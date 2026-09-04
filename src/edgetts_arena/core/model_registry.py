from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import re
from pathlib import Path
from typing import Callable

import yaml

from edgetts_arena.core.base_adapter import BaseTTSAdapter
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.defaults import read_default_text

_DEFAULT_MODELS_CONFIG = Path("config/models_config.yaml")

# 默认模型搜索路径
DEFAULT_MODEL_SEARCH_PATHS: tuple[str, ...] = (
    "${HF_HOME:-~/.cache}/huggingface/hub",
    "./models",
)


def resolve_model_path(
    model_path: str,
    search_paths: tuple[str, ...] | None = None,
    project_root: Path | None = None,
) -> str:
    """
    解析模型路径，支持多级搜索。
    
    如果 model_path 是绝对路径且存在，直接返回。
    如果是相对路径，按搜索路径顺序查找，返回第一个存在的路径。
    如果都不存在，返回相对于项目根目录的路径。
    """
    if not model_path:
        return ""
    
    path = Path(model_path)
    
    # 绝对路径直接返回
    if path.is_absolute():
        return str(path)
    
    # 使用默认搜索路径
    if search_paths is None:
        search_paths = DEFAULT_MODEL_SEARCH_PATHS
    
    if project_root is None:
        project_root = Path.cwd()
    
    # 在搜索路径中查找
    for search_path in search_paths:
        # 展开环境变量
        expanded = _expand_env_var(search_path)
        base = Path(expanded)
        if not base.is_absolute():
            base = project_root / base
        
        candidate = base / model_path
        if candidate.exists():
            return str(candidate)
    
    # 都不存在，返回相对于项目根目录的路径
    return str(project_root / model_path)


def _expand_env_var(value: str) -> str:
    """展开环境变量，支持 ${VAR:-default} 语法"""
    import re
    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(3) if match.group(2) else ""
        return os.environ.get(var_name, os.path.expanduser(default) if default else "")
    
    pattern = r'\$\{([^}:]+)(:-([^}]*))?\}'
    return re.sub(pattern, replace_match, value)


class ModelStatus(str, Enum):
    UNAVAILABLE = "unavailable"
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    name: str
    adapter: str
    enabled: bool = True
    model_path: str = ""
    resolved_model_path: str = ""  # 解析后的实际路径
    keep_in_memory: bool = False
    num_threads: int = 4
    experimental: bool = False
    worker_python: str = ""
    worker_python_env: str = ""
    language_control: bool = False
    # Optional per-model hard inference timeout (seconds). When None, the global
    # BenchmarkService.inference_timeout_sec applies. Slow autoregressive models
    # (e.g. Qwen3-TTS on CPU) override this so long-text cases are not cut off.
    inference_timeout_sec: float | None = None

    def resolve_worker_python(self) -> str:
        configured = self.worker_python.strip()
        if not configured and self.worker_python_env.strip():
            configured = os.environ.get(self.worker_python_env.strip(), "").strip()
        if not configured:
            return ""
        return os.path.expanduser(os.path.expandvars(configured))


@dataclass(slots=True)
class ModelRecord:
    spec: ModelSpec
    status: ModelStatus
    adapter: BaseTTSAdapter | None = None
    error: str | None = None


AdapterFactory = Callable[[], BaseTTSAdapter]


def _default_adapter_factories() -> dict[str, AdapterFactory]:
    from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
    from edgetts_arena.adapters.kokoro_adapter import KokoroTTSAdapter
    from edgetts_arena.adapters.piper_adapter import PiperTTSAdapter
    from edgetts_arena.adapters.qwen3_adapter import Qwen3TTSAdapter
    from edgetts_arena.adapters.qwen3_native_adapter import Qwen3NativeTTSAdapter
    from edgetts_arena.adapters.cosyvoice_adapter import CosyVoiceTTSAdapter
    from edgetts_arena.adapters.melotts_adapter import MeloTTSAdapter

    return {
        "dummy": DummyTTSAdapter,
        "piper": PiperTTSAdapter,
        "kokoro": KokoroTTSAdapter,
        "qwen3": Qwen3TTSAdapter,
        "qwen3_native": Qwen3NativeTTSAdapter,
        "cosyvoice": CosyVoiceTTSAdapter,
        "melotts": MeloTTSAdapter,
    }


class ModelRegistry:
    """Configuration-backed adapter registry with explicit lifecycle state."""

    def __init__(
        self,
        specs: list[ModelSpec],
        adapter_factories: dict[str, AdapterFactory] | None = None,
        search_paths: tuple[str, ...] | None = None,
    ) -> None:
        self._factories: dict[str, AdapterFactory] = _default_adapter_factories()
        if adapter_factories:
            self._factories.update(adapter_factories)
        self._search_paths = search_paths or DEFAULT_MODEL_SEARCH_PATHS
        self._records: dict[str, ModelRecord] = {}
        for spec in specs:
            available = spec.enabled and spec.adapter in self._factories
            status = ModelStatus.UNLOADED if available else ModelStatus.UNAVAILABLE
            self._records[spec.id] = ModelRecord(spec=spec, status=status)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path = _DEFAULT_MODELS_CONFIG,
        adapter_factories: dict[str, AdapterFactory] | None = None,
        search_paths: tuple[str, ...] | None = None,
    ) -> "ModelRegistry":
        config_path = Path(path)
        project_root = config_path.parent.parent if config_path.exists() else Path.cwd()
        
        if config_path.exists():
            text = config_path.read_text(encoding="utf-8")
        elif config_path == _DEFAULT_MODELS_CONFIG:
            text = read_default_text("models_config.yaml")
        else:
            raise FileNotFoundError(config_path)
        raw = yaml.safe_load(text) or {}
        models = raw.get("models", [])
        if not isinstance(models, list):
            raise ValueError("models_config.yaml: 'models' must be a list")
        
        # 使用传入的搜索路径或默认路径
        effective_search_paths = search_paths or DEFAULT_MODEL_SEARCH_PATHS
        
        specs = []
        for item in models:
            worker_python_env = str(item.get("worker_python_env", "") or "")
            worker_python = str(item.get("worker_python", "") or "")
            if not worker_python and worker_python_env:
                worker_python = os.environ.get(worker_python_env, "")
            
            raw_timeout = item.get("inference_timeout_sec")
            inference_timeout_sec = None if raw_timeout is None else float(raw_timeout)
            if inference_timeout_sec is not None and inference_timeout_sec <= 0:
                raise ValueError(
                    f"models_config.yaml: model '{item.get('id')}' inference_timeout_sec must be > 0"
                )

            model_path = str(item.get("model_path", ""))
            resolved_path = resolve_model_path(
                model_path, 
                search_paths=effective_search_paths,
                project_root=project_root,
            )
            
            specs.append(
                ModelSpec(
                    id=str(item["id"]),
                    name=str(item.get("name", item["id"])),
                    adapter=str(item["adapter"]),
                    enabled=bool(item.get("enabled", True)),
                    model_path=model_path,
                    resolved_model_path=resolved_path,
                    keep_in_memory=bool(item.get("keep_in_memory", False)),
                    num_threads=int(item.get("num_threads", 4)),
                    experimental=bool(item.get("experimental", False)),
                    worker_python=worker_python,
                    worker_python_env=worker_python_env,
                    language_control=bool(item.get("language_control", False)),
                    inference_timeout_sec=inference_timeout_sec,
                )
            )
        return cls(specs, adapter_factories=adapter_factories, search_paths=effective_search_paths)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def get_record(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as exc:
            raise ArenaError(
                1002,
                f"model '{model_id}' does not exist",
                error_type="model_not_found",
            ) from exc

    def set_status(
        self,
        model_id: str,
        status: ModelStatus,
        *,
        error: str | None = None,
    ) -> None:
        record = self.get_record(model_id)
        record.status = status
        record.error = error

    def load(
        self,
        model_id: str,
        *,
        num_threads: int | None = None,
    ) -> BaseTTSAdapter:
        record = self.get_record(model_id)
        if record.status == ModelStatus.UNAVAILABLE:
            raise ArenaError(1002, f"model '{model_id}' is unavailable", error_type="model_unavailable")
        if record.adapter is not None and record.status in {ModelStatus.READY, ModelStatus.BUSY}:
            return record.adapter

        factory = self._factories.get(record.spec.adapter)
        if factory is None:
            record.status = ModelStatus.UNAVAILABLE
            raise ArenaError(1002, f"adapter '{record.spec.adapter}' is unavailable", error_type="adapter_unavailable")

        threads = record.spec.num_threads if num_threads is None else int(num_threads)
        if threads < 1:
            raise ValueError("num_threads must be >= 1")

        record.status = ModelStatus.LOADING
        record.error = None
        try:
            adapter = factory()
            # 使用解析后的模型路径
            model_path = record.spec.resolved_model_path or record.spec.model_path
            adapter.load_model(model_path, device="cpu", num_threads=threads)
        except Exception as exc:
            record.status = ModelStatus.ERROR
            record.error = str(exc)
            raise
        record.adapter = adapter
        record.status = ModelStatus.READY
        return adapter

    def unload(self, model_id: str) -> None:
        record = self.get_record(model_id)
        if record.adapter is not None:
            record.adapter.unload_model()
        record.adapter = None
        record.error = None
        record.status = (
            ModelStatus.UNLOADED
            if record.spec.enabled and record.spec.adapter in self._factories
            else ModelStatus.UNAVAILABLE
        )

    def model_info(self, model_id: str) -> dict[str, object]:
        record = self.get_record(model_id)
        factory = self._factories.get(record.spec.adapter)
        capabilities = None
        voices: list[str] = []
        if factory is not None:
            capabilities = factory.capabilities.to_dict()  # type: ignore[attr-defined]
            capabilities["language_control"] = record.spec.language_control
            source = record.adapter if record.adapter is not None else factory
            voices = list(getattr(source, "available_voices", ()))

        default_voice = None
        if voices:
            default_voice = "default" if "default" in voices else voices[0]
        resolved_worker = record.spec.resolve_worker_python()
        worker_mode = "in_process" if record.spec.keep_in_memory else ("external" if resolved_worker else "spawn")

        return {
            "id": record.spec.id,
            "name": record.spec.name,
            "status": record.status.value,
            "experimental": record.spec.experimental,
            "model_version": "unknown",
            "runtime": "unknown",
            "quantization": "unknown",
            "capabilities": capabilities,
            "default_voice": default_voice,
            "voices": voices,
            "worker_mode": worker_mode,
            "worker_python_configured": bool(resolved_worker),
            "model_path": record.spec.model_path,
            "resolved_model_path": record.spec.resolved_model_path,
            "error": record.error,
        }

    def list_models(self) -> list[dict[str, object]]:
        return [self.model_info(model_id) for model_id in self._records]


_WORKER_ENV_PATTERN = re.compile(r"^EDGETTS_ARENA_(.+)_PYTHON$")


def _display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path)


def _bootstrap_env_scripts(project_root: Path, env_var: str) -> tuple[Path | None, Path | None]:
    """Locate generated bootstrap env scripts (env.ps1 / env.sh) for a worker env var.

    ``EDGETTS_ARENA_QWEN3_PYTHON`` maps to ``exports/bootstrap/qwen3/env.*``.
    """
    match = _WORKER_ENV_PATTERN.match(env_var)
    if not match:
        return None, None
    base = project_root / "exports" / "bootstrap" / match.group(1).lower()
    ps1 = base / "env.ps1"
    sh = base / "env.sh"
    return (ps1 if ps1.is_file() else None, sh if sh.is_file() else None)


def collect_worker_env_warnings(
    registry: "ModelRegistry",
    *,
    project_root: Path | None = None,
) -> list[str]:
    """Return actionable warnings for models whose dedicated worker env var is unset.

    A model that declares ``worker_python_env`` but resolves no interpreter falls
    back to in-process ``spawn`` execution inside the main venv, where its heavy
    runtime (qwen-tts / CosyVoice / MeloTTS) is normally absent. That surfaces
    later as a cryptic import error or ``'NoneType' ... splitlines`` instead of a
    clear setup hint. This check names the missing variable and the bootstrap env
    script to source so the failure is preventable up front.
    """
    root = Path(project_root or Path.cwd())
    warnings: list[str] = []
    for model_id in registry.ids():
        spec = registry.get_record(model_id).spec
        if not spec.enabled:
            continue
        env_var = spec.worker_python_env.strip()
        if not env_var or spec.resolve_worker_python():
            continue
        ps1, sh = _bootstrap_env_scripts(root, env_var)
        fixes: list[str] = []
        if ps1 is not None:
            fixes.append(f"PowerShell: . '{_display_path(ps1, root)}'")
        if sh is not None:
            fixes.append(f"bash: source '{_display_path(sh, root)}'")
        if not fixes:
            fixes.append(
                f"set {env_var} to that environment's Python executable "
                f"(prepare it via scripts/bootstrap_extended_model.py)"
            )
        warnings.append(
            f"{model_id} ({spec.name}): worker env var {env_var} is not set; the model "
            f"will fall back to in-process spawn and likely fail to import its runtime. "
            f"Fix -> {' | '.join(fixes)}"
        )
    return warnings
