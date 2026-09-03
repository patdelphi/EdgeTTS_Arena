from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Callable

import yaml

from edgetts_arena.core.base_adapter import BaseTTSAdapter
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.defaults import read_default_text

_DEFAULT_MODELS_CONFIG = Path("config/models_config.yaml")


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
    keep_in_memory: bool = False
    num_threads: int = 4
    experimental: bool = False
    worker_python: str = ""
    worker_python_env: str = ""
    language_control: bool = False

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
    ) -> None:
        self._factories: dict[str, AdapterFactory] = _default_adapter_factories()
        if adapter_factories:
            self._factories.update(adapter_factories)
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
    ) -> "ModelRegistry":
        config_path = Path(path)
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
        specs = []
        for item in models:
            worker_python_env = str(item.get("worker_python_env", "") or "")
            worker_python = str(item.get("worker_python", "") or "")
            if not worker_python and worker_python_env:
                worker_python = os.environ.get(worker_python_env, "")
            specs.append(
                ModelSpec(
                    id=str(item["id"]),
                    name=str(item.get("name", item["id"])),
                    adapter=str(item["adapter"]),
                    enabled=bool(item.get("enabled", True)),
                    model_path=str(item.get("model_path", "")),
                    keep_in_memory=bool(item.get("keep_in_memory", False)),
                    num_threads=int(item.get("num_threads", 4)),
                    experimental=bool(item.get("experimental", False)),
                    worker_python=worker_python,
                    worker_python_env=worker_python_env,
                    language_control=bool(item.get("language_control", False)),
                )
            )
        return cls(specs, adapter_factories=adapter_factories)

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
            adapter.load_model(record.spec.model_path, device="cpu", num_threads=threads)
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
            "error": record.error,
        }

    def list_models(self) -> list[dict[str, object]]:
        return [self.model_info(model_id) for model_id in self._records]
