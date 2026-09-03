from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import yaml

from edgetts_arena.core.base_adapter import BaseTTSAdapter
from edgetts_arena.core.errors import ArenaError


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


@dataclass(slots=True)
class ModelRecord:
    spec: ModelSpec
    status: ModelStatus
    adapter: BaseTTSAdapter | None = None
    error: str | None = None


AdapterFactory = Callable[[], BaseTTSAdapter]


def _default_adapter_factories() -> dict[str, AdapterFactory]:
    # Import adapters only when the registry is instantiated. Importing them at
    # module load time creates a cycle through edgetts_arena.core.__init__.
    from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
    from edgetts_arena.adapters.kokoro_adapter import KokoroTTSAdapter
    from edgetts_arena.adapters.piper_adapter import PiperTTSAdapter
    from edgetts_arena.adapters.qwen3_adapter import Qwen3TTSAdapter

    return {
        "dummy": DummyTTSAdapter,
        "piper": PiperTTSAdapter,
        "kokoro": KokoroTTSAdapter,
        "qwen3": Qwen3TTSAdapter,
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
        path: str | Path = "config/models_config.yaml",
        adapter_factories: dict[str, AdapterFactory] | None = None,
    ) -> "ModelRegistry":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        models = raw.get("models", [])
        if not isinstance(models, list):
            raise ValueError("models_config.yaml: 'models' must be a list")
        specs = [
            ModelSpec(
                id=str(item["id"]),
                name=str(item.get("name", item["id"])),
                adapter=str(item["adapter"]),
                enabled=bool(item.get("enabled", True)),
                model_path=str(item.get("model_path", "")),
                keep_in_memory=bool(item.get("keep_in_memory", False)),
                num_threads=int(item.get("num_threads", 4)),
                experimental=bool(item.get("experimental", False)),
            )
            for item in models
        ]
        return cls(specs, adapter_factories=adapter_factories)

    def ids(self) -> tuple[str, ...]:
        return tuple(self._records)

    def get_record(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as exc:
            raise ArenaError(1002, f"model '{model_id}' does not exist", error_type="model_not_found") from exc

    def load(self, model_id: str) -> BaseTTSAdapter:
        record = self.get_record(model_id)
        if record.status == ModelStatus.UNAVAILABLE:
            raise ArenaError(1002, f"model '{model_id}' is unavailable", error_type="model_unavailable")
        if record.adapter is not None and record.status == ModelStatus.READY:
            return record.adapter

        factory = self._factories.get(record.spec.adapter)
        if factory is None:
            record.status = ModelStatus.UNAVAILABLE
            raise ArenaError(1002, f"adapter '{record.spec.adapter}' is unavailable", error_type="adapter_unavailable")

        record.status = ModelStatus.LOADING
        record.error = None
        try:
            adapter = factory()
            adapter.load_model(
                record.spec.model_path,
                device="cpu",
                num_threads=record.spec.num_threads,
            )
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

    def list_models(self) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for record in self._records.values():
            factory = self._factories.get(record.spec.adapter)
            capabilities = None
            voices: list[str] = []
            if factory is not None:
                capabilities = factory.capabilities.to_dict()  # type: ignore[attr-defined]
                voices = list(getattr(factory, "available_voices", ()))
            result.append(
                {
                    "id": record.spec.id,
                    "name": record.spec.name,
                    "status": record.status.value,
                    "experimental": record.spec.experimental,
                    "capabilities": capabilities,
                    "voices": voices,
                    "error": record.error,
                }
            )
        return result
