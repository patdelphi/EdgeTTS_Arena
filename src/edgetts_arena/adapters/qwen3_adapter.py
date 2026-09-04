from __future__ import annotations

import gc
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

QwenFactory = Callable[[str, int], Any]

_OFFICIAL_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)
_LANGUAGE_ALIASES = {
    "auto": "Auto",
    "zh": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}


class Qwen3TTSAdapter(BaseTTSAdapter):
    """CPU adapter for the official Qwen3-TTS 0.6B CustomVoice runtime.

    The adapter deliberately targets the official 12Hz 0.6B CustomVoice
    checkpoint because its text + built-in speaker contract fits the current
    Arena request model. The Base checkpoint is voice-cloning oriented and is
    intentionally not accepted here.

    Qwen3-TTS is kept experimental in the default registry and is expected to
    run in a dedicated Python environment via ``worker_python_env``.
    """

    id = "qwen3-tts-0.6b"
    capabilities = TTSCapabilities(
        streaming=False,
        seed=False,
        speed=False,
        voices=True,
        voice_clone=False,
        languages=("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"),
    )
    available_voices: tuple[str, ...] = _OFFICIAL_SPEAKERS

    def __init__(
        self,
        *,
        model_factory: QwenFactory | None = None,
        runtime_version: str | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._runtime_version = runtime_version
        self._engine: Any | None = None
        self._model_path: Path | None = None
        self._num_threads = 1
        self.is_loaded = False
        self.available_voices = _OFFICIAL_SPEAKERS

    def load_model(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        if device != "cpu":
            raise ValueError("Qwen3-TTS adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"Qwen3-TTS model directory not found: {path}")
        config = self._read_model_config(path)
        if str(config.get("tts_model_type") or "").lower() != "custom_voice":
            raise ValueError("Qwen3-TTS adapter requires a CustomVoice checkpoint")
        if str(config.get("tts_model_size") or "").lower() != "0b6":
            raise ValueError("Qwen3-TTS adapter requires the 0.6B (0b6) checkpoint")

        factory, runtime_version = self._runtime()
        resolved = path.resolve()
        engine = factory(str(resolved), num_threads)
        voices = self._resolve_voices(engine)
        if not voices:
            raise ArenaError(
                1002,
                "Qwen3-TTS CustomVoice runtime exposes no supported speakers",
                error_type="model_unavailable",
            )

        self._engine = engine
        self._runtime_version = runtime_version
        self._model_path = resolved
        self._num_threads = num_threads
        self.available_voices = voices
        self.is_loaded = True

    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        engine = self._require_loaded()
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if speed != 1.0:
            raise ArenaError(
                1003,
                "Qwen3-TTS 0.6B CustomVoice does not expose semantic speed control",
                error_type="capability_conflict",
            )
        if kwargs.pop("seed", None) is not None:
            raise ArenaError(
                1003,
                "Qwen3-TTS adapter does not expose deterministic seed control",
                error_type="capability_conflict",
            )

        instruct = kwargs.pop("instruct", None)
        if instruct not in {None, ""}:
            raise ArenaError(
                1003,
                "Qwen3-TTS 0.6B CustomVoice does not support instruct control",
                error_type="capability_conflict",
            )
        language = self._normalize_language(kwargs.pop("language", None))
        if kwargs:
            unsupported = ", ".join(sorted(str(key) for key in kwargs))
            raise ValueError(f"unsupported Qwen3-TTS inference options: {unsupported}")

        selected_voice = self._select_voice(voice)
        wavs, sample_rate = engine.generate_custom_voice(
            text=normalized,
            speaker=selected_voice,
            language=language,
        )
        if not isinstance(wavs, (list, tuple)) or not wavs:
            raise RuntimeError("Qwen3-TTS returned no audio")
        audio = np.asarray(wavs[0], dtype=np.float32).reshape(-1)
        if audio.size == 0:
            raise RuntimeError("Qwen3-TTS returned empty audio")

        return TTSOutput(
            audio=audio,
            sample_rate=int(sample_rate),
            metadata={
                "adapter": self.id,
                "voice": selected_voice,
                "language": language,
                "runtime": "QwenLM/Qwen3-TTS qwen-tts",
                "runtime_version": self._runtime_version or "unknown",
                "model_version": self._model_path.name if self._model_path else "unknown",
                "quantization": "fp32",
                "threads": self._num_threads,
                "streaming": False,
                "mode": "custom_voice",
            },
        )

    def unload_model(self) -> None:
        self._engine = None
        self._model_path = None
        self.available_voices = _OFFICIAL_SPEAKERS
        self.is_loaded = False
        gc.collect()

    def _runtime(self) -> tuple[QwenFactory, str]:
        if self._model_factory is not None:
            return self._model_factory, self._runtime_version or "injected"
        try:
            import torch
            from qwen_tts import Qwen3TTSModel
        except ImportError as exc:
            raise ArenaError(
                1002,
                "Qwen3-TTS runtime (qwen-tts) is not importable in this interpreter. "
                "This model must run in its dedicated worker environment: set "
                "EDGETTS_ARENA_QWEN3_PYTHON to that venv's Python (for example by "
                "sourcing exports/bootstrap/qwen3/env.ps1) so the benchmark launches "
                "it with qwen-tts installed.",
                error_type="model_unavailable",
            ) from exc

        def factory(model_dir: str, num_threads: int) -> Any:
            torch.set_num_threads(num_threads)
            return Qwen3TTSModel.from_pretrained(
                model_dir,
                device_map="cpu",
                dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

        try:
            version = importlib.metadata.version("qwen-tts")
        except importlib.metadata.PackageNotFoundError:
            version = "source"
        return factory, version

    @staticmethod
    def _read_model_config(path: Path) -> dict[str, Any]:
        config_path = path / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen3-TTS model config not found: {config_path}")
        try:
            raw = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid Qwen3-TTS config.json: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Qwen3-TTS config.json must be a JSON object")
        return raw

    @staticmethod
    def _resolve_voices(engine: Any) -> tuple[str, ...]:
        getter = getattr(engine, "get_supported_speakers", None)
        reported = getter() if callable(getter) else None
        if not reported:
            return _OFFICIAL_SPEAKERS
        lookup = {str(item).lower(): str(item) for item in reported}
        known = tuple(name for name in _OFFICIAL_SPEAKERS if name.lower() in lookup)
        unknown = tuple(
            lookup[key]
            for key in sorted(lookup)
            if key not in {name.lower() for name in _OFFICIAL_SPEAKERS}
        )
        return known + unknown

    def _select_voice(self, requested: str | None) -> str:
        if not self.available_voices:
            raise ArenaError(1002, "Qwen3-TTS exposes no speakers", error_type="model_unavailable")
        if requested is None:
            return self.available_voices[0]
        wanted = requested.lower()
        for item in self.available_voices:
            if item.lower() == wanted:
                return item
        raise ValueError(f"unknown Qwen3-TTS speaker: {requested}")

    @staticmethod
    def _normalize_language(language: Any) -> str:
        if language is None:
            return "Auto"
        key = str(language).strip().lower()
        try:
            return _LANGUAGE_ALIASES[key]
        except KeyError as exc:
            supported = ", ".join(sorted(_LANGUAGE_ALIASES))
            raise ValueError(f"unsupported Qwen3-TTS language '{language}'; use one of: {supported}") from exc

    def _require_loaded(self) -> Any:
        if not self.is_loaded or self._engine is None:
            raise ModelNotLoadedError(self.id)
        return self._engine
