from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

MeloFactory = Callable[..., Any]
_LANGUAGE_MAP = {"EN": "en", "ES": "es", "FR": "fr", "ZH": "zh", "JP": "ja", "KR": "ko"}


class MeloTTSAdapter(BaseTTSAdapter):
    """CPU adapter for MyShell MeloTTS using explicit local model files."""

    id = "melotts"
    capabilities = TTSCapabilities(
        streaming=False,
        seed=False,
        speed=True,
        voices=True,
        voice_clone=False,
        languages=("en", "es", "fr", "zh", "ja", "ko"),
    )
    available_voices: tuple[str, ...] = ()

    def __init__(self, *, tts_factory: MeloFactory | None = None, runtime_version: str | None = None) -> None:
        self._tts_factory = tts_factory
        self._runtime_version = runtime_version
        self._engine: Any | None = None
        self._descriptor_path: Path | None = None
        self._language_code: str | None = None
        self._num_threads = 1
        self.is_loaded = False
        self.available_voices = ()

    def load_model(self, model_path: str, *, device: str = "cpu", num_threads: int = 4) -> None:
        if device != "cpu":
            raise ValueError("MeloTTS adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        descriptor_path, descriptor = self._read_descriptor(model_path)
        language = str(descriptor.get("language") or "").upper()
        if language not in _LANGUAGE_MAP:
            raise ValueError("MeloTTS descriptor language must be one of EN/ES/FR/ZH/JP/KR")
        config_path = self._local_file(descriptor_path.parent, descriptor.get("config_path"), "config_path")
        ckpt_path = self._local_file(descriptor_path.parent, descriptor.get("ckpt_path"), "ckpt_path")

        factory, runtime_version = self._runtime()
        engine = factory(
            language=language,
            device="cpu",
            use_hf=False,
            config_path=str(config_path),
            ckpt_path=str(ckpt_path),
            num_threads=num_threads,
        )
        speaker_map = dict(getattr(getattr(engine.hps, "data", None), "spk2id", {}) or {})
        if not speaker_map:
            raise RuntimeError("MeloTTS runtime returned no speaker ids")

        self._engine = engine
        self._runtime_version = runtime_version
        self._descriptor_path = descriptor_path
        self._language_code = language
        self._num_threads = num_threads
        self.available_voices = tuple(str(key) for key in speaker_map)
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
        if speed <= 0:
            raise ValueError("speed must be positive")
        if kwargs.get("seed") is not None:
            raise ArenaError(1003, "MeloTTS does not expose deterministic seed control", error_type="capability_conflict")

        selected_voice = voice or self.available_voices[0]
        speaker_map = dict(engine.hps.data.spk2id)
        if selected_voice not in speaker_map:
            raise ValueError(f"unknown MeloTTS speaker: {selected_voice}")
        audio = engine.tts_to_file(
            normalized,
            speaker_map[selected_voice],
            output_path=None,
            speed=speed,
            quiet=True,
        )
        array = np.asarray(audio, dtype=np.float32).reshape(-1)
        if array.size == 0:
            raise RuntimeError("MeloTTS returned empty audio")
        sample_rate = int(engine.hps.data.sampling_rate)
        return TTSOutput(
            audio=array,
            sample_rate=sample_rate,
            metadata={
                "adapter": self.id,
                "voice": selected_voice,
                "language": _LANGUAGE_MAP[self._language_code or "EN"],
                "runtime": "myshell-ai/MeloTTS",
                "runtime_version": self._runtime_version or "unknown",
                "model_version": self._descriptor_path.parent.name if self._descriptor_path else "unknown",
                "quantization": "fp32/model-defined",
                "threads": self._num_threads,
                "streaming": False,
            },
        )

    def unload_model(self) -> None:
        self._engine = None
        self._descriptor_path = None
        self._language_code = None
        self.available_voices = ()
        self.is_loaded = False
        gc.collect()

    def _runtime(self) -> tuple[MeloFactory, str]:
        if self._tts_factory is not None:
            return self._tts_factory, self._runtime_version or "injected"
        try:
            import torch
            from melo.api import TTS
        except ImportError as exc:
            raise ArenaError(
                1002,
                "MeloTTS runtime is not installed. Install the official myshell-ai/MeloTTS package/source separately.",
                error_type="model_unavailable",
            ) from exc

        def factory(**kwargs: Any) -> Any:
            threads = int(kwargs.pop("num_threads"))
            torch.set_num_threads(threads)
            return TTS(**kwargs)

        return factory, "melotts-0.1.2/source"

    @staticmethod
    def _read_descriptor(model_path: str) -> tuple[Path, dict[str, Any]]:
        path = Path(model_path).expanduser()
        descriptor = path / "model.json" if path.is_dir() else path
        if not descriptor.is_file() or descriptor.suffix.lower() != ".json":
            raise FileNotFoundError("MeloTTS model_path must be a model.json descriptor or containing directory")
        raw = json.loads(descriptor.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("MeloTTS model descriptor must be a JSON object")
        return descriptor.resolve(), raw

    @staticmethod
    def _local_file(root: Path, raw: Any, field: str) -> Path:
        if not raw:
            raise ValueError(f"MeloTTS descriptor requires {field}")
        path = Path(str(raw))
        if path.is_absolute():
            candidate = path
        else:
            candidate = root / path
        candidate = candidate.resolve()
        if not candidate.is_file():
            raise FileNotFoundError(f"MeloTTS {field} not found: {candidate}")
        return candidate

    def _require_loaded(self) -> Any:
        if not self.is_loaded or self._engine is None:
            raise ModelNotLoadedError(self.id)
        return self._engine
