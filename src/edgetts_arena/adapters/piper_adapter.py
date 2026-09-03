from __future__ import annotations

import gc
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

VoiceLoader = Callable[..., Any]
SynthesisConfigFactory = Callable[..., Any]


class PiperTTSAdapter(BaseTTSAdapter):
    """Piper ONNX adapter using the maintained ``piper-tts`` Python API."""

    id = "piper"
    capabilities = TTSCapabilities(
        streaming=True,
        seed=False,
        speed=True,
        voices=True,
        voice_clone=False,
        languages=(),
    )
    available_voices: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        voice_loader: VoiceLoader | None = None,
        synthesis_config_factory: SynthesisConfigFactory | None = None,
        runtime_version: str | None = None,
    ) -> None:
        self._voice_loader = voice_loader
        self._synthesis_config_factory = synthesis_config_factory
        self._runtime_version = runtime_version
        self._voice: Any | None = None
        self._model_path: Path | None = None
        self._sample_rate: int | None = None
        self._num_threads = 1
        self.is_loaded = False
        self.available_voices = ()

    def load_model(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        if device != "cpu":
            raise ValueError("Piper adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        model_file = self._resolve_model_file(model_path)
        config_file = Path(f"{model_file}.json")
        if not config_file.is_file():
            raise FileNotFoundError(
                f"Piper voice config not found: {config_file}. "
                "Expected the .onnx.json file next to the ONNX model."
            )

        voice_loader, synthesis_config_factory, runtime_version = self._runtime()
        voice = voice_loader(str(model_file), use_cuda=False)
        config = getattr(voice, "config", None)
        if config is None:
            raise RuntimeError("Piper runtime returned a voice without config")

        sample_rate = int(getattr(config, "sample_rate", 0) or 0)
        if sample_rate <= 0:
            raise RuntimeError("Piper voice config has an invalid sample rate")

        self._voice = voice
        self._synthesis_config_factory = synthesis_config_factory
        self._runtime_version = runtime_version
        self._model_path = model_file
        self._sample_rate = sample_rate
        self._num_threads = num_threads
        self.available_voices = self._discover_voices(config)
        self.is_loaded = True

    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        piper_voice = self._require_loaded()
        normalized = self._validate_text(text)
        syn_config, selected_voice = self._make_synthesis_config(
            voice=voice,
            speed=speed,
            kwargs=kwargs,
        )

        arrays: list[np.ndarray] = []
        sample_rate: int | None = None
        chunk_count = 0
        for chunk in piper_voice.synthesize(normalized, syn_config=syn_config):
            audio, chunk_rate = self._normalize_chunk(chunk)
            if sample_rate is None:
                sample_rate = chunk_rate
            elif chunk_rate != sample_rate:
                raise RuntimeError("Piper returned inconsistent sample rates across chunks")
            arrays.append(audio)
            chunk_count += 1

        if not arrays or sample_rate is None:
            raise RuntimeError("Piper returned no audio")

        return TTSOutput(
            audio=np.concatenate(arrays).astype(np.float32, copy=False),
            sample_rate=sample_rate,
            metadata=self._metadata(
                selected_voice=selected_voice,
                chunk_count=chunk_count,
                streaming=False,
            ),
        )

    def infer_stream(self, text: str, **kwargs: Any) -> Iterator[TTSOutput]:
        piper_voice = self._require_loaded()
        normalized = self._validate_text(text)
        voice = kwargs.pop("voice", None)
        speed = float(kwargs.pop("speed", 1.0))
        syn_config, selected_voice = self._make_synthesis_config(
            voice=voice,
            speed=speed,
            kwargs=kwargs,
        )

        emitted = False
        for chunk_index, chunk in enumerate(
            piper_voice.synthesize(normalized, syn_config=syn_config)
        ):
            audio, sample_rate = self._normalize_chunk(chunk)
            emitted = True
            yield TTSOutput(
                audio=audio,
                sample_rate=sample_rate,
                metadata={
                    **self._metadata(
                        selected_voice=selected_voice,
                        chunk_count=None,
                        streaming=True,
                    ),
                    "chunk_index": chunk_index,
                },
            )
        if not emitted:
            raise RuntimeError("Piper returned no audio")

    def unload_model(self) -> None:
        self._voice = None
        self._model_path = None
        self._sample_rate = None
        self.available_voices = ()
        self.is_loaded = False
        gc.collect()

    def _runtime(self) -> tuple[VoiceLoader, SynthesisConfigFactory, str]:
        if self._voice_loader is not None and self._synthesis_config_factory is not None:
            return (
                self._voice_loader,
                self._synthesis_config_factory,
                self._runtime_version or "injected",
            )

        try:
            from piper import PiperVoice, SynthesisConfig
        except ImportError as exc:
            raise ArenaError(
                1002,
                'Piper runtime is not installed. Install with: pip install -e ".[piper]"',
                error_type="model_unavailable",
            ) from exc

        try:
            runtime_version = importlib_metadata.version("piper-tts")
        except importlib_metadata.PackageNotFoundError:
            runtime_version = "unknown"
        return PiperVoice.load, SynthesisConfig, runtime_version

    @staticmethod
    def _resolve_model_file(model_path: str) -> Path:
        raw = model_path.strip()
        if not raw:
            raise ValueError("model_path must not be empty for Piper")
        path = Path(raw).expanduser()
        if path.is_file():
            if path.suffix.lower() != ".onnx":
                raise ValueError("Piper model_path must point to an .onnx model")
            return path.resolve()
        if path.is_dir():
            candidates = sorted(p for p in path.glob("*.onnx") if p.is_file())
            if not candidates:
                raise FileNotFoundError(f"no Piper .onnx model found in: {path}")
            if len(candidates) > 1:
                names = ", ".join(p.name for p in candidates[:5])
                raise ValueError(
                    f"multiple Piper models found in {path}: {names}. "
                    "Configure model_path to a specific .onnx file."
                )
            return candidates[0].resolve()
        raise FileNotFoundError(f"Piper model path does not exist: {path}")

    @staticmethod
    def _discover_voices(config: Any) -> tuple[str, ...]:
        num_speakers = int(getattr(config, "num_speakers", 1) or 1)
        speaker_id_map = dict(getattr(config, "speaker_id_map", {}) or {})
        if speaker_id_map:
            return tuple(speaker_id_map.keys())
        if num_speakers > 1:
            return tuple(str(index) for index in range(num_speakers))
        return ("default",)

    def _make_synthesis_config(
        self,
        *,
        voice: str | None,
        speed: float,
        kwargs: dict[str, Any],
    ) -> tuple[Any, str]:
        piper_voice = self._require_loaded()
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        if kwargs.get("seed") is not None:
            raise ArenaError(
                1003,
                "Piper does not support deterministic seed control",
                error_type="capability_conflict",
            )

        speaker_id, selected_voice = self._resolve_speaker(
            getattr(piper_voice, "config", None),
            voice,
        )
        config_kwargs: dict[str, Any] = {"length_scale": 1.0 / speed}
        if speaker_id is not None:
            config_kwargs["speaker_id"] = speaker_id

        for name in ("noise_scale", "noise_w_scale", "volume"):
            value = kwargs.get(name)
            if value is not None:
                value = float(value)
                if value < 0 or (name == "volume" and value == 0):
                    raise ValueError(f"{name} must be positive" if name == "volume" else f"{name} must be >= 0")
                config_kwargs[name] = value
        if kwargs.get("normalize_audio") is not None:
            config_kwargs["normalize_audio"] = bool(kwargs["normalize_audio"])

        factory = self._synthesis_config_factory
        if factory is None:
            raise RuntimeError("Piper synthesis config factory is unavailable")
        return factory(**config_kwargs), selected_voice

    @staticmethod
    def _resolve_speaker(config: Any, voice: str | None) -> tuple[int | None, str]:
        num_speakers = int(getattr(config, "num_speakers", 1) or 1)
        speaker_id_map = dict(getattr(config, "speaker_id_map", {}) or {})
        default_speaker_id = int(getattr(config, "default_speaker_id", 0) or 0)

        if num_speakers <= 1:
            if voice not in (None, "default"):
                raise ValueError("this Piper model has only one speaker; use voice='default'")
            return None, "default"

        if voice in (None, "default"):
            selected = next(
                (name for name, sid in speaker_id_map.items() if int(sid) == default_speaker_id),
                str(default_speaker_id),
            )
            return default_speaker_id, selected
        if voice in speaker_id_map:
            return int(speaker_id_map[voice]), voice
        try:
            speaker_id = int(voice)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"unknown Piper speaker: {voice}") from exc
        if not 0 <= speaker_id < num_speakers:
            raise ValueError(f"Piper speaker id out of range: {speaker_id}")
        selected = next(
            (name for name, sid in speaker_id_map.items() if int(sid) == speaker_id),
            str(speaker_id),
        )
        return speaker_id, selected

    @staticmethod
    def _validate_text(text: str) -> str:
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        return normalized

    @staticmethod
    def _normalize_chunk(chunk: Any) -> tuple[np.ndarray, int]:
        audio = np.asarray(getattr(chunk, "audio_float_array", None), dtype=np.float32)
        sample_rate = int(getattr(chunk, "sample_rate", 0) or 0)
        if audio.ndim != 1 or audio.size == 0:
            raise RuntimeError("Piper returned an invalid audio chunk")
        if sample_rate <= 0:
            raise RuntimeError("Piper returned an invalid sample rate")
        return audio, sample_rate

    def _metadata(
        self,
        *,
        selected_voice: str,
        chunk_count: int | None,
        streaming: bool,
    ) -> dict[str, Any]:
        piper_voice = self._require_loaded()
        config = getattr(piper_voice, "config", None)
        return {
            "adapter": self.id,
            "voice": selected_voice,
            "runtime": "piper-tts",
            "runtime_version": self._runtime_version or "unknown",
            "model_version": str(getattr(config, "piper_version", None) or "unknown"),
            "quantization": "model-defined",
            "model_path": str(self._model_path) if self._model_path is not None else None,
            "language": str(getattr(config, "espeak_voice", None) or "unknown"),
            "threads_requested": self._num_threads,
            "chunk_count": chunk_count,
            "streaming": streaming,
        }

    def _require_loaded(self) -> Any:
        if not self.is_loaded or self._voice is None:
            raise ModelNotLoadedError(self.id)
        return self._voice
