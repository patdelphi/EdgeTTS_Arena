from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

CosyVoiceFactory = Callable[[str, int], Any]


class CosyVoiceTTSAdapter(BaseTTSAdapter):
    """SFT-only CPU adapter for the official QwenAudio/CosyVoice runtime.

    This deliberately targets ``CosyVoice-300M-SFT`` style models. CosyVoice2/3
    zero-shot inference requires prompt audio/text and therefore does not fit the
    current Arena voice-id-only request contract without a schema extension.
    """

    id = "cosyvoice"
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
        model_factory: CosyVoiceFactory | None = None,
        runtime_version: str | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._runtime_version = runtime_version
        self._engine: Any | None = None
        self._model_path: Path | None = None
        self._num_threads = 1
        self.is_loaded = False
        self.available_voices = ()

    def load_model(self, model_path: str, *, device: str = "cpu", num_threads: int = 4) -> None:
        if device != "cpu":
            raise ValueError("CosyVoice adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"CosyVoice model directory not found: {path}")
        if not (path / "cosyvoice.yaml").is_file():
            raise ValueError("CosyVoice model directory must contain cosyvoice.yaml")

        factory, runtime_version = self._runtime()
        engine = factory(str(path.resolve()), num_threads)
        voices = tuple(str(item) for item in engine.list_available_spks())
        if not voices:
            raise ArenaError(
                1002,
                "CosyVoice model exposes no SFT speakers; use a CosyVoice-300M-SFT model for this adapter",
                error_type="model_unavailable",
            )

        self._engine = engine
        self._runtime_version = runtime_version
        self._model_path = path.resolve()
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
        if speed <= 0:
            raise ValueError("speed must be positive")
        if kwargs.get("seed") is not None:
            raise ArenaError(1003, "CosyVoice SFT adapter does not expose deterministic seed control", error_type="capability_conflict")
        selected_voice = voice or self.available_voices[0]
        if selected_voice not in self.available_voices:
            raise ValueError(f"unknown CosyVoice SFT speaker: {selected_voice}")

        chunks = list(engine.inference_sft(normalized, selected_voice, stream=False, speed=speed))
        audio = self._merge_chunks(chunks)
        return TTSOutput(audio=audio, sample_rate=int(engine.sample_rate), metadata=self._metadata(selected_voice, streaming=False))

    def infer_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[TTSOutput]:
        engine = self._require_loaded()
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")
        selected_voice = voice or self.available_voices[0]
        if selected_voice not in self.available_voices:
            raise ValueError(f"unknown CosyVoice SFT speaker: {selected_voice}")
        for chunk in engine.inference_sft(normalized, selected_voice, stream=True, speed=speed):
            audio = self._chunk_audio(chunk)
            yield TTSOutput(audio=audio, sample_rate=int(engine.sample_rate), metadata=self._metadata(selected_voice, streaming=True))

    def unload_model(self) -> None:
        self._engine = None
        self._model_path = None
        self.available_voices = ()
        self.is_loaded = False
        gc.collect()

    def _runtime(self) -> tuple[CosyVoiceFactory, str]:
        if self._model_factory is not None:
            return self._model_factory, self._runtime_version or "injected"
        try:
            import torch
            from cosyvoice.cli.cosyvoice import AutoModel
        except ImportError as exc:
            raise ArenaError(
                1002,
                "CosyVoice runtime is not installed. Install the official QwenAudio/CosyVoice source checkout and its dependencies.",
                error_type="model_unavailable",
            ) from exc

        def factory(model_dir: str, num_threads: int) -> Any:
            torch.set_num_threads(num_threads)
            return AutoModel(model_dir=model_dir)

        return factory, "QwenAudio/CosyVoice-source"

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float32)
        return array.reshape(-1)

    @classmethod
    def _chunk_audio(cls, chunk: Any) -> np.ndarray:
        if not isinstance(chunk, dict) or "tts_speech" not in chunk:
            raise RuntimeError("CosyVoice returned a chunk without tts_speech")
        audio = cls._to_numpy(chunk["tts_speech"])
        if audio.size == 0:
            raise RuntimeError("CosyVoice returned an empty audio chunk")
        return audio

    @classmethod
    def _merge_chunks(cls, chunks: list[Any]) -> np.ndarray:
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks")
        return np.concatenate([cls._chunk_audio(chunk) for chunk in chunks]).astype(np.float32, copy=False)

    def _metadata(self, voice: str, *, streaming: bool) -> dict[str, Any]:
        return {
            "adapter": self.id,
            "voice": voice,
            "runtime": "QwenAudio/CosyVoice",
            "runtime_version": self._runtime_version or "unknown",
            "model_version": self._model_path.name if self._model_path else "unknown",
            "quantization": "model-defined",
            "threads": self._num_threads,
            "streaming": streaming,
            "mode": "sft",
        }

    def _require_loaded(self) -> Any:
        if not self.is_loaded or self._engine is None:
            raise ModelNotLoadedError(self.id)
        return self._engine
