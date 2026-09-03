from __future__ import annotations

from typing import Any

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


class Qwen3TTSAdapter(BaseTTSAdapter):
    """Experimental placeholder for Qwen3-TTS 0.6B CPU integration.

    The upstream model is real, but EdgeTTS-Arena deliberately does not bind
    itself to an unverified CPU conversion/runtime. Community ncnn/C++ ports
    are tracked as POC candidates; none is treated as the stable application
    runtime until a reproducible model package, license review, and benchmark
    gate are completed.
    """

    id = "qwen3-tts-0.6b"
    capabilities = TTSCapabilities(
        streaming=False,
        seed=False,
        speed=False,
        voices=False,
        voice_clone=False,
        languages=(),
    )
    available_voices: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.is_loaded = False

    def load_model(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        if device != "cpu":
            raise ValueError("Qwen3-TTS experimental adapter is CPU-targeted only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        raise ArenaError(
            1002,
            "Qwen3-TTS 0.6B is experimental: no CPU runtime has been approved "
            "for EdgeTTS-Arena yet. Keep the model disabled until the CPU POC "
            "passes reproducibility, license, memory, and benchmark gates.",
            error_type="experimental_runtime_unavailable",
        )

    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        raise ModelNotLoadedError(self.id)

    def unload_model(self) -> None:
        self.is_loaded = False
