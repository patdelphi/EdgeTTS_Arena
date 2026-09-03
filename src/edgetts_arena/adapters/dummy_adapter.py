from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ModelNotLoadedError


class DummyTTSAdapter(BaseTTSAdapter):
    """Dependency-light deterministic adapter for CI and end-to-end smoke tests."""

    id = "dummy"
    capabilities = TTSCapabilities(
        streaming=True,
        seed=True,
        speed=True,
        voices=True,
        voice_clone=False,
        languages=("zh", "en"),
    )
    available_voices = ("default", "bright", "deep")

    def __init__(self) -> None:
        self.is_loaded = False
        self.sample_rate = 24_000
        self.num_threads = 1

    def load_model(
        self,
        model_path: str = "",
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        if device != "cpu":
            raise ValueError("dummy adapter only supports cpu")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        self.num_threads = num_threads
        self.is_loaded = True

    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        self._require_loaded()
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if not 0.25 <= speed <= 4.0:
            raise ValueError("speed must be between 0.25 and 4.0")
        selected_voice = voice or "default"
        if selected_voice not in self.available_voices:
            raise ValueError(f"unknown dummy voice: {selected_voice}")

        seed = int(kwargs.get("seed", 0) or 0)
        voice_offset = {"default": 0.0, "bright": 80.0, "deep": -70.0}[selected_voice]
        frequency = 440.0 + voice_offset + float(seed % 37)
        duration = min(3.0, max(0.35, len(normalized) * 0.045 / speed))
        sample_count = max(1, int(self.sample_rate * duration))
        t = np.arange(sample_count, dtype=np.float32) / self.sample_rate
        envelope = np.minimum(1.0, np.arange(sample_count, dtype=np.float32) / 240.0)
        tail = np.minimum(1.0, np.arange(sample_count, 0, -1, dtype=np.float32) / 240.0)
        audio = 0.1 * np.sin(2.0 * np.pi * frequency * t) * envelope * tail

        return TTSOutput(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            metadata={
                "adapter": self.id,
                "voice": selected_voice,
                "seed": seed,
                "runtime": "numpy",
                "model_version": "dummy-v1",
                "quantization": "none",
            },
        )

    def infer_stream(self, text: str, **kwargs: Any) -> Iterator[TTSOutput]:
        output = self.infer(text, **kwargs)
        chunk_size = max(1, int(output.sample_rate * 0.1))
        for start in range(0, output.audio.size, chunk_size):
            chunk = output.audio[start : start + chunk_size]
            yield TTSOutput(
                audio=chunk.copy(),
                sample_rate=output.sample_rate,
                metadata={**output.metadata, "chunk_index": start // chunk_size},
            )

    def unload_model(self) -> None:
        self.is_loaded = False

    def _require_loaded(self) -> None:
        if not self.is_loaded:
            raise ModelNotLoadedError(self.id)
