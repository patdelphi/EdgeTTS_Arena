from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .capabilities import TTSCapabilities


@dataclass(slots=True)
class TTSOutput:
    """Normalized adapter output used by the runtime."""

    audio: np.ndarray
    sample_rate: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        audio = np.asarray(self.audio)
        if audio.ndim != 1:
            raise ValueError("audio must be mono (1-D)")
        if audio.size == 0:
            raise ValueError("audio must not be empty")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if not np.isfinite(audio).all():
            raise ValueError("audio contains NaN or Inf")
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        self.audio = np.clip(audio, -1.0, 1.0)


class BaseTTSAdapter(ABC):
    """Stable SPI implemented by every TTS backend."""

    id: str
    capabilities: TTSCapabilities

    @abstractmethod
    def load_model(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        raise NotImplementedError

    def infer_stream(self, text: str, **kwargs: Any) -> Iterator[TTSOutput]:
        raise NotImplementedError("streaming is not supported by this adapter")

    @abstractmethod
    def unload_model(self) -> None:
        raise NotImplementedError
