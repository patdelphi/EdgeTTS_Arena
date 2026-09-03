from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("audio must be a non-empty mono array")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return target
