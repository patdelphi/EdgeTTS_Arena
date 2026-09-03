import wave
from pathlib import Path

import numpy as np

from edgetts_arena.utils import write_wav


def test_write_wav(tmp_path: Path) -> None:
    audio = np.zeros(2400, dtype=np.float32)
    path = write_wav(tmp_path / "test.wav", audio, 24_000)
    with wave.open(str(path), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 24_000
        assert handle.getnframes() == 2400
