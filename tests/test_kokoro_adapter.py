from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgetts_arena.adapters.kokoro_adapter import KokoroTTSAdapter
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


class FakeKokoro:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_voices(self) -> list[str]:
        return ["af_heart", "bf_emma", "zf_xiaoxiao"]

    def create(self, text: str, **kwargs: Any):
        self.calls.append((text, kwargs))
        return np.full(2400, 0.05, dtype=np.float32), 24_000


def _files(tmp_path: Path) -> Path:
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    model.write_bytes(b"onnx")
    voices.write_bytes(b"voices")
    return model


def _adapter(tmp_path: Path) -> tuple[KokoroTTSAdapter, FakeKokoro, Path]:
    model = _files(tmp_path)
    engine = FakeKokoro()

    def factory(model_file: str, voices_file: str, num_threads: int) -> FakeKokoro:
        assert Path(model_file) == model.resolve()
        assert Path(voices_file) == (tmp_path / "voices-v1.0.bin").resolve()
        assert num_threads == 3
        return engine

    adapter = KokoroTTSAdapter(kokoro_factory=factory, runtime_version="test-runtime")
    return adapter, engine, model


def test_kokoro_requires_load(tmp_path: Path) -> None:
    adapter, _, _ = _adapter(tmp_path)
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")


def test_kokoro_load_and_infer(tmp_path: Path) -> None:
    adapter, engine, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)
    output = adapter.infer(" hello ", voice="bf_emma", speed=1.25)

    assert output.sample_rate == 24_000
    assert output.audio.dtype == np.float32
    assert output.metadata["language"] == "en-gb"
    assert output.metadata["runtime"] == "kokoro-onnx"
    assert output.metadata["runtime_version"] == "test-runtime"
    assert output.metadata["threads"] == 3
    assert engine.calls[-1][0] == "hello"
    assert engine.calls[-1][1]["voice"] == "bf_emma"
    assert engine.calls[-1][1]["speed"] == 1.25


def test_kokoro_language_can_be_explicit(tmp_path: Path) -> None:
    adapter, engine, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)
    adapter.infer("hello", voice="bf_emma", language="en-gb")
    assert engine.calls[-1][1]["lang"] == "en-gb"


def test_kokoro_filters_unverified_direct_text_voices(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)
    assert "zf_xiaoxiao" not in adapter.available_voices
    with pytest.raises(ValueError, match="unknown Kokoro voice"):
        adapter.infer("你好", voice="zf_xiaoxiao", language="cmn")


def test_kokoro_seed_conflict(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)
    with pytest.raises(ArenaError) as exc_info:
        adapter.infer("hello", seed=1)
    assert exc_info.value.code == 1003


def test_kokoro_rejects_unknown_voice(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)
    with pytest.raises(ValueError, match="unknown Kokoro voice"):
        adapter.infer("hello", voice="nope")


def test_kokoro_directory_resolution(tmp_path: Path) -> None:
    _files(tmp_path)
    engine = FakeKokoro()
    adapter = KokoroTTSAdapter(
        kokoro_factory=lambda *_args: engine,
        runtime_version="test-runtime",
    )
    adapter.load_model(str(tmp_path), num_threads=3)
    assert adapter.available_voices[0] == "af_heart"


def test_kokoro_unload(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)
    adapter.unload_model()
    assert adapter.is_loaded is False
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")
