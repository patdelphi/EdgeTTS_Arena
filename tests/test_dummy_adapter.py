import numpy as np
import pytest

from edgetts_arena.adapters import DummyTTSAdapter
from edgetts_arena.core.errors import ModelNotLoadedError


def test_dummy_requires_load() -> None:
    adapter = DummyTTSAdapter()
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")


def test_dummy_lifecycle_and_output() -> None:
    adapter = DummyTTSAdapter()
    adapter.load_model(num_threads=2)
    output = adapter.infer("hello world", voice="bright", speed=1.2, seed=42)
    assert output.audio.ndim == 1
    assert output.audio.dtype == np.float32
    assert output.audio.size > 0
    assert np.isfinite(output.audio).all()
    assert output.sample_rate == 24_000
    assert output.metadata["voice"] == "bright"
    adapter.unload_model()
    assert adapter.is_loaded is False


def test_dummy_is_deterministic_for_same_inputs() -> None:
    adapter = DummyTTSAdapter()
    adapter.load_model()
    first = adapter.infer("same text", seed=7).audio
    second = adapter.infer("same text", seed=7).audio
    np.testing.assert_array_equal(first, second)


def test_dummy_stream_chunks_reconstruct_output() -> None:
    adapter = DummyTTSAdapter()
    adapter.load_model()
    full = adapter.infer("stream me", seed=3)
    chunks = list(adapter.infer_stream("stream me", seed=3))
    reconstructed = np.concatenate([chunk.audio for chunk in chunks])
    np.testing.assert_array_equal(full.audio, reconstructed)
