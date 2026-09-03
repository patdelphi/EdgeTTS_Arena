from __future__ import annotations

import pytest

from edgetts_arena.adapters.qwen3_adapter import Qwen3TTSAdapter
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


def test_qwen3_placeholder_declares_only_implemented_capabilities() -> None:
    adapter = Qwen3TTSAdapter()
    caps = adapter.capabilities
    assert caps.streaming is False
    assert caps.seed is False
    assert caps.speed is False
    assert caps.voices is False
    assert caps.voice_clone is False
    assert caps.languages == ()


def test_qwen3_placeholder_refuses_unapproved_runtime() -> None:
    adapter = Qwen3TTSAdapter()
    with pytest.raises(ArenaError) as exc_info:
        adapter.load_model("models/qwen3-tts-0.6b", num_threads=4)
    assert exc_info.value.code == 1002
    assert exc_info.value.error_type == "experimental_runtime_unavailable"
    assert adapter.is_loaded is False


def test_qwen3_placeholder_never_fakes_audio() -> None:
    adapter = Qwen3TTSAdapter()
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")
