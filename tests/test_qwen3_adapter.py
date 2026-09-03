from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edgetts_arena.adapters.qwen3_adapter import Qwen3TTSAdapter
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


class FakeQwen:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_supported_speakers(self):
        return ["vivian", "ryan", "ono_anna"]

    def get_supported_languages(self):
        return ["auto", "chinese", "english", "japanese"]

    def generate_custom_voice(self, **kwargs):
        self.calls.append(dict(kwargs))
        return [np.linspace(-0.2, 0.2, 160, dtype=np.float32)], 24000


def _model_dir(tmp_path: Path, *, model_type: str = "custom_voice", model_size: str = "0b6") -> Path:
    root = tmp_path / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps({"tts_model_type": model_type, "tts_model_size": model_size}),
        encoding="utf-8",
    )
    return root


def test_qwen3_custom_voice_cpu_contract(tmp_path: Path) -> None:
    engine = FakeQwen()
    captured: dict[str, object] = {}

    def factory(model_dir: str, num_threads: int):
        captured["model_dir"] = model_dir
        captured["num_threads"] = num_threads
        return engine

    model_dir = _model_dir(tmp_path)
    adapter = Qwen3TTSAdapter(model_factory=factory, runtime_version="0.1.1-test")
    adapter.load_model(str(model_dir), num_threads=3)

    assert captured["model_dir"] == str(model_dir.resolve())
    assert captured["num_threads"] == 3
    assert adapter.available_voices == ("Vivian", "Ryan", "Ono_Anna")

    output = adapter.infer("hello from qwen", voice="ryan", language="en")
    assert output.sample_rate == 24000
    assert output.audio.shape == (160,)
    assert output.metadata["runtime"] == "QwenLM/Qwen3-TTS qwen-tts"
    assert output.metadata["runtime_version"] == "0.1.1-test"
    assert output.metadata["voice"] == "Ryan"
    assert output.metadata["language"] == "English"
    assert output.metadata["quantization"] == "fp32"
    assert engine.calls[-1] == {
        "text": "hello from qwen",
        "speaker": "Ryan",
        "language": "English",
    }


def test_qwen3_rejects_wrong_checkpoint_variant(tmp_path: Path) -> None:
    adapter = Qwen3TTSAdapter(model_factory=lambda *_args: FakeQwen())
    with pytest.raises(ValueError, match="CustomVoice"):
        adapter.load_model(str(_model_dir(tmp_path, model_type="base")))


def test_qwen3_rejects_non_06b_checkpoint(tmp_path: Path) -> None:
    adapter = Qwen3TTSAdapter(model_factory=lambda *_args: FakeQwen())
    with pytest.raises(ValueError, match="0.6B"):
        adapter.load_model(str(_model_dir(tmp_path, model_size="1b7")))


def test_qwen3_capability_conflicts_are_explicit(tmp_path: Path) -> None:
    adapter = Qwen3TTSAdapter(model_factory=lambda *_args: FakeQwen())
    adapter.load_model(str(_model_dir(tmp_path)))

    with pytest.raises(ArenaError) as speed_error:
        adapter.infer("x", speed=1.1)
    assert speed_error.value.code == 1003

    with pytest.raises(ArenaError) as seed_error:
        adapter.infer("x", seed=1)
    assert seed_error.value.code == 1003

    with pytest.raises(ArenaError) as instruct_error:
        adapter.infer("x", instruct="very happy")
    assert instruct_error.value.code == 1003


def test_qwen3_language_and_voice_validation(tmp_path: Path) -> None:
    adapter = Qwen3TTSAdapter(model_factory=lambda *_args: FakeQwen())
    adapter.load_model(str(_model_dir(tmp_path)))

    with pytest.raises(ValueError, match="unknown Qwen3-TTS speaker"):
        adapter.infer("x", voice="not-a-speaker")
    with pytest.raises(ValueError, match="unsupported Qwen3-TTS language"):
        adapter.infer("x", language="xx")


def test_qwen3_requires_load() -> None:
    adapter = Qwen3TTSAdapter(model_factory=lambda *_args: FakeQwen())
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")
