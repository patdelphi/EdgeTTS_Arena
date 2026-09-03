from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from edgetts_arena.adapters.melotts_adapter import MeloTTSAdapter
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


class FakeMelo:
    def __init__(self) -> None:
        self.hps = SimpleNamespace(data=SimpleNamespace(spk2id={"ZH": 0, "ZH_ALT": 1}, sampling_rate=44100))
        self.calls = []

    def tts_to_file(self, text, speaker_id, output_path=None, speed=1.0, quiet=False):
        self.calls.append((text, speaker_id, output_path, speed, quiet))
        return np.linspace(-0.2, 0.2, 128, dtype=np.float32)


def _descriptor(tmp_path: Path) -> Path:
    root = tmp_path / "melotts-zh"
    root.mkdir()
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "G.pth").write_bytes(b"checkpoint")
    (root / "model.json").write_text(
        json.dumps({"language": "ZH", "config_path": "config.json", "ckpt_path": "G.pth"}),
        encoding="utf-8",
    )
    return root


def test_melotts_local_descriptor_contract(tmp_path: Path) -> None:
    engine = FakeMelo()
    captured = {}
    def factory(**kwargs):
        captured.update(kwargs)
        return engine

    adapter = MeloTTSAdapter(tts_factory=factory, runtime_version="0.1.2-test")
    adapter.load_model(str(_descriptor(tmp_path)), num_threads=3)
    assert adapter.available_voices == ("ZH", "ZH_ALT")
    assert captured["language"] == "ZH"
    assert captured["use_hf"] is False
    assert captured["num_threads"] == 3

    output = adapter.infer("你好 hello", voice="ZH_ALT", speed=1.1)
    assert output.sample_rate == 44100
    assert output.audio.shape == (128,)
    assert output.metadata["language"] == "zh"
    assert output.metadata["runtime"] == "myshell-ai/MeloTTS"
    assert engine.calls[-1][1] == 1
    assert engine.calls[-1][3] == 1.1


def test_melotts_rejects_bad_descriptor_and_seed(tmp_path: Path) -> None:
    adapter = MeloTTSAdapter(tts_factory=lambda **kwargs: FakeMelo())
    root = tmp_path / "bad"
    root.mkdir()
    (root / "model.json").write_text(json.dumps({"language": "XX"}), encoding="utf-8")
    with pytest.raises(ValueError):
        adapter.load_model(str(root))

    adapter.load_model(str(_descriptor(tmp_path)))
    with pytest.raises(ArenaError) as exc_info:
        adapter.infer("x", seed=1)
    assert exc_info.value.code == 1003


def test_melotts_requires_load() -> None:
    adapter = MeloTTSAdapter(tts_factory=lambda **kwargs: FakeMelo())
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")
