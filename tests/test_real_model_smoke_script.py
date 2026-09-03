from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from edgetts_arena.core.base_adapter import TTSOutput


def _load_script_module():
    path = Path(__file__).parents[1] / "scripts" / "real_model_smoke.py"
    spec = importlib.util.spec_from_file_location("real_model_smoke", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAdapter:
    available_voices = ("demo",)

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def load_model(self, model_path: str, *, device: str, num_threads: int) -> None:
        assert model_path == "fake-model"
        assert device == "cpu"
        assert num_threads == 2
        self.loaded = True

    def infer(self, text: str, **kwargs):
        assert self.loaded is True
        assert text == "smoke text"
        assert kwargs["voice"] == "demo"
        return TTSOutput(
            audio=np.linspace(-0.1, 0.1, 240, dtype=np.float32),
            sample_rate=24000,
            metadata={"runtime": "fake"},
        )

    def unload_model(self) -> None:
        self.unloaded = True


def test_real_model_smoke_writes_audio_and_report(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    adapter = FakeAdapter()
    monkeypatch.setattr(module, "MeloTTSAdapter", lambda: adapter)
    args = argparse.Namespace(
        model="melotts",
        model_path="fake-model",
        text="smoke text",
        voice=None,
        threads=2,
        speed=1.0,
        output=str(tmp_path / "out.wav"),
        report=str(tmp_path / "report.json"),
    )
    report = module.run_gate(args)
    assert (tmp_path / "out.wav").stat().st_size > 44
    assert (tmp_path / "report.json").is_file()
    assert report["voice"] == "demo"
    assert report["metadata"]["runtime"] == "fake"
    assert report["metrics"]["ttfb_ms"] is None
    assert adapter.unloaded is True
