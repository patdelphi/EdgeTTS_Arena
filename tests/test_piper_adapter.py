from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from edgetts_arena.adapters import PiperTTSAdapter
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


@dataclass
class FakeConfig:
    sample_rate: int = 22_050
    num_speakers: int = 2
    speaker_id_map: dict[str, int] | None = None
    default_speaker_id: int = 0
    espeak_voice: str = "en-us"
    piper_version: str = "test-model-v1"

    def __post_init__(self) -> None:
        if self.speaker_id_map is None:
            self.speaker_id_map = {"alice": 0, "bob": 1}


@dataclass
class FakeChunk:
    audio_float_array: np.ndarray
    sample_rate: int = 22_050


class FakeSynthesisConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


class FakeVoice:
    def __init__(self, config: FakeConfig | None = None) -> None:
        self.config = config or FakeConfig()
        self.calls: list[tuple[str, FakeSynthesisConfig]] = []

    def synthesize(self, text: str, syn_config: FakeSynthesisConfig):
        self.calls.append((text, syn_config))
        yield FakeChunk(np.full(100, 0.1, dtype=np.float32), self.config.sample_rate)
        yield FakeChunk(np.full(50, -0.1, dtype=np.float32), self.config.sample_rate)


def _voice_files(tmp_path: Path) -> Path:
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake onnx")
    Path(f"{model}.json").write_text("{}", encoding="utf-8")
    return model


def _adapter(tmp_path: Path, voice: FakeVoice | None = None) -> tuple[PiperTTSAdapter, FakeVoice, Path]:
    model = _voice_files(tmp_path)
    fake_voice = voice or FakeVoice()

    def loader(path: str, *, use_cuda: bool = False) -> FakeVoice:
        assert Path(path) == model.resolve()
        assert use_cuda is False
        return fake_voice

    adapter = PiperTTSAdapter(
        voice_loader=loader,
        synthesis_config_factory=FakeSynthesisConfig,
        runtime_version="test-runtime",
    )
    return adapter, fake_voice, model


def test_piper_requires_load(tmp_path: Path) -> None:
    adapter, _, _ = _adapter(tmp_path)
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")


def test_piper_load_and_infer_normalizes_output(tmp_path: Path) -> None:
    adapter, voice, model = _adapter(tmp_path)
    adapter.load_model(str(model), num_threads=3)

    output = adapter.infer(" hello ", voice="bob", speed=2.0)

    assert output.sample_rate == 22_050
    assert output.audio.dtype == np.float32
    assert output.audio.shape == (150,)
    assert output.metadata["voice"] == "bob"
    assert output.metadata["runtime"] == "piper-tts"
    assert output.metadata["runtime_version"] == "test-runtime"
    assert output.metadata["threads_requested"] == 3
    assert output.metadata["chunk_count"] == 2
    assert voice.calls[-1][0] == "hello"
    assert voice.calls[-1][1].speaker_id == 1
    assert voice.calls[-1][1].length_scale == pytest.approx(0.5)


def test_piper_stream_yields_sentence_chunks(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model))

    chunks = list(adapter.infer_stream("stream me", voice="alice", speed=1.0))

    assert len(chunks) == 2
    assert chunks[0].metadata["streaming"] is True
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1


def test_piper_seed_is_capability_conflict(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model))

    with pytest.raises(ArenaError) as exc_info:
        adapter.infer("hello", seed=7)
    assert exc_info.value.code == 1003


def test_piper_directory_path_must_be_unambiguous(tmp_path: Path) -> None:
    first = tmp_path / "a.onnx"
    second = tmp_path / "b.onnx"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    Path(f"{first}.json").write_text("{}", encoding="utf-8")
    Path(f"{second}.json").write_text("{}", encoding="utf-8")
    adapter = PiperTTSAdapter(
        voice_loader=lambda *_args, **_kwargs: FakeVoice(),
        synthesis_config_factory=FakeSynthesisConfig,
    )

    with pytest.raises(ValueError, match="multiple Piper models"):
        adapter.load_model(str(tmp_path))


def test_piper_single_speaker_rejects_unknown_voice(tmp_path: Path) -> None:
    voice = FakeVoice(FakeConfig(num_speakers=1, speaker_id_map={}))
    adapter, _, model = _adapter(tmp_path, voice=voice)
    adapter.load_model(str(model))

    assert adapter.available_voices == ("default",)
    with pytest.raises(ValueError, match="only one speaker"):
        adapter.infer("hello", voice="bob")


def test_piper_unload_releases_voice(tmp_path: Path) -> None:
    adapter, _, model = _adapter(tmp_path)
    adapter.load_model(str(model))
    adapter.unload_model()

    assert adapter.is_loaded is False
    assert adapter.available_voices == ()
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")
