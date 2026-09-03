from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from edgetts_arena.adapters.cosyvoice_adapter import (
    CosyVoiceTTSAdapter,
    WETEXT_REQUIRED_FILES,
    build_offline_wetext_normalizer,
    validate_wetext_assets,
)
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


class FakeCosyVoice:
    sample_rate = 24000

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool, float]] = []

    def list_available_spks(self):
        return ["中文女", "中文男"]

    def inference_sft(self, text, speaker, stream=False, speed=1.0):
        self.calls.append((text, speaker, stream, speed))
        yield {"tts_speech": np.full((1, 120), 0.1, dtype=np.float32)}
        yield {"tts_speech": np.full((1, 80), -0.1, dtype=np.float32)}


class FakeWetextNormalizer:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls.append(kwargs)

    def normalize(self, text: str) -> str:
        lang = str(self.kwargs["lang"])
        return f"{lang}:{text}"


def _model_dir(tmp_path: Path) -> Path:
    path = tmp_path / "CosyVoice-300M-SFT"
    path.mkdir()
    (path / "cosyvoice.yaml").write_text("sample_rate: 24000\n", encoding="utf-8")
    return path


def _wetext_dir(tmp_path: Path) -> Path:
    path = tmp_path / "wetext"
    for relative in WETEXT_REQUIRED_FILES:
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fst")
    return path


def test_cosyvoice_sft_contract(tmp_path: Path) -> None:
    engine = FakeCosyVoice()
    adapter = CosyVoiceTTSAdapter(model_factory=lambda path, threads: engine, runtime_version="test")
    adapter.load_model(str(_model_dir(tmp_path)), num_threads=2)
    assert adapter.available_voices == ("中文女", "中文男")
    output = adapter.infer("你好", voice="中文女", speed=1.2)
    assert output.sample_rate == 24000
    assert output.audio.shape == (200,)
    assert output.metadata["mode"] == "sft"
    assert output.metadata["threads"] == 2
    assert engine.calls[-1] == ("你好", "中文女", False, 1.2)


def test_cosyvoice_stream_contract(tmp_path: Path) -> None:
    engine = FakeCosyVoice()
    adapter = CosyVoiceTTSAdapter(model_factory=lambda path, threads: engine)
    adapter.load_model(str(_model_dir(tmp_path)))
    chunks = list(adapter.infer_stream("stream", voice="中文男"))
    assert [len(chunk.audio) for chunk in chunks] == [120, 80]
    assert all(chunk.metadata["streaming"] is True for chunk in chunks)
    assert engine.calls[-1][2] is True


def test_cosyvoice_rejects_non_sft_and_seed(tmp_path: Path) -> None:
    adapter = CosyVoiceTTSAdapter(model_factory=lambda path, threads: FakeCosyVoice())
    bad = tmp_path / "missing-yaml"
    bad.mkdir()
    with pytest.raises(ValueError):
        adapter.load_model(str(bad))

    adapter.load_model(str(_model_dir(tmp_path)))
    with pytest.raises(ArenaError) as exc_info:
        adapter.infer("x", seed=1)
    assert exc_info.value.code == 1003


def test_cosyvoice_requires_load() -> None:
    adapter = CosyVoiceTTSAdapter(model_factory=lambda path, threads: FakeCosyVoice())
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")


def test_validate_wetext_assets_requires_all_files(tmp_path: Path) -> None:
    path = _wetext_dir(tmp_path)
    assert validate_wetext_assets(path) == path.resolve()
    (path / "en/tn/tagger.fst").unlink()
    with pytest.raises(FileNotFoundError, match="en/tn/tagger.fst"):
        validate_wetext_assets(path)


def test_offline_wetext_normalizer_uses_explicit_local_paths(tmp_path: Path) -> None:
    FakeWetextNormalizer.calls.clear()
    root = _wetext_dir(tmp_path)
    normalizer_cls = build_offline_wetext_normalizer(FakeWetextNormalizer, root)
    normalizer = normalizer_cls(remove_erhua=False)

    assert normalizer.normalize("你好 123") == "zh:你好 123"
    assert normalizer.normalize("hello 123") == "en:hello 123"
    assert len(FakeWetextNormalizer.calls) == 2
    assert all("tagger_path" in call and "verbalizer_path" in call for call in FakeWetextNormalizer.calls)
    assert all(str(root.resolve()) in str(call["tagger_path"]) for call in FakeWetextNormalizer.calls)


def test_offline_wetext_remove_erhua_uses_explicit_variant(tmp_path: Path) -> None:
    FakeWetextNormalizer.calls.clear()
    root = _wetext_dir(tmp_path)
    normalizer_cls = build_offline_wetext_normalizer(FakeWetextNormalizer, root)
    normalizer_cls(lang="zh", remove_erhua=True)
    assert FakeWetextNormalizer.calls[0]["verbalizer_path"].endswith("zh/tn/verbalizer_remove_erhua.fst")
