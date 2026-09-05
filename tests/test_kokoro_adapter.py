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
    # zf_xiaoxiao 现在在 available_voices 中，因为中文 (cmn) 已添加到支持的语言列表
    assert "zf_xiaoxiao" in adapter.available_voices
    # 使用中文语音和语言应该可以正常工作
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


class FakeZhKokoro:
    """Stand-in for the dedicated v1.1-zh engine (distinct voice names)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_voices(self) -> list[str]:
        return ["af_maple", "bf_vale", "zf_001", "zf_002", "zm_010"]

    def create(self, phonemes: str, **kwargs: Any):
        self.calls.append((phonemes, kwargs))
        return np.full(4800, 0.1, dtype=np.float32), 24_000


def _fake_g2p_factory() -> Any:
    def _factory() -> Any:
        def _g2p(text: str) -> tuple[str, dict[str, Any]]:
            return ("ni3 hao2", {"raw": text})

        return _g2p

    return _factory


def _zh_adapter(
    tmp_path: Path, *, zh_factory: Any = None
) -> tuple[KokoroTTSAdapter, FakeKokoro, FakeZhKokoro]:
    """Adapter with a v1.0 engine plus a discovered zh model under ``tmp_path/zh``."""
    _files(tmp_path)  # kokoro-v1.0.onnx + voices-v1.0.bin
    v1_engine = FakeKokoro()
    zh_engine = FakeZhKokoro()

    zh_dir = tmp_path / "zh"
    zh_dir.mkdir()
    (zh_dir / "kokoro-v1.1-zh.onnx").write_bytes(b"zhonnx")
    (zh_dir / "voices-v1.1-zh.bin").write_bytes(b"zhvoices")
    (zh_dir / "config.json").write_text('{"vocab": {}}', encoding="utf-8")

    def default_zh_factory(model_file, voices_file, config_file, num_threads):
        assert Path(model_file).name == "kokoro-v1.1-zh.onnx"
        assert Path(voices_file).name == "voices-v1.1-zh.bin"
        assert Path(config_file).name == "config.json"
        assert num_threads == 3
        return zh_engine

    adapter = KokoroTTSAdapter(
        kokoro_factory=lambda *_args: v1_engine,
        runtime_version="test-runtime",
        zh_kokoro_factory=zh_factory or default_zh_factory,
        zh_g2p_factory=_fake_g2p_factory(),
    )
    adapter.load_model(str(tmp_path), num_threads=3)
    return adapter, v1_engine, zh_engine


def test_kokoro_discovers_zh_model_beside_v1(tmp_path: Path) -> None:
    adapter, _, _ = _zh_adapter(tmp_path)
    assert adapter._zh_available is True
    assert adapter._zh_model_path is not None
    assert adapter._zh_model_path.name == "kokoro-v1.1-zh.onnx"


def test_kokoro_chinese_routes_to_zh_model(tmp_path: Path) -> None:
    adapter, v1_engine, zh_engine = _zh_adapter(tmp_path)
    output = adapter.infer("你好，世界。")

    assert output.metadata["model_version"] == "v1.1-zh"
    assert output.metadata["phonemizer"] == "misaki-zh"
    assert output.metadata["language"] == "cmn"
    assert output.metadata["voice"] == "zf_001"
    # Phonemes (not raw text) are fed with is_phonemes so tones bypass espeak-ng.
    assert zh_engine.calls[-1][0] == "ni3 hao2"
    assert zh_engine.calls[-1][1]["is_phonemes"] is True
    assert zh_engine.calls[-1][1]["voice"] == "zf_001"
    assert v1_engine.calls == []


def test_kokoro_chinese_explicit_zh_voice_honoured(tmp_path: Path) -> None:
    adapter, _, zh_engine = _zh_adapter(tmp_path)
    output = adapter.infer("你好", voice="zf_002")
    assert output.metadata["voice"] == "zf_002"
    assert output.metadata["model_version"] == "v1.1-zh"
    assert zh_engine.calls[-1][1]["voice"] == "zf_002"


def test_kokoro_chinese_explicit_v1_voice_stays_on_v1(tmp_path: Path) -> None:
    adapter, v1_engine, zh_engine = _zh_adapter(tmp_path)
    output = adapter.infer("你好", voice="zf_xiaoxiao", language="cmn")
    assert output.metadata["model_version"] == "v1.0"
    assert v1_engine.calls[-1][1]["lang"] == "cmn"
    assert zh_engine.calls == []


def test_kokoro_chinese_falls_back_when_zh_load_fails(tmp_path: Path) -> None:
    def boom(*_args: Any) -> Any:
        raise RuntimeError("zh session failed")

    adapter, v1_engine, _ = _zh_adapter(tmp_path, zh_factory=boom)
    output = adapter.infer("你好")
    assert adapter._zh_load_failed is True
    assert output.metadata["model_version"] == "v1.0"
    assert v1_engine.calls[-1][1]["lang"] == "cmn"


def test_kokoro_zh_disabled_without_config(tmp_path: Path) -> None:
    # Only onnx + voices, no config.json -> zh path disabled, Chinese uses v1.0.
    _files(tmp_path)
    zh_dir = tmp_path / "zh"
    zh_dir.mkdir()
    (zh_dir / "kokoro-v1.1-zh.onnx").write_bytes(b"zhonnx")
    (zh_dir / "voices-v1.1-zh.bin").write_bytes(b"zhvoices")
    v1_engine = FakeKokoro()
    adapter = KokoroTTSAdapter(
        kokoro_factory=lambda *_args: v1_engine, runtime_version="test-runtime"
    )
    adapter.load_model(str(tmp_path), num_threads=3)
    assert adapter._zh_available is False
    output = adapter.infer("你好")
    assert output.metadata["model_version"] == "v1.0"


def test_zh_g2p_keeps_mandarin_tones() -> None:
    # Root-cause guard: misaki ZHG2P emits tone digits, which the v1.1-zh vocab
    # contains but espeak-ng 'cmn' + the v1.0 vocab silently dropped.
    pytest.importorskip("misaki")
    from misaki import zh

    phonemes, _ = zh.ZHG2P(version="1.1")("你好")
    assert any(tone in phonemes for tone in "12345")
