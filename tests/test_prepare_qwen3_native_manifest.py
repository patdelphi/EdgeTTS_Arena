from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_qwen3_native_manifest import create_manifest


def test_prepare_qwen3_native_manifest_uses_relative_paths(tmp_path: Path) -> None:
    binary = tmp_path / "runtime" / "qwen_tts"
    binary.parent.mkdir()
    binary.write_text("fake", encoding="utf-8")
    model_dir = tmp_path / "models" / "qwen"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps({"tts_model_type": "custom_voice", "tts_model_size": "0b6"}), encoding="utf-8"
    )
    output = tmp_path / "models" / "qwen3-native" / "int8" / "model.json"
    manifest = create_manifest(argparse.Namespace(
        binary=binary,
        model_dir=model_dir,
        output=output,
        runtime_revision="abc123",
        quantization="int8",
        default_voice="Vivian",
        default_language="Chinese",
    ))
    assert output.is_file()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved == manifest
    assert not Path(saved["binary"]).is_absolute()
    assert not Path(saved["model_dir"]).is_absolute()
    assert saved["runtime_revision"] == "abc123"
    assert saved["quantization"] == "int8"
