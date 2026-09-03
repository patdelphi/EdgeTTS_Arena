from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_qwen3_native_variants import create_variants


def test_prepare_native_variants_creates_matched_int8_int4_manifests(tmp_path: Path) -> None:
    binary = tmp_path / "runtime" / "qwen_tts"
    binary.parent.mkdir()
    binary.write_text("fake", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"tts_model_type": "custom_voice", "tts_model_size": "0b6"}),
        encoding="utf-8",
    )
    variants = create_variants(
        argparse.Namespace(
            binary=binary,
            model_dir=model_dir,
            output_root=tmp_path / "variants",
            runtime_revision="same-revision",
            default_voice="Vivian",
            default_language="Chinese",
        )
    )
    assert variants["int8"]["quantization"] == "int8"
    assert variants["int4"]["quantization"] == "int4"
    for quant in ("int8", "int4"):
        saved = json.loads((tmp_path / "variants" / quant / "model.json").read_text(encoding="utf-8"))
        assert saved["runtime_revision"] == "same-revision"
        assert saved["default_voice"] == "Vivian"
        assert saved["default_language"] == "Chinese"
