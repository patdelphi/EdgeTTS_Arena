from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.prepare_qwen3_native_manifest import DEFAULT_REVISION, write_manifest
except ModuleNotFoundError:  # direct: python scripts/prepare_qwen3_native_variants.py
    from prepare_qwen3_native_manifest import DEFAULT_REVISION, write_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare matched Qwen3 native INT8/INT4 manifests for Arena Blind AB."
    )
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("models/qwen3-native"))
    parser.add_argument("--runtime-revision", default=DEFAULT_REVISION)
    parser.add_argument("--default-voice", default="Vivian")
    parser.add_argument("--default-language", default="Chinese")
    return parser


def create_variants(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    root = args.output_root.expanduser().resolve()
    common = {
        "binary": args.binary,
        "model_dir": args.model_dir,
        "runtime_revision": str(args.runtime_revision),
        "default_voice": str(args.default_voice),
        "default_language": str(args.default_language),
    }
    int8 = write_manifest(
        **common,
        quantization="int8",
        output=root / "int8" / "model.json",
    )
    int4 = write_manifest(
        **common,
        quantization="int4",
        output=root / "int4" / "model.json",
    )
    return {"int8": int8, "int4": int4}


def main() -> int:
    variants = create_variants(build_parser().parse_args())
    print(json.dumps(variants, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
