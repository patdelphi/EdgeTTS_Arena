from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

RUNTIME = "qwen3-tts-c"
DEFAULT_REVISION = "e56ec7e6eabbed608b13bfbd3fba431708b2077f"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a portable Qwen3 native runtime manifest.")
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--runtime-revision", default=DEFAULT_REVISION)
    parser.add_argument("--quantization", choices=("bf16", "int8", "int4"), default="int8")
    parser.add_argument("--default-voice", default="Vivian")
    parser.add_argument("--default-language", default="English")
    return parser


def write_manifest(
    *,
    binary: Path,
    model_dir: Path,
    output: Path,
    runtime_revision: str = DEFAULT_REVISION,
    quantization: str = "int8",
    default_voice: str = "Vivian",
    default_language: str = "English",
) -> dict[str, object]:
    if quantization not in {"bf16", "int8", "int4"}:
        raise ValueError("quantization must be one of bf16/int8/int4")
    binary = binary.expanduser().resolve()
    model_dir = model_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"native runtime binary not found: {binary}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Qwen3 model directory not found: {model_dir}")
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Qwen3 config not found: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("tts_model_type") or "").lower() != "custom_voice":
        raise ValueError("native manifest requires Qwen3 CustomVoice model")
    if str(config.get("tts_model_size") or "").lower() != "0b6":
        raise ValueError("native manifest requires Qwen3 0.6B model")

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runtime": RUNTIME,
        "runtime_revision": str(runtime_revision),
        "binary": os.path.relpath(binary, output.parent),
        "model_dir": os.path.relpath(model_dir, output.parent),
        "quantization": quantization,
        "default_voice": str(default_voice),
        "default_language": str(default_language),
    }
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def create_manifest(args: argparse.Namespace) -> dict[str, object]:
    return write_manifest(
        binary=args.binary,
        model_dir=args.model_dir,
        output=args.output,
        runtime_revision=str(args.runtime_revision),
        quantization=str(args.quantization),
        default_voice=str(args.default_voice),
        default_language=str(args.default_language),
    )


def main() -> int:
    args = build_parser().parse_args()
    manifest = create_manifest(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
