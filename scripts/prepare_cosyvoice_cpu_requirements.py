from __future__ import annotations

import argparse
from pathlib import Path

BLOCKED_PREFIXES = (
    "--extra-index-url",
    "deepspeed==",
    "onnxruntime-gpu==",
    "tensorrt-cu12",
    "torch==",
    "torchaudio==",
    "openai-whisper==",
)


def build_cpu_requirements(source: Path, target: Path) -> dict[str, object]:
    source = source.expanduser().resolve()
    target = target.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"CosyVoice requirements file not found: {source}")

    kept: list[str] = []
    removed: list[str] = []
    for raw in source.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if stripped.startswith(BLOCKED_PREFIXES):
            removed.append(stripped)
        else:
            kept.append(raw)

    if not any(line.startswith("onnxruntime-gpu==") for line in removed):
        raise ValueError("pinned CosyVoice requirements no longer declare onnxruntime-gpu; review CPU filter")
    if not any(line.startswith("torch==") for line in removed):
        raise ValueError("pinned CosyVoice requirements no longer declare torch; review CPU filter")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return {
        "source": str(source),
        "target": str(target),
        "removed": removed,
        "kept_lines": len(kept),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the pinned CPU-only CosyVoice requirements file.")
    parser.add_argument("--source", type=Path, default=Path("vendor/CosyVoice/requirements.txt"))
    parser.add_argument("--output", type=Path, default=Path("vendor/CosyVoice/requirements.cpu.txt"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = build_cpu_requirements(args.source, args.output)
    print(f"wrote {result['target']}")
    for line in result["removed"]:
        print(f"removed: {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
