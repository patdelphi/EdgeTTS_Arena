from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

DEFAULT_REPO_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_OUTPUT = Path("models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice")
REQUIRED_FILES = (
    "config.json",
    "model.safetensors",
    "speech_tokenizer/config.json",
    "speech_tokenizer/model.safetensors",
)


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_snapshot(root: Path) -> dict[str, Any]:
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Qwen3-TTS snapshot is incomplete under {root}; missing: {', '.join(missing)}"
        )

    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Qwen3-TTS config.json must be a JSON object")
    if str(config.get("tts_model_type") or "").lower() != "custom_voice":
        raise ValueError("prepared Qwen3-TTS checkpoint is not CustomVoice")
    if str(config.get("tts_model_size") or "").lower() != "0b6":
        raise ValueError("prepared Qwen3-TTS checkpoint is not the 0.6B (0b6) variant")

    files: dict[str, dict[str, object]] = {}
    for relative in REQUIRED_FILES:
        path = root / relative
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    return {
        "tts_model_type": config.get("tts_model_type"),
        "tts_model_size": config.get("tts_model_size"),
        "files": files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and validate the official Qwen3-TTS 0.6B CustomVoice snapshot."
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face revision/commit to pin.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Run this script inside the dedicated Qwen3 qwen-tts environment."
        ) from exc

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(args.repo_id, revision=args.revision)
    resolved_revision = str(info.sha)
    snapshot_download(
        repo_id=args.repo_id,
        revision=resolved_revision,
        local_dir=str(output),
    )
    validation = validate_snapshot(output)
    manifest = {
        "repo_id": args.repo_id,
        "requested_revision": args.revision,
        "resolved_revision": resolved_revision,
        "output": str(output),
        **validation,
    }
    manifest_path = output / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
