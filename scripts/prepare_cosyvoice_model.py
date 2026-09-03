from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_REPO_ID = "FunAudioLLM/CosyVoice-300M-SFT"
DEFAULT_REVISION = "fbb71de2afe387ed854eebd80b9f3d078c6b9869"
DEFAULT_OUTPUT = Path("models/cosyvoice/CosyVoice-300M-SFT")


def validate_snapshot(root: Path) -> dict[str, object]:
    root = root.expanduser().resolve()
    required = ("cosyvoice.yaml",)
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"CosyVoice snapshot is incomplete under {root}; missing: {', '.join(missing)}"
        )
    return {"required_files": list(required)}


def prepare_snapshot(
    output: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
) -> dict[str, object]:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required inside the dedicated CosyVoice environment") from exc

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(output))
    validation = validate_snapshot(output)
    manifest = {
        "repo_id": repo_id,
        "resolved_revision": revision,
        "output": str(output),
        **validation,
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare the pinned CosyVoice 300M SFT snapshot.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = prepare_snapshot(args.output, repo_id=args.repo_id, revision=args.revision)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
