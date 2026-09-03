from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from edgetts_arena.adapters.cosyvoice_adapter import WETEXT_REQUIRED_FILES, validate_wetext_assets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_wetext(output: Path) -> dict[str, object]:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("modelscope is required; install the pinned CosyVoice runtime first") from exc

    source = Path(snapshot_download("pengzhendong/wetext")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for relative in WETEXT_REQUIRED_FILES:
        source_file = source / relative
        if not source_file.is_file():
            raise SystemExit(f"WeText source asset missing: {source_file}")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_file, target)
        copied[relative] = sha256_file(target)

    validate_wetext_assets(output)
    manifest = {
        "source": "ModelScope:pengzhendong/wetext",
        "purpose": "CosyVoice wetext==0.0.4 offline TN frontend",
        "files": copied,
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch the local WeText FST assets required by the CosyVoice adapter."
    )
    parser.add_argument(
        "--output",
        default="models/cosyvoice/wetext",
        help="Destination directory for explicit local FST files.",
    )
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    manifest = prepare_wetext(output)
    print(json.dumps({"output": str(output), **manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
