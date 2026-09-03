from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from edgetts_arena.adapters.cosyvoice_adapter import WETEXT_REQUIRED_FILES, validate_wetext_assets


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_selected_wetext_assets(output: Path, *, attempts: int = 3) -> None:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise SystemExit("modelscope is required; install the pinned CosyVoice runtime first") from exc

    output.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            # Do not force revision='master' on modelscope==1.20.0. For this
            # public model that path enters revision validation which may require
            # a token. Let the pinned SDK resolve its default public revision.
            snapshot_download(
                "pengzhendong/wetext",
                allow_file_pattern=list(WETEXT_REQUIRED_FILES),
                local_dir=str(output),
            )
            validate_wetext_assets(output)
            return
        except Exception as exc:  # network boundary: preserve exact final error below
            last_error = exc
            if attempt < attempts:
                delay = attempt * 2
                print(
                    f"WeText selective download attempt {attempt}/{attempts} failed: "
                    f"{type(exc).__name__}: {exc}; retrying in {delay}s",
                    flush=True,
                )
                time.sleep(delay)

    assert last_error is not None
    raise SystemExit(
        "failed to download the required CosyVoice WeText assets after "
        f"{attempts} attempts: {type(last_error).__name__}: {last_error}"
    ) from last_error


def prepare_wetext(output: Path) -> dict[str, object]:
    _download_selected_wetext_assets(output)
    root = validate_wetext_assets(output)
    files = {relative: sha256_file(root / relative) for relative in WETEXT_REQUIRED_FILES}
    manifest = {
        "source": "ModelScope:pengzhendong/wetext",
        "revision_resolution": "modelscope==1.20.0 default public revision",
        "download_mode": "selective allow_file_pattern",
        "purpose": "CosyVoice wetext==0.0.4 offline TN frontend",
        "files": files,
    }
    (root / "asset_manifest.json").write_text(
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
