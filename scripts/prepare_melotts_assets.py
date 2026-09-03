from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_REPO_ID = "myshell-ai/MeloTTS-Chinese"
DEFAULT_REVISION = "082ca057e44f1e52ec47e1622a30286019e8a3ef"
DEFAULT_OUTPUT = Path("models/melotts/zh")
MODEL_FILES = ("config.json", "checkpoint.pth")
NLTK_PACKAGES = ("averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "cmudict")


def _sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(root: Path) -> dict[str, object]:
    missing = [name for name in MODEL_FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"MeloTTS assets are incomplete under {root}; missing: {', '.join(missing)}")
    files = {
        name: {"bytes": (root / name).stat().st_size, "sha256": _sha256(root / name)}
        for name in MODEL_FILES
    }
    return {"files": files}


def prepare_assets(
    output: Path,
    *,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = DEFAULT_REVISION,
    prepare_nltk: bool = True,
) -> dict[str, object]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is required inside the dedicated MeloTTS environment") from exc

    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if prepare_nltk:
        try:
            import nltk
        except ImportError as exc:
            raise SystemExit("nltk is required inside the dedicated MeloTTS environment") from exc
        for package in NLTK_PACKAGES:
            if not nltk.download(package, quiet=True):
                raise RuntimeError(f"failed to download NLTK resource: {package}")

    for filename in MODEL_FILES:
        hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=str(output))

    model_descriptor = {"language": "ZH", "config_path": "config.json", "ckpt_path": "checkpoint.pth"}
    (output / "model.json").write_text(
        json.dumps(model_descriptor, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_assets(output)
    manifest = {
        "repo_id": repo_id,
        "resolved_revision": revision,
        "output": str(output),
        "nltk_packages": list(NLTK_PACKAGES) if prepare_nltk else [],
        **validation,
    }
    (output / "asset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare pinned MeloTTS Chinese assets for offline Arena use.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-nltk", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = prepare_assets(
        args.output,
        repo_id=args.repo_id,
        revision=args.revision,
        prepare_nltk=not args.skip_nltk,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
