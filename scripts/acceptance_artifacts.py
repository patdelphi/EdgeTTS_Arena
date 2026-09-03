from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


def archive_path(root: Path) -> Path:
    return root.with_suffix(".zip")


def prepare_output_dir(root: Path, *, overwrite: bool = False) -> tuple[Path, Path]:
    root = root.expanduser().resolve()
    archive = archive_path(root)
    root_has_content = root.is_dir() and any(root.iterdir())
    archive_exists = archive.is_file()
    if (root_has_content or archive_exists) and not overwrite:
        occupied = []
        if root_has_content:
            occupied.append(str(root))
        if archive_exists:
            occupied.append(str(archive))
        raise FileExistsError(
            "acceptance output already exists; choose a new --output-dir or pass --overwrite: "
            + ", ".join(occupied)
        )
    if overwrite:
        if root.exists():
            if root.is_dir():
                shutil.rmtree(root)
            else:
                root.unlink()
        archive.unlink(missing_ok=True)
    root.mkdir(parents=True, exist_ok=True)
    return root, archive


def write_zip(root: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(root))
