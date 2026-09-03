from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path
from typing import Any

from edgetts_arena.core.errors import ArenaError


_RUN_ID_RE = re.compile(r"^run_[A-Za-z0-9_-]+$")
_FILENAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.wav$", re.IGNORECASE)


class RunArtifactStore:
    def __init__(self, root: str | Path = "exports") -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        run_dir = (self.root / run_id).resolve()
        self._assert_within_root(run_dir)
        (run_dir / "audio").mkdir(parents=True, exist_ok=False)
        return run_dir

    def run_dir(self, run_id: str, *, must_exist: bool = True) -> Path:
        self._validate_run_id(run_id)
        path = (self.root / run_id).resolve()
        self._assert_within_root(path)
        if must_exist and not path.is_dir():
            raise ArenaError(1002, f"run '{run_id}' does not exist", error_type="run_not_found")
        return path

    def audio_output_path(self, run_id: str, filename: str) -> Path:
        self._validate_filename(filename)
        run_dir = self.run_dir(run_id)
        path = (run_dir / "audio" / filename).resolve()
        self._assert_within(path, run_dir / "audio")
        return path

    def get_audio_file(self, run_id: str, filename: str) -> Path:
        path = self.audio_output_path(run_id, filename)
        if not path.is_file():
            raise ArenaError(1002, f"audio file '{filename}' does not exist", error_type="audio_not_found")
        return path

    def write_json(self, run_id: str, filename: str, payload: Any) -> Path:
        if filename not in {"benchmark_report.json", "environment.json", "blind_scores.json"}:
            raise ArenaError(1001, "unsupported artifact filename", error_type="invalid_path")
        path = self.run_dir(run_id) / filename
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return path

    def build_export(self, run_id: str) -> Path:
        run_dir = self.run_dir(run_id)
        report = run_dir / "benchmark_report.json"
        environment = run_dir / "environment.json"
        if not report.is_file() or not environment.is_file():
            raise ArenaError(1002, f"run '{run_id}' is incomplete", error_type="run_incomplete")

        zip_path = (run_dir / f"{run_id}.zip").resolve()
        self._assert_within(zip_path, run_dir)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for audio_file in sorted((run_dir / "audio").glob("*.wav")):
                archive.write(audio_file, arcname=f"audio/{audio_file.name}")
            archive.write(report, arcname="benchmark_report.json")
            archive.write(environment, arcname="environment.json")
            blind_scores = run_dir / "blind_scores.json"
            if blind_scores.is_file():
                archive.write(blind_scores, arcname="blind_scores.json")
        return zip_path

    @staticmethod
    def safe_model_filename(model_id: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("._")
        if not safe:
            safe = "model"
        return f"{safe}.wav"

    def _assert_within_root(self, path: Path) -> None:
        self._assert_within(path, self.root)

    @staticmethod
    def _assert_within(path: Path, parent: Path) -> None:
        parent = parent.resolve()
        try:
            path.resolve().relative_to(parent)
        except ValueError as exc:
            raise ArenaError(1001, "path escapes export root", error_type="invalid_path") from exc

    @staticmethod
    def _validate_run_id(run_id: str) -> None:
        if not _RUN_ID_RE.fullmatch(run_id):
            raise ArenaError(1001, "invalid run_id", error_type="invalid_path")

    @staticmethod
    def _validate_filename(filename: str) -> None:
        if Path(filename).name != filename or not _FILENAME_RE.fullmatch(filename):
            raise ArenaError(1001, "invalid audio filename", error_type="invalid_path")
