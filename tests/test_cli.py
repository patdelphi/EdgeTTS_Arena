from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_module_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "cli-smoke.wav"
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "edgetts_arena",
            "dummy",
            "--text",
            "CLI smoke test",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.stat().st_size > 44
