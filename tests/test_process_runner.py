import sys
from pathlib import Path

import pytest

from edgetts_arena.core import ProcessRunner, ProcessTimeoutError
from tests.process_targets import add, fail, sleep_for


def test_process_runner_success() -> None:
    result = ProcessRunner().run(add, 2, 3, timeout_sec=2.0)
    assert result.status == "success"
    assert result.value == 5
    assert result.exit_code == 0


def test_process_runner_reports_worker_error() -> None:
    result = ProcessRunner().run(fail, timeout_sec=2.0)
    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert "expected worker failure" in result.error_message


def test_process_runner_terminates_timeout() -> None:
    with pytest.raises(ProcessTimeoutError):
        ProcessRunner().run(sleep_for, 1.0, timeout_sec=0.05)


def test_external_python_worker_runs_dummy_protocol(tmp_path: Path) -> None:
    audio_path = tmp_path / "external.wav"
    task = {
        "adapter": "dummy",
        "model_path": "",
        "text": "external worker smoke",
        "num_threads": 1,
        "infer_kwargs": {},
        "audio_path": str(audio_path),
    }
    result = ProcessRunner().run_external_worker(
        sys.executable,
        "single",
        task,
        timeout_sec=5.0,
    )
    assert result.status == "success"
    assert result.exit_code == 0
    assert result.value["status"] == "success"
    assert result.value["metrics"]["audio_duration_ms"] > 0
    assert audio_path.exists()
    assert audio_path.stat().st_size > 44


def test_external_python_worker_reports_spawn_failure(tmp_path: Path) -> None:
    missing = tmp_path / "missing-python"
    result = ProcessRunner().run_external_worker(
        str(missing),
        "single",
        {},
        timeout_sec=1.0,
    )
    assert result.status == "error"
    assert result.termination == "spawn_error"
    assert "failed to start external worker" in (result.error_message or "")
