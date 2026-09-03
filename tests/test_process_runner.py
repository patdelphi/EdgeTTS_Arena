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
