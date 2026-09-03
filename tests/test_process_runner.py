import sys
from pathlib import Path

import pytest

from edgetts_arena.core import ProcessRunner, ProcessTimeoutError
from edgetts_arena.core.process_runner import _tail_text
from tests.process_targets import add, fail, memory_error, sleep_for


def _external_dummy_task(audio_path: Path) -> dict[str, object]:
    return {
        "adapter": "dummy",
        "model_path": "",
        "text": "external worker smoke",
        "num_threads": 1,
        "infer_kwargs": {},
        "audio_path": str(audio_path),
    }


def test_process_runner_success() -> None:
    result = ProcessRunner().run(add, 2, 3, timeout_sec=2.0)
    assert result.status == "success"
    assert result.value == 5
    assert result.exit_code == 0
    assert result.mode == "spawn"
    assert result.runtime["protocol"] == "multiprocessing-spawn"
    assert result.runtime["python_version"]
    assert result.oom_suspected is False
    assert result.oom_classification == "none"
    assert result.oom_evidence is None


def test_process_runner_reports_worker_error() -> None:
    result = ProcessRunner().run(fail, timeout_sec=2.0)
    assert result.status == "error"
    assert result.error_type == "RuntimeError"
    assert "expected worker failure" in result.error_message
    assert result.mode == "spawn"
    assert result.oom_classification == "none"


def test_process_runner_reports_explicit_memory_error() -> None:
    result = ProcessRunner().run(memory_error, timeout_sec=2.0)
    assert result.status == "error"
    assert result.error_type == "MemoryError"
    assert "expected worker memory error" in (result.error_message or "")
    assert result.oom_suspected is True
    assert result.oom_classification == "worker_memory_error"
    assert result.oom_evidence == {"source": "worker-exception", "error_type": "MemoryError"}


def test_process_runner_terminates_timeout_without_oom_label() -> None:
    with pytest.raises(ProcessTimeoutError) as captured:
        ProcessRunner().run(sleep_for, 1.0, timeout_sec=0.05)
    diagnostics = captured.value.diagnostics()
    assert diagnostics["mode"] == "spawn"
    assert diagnostics["runtime"]["protocol"] == "multiprocessing-spawn"
    assert diagnostics["oom_suspected"] is False
    assert diagnostics["oom_classification"] == "none"
    assert diagnostics["oom_evidence"] is None


def test_external_python_worker_runs_dummy_protocol(tmp_path: Path) -> None:
    audio_path = tmp_path / "external.wav"
    result = ProcessRunner().run_external_worker(
        sys.executable,
        "single",
        _external_dummy_task(audio_path),
        timeout_sec=5.0,
    )
    assert result.status == "success"
    assert result.exit_code == 0
    assert result.mode == "external"
    assert result.runtime["protocol"] == "external-json"
    assert result.runtime["protocol_version"] == 1
    assert result.runtime["python_version"]
    assert result.runtime["python_implementation"]
    assert result.value["status"] == "success"
    assert "_worker_runtime" not in result.value
    assert result.value["metrics"]["audio_duration_ms"] > 0
    assert result.oom_classification == "none"
    assert audio_path.exists()
    assert audio_path.stat().st_size > 44


def test_external_python_worker_rejects_protocol_mismatch(tmp_path: Path, monkeypatch) -> None:
    import edgetts_arena.core.process_runner as process_runner_module

    monkeypatch.setattr(process_runner_module, "_EXTERNAL_PROTOCOL_VERSION", 999)
    result = ProcessRunner().run_external_worker(
        sys.executable,
        "single",
        _external_dummy_task(tmp_path / "mismatch.wav"),
        timeout_sec=5.0,
    )
    assert result.status == "error"
    assert result.error_type == "ExternalWorkerProtocolError"
    assert result.mode == "external"
    assert "expected 999, got 1" in (result.error_message or "")


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
    assert result.mode == "external"
    assert result.oom_classification == "none"
    assert "failed to start external worker" in (result.error_message or "")


def test_diagnostic_text_is_tail_bounded() -> None:
    text = "prefix-should-disappear\n" + ("x" * 10_000) + "\nimportant-tail"
    bounded = _tail_text(text, limit=256)
    assert bounded is not None
    assert len(bounded) <= 256
    assert bounded.startswith("...<diagnostic output truncated>...")
    assert bounded.endswith("important-tail")
    assert "prefix-should-disappear" not in bounded
