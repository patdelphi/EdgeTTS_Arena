from __future__ import annotations

import sys

import pytest

from edgetts_arena.core.persistent_worker import PersistentExternalWorker, WarmWorkerError


def _worker(model_id: str = "dummy") -> PersistentExternalWorker:
    # The dummy adapter needs no model files and lives in the current venv, so the
    # current interpreter is a valid "dedicated worker python" for the protocol test.
    return PersistentExternalWorker(
        python_executable=sys.executable,
        model_id=model_id,
        adapter="dummy",
        model_path="",
        num_threads=1,
    )


def test_warm_worker_loads_once_and_reuses_across_requests(tmp_path) -> None:
    worker = _worker()
    try:
        worker.start(timeout_sec=60.0)
        assert worker.is_alive()
        assert worker.pid is not None

        first_wav = tmp_path / "first.wav"
        result = worker.submit_infer(
            text="first request", infer_kwargs={"seed": 0},
            audio_path=str(first_wav), timeout_sec=30.0,
        )
        assert result.status == "success"
        assert result.value["status"] == "success"
        assert result.mode == "warm-external"
        assert first_wav.is_file() and first_wav.stat().st_size > 44

        # Second request reuses the same warm process (no reload).
        second_wav = tmp_path / "second.wav"
        result2 = worker.submit_infer(
            text="second request reuses the loaded model", infer_kwargs={"seed": 1},
            audio_path=str(second_wav), timeout_sec=30.0,
        )
        assert result2.value["status"] == "success"
        assert second_wav.is_file()
        assert result2.diagnostics()["pid"] == worker.pid
    finally:
        worker.shutdown()
    assert worker.is_alive() is False


def test_warm_worker_repeated_mode(tmp_path) -> None:
    worker = _worker()
    wav = tmp_path / "repeated.wav"
    try:
        worker.start(timeout_sec=60.0)
        result = worker.submit_repeated(
            text="repeat me", infer_kwargs={"seed": 2}, audio_path=str(wav),
            warmup_runs=1, measured_runs=2, timeout_sec=30.0,
        )
        assert result.value["status"] == "success"
        assert len(result.value["measurements"]) == 2
        assert result.value["audio_written"] is True
        assert wav.is_file()
    finally:
        worker.shutdown()


def test_warm_worker_self_reports_rss(tmp_path) -> None:
    worker = _worker()
    try:
        worker.start(timeout_sec=60.0)
        # The child self-reports RSS in the load frame; parent-side psutil of a
        # child is unreliable on Windows, so the self-report is authoritative.
        assert worker.rss_mb() is not None
        assert worker.rss_mb() > 1.0
    finally:
        worker.shutdown()


def test_warm_worker_start_failure_raises() -> None:
    worker = PersistentExternalWorker(
        python_executable="Z:/definitely/not/a/real/python",
        model_id="dummy", adapter="dummy", model_path="", num_threads=1,
    )
    with pytest.raises(WarmWorkerError):
        worker.start(timeout_sec=5.0)
    assert worker.is_alive() is False


def test_warm_worker_submit_after_shutdown_raises(tmp_path) -> None:
    worker = _worker()
    worker.start(timeout_sec=60.0)
    worker.shutdown()
    with pytest.raises(WarmWorkerError):
        worker.submit_infer(
            text="too late", infer_kwargs={}, audio_path=str(tmp_path / "x.wav"),
            timeout_sec=5.0,
        )
