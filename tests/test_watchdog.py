from __future__ import annotations

from pathlib import Path
from typing import Any

from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.benchmark_suite import BenchmarkPresetSuite, RepeatedBenchmarkService
from edgetts_arena.core.config import ResourceGuardSettings
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec, ModelStatus
from edgetts_arena.core.process_runner import ProcessResult, ProcessRunner, ProcessTimeoutError
from edgetts_arena.core.resource_guard import ResourceGuard


class TimeoutRunner:
    def run(self, target: Any, *args: Any, timeout_sec: float, **kwargs: Any) -> ProcessResult:
        raise ProcessTimeoutError(f"timeout after {timeout_sec}")


class CrashRunner:
    def run(self, target: Any, *args: Any, timeout_sec: float, **kwargs: Any) -> ProcessResult:
        return ProcessResult(
            status="error",
            error_type="WorkerExited",
            error_message="worker crashed",
            exit_code=9,
        )


def _registry(*, keep_in_memory: bool = False) -> ModelRegistry:
    return ModelRegistry(
        [
            ModelSpec(
                id="dummy",
                name="Dummy Adapter",
                adapter="dummy",
                enabled=True,
                keep_in_memory=keep_in_memory,
                num_threads=1,
            )
        ],
        adapter_factories={"dummy": DummyTTSAdapter},
    )


def _guard() -> ResourceGuard:
    return ResourceGuard(
        ResourceGuardSettings(
            min_available_memory_mb_soft=1,
            min_available_memory_mb_hard=1,
            min_available_memory_mb_per_concurrent_model=1,
            max_concurrent_models=4,
        )
    )


def test_single_model_hard_timeout_maps_to_3001_and_recovers_status(tmp_path: Path) -> None:
    registry = _registry()
    service = BenchmarkService(
        registry,
        _guard(),
        RunArtifactStore(tmp_path / "exports"),
        process_runner=TimeoutRunner(),  # type: ignore[arg-type]
        inference_timeout_sec=0.01,
    )
    data = service.run(text="timeout", model_ids=["dummy"])
    result = data["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == 3001
    assert result["error"]["type"] == "inference_timeout"
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED


def test_single_model_worker_crash_maps_to_3002(tmp_path: Path) -> None:
    service = BenchmarkService(
        _registry(),
        _guard(),
        RunArtifactStore(tmp_path / "exports"),
        process_runner=CrashRunner(),  # type: ignore[arg-type]
    )
    result = service.run(text="crash", model_ids=["dummy"])["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == 3002
    assert result["error"]["type"] == "worker_exited"
    assert result["error"]["exit_code"] == 9


def test_repeated_suite_uses_spawn_worker_and_writes_representative_audio(tmp_path: Path) -> None:
    registry = _registry()
    store = RunArtifactStore(tmp_path / "exports")
    service = RepeatedBenchmarkService(
        registry,
        _guard(),
        store,
        preset_suite=BenchmarkPresetSuite.load("config/benchmark_presets.json"),
        process_runner=ProcessRunner(),
        inference_timeout_sec=5.0,
    )
    data = service.run_suite(
        model_ids=["dummy"],
        case_ids=["TC-01"],
        warmup_runs=1,
        measured_runs=2,
        cpu_threads_per_model=1,
        config={"speed": 1.0, "seed": 7},
    )
    result = data["results"][0]
    assert result["status"] == "success"
    assert result["successful_runs"] == 2
    assert len(result["measurements"]) == 2
    assert result["measurements"][0]["audio_url"] is not None
    assert result["measurements"][1]["audio_url"] is None
    assert store.get_audio_file(data["run_id"], "TC-01__dummy.wav").is_file()
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED


def test_repeated_suite_group_timeout_maps_to_3001(tmp_path: Path) -> None:
    registry = _registry()
    service = RepeatedBenchmarkService(
        registry,
        _guard(),
        RunArtifactStore(tmp_path / "exports"),
        preset_suite=BenchmarkPresetSuite.load("config/benchmark_presets.json"),
        process_runner=TimeoutRunner(),  # type: ignore[arg-type]
        inference_timeout_sec=0.5,
    )
    result = service.run_suite(
        model_ids=["dummy"],
        case_ids=["TC-01"],
        warmup_runs=1,
        measured_runs=2,
    )["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == 3001
    assert "1.500s" in result["error"]["message"]
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED


def test_keep_in_memory_suite_stays_in_process_with_explicit_warning(tmp_path: Path) -> None:
    registry = _registry(keep_in_memory=True)
    service = RepeatedBenchmarkService(
        registry,
        _guard(),
        RunArtifactStore(tmp_path / "exports"),
        preset_suite=BenchmarkPresetSuite.load("config/benchmark_presets.json"),
        process_runner=ProcessRunner(),
        inference_timeout_sec=1.0,
    )
    result = service.run_suite(
        model_ids=["dummy"],
        case_ids=["TC-01"],
        warmup_runs=0,
        measured_runs=1,
    )["results"][0]
    assert result["status"] == "success"
    assert any("hard process timeout is not enforced" in warning for warning in result["warnings"])
    assert registry.get_record("dummy").status == ModelStatus.READY


def test_process_runner_records_timeout_cleanup_diagnostics() -> None:
    from tests.process_targets import sleep_for

    runner = ProcessRunner()
    try:
        runner.run(sleep_for, 1.0, timeout_sec=0.05)
    except ProcessTimeoutError as exc:
        diagnostics = exc.diagnostics()
    else:
        raise AssertionError("expected ProcessTimeoutError")
    assert diagnostics["pid"] is not None
    assert diagnostics["exit_code"] is not None
    assert diagnostics["elapsed_ms"] >= 0
    assert diagnostics["termination"] in {"timeout_terminate", "timeout_kill"}
    assert diagnostics["oom_suspected"] is False


def test_process_runner_reports_nonzero_exit_without_queue_result() -> None:
    from tests.process_targets import hard_exit

    result = ProcessRunner().run(hard_exit, 7, timeout_sec=2.0)
    assert result.status == "error"
    assert result.error_type == "WorkerExited"
    assert result.exit_code == 7
    assert result.termination == "exit_code"
    assert result.oom_suspected is False
    assert result.diagnostics()["pid"] is not None


def test_process_runner_sigkill_is_only_oom_suspected() -> None:
    import os
    import pytest

    if os.name == "nt":
        pytest.skip("SIGKILL semantics are POSIX-specific")
    from tests.process_targets import sigkill_self

    result = ProcessRunner().run(sigkill_self, timeout_sec=2.0)
    assert result.status == "error"
    assert result.signal_name == "SIGKILL"
    assert result.oom_suspected is True
    assert "possible OOM or external kill" in (result.error_message or "")


def test_single_model_exposes_worker_diagnostics(tmp_path: Path) -> None:
    service = BenchmarkService(
        _registry(),
        _guard(),
        RunArtifactStore(tmp_path / "exports"),
        process_runner=CrashRunner(),  # type: ignore[arg-type]
    )
    result = service.run(text="crash", model_ids=["dummy"])["results"][0]
    assert "worker" in result
    assert result["worker"]["exit_code"] == 9


def test_repeated_suite_exposes_worker_diagnostics(tmp_path: Path) -> None:
    registry = _registry()
    service = RepeatedBenchmarkService(
        registry,
        _guard(),
        RunArtifactStore(tmp_path / "exports"),
        preset_suite=BenchmarkPresetSuite.load("config/benchmark_presets.json"),
        process_runner=ProcessRunner(),
        inference_timeout_sec=5.0,
    )
    result = service.run_suite(
        model_ids=["dummy"],
        case_ids=["TC-01"],
        warmup_runs=0,
        measured_runs=1,
    )["results"][0]
    assert result["worker"]["pid"] is not None
    assert result["worker"]["exit_code"] == 0
    assert result["worker"]["termination"] == "normal"
