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
from edgetts_arena.utils import write_wav


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


class ExternalOnlyRunner:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, target: Any, *args: Any, timeout_sec: float, **kwargs: Any) -> ProcessResult:
        raise AssertionError("spawn runner must not be used when worker_python is configured")

    def run_external_worker(
        self,
        python_executable: str,
        mode: str,
        task: dict[str, Any],
        *,
        timeout_sec: float,
    ) -> ProcessResult:
        self.calls.append(
            {
                "python": python_executable,
                "mode": mode,
                "task": dict(task),
                "timeout_sec": timeout_sec,
            }
        )
        adapter = DummyTTSAdapter()
        adapter.load_model(num_threads=int(task["num_threads"]))
        output = adapter.infer(str(task["text"]), seed=0)
        write_wav(str(task["audio_path"]), output.audio, output.sample_rate)
        metrics = {
            "inference_time_ms": 1.0,
            "audio_duration_ms": float(len(output.audio) / output.sample_rate * 1000.0),
            "rtf": 0.001,
            "ttfb_ms": None,
            "peak_rss_mb": 1.0,
            "rss_delta_mb": 0.0,
            "avg_cpu_usage_pct": 1.0,
        }
        if mode == "single":
            value = {
                "status": "success",
                "metrics": metrics,
                "metadata": {"runtime": "fake-external"},
                "error": None,
            }
        elif mode == "repeated":
            measurements = [
                {
                    "repeat": repeat,
                    "status": "success",
                    "metrics": metrics,
                    "wrote_representative_audio": repeat == 1,
                    "error": None,
                }
                for repeat in range(1, int(task["measured_runs"]) + 1)
            ]
            value = {
                "status": "success",
                "measurements": measurements,
                "metadata": {"runtime": "fake-external"},
                "audio_written": True,
                "error": None,
            }
        else:
            raise AssertionError(f"unexpected external worker mode: {mode}")
        return ProcessResult(
            status="success",
            value=value,
            exit_code=0,
            pid=4242,
            elapsed_ms=5.0,
            termination="normal",
        )


def _registry(*, keep_in_memory: bool = False, worker_python: str = "") -> ModelRegistry:
    return ModelRegistry(
        [
            ModelSpec(
                id="dummy",
                name="Dummy Adapter",
                adapter="dummy",
                enabled=True,
                keep_in_memory=keep_in_memory,
                num_threads=1,
                worker_python=worker_python,
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


def test_single_model_routes_to_configured_external_python(tmp_path: Path) -> None:
    runner = ExternalOnlyRunner()
    registry = _registry(worker_python="/dedicated/model/python")
    store = RunArtifactStore(tmp_path / "exports")
    service = BenchmarkService(
        registry,
        _guard(),
        store,
        process_runner=runner,  # type: ignore[arg-type]
        inference_timeout_sec=7.0,
    )
    data = service.run(text="external single", model_ids=["dummy"], cpu_threads_per_model=1)
    result = data["results"][0]
    assert result["status"] == "success"
    assert result["worker"]["pid"] == 4242
    assert any("dedicated external Python worker" in warning for warning in result["warnings"])
    assert len(runner.calls) == 1
    assert runner.calls[0]["python"] == "/dedicated/model/python"
    assert runner.calls[0]["mode"] == "single"
    assert runner.calls[0]["timeout_sec"] == 7.0
    assert store.get_audio_file(data["run_id"], "dummy.wav").is_file()
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED


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


def test_repeated_suite_routes_group_to_configured_external_python(tmp_path: Path) -> None:
    runner = ExternalOnlyRunner()
    registry = _registry(worker_python="/dedicated/model/python")
    store = RunArtifactStore(tmp_path / "exports")
    service = RepeatedBenchmarkService(
        registry,
        _guard(),
        store,
        preset_suite=BenchmarkPresetSuite.load("config/benchmark_presets.json"),
        process_runner=runner,  # type: ignore[arg-type]
        inference_timeout_sec=4.0,
    )
    data = service.run_suite(
        model_ids=["dummy"],
        case_ids=["TC-01"],
        warmup_runs=1,
        measured_runs=2,
        cpu_threads_per_model=1,
    )
    result = data["results"][0]
    assert result["status"] == "success"
    assert result["successful_runs"] == 2
    assert result["worker"]["pid"] == 4242
    assert any("dedicated external Python worker" in warning for warning in result["warnings"])
    assert len(runner.calls) == 1
    assert runner.calls[0]["python"] == "/dedicated/model/python"
    assert runner.calls[0]["mode"] == "repeated"
    assert runner.calls[0]["timeout_sec"] == 12.0
    assert runner.calls[0]["task"]["warmup_runs"] == 1
    assert runner.calls[0]["task"]["measured_runs"] == 2
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
