from __future__ import annotations

import json
import zipfile

import pytest

from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_suite import (
    BenchmarkPresetSuite,
    RepeatedBenchmarkService,
    aggregate_measurements,
    summarize_values,
)
from edgetts_arena.core.config import ResourceGuardSettings
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec, ModelStatus
from edgetts_arena.core.resource_guard import ResourceGuard


def make_service(tmp_path) -> RepeatedBenchmarkService:
    registry = ModelRegistry(
        [
            ModelSpec(
                id="dummy",
                name="Dummy Adapter",
                adapter="dummy",
                enabled=True,
                keep_in_memory=True,
                num_threads=1,
            )
        ],
        adapter_factories={"dummy": DummyTTSAdapter},
    )
    guard = ResourceGuard(
        ResourceGuardSettings(
            min_available_memory_mb_soft=1,
            min_available_memory_mb_hard=1,
        )
    )
    return RepeatedBenchmarkService(
        registry,
        guard,
        RunArtifactStore(tmp_path / "exports"),
        preset_suite=BenchmarkPresetSuite.load("config/benchmark_presets.json"),
    )


def test_preset_suite_contains_fixed_tc01_to_tc05() -> None:
    suite = BenchmarkPresetSuite.load("config/benchmark_presets.json")
    assert [case.id for case in suite.cases] == ["TC-01", "TC-02", "TC-03", "TC-04", "TC-05"]
    assert suite.warmup_runs == 1
    assert suite.measured_runs == 3
    tc05 = suite.select(["TC-05"])[0]
    assert len(tc05.text) >= 300
    assert "占位" not in tc05.text


def test_preset_suite_rejects_unknown_case() -> None:
    suite = BenchmarkPresetSuite.load("config/benchmark_presets.json")
    with pytest.raises(ArenaError) as exc:
        suite.select(["TC-99"])
    assert exc.value.code == 1001


def test_summary_statistics_include_p95_and_variance() -> None:
    stats = summarize_values([1.0, 2.0, 3.0, None])
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(2.0)
    assert stats["median"] == pytest.approx(2.0)
    assert stats["min"] == pytest.approx(1.0)
    assert stats["max"] == pytest.approx(3.0)
    assert stats["p95"] == pytest.approx(2.9)
    assert stats["variance"] == pytest.approx(2 / 3)


def test_aggregate_preserves_null_ttfb_for_non_streaming_measurements() -> None:
    measurements = [
        {
            "status": "success",
            "metrics": {
                "inference_time_ms": 10.0,
                "audio_duration_ms": 100.0,
                "rtf": 0.1,
                "ttfb_ms": None,
                "peak_rss_mb": 50.0,
                "rss_delta_mb": 1.0,
                "avg_cpu_usage_pct": 20.0,
            },
        }
    ]
    aggregate = aggregate_measurements(measurements)
    assert aggregate["ttfb_ms"]["count"] == 0
    assert aggregate["ttfb_ms"]["mean"] is None


def test_run_suite_stream_yields_incremental_snapshots(tmp_path) -> None:
    service = make_service(tmp_path)
    snapshots = list(
        service.run_suite_stream(
            model_ids=["dummy"], case_ids=["TC-01", "TC-02"], warmup_runs=0, measured_runs=1
        )
    )
    # One in-progress snapshot per case×model pair, then a single final one.
    assert [complete for _data, complete in snapshots] == [False, False, True]
    assert len(snapshots[0][0]["results"]) == 1
    assert len(snapshots[1][0]["results"]) == 2
    assert snapshots[0][0]["completed_at"] is None
    final = snapshots[-1][0]
    assert final["completed_at"] is not None
    assert len(final["results"]) == 2
    assert [r["case_id"] for r in final["results"]] == ["TC-01", "TC-02"]


def test_repeated_suite_warmup_measurements_report_and_zip(tmp_path) -> None:
    service = make_service(tmp_path)
    data = service.run_suite(
        model_ids=["dummy"],
        case_ids=["TC-01", "TC-02"],
        warmup_runs=1,
        measured_runs=3,
        cpu_threads_per_model=2,
        config={"speed": 1.0, "seed": 7},
    )

    assert data["execution_mode"] == "sequential"
    assert data["warmup_runs"] == 1
    assert data["measured_runs"] == 3
    assert len(data["results"]) == 2
    for result in data["results"]:
        assert result["status"] == "success"
        assert result["successful_runs"] == 3
        assert len(result["measurements"]) == 3
        assert result["audio_url"] is not None
        assert result["aggregate"]["rtf"]["count"] == 3
        assert result["aggregate"]["inference_time_ms"]["p95"] is not None

    run_dir = service.artifact_store.run_dir(data["run_id"])
    report = json.loads((run_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    environment = json.loads((run_dir / "environment.json").read_text(encoding="utf-8"))
    assert report["schema_version"] == "benchmark-suite-report-v1"
    assert report["request"]["case_ids"] == ["TC-01", "TC-02"]
    assert environment["python_version"]
    assert "package_versions" in environment
    assert environment["thread_settings"]["cpu_threads_per_model"] == 2

    archive_path = service.artifact_store.build_export(data["run_id"])
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
    assert "audio/TC-01__dummy.wav" in names
    assert "audio/TC-02__dummy.wav" in names
    assert "benchmark_report.json" in names
    assert "environment.json" in names

    assert service.registry.get_record("dummy").status == ModelStatus.READY


def test_repeated_suite_model_failure_is_reported_not_raised(tmp_path) -> None:
    service = make_service(tmp_path)
    data = service.run_suite(
        model_ids=["missing"],
        case_ids=["TC-01"],
        warmup_runs=0,
        measured_runs=2,
    )
    result = data["results"][0]
    assert result["status"] == "error"
    assert result["error"]["code"] == 1002
