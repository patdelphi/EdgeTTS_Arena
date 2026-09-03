from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.target_device_concurrent_calibration import run_calibration


def _args(tmp_path: Path, **overrides):
    values = {
        "models": ["model-a", "model-b"],
        "text": "concurrent calibration",
        "voice": None,
        "language": None,
        "seed": 42,
        "threads": 2,
        "runs": 2,
        "models_config": tmp_path / "models.yaml",
        "app_config": tmp_path / "app.yaml",
        "output_dir": tmp_path / "calibration",
        "max_rtf_slowdown_ratio": 1.6,
        "max_concurrent_rtf": 2.0,
        "max_concurrent_peak_rss_mb": 4000.0,
        "no_zip": False,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _runner_factory(*, slowdown: float = 1.4, concurrent_rtf: float = 1.4, rss: float = 3200.0):
    state = {"run": 0}

    def runner(*, text, model_ids, execution_mode, cpu_threads_per_model, config):
        state["run"] += 1
        pressure = execution_mode == "concurrent"
        results = []
        for index, model_id in enumerate(model_ids):
            baseline_rtf = 1.0 + index * 0.1
            rtf = concurrent_rtf + index * 0.1 if pressure else baseline_rtf
            if pressure and concurrent_rtf == 1.4:
                rtf = baseline_rtf * slowdown
            results.append(
                {
                    "model_id": model_id,
                    "status": "success",
                    "metrics": {
                        "rtf": rtf,
                        "peak_rss_mb": rss + index * 10.0 if pressure else 2800.0 + index * 10.0,
                        "avg_cpu_usage_pct": 180.0 if pressure else 95.0,
                    },
                    "error": None,
                }
            )
        return {
            "run_id": f"run-{state['run']}",
            "requested_cpu_threads_per_model": cpu_threads_per_model,
            "cpu_threads_per_model": 1 if pressure else cpu_threads_per_model,
            "total_threads_budget": len(model_ids) if pressure else cpu_threads_per_model,
            "resource_warnings": ["clamped"] if pressure else [],
            "results": results,
        }

    return runner


def test_concurrent_calibration_pairs_baseline_and_pressure(tmp_path: Path) -> None:
    report = run_calibration(_args(tmp_path), benchmark_runner=_runner_factory())
    assert report["passed"] is True
    assert report["scope"] == "target-device-concurrent-calibration"
    assert len(report["pairs"]) == 2
    assert report["pairs"][0]["sequential_run_id"] == "run-1"
    assert report["pairs"][0]["concurrent_run_id"] == "run-2"
    assert report["pairs"][0]["concurrent_execution"]["effective_threads_per_model"] == 1
    assert report["aggregate_by_model"]["model-a"]["rtf_slowdown_ratio"]["mean"] == pytest.approx(1.4)
    assert Path(report["archive"]).is_file()
    assert (tmp_path / "calibration" / "calibration_report.json").is_file()
    assert (tmp_path / "calibration" / "environment.json").is_file()


def test_concurrent_calibration_fails_explicit_thresholds(tmp_path: Path) -> None:
    report = run_calibration(
        _args(tmp_path, no_zip=True),
        benchmark_runner=_runner_factory(concurrent_rtf=2.5, rss=4500.0),
    )
    assert report["passed"] is False
    failed = {item["name"] for item in report["checks"] if not item["ok"]}
    assert {"max_rtf_slowdown_ratio", "max_concurrent_rtf", "max_concurrent_peak_rss_mb"} <= failed


def test_concurrent_calibration_requires_two_to_four_unique_models(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2-4 unique"):
        run_calibration(_args(tmp_path, models=["same", "same"]), benchmark_runner=_runner_factory())
    with pytest.raises(ValueError, match="2-4 unique"):
        run_calibration(
            _args(tmp_path, models=["a", "b", "c", "d", "e"]),
            benchmark_runner=_runner_factory(),
        )


def test_concurrent_calibration_records_partial_failure(tmp_path: Path) -> None:
    state = {"count": 0}

    def runner(*, text, model_ids, execution_mode, cpu_threads_per_model, config):
        state["count"] += 1
        results = []
        for model_id in model_ids:
            failed = execution_mode == "concurrent" and model_id == "model-b"
            results.append(
                {
                    "model_id": model_id,
                    "status": "error" if failed else "success",
                    "metrics": None if failed else {"rtf": 1.0, "peak_rss_mb": 1000.0, "avg_cpu_usage_pct": 80.0},
                    "error": {"type": "resource_error"} if failed else None,
                }
            )
        return {
            "run_id": f"run-{state['count']}",
            "requested_cpu_threads_per_model": cpu_threads_per_model,
            "cpu_threads_per_model": cpu_threads_per_model,
            "total_threads_budget": cpu_threads_per_model * len(model_ids),
            "resource_warnings": [],
            "results": results,
        }

    report = run_calibration(
        _args(
            tmp_path,
            runs=1,
            no_zip=True,
            max_rtf_slowdown_ratio=None,
            max_concurrent_rtf=None,
            max_concurrent_peak_rss_mb=None,
        ),
        benchmark_runner=runner,
    )
    assert report["passed"] is False
    assert report["checks"][0]["name"] == "all_pairs_successful"
    assert report["pairs"][0]["success"] is False


def test_concurrent_calibration_rejects_stale_output_by_default(tmp_path: Path) -> None:
    args = _args(tmp_path, runs=1)
    run_calibration(args, benchmark_runner=_runner_factory())
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_calibration(args, benchmark_runner=_runner_factory())


def test_concurrent_calibration_overwrite_cleans_stale_evidence(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    root.mkdir(parents=True)
    stale = root / "stale.json"
    stale.write_text("{}", encoding="utf-8")
    root.with_suffix(".zip").write_bytes(b"old archive")

    report = run_calibration(
        _args(tmp_path, runs=1, overwrite=True), benchmark_runner=_runner_factory()
    )
    assert report["passed"] is True
    assert not stale.exists()
    assert Path(report["archive"]).is_file()
