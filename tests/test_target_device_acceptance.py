from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.target_device_acceptance import run_acceptance


def _args(tmp_path: Path, **overrides):
    values = {
        "model": "qwen3-native",
        "model_path": str(tmp_path / "model.json"),
        "text": "target device test",
        "voice": "Vivian",
        "language": "zh",
        "seed": 42,
        "threads": 2,
        "runs": 3,
        "output_dir": tmp_path / "acceptance",
        "require_arch": "aarch64,arm64",
        "max_rtf": 2.0,
        "max_peak_rss_mb": 3500.0,
        "no_zip": False,
        "overwrite": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _environment(*, cpu_threads_per_model: int):
    return {
        "arch": "aarch64",
        "cpu_effective_cores": 4,
        "available_ram_effective_gb": 7.5,
        "thread_settings": {
            "cpu_threads_per_model": cpu_threads_per_model,
            "openblas_num_threads": None,
        },
    }


def _gate_factory():
    counter = {"value": 0}

    def gate(args):
        counter["value"] += 1
        Path(args.output).write_bytes(b"RIFF" + b"0" * 100)
        report = {
            "metrics": {
                "rtf": 1.5 + counter["value"] * 0.1,
                "peak_rss_mb": 3000.0 + counter["value"],
                "inference_time_ms": 10000.0,
            },
            "metadata": {"quantization": "int8"},
        }
        Path(args.report).write_text(json.dumps(report), encoding="utf-8")
        return report

    return gate


def test_target_device_acceptance_passes_and_builds_archive(tmp_path: Path) -> None:
    report = run_acceptance(
        _args(tmp_path), gate_runner=_gate_factory(), environment_collector=_environment
    )
    assert report["passed"] is True
    assert report["aggregate"]["rtf"]["max"] == 1.8
    assert Path(report["archive"]).is_file()
    assert (tmp_path / "acceptance" / "environment.json").is_file()
    assert (tmp_path / "acceptance" / "acceptance_report.json").is_file()


def test_target_device_acceptance_fails_worst_case_and_bad_openblas(tmp_path: Path) -> None:
    def gate(args):
        return {
            "metrics": {
                "rtf": 2.5,
                "peak_rss_mb": 3600.0,
                "inference_time_ms": 10000.0,
            }
        }

    def environment(*, cpu_threads_per_model: int):
        data = _environment(cpu_threads_per_model=cpu_threads_per_model)
        data["thread_settings"]["openblas_num_threads"] = "8"
        return data

    report = run_acceptance(
        _args(tmp_path, no_zip=True), gate_runner=gate, environment_collector=environment
    )
    assert report["passed"] is False
    failed = {item["name"] for item in report["checks"] if not item["ok"]}
    assert {"native_openblas_unset", "max_rtf", "max_peak_rss_mb"} <= failed


def test_target_device_acceptance_rejects_stale_output_by_default(tmp_path: Path) -> None:
    args = _args(tmp_path, runs=1)
    run_acceptance(args, gate_runner=_gate_factory(), environment_collector=_environment)
    with pytest.raises(FileExistsError, match="--overwrite"):
        run_acceptance(args, gate_runner=_gate_factory(), environment_collector=_environment)


def test_target_device_acceptance_overwrite_cleans_stale_evidence(tmp_path: Path) -> None:
    root = tmp_path / "acceptance"
    root.mkdir(parents=True)
    stale = root / "stale.txt"
    stale.write_text("old evidence", encoding="utf-8")
    root.with_suffix(".zip").write_bytes(b"old archive")

    report = run_acceptance(
        _args(tmp_path, runs=1, overwrite=True),
        gate_runner=_gate_factory(),
        environment_collector=_environment,
    )
    assert report["passed"] is True
    assert not stale.exists()
    assert Path(report["archive"]).is_file()
