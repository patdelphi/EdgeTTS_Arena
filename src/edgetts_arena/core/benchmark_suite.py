from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService, iso_utc, new_run_id, utc_now
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.core.model_registry import ModelRegistry, ModelStatus
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.system_info import collect_system_environment
from edgetts_arena.utils import write_wav


_METRIC_NAMES = (
    "inference_time_ms",
    "audio_duration_ms",
    "rtf",
    "ttfb_ms",
    "peak_rss_mb",
    "rss_delta_mb",
    "avg_cpu_usage_pct",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    name: str
    text: str
    focus: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["focus"] = list(self.focus)
        return payload


@dataclass(frozen=True, slots=True)
class BenchmarkPresetSuite:
    version: str
    warmup_runs: int
    measured_runs: int
    cases: tuple[BenchmarkCase, ...]

    @classmethod
    def load(cls, path: str | Path = "config/benchmark_presets.json") -> "BenchmarkPresetSuite":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        defaults = raw.get("defaults") or {}
        cases = tuple(
            BenchmarkCase(
                id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                text=str(item["text"]),
                focus=tuple(str(value) for value in item.get("focus") or ()),
            )
            for item in raw.get("presets") or ()
        )
        if not cases:
            raise ValueError("benchmark preset suite must contain at least one case")
        if len({case.id for case in cases}) != len(cases):
            raise ValueError("benchmark preset ids must be unique")
        warmup_runs = int(defaults.get("warmup_runs", 1))
        measured_runs = int(defaults.get("measured_runs", 3))
        if warmup_runs < 0 or measured_runs < 1:
            raise ValueError("invalid benchmark repeat defaults")
        return cls(
            version=str(raw.get("version") or "unknown"),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
            cases=cases,
        )

    def select(self, case_ids: Iterable[str] | None = None) -> list[BenchmarkCase]:
        if case_ids is None:
            return list(self.cases)
        requested = list(case_ids)
        by_id = {case.id: case for case in self.cases}
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise ArenaError(
                1001,
                f"unknown benchmark case(s): {', '.join(missing)}",
                error_type="validation_error",
            )
        return [by_id[case_id] for case_id in requested]


def summarize_values(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "p95": None,
            "variance": None,
        }
    array = np.asarray(numeric, dtype=np.float64)
    return {
        "count": len(numeric),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "p95": float(np.percentile(array, 95)),
        "variance": float(statistics.pvariance(numeric)) if len(numeric) > 1 else 0.0,
    }


def aggregate_measurements(measurements: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    successful = [item for item in measurements if item.get("status") == "success"]
    return {
        metric: summarize_values((item.get("metrics") or {}).get(metric) for item in successful)
        for metric in _METRIC_NAMES
    }


class RepeatedBenchmarkService:
    """Runs preset cases with warm-up and repeated measurements under one run_id."""

    def __init__(
        self,
        registry: ModelRegistry,
        resource_guard: ResourceGuard,
        artifact_store: RunArtifactStore,
        *,
        preset_suite: BenchmarkPresetSuite | None = None,
        metrics_collector: MetricsCollector | None = None,
    ) -> None:
        self.registry = registry
        self.resource_guard = resource_guard
        self.artifact_store = artifact_store
        self.preset_suite = preset_suite or BenchmarkPresetSuite.load()
        self.metrics = metrics_collector or MetricsCollector()

    def run_suite(
        self,
        *,
        model_ids: list[str],
        case_ids: list[str] | None = None,
        cpu_threads_per_model: int = 4,
        warmup_runs: int | None = None,
        measured_runs: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not model_ids or len(model_ids) > 4:
            raise ArenaError(1001, "suite models must contain 1 to 4 model ids", error_type="validation_error")
        if len(set(model_ids)) != len(model_ids):
            raise ArenaError(1001, "suite model ids must be unique", error_type="validation_error")

        self.resource_guard.assess(execution_mode="sequential")
        threads = self.resource_guard.clamp_threads(cpu_threads_per_model)
        warmups = self.preset_suite.warmup_runs if warmup_runs is None else int(warmup_runs)
        repeats = self.preset_suite.measured_runs if measured_runs is None else int(measured_runs)
        if not 0 <= warmups <= 10:
            raise ArenaError(1001, "warmup_runs must be between 0 and 10", error_type="validation_error")
        if not 1 <= repeats <= 20:
            raise ArenaError(1001, "measured_runs must be between 1 and 20", error_type="validation_error")

        selected_cases = self.preset_suite.select(case_ids)
        config = dict(config or {})
        run_id = new_run_id()
        started_at = utc_now()
        self.artifact_store.create_run(run_id)
        environment = collect_system_environment(cpu_threads_per_model=threads)

        results: list[dict[str, Any]] = []
        for case in selected_cases:
            for model_id in model_ids:
                results.append(
                    self._run_case_model(
                        run_id=run_id,
                        case=case,
                        model_id=model_id,
                        num_threads=threads,
                        warmup_runs=warmups,
                        measured_runs=repeats,
                        config=config,
                    )
                )

        completed_at = utc_now()
        data = {
            "run_id": run_id,
            "started_at": iso_utc(started_at),
            "completed_at": iso_utc(completed_at),
            "suite_version": self.preset_suite.version,
            "execution_mode": "sequential",
            "cpu_threads_per_model": threads,
            "warmup_runs": warmups,
            "measured_runs": repeats,
            "cases": [case.to_dict() for case in selected_cases],
            "models": list(model_ids),
            "results": results,
        }
        report = {
            "schema_version": "benchmark-suite-report-v1",
            "request": {
                "case_ids": [case.id for case in selected_cases],
                "models": list(model_ids),
                "cpu_threads_per_model": threads,
                "warmup_runs": warmups,
                "measured_runs": repeats,
                "config": config,
            },
            "data": data,
        }
        self.artifact_store.write_json(run_id, "environment.json", environment)
        self.artifact_store.write_json(run_id, "benchmark_report.json", report)
        return data

    def _run_case_model(
        self,
        *,
        run_id: str,
        case: BenchmarkCase,
        model_id: str,
        num_threads: int,
        warmup_runs: int,
        measured_runs: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        measurements: list[dict[str, Any]] = []
        record = None
        loaded = False
        representative_audio_url: str | None = None
        metadata: dict[str, Any] | None = None
        try:
            record = self.registry.get_record(model_id)
            info = self.registry.model_info(model_id)
            if record.status == ModelStatus.UNAVAILABLE:
                raise ArenaError(1002, f"model '{model_id}' is unavailable", error_type="model_unavailable")
            infer_kwargs = BenchmarkService._normalize_infer_config(
                model_id=model_id,
                info=info,
                capabilities=info.get("capabilities") or {},
                config=config,
                warnings=warnings,
            )
            adapter = self.registry.load(model_id, num_threads=num_threads)
            loaded = True
            self.registry.set_status(model_id, ModelStatus.BUSY)

            for _ in range(warmup_runs):
                adapter.infer(case.text, **infer_kwargs)

            for repeat_index in range(1, measured_runs + 1):
                try:
                    output, metrics = self.metrics.measure_inference(adapter, case.text, **infer_kwargs)
                    metadata = dict(output.metadata)
                    audio_url = None
                    if representative_audio_url is None:
                        filename = self.artifact_store.safe_audio_filename(f"{case.id}__{model_id}")
                        path = self.artifact_store.audio_output_path(run_id, filename)
                        write_wav(path, output.audio, output.sample_rate)
                        representative_audio_url = f"/api/v1/audio/download/{run_id}/{filename}"
                        audio_url = representative_audio_url
                    measurements.append(
                        {
                            "repeat": repeat_index,
                            "status": "success",
                            "metrics": metrics.to_dict(),
                            "audio_url": audio_url,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    measurements.append(
                        {
                            "repeat": repeat_index,
                            "status": "error",
                            "metrics": None,
                            "audio_url": None,
                            "error": BenchmarkService._normalize_error(exc),
                        }
                    )

            success_count = sum(item["status"] == "success" for item in measurements)
            status = "success" if success_count == measured_runs else ("partial" if success_count else "error")
            return {
                "case_id": case.id,
                "model_id": model_id,
                "status": status,
                "audio_url": representative_audio_url,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "successful_runs": success_count,
                "measurements": measurements,
                "aggregate": aggregate_measurements(measurements),
                "warnings": warnings,
                "metadata": metadata,
                "error": self._first_measurement_error(measurements) if status != "success" else None,
            }
        except Exception as exc:
            return {
                "case_id": case.id,
                "model_id": model_id,
                "status": "error",
                "audio_url": None,
                "warmup_runs": warmup_runs,
                "measured_runs": measured_runs,
                "successful_runs": 0,
                "measurements": measurements,
                "aggregate": aggregate_measurements(measurements),
                "warnings": warnings,
                "metadata": metadata,
                "error": BenchmarkService._normalize_error(exc),
            }
        finally:
            if record is not None and loaded:
                try:
                    if record.spec.keep_in_memory:
                        self.registry.set_status(model_id, ModelStatus.READY)
                    else:
                        self.registry.unload(model_id)
                except Exception:
                    self.registry.set_status(model_id, ModelStatus.ERROR, error="cleanup failed")

    @staticmethod
    def _first_measurement_error(measurements: list[dict[str, Any]]) -> dict[str, Any] | None:
        for measurement in measurements:
            if measurement.get("error"):
                return measurement["error"]
        return None
