from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService, iso_utc, new_run_id, utc_now
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.core.model_registry import ModelRegistry, ModelStatus
from edgetts_arena.core.persistent_worker import WarmWorkerError
from edgetts_arena.core.process_runner import ProcessRunner, ProcessTimeoutError
from edgetts_arena.core.residency import ResidencyManager
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.system_info import collect_system_environment
from edgetts_arena.core.worker_runtime import run_isolated_repeated_model
from edgetts_arena.defaults import read_default_text
from edgetts_arena.utils import write_wav

_DEFAULT_BENCHMARK_PRESETS = Path("config/benchmark_presets.json")
_METRICS = (
    "inference_time_ms", "audio_duration_ms", "rtf", "ttfb_ms",
    "peak_rss_mb", "rss_delta_mb", "avg_cpu_usage_pct",
)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    id: str
    name: str
    text: str
    focus: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["focus"] = list(self.focus)
        return value


@dataclass(frozen=True, slots=True)
class BenchmarkPresetSuite:
    version: str
    warmup_runs: int
    measured_runs: int
    cases: tuple[BenchmarkCase, ...]

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_BENCHMARK_PRESETS) -> "BenchmarkPresetSuite":
        preset_path = Path(path)
        if preset_path.exists():
            text = preset_path.read_text(encoding="utf-8")
        elif preset_path == _DEFAULT_BENCHMARK_PRESETS:
            text = read_default_text("benchmark_presets.json")
        else:
            raise FileNotFoundError(preset_path)
        raw = json.loads(text)
        defaults = raw.get("defaults") or {}
        cases = tuple(
            BenchmarkCase(
                id=str(item["id"]),
                name=str(item.get("name") or item["id"]),
                text=str(item["text"]),
                focus=tuple(map(str, item.get("focus") or ())),
            )
            for item in raw.get("presets") or ()
        )
        if not cases or len({case.id for case in cases}) != len(cases):
            raise ValueError("benchmark preset suite must contain unique cases")
        warmups = int(defaults.get("warmup_runs", 1))
        repeats = int(defaults.get("measured_runs", 3))
        if warmups < 0 or repeats < 1:
            raise ValueError("invalid benchmark repeat defaults")
        return cls(str(raw.get("version") or "unknown"), warmups, repeats, cases)

    def select(self, case_ids: Iterable[str] | None = None) -> list[BenchmarkCase]:
        if case_ids is None:
            return list(self.cases)
        requested = list(case_ids)
        by_id = {case.id: case for case in self.cases}
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise ArenaError(1001, f"unknown benchmark case(s): {', '.join(missing)}", error_type="validation_error")
        return [by_id[case_id] for case_id in requested]


def summarize_values(values: Iterable[float | int | None]) -> dict[str, float | int | None]:
    numeric = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not numeric:
        return {key: None if key != "count" else 0 for key in ("count", "mean", "median", "min", "max", "p95", "variance")}
    a = np.asarray(numeric, dtype=np.float64)
    return {
        "count": len(numeric), "mean": float(np.mean(a)), "median": float(np.median(a)),
        "min": float(np.min(a)), "max": float(np.max(a)), "p95": float(np.percentile(a, 95)),
        "variance": float(statistics.pvariance(numeric)) if len(numeric) > 1 else 0.0,
    }


def aggregate_measurements(measurements: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    successful = [m for m in measurements if m.get("status") == "success"]
    return {metric: summarize_values((m.get("metrics") or {}).get(metric) for m in successful) for metric in _METRICS}


class RepeatedBenchmarkService:
    """Warm-up + repeated measurements grouped by case/model under one run_id."""

    def __init__(
        self,
        registry: ModelRegistry,
        resource_guard: ResourceGuard,
        artifact_store: RunArtifactStore,
        *,
        preset_suite: BenchmarkPresetSuite | None = None,
        metrics_collector: MetricsCollector | None = None,
        process_runner: ProcessRunner | None = None,
        inference_timeout_sec: float = 60.0,
        isolate_model_processes: bool = True,
        residency: ResidencyManager | None = None,
    ) -> None:
        self.registry = registry
        self.resource_guard = resource_guard
        self.artifact_store = artifact_store
        self.preset_suite = preset_suite or BenchmarkPresetSuite.load()
        self.metrics = metrics_collector or MetricsCollector()
        self.process_runner = process_runner if process_runner is not None else (ProcessRunner() if isolate_model_processes else None)
        self.inference_timeout_sec = float(inference_timeout_sec)
        self.residency = residency

    def run_suite(
        self, *, model_ids: list[str], case_ids: list[str] | None = None,
        cpu_threads_per_model: int = 4, warmup_runs: int | None = None,
        measured_runs: int | None = None, config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Blocking wrapper that returns the final snapshot of :meth:`run_suite_stream`."""
        data: dict[str, Any] = {}
        for data, _complete in self.run_suite_stream(
            model_ids=model_ids, case_ids=case_ids, cpu_threads_per_model=cpu_threads_per_model,
            warmup_runs=warmup_runs, measured_runs=measured_runs, config=config,
        ):
            pass
        return data

    def run_suite_stream(
        self, *, model_ids: list[str], case_ids: list[str] | None = None,
        cpu_threads_per_model: int = 4, warmup_runs: int | None = None,
        measured_runs: int | None = None, config: dict[str, Any] | None = None,
    ) -> Iterator[tuple[dict[str, Any], bool]]:
        """Yield ``(data_snapshot, complete)`` after each case×model pair finishes.

        The final item (``complete=True``) is written to benchmark_report.json /
        environment.json and is exactly what :meth:`run_suite` returns.
        """
        if not model_ids or len(model_ids) > 4 or len(set(model_ids)) != len(model_ids):
            raise ArenaError(1001, "suite models must contain 1 to 4 unique model ids", error_type="validation_error")
        self.resource_guard.assess(execution_mode="sequential")
        threads = self.resource_guard.clamp_threads(cpu_threads_per_model)
        warmups = self.preset_suite.warmup_runs if warmup_runs is None else int(warmup_runs)
        repeats = self.preset_suite.measured_runs if measured_runs is None else int(measured_runs)
        if not 0 <= warmups <= 10 or not 1 <= repeats <= 20:
            raise ArenaError(1001, "warmup_runs must be 0..10 and measured_runs 1..20", error_type="validation_error")

        cases = self.preset_suite.select(case_ids)
        config = dict(config or {})
        run_id, started = new_run_id(), utc_now()
        self.artifact_store.create_run(run_id)

        # Generational eviction: drop models kept warm by a previous run that this
        # suite does not select, before loading anything new.
        residency_evicted: list[str] = []
        if self.residency is not None and self.residency.keep_warm:
            residency_evicted = self.residency.begin_run(list(model_ids))

        def snapshot(results: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
            payload = {
                "run_id": run_id, "started_at": iso_utc(started),
                "completed_at": iso_utc(utc_now()) if complete else None,
                "suite_version": self.preset_suite.version, "execution_mode": "sequential",
                "cpu_threads_per_model": threads, "warmup_runs": warmups, "measured_runs": repeats,
                "cases": [case.to_dict() for case in cases], "models": list(model_ids), "results": results,
            }
            if self.residency is not None:
                payload["residency"] = self.residency.snapshot()
            return payload

        results: list[dict[str, Any]] = []
        for case in cases:
            for model_id in model_ids:
                results.append(self._run_case_model(run_id, case, model_id, threads, warmups, repeats, config))
                yield snapshot(list(results), False), False

        # Keep-warm bookkeeping: trim residents beyond the memory budget, then record
        # the residency state (and what was evicted) into the run environment.
        residency_trimmed: list[str] = []
        if self.residency is not None and self.residency.keep_warm:
            residency_trimmed = self.residency.end_run(list(model_ids))

        data = snapshot(results, True)
        report = {
            "schema_version": "benchmark-suite-report-v1",
            "request": {"case_ids": [c.id for c in cases], "models": list(model_ids), "cpu_threads_per_model": threads,
                        "warmup_runs": warmups, "measured_runs": repeats, "config": config},
            "data": data,
        }
        environment = collect_system_environment(cpu_threads_per_model=threads)
        if self.residency is not None:
            environment["residency"] = {
                **self.residency.snapshot(),
                "evicted_not_selected": residency_evicted,
                "evicted_over_budget": residency_trimmed,
            }
        self.artifact_store.write_json(run_id, "environment.json", environment)
        self.artifact_store.write_json(run_id, "benchmark_report.json", report)
        yield data, True

    def _run_case_model(
        self, run_id: str, case: BenchmarkCase, model_id: str, num_threads: int,
        warmup_runs: int, measured_runs: int, config: dict[str, Any],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        measurements: list[dict[str, Any]] = []
        record = None
        loaded = isolated = False
        warm = False
        in_process_footprint = None
        metadata = None
        audio_url = None
        try:
            record = self.registry.get_record(model_id)
            info = self.registry.model_info(model_id)
            if record.status == ModelStatus.UNAVAILABLE:
                raise ArenaError(1002, f"model '{model_id}' is unavailable", error_type="model_unavailable")
            kwargs = BenchmarkService._normalize_infer_config(
                model_id=model_id, info=info, capabilities=info.get("capabilities") or {}, config=config, warnings=warnings
            )
            filename = self.artifact_store.safe_audio_filename(f"{case.id}__{model_id}")
            path = self.artifact_store.audio_output_path(run_id, filename)

            route = BenchmarkService._execution_route(record.spec, self.process_runner, self.residency)
            # Per-model ceiling + text-scaled budget (shared with single runs),
            # multiplied by the number of synthesis passes in this group.
            per_pass_timeout = BenchmarkService._resolve_timeout_sec(
                record.spec, case.text, self.inference_timeout_sec
            )
            group_timeout = per_pass_timeout * max(1, warmup_runs + measured_runs)

            if route == "warm_worker":
                # Keep-warm: reuse (or start) a dedicated persistent worker so a heavy
                # model stays loaded across cases/runs instead of a one-shot spawn.
                assert self.residency is not None
                warm = True
                self.registry.set_status(model_id, ModelStatus.BUSY)
                load_timeout = max(60.0, min(float(record.spec.timeout_base_sec or per_pass_timeout), 900.0))
                worker = None
                result = None
                try:
                    worker = self.residency.acquire_worker(record.spec, timeout_sec=load_timeout)
                    result = worker.submit_repeated(
                        text=case.text, infer_kwargs=kwargs, audio_path=str(path),
                        warmup_runs=warmup_runs, measured_runs=measured_runs, timeout_sec=group_timeout,
                    )
                except ProcessTimeoutError as exc:
                    self.residency.evict(model_id)
                    warm = False
                    path.unlink(missing_ok=True)
                    return self._error(
                        case, model_id, warmup_runs, measured_runs, warnings,
                        {"code": 3001, "type": "inference_timeout", "message": f"suite case/model group exceeded {group_timeout:.3f}s hard timeout"},
                        worker=exc.diagnostics(),
                    )
                except WarmWorkerError as exc:
                    warnings.append(f"{model_id}: warm worker unavailable ({exc}); falling back to isolated run")
                    self.residency.evict(model_id)
                    warm = False
                    route = "isolated"
                if result is not None:
                    self.residency.mark_worker_used(model_id, footprint_mb=worker.rss_mb() if worker is not None else None)
                    return self._repeated_payload_result(
                        run_id=run_id, case=case, model_id=model_id, filename=filename,
                        warmup_runs=warmup_runs, measured_runs=measured_runs, warnings=warnings,
                        path=path, result=result,
                    )

            if route == "isolated":
                isolated = True
                self.registry.set_status(model_id, ModelStatus.BUSY)
                task = {
                    "adapter": record.spec.adapter, "text": case.text,
                    # Resolved absolute path: the isolated worker may run with a
                    # different CWD where the raw relative model_path would not resolve.
                    "model_path": record.spec.resolved_model_path or record.spec.model_path,
                    "num_threads": num_threads, "infer_kwargs": kwargs, "audio_path": str(path),
                    "warmup_runs": warmup_runs, "measured_runs": measured_runs,
                }
                try:
                    if record.spec.worker_python:
                        warnings.append(f"{model_id}: using dedicated external Python worker")
                        result = self.process_runner.run_external_worker(record.spec.worker_python, "repeated", task, timeout_sec=group_timeout)
                    else:
                        result = self.process_runner.run(run_isolated_repeated_model, task, timeout_sec=group_timeout)
                except ProcessTimeoutError as exc:
                    path.unlink(missing_ok=True)
                    return self._error(
                        case, model_id, warmup_runs, measured_runs, warnings,
                        {"code": 3001, "type": "inference_timeout", "message": f"suite case/model group exceeded {group_timeout:.3f}s hard timeout"},
                        worker=exc.diagnostics(),
                    )
                return self._repeated_payload_result(
                    run_id=run_id, case=case, model_id=model_id, filename=filename,
                    warmup_runs=warmup_runs, measured_runs=measured_runs, warnings=warnings,
                    path=path, result=result,
                )

            if self.process_runner is not None and record.spec.keep_in_memory:
                warnings.append(f"{model_id}: keep_in_memory suite execution remains in-process; hard process timeout is not enforced")
            adapter = self.registry.load(model_id, num_threads=num_threads)
            loaded = True
            self.registry.set_status(model_id, ModelStatus.BUSY)
            for _ in range(warmup_runs):
                adapter.infer(case.text, **kwargs)
            for repeat in range(1, measured_runs + 1):
                try:
                    output, metrics = self.metrics.measure_inference(adapter, case.text, **kwargs)
                    metadata = dict(output.metadata)
                    current_audio = None
                    if audio_url is None:
                        write_wav(path, output.audio, output.sample_rate)
                        audio_url = f"/api/v1/audio/download/{run_id}/{filename}"
                        current_audio = audio_url
                    measurements.append({"repeat": repeat, "status": "success", "metrics": metrics.to_dict(), "audio_url": current_audio, "error": None})
                except Exception as exc:
                    measurements.append({"repeat": repeat, "status": "error", "metrics": None, "audio_url": None, "error": BenchmarkService._normalize_error(exc)})
            deltas = [(m.get("metrics") or {}).get("rss_delta_mb") for m in measurements]
            deltas = [d for d in deltas if isinstance(d, (int, float))]
            in_process_footprint = max(deltas) if deltas else None
            return self._complete(case, model_id, warmup_runs, measured_runs, measurements, warnings, metadata, audio_url)
        except Exception as exc:
            return self._error(case, model_id, warmup_runs, measured_runs, warnings, BenchmarkService._normalize_error(exc), measurements, metadata)
        finally:
            if record is None:
                pass
            elif isolated:
                self.registry.set_status(model_id, ModelStatus.UNLOADED)
            elif warm:
                live = self.residency is not None and self.residency.worker_for(model_id) is not None
                self.registry.set_status(model_id, ModelStatus.READY if live else ModelStatus.UNLOADED)
            elif loaded:
                try:
                    if self.residency is not None and self.residency.keep_warm and not record.spec.keep_in_memory:
                        admission = self.residency.admit_in_process(record.spec, footprint_mb=in_process_footprint)
                        if admission.allowed:
                            self.registry.set_status(model_id, ModelStatus.READY)
                            self.residency.mark_in_process(model_id, in_process_footprint or 0.0)
                        else:
                            warnings.append(f"{model_id}: not kept resident ({admission.reason})")
                            self.registry.unload(model_id)
                    elif record.spec.keep_in_memory:
                        self.registry.set_status(model_id, ModelStatus.READY)
                    else:
                        self.registry.unload(model_id)
                except Exception:
                    self.registry.set_status(model_id, ModelStatus.ERROR, error="cleanup failed")

    def _repeated_payload_result(
        self, *, run_id: str, case: BenchmarkCase, model_id: str, filename: str,
        warmup_runs: int, measured_runs: int, warnings: list[str], path: Any, result: Any,
    ) -> dict[str, Any]:
        """Map a repeated-mode worker ``ProcessResult`` (one-shot or warm) to a group result."""
        if result.status != "success":
            path.unlink(missing_ok=True)
            return self._error(
                case, model_id, warmup_runs, measured_runs, warnings,
                {"code": 3002, "type": "worker_exited", "message": result.error_message or "isolated suite worker exited unexpectedly", "exit_code": result.exit_code},
                worker=result.diagnostics(),
            )
        payload = dict(result.value or {})
        if payload.get("status") != "success":
            path.unlink(missing_ok=True)
            return self._error(
                case, model_id, warmup_runs, measured_runs, warnings,
                payload.get("error") or {"code": 3002, "type": "worker_error", "message": "isolated suite worker failed"},
                metadata=payload.get("metadata"), worker=result.diagnostics(),
            )
        metadata = payload.get("metadata")
        audio_url = f"/api/v1/audio/download/{run_id}/{filename}" if payload.get("audio_written") else None
        measurements: list[dict[str, Any]] = []
        for raw in payload.get("measurements") or []:
            item = dict(raw)
            wrote = bool(item.pop("wrote_representative_audio", False))
            item["audio_url"] = audio_url if wrote else None
            measurements.append(item)
        return self._complete(case, model_id, warmup_runs, measured_runs, measurements, warnings, metadata, audio_url, worker=result.diagnostics())

    @staticmethod
    def _complete(case, model_id, warmups, repeats, measurements, warnings, metadata, audio_url, worker=None) -> dict[str, Any]:
        count = sum(m.get("status") == "success" for m in measurements)
        status = "success" if count == repeats else ("partial" if count else "error")
        return {
            "case_id": case.id, "model_id": model_id, "status": status, "audio_url": audio_url,
            "warmup_runs": warmups, "measured_runs": repeats, "successful_runs": count,
            "measurements": measurements, "aggregate": aggregate_measurements(measurements),
            "warnings": warnings, "metadata": metadata, "worker": worker,
            "error": RepeatedBenchmarkService._first_measurement_error(measurements) if status != "success" else None,
        }

    @staticmethod
    def _error(case, model_id, warmups, repeats, warnings, error, measurements=None, metadata=None, worker=None) -> dict[str, Any]:
        measurements = measurements or []
        return {
            "case_id": case.id, "model_id": model_id, "status": "error", "audio_url": None,
            "warmup_runs": warmups, "measured_runs": repeats, "successful_runs": 0,
            "measurements": measurements, "aggregate": aggregate_measurements(measurements),
            "warnings": warnings, "metadata": metadata, "worker": worker, "error": error,
        }

    @staticmethod
    def _first_measurement_error(measurements: list[dict[str, Any]]) -> dict[str, Any] | None:
        return next((m["error"] for m in measurements if m.get("error")), None)
