from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Iterator
from uuid import uuid4

from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec, ModelStatus
from edgetts_arena.core.process_runner import ProcessRunner, ProcessTimeoutError
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.system_info import collect_system_environment
from edgetts_arena.core.worker_runtime import run_isolated_model
from edgetts_arena.utils import write_wav


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_run_id(now: datetime | None = None) -> str:
    stamp = (now or utc_now()).strftime("%Y%m%d_%H%M%S")
    return f"run_{stamp}_{uuid4().hex[:6]}"


class BenchmarkService:
    def __init__(
        self,
        registry: ModelRegistry,
        resource_guard: ResourceGuard,
        artifact_store: RunArtifactStore,
        *,
        metrics_collector: MetricsCollector | None = None,
        process_runner: ProcessRunner | None = None,
        inference_timeout_sec: float = 60.0,
        isolate_model_processes: bool = True,
    ) -> None:
        self.registry = registry
        self.resource_guard = resource_guard
        self.artifact_store = artifact_store
        self.metrics = metrics_collector or MetricsCollector()
        self.process_runner = (
            process_runner
            if process_runner is not None
            else (ProcessRunner() if isolate_model_processes else None)
        )
        self.inference_timeout_sec = float(inference_timeout_sec)
        self._artifact_lock = Lock()

    def run(
        self,
        *,
        text: str,
        model_ids: list[str],
        execution_mode: str = "sequential",
        cpu_threads_per_model: int = 4,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Blocking wrapper that returns the final snapshot of :meth:`run_stream`."""
        data: dict[str, Any] = {}
        for data, _complete in self.run_stream(
            text=text,
            model_ids=model_ids,
            execution_mode=execution_mode,
            cpu_threads_per_model=cpu_threads_per_model,
            config=config,
        ):
            pass
        return data

    def run_stream(
        self,
        *,
        text: str,
        model_ids: list[str],
        execution_mode: str = "sequential",
        cpu_threads_per_model: int = 4,
        config: dict[str, Any] | None = None,
    ) -> Iterator[tuple[dict[str, Any], bool]]:
        """Yield ``(data_snapshot, complete)`` after each model finishes.

        Lets the UI render results incrementally instead of waiting for the whole
        run. ``results`` stays index-aligned to ``model_ids`` (sequential finishes
        in order; concurrent fills slots as futures complete). The final item
        (``complete=True``) is written to benchmark_report.json / environment.json
        and is exactly what :meth:`run` returns.
        """
        if execution_mode not in {"sequential", "concurrent"}:
            raise ArenaError(1001, "invalid execution_mode", error_type="validation_error")

        plan = self.resource_guard.plan_execution(
            execution_mode=execution_mode,
            model_count=len(model_ids),
            requested_threads_per_model=cpu_threads_per_model,
        )
        threads = plan.threads_per_model
        config = dict(config or {})
        run_id = new_run_id()
        started_at = utc_now()
        environment = collect_system_environment(cpu_threads_per_model=threads)
        environment["execution_plan"] = plan.to_dict()
        self.artifact_store.create_run(run_id)

        def snapshot(results: list[dict[str, Any]], complete: bool) -> dict[str, Any]:
            return {
                "run_id": run_id,
                "started_at": iso_utc(started_at),
                "completed_at": iso_utc(utc_now()) if complete else None,
                "execution_mode": execution_mode,
                "execution_profile": plan.profile,
                "requested_cpu_threads_per_model": cpu_threads_per_model,
                "cpu_threads_per_model": threads,
                "total_threads_budget": plan.total_threads_budget,
                "resource_warnings": list(plan.warnings),
                "results": results,
            }

        aligned: list[dict[str, Any] | None] = [None] * len(model_ids)
        if execution_mode == "sequential":
            for index, model_id in enumerate(model_ids):
                model_started_at = utc_now()
                result = self._run_model(
                    run_id=run_id,
                    model_id=model_id,
                    text=text,
                    num_threads=threads,
                    config=config,
                )
                model_completed_at = utc_now()
                # 计算模型运行时长（秒）
                result["model_duration_sec"] = (model_completed_at - model_started_at).total_seconds()
                result["model_started_at"] = iso_utc(model_started_at)
                result["model_completed_at"] = iso_utc(model_completed_at)
                aligned[index] = result
                yield snapshot([item for item in aligned if item is not None], False), False
        else:
            with ThreadPoolExecutor(max_workers=len(model_ids), thread_name_prefix="arena-model") as pool:
                futures = {
                    pool.submit(
                        self._run_model,
                        run_id=run_id,
                        model_id=model_id,
                        text=text,
                        num_threads=threads,
                        config=config,
                    ): index
                    for index, model_id in enumerate(model_ids)
                }
                for future in as_completed(futures):
                    aligned[futures[future]] = future.result()
                    yield snapshot([item for item in aligned if item is not None], False), False

        data = snapshot([item for item in aligned if item is not None], True)
        self.artifact_store.write_json(run_id, "environment.json", environment)
        self.artifact_store.write_json(
            run_id,
            "benchmark_report.json",
            {
                "request": {
                    "text": text,
                    "models": model_ids,
                    "execution_mode": execution_mode,
                    "execution_profile": plan.profile,
                    "requested_cpu_threads_per_model": cpu_threads_per_model,
                    "cpu_threads_per_model": threads,
                    "total_threads_budget": plan.total_threads_budget,
                    "resource_warnings": list(plan.warnings),
                    "config": config,
                },
                "data": data,
            },
        )
        yield data, True

    def _run_model(
        self,
        *,
        run_id: str,
        model_id: str,
        text: str,
        num_threads: int,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        warnings: list[str] = []
        record = None
        loaded_for_run = False
        isolated_run = False
        try:
            record = self.registry.get_record(model_id)
            info = self.registry.model_info(model_id)
            if record.status == ModelStatus.UNAVAILABLE:
                raise ArenaError(1002, f"model '{model_id}' is unavailable", error_type="model_unavailable")

            capabilities = info.get("capabilities") or {}
            timeout_sec = self._resolve_timeout_sec(record.spec, text, self.inference_timeout_sec)
            infer_kwargs = self._normalize_infer_config(
                model_id=model_id,
                info=info,
                capabilities=capabilities,
                config=config,
                warnings=warnings,
            )
            filename = self.artifact_store.safe_model_filename(model_id)
            path = self.artifact_store.audio_output_path(run_id, filename)

            if self.process_runner is not None and not record.spec.keep_in_memory:
                isolated_run = True
                self.registry.set_status(model_id, ModelStatus.BUSY)
                task = {
                    "adapter": record.spec.adapter,
                    # Use the resolved absolute path: the isolated worker may run
                    # with a different CWD, where the raw relative model_path would
                    # not resolve (a source of spurious 1001/3002 worker errors).
                    "model_path": record.spec.resolved_model_path or record.spec.model_path,
                    "text": text,
                    "num_threads": num_threads,
                    "infer_kwargs": infer_kwargs,
                    "audio_path": str(path),
                }
                try:
                    if record.spec.worker_python:
                        warnings.append(f"{model_id}: using dedicated external Python worker")
                        process_result = self.process_runner.run_external_worker(
                            record.spec.worker_python,
                            "single",
                            task,
                            timeout_sec=timeout_sec,
                        )
                    else:
                        process_result = self.process_runner.run(
                            run_isolated_model, task, timeout_sec=timeout_sec
                        )
                except ProcessTimeoutError as exc:
                    path.unlink(missing_ok=True)
                    return {
                        "model_id": model_id,
                        "status": "error",
                        "audio_url": None,
                        "metrics": None,
                        "warnings": warnings,
                        "error": {
                            "code": 3001,
                            "type": "inference_timeout",
                            "message": f"model inference exceeded {timeout_sec:.3f}s hard timeout",
                        },
                        "metadata": None,
                        "worker": exc.diagnostics(),
                    }
                if process_result.status != "success":
                    path.unlink(missing_ok=True)
                    return {
                        "model_id": model_id,
                        "status": "error",
                        "audio_url": None,
                        "metrics": None,
                        "warnings": warnings,
                        "error": {
                            "code": 3002,
                            "type": "worker_exited",
                            "message": process_result.error_message or "isolated worker exited unexpectedly",
                            "exit_code": process_result.exit_code,
                        },
                        "metadata": None,
                        "worker": process_result.diagnostics(),
                    }
                payload = dict(process_result.value or {})
                if payload.get("status") != "success":
                    path.unlink(missing_ok=True)
                    return {
                        "model_id": model_id,
                        "status": "error",
                        "audio_url": None,
                        "metrics": None,
                        "warnings": warnings,
                        "error": payload.get("error") or {
                            "code": 3002, "type": "worker_error", "message": "isolated worker failed"
                        },
                        "metadata": payload.get("metadata"),
                        "worker": process_result.diagnostics(),
                    }
                return {
                    "model_id": model_id,
                    "status": "success",
                    "audio_url": f"/api/v1/audio/download/{run_id}/{filename}",
                    "metrics": payload.get("metrics"),
                    "config": config,
                    "warnings": warnings,
                    "error": None,
                    "metadata": payload.get("metadata"),
                    "worker": process_result.diagnostics(),
                }

            if self.process_runner is not None and record.spec.keep_in_memory:
                warnings.append(
                    f"{model_id}: keep_in_memory uses in-process execution; hard process timeout is not enforced"
                )
            adapter = self.registry.load(model_id, num_threads=num_threads)
            loaded_for_run = True
            self.registry.set_status(model_id, ModelStatus.BUSY)
            output, metrics = self.metrics.measure_inference(adapter, text, **infer_kwargs)
            with self._artifact_lock:
                write_wav(path, output.audio, output.sample_rate)

            return {
                "model_id": model_id,
                "status": "success",
                "audio_url": f"/api/v1/audio/download/{run_id}/{filename}",
                "metrics": metrics.to_dict(),
                "config": config,
                "warnings": warnings,
                "error": None,
                "metadata": output.metadata,
                "worker": None,
            }
        except Exception as exc:
            error = self._normalize_error(exc)
            return {
                "model_id": model_id,
                "status": "error",
                "audio_url": None,
                "metrics": None,
                "warnings": warnings,
                "error": error,
                "metadata": None,
                "worker": None,
            }
        finally:
            if record is not None and isolated_run:
                self.registry.set_status(model_id, ModelStatus.UNLOADED)
            elif record is not None and loaded_for_run:
                try:
                    if record.spec.keep_in_memory:
                        self.registry.set_status(model_id, ModelStatus.READY)
                    else:
                        self.registry.unload(model_id)
                except Exception:
                    self.registry.set_status(model_id, ModelStatus.ERROR, error="cleanup failed")

    @staticmethod
    def _resolve_timeout_sec(spec: ModelSpec, text: str, global_default: float) -> float:
        """Hard inference timeout for synthesizing ``text`` once on ``spec``.

        Static models (no ``timeout_per_char_sec``) use ``inference_timeout_sec`` or
        the global default. Models that declare a per-char rate scale the budget with
        text length so short text fails fast on a real hang while long text is not
        spuriously cut off::

            timeout = clamp(base + per_char * len(text), base, ceiling)

        ``base`` (model load + fixed overhead) defaults to the global timeout when
        omitted; ``ceiling`` is ``inference_timeout_sec`` (or the global default when
        unset), so pathological long text can never request an unbounded budget.
        """
        if spec.timeout_per_char_sec is None:
            return (
                global_default
                if spec.inference_timeout_sec is None
                else float(spec.inference_timeout_sec)
            )
        base = global_default if spec.timeout_base_sec is None else float(spec.timeout_base_sec)
        ceiling = (
            global_default
            if spec.inference_timeout_sec is None
            else float(spec.inference_timeout_sec)
        )
        scaled = base + float(spec.timeout_per_char_sec) * len(text)
        # Never below the fixed load allowance, never above the hard ceiling.
        return max(base, min(scaled, ceiling))

    @staticmethod
    def _normalize_infer_config(
        *,
        model_id: str,
        info: dict[str, object],
        capabilities: dict[str, object],
        config: dict[str, Any],
        warnings: list[str],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        speed = float(config.get("speed", 1.0))
        if speed != 1.0 and not bool(capabilities.get("speed", False)):
            raise ArenaError(
                1003,
                f"model '{model_id}' does not support speed control",
                error_type="capability_conflict",
            )
        if bool(capabilities.get("speed", False)):
            kwargs["speed"] = speed

        voice = config.get("voice")
        if voice is not None:
            if not bool(capabilities.get("voices", False)):
                raise ArenaError(
                    1003,
                    f"model '{model_id}' does not support voice selection",
                    error_type="capability_conflict",
                )
            voices = list(info.get("voices") or [])
            if voices and voice not in voices:
                raise ArenaError(
                    1003,
                    f"voice '{voice}' is not available for model '{model_id}'",
                    error_type="capability_conflict",
                )
            kwargs["voice"] = voice

        language = config.get("language")
        if language is not None:
            normalized_language = str(language).strip().lower()
            if not normalized_language:
                raise ArenaError(1001, "language must not be blank", error_type="validation_error")
            if not bool(capabilities.get("language_control", False)):
                raise ArenaError(
                    1003,
                    f"model '{model_id}' does not support explicit language control",
                    error_type="capability_conflict",
                )
            languages = [str(item).strip().lower() for item in capabilities.get("languages") or []]
            if languages and normalized_language not in languages:
                raise ArenaError(
                    1003,
                    f"language '{normalized_language}' is not available for model '{model_id}'",
                    error_type="capability_conflict",
                )
            kwargs["language"] = normalized_language

        seed = config.get("seed")
        if seed is not None:
            if bool(capabilities.get("seed", False)):
                kwargs["seed"] = int(seed)
            else:
                warnings.append(f"{model_id}: seed ignored because the model does not support deterministic seed")

        sample_rate = config.get("sample_rate")
        if sample_rate is not None:
            warnings.append(
                f"{model_id}: sample_rate override is not supported; native model sample rate is retained"
            )
        return kwargs

    @staticmethod
    def _normalize_error(exc: Exception) -> dict[str, Any]:
        if isinstance(exc, ArenaError):
            return {"code": exc.code, "type": exc.error_type, "message": exc.message}
        if isinstance(exc, (ValueError, FileNotFoundError)):
            return {"code": 1001, "type": "invalid_model_input", "message": str(exc)}
        return {"code": 3002, "type": "model_internal_error", "message": str(exc)}
