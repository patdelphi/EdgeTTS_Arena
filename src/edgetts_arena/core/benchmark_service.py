from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.core.model_registry import ModelRegistry, ModelStatus
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.system_info import collect_system_environment
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
    ) -> None:
        self.registry = registry
        self.resource_guard = resource_guard
        self.artifact_store = artifact_store
        self.metrics = metrics_collector or MetricsCollector()
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

        if execution_mode == "sequential":
            results = [
                self._run_model(
                    run_id=run_id,
                    model_id=model_id,
                    text=text,
                    num_threads=threads,
                    config=config,
                )
                for model_id in model_ids
            ]
        else:
            indexed: dict[int, dict[str, Any]] = {}
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
                    indexed[futures[future]] = future.result()
            results = [indexed[index] for index in range(len(model_ids))]

        completed_at = utc_now()
        data = {
            "run_id": run_id,
            "started_at": iso_utc(started_at),
            "completed_at": iso_utc(completed_at),
            "execution_mode": execution_mode,
            "execution_profile": plan.profile,
            "requested_cpu_threads_per_model": cpu_threads_per_model,
            "cpu_threads_per_model": threads,
            "total_threads_budget": plan.total_threads_budget,
            "resource_warnings": list(plan.warnings),
            "results": results,
        }
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
        return data

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
        try:
            record = self.registry.get_record(model_id)
            info = self.registry.model_info(model_id)
            if record.status == ModelStatus.UNAVAILABLE:
                raise ArenaError(1002, f"model '{model_id}' is unavailable", error_type="model_unavailable")

            capabilities = info.get("capabilities") or {}
            infer_kwargs = self._normalize_infer_config(
                model_id=model_id,
                info=info,
                capabilities=capabilities,
                config=config,
                warnings=warnings,
            )
            adapter = self.registry.load(model_id, num_threads=num_threads)
            loaded_for_run = True
            self.registry.set_status(model_id, ModelStatus.BUSY)
            output, metrics = self.metrics.measure_inference(adapter, text, **infer_kwargs)

            filename = self.artifact_store.safe_model_filename(model_id)
            with self._artifact_lock:
                path = self.artifact_store.audio_output_path(run_id, filename)
                write_wav(path, output.audio, output.sample_rate)

            return {
                "model_id": model_id,
                "status": "success",
                "audio_url": f"/api/v1/audio/download/{run_id}/{filename}",
                "metrics": metrics.to_dict(),
                "warnings": warnings,
                "error": None,
                "metadata": output.metadata,
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
            }
        finally:
            if record is not None and loaded_for_run:
                try:
                    if record.spec.keep_in_memory:
                        self.registry.set_status(model_id, ModelStatus.READY)
                    else:
                        self.registry.unload(model_id)
                except Exception:
                    self.registry.set_status(model_id, ModelStatus.ERROR, error="cleanup failed")

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
