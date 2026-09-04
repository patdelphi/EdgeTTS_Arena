from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from edgetts_arena.api.schemas import APIEnvelope, BenchmarkRunRequest, BenchmarkSuiteRunRequest, StreamingStart
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.benchmark_suite import BenchmarkPresetSuite, RepeatedBenchmarkService
from edgetts_arena.core.config import AppSettings, load_settings
from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.model_registry import ModelRegistry, ModelStatus
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.system_info import collect_system_environment


logger = logging.getLogger(__name__)


_ERROR_HTTP_STATUS = {
    1001: 400,
    1002: 404,
    1003: 409,
    2001: 503,
    2002: 503,
    3001: 504,
    3002: 500,
}


def _error_payload(
    *,
    code: int,
    message: str,
    error_type: str,
    details: Any | None = None,
) -> dict[str, Any]:
    return {
        "code": code,
        "message": message,
        "data": None,
        "error": {"type": error_type, "details": details},
    }


def _success(data: Any) -> dict[str, Any]:
    return {"code": 200, "message": "success", "data": data, "error": None}


def create_app(
    *,
    settings: AppSettings | None = None,
    registry: ModelRegistry | None = None,
    exports_root: str | Path = "exports",
) -> FastAPI:
    settings = settings or load_settings()
    registry = registry or ModelRegistry.from_yaml(search_paths=settings.model_search_paths)
    artifact_store = RunArtifactStore(exports_root)
    resource_guard = ResourceGuard(settings.resource_guard)
    benchmark_service = BenchmarkService(
        registry, resource_guard, artifact_store, inference_timeout_sec=settings.inference_timeout_sec
    )
    preset_suite = BenchmarkPresetSuite.load()
    repeated_benchmark_service = RepeatedBenchmarkService(
        registry, resource_guard, artifact_store, preset_suite=preset_suite,
        inference_timeout_sec=settings.inference_timeout_sec,
    )

    app = FastAPI(
        title="EdgeTTS-Arena API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = settings
    app.state.registry = registry
    app.state.artifact_store = artifact_store
    app.state.benchmark_service = benchmark_service
    app.state.benchmark_preset_suite = preset_suite
    app.state.repeated_benchmark_service = repeated_benchmark_service

    # Surface missing dedicated worker environments (Qwen3 / CosyVoice / MeloTTS)
    # at startup so API-only deployments also get the actionable hint. Purely a
    # notice: it never blocks app creation.
    try:
        from edgetts_arena.core.model_registry import collect_worker_env_warnings

        for message in collect_worker_env_warnings(registry):
            logger.warning("worker env not configured: %s", message)
    except Exception:  # pragma: no cover - defensive startup notice
        pass

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=_error_payload(
                code=1001,
                message="invalid request",
                error_type="validation_error",
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(ArenaError)
    async def arena_exception_handler(request: Request, exc: ArenaError) -> JSONResponse:
        return JSONResponse(
            status_code=_ERROR_HTTP_STATUS.get(exc.code, 500),
            content=_error_payload(
                code=exc.code,
                message=exc.message,
                error_type=exc.error_type,
            ),
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/v1/system/models", response_model=APIEnvelope)
    async def system_models() -> dict[str, Any]:
        return _success(
            {
                "system_env": collect_system_environment(),
                "models": registry.list_models(),
            }
        )

    @app.post("/api/v1/benchmark/run", response_model=APIEnvelope)
    async def benchmark_run(payload: BenchmarkRunRequest) -> dict[str, Any]:
        data = await asyncio.to_thread(
            benchmark_service.run,
            text=payload.text,
            model_ids=payload.models,
            execution_mode=payload.execution_mode,
            cpu_threads_per_model=payload.cpu_threads_per_model,
            config=payload.config.model_dump(),
        )
        return _success(data)

    @app.get("/api/v1/benchmark/presets", response_model=APIEnvelope)
    async def benchmark_presets() -> dict[str, Any]:
        return _success(
            {
                "version": preset_suite.version,
                "defaults": {
                    "warmup_runs": preset_suite.warmup_runs,
                    "measured_runs": preset_suite.measured_runs,
                },
                "cases": [case.to_dict() for case in preset_suite.cases],
            }
        )

    @app.post("/api/v1/benchmark/suite", response_model=APIEnvelope)
    async def benchmark_suite_run(payload: BenchmarkSuiteRunRequest) -> dict[str, Any]:
        data = await asyncio.to_thread(
            repeated_benchmark_service.run_suite,
            model_ids=payload.models,
            case_ids=payload.case_ids,
            cpu_threads_per_model=payload.cpu_threads_per_model,
            warmup_runs=payload.warmup_runs,
            measured_runs=payload.measured_runs,
            config=payload.config.model_dump(),
        )
        return _success(data)

    @app.get("/api/v1/audio/download/{run_id}/{filename}")
    async def audio_download(run_id: str, filename: str) -> FileResponse:
        path = artifact_store.get_audio_file(run_id, filename)
        return FileResponse(
            path,
            media_type="audio/wav",
            filename=filename,
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.get("/api/v1/export/{run_id}")
    async def export_run(run_id: str) -> FileResponse:
        path = artifact_store.build_export(run_id)
        return FileResponse(
            path,
            media_type="application/zip",
            filename=path.name,
            headers={"X-Content-Type-Options": "nosniff"},
        )

    @app.websocket("/api/v1/tts/stream")
    async def stream_tts(websocket: WebSocket) -> None:
        await websocket.accept()
        model_id = (websocket.query_params.get("model") or "").strip()
        if not model_id:
            await _ws_error(websocket, 1001, "model query parameter is required", "validation_error")
            return

        try:
            record = registry.get_record(model_id)
            info = registry.model_info(model_id)
            capabilities = info.get("capabilities") or {}
            if record.status.value == "unavailable":
                raise ArenaError(
                    1002,
                    f"model '{model_id}' is unavailable",
                    error_type="model_unavailable",
                )
            if not bool(capabilities.get("streaming", False)):
                raise ArenaError(
                    1003,
                    f"model '{model_id}' does not support streaming",
                    error_type="capability_conflict",
                )

            try:
                event = StreamingStart.model_validate(await websocket.receive_json())
            except ValidationError as exc:
                await websocket.send_json(
                    {
                        "event": "error",
                        "code": 1001,
                        "type": "validation_error",
                        "message": "invalid start event",
                        "details": jsonable_encoder(exc.errors()),
                    }
                )
                await websocket.close(code=1008)
                return

            threads = resource_guard.clamp_threads(settings.default_num_threads)
            adapter = await asyncio.to_thread(registry.load, model_id, num_threads=threads)
            registry.set_status(model_id, ModelStatus.BUSY)
            await websocket.send_json({"event": "started", "model_id": model_id})

            iterator = adapter.infer_stream(
                event.text,
                voice=event.voice,
                speed=event.speed,
            )
            started = time.perf_counter()
            first_chunk_at: float | None = None
            sample_rate: int | None = None
            total_samples = 0
            chunk_idx = 0

            while True:
                chunk = await asyncio.to_thread(_next_or_none, iterator)
                if chunk is None:
                    break
                if sample_rate is None:
                    sample_rate = chunk.sample_rate
                elif chunk.sample_rate != sample_rate:
                    raise RuntimeError("stream returned inconsistent sample rates")

                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    await websocket.send_json(
                        {
                            "event": "first_chunk",
                            "ttfb_ms": (first_chunk_at - started) * 1000.0,
                            "chunk_idx": chunk_idx,
                            "sample_rate": chunk.sample_rate,
                            "encoding": "pcm_s16le",
                        }
                    )
                total_samples += chunk.audio.size
                await websocket.send_bytes(_pcm16_bytes(chunk.audio))
                chunk_idx += 1

            if sample_rate is None or total_samples == 0:
                raise RuntimeError("stream returned no audio")
            elapsed = max(time.perf_counter() - started, 1e-9)
            duration_sec = total_samples / sample_rate
            await websocket.send_json(
                {
                    "event": "complete",
                    "audio_duration_ms": duration_sec * 1000.0,
                    "rtf": elapsed / max(duration_sec, 1e-9),
                    "chunks": chunk_idx,
                }
            )
            if record.spec.keep_in_memory:
                registry.set_status(model_id, ModelStatus.READY)
            else:
                await asyncio.to_thread(registry.unload, model_id)
        except WebSocketDisconnect:
            logger.info("stream client disconnected model=%s", model_id)
        except ArenaError as exc:
            await _ws_error(websocket, exc.code, exc.message, exc.error_type)
        except Exception as exc:
            logger.exception("stream failure model=%s", model_id)
            await _ws_error(websocket, 3002, str(exc), "model_internal_error")
        finally:
            try:
                current = registry.get_record(model_id)
                if current.adapter is not None and current.status.value == "busy":
                    if current.spec.keep_in_memory:
                        registry.set_status(model_id, ModelStatus.READY)
                    else:
                        await asyncio.to_thread(registry.unload, model_id)
            except Exception:
                pass

    return app


def _next_or_none(iterator: Iterator[Any]) -> Any | None:
    try:
        return next(iterator)
    except StopIteration:
        return None


def _pcm16_bytes(audio: np.ndarray) -> bytes:
    samples = np.asarray(audio, dtype=np.float32)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


async def _ws_error(
    websocket: WebSocket,
    code: int,
    message: str,
    error_type: str,
) -> None:
    try:
        await websocket.send_json(
            {"event": "error", "code": code, "type": error_type, "message": message}
        )
        await websocket.close(code=1008)
    except RuntimeError:
        pass
