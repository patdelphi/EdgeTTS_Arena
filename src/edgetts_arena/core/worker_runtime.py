from __future__ import annotations

from typing import Any

from edgetts_arena.core.errors import ArenaError
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.utils import write_wav


def _adapter_factory(adapter_name: str):
    if adapter_name == "dummy":
        from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
        return DummyTTSAdapter
    if adapter_name == "piper":
        from edgetts_arena.adapters.piper_adapter import PiperTTSAdapter
        return PiperTTSAdapter
    if adapter_name == "kokoro":
        from edgetts_arena.adapters.kokoro_adapter import KokoroTTSAdapter
        return KokoroTTSAdapter
    if adapter_name == "qwen3":
        from edgetts_arena.adapters.qwen3_adapter import Qwen3TTSAdapter
        return Qwen3TTSAdapter
    if adapter_name == "cosyvoice":
        from edgetts_arena.adapters.cosyvoice_adapter import CosyVoiceTTSAdapter
        return CosyVoiceTTSAdapter
    if adapter_name == "melotts":
        from edgetts_arena.adapters.melotts_adapter import MeloTTSAdapter
        return MeloTTSAdapter
    raise ArenaError(1002, f"adapter '{adapter_name}' is unavailable", error_type="adapter_unavailable")


def run_isolated_model(task: dict[str, Any]) -> dict[str, Any]:
    """Child-process target. Writes audio to a prevalidated path and returns only small metadata."""
    adapter = None
    try:
        factory = _adapter_factory(str(task["adapter"]))
        adapter = factory()
        adapter.load_model(
            str(task.get("model_path") or ""),
            device="cpu",
            num_threads=int(task["num_threads"]),
        )
        output, metrics = MetricsCollector().measure_inference(
            adapter,
            str(task["text"]),
            **dict(task.get("infer_kwargs") or {}),
        )
        write_wav(str(task["audio_path"]), output.audio, output.sample_rate)
        return {
            "status": "success",
            "metrics": metrics.to_dict(),
            "metadata": output.metadata,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "metrics": None,
            "metadata": None,
            "error": _normalize_worker_error(exc),
        }
    finally:
        if adapter is not None:
            try:
                adapter.unload_model()
            except Exception:
                pass



def run_isolated_repeated_model(task: dict[str, Any]) -> dict[str, Any]:
    """Run one case/model warm-up + repeated measurement group in one child process."""
    adapter = None
    try:
        factory = _adapter_factory(str(task["adapter"]))
        adapter = factory()
        adapter.load_model(
            str(task.get("model_path") or ""),
            device="cpu",
            num_threads=int(task["num_threads"]),
        )
        text = str(task["text"])
        infer_kwargs = dict(task.get("infer_kwargs") or {})
        for _ in range(int(task.get("warmup_runs") or 0)):
            adapter.infer(text, **infer_kwargs)

        collector = MetricsCollector()
        measurements: list[dict[str, Any]] = []
        metadata: dict[str, Any] | None = None
        audio_written = False
        for repeat_index in range(1, int(task["measured_runs"]) + 1):
            try:
                output, metrics = collector.measure_inference(adapter, text, **infer_kwargs)
                metadata = dict(output.metadata)
                wrote_audio = False
                if not audio_written:
                    write_wav(str(task["audio_path"]), output.audio, output.sample_rate)
                    audio_written = True
                    wrote_audio = True
                measurements.append(
                    {
                        "repeat": repeat_index,
                        "status": "success",
                        "metrics": metrics.to_dict(),
                        "wrote_representative_audio": wrote_audio,
                        "error": None,
                    }
                )
            except Exception as exc:
                measurements.append(
                    {
                        "repeat": repeat_index,
                        "status": "error",
                        "metrics": None,
                        "wrote_representative_audio": False,
                        "error": _normalize_worker_error(exc),
                    }
                )
        return {
            "status": "success",
            "measurements": measurements,
            "metadata": metadata,
            "audio_written": audio_written,
            "error": None,
        }
    except Exception as exc:
        return {
            "status": "error",
            "measurements": [],
            "metadata": None,
            "audio_written": False,
            "error": _normalize_worker_error(exc),
        }
    finally:
        if adapter is not None:
            try:
                adapter.unload_model()
            except Exception:
                pass

def _normalize_worker_error(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, ArenaError):
        return {"code": exc.code, "type": exc.error_type, "message": exc.message}
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return {"code": 1001, "type": "invalid_model_input", "message": str(exc)}
    if isinstance(exc, MemoryError):
        return {"code": 2002, "type": "worker_memory_error", "message": str(exc) or "worker memory error"}
    return {"code": 3002, "type": "model_internal_error", "message": str(exc)}
