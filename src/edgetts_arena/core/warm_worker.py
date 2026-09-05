"""Persistent (warm) external worker child process.

Unlike :mod:`edgetts_arena.core.external_worker` — which loads a model, runs one
task and exits — this worker stays alive across many requests so a heavy model
(Qwen3 / CosyVoice / MeloTTS, whose cold load can take tens of seconds) is only
loaded once and reused until the residency manager decides to evict it.

Line-delimited JSON protocol over stdin/stdout. Every command is a single JSON
object with an ``op`` field; every response is one framed line prefixed with
``__EDGETTS_ARENA_RESULT__=`` so the parent can ignore any library noise that a
model runtime prints to stdout.

Commands::

    {"op": "load", "adapter", "model_path", "num_threads"}   -> load once
    {"op": "infer", "text", "infer_kwargs", "audio_path"}    -> reuse loaded model
    {"op": "infer_repeated", ..., "warmup_runs", "measured_runs"}
    {"op": "ping"}                                           -> liveness + rss
    {"op": "shutdown"}                                       -> exit 0

On stdin EOF (parent gone) the worker exits so it can never be orphaned.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from typing import Any

RESULT_PREFIX = "__EDGETTS_ARENA_RESULT__="
WARM_PROTOCOL_VERSION = 1

# The single adapter instance kept warm for the lifetime of this process.
_ADAPTER: Any = None


def _rss_mb() -> float:
    """This process' resident set size in MB (best effort)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:  # pragma: no cover - psutil always present in worker venvs
        return 0.0


def _emit(payload: dict[str, Any]) -> None:
    framed = dict(payload)
    framed["_worker_runtime"] = {
        "protocol": "warm-json",
        "protocol_version": WARM_PROTOCOL_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }
    framed.setdefault("rss_mb", _rss_mb())
    sys.stdout.write(
        RESULT_PREFIX + json.dumps(framed, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    sys.stdout.flush()


def _normalize_error(exc: Exception) -> dict[str, Any]:
    from edgetts_arena.core.worker_runtime import _normalize_worker_error

    return _normalize_worker_error(exc)


def _load(cmd: dict[str, Any]) -> dict[str, Any]:
    global _ADAPTER
    from edgetts_arena.core.worker_runtime import _adapter_factory

    factory = _adapter_factory(str(cmd["adapter"]))
    adapter = factory()
    adapter.load_model(
        str(cmd.get("model_path") or ""), device="cpu", num_threads=int(cmd["num_threads"])
    )
    _ADAPTER = adapter
    return {"op": "load", "status": "ok"}


def _infer(cmd: dict[str, Any]) -> dict[str, Any]:
    from edgetts_arena.core.metrics_collector import MetricsCollector
    from edgetts_arena.utils import write_wav

    if _ADAPTER is None:
        return {
            "op": "infer",
            "status": "error",
            "metrics": None,
            "metadata": None,
            "error": {
                "code": 3002,
                "type": "worker_not_loaded",
                "message": "warm worker received 'infer' before a successful 'load'",
            },
        }
    output, metrics = MetricsCollector().measure_inference(
        _ADAPTER, str(cmd["text"]), **dict(cmd.get("infer_kwargs") or {})
    )
    write_wav(str(cmd["audio_path"]), output.audio, output.sample_rate)
    return {
        "op": "infer",
        "status": "success",
        "metrics": metrics.to_dict(),
        "metadata": output.metadata,
        "error": None,
    }


def _infer_repeated(cmd: dict[str, Any]) -> dict[str, Any]:
    from edgetts_arena.core.metrics_collector import MetricsCollector
    from edgetts_arena.utils import write_wav

    if _ADAPTER is None:
        return {
            "op": "infer_repeated",
            "status": "error",
            "measurements": [],
            "metadata": None,
            "audio_written": False,
            "error": {
                "code": 3002,
                "type": "worker_not_loaded",
                "message": "warm worker received 'infer_repeated' before a successful 'load'",
            },
        }
    text = str(cmd["text"])
    infer_kwargs = dict(cmd.get("infer_kwargs") or {})
    for _ in range(int(cmd.get("warmup_runs") or 0)):
        _ADAPTER.infer(text, **infer_kwargs)
    collector = MetricsCollector()
    measurements: list[dict[str, Any]] = []
    metadata: dict[str, Any] | None = None
    audio_written = False
    for repeat_index in range(1, int(cmd["measured_runs"]) + 1):
        try:
            output, metrics = collector.measure_inference(_ADAPTER, text, **infer_kwargs)
            metadata = dict(output.metadata)
            wrote_audio = False
            if not audio_written:
                write_wav(str(cmd["audio_path"]), output.audio, output.sample_rate)
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
                    "error": _normalize_error(exc),
                }
            )
    return {
        "op": "infer_repeated",
        "status": "success",
        "measurements": measurements,
        "metadata": metadata,
        "audio_written": audio_written,
        "error": None,
    }


def _dispatch(cmd: dict[str, Any]) -> int | None:
    """Handle one command; return an exit code when the loop should stop."""
    op = cmd.get("op")
    try:
        if op == "load":
            _emit(_load(cmd))
        elif op == "infer":
            _emit(_infer(cmd))
        elif op == "infer_repeated":
            _emit(_infer_repeated(cmd))
        elif op == "ping":
            _emit({"op": "ping", "status": "ok"})
        elif op == "shutdown":
            _emit({"op": "shutdown", "status": "ok"})
            return 0
        else:
            _emit(
                {
                    "op": "error",
                    "status": "error",
                    "error": {
                        "code": 1001,
                        "type": "worker_protocol",
                        "message": f"unknown warm-worker op: {op!r}",
                    },
                }
            )
    except Exception as exc:  # keep the worker alive across a failed request
        _emit({"op": op or "error", "status": "error", "error": _normalize_error(exc)})
    return None


def main(argv: list[str] | None = None) -> int:
    # readline() (not `for line in stdin`) avoids read-ahead buffering that would
    # stall an interactive request/response loop.
    while True:
        line = sys.stdin.readline()
        if not line:  # EOF: parent closed the pipe or exited
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
            if not isinstance(cmd, dict):
                raise ValueError("command must be a JSON object")
        except Exception as exc:
            _emit(
                {
                    "op": "error",
                    "status": "error",
                    "error": {
                        "code": 1001,
                        "type": "worker_protocol",
                        "message": f"invalid warm-worker command: {exc}",
                    },
                }
            )
            continue
        exit_code = _dispatch(cmd)
        if exit_code is not None:
            return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
