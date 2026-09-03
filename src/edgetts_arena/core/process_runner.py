from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import queue
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

_EXTERNAL_RESULT_PREFIX = "__EDGETTS_ARENA_RESULT__="
_EXTERNAL_PROTOCOL_VERSION = 1
_DIAGNOSTIC_TEXT_LIMIT = 8192


def _exit_diagnostics(exit_code: int | None) -> tuple[str, str | None, bool]:
    if exit_code is None:
        return "unknown", None, False
    if exit_code == 0:
        return "normal", None, False
    if exit_code < 0:
        signal_number = -exit_code
        try:
            signal_name = signal.Signals(signal_number).name
        except (ValueError, AttributeError):
            signal_name = f"SIGNAL_{signal_number}"
        oom_suspected = signal_number == getattr(signal, "SIGKILL", 9)
        return "signal", signal_name, oom_suspected
    return "exit_code", None, False


def _tail_text(value: str | None, *, limit: int = _DIAGNOSTIC_TEXT_LIMIT) -> str | None:
    if not value:
        return None
    text = value.strip()
    if len(text) <= limit:
        return text
    marker = "...<diagnostic output truncated>...\n"
    return marker + text[-max(0, limit - len(marker)) :]


def _current_runtime(protocol: str) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
    }


@dataclass(frozen=True, slots=True)
class ProcessResult:
    status: str
    value: Any = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    exit_code: int | None = None
    pid: int | None = None
    elapsed_ms: float | None = None
    termination: str | None = None
    signal_name: str | None = None
    oom_suspected: bool = False
    mode: str | None = None
    runtime: dict[str, Any] | None = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "termination": self.termination,
            "signal": self.signal_name,
            "oom_suspected": self.oom_suspected,
            "runtime": self.runtime,
        }


class ProcessTimeoutError(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        pid: int | None = None,
        exit_code: int | None = None,
        elapsed_ms: float | None = None,
        termination: str = "timeout_terminate",
        signal_name: str | None = None,
        mode: str | None = None,
        runtime: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.pid = pid
        self.exit_code = exit_code
        self.elapsed_ms = elapsed_ms
        self.termination = termination
        self.signal_name = signal_name
        self.mode = mode
        self.runtime = runtime

    def diagnostics(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "termination": self.termination,
            "signal": self.signal_name,
            "oom_suspected": False,
            "runtime": self.runtime,
        }


def _worker_entry(
    result_queue: mp.Queue,
    target: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    try:
        result_queue.put(("success", target(*args, **kwargs), None, None, None))
    except BaseException as exc:
        result_queue.put(
            (
                "error",
                None,
                type(exc).__name__,
                str(exc),
                traceback.format_exc(),
            )
        )


class ProcessRunner:
    """Runs isolated model work in either the current Python or a dedicated venv Python."""

    def __init__(self, *, start_method: str = "spawn") -> None:
        self._context = mp.get_context(start_method)

    def run(
        self,
        target: Callable[..., Any],
        *args: Any,
        timeout_sec: float,
        **kwargs: Any,
    ) -> ProcessResult:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")

        started = time.perf_counter()
        runtime = _current_runtime("multiprocessing-spawn")
        result_queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_worker_entry,
            args=(result_queue, target, args, kwargs),
        )
        process.start()
        pid = process.pid
        process.join(timeout_sec)

        if process.is_alive():
            termination = "timeout_terminate"
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                termination = "timeout_kill"
                process.kill()
                process.join(timeout=2.0)
            exit_code = process.exitcode
            _, signal_name, _ = _exit_diagnostics(exit_code)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result_queue.close()
            result_queue.cancel_join_thread()
            process.close()
            raise ProcessTimeoutError(
                f"child process exceeded timeout of {timeout_sec:.3f}s",
                pid=pid,
                exit_code=exit_code,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                mode="spawn",
                runtime=runtime,
            )

        exit_code = process.exitcode
        termination, signal_name, oom_suspected = _exit_diagnostics(exit_code)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        try:
            status, value, error_type, error_message, tb = result_queue.get(timeout=0.5)
        except queue.Empty:
            if oom_suspected:
                message = (
                    "worker exited without returning a result after SIGKILL; "
                    "possible OOM or external kill"
                )
            else:
                message = "worker exited without returning a result"
            return ProcessResult(
                status="error",
                error_type="WorkerExited",
                error_message=message,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
                mode="spawn",
                runtime=runtime,
            )
        finally:
            result_queue.close()
            result_queue.cancel_join_thread()
            process.close()

        return ProcessResult(
            status=status,
            value=value,
            error_type=error_type,
            error_message=_tail_text(error_message),
            traceback=_tail_text(tb),
            exit_code=exit_code,
            pid=pid,
            elapsed_ms=elapsed_ms,
            termination=termination,
            signal_name=signal_name,
            oom_suspected=oom_suspected,
            mode="spawn",
            runtime=runtime,
        )

    def run_external_worker(
        self,
        python_executable: str,
        mode: str,
        task: dict[str, Any],
        *,
        timeout_sec: float,
    ) -> ProcessResult:
        """Run the versioned JSON worker protocol under another Python interpreter/venv."""
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        if mode not in {"single", "repeated"}:
            raise ValueError("external worker mode must be 'single' or 'repeated'")
        executable = str(python_executable).strip()
        if not executable:
            raise ValueError("python_executable must not be empty")

        env = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        started = time.perf_counter()
        try:
            process = subprocess.Popen(
                [executable, "-m", "edgetts_arena.core.external_worker", mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                env=env,
            )
        except OSError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return ProcessResult(
                status="error",
                error_type=type(exc).__name__,
                error_message=_tail_text(f"failed to start external worker: {exc}"),
                elapsed_ms=elapsed_ms,
                termination="spawn_error",
                mode="external",
            )

        pid = process.pid
        try:
            stdout, stderr = process.communicate(
                json.dumps(task, ensure_ascii=False), timeout=timeout_sec
            )
        except subprocess.TimeoutExpired:
            termination = "timeout_terminate"
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                termination = "timeout_kill"
                process.kill()
                process.wait(timeout=2.0)
            exit_code = process.returncode
            _, signal_name, _ = _exit_diagnostics(exit_code)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            raise ProcessTimeoutError(
                f"external worker exceeded timeout of {timeout_sec:.3f}s",
                pid=pid,
                exit_code=exit_code,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                mode="external",
            )

        exit_code = process.returncode
        termination, signal_name, oom_suspected = _exit_diagnostics(exit_code)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        stderr_tail = _tail_text(stderr)
        if exit_code != 0:
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerExited",
                error_message=stderr_tail or f"external worker exited with code {exit_code}",
                traceback=stderr_tail,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
                mode="external",
            )

        result_line = next(
            (line for line in reversed(stdout.splitlines()) if line.startswith(_EXTERNAL_RESULT_PREFIX)),
            None,
        )
        if result_line is None:
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerProtocolError",
                error_message="external worker completed without a result frame",
                traceback=stderr_tail,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
                mode="external",
            )
        try:
            value = json.loads(result_line[len(_EXTERNAL_RESULT_PREFIX) :])
        except json.JSONDecodeError as exc:
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerProtocolError",
                error_message=f"invalid external worker result JSON: {exc}",
                traceback=stderr_tail,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
                mode="external",
            )
        if not isinstance(value, dict):
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerProtocolError",
                error_message="external worker result frame must contain a JSON object",
                traceback=stderr_tail,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
                mode="external",
            )

        raw_runtime = value.pop("_worker_runtime", None)
        if not isinstance(raw_runtime, dict) or raw_runtime.get("protocol_version") != _EXTERNAL_PROTOCOL_VERSION:
            found = raw_runtime.get("protocol_version") if isinstance(raw_runtime, dict) else None
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerProtocolError",
                error_message=(
                    f"external worker protocol mismatch: expected {_EXTERNAL_PROTOCOL_VERSION}, got {found}"
                ),
                traceback=stderr_tail,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
                mode="external",
            )
        runtime = {"protocol": "external-json", **raw_runtime}
        return ProcessResult(
            status="success",
            value=value,
            exit_code=exit_code,
            pid=pid,
            elapsed_ms=elapsed_ms,
            termination=termination,
            signal_name=signal_name,
            oom_suspected=oom_suspected,
            mode="external",
            runtime=runtime,
        )
