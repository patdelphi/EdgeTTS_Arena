from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

_EXTERNAL_RESULT_PREFIX = "__EDGETTS_ARENA_RESULT__="


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
        # SIGKILL is a common symptom of a kernel/cgroup OOM kill, but it can
        # also be an external administrative kill. Keep this deliberately
        # probabilistic rather than claiming an OOM without kernel evidence.
        oom_suspected = signal_number == getattr(signal, "SIGKILL", 9)
        return "signal", signal_name, oom_suspected
    return "exit_code", None, False


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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "termination": self.termination,
            "signal": self.signal_name,
            "oom_suspected": self.oom_suspected,
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
    ) -> None:
        super().__init__(message)
        self.pid = pid
        self.exit_code = exit_code
        self.elapsed_ms = elapsed_ms
        self.termination = termination
        self.signal_name = signal_name

    def diagnostics(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "exit_code": self.exit_code,
            "elapsed_ms": self.elapsed_ms,
            "termination": self.termination,
            "signal": self.signal_name,
            # A watchdog-triggered SIGKILL/SIGTERM is caused by us, so it is
            # not evidence of OOM.
            "oom_suspected": False,
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
            )
        finally:
            result_queue.close()
            result_queue.cancel_join_thread()
            process.close()

        return ProcessResult(
            status=status,
            value=value,
            error_type=error_type,
            error_message=error_message,
            traceback=tb,
            exit_code=exit_code,
            pid=pid,
            elapsed_ms=elapsed_ms,
            termination=termination,
            signal_name=signal_name,
            oom_suspected=oom_suspected,
        )

    def run_external_worker(
        self,
        python_executable: str,
        mode: str,
        task: dict[str, Any],
        *,
        timeout_sec: float,
    ) -> ProcessResult:
        """Run the JSON worker protocol under another Python interpreter/venv."""
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
                error_message=f"failed to start external worker: {exc}",
                elapsed_ms=elapsed_ms,
                termination="spawn_error",
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
            )

        exit_code = process.returncode
        termination, signal_name, oom_suspected = _exit_diagnostics(exit_code)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if exit_code != 0:
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerExited",
                error_message=(stderr.strip() or f"external worker exited with code {exit_code}"),
                traceback=stderr or None,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
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
                traceback=stderr or None,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
            )
        try:
            value = json.loads(result_line[len(_EXTERNAL_RESULT_PREFIX):])
        except json.JSONDecodeError as exc:
            return ProcessResult(
                status="error",
                error_type="ExternalWorkerProtocolError",
                error_message=f"invalid external worker result JSON: {exc}",
                traceback=stderr or None,
                exit_code=exit_code,
                pid=pid,
                elapsed_ms=elapsed_ms,
                termination=termination,
                signal_name=signal_name,
                oom_suspected=oom_suspected,
            )
        return ProcessResult(
            status="success",
            value=value,
            exit_code=exit_code,
            pid=pid,
            elapsed_ms=elapsed_ms,
            termination=termination,
            signal_name=signal_name,
            oom_suspected=oom_suspected,
        )
