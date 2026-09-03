from __future__ import annotations

import multiprocessing as mp
import queue
import signal
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable


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
    """Runs picklable callables in a terminable child process."""

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
