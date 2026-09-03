from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class ProcessResult:
    status: str
    value: Any = None
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    exit_code: int | None = None


class ProcessTimeoutError(TimeoutError):
    pass


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
        result_queue = self._context.Queue(maxsize=1)
        process = self._context.Process(
            target=_worker_entry,
            args=(result_queue, target, args, kwargs),
        )
        process.start()
        process.join(timeout_sec)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
            if process.is_alive():
                process.kill()
                process.join(timeout=2.0)
            raise ProcessTimeoutError(
                f"child process exceeded timeout of {timeout_sec:.3f}s"
            )

        try:
            status, value, error_type, error_message, tb = result_queue.get(timeout=0.5)
        except queue.Empty:
            return ProcessResult(
                status="error",
                error_type="WorkerExited",
                error_message="worker exited without returning a result",
                exit_code=process.exitcode,
            )
        finally:
            result_queue.close()

        return ProcessResult(
            status=status,
            value=value,
            error_type=error_type,
            error_message=error_message,
            traceback=tb,
            exit_code=process.exitcode,
        )
