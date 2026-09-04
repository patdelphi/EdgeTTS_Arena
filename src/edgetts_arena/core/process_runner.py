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

from edgetts_arena.core.oom_diagnostics import (
    application_error_type,
    classify_oom,
    read_linux_cgroup_memory_events,
)

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
    oom_classification: str = "none"
    oom_evidence: dict[str, Any] | None = None
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
            "oom_classification": self.oom_classification,
            "oom_evidence": self.oom_evidence,
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
            "oom_classification": "none",
            "oom_evidence": None,
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


def _oom_details(
    *,
    exit_code: int | None,
    error_type: str | None,
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> tuple[str, bool, dict[str, Any] | None]:
    return classify_oom(
        exit_code=exit_code,
        error_type=error_type,
        before=before,
        after=after,
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
        cgroup_before = read_linux_cgroup_memory_events()
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
        termination, signal_name, _ = _exit_diagnostics(exit_code)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cgroup_after = read_linux_cgroup_memory_events()
        try:
            status, value, error_type, error_message, tb = result_queue.get(timeout=0.5)
        except queue.Empty:
            oom_classification, oom_suspected, oom_evidence = _oom_details(
                exit_code=exit_code,
                error_type=None,
                before=cgroup_before,
                after=cgroup_after,
            )
            if oom_classification == "cgroup_oom_kill_observed":
                message = "worker exited after SIGKILL while cgroup memory.events recorded an oom_kill increment"
            elif oom_suspected:
                message = "worker exited without returning a result after SIGKILL; possible OOM or external kill"
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
                oom_classification=oom_classification,
                oom_evidence=oom_evidence,
                mode="spawn",
                runtime=runtime,
            )
        finally:
            result_queue.close()
            result_queue.cancel_join_thread()
            process.close()

        effective_error_type = error_type or application_error_type(value)
        oom_classification, oom_suspected, oom_evidence = _oom_details(
            exit_code=exit_code,
            error_type=effective_error_type,
            before=cgroup_before,
            after=cgroup_after,
        )
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
            oom_classification=oom_classification,
            oom_evidence=oom_evidence,
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
        # Pin the worker's std streams to UTF-8 so the JSON protocol survives a
        # non-UTF-8 locale (e.g. Windows cp936/GBK). The parent talks UTF-8 below
        # (Popen encoding="utf-8"); without this the worker decodes stdin as the
        # locale codec (turning non-ASCII text into mojibake / lone surrogates that
        # break Rust tokenizers -> "TextEncodeInput must be Union[...]"), and it
        # re-encodes its UTF-8 result frame as GBK on stdout, which the parent then
        # fails to decode -> the frame is lost ("completed without a result frame").
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        started = time.perf_counter()
        cgroup_before = read_linux_cgroup_memory_events()
        try:
            process = subprocess.Popen(
                [executable, "-m", "edgetts_arena.core.external_worker", mode],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                # Never let a stray non-UTF-8 byte (e.g. a native library writing
                # GBK straight to fd 1/2) kill the reader thread and drop the whole
                # result frame; replace undecodable bytes instead of raising.
                errors="replace",
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
        termination, signal_name, _ = _exit_diagnostics(exit_code)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        cgroup_after = read_linux_cgroup_memory_events()
        stderr_tail = _tail_text(stderr)
        if exit_code != 0:
            oom_classification, oom_suspected, oom_evidence = _oom_details(
                exit_code=exit_code,
                error_type=None,
                before=cgroup_before,
                after=cgroup_after,
            )
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
                oom_classification=oom_classification,
                oom_evidence=oom_evidence,
                mode="external",
            )

        result_line = next(
            (line for line in reversed((stdout or "").splitlines()) if line.startswith(_EXTERNAL_RESULT_PREFIX)),
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
                mode="external",
            )
        runtime = {"protocol": "external-json", **raw_runtime}
        effective_error_type = application_error_type(value)
        oom_classification, oom_suspected, oom_evidence = _oom_details(
            exit_code=exit_code,
            error_type=effective_error_type,
            before=cgroup_before,
            after=cgroup_after,
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
            oom_classification=oom_classification,
            oom_evidence=oom_evidence,
            mode="external",
            runtime=runtime,
        )
