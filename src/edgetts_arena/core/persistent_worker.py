"""Parent-side handle for a persistent (warm) external worker process.

Wraps one long-lived ``edgetts_arena.core.warm_worker`` subprocess running in a
model's dedicated venv. The model is loaded once (:meth:`start`) and then reused
for many :meth:`submit_infer` / :meth:`submit_repeated` calls, so heavy models are
not re-loaded (a tens-of-seconds cold start) on every benchmark run.

Design notes:
- A background reader thread turns stdout into a queue of result frames, so a
  request can be awaited with a real timeout without blocking the event loop.
- stderr is drained concurrently to avoid pipe deadlock and to keep the tail for
  crash diagnostics.
- Requests are serialized with a lock: one warm worker serves one model, one task
  at a time. Concurrent Arena runs therefore never interleave on the same worker.
- On inference timeout the worker is killed (it is wedged) and a
  :class:`ProcessTimeoutError` is raised, mirroring the one-shot runner so the
  caller's existing timeout handling applies unchanged.
- Any transport failure (crash, broken pipe, protocol error) raises
  :class:`WarmWorkerError` so the caller can discard the worker and fall back to
  a one-shot run instead of surfacing a confusing error.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import psutil

from edgetts_arena.core.process_runner import (
    ProcessResult,
    ProcessTimeoutError,
    _exit_diagnostics,
    _tail_text,
)
from edgetts_arena.core.warm_worker import RESULT_PREFIX, WARM_PROTOCOL_VERSION


class WarmWorkerError(RuntimeError):
    """A warm worker could not serve a request (crashed, protocol error, unloadable)."""


class _FrameTimeout(Exception):
    """Internal: no result frame arrived before the deadline."""


class PersistentExternalWorker:
    def __init__(
        self,
        *,
        python_executable: str,
        model_id: str,
        adapter: str,
        model_path: str,
        num_threads: int,
    ) -> None:
        self.python_executable = str(python_executable)
        self.model_id = model_id
        self.adapter_name = adapter
        self.model_path = model_path
        self.num_threads = int(num_threads)
        self.pid: int | None = None
        self.last_rss_mb: float | None = None

        self._process: subprocess.Popen[str] | None = None
        self._frames: "queue.Queue[dict[str, Any] | None]" = queue.Queue()
        self._stderr: deque[str] = deque(maxlen=200)
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------ lifecycle
    def start(self, *, timeout_sec: float) -> None:
        """Spawn the worker and load the model once. Raises WarmWorkerError on failure."""
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        executable = self.python_executable.strip()
        if not executable:
            raise WarmWorkerError(f"warm worker for '{self.model_id}' has no Python executable")

        env = os.environ.copy()
        src_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src_root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        # Pin UTF-8 std streams so the JSON protocol survives a non-UTF-8 locale
        # (Windows cp936/GBK), exactly like the one-shot external worker.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        try:
            self._process = subprocess.Popen(
                [executable, "-m", "edgetts_arena.core.warm_worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
        except OSError as exc:
            raise WarmWorkerError(
                f"failed to start warm worker for '{self.model_id}': {exc}"
            ) from exc

        self.pid = self._process.pid
        threading.Thread(target=self._read_stdout, daemon=True).start()
        threading.Thread(target=self._read_stderr, daemon=True).start()

        self._send(
            {
                "op": "load",
                "adapter": self.adapter_name,
                "model_path": self.model_path,
                "num_threads": self.num_threads,
            }
        )
        try:
            frame = self._await_frame(timeout_sec)
        except (_FrameTimeout, WarmWorkerError) as exc:
            self.shutdown()
            raise WarmWorkerError(
                f"warm worker for '{self.model_id}' did not load within {timeout_sec:.0f}s: {exc}"
            ) from exc
        if frame.get("op") != "load" or frame.get("status") != "ok":
            error = frame.get("error") or {}
            self.shutdown()
            raise WarmWorkerError(
                f"warm worker failed to load '{self.model_id}': "
                f"{error.get('message') or frame.get('status') or 'unknown error'}"
            )
        self.last_rss_mb = self._rss_from_frame(frame)

    def is_alive(self) -> bool:
        return (
            not self._closed
            and self._process is not None
            and self._process.poll() is None
        )

    def shutdown(self) -> None:
        """Gracefully stop the worker; idempotent and safe to call from cleanup."""
        if self._closed:
            return
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                stdin = process.stdin
                if stdin is not None and not stdin.closed:
                    try:
                        stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                        stdin.flush()
                    except (BrokenPipeError, ValueError, OSError):
                        pass
                try:
                    process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    pass
        finally:
            self._kill()

    def _kill(self) -> None:
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2.0)
        except (OSError, subprocess.SubprocessError):
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass

    # ------------------------------------------------------------------ requests
    def submit_infer(
        self,
        *,
        text: str,
        infer_kwargs: dict[str, Any],
        audio_path: str,
        timeout_sec: float,
    ) -> ProcessResult:
        return self._submit(
            {
                "op": "infer",
                "text": text,
                "infer_kwargs": infer_kwargs,
                "audio_path": audio_path,
            },
            timeout_sec=timeout_sec,
        )

    def submit_repeated(
        self,
        *,
        text: str,
        infer_kwargs: dict[str, Any],
        audio_path: str,
        warmup_runs: int,
        measured_runs: int,
        timeout_sec: float,
    ) -> ProcessResult:
        return self._submit(
            {
                "op": "infer_repeated",
                "text": text,
                "infer_kwargs": infer_kwargs,
                "audio_path": audio_path,
                "warmup_runs": int(warmup_runs),
                "measured_runs": int(measured_runs),
            },
            timeout_sec=timeout_sec,
        )

    def _submit(self, cmd: dict[str, Any], *, timeout_sec: float) -> ProcessResult:
        if timeout_sec <= 0:
            raise ValueError("timeout_sec must be positive")
        with self._lock:
            if not self.is_alive():
                raise WarmWorkerError(
                    f"warm worker '{self.model_id}' is not running: {self._stderr_tail()}"
                )
            started = time.perf_counter()
            self._send(cmd)
            try:
                frame = self._await_frame(timeout_sec)
            except _FrameTimeout:
                exit_code = self._kill_and_exit_code()
                _, signal_name, _ = _exit_diagnostics(exit_code)
                raise ProcessTimeoutError(
                    f"warm worker '{self.model_id}' exceeded timeout of {timeout_sec:.3f}s",
                    pid=self.pid,
                    exit_code=exit_code,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    termination="timeout_kill",
                    signal_name=signal_name,
                    mode="warm-external",
                )
            except WarmWorkerError:
                self._kill_and_exit_code()
                raise
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.last_rss_mb = self._rss_from_frame(frame) or self.last_rss_mb
            return self._to_result(frame, elapsed_ms)

    def rss_mb(self) -> float | None:
        """Resident size of the worker (MB).

        The worker self-reports its RSS in every result frame, which is the
        authoritative value: parent-side ``psutil`` measurement of a *child*
        process is unreliable on some Windows setups (it can report a few MB for a
        process actually holding tens). Parent-side is only a last-resort fallback
        before the first frame arrives.
        """
        if self.last_rss_mb is not None:
            return self.last_rss_mb
        if self.pid is None:
            return None
        try:
            return psutil.Process(self.pid).memory_info().rss / (1024 * 1024)
        except psutil.Error:
            return None

    # ------------------------------------------------------------------ internals
    def _to_result(self, frame: dict[str, Any], elapsed_ms: float) -> ProcessResult:
        runtime_raw = frame.get("_worker_runtime")
        runtime = (
            {"protocol": "warm-json", **runtime_raw}
            if isinstance(runtime_raw, dict)
            and runtime_raw.get("protocol_version") == WARM_PROTOCOL_VERSION
            else {"protocol": "warm-json", "protocol_version": WARM_PROTOCOL_VERSION}
        )
        # A received frame means the transport succeeded; the application-level
        # status lives inside the frame (value["status"]), matching the one-shot
        # runner so _run_model's payload handling applies unchanged.
        return ProcessResult(
            status="success",
            value=frame,
            exit_code=0,
            pid=self.pid,
            elapsed_ms=elapsed_ms,
            termination=None,
            signal_name=None,
            mode="warm-external",
            runtime=runtime,
        )

    def _rss_from_frame(self, frame: dict[str, Any]) -> float | None:
        value = frame.get("rss_mb")
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number >= 0 else None

    def _send(self, cmd: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.stdin.closed:
            raise WarmWorkerError(f"warm worker '{self.model_id}' stdin is unavailable")
        try:
            process.stdin.write(json.dumps(cmd, ensure_ascii=False) + "\n")
            process.stdin.flush()
        except (BrokenPipeError, ValueError, OSError) as exc:
            raise WarmWorkerError(
                f"warm worker '{self.model_id}' pipe broken: {exc}"
            ) from exc

    def _await_frame(self, timeout_sec: float) -> dict[str, Any]:
        deadline = time.perf_counter() + timeout_sec
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise _FrameTimeout()
            try:
                frame = self._frames.get(timeout=min(remaining, 0.25))
            except queue.Empty:
                if not self.is_alive():
                    raise WarmWorkerError(
                        f"warm worker '{self.model_id}' exited: {self._stderr_tail()}"
                    )
                continue
            if frame is None:  # stdout closed sentinel
                raise WarmWorkerError(
                    f"warm worker '{self.model_id}' closed stdout: {self._stderr_tail()}"
                )
            return frame

    def _read_stdout(self) -> None:
        process = self._process
        assert process is not None and process.stdout is not None
        try:
            for line in process.stdout:
                line = line.rstrip("\n")
                if not line.startswith(RESULT_PREFIX):
                    continue  # ignore library noise printed to stdout
                try:
                    self._frames.put(json.loads(line[len(RESULT_PREFIX) :]))
                except json.JSONDecodeError:
                    continue
        except (ValueError, OSError):
            pass
        finally:
            self._frames.put(None)

    def _read_stderr(self) -> None:
        process = self._process
        assert process is not None and process.stderr is not None
        try:
            for line in process.stderr:
                self._stderr.append(line.rstrip("\n"))
        except (ValueError, OSError):
            pass

    def _stderr_tail(self) -> str | None:
        return _tail_text("\n".join(self._stderr)) if self._stderr else None

    def _kill_and_exit_code(self) -> int | None:
        self._closed = True
        self._kill()
        return self._process.exitcode if self._process is not None else None
