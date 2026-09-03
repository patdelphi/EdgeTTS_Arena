from __future__ import annotations

import os
import signal
from pathlib import Path
from typing import Any

_MEMORY_ERROR_TYPES = {"MemoryError", "worker_memory_error"}


def parse_memory_events(text: str) -> dict[str, int]:
    counters: dict[str, int] = {}
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            counters[key] = int(value)
        except ValueError:
            continue
    return counters


def read_linux_cgroup_memory_events(
    *,
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    cgroup_root: Path = Path("/sys/fs/cgroup"),
) -> dict[str, Any] | None:
    if os.name != "posix" or not proc_cgroup.is_file():
        return None
    try:
        relative: str | None = None
        for raw in proc_cgroup.read_text(encoding="utf-8").splitlines():
            fields = raw.split(":", 2)
            if len(fields) == 3 and fields[0] == "0" and fields[1] == "":
                relative = fields[2]
                break
        if relative is None:
            return None
        events_path = cgroup_root / relative.lstrip("/") / "memory.events"
        if not events_path.is_file():
            return None
        counters = parse_memory_events(events_path.read_text(encoding="utf-8"))
        return {"source": "linux-cgroup-v2", "path": str(events_path), "counters": counters}
    except (OSError, UnicodeError):
        return None


def application_error_type(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    error = value.get("error")
    if isinstance(error, dict) and error.get("type"):
        return str(error["type"])
    measurements = value.get("measurements")
    if isinstance(measurements, list):
        for item in measurements:
            if not isinstance(item, dict):
                continue
            nested = item.get("error")
            if isinstance(nested, dict) and nested.get("type") in _MEMORY_ERROR_TYPES:
                return str(nested["type"])
    return None


def classify_oom(
    *,
    exit_code: int | None,
    error_type: str | None = None,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
) -> tuple[str, bool, dict[str, Any] | None]:
    if error_type in _MEMORY_ERROR_TYPES:
        return (
            "worker_memory_error",
            True,
            {"source": "worker-exception", "error_type": error_type},
        )

    sigkill = getattr(signal, "SIGKILL", 9)
    killed = exit_code is not None and exit_code < 0 and -exit_code == sigkill
    if not killed:
        return "none", False, None

    evidence = _cgroup_delta(before, after)
    if evidence is not None and int(evidence.get("oom_kill_delta") or 0) > 0:
        return "cgroup_oom_kill_observed", True, evidence
    if evidence is not None:
        evidence = {**evidence, "note": "SIGKILL observed without cgroup oom_kill increment"}
    else:
        evidence = {"source": "process-exit", "signal": "SIGKILL"}
    return "sigkill_suspected", True, evidence


def _cgroup_delta(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    if before.get("source") != "linux-cgroup-v2" or after.get("source") != "linux-cgroup-v2":
        return None
    if before.get("path") != after.get("path"):
        return None
    before_counters = before.get("counters")
    after_counters = after.get("counters")
    if not isinstance(before_counters, dict) or not isinstance(after_counters, dict):
        return None
    keys = {"oom", "oom_kill", "oom_group_kill", "high", "max"}
    deltas = {
        f"{key}_delta": int(after_counters.get(key, 0)) - int(before_counters.get(key, 0))
        for key in keys
    }
    return {
        "source": "linux-cgroup-v2-memory.events",
        "path": str(after.get("path")),
        **deltas,
    }
