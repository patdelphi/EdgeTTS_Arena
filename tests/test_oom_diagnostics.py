from __future__ import annotations

import signal
from pathlib import Path

from edgetts_arena.core.oom_diagnostics import (
    application_error_type,
    classify_oom,
    parse_memory_events,
    read_linux_cgroup_memory_events,
)


def test_parse_memory_events_ignores_malformed_lines() -> None:
    parsed = parse_memory_events("low 0\nhigh 2\noom 3\noom_kill 1\nbad\nmax nope\n")
    assert parsed == {"low": 0, "high": 2, "oom": 3, "oom_kill": 1}


def test_cgroup_sigkill_observation_is_stronger_than_plain_sigkill() -> None:
    before = {
        "source": "linux-cgroup-v2",
        "path": "/sys/fs/cgroup/demo/memory.events",
        "counters": {"oom": 2, "oom_kill": 1, "oom_group_kill": 0},
    }
    after = {
        "source": "linux-cgroup-v2",
        "path": "/sys/fs/cgroup/demo/memory.events",
        "counters": {"oom": 3, "oom_kill": 2, "oom_group_kill": 0},
    }
    classification, suspected, evidence = classify_oom(
        exit_code=-getattr(signal, "SIGKILL", 9), before=before, after=after
    )
    assert classification == "cgroup_oom_kill_observed"
    assert suspected is True
    assert evidence and evidence["oom_kill_delta"] == 1
    assert evidence["oom_delta"] == 1


def test_sigkill_without_cgroup_counter_remains_suspected() -> None:
    classification, suspected, evidence = classify_oom(
        exit_code=-getattr(signal, "SIGKILL", 9), before=None, after=None
    )
    assert classification == "sigkill_suspected"
    assert suspected is True
    assert evidence == {"source": "process-exit", "signal": "SIGKILL"}


def test_non_sigkill_is_not_classified_as_oom() -> None:
    classification, suspected, evidence = classify_oom(exit_code=2)
    assert classification == "none"
    assert suspected is False
    assert evidence is None


def test_worker_memory_error_is_explicit_evidence() -> None:
    classification, suspected, evidence = classify_oom(
        exit_code=0, error_type="worker_memory_error"
    )
    assert classification == "worker_memory_error"
    assert suspected is True
    assert evidence == {"source": "worker-exception", "error_type": "worker_memory_error"}


def test_application_error_type_finds_single_and_repeated_memory_error() -> None:
    assert application_error_type({"error": {"type": "worker_memory_error"}}) == "worker_memory_error"
    repeated = {
        "error": None,
        "measurements": [
            {"status": "success", "error": None},
            {"status": "error", "error": {"type": "worker_memory_error"}},
        ],
    }
    assert application_error_type(repeated) == "worker_memory_error"


def test_read_linux_cgroup_memory_events_from_fixture(tmp_path: Path, monkeypatch) -> None:
    import edgetts_arena.core.oom_diagnostics as oom_module

    monkeypatch.setattr(oom_module.os, "name", "posix")
    proc = tmp_path / "cgroup"
    proc.write_text("0::/demo.slice\n", encoding="utf-8")
    root = tmp_path / "sysfs"
    events = root / "demo.slice" / "memory.events"
    events.parent.mkdir(parents=True)
    events.write_text("oom 4\noom_kill 2\nhigh 1\n", encoding="utf-8")

    snapshot = read_linux_cgroup_memory_events(proc_cgroup=proc, cgroup_root=root)
    assert snapshot is not None
    assert snapshot["source"] == "linux-cgroup-v2"
    assert snapshot["counters"]["oom_kill"] == 2
