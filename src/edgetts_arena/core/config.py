from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class ResourceGuardSettings:
    min_available_memory_mb_soft: int = 1536
    min_available_memory_mb_hard: int = 768


@dataclass(frozen=True, slots=True)
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    default_num_threads: int = 4
    inference_timeout_sec: int = 60
    resource_guard: ResourceGuardSettings = ResourceGuardSettings()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("application config root must be a mapping")
    return data


def load_settings(path: str | Path = "config/app_config.yaml") -> AppSettings:
    raw = _read_yaml(Path(path))
    guard_raw = raw.get("resource_guard", {}) or {}
    guard = ResourceGuardSettings(
        min_available_memory_mb_soft=int(
            os.getenv(
                "EDGETTS_ARENA_MIN_MEMORY_SOFT_MB",
                guard_raw.get("min_available_memory_mb_soft", 1536),
            )
        ),
        min_available_memory_mb_hard=int(
            os.getenv(
                "EDGETTS_ARENA_MIN_MEMORY_HARD_MB",
                guard_raw.get("min_available_memory_mb_hard", 768),
            )
        ),
    )
    return AppSettings(
        host=os.getenv("EDGETTS_ARENA_HOST", raw.get("host", "127.0.0.1")),
        port=int(os.getenv("EDGETTS_ARENA_PORT", raw.get("port", 8000))),
        log_level=os.getenv("EDGETTS_ARENA_LOG_LEVEL", raw.get("log_level", "INFO")),
        default_num_threads=int(
            os.getenv(
                "EDGETTS_ARENA_DEFAULT_THREADS",
                raw.get("default_num_threads", 4),
            )
        ),
        inference_timeout_sec=int(
            os.getenv(
                "EDGETTS_ARENA_INFERENCE_TIMEOUT_SEC",
                raw.get("inference_timeout_sec", 60),
            )
        ),
        resource_guard=guard,
    )
