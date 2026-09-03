from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

import psutil

from edgetts_arena.core.config import ResourceGuardSettings
from edgetts_arena.core.errors import ArenaError


@dataclass(frozen=True, slots=True)
class ResourceAssessment:
    available_memory_mb: float
    level: str
    warnings: tuple[str, ...] = ()


class ResourceGuard:
    def __init__(
        self,
        settings: ResourceGuardSettings,
        *,
        available_memory_provider: Callable[[], int] | None = None,
        cpu_count_provider: Callable[[], int | None] | None = None,
    ) -> None:
        self.settings = settings
        self._available_memory_provider = (
            available_memory_provider or (lambda: psutil.virtual_memory().available)
        )
        self._cpu_count_provider = cpu_count_provider or os.cpu_count

    def assess(self, *, execution_mode: str = "sequential") -> ResourceAssessment:
        available_mb = self._available_memory_provider() / (1024 * 1024)
        hard = self.settings.min_available_memory_mb_hard
        soft = self.settings.min_available_memory_mb_soft
        if available_mb < hard:
            raise ArenaError(
                2002,
                f"available memory {available_mb:.0f} MB is below hard threshold {hard} MB",
                error_type="memory_guard_hard",
            )
        if available_mb < soft:
            if execution_mode == "concurrent":
                raise ArenaError(
                    2001,
                    f"concurrent execution blocked below soft memory threshold {soft} MB",
                    error_type="resource_guard_soft",
                )
            return ResourceAssessment(
                available_memory_mb=available_mb,
                level="soft",
                warnings=("available memory is below the configured soft threshold",),
            )
        return ResourceAssessment(available_memory_mb=available_mb, level="ok")

    def clamp_threads(self, requested: int) -> int:
        if requested < 1:
            raise ValueError("requested threads must be >= 1")
        logical = max(1, int(self._cpu_count_provider() or 1))
        return min(requested, logical)
