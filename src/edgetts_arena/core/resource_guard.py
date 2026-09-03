from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Callable

import psutil

from edgetts_arena.core.config import ResourceGuardSettings
from edgetts_arena.core.errors import ArenaError


def _quota_to_cpu_count(quota: int, period: int) -> int | None:
    """Convert a cgroup CPU quota into a conservative whole-core budget."""
    if quota <= 0 or period <= 0:
        return None
    return max(1, quota // period)


def _cgroup_cpu_quota_count() -> int | None:
    """Best-effort Linux cgroup v2/v1 CPU quota detection."""
    try:
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.is_file():
            quota_raw, period_raw = cpu_max.read_text(encoding="utf-8").strip().split()[:2]
            if quota_raw != "max":
                value = _quota_to_cpu_count(int(quota_raw), int(period_raw))
                if value is not None:
                    return value
    except (OSError, ValueError, IndexError):
        pass

    v1_candidates = (
        (Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")),
        (Path("/sys/fs/cgroup/cpu.cfs_quota_us"), Path("/sys/fs/cgroup/cpu.cfs_period_us")),
    )
    for quota_path, period_path in v1_candidates:
        try:
            if not quota_path.is_file() or not period_path.is_file():
                continue
            value = _quota_to_cpu_count(
                int(quota_path.read_text(encoding="utf-8").strip()),
                int(period_path.read_text(encoding="utf-8").strip()),
            )
            if value is not None:
                return value
        except (OSError, ValueError):
            continue
    return None


def effective_cpu_count() -> int:
    """Return the conservative CPU budget effectively available to this process."""
    candidates: list[int] = []

    process_count = getattr(os, "process_cpu_count", None)
    if callable(process_count):
        try:
            value = process_count()
            if value:
                candidates.append(max(1, int(value)))
        except (OSError, TypeError, ValueError):
            pass

    affinity = getattr(os, "sched_getaffinity", None)
    if callable(affinity):
        try:
            value = len(affinity(0))
            if value:
                candidates.append(max(1, int(value)))
        except (OSError, TypeError, ValueError):
            pass

    quota_count = _cgroup_cpu_quota_count()
    if quota_count is not None:
        candidates.append(quota_count)

    candidates.append(max(1, int(os.cpu_count() or 1)))
    return max(1, min(candidates))


@dataclass(frozen=True, slots=True)
class ResourceAssessment:
    available_memory_mb: float
    level: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    execution_mode: str
    model_count: int
    requested_threads_per_model: int
    threads_per_model: int
    total_threads_budget: int
    logical_cores: int
    available_memory_mb: float
    minimum_memory_budget_mb: int
    warnings: tuple[str, ...] = ()

    @property
    def profile(self) -> str:
        return "pressure" if self.execution_mode == "concurrent" else "baseline"

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "execution_mode": self.execution_mode,
            "model_count": self.model_count,
            "requested_threads_per_model": self.requested_threads_per_model,
            "threads_per_model": self.threads_per_model,
            "total_threads_budget": self.total_threads_budget,
            "logical_cores": self.logical_cores,
            "available_memory_mb": round(self.available_memory_mb, 1),
            "minimum_memory_budget_mb": self.minimum_memory_budget_mb,
            "warnings": list(self.warnings),
        }


class ResourceGuard:
    def __init__(
        self,
        settings: ResourceGuardSettings,
        *,
        available_memory_provider: Callable[[], int] | None = None,
        cpu_count_provider: Callable[[], int | None] | None = None,
    ) -> None:
        self.settings = settings
        self._available_memory_provider = available_memory_provider or (lambda: psutil.virtual_memory().available)
        self._cpu_count_provider = cpu_count_provider or effective_cpu_count

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

    def plan_execution(
        self,
        *,
        execution_mode: str,
        model_count: int,
        requested_threads_per_model: int,
    ) -> ExecutionPlan:
        if execution_mode not in {"sequential", "concurrent"}:
            raise ValueError("execution_mode must be sequential or concurrent")
        if model_count < 1:
            raise ValueError("model_count must be >= 1")
        if requested_threads_per_model < 1:
            raise ValueError("requested_threads_per_model must be >= 1")

        assessment = self.assess(execution_mode=execution_mode)
        logical = max(1, int(self._cpu_count_provider() or 1))
        warnings = list(assessment.warnings)

        if execution_mode == "sequential":
            threads = min(requested_threads_per_model, logical)
            if threads < requested_threads_per_model:
                warnings.append(
                    f"threads/model clamped from {requested_threads_per_model} to {threads} by logical core count"
                )
            return ExecutionPlan(
                execution_mode=execution_mode,
                model_count=model_count,
                requested_threads_per_model=requested_threads_per_model,
                threads_per_model=threads,
                total_threads_budget=threads,
                logical_cores=logical,
                available_memory_mb=assessment.available_memory_mb,
                minimum_memory_budget_mb=self.settings.min_available_memory_mb_hard,
                warnings=tuple(warnings),
            )

        if model_count > self.settings.max_concurrent_models:
            raise ArenaError(
                2001,
                f"concurrent model count {model_count} exceeds configured maximum {self.settings.max_concurrent_models}",
                error_type="concurrent_model_limit",
            )
        if model_count > logical:
            raise ArenaError(
                2001,
                f"concurrent model count {model_count} exceeds effective CPU budget of {logical} cores",
                error_type="concurrent_cpu_budget",
            )

        minimum_memory = max(
            self.settings.min_available_memory_mb_soft,
            model_count * self.settings.min_available_memory_mb_per_concurrent_model,
        )
        if assessment.available_memory_mb < minimum_memory:
            raise ArenaError(
                2001,
                f"concurrent execution requires at least {minimum_memory} MB available for {model_count} models; "
                f"found {assessment.available_memory_mb:.0f} MB",
                error_type="concurrent_memory_budget",
            )

        fair_share = max(1, logical // model_count)
        threads = min(requested_threads_per_model, fair_share)
        if threads < requested_threads_per_model:
            warnings.append(
                f"concurrent threads/model clamped from {requested_threads_per_model} to {threads} "
                f"to fit {model_count} models within {logical} logical cores"
            )
        return ExecutionPlan(
            execution_mode=execution_mode,
            model_count=model_count,
            requested_threads_per_model=requested_threads_per_model,
            threads_per_model=threads,
            total_threads_budget=threads * model_count,
            logical_cores=logical,
            available_memory_mb=assessment.available_memory_mb,
            minimum_memory_budget_mb=minimum_memory,
            warnings=tuple(warnings),
        )
