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
