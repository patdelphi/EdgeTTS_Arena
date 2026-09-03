import pytest

from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.config import ResourceGuardSettings
from edgetts_arena.core.errors import ArenaError


MB = 1024 * 1024


def test_resource_guard_ok_soft_and_hard() -> None:
    settings = ResourceGuardSettings(min_available_memory_mb_soft=1000, min_available_memory_mb_hard=500)
    guard = ResourceGuard(settings, available_memory_provider=lambda: 1200 * MB)
    assert guard.assess().level == "ok"

    soft = ResourceGuard(settings, available_memory_provider=lambda: 700 * MB)
    assert soft.assess(execution_mode="sequential").level == "soft"
    with pytest.raises(ArenaError) as exc_info:
        soft.assess(execution_mode="concurrent")
    assert exc_info.value.code == 2001

    hard = ResourceGuard(settings, available_memory_provider=lambda: 400 * MB)
    with pytest.raises(ArenaError) as exc_info:
        hard.assess()
    assert exc_info.value.code == 2002


def test_resource_guard_clamps_threads() -> None:
    guard = ResourceGuard(ResourceGuardSettings(), available_memory_provider=lambda: 4096 * MB, cpu_count_provider=lambda: 4)
    assert guard.clamp_threads(2) == 2
    assert guard.clamp_threads(16) == 4


def test_concurrent_plan_fairly_splits_cpu_budget() -> None:
    guard = ResourceGuard(
        ResourceGuardSettings(
            min_available_memory_mb_soft=1000,
            min_available_memory_mb_hard=500,
            min_available_memory_mb_per_concurrent_model=400,
            max_concurrent_models=4,
        ),
        available_memory_provider=lambda: 4096 * MB,
        cpu_count_provider=lambda: 8,
    )
    plan = guard.plan_execution(
        execution_mode="concurrent",
        model_count=4,
        requested_threads_per_model=4,
    )
    assert plan.profile == "pressure"
    assert plan.threads_per_model == 2
    assert plan.total_threads_budget == 8
    assert plan.minimum_memory_budget_mb == 1600
    assert plan.warnings


def test_concurrent_plan_rejects_memory_and_model_budget() -> None:
    settings = ResourceGuardSettings(
        min_available_memory_mb_soft=1000,
        min_available_memory_mb_hard=500,
        min_available_memory_mb_per_concurrent_model=700,
        max_concurrent_models=2,
    )
    low = ResourceGuard(
        settings, available_memory_provider=lambda: 1200 * MB, cpu_count_provider=lambda: 8
    )
    with pytest.raises(ArenaError) as exc_info:
        low.plan_execution(execution_mode="concurrent", model_count=2, requested_threads_per_model=2)
    assert exc_info.value.error_type == "concurrent_memory_budget"

    enough = ResourceGuard(
        settings, available_memory_provider=lambda: 4096 * MB, cpu_count_provider=lambda: 8
    )
    with pytest.raises(ArenaError) as exc_info:
        enough.plan_execution(execution_mode="concurrent", model_count=3, requested_threads_per_model=2)
    assert exc_info.value.error_type == "concurrent_model_limit"
