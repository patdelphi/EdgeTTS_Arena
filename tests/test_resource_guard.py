import pytest

from edgetts_arena.core import ResourceGuard
from edgetts_arena.core.config import ResourceGuardSettings
from edgetts_arena.core.errors import ArenaError


MB = 1024 * 1024


def test_resource_guard_ok_soft_and_hard() -> None:
    settings = ResourceGuardSettings(
        min_available_memory_mb_soft=1000,
        min_available_memory_mb_hard=500,
    )
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
    guard = ResourceGuard(
        ResourceGuardSettings(),
        available_memory_provider=lambda: 4096 * MB,
        cpu_count_provider=lambda: 4,
    )
    assert guard.clamp_threads(2) == 2
    assert guard.clamp_threads(16) == 4
