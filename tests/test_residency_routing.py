from __future__ import annotations

from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.model_registry import ModelSpec

_ROUTE = BenchmarkService._execution_route


class _Res:
    """Minimal residency stand-in exposing only the ``keep_warm`` flag."""

    def __init__(self, keep_warm: bool) -> None:
        self.keep_warm = keep_warm


def _spec(**kwargs) -> ModelSpec:
    base = dict(
        id="m", name="m", adapter="dummy", enabled=True, keep_in_memory=False, num_threads=1
    )
    base.update(kwargs)
    return ModelSpec(**base)


def test_keep_in_memory_is_always_in_process() -> None:
    spec = _spec(keep_in_memory=True, worker_python="/fake/python")
    assert _ROUTE(spec, object(), _Res(True)) == "in_process"
    assert _ROUTE(spec, object(), _Res(False)) == "in_process"


def test_no_process_runner_is_in_process() -> None:
    spec = _spec(worker_python="/fake/python")
    assert _ROUTE(spec, None, _Res(True)) == "in_process"


def test_eager_worker_model_is_isolated() -> None:
    spec = _spec(worker_python="/fake/python")
    assert _ROUTE(spec, object(), _Res(False)) == "isolated"


def test_missing_residency_is_isolated() -> None:
    spec = _spec(worker_python="/fake/python")
    assert _ROUTE(spec, object(), None) == "isolated"


def test_keep_warm_resolved_worker_is_warm_worker() -> None:
    spec = _spec(worker_python="/fake/python")
    assert _ROUTE(spec, object(), _Res(True)) == "warm_worker"


def test_keep_warm_unresolved_worker_env_falls_back_to_isolated(monkeypatch) -> None:
    monkeypatch.delenv("EDGETTS_ARENA_TEST_UNSET", raising=False)
    spec = _spec(worker_python_env="EDGETTS_ARENA_TEST_UNSET")
    assert spec.resolve_worker_python() == ""
    assert _ROUTE(spec, object(), _Res(True)) == "isolated"


def test_keep_warm_worker_env_resolves_to_warm_worker(monkeypatch) -> None:
    monkeypatch.setenv("EDGETTS_ARENA_TEST_SET", "/fake/from/env")
    spec = _spec(worker_python_env="EDGETTS_ARENA_TEST_SET")
    assert spec.resolve_worker_python() != ""
    assert _ROUTE(spec, object(), _Res(True)) == "warm_worker"


def test_keep_warm_light_model_is_in_process() -> None:
    spec = _spec()
    assert spec.resolve_worker_python() == ""
    assert _ROUTE(spec, object(), _Res(True)) == "in_process"
