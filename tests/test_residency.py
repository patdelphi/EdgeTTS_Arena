from __future__ import annotations

from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
from edgetts_arena.core.config import ResidencySettings, ResourceGuardSettings
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec, ModelStatus
from edgetts_arena.core.residency import ResidencyManager
from edgetts_arena.core.resource_guard import ResourceGuard

_MB = 1024 * 1024


class FakeWorker:
    """Stand-in for PersistentExternalWorker: no subprocess, records lifecycle."""

    instances: list["FakeWorker"] = []

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self.started = False
        self.shutdowns = 0
        self._alive = False
        self.rss = 40.0
        FakeWorker.instances.append(self)

    def start(self, *, timeout_sec: float) -> None:
        self.started = True
        self._alive = True

    def is_alive(self) -> bool:
        return self._alive

    def shutdown(self) -> None:
        self.shutdowns += 1
        self._alive = False

    def rss_mb(self) -> float | None:
        return self.rss if self._alive else None


def _registry(ids: list[str], *, worker_ids: set[str] | None = None) -> ModelRegistry:
    worker_ids = worker_ids or set()
    specs = [
        ModelSpec(
            id=model_id,
            name=model_id,
            adapter="dummy",
            enabled=True,
            keep_in_memory=False,
            num_threads=1,
            worker_python="/fake/worker/python" if model_id in worker_ids else "",
        )
        for model_id in ids
    ]
    return ModelRegistry(specs, adapter_factories={"dummy": DummyTTSAdapter})


def _manager(
    registry: ModelRegistry,
    *,
    mode: str = "keep_warm",
    memory_aware: bool = False,
    budget_mb: int = 4096,
    available_mb: float = 8192.0,
    soft_mb: int = 512,
) -> ResidencyManager:
    FakeWorker.instances.clear()
    guard = ResourceGuard(ResourceGuardSettings(min_available_memory_mb_soft=soft_mb))
    return ResidencyManager(
        registry,
        ResidencySettings(mode=mode, memory_aware=memory_aware, resident_memory_budget_mb=budget_mb),
        guard,
        available_memory_provider=lambda: int(available_mb * _MB),
        worker_factory=lambda spec: FakeWorker(spec),
    )


# --------------------------------------------------------------------------- mode
def test_eager_mode_is_inert() -> None:
    registry = _registry(["a", "b"])
    manager = _manager(registry, mode="eager")
    assert manager.keep_warm is False
    manager.mark_in_process("a", 10.0)
    assert manager.begin_run(["b"]) == []
    assert manager.end_run(["b"]) == []
    # nothing tracked while eager
    assert manager.snapshot()["residents"] == []


def test_residency_kind_routes_worker_vs_in_process() -> None:
    registry = _registry(["a", "w"], worker_ids={"w"})
    manager = _manager(registry)
    assert manager.residency_kind(registry.get_record("a").spec) == "in_process"
    assert manager.residency_kind(registry.get_record("w").spec) == "worker"


# ------------------------------------------------------- generational eviction (①)
def test_begin_run_evicts_warm_models_not_selected() -> None:
    registry = _registry(["a", "b", "c"])
    manager = _manager(registry)
    # load a/b/c in-process so eviction has a real adapter to unload
    for model_id in ("a", "b", "c"):
        registry.load(model_id, num_threads=1)
        manager.mark_in_process(model_id, 10.0)
        assert registry.get_record(model_id).status == ModelStatus.READY

    evicted = manager.begin_run(["a", "c"])  # next run drops b
    assert evicted == ["b"]
    assert registry.get_record("b").status == ModelStatus.UNLOADED
    assert registry.get_record("a").status == ModelStatus.READY
    assert registry.get_record("c").status == ModelStatus.READY
    assert {r["model_id"] for r in manager.snapshot()["residents"]} == {"a", "c"}


def test_worker_eviction_shuts_down_subprocess() -> None:
    registry = _registry(["w"], worker_ids={"w"})
    manager = _manager(registry)
    spec = registry.get_record("w").spec
    worker = manager.acquire_worker(spec, timeout_sec=5.0)
    assert worker.started and worker.is_alive()

    evicted = manager.begin_run([])  # not selected next run
    assert evicted == ["w"]
    assert worker.shutdowns == 1
    assert worker.is_alive() is False
    assert manager.worker_for("w") is None


def test_acquire_worker_reuses_live_worker() -> None:
    registry = _registry(["w"], worker_ids={"w"})
    manager = _manager(registry)
    spec = registry.get_record("w").spec
    first = manager.acquire_worker(spec, timeout_sec=5.0)
    second = manager.acquire_worker(spec, timeout_sec=5.0)
    assert first is second
    assert len(FakeWorker.instances) == 1  # started only once
    assert first.started


def test_acquire_worker_restarts_dead_worker() -> None:
    registry = _registry(["w"], worker_ids={"w"})
    manager = _manager(registry)
    spec = registry.get_record("w").spec
    first = manager.acquire_worker(spec, timeout_sec=5.0)
    first.shutdown()  # simulate crash/exit between runs
    second = manager.acquire_worker(spec, timeout_sec=5.0)
    assert second is not first
    assert len(FakeWorker.instances) == 2
    assert second.is_alive()


# --------------------------------------------------------- memory-aware budget (②)
def test_end_run_trims_lru_beyond_budget() -> None:
    registry = _registry(["a", "b", "c"])
    manager = _manager(registry, memory_aware=True, budget_mb=100)
    for model_id in ("a", "b", "c"):
        registry.load(model_id, num_threads=1)
        manager.mark_in_process(model_id, 60.0)  # 180 total > 100 budget

    evicted = manager.end_run(["a", "b", "c"])
    # LRU (a, then b) evicted until <= budget; newest (c) protected last
    assert evicted == ["a", "b"]
    assert registry.get_record("a").status == ModelStatus.UNLOADED
    assert registry.get_record("b").status == ModelStatus.UNLOADED
    assert registry.get_record("c").status == ModelStatus.READY
    assert manager.snapshot()["resident_footprint_mb"] == 60.0


def test_admit_in_process_denied_when_single_model_exceeds_budget() -> None:
    registry = _registry(["a"])
    manager = _manager(registry, memory_aware=True, budget_mb=50)
    manager.mark_in_process("a", 60.0)  # learn a 60MB footprint hint
    manager.evict("a")  # not resident now, hint retained

    admission = manager.admit_in_process(registry.get_record("a").spec)
    assert admission.allowed is False
    assert "budget" in admission.reason


def test_admit_in_process_evicts_lru_to_make_room() -> None:
    registry = _registry(["a", "b"])
    manager = _manager(registry, memory_aware=True, budget_mb=100)
    registry.load("a", num_threads=1)
    manager.mark_in_process("a", 60.0)
    manager._footprint_hint["b"] = 60.0  # b's footprint learned from a prior run

    # Admitting b (60MB) alongside resident a (60MB) = 120 > 100 -> evict LRU (a).
    admission = manager.admit_in_process(registry.get_record("b").spec)
    assert admission.allowed is True
    assert "a" in admission.evicted
    assert registry.get_record("a").status == ModelStatus.UNLOADED


def test_available_memory_floor_blocks_residency() -> None:
    registry = _registry(["a"])
    # Only 600MB available, soft floor 512, model needs ~200 -> below floor.
    manager = _manager(registry, memory_aware=True, budget_mb=4096, available_mb=600.0, soft_mb=512)
    manager.mark_in_process("a", 200.0)
    manager.evict("a")
    admission = manager.admit_in_process(registry.get_record("a").spec)
    assert admission.allowed is False
    assert "floor" in admission.reason


def test_memory_unaware_never_trims() -> None:
    registry = _registry(["a", "b", "c"])
    manager = _manager(registry, memory_aware=False, budget_mb=1, available_mb=1.0)
    for model_id in ("a", "b", "c"):
        registry.load(model_id, num_threads=1)
        manager.mark_in_process(model_id, 500.0)
    assert manager.end_run(["a", "b", "c"]) == []
    assert len(manager.snapshot()["residents"]) == 3


# ------------------------------------------------------------- introspection / all
def test_snapshot_reports_state() -> None:
    registry = _registry(["a", "w"], worker_ids={"w"})
    manager = _manager(registry, memory_aware=True, budget_mb=4096)
    registry.load("a", num_threads=1)
    manager.mark_in_process("a", 12.5)
    manager.acquire_worker(registry.get_record("w").spec, timeout_sec=5.0)

    snap = manager.snapshot()
    assert snap["mode"] == "keep_warm"
    assert snap["memory_aware"] is True
    assert snap["resident_memory_budget_mb"] == 4096
    kinds = {r["model_id"]: r["kind"] for r in snap["residents"]}
    assert kinds == {"a": "in_process", "w": "worker"}
    assert snap["available_memory_mb"] == 8192.0


def test_evict_all_shuts_down_everything() -> None:
    registry = _registry(["a", "w"], worker_ids={"w"})
    manager = _manager(registry)
    registry.load("a", num_threads=1)
    manager.mark_in_process("a", 10.0)
    worker = manager.acquire_worker(registry.get_record("w").spec, timeout_sec=5.0)

    manager.evict_all()
    assert manager.snapshot()["residents"] == []
    assert worker.shutdowns == 1
    assert registry.get_record("a").status == ModelStatus.UNLOADED
