"""Model residency policy: warm cache, generational eviction, memory-aware admission.

This centralizes *which models stay loaded between runs* — previously hard-coded
in each service's ``finally`` block as "keep_in_memory stays, everything else is
unloaded immediately".

Two residency modes (see :class:`~edgetts_arena.core.config.ResidencySettings`):

``eager`` (default)
    Preserve the legacy behavior exactly. The manager is inert: services unload
    non-``keep_in_memory`` models after every run and never ask the manager for a
    warm worker.

``keep_warm``
    The last batch of models used by a run stays resident. A model is only evicted
    when a *later* run does not select it (generational eviction), or when the
    memory budget / available-memory floor forces an LRU eviction.

Residency is realized differently per model kind:

* ``in_process`` — models with no dedicated worker interpreter (piper / kokoro /
  dummy). Their adapter lives in the main process (``registry`` keeps it READY);
  eviction calls :meth:`ModelRegistry.unload`.
* ``worker`` — models that resolve a dedicated venv interpreter (qwen3 /
  cosyvoice / melotts). They are kept warm as a persistent
  :class:`~edgetts_arena.core.persistent_worker.PersistentExternalWorker`
  subprocess so their tens-of-seconds cold load is paid once, not per run.

Memory-aware admission (``memory_aware=True``) uses each model's measured
footprint (in-process RSS delta, or the worker's self-reported RSS) plus
:func:`effective_available_memory_bytes` to decide how many models may stay warm,
evicting least-recently-used residents to stay within budget.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from edgetts_arena.core.config import ResidencySettings
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec, ModelStatus
from edgetts_arena.core.persistent_worker import PersistentExternalWorker
from edgetts_arena.core.resource_guard import ResourceGuard, effective_available_memory_bytes

WorkerFactory = Callable[[ModelSpec], PersistentExternalWorker]


def _default_worker_factory(spec: ModelSpec) -> PersistentExternalWorker:
    return PersistentExternalWorker(
        python_executable=spec.resolve_worker_python(),
        model_id=spec.id,
        adapter=spec.adapter,
        model_path=spec.resolved_model_path or spec.model_path,
        num_threads=spec.num_threads,
    )


@dataclass(slots=True)
class WarmEntry:
    model_id: str
    kind: str  # "in_process" | "worker"
    footprint_mb: float = 0.0
    generation: int = 0
    worker: PersistentExternalWorker | None = None


@dataclass(frozen=True, slots=True)
class Admission:
    allowed: bool
    reason: str = ""
    evicted: tuple[str, ...] = ()


class ResidencyManager:
    """Cross-run warm cache with generational + memory-budget eviction."""

    def __init__(
        self,
        registry: ModelRegistry,
        settings: ResidencySettings,
        resource_guard: ResourceGuard | None = None,
        *,
        available_memory_provider: Callable[[], int] | None = None,
        worker_factory: WorkerFactory | None = None,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.resource_guard = resource_guard
        self._available_memory = available_memory_provider or effective_available_memory_bytes
        self._worker_factory = worker_factory or _default_worker_factory
        self._warm: "OrderedDict[str, WarmEntry]" = OrderedDict()
        self._footprint_hint: dict[str, float] = {}
        self._model_locks: dict[str, threading.Lock] = {}
        self._generation = 0
        self._lock = threading.RLock()
        self._eviction_log: list[str] = []

    # ------------------------------------------------------------------ properties
    @property
    def keep_warm(self) -> bool:
        return bool(self.settings.keep_warm)

    @property
    def memory_aware(self) -> bool:
        return bool(self.settings.memory_aware)

    def residency_kind(self, spec: ModelSpec) -> str:
        """``worker`` when a dedicated interpreter resolves, else ``in_process``."""
        return "worker" if spec.resolve_worker_python() else "in_process"

    def available_memory_mb(self) -> float:
        return self._available_memory() / (1024 * 1024)

    def _memory_floor_mb(self) -> float:
        if self.resource_guard is not None:
            return float(self.resource_guard.settings.min_available_memory_mb_soft)
        return 512.0

    def configure(self, settings: ResidencySettings) -> dict[str, object]:
        """Apply a new residency policy at runtime (driven by the UI/API control).

        Switching to ``eager`` releases every warm resident, since eager mode never
        keeps models loaded between runs. Tightening the budget while keep-warm is
        active trims LRU residents down to the new budget. Returns the new snapshot.
        """
        with self._lock:
            was_keep_warm = self.settings.keep_warm
            self.settings = settings
            if was_keep_warm and not settings.keep_warm:
                for model_id in list(self._warm):
                    self._evict_locked(model_id, reason="mode_eager")
            elif settings.keep_warm and settings.memory_aware:
                self._trim_to_budget_locked(protect_newest=True)
            return self.snapshot()

    # ------------------------------------------------------------------ run lifecycle
    def begin_run(self, selected_ids: list[str]) -> list[str]:
        """Generational eviction: drop warm models not selected by this run.

        Called once at the start of a run. Returns the evicted model ids.
        """
        if not self.keep_warm:
            return []
        selected = set(selected_ids)
        with self._lock:
            self._generation += 1
            evicted: list[str] = []
            for model_id in list(self._warm):
                if model_id not in selected:
                    self._evict_locked(model_id, reason="not_selected")
                    evicted.append(model_id)
            return evicted

    def end_run(self, selected_ids: list[str]) -> list[str]:
        """Trim residents beyond the memory budget (LRU), protecting the newest.

        Called once after a run completes. Returns the evicted model ids.
        """
        if not self.keep_warm or not self.memory_aware:
            return []
        with self._lock:
            return self._trim_to_budget_locked(protect_newest=True)

    # ------------------------------------------------------------------ worker cache
    def acquire_worker(self, spec: ModelSpec, *, timeout_sec: float) -> PersistentExternalWorker:
        """Return a live warm worker for ``spec``, starting one if needed.

        Raises :class:`WarmWorkerError` (from the worker) if it cannot be started;
        the caller should then fall back to a one-shot run.
        """
        with self._lock:
            entry = self._warm.get(spec.id)
            if entry is not None and entry.kind == "worker" and entry.worker is not None:
                if entry.worker.is_alive():
                    self._warm.move_to_end(spec.id)
                    entry.generation = self._generation
                    return entry.worker
                self._evict_locked(spec.id, reason="dead_worker")
            model_lock = self._model_locks.setdefault(spec.id, threading.Lock())

        # Start outside the global lock so different models can warm in parallel.
        with model_lock:
            with self._lock:
                entry = self._warm.get(spec.id)
                if entry is not None and entry.kind == "worker" and entry.worker is not None and entry.worker.is_alive():
                    self._warm.move_to_end(spec.id)
                    return entry.worker
                # A stale in-process resident (e.g. loaded by the WS streaming path)
                # must be released before this model becomes a dedicated worker,
                # otherwise its adapter would leak untracked in the main process.
                if entry is not None and entry.kind != "worker":
                    self._evict_locked(spec.id, reason="promote_to_worker")
                estimate = self._footprint_hint.get(spec.id, 0.0)
                self._make_room_locked(estimate_mb=estimate, incoming=spec.id)

            worker = self._worker_factory(spec)
            worker.start(timeout_sec=timeout_sec)

            with self._lock:
                footprint = worker.rss_mb() or 0.0
                self._footprint_hint[spec.id] = footprint
                self._warm[spec.id] = WarmEntry(
                    model_id=spec.id,
                    kind="worker",
                    footprint_mb=footprint,
                    generation=self._generation,
                    worker=worker,
                )
                self._warm.move_to_end(spec.id)
            return worker

    def worker_for(self, model_id: str) -> PersistentExternalWorker | None:
        with self._lock:
            entry = self._warm.get(model_id)
            if entry is not None and entry.kind == "worker" and entry.worker is not None and entry.worker.is_alive():
                return entry.worker
            return None

    def mark_worker_used(self, model_id: str, footprint_mb: float | None = None) -> None:
        if not self.keep_warm:
            return
        with self._lock:
            entry = self._warm.get(model_id)
            if entry is None:
                return
            if footprint_mb is not None and footprint_mb >= 0:
                entry.footprint_mb = float(footprint_mb)
                self._footprint_hint[model_id] = float(footprint_mb)
            entry.generation = self._generation
            self._warm.move_to_end(model_id)

    def mark_in_process(self, model_id: str, footprint_mb: float) -> None:
        """Record an in-process resident model (its adapter stays in the registry)."""
        if not self.keep_warm:
            return
        with self._lock:
            entry = self._warm.get(model_id)
            if entry is None:
                entry = WarmEntry(model_id=model_id, kind="in_process")
                self._warm[model_id] = entry
            elif entry.kind == "worker" and entry.worker is not None:
                # Was a dedicated worker; shut it down before tracking the model as
                # in-process so the subprocess is not orphaned.
                try:
                    entry.worker.shutdown()
                except Exception:
                    pass
            entry.kind = "in_process"
            entry.worker = None
            entry.footprint_mb = max(0.0, float(footprint_mb or 0.0))
            entry.generation = self._generation
            self._footprint_hint[model_id] = entry.footprint_mb
            self._warm.move_to_end(model_id)

    # ------------------------------------------------------------------ admission
    def admit_in_process(self, spec: ModelSpec, *, footprint_mb: float | None = None) -> Admission:
        """Memory gate before *keeping* an in-process model resident.

        Never blocks the run itself — the model is loaded and used regardless — but
        decides whether it may stay warm afterwards. When memory is tight, evicts
        LRU residents to make room; if still over budget, reports not-allowed so the
        caller leaves the model unloaded after the run.

        ``footprint_mb`` is the just-measured RSS delta for this run; when given it
        refreshes the estimate so the very first keep-warm decision is accurate.
        """
        if not self.keep_warm or not self.memory_aware:
            return Admission(True)
        if footprint_mb is not None and footprint_mb >= 0:
            self._footprint_hint[spec.id] = float(footprint_mb)
        estimate = self._footprint_hint.get(spec.id, 0.0)
        with self._lock:
            evicted = self._make_room_locked(estimate_mb=estimate, incoming=spec.id)
            allowed, reason = self._fits_locked(estimate_mb=estimate, incoming=spec.id)
            return Admission(allowed, reason, tuple(evicted))

    # ------------------------------------------------------------------ eviction
    def evict(self, model_id: str) -> None:
        with self._lock:
            self._evict_locked(model_id, reason="manual")

    def evict_all(self) -> None:
        """Shut down every resident (used on app shutdown to avoid orphan workers)."""
        with self._lock:
            for model_id in list(self._warm):
                self._evict_locked(model_id, reason="shutdown")

    def _evict_locked(self, model_id: str, *, reason: str) -> None:
        entry = self._warm.pop(model_id, None)
        if entry is None:
            return
        if entry.kind == "worker" and entry.worker is not None:
            try:
                entry.worker.shutdown()
            except Exception:
                pass
        else:
            try:
                self.registry.unload(model_id)
            except Exception:
                try:
                    self.registry.set_status(model_id, ModelStatus.ERROR, error="eviction failed")
                except Exception:
                    pass
        self._log_eviction(f"{model_id}:{reason}")

    def _log_eviction(self, note: str) -> None:
        self._eviction_log.append(note)
        if len(self._eviction_log) > 20:
            del self._eviction_log[:-20]

    def _resident_footprint_locked(self) -> float:
        return sum(entry.footprint_mb for entry in self._warm.values())

    def _make_room_locked(self, *, estimate_mb: float, incoming: str) -> list[str]:
        """Evict LRU residents until ``incoming`` fits the budget/memory floor.

        ``incoming`` is never evicted here (it is not yet resident). May evict
        members of the current run's batch if their combined footprint cannot fit —
        that is the memory-aware cap on how many models stay warm.
        """
        if not self.memory_aware:
            return []
        evicted: list[str] = []
        while self._warm:
            fits, _ = self._fits_locked(estimate_mb=estimate_mb, incoming=incoming)
            if fits:
                break
            oldest_id = next(iter(self._warm))
            if oldest_id == incoming:
                self._warm.move_to_end(oldest_id)
                if len(self._warm) == 1:
                    break
                oldest_id = next(iter(self._warm))
            self._evict_locked(oldest_id, reason="memory_budget")
            evicted.append(oldest_id)
        return evicted

    def _fits_locked(self, *, estimate_mb: float, incoming: str | None = None) -> tuple[bool, str]:
        budget = float(self.settings.resident_memory_budget_mb)
        # Exclude ``incoming`` from the resident sum so re-admitting an already-warm
        # model does not double-count its own footprint against the estimate.
        if incoming is not None:
            base = sum(
                entry.footprint_mb for mid, entry in self._warm.items() if mid != incoming
            )
        else:
            base = self._resident_footprint_locked()
        projected = base + max(0.0, estimate_mb)
        if projected > budget:
            return False, (
                f"resident footprint {projected:.0f} MB would exceed budget {budget:.0f} MB"
            )
        floor = self._memory_floor_mb()
        available = self.available_memory_mb()
        if available - max(0.0, estimate_mb) < floor:
            return False, (
                f"available memory {available:.0f} MB minus {estimate_mb:.0f} MB "
                f"is below floor {floor:.0f} MB"
            )
        return True, ""

    def _trim_to_budget_locked(self, *, protect_newest: bool) -> list[str]:
        evicted: list[str] = []
        while self._resident_footprint_locked() > float(self.settings.resident_memory_budget_mb):
            if len(self._warm) <= (1 if protect_newest else 0):
                break
            oldest_id = next(iter(self._warm))
            self._evict_locked(oldest_id, reason="memory_budget")
            evicted.append(oldest_id)
        return evicted

    # ------------------------------------------------------------------ introspection
    def snapshot(self) -> dict[str, object]:
        with self._lock:
            residents = [
                {
                    "model_id": entry.model_id,
                    "kind": entry.kind,
                    "footprint_mb": round(entry.footprint_mb, 1),
                    "generation": entry.generation,
                }
                for entry in self._warm.values()
            ]
            return {
                "mode": self.settings.mode,
                "memory_aware": self.memory_aware,
                "resident_memory_budget_mb": self.settings.resident_memory_budget_mb,
                "resident_footprint_mb": round(self._resident_footprint_locked(), 1),
                "available_memory_mb": round(self.available_memory_mb(), 1),
                "residents": residents,
                "recent_evictions": list(self._eviction_log),
            }
