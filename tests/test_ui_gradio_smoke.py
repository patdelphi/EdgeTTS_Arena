import pytest


gr = pytest.importorskip("gradio")

from fastapi.testclient import TestClient

from edgetts_arena.api.app import create_app
from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.config import AppSettings, ResourceGuardSettings
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.ui.gradio_app import build_arena_ui


def _runtime(tmp_path):
    registry = ModelRegistry(
        [
            ModelSpec(
                id="dummy",
                name="Dummy",
                adapter="dummy",
                enabled=True,
                keep_in_memory=True,
                num_threads=1,
            )
        ],
        adapter_factories={"dummy": DummyTTSAdapter},
    )
    store = RunArtifactStore(tmp_path / "exports")
    guard = ResourceGuard(
        ResourceGuardSettings(
            min_available_memory_mb_soft=1,
            min_available_memory_mb_hard=1,
        )
    )
    service = BenchmarkService(registry, guard, store)
    return registry, store, service


def test_gradio_blocks_build_without_starting_server(tmp_path) -> None:
    registry, store, service = _runtime(tmp_path)
    demo = build_arena_ui(registry, service, store)
    assert isinstance(demo, gr.Blocks)
    assert demo.title == "EdgeTTS-Arena"
    assert demo.get_config_file()["components"]


def test_gradio_blocks_mount_on_stage3_fastapi(tmp_path) -> None:
    registry, store, service = _runtime(tmp_path)
    settings = AppSettings(
        resource_guard=ResourceGuardSettings(
            min_available_memory_mb_soft=1,
            min_available_memory_mb_hard=1,
        )
    )
    app = create_app(settings=settings, registry=registry, exports_root=store.root)
    demo = build_arena_ui(registry, app.state.benchmark_service, app.state.artifact_store)
    mounted = gr.mount_gradio_app(
        app,
        demo,
        path="/arena",
        allowed_paths=[str(store.root)],
    )
    client = TestClient(mounted)
    assert client.get("/healthz").status_code == 200
    response = client.get("/arena/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
