from __future__ import annotations

import io
import zipfile

from fastapi.testclient import TestClient

from edgetts_arena.api import create_app
from edgetts_arena.adapters.dummy_adapter import DummyTTSAdapter
from edgetts_arena.core.config import AppSettings, ResourceGuardSettings
from edgetts_arena.core.model_registry import ModelRegistry, ModelSpec


def make_client(tmp_path) -> TestClient:
    registry = ModelRegistry(
        [
            ModelSpec(
                id="dummy",
                name="Dummy Adapter",
                adapter="dummy",
                enabled=True,
                keep_in_memory=True,
                num_threads=1,
            ),
            ModelSpec(
                id="nostream",
                name="No Stream",
                adapter="qwen3",
                enabled=True,
                experimental=True,
            ),
        ],
        adapter_factories={"dummy": DummyTTSAdapter},
    )
    settings = AppSettings(
        default_num_threads=2,
        resource_guard=ResourceGuardSettings(
            min_available_memory_mb_soft=1,
            min_available_memory_mb_hard=1,
        ),
    )
    app = create_app(
        settings=settings,
        registry=registry,
        exports_root=tmp_path / "exports",
    )
    return TestClient(app)


def test_models_endpoint_reports_environment_and_capabilities(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/v1/system/models")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == 200
    assert body["data"]["system_env"]["cpu_logical_cores"] >= 1
    assert body["data"]["system_env"]["cpu_effective_cores"] >= 1
    assert body["data"]["system_env"]["cpu_effective_cores"] <= body["data"]["system_env"]["cpu_logical_cores"]
    dummy = next(item for item in body["data"]["models"] if item["id"] == "dummy")
    assert dummy["capabilities"]["streaming"] is True
    assert dummy["default_voice"] == "default"


def test_benchmark_is_sync_and_model_failure_is_isolated(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/benchmark/run",
        json={
            "text": "EdgeTTS Arena API benchmark.",
            "models": ["dummy", "missing"],
            "execution_mode": "sequential",
            "cpu_threads_per_model": 2,
            "config": {"seed": 7, "speed": 1.0},
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["run_id"].startswith("run_")
    assert len(data["results"]) == 2

    success, failure = data["results"]
    assert success["model_id"] == "dummy"
    assert success["status"] == "success"
    assert success["metrics"]["ttfb_ms"] is None
    assert success["audio_url"].endswith("/dummy.wav")

    assert failure["model_id"] == "missing"
    assert failure["status"] == "error"
    assert failure["error"]["code"] == 1002


def test_benchmark_audio_download_and_export(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/benchmark/run",
        json={"text": "download test", "models": ["dummy"]},
    )
    data = response.json()["data"]
    run_id = data["run_id"]
    audio_url = data["results"][0]["audio_url"]

    audio = client.get(audio_url)
    assert audio.status_code == 200
    assert audio.headers["content-type"].startswith("audio/wav")
    assert audio.content.startswith(b"RIFF")

    export = client.get(f"/api/v1/export/{run_id}")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        names = set(archive.namelist())
    assert "audio/dummy.wav" in names
    assert "benchmark_report.json" in names
    assert "environment.json" in names


def test_validation_error_uses_application_envelope(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/benchmark/run",
        json={"text": "x", "models": ["dummy", "dummy"]},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] == 1001
    assert body["error"]["type"] == "validation_error"


def test_streaming_capability_gate_rejects_non_streaming_model(tmp_path) -> None:
    client = make_client(tmp_path)
    with client.websocket_connect("/api/v1/tts/stream?model=nostream") as websocket:
        message = websocket.receive_json()
        assert message["event"] == "error"
        assert message["code"] == 1003
        assert message["type"] == "capability_conflict"


def test_dummy_streaming_emits_first_chunk_and_complete(tmp_path) -> None:
    client = make_client(tmp_path)
    with client.websocket_connect("/api/v1/tts/stream?model=dummy") as websocket:
        websocket.send_json(
            {"action": "start", "text": "stream test", "speed": 1.0, "voice": "default"}
        )
        started = websocket.receive_json()
        assert started["event"] == "started"

        first = websocket.receive_json()
        assert first["event"] == "first_chunk"
        assert first["encoding"] == "pcm_s16le"
        assert websocket.receive_bytes()

        complete = None
        for _ in range(20):
            message = websocket.receive()
            if message.get("text"):
                import json

                payload = json.loads(message["text"])
                if payload.get("event") == "complete":
                    complete = payload
                    break
        assert complete is not None
        assert complete["chunks"] >= 1


def test_concurrent_benchmark_reports_pressure_execution_plan(tmp_path) -> None:
    registry = ModelRegistry(
        [
            ModelSpec(id="dummy-a", name="Dummy A", adapter="dummy", enabled=True, keep_in_memory=False),
            ModelSpec(id="dummy-b", name="Dummy B", adapter="dummy", enabled=True, keep_in_memory=False),
        ],
        adapter_factories={"dummy": DummyTTSAdapter},
    )
    settings = AppSettings(
        resource_guard=ResourceGuardSettings(
            min_available_memory_mb_soft=1,
            min_available_memory_mb_hard=1,
            min_available_memory_mb_per_concurrent_model=1,
            max_concurrent_models=4,
        )
    )
    client = TestClient(create_app(settings=settings, registry=registry, exports_root=tmp_path / "exports"))
    response = client.post(
        "/api/v1/benchmark/run",
        json={
            "text": "pressure profile",
            "models": ["dummy-a", "dummy-b"],
            "execution_mode": "concurrent",
            "cpu_threads_per_model": 64,
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["execution_profile"] == "pressure"
    assert data["requested_cpu_threads_per_model"] == 64
    assert data["total_threads_budget"] <= data["cpu_threads_per_model"] * 2
    assert data["resource_warnings"]
