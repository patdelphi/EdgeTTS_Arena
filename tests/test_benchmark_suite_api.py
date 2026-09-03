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
            )
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
    return TestClient(
        create_app(settings=settings, registry=registry, exports_root=tmp_path / "exports")
    )


def test_presets_endpoint_exposes_tc01_to_tc05(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.get("/api/v1/benchmark/presets")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["defaults"] == {"warmup_runs": 1, "measured_runs": 3}
    assert [case["id"] for case in data["cases"]] == ["TC-01", "TC-02", "TC-03", "TC-04", "TC-05"]


def test_suite_endpoint_returns_repeated_statistics_and_export(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/benchmark/suite",
        json={
            "models": ["dummy"],
            "case_ids": ["TC-01"],
            "warmup_runs": 1,
            "measured_runs": 3,
            "cpu_threads_per_model": 2,
            "config": {"seed": 42, "speed": 1.0},
        },
    )
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["results"][0]["aggregate"]["rtf"]["count"] == 3
    assert len(data["results"][0]["measurements"]) == 3

    exported = client.get(f"/api/v1/export/{data['run_id']}")
    assert exported.status_code == 200
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        names = set(archive.namelist())
    assert "audio/TC-01__dummy.wav" in names


def test_suite_endpoint_rejects_duplicate_case_ids(tmp_path) -> None:
    client = make_client(tmp_path)
    response = client.post(
        "/api/v1/benchmark/suite",
        json={"models": ["dummy"], "case_ids": ["TC-01", "TC-01"]},
    )
    assert response.status_code == 400
    assert response.json()["code"] == 1001
