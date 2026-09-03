from pathlib import Path

from edgetts_arena.core import ModelRegistry, ModelStatus


def test_registry_reads_config_and_lazy_loads_dummy() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    assert "dummy" in registry.ids()
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED
    assert registry.get_record("piper").status == ModelStatus.UNAVAILABLE

    adapter = registry.load("dummy")
    assert adapter.id == "dummy"
    assert registry.get_record("dummy").status == ModelStatus.READY

    registry.unload("dummy")
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED


def test_registry_list_exposes_capabilities() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    dummy = next(item for item in registry.list_models() if item["id"] == "dummy")
    assert dummy["status"] == "unloaded"
    assert dummy["capabilities"]["streaming"] is True
    assert "default" in dummy["voices"]
    assert dummy["worker_mode"] == "in_process"


def test_registry_keeps_qwen3_experimental_disabled() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    record = registry.get_record("qwen3-tts-0.6b")
    assert record.spec.experimental is True
    assert record.status == ModelStatus.UNAVAILABLE
    item = next(item for item in registry.list_models() if item["id"] == "qwen3-tts-0.6b")
    assert item["experimental"] is True
    assert item["status"] == "unavailable"
    assert item["capabilities"]["voice_clone"] is False


def test_registry_reads_dedicated_worker_python(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - id: isolated-dummy\n"
        "    name: Isolated Dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n"
        "    keep_in_memory: false\n"
        "    worker_python: /opt/arena-venvs/dummy/bin/python\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_yaml(config)
    record = registry.get_record("isolated-dummy")
    assert record.spec.worker_python == "/opt/arena-venvs/dummy/bin/python"
    assert record.status == ModelStatus.UNLOADED
    item = registry.model_info("isolated-dummy")
    assert item["worker_mode"] == "external"
    assert item["worker_python_configured"] is True


def test_registry_resolves_worker_python_from_environment(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - id: isolated-dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n"
        "    keep_in_memory: false\n"
        "    worker_python_env: EDGETTS_ARENA_TEST_WORKER_PYTHON\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("EDGETTS_ARENA_TEST_WORKER_PYTHON", "~/arena-worker/bin/python")
    registry = ModelRegistry.from_yaml(config)
    record = registry.get_record("isolated-dummy")
    assert record.spec.worker_python == "~/arena-worker/bin/python"
    assert record.spec.worker_python_env == "EDGETTS_ARENA_TEST_WORKER_PYTHON"
    assert record.spec.resolve_worker_python().endswith("arena-worker/bin/python")
    assert registry.model_info("isolated-dummy")["worker_mode"] == "external"
