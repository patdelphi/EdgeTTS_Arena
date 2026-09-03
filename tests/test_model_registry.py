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


def test_registry_keeps_qwen3_experimental_disabled() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    record = registry.get_record("qwen3-tts-0.6b")
    assert record.spec.experimental is True
    assert record.status == ModelStatus.UNAVAILABLE
    item = next(item for item in registry.list_models() if item["id"] == "qwen3-tts-0.6b")
    assert item["experimental"] is True
    assert item["status"] == "unavailable"
    assert item["capabilities"]["voice_clone"] is False
