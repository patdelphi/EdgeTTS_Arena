from __future__ import annotations

from pathlib import Path

from edgetts_arena.core.benchmark_suite import BenchmarkPresetSuite
from edgetts_arena.core.config import load_settings
from edgetts_arena.core.model_registry import ModelRegistry
from edgetts_arena.defaults import read_default_text


def test_packaged_defaults_match_repository_configs() -> None:
    assert read_default_text("app_config.yaml") == Path("config/app_config.yaml").read_text(encoding="utf-8")
    assert read_default_text("models_config.yaml") == Path("config/models_config.yaml").read_text(encoding="utf-8")
    assert read_default_text("benchmark_presets.json") == Path("config/benchmark_presets.json").read_text(encoding="utf-8")


def test_default_configuration_falls_back_to_package_resources(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    settings = load_settings()
    registry = ModelRegistry.from_yaml()
    suite = BenchmarkPresetSuite.load()

    assert settings.port == 8000
    assert settings.inference_timeout_sec == 60
    assert "dummy" in registry.ids()
    assert registry.get_record("dummy").spec.enabled is True
    assert suite.version == "1.0"
    assert [case.id for case in suite.cases] == ["TC-01", "TC-02", "TC-03", "TC-04", "TC-05"]


def test_custom_missing_registry_and_preset_paths_still_fail(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    try:
        ModelRegistry.from_yaml(missing)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("custom missing models config should fail")

    missing_json = tmp_path / "missing.json"
    try:
        BenchmarkPresetSuite.load(missing_json)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("custom missing benchmark preset path should fail")
