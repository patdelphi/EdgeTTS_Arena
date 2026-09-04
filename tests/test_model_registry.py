from pathlib import Path

import pytest

from edgetts_arena.core import ModelRegistry, ModelStatus


def test_registry_reads_config_and_lazy_loads_dummy() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    assert "dummy" in registry.ids()
    assert registry.get_record("dummy").status == ModelStatus.UNLOADED
    # piper 现在已启用，状态为 UNLOADED（等待加载）
    assert registry.get_record("piper").status == ModelStatus.UNLOADED

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
    assert dummy["capabilities"]["language_control"] is False
    assert "default" in dummy["voices"]
    assert dummy["worker_mode"] == "in_process"


def test_registry_keeps_qwen_variants_experimental_disabled() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    expected_threads = {
        "qwen3-tts-0.6b": 4,
        "qwen3-tts-0.6b-native-int8": 2,
        "qwen3-tts-0.6b-native-int4": 4,
    }
    # The native INT8/INT4 variants need a compiled C runtime that cannot be
    # bootstrapped on native Windows, so they ship disabled (status UNAVAILABLE).
    disabled_variants = {"qwen3-tts-0.6b-native-int8", "qwen3-tts-0.6b-native-int4"}
    for model_id, threads in expected_threads.items():
        record = registry.get_record(model_id)
        assert record.spec.experimental is True
        assert record.spec.language_control is True
        assert record.spec.num_threads == threads
        expected_status = (
            ModelStatus.UNAVAILABLE if model_id in disabled_variants else ModelStatus.UNLOADED
        )
        assert record.status == expected_status
        item = registry.model_info(model_id)
        assert item["experimental"] is True
        assert item["capabilities"]["language_control"] is True


def test_registry_applies_per_model_inference_timeout() -> None:
    registry = ModelRegistry.from_yaml("config/models_config.yaml")
    # Qwen3-TTS 0.6B is slow on CPU for long text, so it ships a higher hard-timeout
    # ceiling; every other model inherits the global default (spec value is None).
    assert registry.get_record("qwen3-tts-0.6b").spec.inference_timeout_sec == 1800.0
    # CosyVoice 300M SFT is also slow on CPU and ships a raised ceiling.
    assert registry.get_record("cosyvoice-300m-sft").spec.inference_timeout_sec == 900.0
    assert registry.get_record("dummy").spec.inference_timeout_sec is None
    assert registry.get_record("piper").spec.inference_timeout_sec is None
    # The two slow models also scale the budget with text length (base + per-char),
    # capped by the ceiling above; fast models leave both unset.
    qwen3 = registry.get_record("qwen3-tts-0.6b").spec
    assert (qwen3.timeout_base_sec, qwen3.timeout_per_char_sec) == (120.0, 3.0)
    cosy = registry.get_record("cosyvoice-300m-sft").spec
    assert (cosy.timeout_base_sec, cosy.timeout_per_char_sec) == (90.0, 1.5)
    assert registry.get_record("piper").spec.timeout_per_char_sec is None


def test_registry_parses_and_validates_inference_timeout(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - id: slow-dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n"
        "    inference_timeout_sec: 1800\n"
        "  - id: fast-dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_yaml(config)
    assert registry.get_record("slow-dummy").spec.inference_timeout_sec == 1800.0
    assert registry.get_record("fast-dummy").spec.inference_timeout_sec is None

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "models:\n"
        "  - id: bad-dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n"
        "    inference_timeout_sec: 0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inference_timeout_sec must be > 0"):
        ModelRegistry.from_yaml(bad)


def test_registry_parses_and_validates_text_scaled_timeout(tmp_path: Path) -> None:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - id: scaled-dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n"
        "    inference_timeout_sec: 1800\n"
        "    timeout_base_sec: 120\n"
        "    timeout_per_char_sec: 3.0\n"
        "  - id: plain-dummy\n"
        "    adapter: dummy\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    registry = ModelRegistry.from_yaml(config)
    spec = registry.get_record("scaled-dummy").spec
    assert (spec.timeout_base_sec, spec.timeout_per_char_sec) == (120.0, 3.0)
    plain = registry.get_record("plain-dummy").spec
    assert plain.timeout_base_sec is None and plain.timeout_per_char_sec is None

    for field, value, message in (
        ("timeout_base_sec", -1, "timeout_base_sec must be >= 0"),
        ("timeout_per_char_sec", 0, "timeout_per_char_sec must be > 0"),
    ):
        bad = tmp_path / f"bad_{field}.yaml"
        bad.write_text(
            "models:\n"
            "  - id: bad-dummy\n"
            "    adapter: dummy\n"
            "    enabled: true\n"
            f"    {field}: {value}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=message):
            ModelRegistry.from_yaml(bad)


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


def _worker_env_config(tmp_path: Path, env_var: str) -> Path:
    config = tmp_path / "models.yaml"
    config.write_text(
        "models:\n"
        "  - id: qwen3-tts-0.6b\n"
        "    name: Qwen3-TTS 0.6B CustomVoice\n"
        "    adapter: dummy\n"
        "    enabled: true\n"
        "    keep_in_memory: false\n"
        f"    worker_python_env: {env_var}\n",
        encoding="utf-8",
    )
    return config


def test_collect_worker_env_warnings_flags_unset_env(tmp_path: Path, monkeypatch) -> None:
    from edgetts_arena.core.model_registry import collect_worker_env_warnings

    monkeypatch.delenv("EDGETTS_ARENA_QWEN3_PYTHON", raising=False)
    registry = ModelRegistry.from_yaml(_worker_env_config(tmp_path, "EDGETTS_ARENA_QWEN3_PYTHON"))
    warnings = collect_worker_env_warnings(registry, project_root=tmp_path)
    assert len(warnings) == 1
    assert "EDGETTS_ARENA_QWEN3_PYTHON" in warnings[0]
    assert "qwen3-tts-0.6b" in warnings[0]
    # No generated bootstrap script yet -> fall back to the generic setup hint.
    assert "bootstrap_extended_model.py" in warnings[0]


def test_collect_worker_env_warnings_points_to_bootstrap_script(tmp_path: Path, monkeypatch) -> None:
    from edgetts_arena.core.model_registry import collect_worker_env_warnings

    monkeypatch.delenv("EDGETTS_ARENA_QWEN3_PYTHON", raising=False)
    script_dir = tmp_path / "exports" / "bootstrap" / "qwen3"
    script_dir.mkdir(parents=True)
    (script_dir / "env.ps1").write_text("# env\n", encoding="utf-8")
    (script_dir / "env.sh").write_text("# env\n", encoding="utf-8")
    registry = ModelRegistry.from_yaml(_worker_env_config(tmp_path, "EDGETTS_ARENA_QWEN3_PYTHON"))
    warnings = collect_worker_env_warnings(registry, project_root=tmp_path)
    assert len(warnings) == 1
    assert "exports/bootstrap/qwen3/env.ps1" in warnings[0]
    assert "exports/bootstrap/qwen3/env.sh" in warnings[0]


def test_collect_worker_env_warnings_silent_when_configured(tmp_path: Path, monkeypatch) -> None:
    from edgetts_arena.core.model_registry import collect_worker_env_warnings

    monkeypatch.setenv("EDGETTS_ARENA_QWEN3_PYTHON", str(tmp_path / "worker-python"))
    registry = ModelRegistry.from_yaml(_worker_env_config(tmp_path, "EDGETTS_ARENA_QWEN3_PYTHON"))
    assert collect_worker_env_warnings(registry, project_root=tmp_path) == []
