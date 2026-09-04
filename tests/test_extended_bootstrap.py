from __future__ import annotations

import json
from pathlib import Path

from scripts.bootstrap_extended_model import (
    COSY_MODEL_REVISION,
    MELO_MODEL_REVISION,
    QWEN_REVISION,
    build_plan,
    write_bootstrap_artifacts,
)
from scripts.prepare_cosyvoice_cpu_requirements import build_cpu_requirements
from scripts.prepare_cosyvoice_model import validate_snapshot as validate_cosyvoice_snapshot
from scripts.prepare_melotts_assets import validate_assets as validate_melotts_assets


def _commands(plan) -> list[list[str]]:
    return [list(step.command) for step in plan.steps]


def test_qwen_bootstrap_plan_is_pinned_and_explicit(tmp_path: Path) -> None:
    plan = build_plan(
        "qwen3",
        repo_root=tmp_path,
        bootstrap_python="python3.11",
        include_assets=True,
    )
    commands = _commands(plan)
    assert plan.recommended_python == "3.11"
    assert plan.worker_env_name == "EDGETTS_ARENA_QWEN3_PYTHON"
    assert any("torch==2.8.0" in command for command in commands)
    assert any("qwen-tts==0.1.1" in command for command in commands)
    assert any(QWEN_REVISION in command for command in commands)
    assert commands[-1][3:5] == ["doctor", "--worker"]
    assert "qwen3-tts-0.6b" in commands[-1]


def test_melotts_bootstrap_plan_reuses_gate_pins(tmp_path: Path) -> None:
    plan = build_plan(
        "melotts",
        repo_root=tmp_path,
        bootstrap_python="python3.10",
        include_assets=True,
    )
    commands = _commands(plan)
    assert plan.required_tools == ("git",)
    assert any("MeloTTS.git@209145371cff8fc3bd60d7be902ea69cbdb7965a" in item for command in commands for item in command)
    assert any(MELO_MODEL_REVISION in command for command in commands)
    assert any(command[-2:] == ["unidic", "download"] for command in commands)
    worker_python = Path(plan.environment["EDGETTS_ARENA_MELOTTS_PYTHON"])
    assert worker_python.parent.parent == tmp_path / ".venv-melotts"
    assert worker_python.name in {"python", "python.exe"}


def test_cosyvoice_bootstrap_plan_exports_runtime_paths(tmp_path: Path) -> None:
    plan = build_plan(
        "cosyvoice",
        repo_root=tmp_path,
        bootstrap_python="python3.10",
        include_assets=True,
    )
    commands = _commands(plan)
    assert plan.required_tools == ("git", "sox")
    assert len(plan.pythonpath_entries) == 2
    # 使用 normpath 确保跨平台路径比较
    from os.path import normpath
    actual = normpath(plan.environment["EDGETTS_ARENA_COSYVOICE_WETEXT_DIR"])
    expected = normpath("models/cosyvoice/wetext")
    assert actual.replace("\\", "/").endswith(expected.replace("\\", "/"))
    assert any(COSY_MODEL_REVISION in command for command in commands)
    clone = next(step for step in plan.steps if step.label == "clone CosyVoice source")
    # 跨平台路径比较
    assert clone.skip_if_exists and clone.skip_if_exists.replace("\\", "/").endswith("vendor/CosyVoice/.git")


def test_bootstrap_can_skip_heavy_assets(tmp_path: Path) -> None:
    plan = build_plan(
        "qwen3",
        repo_root=tmp_path,
        bootstrap_python="python3.11",
        include_assets=False,
    )
    assert all("prepare pinned Qwen3 snapshot" != step.label for step in plan.steps)
    assert plan.steps[-1].label == "worker doctor"


def test_bootstrap_writes_portable_environment_files(tmp_path: Path) -> None:
    plan = build_plan(
        "cosyvoice",
        repo_root=tmp_path,
        bootstrap_python="python3.10",
        include_assets=False,
        include_doctor=False,
    )
    artifacts = write_bootstrap_artifacts(plan, tmp_path / "exports")
    saved = json.loads(Path(artifacts["plan"]).read_text(encoding="utf-8"))
    assert saved["model"] == "cosyvoice"
    env_sh = Path(artifacts["env_sh"]).read_text(encoding="utf-8")
    env_ps1 = Path(artifacts["env_ps1"]).read_text(encoding="utf-8")
    assert "EDGETTS_ARENA_COSYVOICE_PYTHON" in env_sh
    assert "PYTHONPATH" in env_sh
    assert "$env:EDGETTS_ARENA_COSYVOICE_PYTHON" in env_ps1


def test_cosyvoice_cpu_requirements_filter_is_guarded(tmp_path: Path) -> None:
    source = tmp_path / "requirements.txt"
    target = tmp_path / "requirements.cpu.txt"
    source.write_text(
        "torch==2.3.1\n"
        "torchaudio==2.3.1\n"
        "onnxruntime-gpu==1.18.0\n"
        "tensorrt-cu12==10.0\n"
        "openai-whisper==20231117\n"
        "numpy==1.26.4\n",
        encoding="utf-8",
    )
    result = build_cpu_requirements(source, target)
    text = target.read_text(encoding="utf-8")
    assert "numpy==1.26.4" in text
    assert "torch==" not in text
    assert "onnxruntime-gpu==" not in text
    assert len(result["removed"]) == 5


def test_offline_asset_validators_do_not_need_network(tmp_path: Path) -> None:
    melo = tmp_path / "melo"
    melo.mkdir()
    (melo / "config.json").write_text("{}", encoding="utf-8")
    (melo / "checkpoint.pth").write_bytes(b"checkpoint")
    melo_result = validate_melotts_assets(melo)
    assert set(melo_result["files"]) == {"config.json", "checkpoint.pth"}

    cosy = tmp_path / "cosy"
    cosy.mkdir()
    (cosy / "cosyvoice.yaml").write_text("sample_rate: 22050\n", encoding="utf-8")
    cosy_result = validate_cosyvoice_snapshot(cosy)
    assert cosy_result["required_files"] == ["cosyvoice.yaml"]
