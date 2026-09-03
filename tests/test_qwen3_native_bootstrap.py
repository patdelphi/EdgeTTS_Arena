from __future__ import annotations

from pathlib import Path

import pytest

from scripts.bootstrap_qwen3_native import (
    MODEL_REVISION,
    RUNTIME_REVISION,
    build_plan,
    execute_plan,
    write_plan,
)


def _commands(plan) -> list[list[str]]:
    return [list(step.command) for step in plan.steps]


def test_native_bootstrap_plan_pins_runtime_model_and_variants(tmp_path: Path) -> None:
    plan = build_plan(repo_root=tmp_path, bootstrap_python="python3.11")
    commands = _commands(plan)
    assert plan.runtime_revision == RUNTIME_REVISION
    assert plan.model_revision == MODEL_REVISION
    assert plan.required_tools == ("git", "make", "gcc")
    assert any(command[-3:] == ["-C", str(tmp_path / "runtime" / "qwen3-tts-c"), "blas"] for command in commands)
    assert any(command[-1:] == ["--caps"] for command in commands)
    assert any(command[-1:] == ["--self-test"] for command in commands)
    assert any(MODEL_REVISION in command for command in commands)
    variants = next(step for step in plan.steps if step.label == "create matched INT8/INT4 manifests")
    assert RUNTIME_REVISION in variants.command
    assert str(tmp_path / "models" / "qwen3-native") in variants.command
    assert plan.steps[-1].label == "Arena native adapter preflight"


def test_native_bootstrap_can_reuse_existing_model_assets(tmp_path: Path) -> None:
    plan = build_plan(
        repo_root=tmp_path,
        bootstrap_python="python3.11",
        include_model_assets=False,
    )
    assert all(step.label != "prepare pinned official Qwen3 model" for step in plan.steps)
    assert any(step.label == "create matched INT8/INT4 manifests" for step in plan.steps)


def test_native_bootstrap_clone_is_idempotent_by_source_marker(tmp_path: Path) -> None:
    plan = build_plan(repo_root=tmp_path, bootstrap_python="python3.11")
    clone = next(step for step in plan.steps if step.label == "clone pinned native runtime source")
    assert clone.skip_if_exists == str(tmp_path / "runtime" / "qwen3-tts-c" / ".git")


def test_native_bootstrap_writes_auditable_plan(tmp_path: Path) -> None:
    plan = build_plan(repo_root=tmp_path, bootstrap_python="python3.11")
    path = write_plan(plan, tmp_path / "exports" / "bootstrap_plan.json")
    text = path.read_text(encoding="utf-8")
    assert RUNTIME_REVISION in text
    assert MODEL_REVISION in text
    assert "build native BLAS runtime" in text


def test_native_bootstrap_rejects_windows_execution(tmp_path: Path, monkeypatch) -> None:
    import scripts.bootstrap_qwen3_native as module

    plan = build_plan(repo_root=tmp_path, bootstrap_python="python3.11", include_model_assets=False)
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="not a supported native Windows baseline"):
        execute_plan(plan)
