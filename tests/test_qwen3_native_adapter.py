from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
import soundfile as sf

from edgetts_arena.adapters.qwen3_native_adapter import Qwen3NativeTTSAdapter
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError


def _fixture(tmp_path: Path, *, quantization: str = "int8") -> tuple[Path, Path, Path]:
    runtime = tmp_path / "runtime"
    runtime.mkdir(parents=True)
    binary = runtime / "qwen_tts"
    binary.write_text("fake", encoding="utf-8")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"tts_model_type": "custom_voice", "tts_model_size": "0b6"}),
        encoding="utf-8",
    )
    manifest_dir = tmp_path / "manifest"
    manifest_dir.mkdir()
    manifest = manifest_dir / "model.json"
    manifest.write_text(
        json.dumps({
            "runtime": "qwen3-tts-c",
            "runtime_revision": "deadbeef",
            "binary": "../runtime/qwen_tts",
            "model_dir": "../model",
            "quantization": quantization,
            "default_voice": "Vivian",
            "default_language": "English",
        }),
        encoding="utf-8",
    )
    return manifest, binary, model_dir


def test_qwen3_native_int8_contract(tmp_path: Path) -> None:
    manifest, binary, model_dir = _fixture(tmp_path)
    calls: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(command))
        if command[-1] == "--caps":
            return subprocess.CompletedProcess(command, 0, stdout="AVX2 + FMA\n", stderr="")
        output = Path(command[command.index("-o") + 1])
        sf.write(output, np.linspace(-0.2, 0.2, 2400, dtype=np.float32), 24000)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    adapter = Qwen3NativeTTSAdapter(command_runner=runner)
    adapter.load_model(str(manifest), num_threads=3)
    output = adapter.infer("native qwen", voice="vivian", language="zh", seed=42)

    assert calls[0] == [str(binary.resolve()), "--caps"]
    synthesis = calls[1]
    assert synthesis[synthesis.index("-d") + 1] == str(model_dir.resolve())
    assert synthesis[synthesis.index("-s") + 1] == "Vivian"
    assert synthesis[synthesis.index("-l") + 1] == "Chinese"
    assert synthesis[synthesis.index("-j") + 1] == "3"
    assert "--int8" in synthesis
    assert synthesis[synthesis.index("--seed") + 1] == "42"
    assert output.sample_rate == 24000
    assert output.audio.size == 2400
    assert output.metadata["runtime"] == "gabriele-mastrapasqua/qwen3-tts"
    assert output.metadata["runtime_revision"] == "deadbeef"
    assert output.metadata["quantization"] == "int8"
    assert output.metadata["runtime_caps"] == "AVX2 + FMA"


def test_qwen3_native_int4_flag(tmp_path: Path) -> None:
    manifest, _, _ = _fixture(tmp_path, quantization="int4")
    commands: list[list[str]] = []

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        if command[-1] == "--caps":
            return subprocess.CompletedProcess(command, 0, stdout="caps", stderr="")
        output = Path(command[command.index("-o") + 1])
        sf.write(output, np.ones(1200, dtype=np.float32) * 0.05, 24000)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    adapter = Qwen3NativeTTSAdapter(command_runner=runner)
    adapter.load_model(str(manifest), num_threads=2)
    adapter.infer("x")
    assert "--int4" in commands[-1]
    assert "--int8" not in commands[-1]


def test_qwen3_native_rejects_bad_manifest_and_checkpoint(tmp_path: Path) -> None:
    manifest, _, _ = _fixture(tmp_path)
    bad = json.loads(manifest.read_text(encoding="utf-8"))
    bad["runtime"] = "other"
    manifest.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime"):
        Qwen3NativeTTSAdapter(command_runner=lambda cmd: subprocess.CompletedProcess(cmd, 0)).load_model(str(manifest))

    manifest, _, model_dir = _fixture(tmp_path / "second")
    (model_dir / "config.json").write_text(
        json.dumps({"tts_model_type": "base", "tts_model_size": "0b6"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="CustomVoice"):
        Qwen3NativeTTSAdapter(command_runner=lambda cmd: subprocess.CompletedProcess(cmd, 0)).load_model(str(manifest))


def test_qwen3_native_preflight_failure_is_model_unavailable(tmp_path: Path) -> None:
    manifest, _, _ = _fixture(tmp_path)
    adapter = Qwen3NativeTTSAdapter(
        command_runner=lambda command: subprocess.CompletedProcess(command, 2, stdout="", stderr="unsupported ISA")
    )
    with pytest.raises(ArenaError) as error:
        adapter.load_model(str(manifest))
    assert error.value.code == 1002
    assert error.value.error_type == "model_unavailable"


def test_qwen3_native_capability_and_load_guards(tmp_path: Path) -> None:
    adapter = Qwen3NativeTTSAdapter()
    with pytest.raises(ModelNotLoadedError):
        adapter.infer("hello")

    manifest, _, _ = _fixture(tmp_path)

    def runner(command: list[str]) -> subprocess.CompletedProcess[str]:
        if command[-1] == "--caps":
            return subprocess.CompletedProcess(command, 0, stdout="caps", stderr="")
        output = Path(command[command.index("-o") + 1])
        sf.write(output, np.ones(1200, dtype=np.float32) * 0.05, 24000)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    adapter = Qwen3NativeTTSAdapter(command_runner=runner)
    adapter.load_model(str(manifest))
    with pytest.raises(ArenaError):
        adapter.infer("hello", speed=1.1)
    with pytest.raises(ValueError, match="speaker"):
        adapter.infer("hello", voice="unknown")
    with pytest.raises(ValueError, match="language"):
        adapter.infer("hello", language="xx")
