from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def test_module_cli_smoke(tmp_path: Path) -> None:
    output = tmp_path / "cli-smoke.wav"
    result = subprocess.run(
        [sys.executable, "-m", "edgetts_arena", "dummy", "--text", "CLI smoke test", "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    assert result.returncode == 0, result.stderr
    assert output.exists()
    assert output.stat().st_size > 44


def test_doctor_cli_core_baseline(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "edgetts_arena", "doctor", "--exports-root", str(tmp_path / "exports")],
        check=False,
        capture_output=True,
        text=True,
        env=_cli_env(),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ready"] is True
    assert report["workers_requested"] is False
    assert report["worker_models_requested"] == []
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["dummy_synthesis"]["ok"] is True
    assert checks["fastapi_app"]["ok"] is True
    assert checks["exports_writable"]["ok"] is True


def test_doctor_cli_validates_external_worker_protocol(tmp_path: Path) -> None:
    env = _cli_env()
    env["EDGETTS_ARENA_QWEN3_PYTHON"] = sys.executable
    env["EDGETTS_ARENA_COSYVOICE_PYTHON"] = sys.executable
    env["EDGETTS_ARENA_MELOTTS_PYTHON"] = sys.executable
    result = subprocess.run(
        [sys.executable, "-m", "edgetts_arena", "doctor", "--workers", "--exports-root", str(tmp_path / "exports")],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["ready"] is True
    assert report["workers_requested"] is True
    assert report["worker_models_requested"] == []
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["worker:qwen3-tts-0.6b"]["ok"] is True
    assert checks["worker:cosyvoice-300m-sft"]["ok"] is True
    assert checks["worker:melotts-zh"]["ok"] is True


def test_doctor_cli_can_probe_one_worker_without_other_envs(tmp_path: Path) -> None:
    env = _cli_env()
    env.pop("EDGETTS_ARENA_COSYVOICE_PYTHON", None)
    env.pop("EDGETTS_ARENA_MELOTTS_PYTHON", None)
    env["EDGETTS_ARENA_QWEN3_PYTHON"] = sys.executable
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "edgetts_arena",
            "doctor",
            "--worker",
            "qwen3-tts-0.6b",
            "--exports-root",
            str(tmp_path / "exports"),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stdout
    report = json.loads(result.stdout)
    assert report["ready"] is True
    assert report["workers_requested"] is True
    assert report["worker_models_requested"] == ["qwen3-tts-0.6b"]
    worker_checks = [item for item in report["checks"] if item["name"].startswith("worker:")]
    assert [item["name"] for item in worker_checks] == ["worker:qwen3-tts-0.6b"]
    assert worker_checks[0]["ok"] is True


def test_doctor_parser_supports_ui_workers_and_worker_filter() -> None:
    from edgetts_arena.cli import build_parser

    args = build_parser().parse_args(
        [
            "doctor",
            "--ui",
            "--workers",
            "--worker",
            "qwen3-tts-0.6b",
            "--worker",
            "melotts-zh",
            "--exports-root",
            "doctor-out",
        ]
    )
    assert args.command == "doctor"
    assert args.ui is True
    assert args.workers is True
    assert args.worker_models == ["qwen3-tts-0.6b", "melotts-zh"]
    assert args.exports_root == Path("doctor-out")


def test_piper_cli_parser_requires_model() -> None:
    from edgetts_arena.cli import build_parser

    args = build_parser().parse_args(
        ["piper", "--model", "voice.onnx", "--text", "hello", "--speed", "1.25"]
    )
    assert args.command == "piper"
    assert args.model == Path("voice.onnx")
    assert args.speed == 1.25


def test_kokoro_cli_parser() -> None:
    from edgetts_arena.cli import build_parser

    args = build_parser().parse_args(
        ["kokoro", "--model", "kokoro-v1.0.int8.onnx", "--voice", "af_heart"]
    )
    assert args.command == "kokoro"
    assert args.model == Path("kokoro-v1.0.int8.onnx")
    assert args.voice == "af_heart"
