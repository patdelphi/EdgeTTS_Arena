from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

QWEN_REVISION = "f3d1af06e4eaefac12b1ffa6726f9eef674a6f02"
MELO_SOURCE_REVISION = "209145371cff8fc3bd60d7be902ea69cbdb7965a"
MELO_MODEL_REVISION = "082ca057e44f1e52ec47e1622a30286019e8a3ef"
COSY_SOURCE_REVISION = "074ca6dc9e80a2f424f1f74b48bdd7d3fea531cc"
COSY_MODEL_REVISION = "fbb71de2afe387ed854eebd80b9f3d078c6b9869"


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str] | None = None
    skip_if_exists: str | None = None


@dataclass(frozen=True)
class BootstrapPlan:
    model: str
    model_id: str
    recommended_python: str
    bootstrap_python: str
    venv: str
    worker_python: str
    worker_env_name: str
    repo_root: str
    required_tools: tuple[str, ...]
    requires_network: bool
    steps: tuple[Step, ...]
    environment: dict[str, str]
    pythonpath_entries: tuple[str, ...]


def _venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _python_version_check(python: str, major: int, minor: int) -> tuple[str, ...]:
    code = (
        "import sys; expected=(%d,%d); actual=sys.version_info[:2]; "
        "assert actual==expected, f'expected Python {expected[0]}.{expected[1]}, got {actual[0]}.{actual[1]}'"
    ) % (major, minor)
    return (python, "-c", code)


def _base_steps(repo_root: Path, venv: Path, bootstrap_python: str, version: tuple[int, int]) -> list[Step]:
    worker_python = _venv_python(venv)
    return [
        Step("check bootstrap Python", _python_version_check(bootstrap_python, *version), str(repo_root)),
        Step("create dedicated venv", (bootstrap_python, "-m", "venv", str(venv)), str(repo_root)),
        Step(
            "upgrade pip",
            (str(worker_python), "-m", "pip", "install", "--upgrade", "pip"),
            str(repo_root),
        ),
        Step(
            "install EdgeTTS-Arena source",
            (str(worker_python), "-m", "pip", "install", "-e", str(repo_root)),
            str(repo_root),
        ),
    ]


def _doctor_step(
    repo_root: Path,
    worker_python: Path,
    model_id: str,
    env: dict[str, str],
) -> Step:
    return Step(
        "worker doctor",
        (
            str(worker_python),
            "-m",
            "edgetts_arena.cli",
            "doctor",
            "--worker",
            model_id,
            "--exports-root",
            str(repo_root / "exports" / "bootstrap-doctor" / model_id),
        ),
        str(repo_root),
        env=env,
    )


def build_plan(
    model: str,
    *,
    repo_root: Path,
    bootstrap_python: str,
    venv: Path | None = None,
    include_assets: bool = True,
    include_doctor: bool = True,
) -> BootstrapPlan:
    repo_root = repo_root.expanduser().resolve()
    if model not in {"qwen3", "melotts", "cosyvoice"}:
        raise ValueError(f"unsupported bootstrap model: {model}")

    default_venv = repo_root / f".venv-{model}"
    venv = (venv or default_venv).expanduser().resolve()
    worker_python = _venv_python(venv)
    steps: list[Step]
    env: dict[str, str]
    pythonpath_entries: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()

    if model == "qwen3":
        model_id = "qwen3-tts-0.6b"
        worker_env = "EDGETTS_ARENA_QWEN3_PYTHON"
        steps = _base_steps(repo_root, venv, bootstrap_python, (3, 11))
        steps.extend(
            [
                Step(
                    "install CPU PyTorch",
                    (
                        str(worker_python), "-m", "pip", "install", "--index-url",
                        "https://download.pytorch.org/whl/cpu", "torch==2.8.0", "torchaudio==2.8.0",
                    ),
                    str(repo_root),
                ),
                Step(
                    "install official Qwen3-TTS runtime",
                    (str(worker_python), "-m", "pip", "install", "qwen-tts==0.1.1"),
                    str(repo_root),
                ),
                Step(
                    "Qwen3 runtime preflight",
                    (
                        str(worker_python), "-c",
                        "import torch, qwen_tts; assert not torch.cuda.is_available(); print(torch.__version__, qwen_tts.__file__)",
                    ),
                    str(repo_root),
                ),
            ]
        )
        if include_assets:
            steps.append(
                Step(
                    "prepare pinned Qwen3 snapshot",
                    (
                        str(worker_python), str(repo_root / "scripts" / "prepare_qwen3_model.py"),
                        "--revision", QWEN_REVISION,
                        "--output", str(repo_root / "models" / "qwen3" / "Qwen3-TTS-12Hz-0.6B-CustomVoice"),
                    ),
                    str(repo_root),
                )
            )
        env = {worker_env: str(worker_python)}
        recommended = "3.11"

    elif model == "melotts":
        model_id = "melotts-zh"
        worker_env = "EDGETTS_ARENA_MELOTTS_PYTHON"
        required_tools = ("git",)
        steps = _base_steps(repo_root, venv, bootstrap_python, (3, 10))
        steps.extend(
            [
                Step(
                    "install CPU PyTorch",
                    (
                        str(worker_python), "-m", "pip", "install", "--index-url",
                        "https://download.pytorch.org/whl/cpu", "torch==2.8.0", "torchaudio==2.8.0",
                    ),
                    str(repo_root),
                ),
                Step(
                    "install pinned MeloTTS source",
                    (
                        str(worker_python), "-m", "pip", "install",
                        f"git+https://github.com/myshell-ai/MeloTTS.git@{MELO_SOURCE_REVISION}",
                    ),
                    str(repo_root),
                ),
                Step("prepare UniDic dictionary", (str(worker_python), "-m", "unidic", "download"), str(repo_root)),
                Step(
                    "MeloTTS runtime preflight",
                    (
                        str(worker_python), "-c",
                        "import MeCab; from melo.api import TTS; assert MeCab.Tagger().parse('テスト'); print(TTS)",
                    ),
                    str(repo_root),
                ),
            ]
        )
        if include_assets:
            steps.append(
                Step(
                    "prepare pinned MeloTTS assets",
                    (
                        str(worker_python), str(repo_root / "scripts" / "prepare_melotts_assets.py"),
                        "--revision", MELO_MODEL_REVISION,
                        "--output", str(repo_root / "models" / "melotts" / "zh"),
                    ),
                    str(repo_root),
                )
            )
        env = {worker_env: str(worker_python)}
        recommended = "3.10"

    else:
        model_id = "cosyvoice-300m-sft"
        worker_env = "EDGETTS_ARENA_COSYVOICE_PYTHON"
        required_tools = ("git", "sox")
        vendor = repo_root / "vendor" / "CosyVoice"
        wetext = repo_root / "models" / "cosyvoice" / "wetext"
        pythonpath_entries = (str(vendor), str(vendor / "third_party" / "Matcha-TTS"))
        steps = _base_steps(repo_root, venv, bootstrap_python, (3, 10))
        steps.extend(
            [
                Step(
                    "install build tools",
                    (str(worker_python), "-m", "pip", "install", "setuptools==80.9.0", "wheel"),
                    str(repo_root),
                ),
                Step(
                    "clone CosyVoice source",
                    ("git", "clone", "https://github.com/QwenAudio/CosyVoice.git", str(vendor)),
                    str(repo_root),
                    skip_if_exists=str(vendor / ".git"),
                ),
                Step("checkout pinned CosyVoice source", ("git", "-C", str(vendor), "checkout", COSY_SOURCE_REVISION), str(repo_root)),
                Step(
                    "initialize CosyVoice submodules",
                    ("git", "-C", str(vendor), "submodule", "update", "--init", "--recursive"),
                    str(repo_root),
                ),
                Step(
                    "build CPU-only CosyVoice requirements",
                    (
                        str(worker_python), str(repo_root / "scripts" / "prepare_cosyvoice_cpu_requirements.py"),
                        "--source", str(vendor / "requirements.txt"),
                        "--output", str(vendor / "requirements.cpu.txt"),
                    ),
                    str(repo_root),
                ),
                Step(
                    "install CPU PyTorch",
                    (
                        str(worker_python), "-m", "pip", "install", "--index-url",
                        "https://download.pytorch.org/whl/cpu", "torch==2.3.1", "torchaudio==2.3.1",
                    ),
                    str(repo_root),
                ),
                Step(
                    "install pinned Whisper",
                    (str(worker_python), "-m", "pip", "install", "--no-build-isolation", "openai-whisper==20231117"),
                    str(repo_root),
                ),
                Step(
                    "install CosyVoice CPU requirements",
                    (str(worker_python), "-m", "pip", "install", "-r", str(vendor / "requirements.cpu.txt")),
                    str(repo_root),
                ),
                Step(
                    "install CPU ONNX runtime",
                    (str(worker_python), "-m", "pip", "install", "onnxruntime==1.18.0", "huggingface_hub"),
                    str(repo_root),
                ),
            ]
        )
        runtime_env = {
            "EDGETTS_ARENA_COSYVOICE_WETEXT_DIR": str(wetext),
            "PYTHONPATH": os.pathsep.join(pythonpath_entries),
        }
        steps.append(
            Step(
                "CosyVoice runtime preflight",
                (
                    str(worker_python), "-c",
                    "import torch, onnxruntime as ort, whisper; from cosyvoice.cli.cosyvoice import AutoModel; "
                    "assert not torch.cuda.is_available(); assert 'CPUExecutionProvider' in ort.get_available_providers(); print(AutoModel)",
                ),
                str(repo_root),
                env=runtime_env,
            )
        )
        if include_assets:
            steps.extend(
                [
                    Step(
                        "prepare pinned CosyVoice model",
                        (
                            str(worker_python), str(repo_root / "scripts" / "prepare_cosyvoice_model.py"),
                            "--revision", COSY_MODEL_REVISION,
                            "--output", str(repo_root / "models" / "cosyvoice" / "CosyVoice-300M-SFT"),
                        ),
                        str(repo_root),
                    ),
                    Step(
                        "prepare local WeText frontend",
                        (
                            str(worker_python), str(repo_root / "scripts" / "prepare_cosyvoice_frontend.py"),
                            "--output", str(wetext),
                        ),
                        str(repo_root),
                    ),
                ]
            )
        env = {
            worker_env: str(worker_python),
            "EDGETTS_ARENA_COSYVOICE_WETEXT_DIR": str(wetext),
        }
        recommended = "3.10"

    doctor_env = dict(env)
    if pythonpath_entries:
        doctor_env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    if include_doctor:
        steps.append(_doctor_step(repo_root, worker_python, model_id, doctor_env))

    return BootstrapPlan(
        model=model,
        model_id=model_id,
        recommended_python=recommended,
        bootstrap_python=str(Path(bootstrap_python).expanduser()),
        venv=str(venv),
        worker_python=str(worker_python),
        worker_env_name=worker_env,
        repo_root=str(repo_root),
        required_tools=required_tools,
        requires_network=True,
        steps=tuple(steps),
        environment=env,
        pythonpath_entries=pythonpath_entries,
    )


def _merged_env(overrides: dict[str, str] | None) -> dict[str, str] | None:
    if not overrides:
        return None
    env = os.environ.copy()
    for key, value in overrides.items():
        if key == "PYTHONPATH" and env.get("PYTHONPATH"):
            env[key] = value + os.pathsep + env["PYTHONPATH"]
        else:
            env[key] = value
    return env


def execute_plan(
    plan: BootstrapPlan,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    missing_tools = [tool for tool in plan.required_tools if shutil.which(tool) is None]
    if missing_tools:
        raise RuntimeError(f"missing required host tools: {', '.join(missing_tools)}")

    for index, step in enumerate(plan.steps, start=1):
        if step.skip_if_exists and Path(step.skip_if_exists).exists():
            print(f"[{index}/{len(plan.steps)}] skip {step.label}: {step.skip_if_exists} exists", flush=True)
            continue
        print(f"[{index}/{len(plan.steps)}] {step.label}", flush=True)
        runner(
            list(step.command),
            cwd=step.cwd,
            env=_merged_env(step.env),
            check=True,
            text=True,
        )


def _shell_exports(plan: BootstrapPlan) -> str:
    lines = ["# Source this file before starting EdgeTTS-Arena."]
    for key, value in plan.environment.items():
        lines.append(f"export {key}={shlex.quote(value)}")
    if plan.pythonpath_entries:
        joined = os.pathsep.join(plan.pythonpath_entries)
        lines.append(f"export PYTHONPATH={shlex.quote(joined)}${{PYTHONPATH:+{os.pathsep}$PYTHONPATH}}")
    return "\n".join(lines) + "\n"


def _powershell_exports(plan: BootstrapPlan) -> str:
    lines = ["# Run this file before starting EdgeTTS-Arena."]
    for key, value in plan.environment.items():
        escaped = value.replace("'", "''")
        lines.append(f"$env:{key} = '{escaped}'")
    if plan.pythonpath_entries:
        joined = ";".join(plan.pythonpath_entries).replace("'", "''")
        lines.append(f"$env:PYTHONPATH = '{joined}' + $(if ($env:PYTHONPATH) {{ ';' + $env:PYTHONPATH }} else {{ '' }})")
    return "\n".join(lines) + "\n"


def write_bootstrap_artifacts(plan: BootstrapPlan, output_dir: Path) -> dict[str, str]:
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "bootstrap_plan.json"
    env_sh = output_dir / "env.sh"
    env_ps1 = output_dir / "env.ps1"
    plan_path.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    env_sh.write_text(_shell_exports(plan), encoding="utf-8")
    env_ps1.write_text(_powershell_exports(plan), encoding="utf-8")
    return {"plan": str(plan_path), "env_sh": str(env_sh), "env_ps1": str(env_ps1)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute a pinned dedicated-worker bootstrap for an extended TTS model."
    )
    parser.add_argument("model", choices=("qwen3", "melotts", "cosyvoice"))
    parser.add_argument("--python", default=sys.executable, help="Bootstrap Python executable; version is checked before venv creation.")
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--skip-assets", action="store_true", help="Install runtime/venv only; do not download model assets.")
    parser.add_argument("--no-doctor", action="store_true", help="Skip the final targeted worker Doctor probe.")
    parser.add_argument("--execute", action="store_true", help="Actually create/install/download. Without this flag only the plan is printed.")
    parser.add_argument("--output-dir", type=Path, help="Where to write bootstrap_plan.json and env.sh/env.ps1 after execution.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = build_plan(
        args.model,
        repo_root=args.repo_root,
        bootstrap_python=args.python,
        venv=args.venv,
        include_assets=not args.skip_assets,
        include_doctor=not args.no_doctor,
    )
    print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
    if not args.execute:
        print("Plan only. Re-run with --execute to create the venv and perform network-heavy installation.")
        return 0

    execute_plan(plan)
    output_dir = args.output_dir or (Path(plan.repo_root) / "exports" / "bootstrap" / plan.model)
    artifacts = write_bootstrap_artifacts(plan, output_dir)
    print(json.dumps({"status": "ready", "artifacts": artifacts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
