from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from scripts.prepare_qwen3_model import DEFAULT_REVISION as MODEL_REVISION
from scripts.prepare_qwen3_native_manifest import DEFAULT_REVISION as RUNTIME_REVISION

RUNTIME_REPO = "https://github.com/gabriele-mastrapasqua/qwen3-tts.git"
MODEL_DIR_NAME = "Qwen3-TTS-12Hz-0.6B-CustomVoice"


@dataclass(frozen=True)
class Step:
    label: str
    command: tuple[str, ...]
    cwd: str
    skip_if_exists: str | None = None


@dataclass(frozen=True)
class NativeBootstrapPlan:
    repo_root: str
    bootstrap_python: str
    bootstrap_venv: str
    runtime_dir: str
    runtime_revision: str
    model_dir: str
    model_revision: str
    variants_dir: str
    required_tools: tuple[str, ...]
    host_note: str
    requires_network: bool
    steps: tuple[Step, ...]


def _venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _host_note() -> str:
    system = platform.system().lower()
    if system == "linux":
        return "Requires a C toolchain plus OpenBLAS development headers/library (for example build-essential + libopenblas-dev)."
    if system == "darwin":
        return "Requires Xcode Command Line Tools; the pinned Makefile uses Apple Accelerate on macOS."
    if system == "windows":
        return "Native bootstrap is not a supported Windows baseline; use WSL/Linux or provide a compatible make/gcc/OpenBLAS toolchain explicitly."
    return "Requires git, make, gcc and a BLAS implementation compatible with the pinned runtime Makefile."


def build_plan(
    *,
    repo_root: Path,
    bootstrap_python: str,
    bootstrap_venv: Path | None = None,
    runtime_dir: Path | None = None,
    model_dir: Path | None = None,
    variants_dir: Path | None = None,
    include_model_assets: bool = True,
) -> NativeBootstrapPlan:
    repo_root = repo_root.expanduser().resolve()
    bootstrap_venv = (bootstrap_venv or repo_root / ".venv-qwen3-native-bootstrap").expanduser().resolve()
    runtime_dir = (runtime_dir or repo_root / "runtime" / "qwen3-tts-c").expanduser().resolve()
    model_dir = (model_dir or repo_root / "models" / "qwen3" / MODEL_DIR_NAME).expanduser().resolve()
    variants_dir = (variants_dir or repo_root / "models" / "qwen3-native").expanduser().resolve()
    python = _venv_python(bootstrap_venv)

    version_check = (
        bootstrap_python,
        "-c",
        "import sys; actual=sys.version_info[:2]; assert actual==(3,11), f'expected Python 3.11, got {actual[0]}.{actual[1]}'",
    )
    steps: list[Step] = [
        Step("check bootstrap Python", version_check, str(repo_root)),
        Step("create bootstrap venv", (bootstrap_python, "-m", "venv", str(bootstrap_venv)), str(repo_root)),
        Step("upgrade pip", (str(python), "-m", "pip", "install", "--upgrade", "pip"), str(repo_root)),
        Step(
            "install Arena preparation runtime",
            (str(python), "-m", "pip", "install", "-e", str(repo_root), "huggingface_hub>=0.34"),
            str(repo_root),
        ),
        Step(
            "clone pinned native runtime source",
            ("git", "clone", RUNTIME_REPO, str(runtime_dir)),
            str(repo_root),
            skip_if_exists=str(runtime_dir / ".git"),
        ),
        Step(
            "checkout pinned native runtime revision",
            ("git", "-C", str(runtime_dir), "checkout", RUNTIME_REVISION),
            str(repo_root),
        ),
        Step("build native BLAS runtime", ("make", "-C", str(runtime_dir), "blas"), str(repo_root)),
        Step("native ISA capabilities", (str(runtime_dir / "qwen_tts"), "--caps"), str(repo_root)),
        Step("native runtime self-test", (str(runtime_dir / "qwen_tts"), "--self-test"), str(repo_root)),
    ]
    if include_model_assets:
        steps.append(
            Step(
                "prepare pinned official Qwen3 model",
                (
                    str(python),
                    str(repo_root / "scripts" / "prepare_qwen3_model.py"),
                    "--revision",
                    MODEL_REVISION,
                    "--output",
                    str(model_dir),
                ),
                str(repo_root),
            )
        )
    steps.extend(
        [
            Step(
                "create matched INT8/INT4 manifests",
                (
                    str(python),
                    str(repo_root / "scripts" / "prepare_qwen3_native_variants.py"),
                    "--binary",
                    str(runtime_dir / "qwen_tts"),
                    "--model-dir",
                    str(model_dir),
                    "--output-root",
                    str(variants_dir),
                    "--runtime-revision",
                    RUNTIME_REVISION,
                    "--default-voice",
                    "Vivian",
                    "--default-language",
                    "Chinese",
                ),
                str(repo_root),
            ),
            Step(
                "Arena native adapter preflight",
                (
                    str(python),
                    "-c",
                    "from edgetts_arena.adapters.qwen3_native_adapter import Qwen3NativeTTSAdapter; "
                    f"a=Qwen3NativeTTSAdapter(); a.load_model(r'{variants_dir / 'int8' / 'model.json'}', num_threads=2); "
                    "print(a.available_voices); a.unload_model()",
                ),
                str(repo_root),
            ),
        ]
    )
    return NativeBootstrapPlan(
        repo_root=str(repo_root),
        bootstrap_python=str(bootstrap_python),
        bootstrap_venv=str(bootstrap_venv),
        runtime_dir=str(runtime_dir),
        runtime_revision=RUNTIME_REVISION,
        model_dir=str(model_dir),
        model_revision=MODEL_REVISION,
        variants_dir=str(variants_dir),
        required_tools=("git", "make", "gcc"),
        host_note=_host_note(),
        requires_network=True,
        steps=tuple(steps),
    )


def execute_plan(
    plan: NativeBootstrapPlan,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    if platform.system().lower() == "windows":
        raise RuntimeError(
            "qwen3-native bootstrap is not a supported native Windows baseline; use WSL/Linux or prepare a compatible toolchain manually"
        )
    missing = [tool for tool in plan.required_tools if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"missing required host tools: {', '.join(missing)}")
    for index, step in enumerate(plan.steps, start=1):
        if step.skip_if_exists and Path(step.skip_if_exists).exists():
            print(f"[{index}/{len(plan.steps)}] skip {step.label}: {step.skip_if_exists} exists", flush=True)
            continue
        print(f"[{index}/{len(plan.steps)}] {step.label}", flush=True)
        runner(list(step.command), cwd=step.cwd, check=True, text=True)


def write_plan(plan: NativeBootstrapPlan, output: Path) -> Path:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(plan), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the pinned Qwen3 pure-C INT8/INT4 local bootstrap."
    )
    parser.add_argument("--python", default="python3.11")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--venv", type=Path)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--variants-dir", type=Path)
    parser.add_argument("--skip-model-assets", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--plan-output", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    plan = build_plan(
        repo_root=args.repo_root,
        bootstrap_python=args.python,
        bootstrap_venv=args.venv,
        runtime_dir=args.runtime_dir,
        model_dir=args.model_dir,
        variants_dir=args.variants_dir,
        include_model_assets=not args.skip_model_assets,
    )
    print(json.dumps(asdict(plan), ensure_ascii=False, indent=2))
    if not args.execute:
        print("Plan only. Install the host dependencies described in host_note, then re-run with --execute.")
        return 0
    execute_plan(plan)
    output = args.plan_output or Path(plan.repo_root) / "exports" / "bootstrap" / "qwen3-native" / "bootstrap_plan.json"
    path = write_plan(plan, output)
    print(json.dumps({"status": "ready", "plan": str(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
