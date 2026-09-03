from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Callable

try:
    from scripts.acceptance_artifacts import prepare_output_dir, write_zip
    from scripts.real_model_smoke import run_gate
except ModuleNotFoundError:  # direct: python scripts/target_device_acceptance.py
    from acceptance_artifacts import prepare_output_dir, write_zip
    from real_model_smoke import run_gate

from edgetts_arena.core.system_info import collect_system_environment

GateRunner = Callable[[argparse.Namespace], dict[str, object]]
EnvironmentCollector = Callable[..., dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a reproducible real-model acceptance pack on an actual target device."
    )
    parser.add_argument("model", choices=("melotts", "cosyvoice", "qwen3", "qwen3-native"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--language")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("exports/target-device"))
    parser.add_argument(
        "--require-arch",
        help="comma-separated accepted platform.machine() values, e.g. aarch64,arm64",
    )
    parser.add_argument("--max-rtf", type=float)
    parser.add_argument("--max-peak-rss-mb", type=float)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing acceptance output directory/archive before running",
    )
    return parser


def _stats(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _accepted_arches(raw: str | None) -> set[str]:
    if raw is None:
        return set()
    return {value.strip().lower() for value in raw.split(",") if value.strip()}


def run_acceptance(
    args: argparse.Namespace,
    *,
    gate_runner: GateRunner = run_gate,
    environment_collector: EnvironmentCollector = collect_system_environment,
) -> dict[str, object]:
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.max_rtf is not None and args.max_rtf <= 0:
        raise ValueError("max_rtf must be positive")
    if args.max_peak_rss_mb is not None and args.max_peak_rss_mb <= 0:
        raise ValueError("max_peak_rss_mb must be positive")

    root, archive = prepare_output_dir(
        Path(args.output_dir), overwrite=bool(getattr(args, "overwrite", False))
    )
    environment = environment_collector(cpu_threads_per_model=args.threads)
    (root / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    run_reports: list[dict[str, object]] = []
    run_errors: list[dict[str, object]] = []
    for index in range(1, args.runs + 1):
        output = root / f"run-{index:02d}.wav"
        report_path = root / f"run-{index:02d}.json"
        gate_args = argparse.Namespace(
            model=args.model,
            model_path=args.model_path,
            text=args.text,
            voice=args.voice,
            language=args.language,
            seed=args.seed,
            threads=args.threads,
            speed=1.0,
            output=str(output),
            report=str(report_path),
        )
        try:
            report = gate_runner(gate_args)
            run_reports.append(dict(report))
        except Exception as exc:  # acceptance boundary must still produce a report
            run_errors.append(
                {"run": index, "type": type(exc).__name__, "message": str(exc)}
            )

    rtf_values = [
        float((report.get("metrics") or {}).get("rtf"))
        for report in run_reports
        if (report.get("metrics") or {}).get("rtf") is not None
    ]
    rss_values = [
        float((report.get("metrics") or {}).get("peak_rss_mb"))
        for report in run_reports
        if (report.get("metrics") or {}).get("peak_rss_mb") is not None
    ]
    inference_values = [
        float((report.get("metrics") or {}).get("inference_time_ms"))
        for report in run_reports
        if (report.get("metrics") or {}).get("inference_time_ms") is not None
    ]

    checks: list[dict[str, object]] = [
        _check(
            "all_runs_successful",
            len(run_reports) == args.runs and not run_errors,
            f"successful={len(run_reports)}/{args.runs}",
        )
    ]

    accepted_arches = _accepted_arches(args.require_arch)
    if accepted_arches:
        actual_arch = str(environment.get("arch") or "unknown").lower()
        checks.append(
            _check(
                "required_arch",
                actual_arch in accepted_arches,
                f"actual={actual_arch}; accepted={','.join(sorted(accepted_arches))}",
            )
        )

    thread_settings = environment.get("thread_settings") or {}
    if args.model == "qwen3-native":
        openblas = thread_settings.get("openblas_num_threads") if isinstance(thread_settings, dict) else None
        checks.append(
            _check(
                "native_openblas_unset",
                not openblas,
                "OPENBLAS_NUM_THREADS is unset"
                if not openblas
                else f"OPENBLAS_NUM_THREADS={openblas}; unset it so the pinned runtime controls BLAS threads",
            )
        )

    if args.max_rtf is not None:
        worst_rtf = max(rtf_values) if rtf_values else float("inf")
        checks.append(
            _check(
                "max_rtf",
                worst_rtf <= args.max_rtf,
                f"worst={worst_rtf:.4f}; limit={args.max_rtf:.4f}",
            )
        )
    if args.max_peak_rss_mb is not None:
        worst_rss = max(rss_values) if rss_values else float("inf")
        checks.append(
            _check(
                "max_peak_rss_mb",
                worst_rss <= args.max_peak_rss_mb,
                f"worst={worst_rss:.1f}MB; limit={args.max_peak_rss_mb:.1f}MB",
            )
        )

    passed = all(bool(check["ok"]) for check in checks)
    report: dict[str, object] = {
        "passed": passed,
        "scope": "target-device-real-synthesis",
        "model": args.model,
        "model_path": str(Path(args.model_path).expanduser().resolve()),
        "request": {
            "text": args.text,
            "voice": args.voice,
            "language": args.language,
            "seed": args.seed,
            "threads": args.threads,
            "runs": args.runs,
        },
        "thresholds": {
            "require_arch": args.require_arch,
            "max_rtf": args.max_rtf,
            "max_peak_rss_mb": args.max_peak_rss_mb,
        },
        "performance_thresholds_supplied": bool(
            args.max_rtf is not None or args.max_peak_rss_mb is not None
        ),
        "environment": environment,
        "aggregate": {
            "rtf": _stats(rtf_values),
            "peak_rss_mb": _stats(rss_values),
            "inference_time_ms": _stats(inference_values),
        },
        "checks": checks,
        "runs": run_reports,
        "errors": run_errors,
    }
    if not args.no_zip:
        report["archive"] = str(archive)
    report_path = root / "acceptance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_zip:
        write_zip(root, archive)
    return report


def main() -> int:
    report = run_acceptance(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
