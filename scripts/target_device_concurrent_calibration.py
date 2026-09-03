from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.acceptance_artifacts import prepare_output_dir, write_zip
except ModuleNotFoundError:  # direct: python scripts/target_device_concurrent_calibration.py
    from acceptance_artifacts import prepare_output_dir, write_zip

from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.config import load_settings
from edgetts_arena.core.model_registry import ModelRegistry
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.system_info import collect_system_environment

BenchmarkRunner = Callable[..., dict[str, Any]]


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


def _metric(result: dict[str, Any], name: str) -> float | None:
    metrics = result.get("metrics")
    if result.get("status") != "success" or not isinstance(metrics, dict):
        return None
    value = metrics.get(name)
    return None if value is None else float(value)


def _result_map(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["model_id"]): item
        for item in run.get("results") or []
        if isinstance(item, dict) and item.get("model_id")
    }


def _check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"name": name, "ok": bool(ok), "detail": detail}


def build_real_runner(*, exports_root: Path, models_config: Path, app_config: Path) -> BenchmarkRunner:
    settings = load_settings(app_config)
    registry = ModelRegistry.from_yaml(models_config)
    service = BenchmarkService(
        registry,
        ResourceGuard(settings.resource_guard),
        RunArtifactStore(exports_root),
        inference_timeout_sec=float(settings.inference_timeout_sec),
    )
    return service.run


def run_calibration(
    args: argparse.Namespace,
    *,
    benchmark_runner: BenchmarkRunner | None = None,
) -> dict[str, Any]:
    model_ids = list(dict.fromkeys(str(value) for value in args.models))
    if not 2 <= len(model_ids) <= 4:
        raise ValueError("concurrent calibration requires 2-4 unique model ids")
    if args.runs < 1:
        raise ValueError("runs must be >= 1")
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    for name in ("max_rtf_slowdown_ratio", "max_concurrent_rtf", "max_concurrent_peak_rss_mb"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    root, archive = prepare_output_dir(
        Path(args.output_dir), overwrite=bool(getattr(args, "overwrite", False))
    )
    run_root = root / "runs"
    environment = collect_system_environment(cpu_threads_per_model=args.threads)
    (root / "environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    runner = benchmark_runner or build_real_runner(
        exports_root=run_root,
        models_config=Path(args.models_config),
        app_config=Path(args.app_config),
    )
    config = {
        "voice": args.voice,
        "language": args.language,
        "seed": args.seed,
        "speed": 1.0,
        "sample_rate": None,
    }

    pairs: list[dict[str, Any]] = []
    per_model: dict[str, dict[str, list[float]]] = {
        model_id: {
            "sequential_rtf": [],
            "concurrent_rtf": [],
            "rtf_slowdown_ratio": [],
            "sequential_peak_rss_mb": [],
            "concurrent_peak_rss_mb": [],
            "peak_rss_delta_mb": [],
            "sequential_cpu_pct": [],
            "concurrent_cpu_pct": [],
        }
        for model_id in model_ids
    }

    successful_pair_count = 0
    for index in range(1, args.runs + 1):
        sequential = runner(
            text=args.text,
            model_ids=model_ids,
            execution_mode="sequential",
            cpu_threads_per_model=args.threads,
            config=config,
        )
        concurrent = runner(
            text=args.text,
            model_ids=model_ids,
            execution_mode="concurrent",
            cpu_threads_per_model=args.threads,
            config=config,
        )
        sequential_map = _result_map(sequential)
        concurrent_map = _result_map(concurrent)
        pair_models: dict[str, Any] = {}
        pair_success = True

        for model_id in model_ids:
            base = sequential_map.get(model_id, {})
            pressure = concurrent_map.get(model_id, {})
            base_rtf = _metric(base, "rtf")
            pressure_rtf = _metric(pressure, "rtf")
            base_rss = _metric(base, "peak_rss_mb")
            pressure_rss = _metric(pressure, "peak_rss_mb")
            base_cpu = _metric(base, "avg_cpu_usage_pct")
            pressure_cpu = _metric(pressure, "avg_cpu_usage_pct")
            slowdown = (
                pressure_rtf / base_rtf
                if base_rtf is not None and pressure_rtf is not None and base_rtf > 0
                else None
            )
            rss_delta = (
                pressure_rss - base_rss
                if base_rss is not None and pressure_rss is not None
                else None
            )
            pair_success = pair_success and base.get("status") == "success" and pressure.get("status") == "success"
            pair_models[model_id] = {
                "sequential": base,
                "concurrent": pressure,
                "rtf_slowdown_ratio": slowdown,
                "peak_rss_delta_mb": rss_delta,
            }
            values = per_model[model_id]
            for key, value in (
                ("sequential_rtf", base_rtf),
                ("concurrent_rtf", pressure_rtf),
                ("rtf_slowdown_ratio", slowdown),
                ("sequential_peak_rss_mb", base_rss),
                ("concurrent_peak_rss_mb", pressure_rss),
                ("peak_rss_delta_mb", rss_delta),
                ("sequential_cpu_pct", base_cpu),
                ("concurrent_cpu_pct", pressure_cpu),
            ):
                if value is not None:
                    values[key].append(float(value))

        if pair_success:
            successful_pair_count += 1
        pairs.append(
            {
                "iteration": index,
                "success": pair_success,
                "sequential_run_id": sequential.get("run_id"),
                "concurrent_run_id": concurrent.get("run_id"),
                "sequential_execution": {
                    "requested_threads_per_model": sequential.get("requested_cpu_threads_per_model"),
                    "effective_threads_per_model": sequential.get("cpu_threads_per_model"),
                    "total_threads_budget": sequential.get("total_threads_budget"),
                    "resource_warnings": sequential.get("resource_warnings") or [],
                },
                "concurrent_execution": {
                    "requested_threads_per_model": concurrent.get("requested_cpu_threads_per_model"),
                    "effective_threads_per_model": concurrent.get("cpu_threads_per_model"),
                    "total_threads_budget": concurrent.get("total_threads_budget"),
                    "resource_warnings": concurrent.get("resource_warnings") or [],
                },
                "models": pair_models,
            }
        )

    aggregates: dict[str, Any] = {
        model_id: {name: _stats(values) for name, values in metrics.items()}
        for model_id, metrics in per_model.items()
    }
    checks: list[dict[str, object]] = [
        _check(
            "all_pairs_successful",
            successful_pair_count == args.runs,
            f"successful_pairs={successful_pair_count}/{args.runs}",
        )
    ]

    if args.max_rtf_slowdown_ratio is not None:
        worst = max(
            (
                value
                for metrics in per_model.values()
                for value in metrics["rtf_slowdown_ratio"]
            ),
            default=float("inf"),
        )
        checks.append(
            _check(
                "max_rtf_slowdown_ratio",
                worst <= args.max_rtf_slowdown_ratio,
                f"worst={worst:.4f}; limit={args.max_rtf_slowdown_ratio:.4f}",
            )
        )
    if args.max_concurrent_rtf is not None:
        worst = max(
            (value for metrics in per_model.values() for value in metrics["concurrent_rtf"]),
            default=float("inf"),
        )
        checks.append(
            _check(
                "max_concurrent_rtf",
                worst <= args.max_concurrent_rtf,
                f"worst={worst:.4f}; limit={args.max_concurrent_rtf:.4f}",
            )
        )
    if args.max_concurrent_peak_rss_mb is not None:
        worst = max(
            (
                value
                for metrics in per_model.values()
                for value in metrics["concurrent_peak_rss_mb"]
            ),
            default=float("inf"),
        )
        checks.append(
            _check(
                "max_concurrent_peak_rss_mb",
                worst <= args.max_concurrent_peak_rss_mb,
                f"worst={worst:.1f}MB; limit={args.max_concurrent_peak_rss_mb:.1f}MB",
            )
        )

    report: dict[str, Any] = {
        "passed": all(bool(check["ok"]) for check in checks),
        "scope": "target-device-concurrent-calibration",
        "models": model_ids,
        "request": {
            "text": args.text,
            "voice": args.voice,
            "language": args.language,
            "seed": args.seed,
            "requested_threads_per_model": args.threads,
            "runs": args.runs,
        },
        "thresholds": {
            "max_rtf_slowdown_ratio": args.max_rtf_slowdown_ratio,
            "max_concurrent_rtf": args.max_concurrent_rtf,
            "max_concurrent_peak_rss_mb": args.max_concurrent_peak_rss_mb,
        },
        "environment": environment,
        "aggregate_by_model": aggregates,
        "checks": checks,
        "pairs": pairs,
    }
    if not args.no_zip:
        report["archive"] = str(archive)
    (root / "calibration_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if not args.no_zip:
        write_zip(root, archive)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate sequential baseline versus concurrent pressure on an actual target device."
    )
    parser.add_argument("--models", nargs="+", required=True, help="2-4 enabled model ids")
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--language")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--models-config", type=Path, default=Path("config/models_config.yaml"))
    parser.add_argument("--app-config", type=Path, default=Path("config/app_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("exports/target-concurrent"))
    parser.add_argument("--max-rtf-slowdown-ratio", type=float)
    parser.add_argument("--max-concurrent-rtf", type=float)
    parser.add_argument("--max-concurrent-peak-rss-mb", type=float)
    parser.add_argument("--no-zip", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="delete an existing calibration output directory/archive before running",
    )
    return parser


def main() -> int:
    report = run_calibration(build_parser().parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
