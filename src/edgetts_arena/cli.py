from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from edgetts_arena.adapters import DummyTTSAdapter, KokoroTTSAdapter, PiperTTSAdapter
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_suite import BenchmarkPresetSuite, RepeatedBenchmarkService
from edgetts_arena.core.config import load_settings
from edgetts_arena.core.model_registry import ModelRegistry, collect_worker_env_warnings
from edgetts_arena.core.process_runner import ProcessRunner
from edgetts_arena.core.residency import ResidencyManager
from edgetts_arena.core.resource_guard import ResourceGuard
from edgetts_arena.core.logging import configure_logging
from edgetts_arena.utils import write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgetts-arena")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="validate the local deployment baseline")
    doctor.add_argument("--ui", action="store_true", help="also validate Gradio UI mounting")
    doctor.add_argument(
        "--workers",
        action="store_true",
        help="also validate configured dedicated Python worker interpreters",
    )
    doctor.add_argument(
        "--worker",
        dest="worker_models",
        action="append",
        default=None,
        metavar="MODEL_ID",
        help="limit worker validation to a model id; repeatable and implies --workers",
    )
    doctor.add_argument("--exports-root", type=Path, default=Path("exports"))

    dummy = subparsers.add_parser("dummy", help="generate a deterministic dummy WAV")
    dummy.add_argument("--text", default="EdgeTTS Arena smoke test")
    dummy.add_argument("--voice", default="default")
    dummy.add_argument("--speed", type=float, default=1.0)
    dummy.add_argument("--seed", type=int, default=0)
    dummy.add_argument("--output", type=Path, default=Path("exports/dummy.wav"))

    piper = subparsers.add_parser("piper", help="synthesize with a local Piper ONNX voice")
    piper.add_argument("--model", type=Path, required=True, help=".onnx file or a directory containing exactly one .onnx voice")
    piper.add_argument("--text", default="EdgeTTS Arena Piper smoke test")
    piper.add_argument("--voice", default=None, help="speaker name/id for multi-speaker Piper voices")
    piper.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier")
    piper.add_argument("--threads", type=int, default=4)
    piper.add_argument("--output", type=Path, default=Path("exports/piper.wav"))

    kokoro = subparsers.add_parser("kokoro", help="synthesize with a local Kokoro v1 ONNX model")
    kokoro.add_argument("--model", type=Path, required=True, help="kokoro-v1*.onnx file or containing directory")
    kokoro.add_argument("--text", default="EdgeTTS Arena Kokoro smoke test")
    kokoro.add_argument("--voice", default=None, help="Kokoro voice id; defaults to af_heart/af_sarah")
    kokoro.add_argument("--language", default=None, help="direct-text language; currently en-us/en-gb")
    kokoro.add_argument("--speed", type=float, default=1.0, help="speech speed multiplier (0.5-2.0)")
    kokoro.add_argument("--threads", type=int, default=4)
    kokoro.add_argument("--output", type=Path, default=Path("exports/kokoro.wav"))

    suite = subparsers.add_parser("suite", help="run the standard repeated benchmark suite")
    suite.add_argument("--models", nargs="+", required=True, help="1-4 configured model ids")
    suite.add_argument("--cases", nargs="+", default=None, help="case ids; defaults to TC-01..TC-05")
    suite.add_argument("--warmup-runs", type=int, default=None)
    suite.add_argument("--measured-runs", type=int, default=None)
    suite.add_argument("--threads", type=int, default=4)
    suite.add_argument(
        "--language",
        default=None,
        help="explicit language code for models with language_control capability (for example zh/en/ja)",
    )
    suite.add_argument("--exports-root", type=Path, default=Path("exports"))

    serve = subparsers.add_parser("serve", help="start the local FastAPI service")
    serve.add_argument("--host", default=None, help="bind host; defaults to app config")
    serve.add_argument("--port", type=int, default=None, help="bind port; defaults to app config")
    serve.add_argument("--reload", action="store_true", help="enable uvicorn auto-reload for development")
    serve.add_argument("--ui", action="store_true", help="mount the optional Gradio Arena UI at /arena")

    download = subparsers.add_parser("download", help="download TTS models")
    download.add_argument(
        "model",
        nargs="?",
        default=None,
        help="model id to download; if omitted, list available models",
    )
    download.add_argument(
        "--list",
        action="store_true",
        dest="list_models",
        help="list all downloadable models",
    )
    download.add_argument(
        "--all",
        action="store_true",
        dest="download_all",
        help="download all available models",
    )
    return parser


def collect_doctor_report(
    *,
    exports_root: Path,
    check_ui: bool = False,
    check_workers: bool = False,
    worker_model_ids: list[str] | None = None,
) -> dict[str, object]:
    checks: list[dict[str, object]] = []

    def record(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    selected_worker_ids = list(dict.fromkeys(worker_model_ids or []))
    check_workers = bool(check_workers or selected_worker_ids)

    py_ok = (3, 10) <= sys.version_info[:2] < (3, 14)
    record("python", py_ok, platform.python_version())

    try:
        settings = load_settings()
        record("config", True, f"host={settings.host} port={settings.port}")
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        settings = None
        record("config", False, f"{type(exc).__name__}: {exc}")

    try:
        exports_root.mkdir(parents=True, exist_ok=True)
        probe = exports_root / ".edgetts-arena-doctor"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        record("exports_writable", True, str(exports_root.resolve()))
    except Exception as exc:
        record("exports_writable", False, f"{type(exc).__name__}: {exc}")

    try:
        adapter = DummyTTSAdapter()
        adapter.load_model(num_threads=1)
        output = adapter.infer("local deployment doctor", seed=0)
        record(
            "dummy_synthesis",
            bool(output.audio.size and output.sample_rate > 0),
            f"samples={output.audio.size} sample_rate={output.sample_rate}",
        )
    except Exception as exc:
        record("dummy_synthesis", False, f"{type(exc).__name__}: {exc}")

    try:
        from edgetts_arena.api.app import create_app

        app = create_app(exports_root=str(exports_root))
        routes = {route.path for route in app.routes}
        required = {"/healthz", "/api/v1/system/models", "/api/v1/benchmark/run"}
        missing = sorted(required - routes)
        record("fastapi_app", not missing, "routes ok" if not missing else f"missing routes: {missing}")
    except Exception as exc:
        record("fastapi_app", False, f"{type(exc).__name__}: {exc}")

    if check_ui:
        try:
            from edgetts_arena.ui.gradio_app import create_full_app

            ui_app = create_full_app(exports_root=str(exports_root))
            route_paths = {route.path for route in ui_app.routes}
            mounted = any(path.startswith("/arena") for path in route_paths)
            record("gradio_ui", mounted, "mounted at /arena" if mounted else "Arena route not mounted")
        except Exception as exc:
            record("gradio_ui", False, f"{type(exc).__name__}: {exc}")

    if check_workers:
        try:
            registry = ModelRegistry.from_yaml()
            runner = ProcessRunner()
            all_worker_specs = {
                model_id: registry.get_record(model_id).spec
                for model_id in registry.ids()
                if registry.get_record(model_id).spec.worker_python
                or registry.get_record(model_id).spec.worker_python_env
            }
            worker_specs = []
            if selected_worker_ids:
                known_ids = set(registry.ids())
                for model_id in selected_worker_ids:
                    if model_id not in known_ids:
                        record(f"worker:{model_id}", False, "unknown model id")
                        continue
                    spec = all_worker_specs.get(model_id)
                    if spec is None:
                        record(
                            f"worker:{model_id}",
                            False,
                            "model does not declare a dedicated worker interpreter",
                        )
                        continue
                    worker_specs.append(spec)
            else:
                worker_specs = list(all_worker_specs.values())

            if not worker_specs and not selected_worker_ids:
                record("external_workers", True, "no dedicated worker interpreters configured")

            for spec in worker_specs:
                executable = spec.resolve_worker_python()
                check_name = f"worker:{spec.id}"
                if not executable:
                    detail = (
                        f"environment variable {spec.worker_python_env} is not set"
                        if spec.worker_python_env
                        else "worker_python is empty"
                    )
                    record(check_name, False, detail)
                    continue
                probe_audio = exports_root / f".edgetts-arena-worker-{spec.id}.wav"
                task = {
                    "adapter": "dummy",
                    "model_path": "",
                    "text": "external worker doctor",
                    "num_threads": 1,
                    "infer_kwargs": {"seed": 0},
                    "audio_path": str(probe_audio),
                }
                try:
                    result = runner.run_external_worker(
                        executable,
                        "single",
                        task,
                        timeout_sec=10.0,
                    )
                    payload = result.value if isinstance(result.value, dict) else {}
                    ok = (
                        result.status == "success"
                        and payload.get("status") == "success"
                        and probe_audio.is_file()
                        and probe_audio.stat().st_size > 44
                    )
                    detail = (
                        f"external protocol ok pid={result.pid}"
                        if ok
                        else result.error_message or str(payload.get("error") or "external worker probe failed")
                    )
                    record(check_name, ok, detail)
                finally:
                    probe_audio.unlink(missing_ok=True)
        except Exception as exc:
            record("external_workers", False, f"{type(exc).__name__}: {exc}")

    ready = all(bool(check["ok"]) for check in checks)
    try:
        worker_env_warnings = collect_worker_env_warnings(ModelRegistry.from_yaml())
    except Exception:  # pragma: no cover - defensive; never fail doctor on this notice
        worker_env_warnings = []
    return {
        "ready": ready,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ui_requested": check_ui,
        "workers_requested": check_workers,
        "worker_models_requested": selected_worker_ids,
        "worker_env_warnings": worker_env_warnings,
        "checks": checks,
    }


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()
    configure_logging(settings.log_level)

    if args.command == "doctor":
        report = collect_doctor_report(
            exports_root=args.exports_root,
            check_ui=args.ui,
            check_workers=args.workers,
            worker_model_ids=args.worker_models,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ready"] else 1

    if args.command == "dummy":
        adapter = DummyTTSAdapter()
        adapter.load_model(num_threads=1)
        output = adapter.infer(
            args.text,
            voice=args.voice,
            speed=args.speed,
            seed=args.seed,
        )
        path = write_wav(args.output, output.audio, output.sample_rate)
        print(path)
        return 0

    if args.command == "piper":
        adapter = PiperTTSAdapter()
        adapter.load_model(str(args.model), num_threads=args.threads)
        output = adapter.infer(
            args.text,
            voice=args.voice,
            speed=args.speed,
        )
        path = write_wav(args.output, output.audio, output.sample_rate)
        print(path)
        return 0

    if args.command == "kokoro":
        adapter = KokoroTTSAdapter()
        adapter.load_model(str(args.model), num_threads=args.threads)
        infer_kwargs = {} if args.language is None else {"language": args.language}
        output = adapter.infer(
            args.text,
            voice=args.voice,
            speed=args.speed,
            **infer_kwargs,
        )
        path = write_wav(args.output, output.audio, output.sample_rate)
        print(path)
        return 0

    if args.command == "suite":
        registry = ModelRegistry.from_yaml()
        store = RunArtifactStore(args.exports_root)
        guard = ResourceGuard(settings.resource_guard)
        # Keep-warm lets a heavy model load once and be reused across every case in
        # this suite (eager mode re-spawns a worker per case). evict_all on exit so
        # no warm subprocess outlives the CLI run.
        residency = ResidencyManager(registry, settings.residency, guard)
        service = RepeatedBenchmarkService(
            registry,
            guard,
            store,
            preset_suite=BenchmarkPresetSuite.load(),
            inference_timeout_sec=settings.inference_timeout_sec,
            residency=residency,
        )
        try:
            data = service.run_suite(
                model_ids=list(args.models),
                case_ids=None if args.cases is None else list(args.cases),
                cpu_threads_per_model=args.threads,
                warmup_runs=args.warmup_runs,
                measured_runs=args.measured_runs,
                config={
                    "speed": 1.0,
                    "voice": None,
                    "language": None if args.language is None else args.language.strip().lower(),
                    "seed": None,
                    "sample_rate": None,
                },
            )
        finally:
            residency.evict_all()
        export_path = store.build_export(data["run_id"])
        print(f"{data['run_id']}\t{export_path}")
        return 0

    if args.command == "serve":
        import logging

        import uvicorn

        host = args.host or settings.host
        port = args.port or settings.port
        if host in {"0.0.0.0", "::"}:
            logging.getLogger(__name__).warning(
                "API is binding to %s and may be reachable from other devices", host
            )
        # Startup self-check: dedicated worker environments (Qwen3 / CosyVoice /
        # MeloTTS) are routed through env vars set by exports/bootstrap/*/env.ps1.
        # If one is missing the model silently falls back to in-process spawn and
        # fails with a cryptic import/NoneType error, so warn clearly up front.
        worker_env_warnings = collect_worker_env_warnings(
            ModelRegistry.from_yaml(search_paths=settings.model_search_paths)
        )
        if worker_env_warnings:
            print(
                "[startup-check] Dedicated worker environment is NOT configured for "
                f"{len(worker_env_warnings)} model(s); they will fail until you source "
                "their env script in THIS shell before starting:",
                file=sys.stderr,
            )
            for message in worker_env_warnings:
                print(f"  - {message}", file=sys.stderr)
        app_factory = (
            "edgetts_arena.ui.gradio_app:create_full_app"
            if args.ui
            else "edgetts_arena.api.app:create_app"
        )
        uvicorn.run(
            app_factory,
            factory=True,
            host=host,
            port=port,
            reload=args.reload,
            log_level=settings.log_level.lower(),
        )
        return 0

    if args.command == "download":
        from edgetts_arena.core.model_downloader import (
            download_model,
            list_downloadable_models,
            check_model_downloaded,
        )
        
        project_root = Path.cwd()
        search_paths = settings.model_search_paths
        
        # 列出可下载模型
        if args.list_models or (not args.model and not args.download_all):
            models = list_downloadable_models()
            print("可下载的模型列表:")
            print("-" * 60)
            for m in models:
                status = "✓ 已下载" if check_model_downloaded(m["id"], search_paths, project_root) else "○ 未下载"
                print(f"  {m['id']:<20} {status:<10} ~{m['size_mb']}MB")
                print(f"    {m['description']}")
                print(f"    来源: {m['repo_id']}")
                print()
            print(f"搜索路径: {search_paths}")
            return 0
        
        # 下载所有模型
        if args.download_all:
            models = list_downloadable_models()
            results = []
            for m in models:
                print(f"\n正在下载 {m['id']}...")
                def progress(p):
                    if p.status == "downloading":
                        print(f"  {p.message}", end="\r")
                result = download_model(m["id"], search_paths, project_root, progress)
                results.append(result)
                print(f"  {result['message']}")
            
            success = sum(1 for r in results if r["success"])
            print(f"\n完成: {success}/{len(results)} 个模型下载成功")
            return 0 if success == len(results) else 1
        
        # 下载指定模型
        if args.model:
            def progress(p):
                if p.status == "downloading":
                    print(f"  {p.message}", end="\r")
                elif p.status == "complete":
                    print(f"  {p.message}")
            
            print(f"正在下载 {args.model}...")
            result = download_model(args.model, search_paths, project_root, progress)
            print(result["message"])
            return 0 if result["success"] else 1
        
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
