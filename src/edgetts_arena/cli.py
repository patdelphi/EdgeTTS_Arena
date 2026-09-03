from __future__ import annotations

import argparse
from pathlib import Path

from edgetts_arena.adapters import DummyTTSAdapter, KokoroTTSAdapter, PiperTTSAdapter
from edgetts_arena.core.config import load_settings
from edgetts_arena.core.logging import configure_logging
from edgetts_arena.utils import write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgetts-arena")
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    serve = subparsers.add_parser("serve", help="start the local FastAPI service")
    serve.add_argument("--host", default=None, help="bind host; defaults to app config")
    serve.add_argument("--port", type=int, default=None, help="bind port; defaults to app config")
    serve.add_argument("--reload", action="store_true", help="enable uvicorn auto-reload for development")
    serve.add_argument("--ui", action="store_true", help="mount the optional Gradio Arena UI at /arena")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    settings = load_settings()
    configure_logging(settings.log_level)

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

    if args.command == "serve":
        import logging

        import uvicorn

        host = args.host or settings.host
        port = args.port or settings.port
        if host in {"0.0.0.0", "::"}:
            logging.getLogger(__name__).warning(
                "API is binding to %s and may be reachable from other devices", host
            )
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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
