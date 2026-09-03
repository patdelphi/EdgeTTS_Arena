from __future__ import annotations

import argparse
from pathlib import Path

from edgetts_arena.adapters import DummyTTSAdapter, PiperTTSAdapter
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
    return parser


def main() -> int:
    args = build_parser().parse_args()
    configure_logging()

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

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
