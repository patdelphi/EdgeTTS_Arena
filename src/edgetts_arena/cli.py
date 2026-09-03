from __future__ import annotations

import argparse
from pathlib import Path

from edgetts_arena.adapters import DummyTTSAdapter
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
