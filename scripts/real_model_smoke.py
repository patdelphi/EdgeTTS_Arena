from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgetts_arena.adapters.cosyvoice_adapter import CosyVoiceTTSAdapter
from edgetts_arena.adapters.melotts_adapter import MeloTTSAdapter
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.utils import write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an explicit real-model CPU smoke gate.")
    parser.add_argument("model", choices=("melotts", "cosyvoice"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    if args.speed <= 0:
        raise ValueError("speed must be positive")

    adapter = MeloTTSAdapter() if args.model == "melotts" else CosyVoiceTTSAdapter()
    try:
        adapter.load_model(args.model_path, device="cpu", num_threads=args.threads)
        voices = tuple(getattr(adapter, "available_voices", ()))
        voice = args.voice or (voices[0] if voices else None)
        kwargs: dict[str, object] = {"speed": args.speed}
        if voice is not None:
            kwargs["voice"] = voice
        output, metrics = MetricsCollector().measure_inference(adapter, args.text, **kwargs)
        output_path = Path(args.output)
        write_wav(output_path, output.audio, output.sample_rate)
        report: dict[str, object] = {
            "model": args.model,
            "model_path": str(Path(args.model_path).resolve()),
            "voice": voice,
            "threads": args.threads,
            "speed": args.speed,
            "sample_rate": output.sample_rate,
            "samples": int(output.audio.size),
            "metrics": metrics.to_dict(),
            "metadata": output.metadata,
            "output": str(output_path.resolve()),
        }
        if args.report:
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return report
    finally:
        adapter.unload_model()


def main() -> int:
    args = build_parser().parse_args()
    run_gate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
