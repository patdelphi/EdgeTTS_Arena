from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from edgetts_arena.adapters.cosyvoice_adapter import CosyVoiceTTSAdapter
from edgetts_arena.adapters.melotts_adapter import MeloTTSAdapter
from edgetts_arena.adapters.qwen3_adapter import Qwen3TTSAdapter
from edgetts_arena.core.metrics_collector import MetricsCollector
from edgetts_arena.utils import write_wav


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an explicit real-model CPU smoke gate.")
    parser.add_argument("model", choices=("melotts", "cosyvoice", "qwen3"))
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--voice")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report")
    return parser


def _adapter(model: str):
    if model == "melotts":
        return MeloTTSAdapter()
    if model == "cosyvoice":
        return CosyVoiceTTSAdapter()
    if model == "qwen3":
        return Qwen3TTSAdapter()
    raise ValueError(f"unknown model gate: {model}")


def run_gate(args: argparse.Namespace) -> dict[str, object]:
    if args.threads < 1:
        raise ValueError("threads must be >= 1")
    if args.speed <= 0:
        raise ValueError("speed must be positive")

    adapter = _adapter(args.model)
    try:
        adapter.load_model(args.model_path, device="cpu", num_threads=args.threads)
        voices = tuple(getattr(adapter, "available_voices", ()))
        voice = args.voice or (voices[0] if voices else None)
        kwargs: dict[str, object] = {"speed": args.speed}
        if voice is not None:
            kwargs["voice"] = voice
        output, metrics = MetricsCollector().measure_inference(adapter, args.text, **kwargs)
        audio = np.asarray(output.audio, dtype=np.float32).reshape(-1)
        duration_sec = float(audio.size / output.sample_rate)
        peak_abs = float(np.max(np.abs(audio)))
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        if not np.isfinite(audio).all():
            raise RuntimeError("real-model gate produced non-finite audio")
        if duration_sec < 0.1:
            raise RuntimeError(f"real-model gate produced implausibly short audio: {duration_sec:.4f}s")
        if peak_abs < 1e-4 or rms < 1e-5:
            raise RuntimeError(
                f"real-model gate produced silent/near-silent audio: peak={peak_abs:.6g}, rms={rms:.6g}"
            )

        output_path = Path(args.output)
        write_wav(output_path, audio, output.sample_rate)
        report: dict[str, object] = {
            "model": args.model,
            "model_path": str(Path(args.model_path).resolve()),
            "voice": voice,
            "threads": args.threads,
            "speed": args.speed,
            "sample_rate": output.sample_rate,
            "samples": int(audio.size),
            "audio_sanity": {
                "duration_sec": duration_sec,
                "peak_abs": peak_abs,
                "rms": rms,
            },
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
