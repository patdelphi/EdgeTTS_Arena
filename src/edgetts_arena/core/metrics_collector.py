from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput


@dataclass(frozen=True, slots=True)
class InferenceMetrics:
    inference_time_ms: float
    audio_duration_ms: float
    rtf: float
    peak_rss_mb: float
    rss_delta_mb: float
    avg_cpu_usage_pct: float
    ttfb_ms: float | None = None

    def to_dict(self) -> dict[str, float | None]:
        return asdict(self)


class MetricsCollector:
    """Collects adapter-independent wall, RSS and process CPU metrics."""

    def __init__(self, *, sample_interval_sec: float = 0.01) -> None:
        self.sample_interval_sec = max(0.001, sample_interval_sec)

    def measure_inference(
        self,
        adapter: BaseTTSAdapter,
        text: str,
        **kwargs: Any,
    ) -> tuple[TTSOutput, InferenceMetrics]:
        process = psutil.Process(os.getpid())
        baseline_rss = process.memory_info().rss
        peak_rss = baseline_rss
        stop = threading.Event()

        def sample_memory() -> None:
            nonlocal peak_rss
            while not stop.wait(self.sample_interval_sec):
                try:
                    peak_rss = max(peak_rss, process.memory_info().rss)
                except psutil.Error:
                    return

        sampler = threading.Thread(target=sample_memory, daemon=True)
        cpu_before = process.cpu_times()
        started = time.perf_counter()
        sampler.start()
        try:
            output = adapter.infer(text, **kwargs)
        finally:
            elapsed = max(time.perf_counter() - started, 1e-9)
            stop.set()
            sampler.join(timeout=self.sample_interval_sec * 3)

        final_rss = process.memory_info().rss
        peak_rss = max(peak_rss, final_rss)
        cpu_after = process.cpu_times()
        cpu_seconds = max(
            0.0,
            (cpu_after.user + cpu_after.system) - (cpu_before.user + cpu_before.system),
        )
        audio_duration_sec = output.audio.size / output.sample_rate
        metrics = InferenceMetrics(
            inference_time_ms=elapsed * 1000.0,
            audio_duration_ms=audio_duration_sec * 1000.0,
            rtf=elapsed / max(audio_duration_sec, 1e-9),
            peak_rss_mb=peak_rss / (1024 * 1024),
            rss_delta_mb=max(0, peak_rss - baseline_rss) / (1024 * 1024),
            avg_cpu_usage_pct=(cpu_seconds / elapsed) * 100.0,
            ttfb_ms=None,
        )
        return output, metrics
