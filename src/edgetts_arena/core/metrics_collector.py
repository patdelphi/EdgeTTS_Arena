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
    """Collect wall, RSS and CPU metrics, including declared native child work."""

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

        parent_peak_mb = peak_rss / (1024 * 1024)
        parent_delta_mb = max(0, peak_rss - baseline_rss) / (1024 * 1024)
        parent_cpu_pct = (cpu_seconds / elapsed) * 100.0
        child_peak_mb = self._metadata_metric(output, "subprocess_peak_rss_mb") or 0.0
        child_cpu_pct = self._metadata_metric(output, "subprocess_avg_cpu_usage_pct") or 0.0

        metrics = InferenceMetrics(
            inference_time_ms=elapsed * 1000.0,
            audio_duration_ms=audio_duration_sec * 1000.0,
            rtf=elapsed / max(audio_duration_sec, 1e-9),
            # A subprocess-backed adapter keeps the Python worker alive while
            # the native child runs, so their resident sets are additive.
            peak_rss_mb=parent_peak_mb + child_peak_mb,
            rss_delta_mb=parent_delta_mb + child_peak_mb,
            avg_cpu_usage_pct=parent_cpu_pct + child_cpu_pct,
            ttfb_ms=None,
        )
        return output, metrics

    @staticmethod
    def _metadata_metric(output: TTSOutput, key: str) -> float | None:
        value = output.metadata.get(key)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number >= 0 else None
