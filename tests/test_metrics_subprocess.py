from __future__ import annotations

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.metrics_collector import MetricsCollector


class SubprocessMetricsAdapter(BaseTTSAdapter):
    id = "subprocess-metrics"
    capabilities = TTSCapabilities()

    def load_model(self, model_path: str, *, device: str = "cpu", num_threads: int = 4) -> None:
        return None

    def infer(self, text: str, *, voice: str | None = None, speed: float = 1.0, **kwargs):
        return TTSOutput(
            audio=np.ones(240, dtype=np.float32) * 0.01,
            sample_rate=24000,
            metadata={
                "subprocess_peak_rss_mb": 321.5,
                "subprocess_avg_cpu_usage_pct": 175.0,
            },
        )

    def unload_model(self) -> None:
        return None


def test_metrics_include_declared_native_child_resources() -> None:
    _, metrics = MetricsCollector(sample_interval_sec=0.001).measure_inference(
        SubprocessMetricsAdapter(), "test"
    )
    assert metrics.peak_rss_mb >= 321.5
    assert metrics.rss_delta_mb >= 321.5
    assert metrics.avg_cpu_usage_pct >= 175.0
