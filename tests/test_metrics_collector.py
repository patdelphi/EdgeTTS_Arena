import math

from edgetts_arena.adapters import DummyTTSAdapter
from edgetts_arena.core import MetricsCollector


def test_metrics_collector_non_streaming_metrics() -> None:
    adapter = DummyTTSAdapter()
    adapter.load_model()
    output, metrics = MetricsCollector(sample_interval_sec=0.001).measure_inference(
        adapter,
        "metrics test",
        seed=2,
    )
    assert output.audio.size > 0
    assert metrics.inference_time_ms > 0
    assert metrics.audio_duration_ms > 0
    assert math.isfinite(metrics.rtf)
    assert metrics.rtf >= 0
    assert metrics.peak_rss_mb > 0
    assert metrics.rss_delta_mb >= 0
    assert metrics.avg_cpu_usage_pct >= 0
    assert metrics.ttfb_ms is None
