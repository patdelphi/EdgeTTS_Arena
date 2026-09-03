from edgetts_arena.ui.presenter import (
    capability_view,
    comparison_rows,
    format_result_card,
    model_choices,
    status_rows,
)


def _models():
    return [
        {
            "id": "a",
            "name": "Model A",
            "status": "unloaded",
            "experimental": False,
            "voices": ["default", "alt"],
            "capabilities": {
                "streaming": True,
                "seed": True,
                "speed": True,
                "voices": True,
                "voice_clone": False,
                "language_control": True,
                "languages": ["en"],
            },
        },
        {
            "id": "b",
            "name": "Model B",
            "status": "unloaded",
            "experimental": False,
            "voices": ["voice-b"],
            "capabilities": {
                "streaming": False,
                "seed": False,
                "speed": True,
                "voices": True,
                "voice_clone": False,
                "language_control": False,
                "languages": ["en-gb"],
            },
        },
    ]


def test_capability_view_is_selection_driven() -> None:
    single = capability_view(_models(), ["a"])
    assert single["speed_enabled"] is True
    assert single["seed_enabled"] is True
    assert single["voice_enabled"] is True
    assert single["voices"] == ["default", "alt"]
    assert single["language_enabled"] is True
    assert single["languages"] == ["en"]
    assert single["streaming_enabled"] is True

    multi = capability_view(_models(), ["a", "b"])
    assert multi["speed_enabled"] is True
    assert multi["seed_enabled"] is True
    assert multi["seed_partial"] is True
    assert multi["voice_enabled"] is False
    assert multi["language_enabled"] is False
    assert multi["languages"] == []
    assert multi["streaming_enabled"] is False


def test_model_status_and_choices_include_runtime_state() -> None:
    choices = model_choices(_models())
    assert choices[0][1] == "a"
    assert "unloaded" in choices[0][0]
    rows = status_rows(_models())
    assert rows[0][0] == "Model A"
    assert "streaming" in rows[0][3]
    assert "language_control" in rows[0][3]


def test_result_presenter_shows_non_streaming_ttfb_as_na() -> None:
    result = {
        "model_id": "a",
        "status": "success",
        "metrics": {
            "inference_time_ms": 100.0,
            "audio_duration_ms": 1000.0,
            "rtf": 0.1,
            "peak_rss_mb": 200.0,
            "rss_delta_mb": 20.0,
            "avg_cpu_usage_pct": 50.0,
            "ttfb_ms": None,
        },
        "warnings": [],
        "error": None,
    }
    card = format_result_card(result, {"a": "Model A"})
    assert "N/A (non-streaming)" in card
    rows = comparison_rows([result], {"a": "Model A"})
    assert rows[0][8] == "N/A"


def test_suite_result_rows_uses_aggregate_statistics() -> None:
    from edgetts_arena.ui.presenter import suite_result_rows

    rows = suite_result_rows(
        [
            {
                "case_id": "TC-01",
                "model_id": "dummy",
                "status": "success",
                "successful_runs": 3,
                "measured_runs": 3,
                "aggregate": {
                    "inference_time_ms": {"mean": 10.0},
                    "rtf": {"mean": 0.2, "p95": 0.25},
                    "peak_rss_mb": {"mean": 100.0},
                    "avg_cpu_usage_pct": {"mean": 35.0},
                },
            }
        ],
        {"dummy": "Dummy"},
    )
    assert rows == [["TC-01", "Dummy", "success", "3/3", 10.0, 0.2, 0.25, 100.0, 35.0]]
