from __future__ import annotations

from typing import Any


def model_choices(models: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for model in models:
        status = str(model.get("status") or "unknown")
        experimental = " · experimental" if model.get("experimental") else ""
        label = f"{model.get('name', model.get('id'))} · {status}{experimental}"
        choices.append((label, str(model["id"])))
    return choices


def usable_model_ids(models: list[dict[str, Any]]) -> list[str]:
    return [str(item["id"]) for item in models if item.get("status") != "unavailable"]


def status_rows(models: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for model in models:
        capabilities = model.get("capabilities") or {}
        feature_names = [
            name
            for name in ("streaming", "seed", "speed", "voices", "voice_clone")
            if capabilities.get(name)
        ]
        languages = ", ".join(capabilities.get("languages") or []) or "—"
        rows.append(
            [
                model.get("name", model.get("id")),
                model.get("status", "unknown"),
                "yes" if model.get("experimental") else "no",
                ", ".join(feature_names) or "—",
                languages,
            ]
        )
    return rows


def capability_view(
    models: list[dict[str, Any]], selected_ids: list[str] | None
) -> dict[str, Any]:
    selected_ids = list(selected_ids or [])
    by_id = {str(item["id"]): item for item in models}
    selected = [by_id[item] for item in selected_ids if item in by_id]
    if not selected:
        return {
            "speed_enabled": False,
            "seed_enabled": False,
            "seed_partial": False,
            "voice_enabled": False,
            "voices": [],
            "streaming_enabled": False,
            "summary": "Select at least one model.",
        }

    caps = [item.get("capabilities") or {} for item in selected]
    speed_enabled = all(bool(cap.get("speed")) for cap in caps)
    seed_count = sum(bool(cap.get("seed")) for cap in caps)
    seed_enabled = seed_count > 0
    seed_partial = 0 < seed_count < len(selected)
    streaming_enabled = len(selected) == 1 and bool(caps[0].get("streaming"))

    voices: list[str] = []
    voice_enabled = False
    if len(selected) == 1 and bool(caps[0].get("voices")):
        voices = [str(value) for value in selected[0].get("voices") or []]
        voice_enabled = bool(voices)

    unavailable = [str(item["id"]) for item in selected if item.get("status") == "unavailable"]
    lines = [
        f"Selected: **{len(selected)}** model(s)",
        f"Speed control: **{'enabled' if speed_enabled else 'disabled'}**",
        "Seed: **partial support**" if seed_partial else f"Seed: **{'enabled' if seed_enabled else 'disabled'}**",
        f"Streaming preview capable: **{'yes' if streaming_enabled else 'no'}**",
    ]
    if len(selected) > 1:
        lines.append("Voice selection uses each model's default in multi-model Arena mode.")
    if unavailable:
        lines.append("Unavailable: " + ", ".join(unavailable))
    return {
        "speed_enabled": speed_enabled,
        "seed_enabled": seed_enabled,
        "seed_partial": seed_partial,
        "voice_enabled": voice_enabled,
        "voices": voices,
        "streaming_enabled": streaming_enabled,
        "summary": "  \n".join(lines),
    }


def format_result_card(
    result: dict[str, Any] | None,
    model_names: dict[str, str] | None = None,
) -> str:
    if not result:
        return "_No result_"
    model_id = str(result.get("model_id") or "unknown")
    display = (model_names or {}).get(model_id, model_id)
    if result.get("status") != "success":
        error = result.get("error") or {}
        return (
            f"### {display}\n"
            f"**Status:** error  \n"
            f"**{error.get('type', 'error')} ({error.get('code', '—')}):** "
            f"{error.get('message', 'Unknown model error')}"
        )

    metrics = result.get("metrics") or {}
    ttfb = metrics.get("ttfb_ms")
    ttfb_text = "N/A (non-streaming)" if ttfb is None else f"{float(ttfb):.1f} ms"
    warnings = result.get("warnings") or []
    warning_text = "<br>".join(str(item) for item in warnings) if warnings else "—"
    return (
        f"### {display}\n"
        f"**Status:** success  \n"
        f"Inference: **{_fmt(metrics.get('inference_time_ms'))} ms** · "
        f"Audio: **{_fmt(metrics.get('audio_duration_ms'))} ms** · "
        f"RTF: **{_fmt(metrics.get('rtf'), 3)}**  \n"
        f"Peak RSS: **{_fmt(metrics.get('peak_rss_mb'))} MB** · "
        f"RSS Δ: **{_fmt(metrics.get('rss_delta_mb'))} MB** · "
        f"CPU: **{_fmt(metrics.get('avg_cpu_usage_pct'))}%**  \n"
        f"TTFB: **{ttfb_text}**  \n"
        f"Warnings: {warning_text}"
    )


def comparison_rows(
    results: list[dict[str, Any]],
    model_names: dict[str, str] | None = None,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for result in results:
        model_id = str(result.get("model_id") or "unknown")
        display = (model_names or {}).get(model_id, model_id)
        metrics = result.get("metrics") or {}
        error = result.get("error") or {}
        ttfb = metrics.get("ttfb_ms")
        rows.append(
            [
                display,
                result.get("status", "unknown"),
                _number(metrics.get("inference_time_ms")),
                _number(metrics.get("audio_duration_ms")),
                _number(metrics.get("rtf")),
                _number(metrics.get("peak_rss_mb")),
                _number(metrics.get("rss_delta_mb")),
                _number(metrics.get("avg_cpu_usage_pct")),
                "N/A" if ttfb is None else _number(ttfb),
                error.get("message", "") if result.get("status") != "success" else "",
            ]
        )
    return rows


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None
