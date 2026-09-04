from __future__ import annotations

from typing import Any


def model_choices(models: list[dict[str, Any]]) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for model in models:
        # Models explicitly disabled in config (e.g. the native-quantized Qwen3
        # variants that cannot be bootstrapped on this platform) are hidden from
        # the picker entirely instead of showing as a dead "unavailable" row.
        if model.get("enabled") is False:
            continue
        status = str(model.get("status") or "unknown")
        experimental = " · 实验性" if model.get("experimental") else ""
        label = f"{model.get('name', model.get('id'))} · {status}{experimental}"
        choices.append((label, str(model["id"])))
    return choices


def usable_model_ids(models: list[dict[str, Any]]) -> list[str]:
    return [
        str(item["id"])
        for item in models
        if item.get("status") != "unavailable" and item.get("enabled") is not False
    ]


def status_rows(models: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for model in models:
        if model.get("enabled") is False:
            continue
        capabilities = model.get("capabilities") or {}
        feature_names = [
            name
            for name in ("streaming", "seed", "speed", "voices", "voice_clone", "language_control")
            if capabilities.get(name)
        ]
        languages = ", ".join(capabilities.get("languages") or []) or "—"
        rows.append(
            [
                model.get("name", model.get("id")),
                model.get("status", "unknown"),
                "是" if model.get("experimental") else "否",
                ", ".join(feature_names) or "—",
                languages,
            ]
        )
    return rows


def _common_values(groups: list[list[str]]) -> list[str]:
    if not groups or any(not group for group in groups):
        return []
    common = set(groups[0])
    for group in groups[1:]:
        common.intersection_update(group)
    return [value for value in groups[0] if value in common]


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
            "language_enabled": False,
            "languages": [],
            "streaming_enabled": False,
            "summary": "请至少选择一个模型。",
        }

    caps = [item.get("capabilities") or {} for item in selected]
    speed_enabled = all(bool(cap.get("speed")) for cap in caps)
    seed_count = sum(bool(cap.get("seed")) for cap in caps)
    seed_enabled = seed_count > 0
    seed_partial = 0 < seed_count < len(selected)
    streaming_enabled = len(selected) == 1 and bool(caps[0].get("streaming"))

    voices: list[str] = []
    if all(bool(cap.get("voices")) for cap in caps):
        voices = _common_values(
            [[str(value) for value in item.get("voices") or []] for item in selected]
        )
    voice_enabled = bool(voices)

    languages: list[str] = []
    if all(bool(cap.get("language_control")) for cap in caps):
        languages = _common_values(
            [[str(value) for value in cap.get("languages") or []] for cap in caps]
        )
    language_enabled = bool(languages)

    unavailable = [str(item["id"]) for item in selected if item.get("status") == "unavailable"]
    lines = [
        f"已选择: **{len(selected)}** 个模型",
        f"语速控制: **{'已启用' if speed_enabled else '已禁用'}**",
        "随机种子: **部分支持**" if seed_partial else f"随机种子: **{'已启用' if seed_enabled else '已禁用'}**",
        f"共享音色控制: **{'已启用' if voice_enabled else '已禁用'}**",
        f"共享语言控制: **{'已启用' if language_enabled else '已禁用'}**",
        f"流式预览能力: **{'是' if streaming_enabled else '否'}**",
    ]
    if len(selected) > 1 and not voice_enabled:
        lines.append("无共同可选音色，每个模型将使用默认音色。")
    if len(selected) > 1 and not language_enabled:
        lines.append("无共同明确语言，每个模型将使用默认语言。")
    if unavailable:
        lines.append("不可用: " + ", ".join(unavailable))
    return {
        "speed_enabled": speed_enabled,
        "seed_enabled": seed_enabled,
        "seed_partial": seed_partial,
        "voice_enabled": voice_enabled,
        "voices": voices,
        "language_enabled": language_enabled,
        "languages": languages,
        "streaming_enabled": streaming_enabled,
        "summary": "  \n".join(lines),
    }


def format_result_card(
    result: dict[str, Any] | None,
    model_names: dict[str, str] | None = None,
) -> str:
    if not result:
        return "_暂无结果_"
    model_id = str(result.get("model_id") or "unknown")
    display = (model_names or {}).get(model_id, model_id)
    if result.get("status") != "success":
        error = result.get("error") or {}
        return (
            f"### {display}\n"
            f"**状态:** ❌ 错误  \n"
            f"**{error.get('type', 'error')} ({error.get('code', '—')}):** "
            f"{error.get('message', '未知模型错误')}"
        )

    metrics = result.get("metrics") or {}
    config = result.get("config") or {}
    ttfb = metrics.get("ttfb_ms")
    ttfb_text = "不适用 (非流式)" if ttfb is None else f"{float(ttfb):.1f} ms"
    warnings = result.get("warnings") or []
    warning_text = "<br>".join(str(item) for item in warnings) if warnings else "—"
    
    # 获取调用配置信息
    voice = config.get("voice") or "默认"
    language = config.get("language") or "默认"
    speed = config.get("speed") or 1.0
    seed = config.get("seed")
    seed_text = str(seed) if seed is not None else "无"
    threads = config.get("num_threads") or "—"
    sample_rate = metrics.get("sample_rate") or "—"
    audio_samples = metrics.get("audio_samples") or "—"
    execution_mode = result.get("execution_mode") or "—"
    
    # 获取模型运行时长信息（顺序执行时）
    model_duration_sec = result.get("model_duration_sec")
    model_started_at = result.get("model_started_at")
    model_completed_at = result.get("model_completed_at")
    
    duration_text = ""
    if model_duration_sec is not None:
        duration_text = (
            f"---\n"
            f"**运行时间:**\n"
            f"- 总耗时: **{model_duration_sec:.2f} 秒**\n"
            f"- 开始时间: `{model_started_at}`\n"
            f"- 完成时间: `{model_completed_at}`\n"
        )
    
    return (
        f"### {display}\n"
        f"**状态:** ✅ 成功  \n"
        f"---\n"
        f"**调用参数:**\n"
        f"- 音色: `{voice}`\n"
        f"- 语言: `{language}`\n"
        f"- 语速: `{speed}x`\n"
        f"- 随机种子: `{seed_text}`\n"
        f"- 线程数: `{threads}`\n"
        f"---\n"
        f"**性能指标:**\n"
        f"- 推理耗时: **{_fmt(metrics.get('inference_time_ms'))} ms**\n"
        f"- 音频时长: **{_fmt(metrics.get('audio_duration_ms'))} ms**\n"
        f"- 实时率(RTF): **{_fmt(metrics.get('rtf'), 3)}**\n"
        f"- 首包延迟(TTFB): **{ttfb_text}**\n"
        f"---\n"
        f"**资源占用:**\n"
        f"- 峰值内存: **{_fmt(metrics.get('peak_rss_mb'))} MB**\n"
        f"- 内存增量: **{_fmt(metrics.get('rss_delta_mb'))} MB**\n"
        f"- CPU使用率: **{_fmt(metrics.get('avg_cpu_usage_pct'))}%**\n"
        f"---\n"
        f"**音频信息:**\n"
        f"- 采样率: `{sample_rate} Hz`\n"
        f"- 采样数: `{audio_samples}`\n"
        f"- 执行模式: `{execution_mode}`\n"
        f"{duration_text}"
        f"---\n"
        f"**警告:** {warning_text}"
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
                "不适用" if ttfb is None else _number(ttfb),
                error.get("message", "") if result.get("status") != "success" else "",
            ]
        )
    return rows


def _fmt(value: Any, digits: int = 1) -> str:
    if value is None:
        return "不适用"
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


def suite_result_rows(
    results: list[dict[str, Any]],
    model_names: dict[str, str] | None = None,
) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for result in results:
        model_id = str(result.get("model_id") or "unknown")
        display = (model_names or {}).get(model_id, model_id)
        aggregate = result.get("aggregate") or {}
        inference = aggregate.get("inference_time_ms") or {}
        rtf = aggregate.get("rtf") or {}
        rss = aggregate.get("peak_rss_mb") or {}
        cpu = aggregate.get("avg_cpu_usage_pct") or {}
        rows.append(
            [
                result.get("case_id", "unknown"),
                display,
                result.get("status", "unknown"),
                f"{result.get('successful_runs', 0)}/{result.get('measured_runs', 0)}",
                _number(inference.get("mean")),
                _number(rtf.get("mean")),
                _number(rtf.get("p95")),
                _number(rss.get("mean")),
                _number(cpu.get("mean")),
            ]
        )
    return rows
