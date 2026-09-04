from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from fastapi.responses import RedirectResponse

from edgetts_arena.api.app import create_app
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.benchmark_suite import BenchmarkPresetSuite, RepeatedBenchmarkService
from edgetts_arena.core.blind_ab import (
    blind_session_complete,
    create_blind_session,
    record_blind_rating,
    reveal_blind_session,
)
from edgetts_arena.core.model_registry import ModelRegistry
from edgetts_arena.core.system_info import collect_system_environment
from edgetts_arena.ui.presenter import (
    capability_view,
    comparison_rows,
    format_result_card,
    model_choices,
    status_rows,
    suite_result_rows,
    usable_model_ids,
)

# 表头定义
STATUS_HEADERS = ["模型", "状态", "实验性", "能力", "语言"]
COMPARISON_HEADERS = [
    "模型", "状态", "推理毫秒", "音频毫秒", "RTF", "峰值RSS MB",
    "RSS增量 MB", "CPU %", "TTFB毫秒", "错误",
]
SUITE_HEADERS = [
    "用例", "模型", "状态", "成功", "推理平均毫秒", "RTF平均",
    "RTF P95", "峰值RSS平均 MB", "CPU平均 %",
]


def build_arena_ui(
    registry: ModelRegistry,
    benchmark_service: BenchmarkService,
    artifact_store: RunArtifactStore,
    repeated_benchmark_service: RepeatedBenchmarkService | None = None,
) -> gr.Blocks:
    from edgetts_arena.core.config import load_settings
    settings = load_settings()
    
    suite_service = repeated_benchmark_service or RepeatedBenchmarkService(
        registry,
        benchmark_service.resource_guard,
        artifact_store,
        preset_suite=BenchmarkPresetSuite.load(),
    )
    presets = suite_service.preset_suite
    initial_models = registry.list_models()
    default_models = usable_model_ids(initial_models)[:2] or usable_model_ids(initial_models)[:1]
    initial_caps = capability_view(initial_models, default_models)

    with gr.Blocks(title="EdgeTTS-Arena") as demo:
        model_state = gr.State(initial_models)
        run_state = gr.State(None)
        blind_state = gr.State(None)

        gr.Markdown("# EdgeTTS-Arena\nCPU优先的本地TTS对比平台与可复现基准测试套件。")
        with gr.Row():
            system = gr.Markdown(_system_markdown())
            refresh = gr.Button("刷新模型状态")
        status = gr.Dataframe(
            status_rows(initial_models), headers=STATUS_HEADERS, type="array",
            interactive=False, label="运行时模型状态",
        )

        with gr.Tab("Arena"):
            with gr.Row():
                with gr.Column(scale=1, min_width=320):
                    preset = gr.Dropdown(
                        [("自定义", "__custom__")] + [(f"{c.id} · {c.name}", c.id) for c in presets.cases],
                        value="__custom__", label="预设",
                    )
                    text = gr.Textbox(
                        "EdgeTTS-Arena 在相同CPU上对比本地TTS模型的表现。",
                        label="文本", lines=6,
                    )
                    models = gr.Dropdown(
                        model_choices(initial_models), value=default_models, multiselect=True,
                        max_choices=4, label="模型 (1–4)",
                    )
                    mode = gr.Radio(
                        [("顺序执行", "sequential"), ("并发执行", "concurrent")],
                        value="sequential", label="执行模式",
                    )
                with gr.Column(scale=1, min_width=320):
                    threads = gr.Slider(1, 16, value=4, step=1, label="CPU线程数/模型")
                    with gr.Row():
                        speed = gr.Slider(
                            .5, 2.0, value=1.0, step=.05, label="语速",
                            interactive=initial_caps["speed_enabled"],
                        )
                        seed = gr.Number(
                            value=None, precision=0, label="随机种子",
                            interactive=initial_caps["seed_enabled"],
                        )
                    voice = gr.Dropdown(
                        initial_caps["voices"], value=None, label="音色",
                        interactive=initial_caps["voice_enabled"],
                    )
                    language = gr.Dropdown(
                        initial_caps["languages"], value=None, label="语言",
                        interactive=initial_caps["language_enabled"],
                    )
                    cap_summary = gr.Markdown(initial_caps["summary"])
            run_btn = gr.Button("运行基准测试", variant="primary")

            run_summary = gr.Markdown("_尚未运行基准测试。_")
            audios: list[gr.Audio] = []
            cards: list[gr.Markdown] = []
            for row in range(2):
                with gr.Row(equal_height=False):
                    for col in range(2):
                        idx = row * 2 + col
                        with gr.Column(scale=1):
                            audios.append(gr.Audio(type="filepath", label=f"结果 {idx + 1}", interactive=False))
                            cards.append(gr.Markdown("_暂无结果_", elem_classes=["result-card"]))
            comparison = gr.Dataframe(
                [], headers=COMPARISON_HEADERS, type="array", interactive=False, label="对比",
            )
            export_file = gr.File(label="运行结果导出 ZIP", interactive=False)

        with gr.Tab("标准测试套件"):
            gr.Markdown(
                "TC-01~TC-05 顺序执行。预热测量数据将被丢弃；原始测量运行和汇总统计数据共享同一个 run_id。"
            )
            suite_cases = gr.Dropdown(
                [(f"{c.id} · {c.name}", c.id) for c in presets.cases],
                value=[c.id for c in presets.cases], multiselect=True, max_choices=5, label="测试用例",
            )
            suite_models = gr.Dropdown(
                model_choices(initial_models), value=default_models, multiselect=True,
                max_choices=4, label="模型",
            )
            with gr.Row():
                warmups = gr.Slider(0, 5, value=presets.warmup_runs, step=1, label="预热次数")
                repeats = gr.Slider(1, 10, value=presets.measured_runs, step=1, label="测量次数")
                suite_threads = gr.Slider(1, 16, value=4, step=1, label="CPU线程数/模型")
            suite_btn = gr.Button("运行标准测试套件", variant="primary")
            suite_summary = gr.Markdown("_尚未运行标准测试套件。_")
            suite_table = gr.Dataframe(
                [], headers=SUITE_HEADERS, type="array", interactive=False,
                label="重复基准测试汇总",
            )
            suite_export = gr.File(label="测试结果导出 ZIP", interactive=False)

        with gr.Tab("盲测AB"):
            gr.Markdown("在 Arena 运行至少两个模型成功后开始盲测。")
            start_blind = gr.Button("开始盲测AB", interactive=False)
            blind_audio: list[gr.Audio] = []
            with gr.Row():
                for i in range(4):
                    blind_audio.append(
                        gr.Audio(type="filepath", label=f"样本 {chr(65+i)}", visible=False, interactive=False)
                    )
            with gr.Row():
                score_sample = gr.Dropdown([], label="匿名样本")
                natural = gr.Slider(1, 5, value=3, step=1, label="自然度")
                intelligible = gr.Slider(1, 5, value=3, step=1, label="清晰度")
                prosody = gr.Slider(1, 5, value=3, step=1, label="韵律")
            save_score = gr.Button("保存样本评分")
            blind_progress = gr.Markdown("_盲测会话未开始。_")
            reveal = gr.Button("揭晓模型", interactive=False)
            reveal_output = gr.Markdown("")

        with gr.Tab("模型管理"):
            gr.Markdown("## 📦 模型下载管理\n\n模型优先存储在 Hugging Face Hub 缓存目录，其次使用项目目录。")
            from edgetts_arena.core.model_downloader import list_downloadable_models, check_model_downloaded
            from pathlib import Path as _Path

            _project_root = _Path.cwd()
            _search_paths = settings.model_search_paths if hasattr(settings, 'model_search_paths') else (
                "${HF_HOME:-~/.cache}/huggingface/hub",
                "./models",
            )

            downloadable = list_downloadable_models()

            # 模型下载状态表格
            def _get_download_status():
                rows = []
                for m in downloadable:
                    downloaded = check_model_downloaded(m["id"], _search_paths, _project_root)
                    status = "✅ 已下载" if downloaded else "⬜ 未下载"
                    rows.append([m["id"], m["description"], f"~{m['size_mb']}MB", m["repo_id"], status])
                return rows

            with gr.Row():
                with gr.Column(scale=3):
                    download_table = gr.Dataframe(
                        _get_download_status(),
                        headers=["模型ID", "描述", "大小", "来源", "状态"],
                        type="array",
                        interactive=False,
                        label="可下载模型列表",
                        wrap=True,
                    )
                    refresh_status_btn = gr.Button("🔄 刷新状态", size="sm")
                
                with gr.Column(scale=2):
                    gr.Markdown("### 下载操作")
                    download_model_input = gr.Dropdown(
                        [m["id"] for m in downloadable],
                        label="选择要下载的模型",
                        info="从上方列表中选择模型ID",
                    )
                    with gr.Row():
                        download_btn = gr.Button("⬇️ 下载选中模型", variant="primary", size="lg")
                    with gr.Row():
                        download_all_btn = gr.Button("📥 下载全部模型", size="lg")
                    
                    gr.Markdown("---")
                    gr.Markdown(
                        f"**📁 搜索路径优先级:**\n\n"
                        + "\n".join(f"{i+1}. `{p}`" for i, p in enumerate(_search_paths))
                    )

            # 下载状态输出 - 使用更大的区域
            gr.Markdown("### 📊 下载状态")
            download_output = gr.Markdown(
                "_请选择要下载的模型，然后点击下载按钮。_\n\n"
                "> 💡 **提示**: 首次下载可能需要较长时间，请耐心等待。",
                elem_classes=["download-status-box"],
            )

        def refresh_runtime() -> tuple[Any, ...]:
            current = registry.list_models()
            defaults = usable_model_ids(current)[:2] or usable_model_ids(current)[:1]
            caps = capability_view(current, defaults)
            choices = model_choices(current)
            return (
                current,
                gr.update(choices=choices, value=defaults),
                gr.update(choices=choices, value=defaults),
                status_rows(current),
                _system_markdown(),
                gr.update(interactive=caps["speed_enabled"], value=1.0),
                gr.update(interactive=caps["seed_enabled"], value=None),
                gr.update(choices=caps["voices"], value=None, interactive=caps["voice_enabled"]),
                gr.update(
                    choices=caps["languages"], value=None,
                    interactive=caps["language_enabled"],
                ),
                caps["summary"],
            )

        def selected_changed(ids: list[str] | None, current: list[dict[str, Any]]) -> tuple[Any, ...]:
            caps = capability_view(current or registry.list_models(), ids)
            return (
                gr.update(interactive=caps["speed_enabled"], value=1.0),
                gr.update(interactive=caps["seed_enabled"], value=None),
                gr.update(choices=caps["voices"], value=None, interactive=caps["voice_enabled"]),
                gr.update(
                    choices=caps["languages"], value=None,
                    interactive=caps["language_enabled"],
                ),
                caps["summary"],
            )

        def preset_changed(case_id: str | None) -> Any:
            if not case_id or case_id == "__custom__":
                return gr.update()
            return gr.update(value=presets.select([case_id])[0].text)

        def run_arena(
            input_text: str, ids: list[str] | None, execution: str, thread_count: int | float,
            speed_value: float, seed_value: int | float | None, voice_value: str | None,
            language_value: str | None, current: list[dict[str, Any]],
        ) -> tuple[Any, ...]:
            selected = list(ids or [])
            if not selected:
                raise gr.Error("请至少选择一个模型。")
            data = benchmark_service.run(
                text=input_text, model_ids=selected, execution_mode=execution,
                cpu_threads_per_model=int(thread_count),
                config={
                    "speed": float(speed_value),
                    "seed": None if seed_value is None else int(seed_value),
                    "voice": voice_value or None,
                    "language": language_value or None,
                    "sample_rate": None,
                },
            )
            names = {str(x["id"]): str(x.get("name") or x["id"]) for x in (current or registry.list_models())}
            audio_values, card_values = [], []
            for result in data["results"][:4]:
                audio_values.append(_result_audio_path(result, data["run_id"], artifact_store))
                card_values.append(format_result_card(result, names))
            audio_values += [None] * (4 - len(audio_values))
            card_values += ["_暂无结果_"] * (4 - len(card_values))
            success = sum(r.get("status") == "success" for r in data["results"])
            zip_path = artifact_store.build_export(data["run_id"])
            summary = (
                f"### 运行 `{data['run_id']}`\n模式: **{data['execution_mode']}** · "
                f"每模型线程: **{data['cpu_threads_per_model']}** · 成功: **{success}/{len(data['results'])}**"
            )
            return (
                data, summary, str(zip_path), *audio_values, *card_values,
                comparison_rows(data["results"], names), gr.update(interactive=success >= 2),
                None, "_盲测会话未开始。_", "", gr.update(interactive=False),
            )

        def run_suite(
            case_ids: list[str] | None, ids: list[str] | None, warm: int | float,
            measured: int | float, thread_count: int | float, current: list[dict[str, Any]],
        ) -> tuple[str, list[list[Any]], str]:
            if not case_ids or not ids:
                raise gr.Error("请至少选择一个测试用例和一个模型。")
            data = suite_service.run_suite(
                model_ids=list(ids), case_ids=list(case_ids), warmup_runs=int(warm),
                measured_runs=int(measured), cpu_threads_per_model=int(thread_count),
                config={
                    "speed": 1.0, "voice": None, "language": None,
                    "seed": None, "sample_rate": None,
                },
            )
            names = {str(x["id"]): str(x.get("name") or x["id"]) for x in (current or registry.list_models())}
            success = sum(r.get("status") == "success" for r in data["results"])
            summary = (
                f"### 测试套件 `{data['run_id']}`\n用例: **{len(data['cases'])}** · 模型: **{len(data['models'])}** · "
                f"预热: **{data['warmup_runs']}** · 测量: **{data['measured_runs']}** · "
                f"成功配对: **{success}/{len(data['results'])}**"
            )
            return summary, suite_result_rows(data["results"], names), str(artifact_store.build_export(data["run_id"]))

        def begin_blind(data: dict[str, Any] | None) -> tuple[Any, ...]:
            if not data:
                raise gr.Error("请先运行 Arena 基准测试。")
            session = create_blind_session(data["run_id"], data["results"])
            by_model = {str(r["model_id"]): r for r in data["results"]}
            audio_updates = []
            for i in range(4):
                if i < len(session["samples"]):
                    sample = session["samples"][i]
                    path = _result_audio_path(by_model[sample["model_id"]], data["run_id"], artifact_store)
                    audio_updates.append(gr.update(value=path, label=f"样本 {sample['label']}", visible=True))
                else:
                    audio_updates.append(gr.update(value=None, visible=False))
            labels = [s["label"] for s in session["samples"]]
            return session, *audio_updates, gr.update(choices=labels, value=labels[0]), _blind_progress(session), "", gr.update(interactive=False)

        def score_blind(
            session: dict[str, Any] | None, label: str | None, n: int | float,
            i: int | float, p: int | float,
        ) -> tuple[Any, str, Any]:
            if not session or not label:
                raise gr.Error("请先开始盲测AB并选择一个样本。")
            updated = record_blind_rating(session, label, naturalness=n, intelligibility=i, prosody=p)
            return updated, _blind_progress(updated), gr.update(interactive=blind_session_complete(updated))

        def reveal_blind(session: dict[str, Any] | None) -> tuple[Any, str, str, Any]:
            if not session:
                raise gr.Error("请先开始盲测AB。")
            revealed = reveal_blind_session(session)
            artifact_store.write_json(revealed["run_id"], "blind_scores.json", revealed)
            ratings = revealed.get("ratings") or {}
            lines = ["### 揭晓结果"] + [
                f"- **样本 {s['label']} → {s['model_id']}** — 自然度 {ratings[s['label']]['naturalness']}/5, "
                f"清晰度 {ratings[s['label']]['intelligibility']}/5, 韵律 {ratings[s['label']]['prosody']}/5"
                for s in revealed["samples"]
            ]
            return revealed, "\n".join(lines), str(artifact_store.build_export(revealed["run_id"])), gr.update(interactive=False)

        # 模型下载处理函数
        def download_single_model(model_id: str | None) -> tuple[str, list[list[Any]]]:
            if not model_id:
                return "⚠️ **请先选择一个模型。**", _get_download_status()
            from edgetts_arena.core.model_downloader import download_model
            try:
                result = download_model(model_id, _search_paths, _project_root)
                if result["success"]:
                    msg = f"### ✅ 下载成功\n\n**模型:** `{model_id}`\n\n**路径:** `{result['path']}`\n\n{result['message']}"
                else:
                    msg = f"### ❌ 下载失败\n\n**模型:** `{model_id}`\n\n**错误:** {result.get('error', '未知错误')}"
            except Exception as exc:
                msg = f"### ❌ 下载出错\n\n**模型:** `{model_id}`\n\n**异常:** {exc}"
            return msg, _get_download_status()
        
        def download_all_models() -> tuple[str, list[list[Any]]]:
            from edgetts_arena.core.model_downloader import download_model
            results = []
            success_count = 0
            for m in downloadable:
                result = download_model(m["id"], _search_paths, _project_root)
                if result["success"]:
                    success_count += 1
                    results.append(f"- ✅ **{m['id']}**: {result['message']}")
                else:
                    results.append(f"- ❌ **{m['id']}**: {result.get('error', '失败')}")
            
            msg = f"### 📥 批量下载完成\n\n**成功:** {success_count}/{len(downloadable)}\n\n" + "\n".join(results)
            return msg, _get_download_status()
        
        def refresh_download_status() -> list[list[Any]]:
            return _get_download_status()

        refresh_outputs = [
            model_state, models, suite_models, status, system, speed, seed, voice, language, cap_summary,
        ]
        refresh.click(refresh_runtime, outputs=refresh_outputs)
        demo.load(refresh_runtime, outputs=refresh_outputs)
        models.change(
            selected_changed, [models, model_state], [speed, seed, voice, language, cap_summary]
        )
        preset.change(preset_changed, preset, text)
        run_btn.click(
            run_arena,
            [text, models, mode, threads, speed, seed, voice, language, model_state],
            [run_state, run_summary, export_file, *audios, *cards, comparison, start_blind,
             blind_state, blind_progress, reveal_output, reveal],
        )
        suite_btn.click(
            run_suite,
            [suite_cases, suite_models, warmups, repeats, suite_threads, model_state],
            [suite_summary, suite_table, suite_export],
        )
        start_blind.click(
            begin_blind, run_state,
            [blind_state, *blind_audio, score_sample, blind_progress, reveal_output, reveal],
        )
        save_score.click(
            score_blind, [blind_state, score_sample, natural, intelligible, prosody],
            [blind_state, blind_progress, reveal],
        )
        reveal.click(reveal_blind, blind_state, [blind_state, reveal_output, export_file, reveal])
        
        # 模型下载按钮事件 - 同时更新状态输出和表格
        download_btn.click(download_single_model, download_model_input, [download_output, download_table])
        download_all_btn.click(download_all_models, outputs=[download_output, download_table])
        refresh_status_btn.click(refresh_download_status, outputs=download_table)

    return demo


def create_full_app(*, exports_root: str | Path = "exports"):
    app = create_app(exports_root=exports_root)
    store = app.state.artifact_store
    demo = build_arena_ui(
        app.state.registry, app.state.benchmark_service, store, app.state.repeated_benchmark_service
    )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/arena/")

    return gr.mount_gradio_app(
        app, demo, path="/arena", allowed_paths=[str(store.root)], show_error=True
    )


def _result_audio_path(result: dict[str, Any], run_id: str, store: RunArtifactStore) -> str | None:
    if result.get("status") != "success" or not result.get("audio_url"):
        return None
    return str(store.get_audio_file(run_id, str(result["audio_url"]).rsplit("/", 1)[-1]))


def _system_markdown() -> str:
    env = collect_system_environment()
    return (
        f"**系统** — {env['os']} / {env['arch']} · 逻辑核心 {env['cpu_logical_cores']} · "
        f"内存 {env['available_ram_gb']:.1f}/{env['total_ram_gb']:.1f} GB 可用"
    )


def _blind_progress(session: dict[str, Any]) -> str:
    labels = [s["label"] for s in session.get("samples", [])]
    ratings = session.get("ratings") or {}
    missing = [label for label in labels if label not in ratings]
    return (
        f"已评分 **{len(labels)-len(missing)}/{len(labels)}** 个样本。"
        + ("剩余: " + ", ".join(f"样本 {x}" for x in missing) if missing else "所有样本已评分，可以揭晓结果。")
    )
