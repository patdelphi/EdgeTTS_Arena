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

STATUS_HEADERS = ["Model", "Status", "Experimental", "Capabilities", "Languages"]
COMPARISON_HEADERS = [
    "Model", "Status", "Inference ms", "Audio ms", "RTF", "Peak RSS MB",
    "RSS Δ MB", "CPU %", "TTFB ms", "Error",
]
SUITE_HEADERS = [
    "Case", "Model", "Status", "Successful", "Inference mean ms", "RTF mean",
    "RTF P95", "Peak RSS mean MB", "CPU mean %",
]


def build_arena_ui(
    registry: ModelRegistry,
    benchmark_service: BenchmarkService,
    artifact_store: RunArtifactStore,
    repeated_benchmark_service: RepeatedBenchmarkService | None = None,
) -> gr.Blocks:
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

        gr.Markdown("# EdgeTTS-Arena\nCPU-first local TTS Arena and reproducible benchmark suite.")
        with gr.Row():
            system = gr.Markdown(_system_markdown())
            refresh = gr.Button("Refresh model status")
        status = gr.Dataframe(
            status_rows(initial_models), headers=STATUS_HEADERS, type="array",
            interactive=False, label="Runtime model status",
        )

        with gr.Tab("Arena"):
            with gr.Row():
                with gr.Column(scale=4):
                    preset = gr.Dropdown(
                        [("Custom", "__custom__")] + [(f"{c.id} · {c.name}", c.id) for c in presets.cases],
                        value="__custom__", label="Preset",
                    )
                    text = gr.Textbox(
                        "EdgeTTS-Arena compares local TTS models on the same CPU.",
                        label="Text", lines=6,
                    )
                    models = gr.Dropdown(
                        model_choices(initial_models), value=default_models, multiselect=True,
                        max_choices=4, label="Models (1–4)",
                    )
                    mode = gr.Radio(
                        [("Sequential", "sequential"), ("Concurrent", "concurrent")],
                        value="sequential", label="Execution",
                    )
                    threads = gr.Slider(1, 16, value=4, step=1, label="CPU threads / model")
                    with gr.Row():
                        speed = gr.Slider(
                            .5, 2.0, value=1.0, step=.05, label="Speed",
                            interactive=initial_caps["speed_enabled"],
                        )
                        seed = gr.Number(
                            value=None, precision=0, label="Seed",
                            interactive=initial_caps["seed_enabled"],
                        )
                    voice = gr.Dropdown(
                        initial_caps["voices"], value=None, label="Voice",
                        interactive=initial_caps["voice_enabled"],
                    )
                    language = gr.Dropdown(
                        initial_caps["languages"], value=None, label="Language",
                        interactive=initial_caps["language_enabled"],
                    )
                    cap_summary = gr.Markdown(initial_caps["summary"])
                    run_btn = gr.Button("Run benchmark", variant="primary")
                with gr.Column(scale=6):
                    run_summary = gr.Markdown("_No benchmark run yet._")
                    export_file = gr.File(label="Run export ZIP", interactive=False)

            audios: list[gr.Audio] = []
            cards: list[gr.Markdown] = []
            for row in range(2):
                with gr.Row():
                    for col in range(2):
                        idx = row * 2 + col
                        with gr.Column():
                            audios.append(gr.Audio(type="filepath", label=f"Result {idx + 1}", interactive=False))
                            cards.append(gr.Markdown("_No result_"))
            comparison = gr.Dataframe(
                [], headers=COMPARISON_HEADERS, type="array", interactive=False, label="Comparison",
            )

        with gr.Tab("Standard Suite"):
            gr.Markdown(
                "TC-01~TC-05 run sequentially. Warm-up measurements are discarded; raw measured runs "
                "and aggregate statistics share one run_id."
            )
            suite_cases = gr.Dropdown(
                [(f"{c.id} · {c.name}", c.id) for c in presets.cases],
                value=[c.id for c in presets.cases], multiselect=True, max_choices=5, label="Cases",
            )
            suite_models = gr.Dropdown(
                model_choices(initial_models), value=default_models, multiselect=True,
                max_choices=4, label="Models",
            )
            with gr.Row():
                warmups = gr.Slider(0, 5, value=presets.warmup_runs, step=1, label="Warm-up runs")
                repeats = gr.Slider(1, 10, value=presets.measured_runs, step=1, label="Measured runs")
                suite_threads = gr.Slider(1, 16, value=4, step=1, label="CPU threads / model")
            suite_btn = gr.Button("Run standard suite", variant="primary")
            suite_summary = gr.Markdown("_No standard suite run yet._")
            suite_table = gr.Dataframe(
                [], headers=SUITE_HEADERS, type="array", interactive=False,
                label="Repeated benchmark aggregates",
            )
            suite_export = gr.File(label="Suite export ZIP", interactive=False)

        with gr.Tab("Blind AB"):
            gr.Markdown("Start after an Arena run with at least two successful models.")
            start_blind = gr.Button("Start Blind AB", interactive=False)
            blind_audio: list[gr.Audio] = []
            with gr.Row():
                for i in range(4):
                    blind_audio.append(
                        gr.Audio(type="filepath", label=f"Sample {chr(65+i)}", visible=False, interactive=False)
                    )
            with gr.Row():
                score_sample = gr.Dropdown([], label="Anonymous sample")
                natural = gr.Slider(1, 5, value=3, step=1, label="Naturalness")
                intelligible = gr.Slider(1, 5, value=3, step=1, label="Intelligibility")
                prosody = gr.Slider(1, 5, value=3, step=1, label="Prosody")
            save_score = gr.Button("Save sample score")
            blind_progress = gr.Markdown("_Blind session not started._")
            reveal = gr.Button("Reveal models", interactive=False)
            reveal_output = gr.Markdown("")

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
                raise gr.Error("Select at least one model.")
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
            card_values += ["_No result_"] * (4 - len(card_values))
            success = sum(r.get("status") == "success" for r in data["results"])
            zip_path = artifact_store.build_export(data["run_id"])
            summary = (
                f"### Run `{data['run_id']}`\nMode: **{data['execution_mode']}** · "
                f"Threads/model: **{data['cpu_threads_per_model']}** · Success: **{success}/{len(data['results'])}**"
            )
            return (
                data, summary, str(zip_path), *audio_values, *card_values,
                comparison_rows(data["results"], names), gr.update(interactive=success >= 2),
                None, "_Blind session not started._", "", gr.update(interactive=False),
            )

        def run_suite(
            case_ids: list[str] | None, ids: list[str] | None, warm: int | float,
            measured: int | float, thread_count: int | float, current: list[dict[str, Any]],
        ) -> tuple[str, list[list[Any]], str]:
            if not case_ids or not ids:
                raise gr.Error("Select at least one case and one model.")
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
                f"### Suite `{data['run_id']}`\nCases: **{len(data['cases'])}** · Models: **{len(data['models'])}** · "
                f"Warm-up: **{data['warmup_runs']}** · Measured: **{data['measured_runs']}** · "
                f"Successful pairs: **{success}/{len(data['results'])}**"
            )
            return summary, suite_result_rows(data["results"], names), str(artifact_store.build_export(data["run_id"]))

        def begin_blind(data: dict[str, Any] | None) -> tuple[Any, ...]:
            if not data:
                raise gr.Error("Run an Arena benchmark first.")
            session = create_blind_session(data["run_id"], data["results"])
            by_model = {str(r["model_id"]): r for r in data["results"]}
            audio_updates = []
            for i in range(4):
                if i < len(session["samples"]):
                    sample = session["samples"][i]
                    path = _result_audio_path(by_model[sample["model_id"]], data["run_id"], artifact_store)
                    audio_updates.append(gr.update(value=path, label=f"Sample {sample['label']}", visible=True))
                else:
                    audio_updates.append(gr.update(value=None, visible=False))
            labels = [s["label"] for s in session["samples"]]
            return session, *audio_updates, gr.update(choices=labels, value=labels[0]), _blind_progress(session), "", gr.update(interactive=False)

        def score_blind(
            session: dict[str, Any] | None, label: str | None, n: int | float,
            i: int | float, p: int | float,
        ) -> tuple[Any, str, Any]:
            if not session or not label:
                raise gr.Error("Start Blind AB and select a sample.")
            updated = record_blind_rating(session, label, naturalness=n, intelligibility=i, prosody=p)
            return updated, _blind_progress(updated), gr.update(interactive=blind_session_complete(updated))

        def reveal_blind(session: dict[str, Any] | None) -> tuple[Any, str, str, Any]:
            if not session:
                raise gr.Error("Start Blind AB first.")
            revealed = reveal_blind_session(session)
            artifact_store.write_json(revealed["run_id"], "blind_scores.json", revealed)
            ratings = revealed.get("ratings") or {}
            lines = ["### Reveal"] + [
                f"- **Sample {s['label']} → {s['model_id']}** — Naturalness {ratings[s['label']]['naturalness']}/5, "
                f"Intelligibility {ratings[s['label']]['intelligibility']}/5, Prosody {ratings[s['label']]['prosody']}/5"
                for s in revealed["samples"]
            ]
            return revealed, "\n".join(lines), str(artifact_store.build_export(revealed["run_id"])), gr.update(interactive=False)

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
        f"**System** — {env['os']} / {env['arch']} · logical cores {env['cpu_logical_cores']} · "
        f"RAM {env['available_ram_gb']:.1f}/{env['total_ram_gb']:.1f} GB available"
    )


def _blind_progress(session: dict[str, Any]) -> str:
    labels = [s["label"] for s in session.get("samples", [])]
    ratings = session.get("ratings") or {}
    missing = [label for label in labels if label not in ratings]
    return (
        f"Scored **{len(labels)-len(missing)}/{len(labels)}** samples. "
        + ("Remaining: " + ", ".join(f"Sample {x}" for x in missing) if missing else "All samples scored; reveal enabled.")
    )
