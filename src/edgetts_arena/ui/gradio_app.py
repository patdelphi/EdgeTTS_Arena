from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr
from fastapi.responses import RedirectResponse

from edgetts_arena.api.app import create_app
from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.benchmark_service import BenchmarkService
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
    usable_model_ids,
)


COMPARISON_HEADERS = [
    "Model",
    "Status",
    "Inference ms",
    "Audio ms",
    "RTF",
    "Peak RSS MB",
    "RSS Δ MB",
    "CPU %",
    "TTFB ms",
    "Error",
]
STATUS_HEADERS = ["Model", "Status", "Experimental", "Capabilities", "Languages"]


def build_arena_ui(
    registry: ModelRegistry,
    benchmark_service: BenchmarkService,
    artifact_store: RunArtifactStore,
) -> gr.Blocks:
    initial_models = registry.list_models()
    usable = usable_model_ids(initial_models)
    default_models = usable[:2] if len(usable) >= 2 else usable[:1]
    initial_caps = capability_view(initial_models, default_models)

    with gr.Blocks(title="EdgeTTS-Arena") as demo:
        models_state = gr.State(initial_models)
        run_state = gr.State(None)
        blind_state = gr.State(None)

        gr.Markdown(
            "# EdgeTTS-Arena\n"
            "CPU-first local TTS comparison and benchmark workspace. "
            "Stage 4 UI reads model capabilities from the same runtime used by the API."
        )

        with gr.Row():
            system_summary = gr.Markdown(_system_markdown())
            refresh_button = gr.Button("Refresh model status", variant="secondary")

        model_status = gr.Dataframe(
            value=status_rows(initial_models),
            headers=STATUS_HEADERS,
            type="array",
            interactive=False,
            label="Runtime model status",
            wrap=True,
        )

        with gr.Row():
            with gr.Column(scale=4):
                text = gr.Textbox(
                    label="Text",
                    value="EdgeTTS-Arena compares local TTS models on the same CPU.",
                    lines=6,
                    max_lines=12,
                )
                models = gr.Dropdown(
                    choices=model_choices(initial_models),
                    value=default_models,
                    multiselect=True,
                    max_choices=4,
                    label="Models (1–4; 2–4 recommended for comparison)",
                )
                with gr.Row():
                    execution_mode = gr.Radio(
                        choices=[("Sequential (default)", "sequential"), ("Concurrent", "concurrent")],
                        value="sequential",
                        label="Execution",
                    )
                    threads = gr.Slider(
                        minimum=1,
                        maximum=16,
                        step=1,
                        value=4,
                        label="CPU threads / model",
                    )
                with gr.Row():
                    speed = gr.Slider(
                        minimum=0.5,
                        maximum=2.0,
                        step=0.05,
                        value=1.0,
                        label="Speed",
                        interactive=initial_caps["speed_enabled"],
                    )
                    seed = gr.Number(
                        value=None,
                        precision=0,
                        label="Seed",
                        info="Partial support is allowed; unsupported models emit a warning.",
                        interactive=initial_caps["seed_enabled"],
                    )
                voice = gr.Dropdown(
                    choices=initial_caps["voices"],
                    value=None,
                    label="Voice",
                    info="Enabled for a single selected model. Multi-model Arena uses per-model defaults.",
                    interactive=initial_caps["voice_enabled"],
                )
                capability_summary = gr.Markdown(initial_caps["summary"])
                run_button = gr.Button("Run benchmark", variant="primary")

            with gr.Column(scale=6):
                run_summary = gr.Markdown("_No benchmark run yet._")
                export_file = gr.File(label="Run export ZIP", interactive=False)

        gr.Markdown("## Arena Results")
        result_audio: list[gr.Audio] = []
        result_cards: list[gr.Markdown] = []
        for row_index in range(2):
            with gr.Row():
                for col_index in range(2):
                    index = row_index * 2 + col_index
                    with gr.Column():
                        audio = gr.Audio(
                            value=None,
                            type="filepath",
                            label=f"Result {index + 1}",
                            interactive=False,
                        )
                        card = gr.Markdown("_No result_")
                        result_audio.append(audio)
                        result_cards.append(card)

        comparison = gr.Dataframe(
            value=[],
            headers=COMPARISON_HEADERS,
            type="array",
            interactive=False,
            label="Comparison",
            wrap=True,
        )

        gr.Markdown("## Blind AB")
        gr.Markdown(
            "Start Blind AB after a run with at least two successful models. "
            "Model identity stays hidden until every anonymous sample has a 1–5 rating."
        )
        start_blind = gr.Button("Start Blind AB", interactive=False)

        blind_audio: list[gr.Audio] = []
        with gr.Row():
            for index in range(4):
                blind_audio.append(
                    gr.Audio(
                        value=None,
                        type="filepath",
                        label=f"Sample {chr(ord('A') + index)}",
                        interactive=False,
                        visible=False,
                    )
                )

        with gr.Row():
            score_sample = gr.Dropdown(
                choices=[],
                value=None,
                label="Anonymous sample",
                interactive=True,
            )
            naturalness = gr.Slider(1, 5, value=3, step=1, label="Naturalness")
            intelligibility = gr.Slider(1, 5, value=3, step=1, label="Intelligibility")
            prosody = gr.Slider(1, 5, value=3, step=1, label="Prosody")
        save_score = gr.Button("Save sample score")
        blind_progress = gr.Markdown("_Blind session not started._")
        reveal_button = gr.Button("Reveal models", interactive=False)
        reveal_output = gr.Markdown("")

        def refresh() -> tuple[Any, ...]:
            current = registry.list_models()
            current_usable = usable_model_ids(current)
            defaults = current_usable[:2] if len(current_usable) >= 2 else current_usable[:1]
            caps = capability_view(current, defaults)
            return (
                current,
                gr.update(choices=model_choices(current), value=defaults),
                status_rows(current),
                _system_markdown(),
                gr.update(interactive=caps["speed_enabled"], value=1.0),
                gr.update(interactive=caps["seed_enabled"], value=None),
                gr.update(
                    choices=caps["voices"],
                    value=None,
                    interactive=caps["voice_enabled"],
                ),
                caps["summary"],
            )

        def selection_changed(
            selected_ids: list[str] | None, current_models: list[dict[str, Any]]
        ) -> tuple[Any, ...]:
            caps = capability_view(current_models or registry.list_models(), selected_ids)
            return (
                gr.update(interactive=caps["speed_enabled"], value=1.0),
                gr.update(interactive=caps["seed_enabled"], value=None),
                gr.update(
                    choices=caps["voices"],
                    value=None,
                    interactive=caps["voice_enabled"],
                ),
                caps["summary"],
            )

        def run_benchmark(
            input_text: str,
            selected_ids: list[str] | None,
            mode: str,
            thread_count: int | float,
            speed_value: float,
            seed_value: int | float | None,
            voice_value: str | None,
            current_models: list[dict[str, Any]],
        ) -> tuple[Any, ...]:
            selected = list(selected_ids or [])
            if not selected:
                raise gr.Error("Select at least one model.")
            if len(selected) > 4:
                raise gr.Error("Arena supports at most four models.")

            config: dict[str, Any] = {
                "speed": float(speed_value),
                "seed": None if seed_value is None else int(seed_value),
                "sample_rate": None,
                "voice": voice_value or None,
            }
            data = benchmark_service.run(
                text=input_text,
                model_ids=selected,
                execution_mode=mode,
                cpu_threads_per_model=int(thread_count),
                config=config,
            )
            results = data["results"]
            names = {
                str(item["id"]): str(item.get("name") or item["id"])
                for item in (current_models or registry.list_models())
            }

            audio_values: list[str | None] = []
            cards: list[str] = []
            for result in results[:4]:
                audio_values.append(_result_audio_path(result, data["run_id"], artifact_store))
                cards.append(format_result_card(result, names))
            while len(audio_values) < 4:
                audio_values.append(None)
                cards.append("_No result_")

            export_path = artifact_store.build_export(data["run_id"])
            success_count = sum(item.get("status") == "success" for item in results)
            summary = (
                f"### Run `{data['run_id']}`\n"
                f"Mode: **{data['execution_mode']}** · Threads/model: **{data['cpu_threads_per_model']}** · "
                f"Success: **{success_count}/{len(results)}**"
            )
            return (
                data,
                summary,
                str(export_path),
                *audio_values,
                *cards,
                comparison_rows(results, names),
                gr.update(interactive=success_count >= 2),
                None,
                "_Blind session not started._",
                "",
                gr.update(interactive=False),
            )

        def begin_blind(data: dict[str, Any] | None) -> tuple[Any, ...]:
            if not data:
                raise gr.Error("Run a benchmark first.")
            session = create_blind_session(data["run_id"], data["results"])
            by_model = {str(item["model_id"]): item for item in data["results"]}
            audio_updates: list[Any] = []
            for index in range(4):
                if index < len(session["samples"]):
                    sample = session["samples"][index]
                    result = by_model[sample["model_id"]]
                    path = _result_audio_path(result, data["run_id"], artifact_store)
                    audio_updates.append(
                        gr.update(value=path, label=f"Sample {sample['label']}", visible=True)
                    )
                else:
                    audio_updates.append(gr.update(value=None, visible=False))
            labels = [item["label"] for item in session["samples"]]
            return (
                session,
                *audio_updates,
                gr.update(choices=labels, value=labels[0] if labels else None),
                _blind_progress(session),
                "",
                gr.update(interactive=False),
            )

        def save_blind_score(
            session: dict[str, Any] | None,
            label: str | None,
            natural: int | float,
            intelligible: int | float,
            prosody_value: int | float,
        ) -> tuple[Any, ...]:
            if not session:
                raise gr.Error("Start Blind AB first.")
            if not label:
                raise gr.Error("Choose an anonymous sample.")
            updated = record_blind_rating(
                session,
                label,
                naturalness=natural,
                intelligibility=intelligible,
                prosody=prosody_value,
            )
            return (
                updated,
                _blind_progress(updated),
                gr.update(interactive=blind_session_complete(updated)),
            )

        def reveal_blind(session: dict[str, Any] | None) -> tuple[Any, ...]:
            if not session:
                raise gr.Error("Start Blind AB first.")
            revealed = reveal_blind_session(session)
            artifact_store.write_json(revealed["run_id"], "blind_scores.json", revealed)
            export_path = artifact_store.build_export(revealed["run_id"])
            lines = ["### Reveal"]
            ratings = revealed.get("ratings") or {}
            for sample in revealed["samples"]:
                score = ratings.get(sample["label"], {})
                lines.append(
                    f"- **Sample {sample['label']} → {sample['model_id']}** — "
                    f"Naturalness {score.get('naturalness', '—')}/5, "
                    f"Intelligibility {score.get('intelligibility', '—')}/5, "
                    f"Prosody {score.get('prosody', '—')}/5"
                )
            lines.append(f"\nScores saved to run `{revealed['run_id']}` and included in the export ZIP.")
            return revealed, "\n".join(lines), str(export_path), gr.update(interactive=False)

        refresh_outputs = [
            models_state,
            models,
            model_status,
            system_summary,
            speed,
            seed,
            voice,
            capability_summary,
        ]
        refresh_button.click(refresh, outputs=refresh_outputs)
        demo.load(refresh, outputs=refresh_outputs)
        models.change(
            selection_changed,
            inputs=[models, models_state],
            outputs=[speed, seed, voice, capability_summary],
        )

        run_button.click(
            run_benchmark,
            inputs=[text, models, execution_mode, threads, speed, seed, voice, models_state],
            outputs=[
                run_state,
                run_summary,
                export_file,
                *result_audio,
                *result_cards,
                comparison,
                start_blind,
                blind_state,
                blind_progress,
                reveal_output,
                reveal_button,
            ],
        )
        start_blind.click(
            begin_blind,
            inputs=[run_state],
            outputs=[
                blind_state,
                *blind_audio,
                score_sample,
                blind_progress,
                reveal_output,
                reveal_button,
            ],
        )
        save_score.click(
            save_blind_score,
            inputs=[blind_state, score_sample, naturalness, intelligibility, prosody],
            outputs=[blind_state, blind_progress, reveal_button],
        )
        reveal_button.click(
            reveal_blind,
            inputs=[blind_state],
            outputs=[blind_state, reveal_output, export_file, reveal_button],
        )

    return demo


def create_full_app(*, exports_root: str | Path = "exports"):
    base_app = create_app(exports_root=exports_root)
    store = base_app.state.artifact_store
    demo = build_arena_ui(
        base_app.state.registry,
        base_app.state.benchmark_service,
        store,
    )

    @base_app.get("/", include_in_schema=False)
    async def arena_redirect() -> RedirectResponse:
        return RedirectResponse(url="/arena/")

    return gr.mount_gradio_app(
        base_app,
        demo,
        path="/arena",
        allowed_paths=[str(store.root)],
        show_error=True,
    )


def _result_audio_path(
    result: dict[str, Any], run_id: str, artifact_store: RunArtifactStore
) -> str | None:
    if result.get("status") != "success" or not result.get("audio_url"):
        return None
    filename = str(result["audio_url"]).rsplit("/", 1)[-1]
    return str(artifact_store.get_audio_file(run_id, filename))


def _system_markdown() -> str:
    env = collect_system_environment()
    return (
        "**System** — "
        f"{env['os']} / {env['arch']} · "
        f"logical cores {env['cpu_logical_cores']} · "
        f"RAM {env['available_ram_gb']:.1f}/{env['total_ram_gb']:.1f} GB available"
    )


def _blind_progress(session: dict[str, Any]) -> str:
    labels = [item["label"] for item in session.get("samples", [])]
    ratings = session.get("ratings") or {}
    completed = [label for label in labels if label in ratings]
    waiting = [label for label in labels if label not in ratings]
    text = f"Scored **{len(completed)}/{len(labels)}** samples."
    if waiting:
        text += " Remaining: " + ", ".join(f"Sample {label}" for label in waiting) + "."
    else:
        text += " All samples scored; reveal is now enabled."
    return text
