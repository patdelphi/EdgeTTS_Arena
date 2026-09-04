from __future__ import annotations

import gc
import os
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

CosyVoiceFactory = Callable[[str, int], Any]

WETEXT_REQUIRED_FILES = (
    "en/tn/tagger.fst",
    "en/tn/verbalizer.fst",
    "zh/tn/tagger.fst",
    "zh/tn/verbalizer.fst",
    "zh/tn/verbalizer_remove_erhua.fst",
)


def validate_wetext_assets(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    missing = [relative for relative in WETEXT_REQUIRED_FILES if not (root / relative).is_file()]
    if missing:
        raise FileNotFoundError(
            "CosyVoice WeText frontend assets are incomplete under "
            f"{root}; missing: {', '.join(missing)}"
        )
    return root


def build_offline_wetext_normalizer(normalizer_cls: type[Any], root: str | Path) -> type[Any]:
    """Build a drop-in ``wetext.Normalizer`` that never calls snapshot_download.

    ``wetext==0.0.4`` downloads ``pengzhendong/wetext`` whenever tagger/verbalizer
    paths are omitted. CosyVoice imports that class directly. This wrapper keeps
    the old runtime but always supplies explicit local FST paths, preserving the
    pinned upstream behavior without hidden network access during model load.
    """

    assets = validate_wetext_assets(root)

    class OfflineNormalizer:
        def __init__(
            self,
            *,
            lang: str = "auto",
            operator: str = "tn",
            remove_erhua: bool = False,
            enable_0_to_9: bool = False,
            **kwargs: Any,
        ) -> None:
            if kwargs:
                unsupported = ", ".join(sorted(kwargs))
                raise ValueError(f"unsupported offline WeText options: {unsupported}")
            if operator != "tn":
                raise ValueError("CosyVoice offline WeText bridge currently supports TN only")
            if lang not in {"auto", "en", "zh"}:
                raise ValueError("CosyVoice offline WeText language must be auto, en, or zh")
            if enable_0_to_9:
                raise ValueError("enable_0_to_9 is only meaningful for ITN and is not supported here")

            self.lang = lang
            self._models: dict[str, Any] = {}
            languages = ("en", "zh") if lang == "auto" else (lang,)
            for language in languages:
                verbalizer = "verbalizer.fst"
                if language == "zh" and remove_erhua:
                    verbalizer = "verbalizer_remove_erhua.fst"
                self._models[language] = normalizer_cls(
                    tagger_path=str(assets / language / "tn" / "tagger.fst"),
                    verbalizer_path=str(assets / language / "tn" / verbalizer),
                    lang=language,
                    operator="tn",
                    remove_erhua=remove_erhua,
                )

        @staticmethod
        def _contains_chinese(text: str) -> bool:
            return any("\u4e00" <= char <= "\u9fff" for char in text)

        def normalize(self, text: str) -> str:
            language = self.lang
            if language == "auto":
                language = "zh" if self._contains_chinese(text) else "en"
            return self._models[language].normalize(text)

    OfflineNormalizer.__name__ = "OfflineCosyVoiceWetextNormalizer"
    return OfflineNormalizer


class CosyVoiceTTSAdapter(BaseTTSAdapter):
    """SFT-only CPU adapter for the official QwenAudio/CosyVoice runtime.

    This deliberately targets ``CosyVoice-300M-SFT`` style models. CosyVoice2/3
    zero-shot inference requires prompt audio/text and therefore does not fit the
    current Arena voice-id-only request contract without a schema extension.

    For reproducible deployment the pinned ``wetext==0.0.4`` frontend is forced
    to use local FST files. Set ``EDGETTS_ARENA_COSYVOICE_WETEXT_DIR`` or place
    them in ``<CosyVoice model parent>/wetext``.
    """

    id = "cosyvoice"
    capabilities = TTSCapabilities(
        streaming=True,
        seed=False,
        speed=True,
        voices=True,
        voice_clone=False,
        languages=(),
    )
    available_voices: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        model_factory: CosyVoiceFactory | None = None,
        runtime_version: str | None = None,
        wetext_dir: str | None = None,
    ) -> None:
        self._model_factory = model_factory
        self._runtime_version = runtime_version
        self._configured_wetext_dir = Path(wetext_dir).expanduser() if wetext_dir else None
        self._resolved_wetext_dir: Path | None = None
        self._engine: Any | None = None
        self._model_path: Path | None = None
        self._num_threads = 1
        self.is_loaded = False
        self.available_voices = ()

    def load_model(self, model_path: str, *, device: str = "cpu", num_threads: int = 4) -> None:
        if device != "cpu":
            raise ValueError("CosyVoice adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f"CosyVoice model directory not found: {path}")
        if not (path / "cosyvoice.yaml").is_file():
            raise ValueError("CosyVoice model directory must contain cosyvoice.yaml")

        factory, runtime_version = self._runtime(path.resolve())
        engine = factory(str(path.resolve()), num_threads)
        voices = tuple(str(item) for item in engine.list_available_spks())
        if not voices:
            raise ArenaError(
                1002,
                "CosyVoice model exposes no SFT speakers; use a CosyVoice-300M-SFT model for this adapter",
                error_type="model_unavailable",
            )

        self._engine = engine
        self._runtime_version = runtime_version
        self._model_path = path.resolve()
        self._num_threads = num_threads
        self.available_voices = voices
        self.is_loaded = True

    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        engine = self._require_loaded()
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")
        if kwargs.get("seed") is not None:
            raise ArenaError(1003, "CosyVoice SFT adapter does not expose deterministic seed control", error_type="capability_conflict")
        selected_voice = voice or self.available_voices[0]
        if selected_voice not in self.available_voices:
            raise ValueError(f"unknown CosyVoice SFT speaker: {selected_voice}")

        chunks = list(engine.inference_sft(normalized, selected_voice, stream=False, speed=speed))
        audio = self._merge_chunks(chunks)
        return TTSOutput(audio=audio, sample_rate=int(engine.sample_rate), metadata=self._metadata(selected_voice, streaming=False))

    def infer_stream(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> Iterator[TTSOutput]:
        engine = self._require_loaded()
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if speed <= 0:
            raise ValueError("speed must be positive")
        selected_voice = voice or self.available_voices[0]
        if selected_voice not in self.available_voices:
            raise ValueError(f"unknown CosyVoice SFT speaker: {selected_voice}")
        for chunk in engine.inference_sft(normalized, selected_voice, stream=True, speed=speed):
            audio = self._chunk_audio(chunk)
            yield TTSOutput(audio=audio, sample_rate=int(engine.sample_rate), metadata=self._metadata(selected_voice, streaming=True))

    def unload_model(self) -> None:
        self._engine = None
        self._model_path = None
        self.available_voices = ()
        self.is_loaded = False
        gc.collect()

    def _runtime(self, model_path: Path) -> tuple[CosyVoiceFactory, str]:
        if self._model_factory is not None:
            return self._model_factory, self._runtime_version or "injected"
        try:
            import torch
            import wetext
        except ImportError as exc:
            raise ArenaError(
                1002,
                "CosyVoice runtime is not installed. Install the official QwenAudio/CosyVoice source checkout and its dependencies.",
                error_type="model_unavailable",
            ) from exc

        wetext_dir = self._configured_wetext_dir
        if wetext_dir is None:
            configured = os.environ.get("EDGETTS_ARENA_COSYVOICE_WETEXT_DIR")
            wetext_dir = Path(configured).expanduser() if configured else model_path.parent / "wetext"
        try:
            self._resolved_wetext_dir = validate_wetext_assets(wetext_dir)
        except FileNotFoundError as exc:
            raise ArenaError(
                1002,
                f"{exc}. Run scripts/prepare_cosyvoice_frontend.py before loading CosyVoice.",
                error_type="model_unavailable",
            ) from exc

        offline_normalizer = build_offline_wetext_normalizer(wetext.Normalizer, self._resolved_wetext_dir)
        wetext.Normalizer = offline_normalizer

        # Defence in depth: the offline bridge above always supplies local FST
        # paths, so wetext never needs the network. Hard-block the modelscope
        # download that wetext falls back to when those paths are missing;
        # otherwise a network failure surfaces as the cryptic
        # "'NoneType' object has no attribute 'splitlines'" (3002) deep inside
        # modelscope instead of a clear, actionable offline-assets error.
        import wetext.wetext as wetext_impl

        resolved_dir = self._resolved_wetext_dir

        def _blocked_snapshot_download(*_args: Any, **_kwargs: Any) -> Any:
            raise ArenaError(
                1002,
                "CosyVoice WeText frontend attempted a network download "
                "(modelscope snapshot_download of 'pengzhendong/wetext'). Offline "
                f"operation requires local FST assets under {resolved_dir}; run "
                "scripts/prepare_cosyvoice_frontend.py and set "
                "EDGETTS_ARENA_COSYVOICE_WETEXT_DIR if they are missing.",
                error_type="model_unavailable",
            )

        wetext_impl.snapshot_download = _blocked_snapshot_download

        try:
            from cosyvoice.cli.cosyvoice import AutoModel
            import cosyvoice.cli.frontend as frontend_module
        except ImportError as exc:
            raise ArenaError(
                1002,
                "CosyVoice runtime is not installed. Install the official QwenAudio/CosyVoice source checkout and its dependencies.",
                error_type="model_unavailable",
            ) from exc

        # Cover both first import and already-imported frontend modules.
        frontend_module.ZhNormalizer = offline_normalizer
        frontend_module.EnNormalizer = offline_normalizer

        def factory(model_dir: str, num_threads: int) -> Any:
            torch.set_num_threads(num_threads)
            return AutoModel(model_dir=model_dir)

        return factory, "QwenAudio/CosyVoice-source"

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float32)
        return array.reshape(-1)

    @classmethod
    def _chunk_audio(cls, chunk: Any) -> np.ndarray:
        if not isinstance(chunk, dict) or "tts_speech" not in chunk:
            raise RuntimeError("CosyVoice returned a chunk without tts_speech")
        audio = cls._to_numpy(chunk["tts_speech"])
        if audio.size == 0:
            raise RuntimeError("CosyVoice returned an empty audio chunk")
        return audio

    @classmethod
    def _merge_chunks(cls, chunks: list[Any]) -> np.ndarray:
        if not chunks:
            raise RuntimeError("CosyVoice returned no audio chunks")
        return np.concatenate([cls._chunk_audio(chunk) for chunk in chunks]).astype(np.float32, copy=False)

    def _metadata(self, voice: str, *, streaming: bool) -> dict[str, Any]:
        return {
            "adapter": self.id,
            "voice": voice,
            "runtime": "QwenAudio/CosyVoice",
            "runtime_version": self._runtime_version or "unknown",
            "model_version": self._model_path.name if self._model_path else "unknown",
            "quantization": "model-defined",
            "threads": self._num_threads,
            "streaming": streaming,
            "mode": "sft",
            "wetext_dir": str(self._resolved_wetext_dir) if self._resolved_wetext_dir else None,
        }

    def _require_loaded(self) -> Any:
        if not self.is_loaded or self._engine is None:
            raise ModelNotLoadedError(self.id)
        return self._engine
