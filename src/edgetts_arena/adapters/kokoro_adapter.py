from __future__ import annotations

import gc
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable

import numpy as np

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

KokoroFactory = Callable[[str, str, int], Any]
# Builds the dedicated Chinese engine: (model, voices, vocab_config, threads).
ZhKokoroFactory = Callable[[str, str, str, int], Any]
# Builds the Mandarin G2P; the returned object maps text -> (phonemes, aux).
ZhG2PFactory = Callable[[], Any]


class KokoroTTSAdapter(BaseTTSAdapter):
    """CPU adapter for Kokoro through ``kokoro-onnx``.

    English and other Latin/IPA languages run on the v1.0 model with espeak-ng.
    Mandarin (``cmn``) instead prefers a dedicated ``v1.1-zh`` model discovered
    under ``<model_dir>/zh`` and phonemized with ``misaki``'s ``ZHG2P``: the v1.0
    vocab carries no tone marks, so espeak-ng ``cmn`` strips Mandarin tones and
    the result is unintelligible. When the zh model is absent the adapter falls
    back to the v1.0 espeak path so behaviour degrades gracefully.
    """

    id = "kokoro"
    capabilities = TTSCapabilities(
        streaming=False,
        seed=False,
        speed=True,
        voices=True,
        voice_clone=False,
        languages=("en-us", "en-gb", "cmn", "ja", "ko", "fr-fr", "es", "it", "pt-br", "hi"),
    )
    available_voices: tuple[str, ...] = ()

    def __init__(
        self,
        *,
        kokoro_factory: KokoroFactory | None = None,
        runtime_version: str | None = None,
        zh_kokoro_factory: ZhKokoroFactory | None = None,
        zh_g2p_factory: ZhG2PFactory | None = None,
    ) -> None:
        self._kokoro_factory = kokoro_factory
        self._runtime_version = runtime_version
        self._zh_kokoro_factory = zh_kokoro_factory
        self._zh_g2p_factory = zh_g2p_factory
        self._engine: Any | None = None
        self._model_path: Path | None = None
        self._voices_path: Path | None = None
        self._model_dir: Path | None = None
        self._num_threads = 1
        self.is_loaded = False
        self.available_voices = ()
        # Dedicated Chinese (v1.1-zh) engine: discovered at load, built on demand.
        self._zh_model_path: Path | None = None
        self._zh_voices_path: Path | None = None
        self._zh_config_path: Path | None = None
        self._zh_available = False
        self._zh_load_failed = False
        self._zh_engine: Any | None = None
        self._zh_g2p: Any | None = None
        self._zh_voices: tuple[str, ...] = ()

    def load_model(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        if device != "cpu":
            raise ValueError("Kokoro adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        model_file, voices_file = self._resolve_model_files(model_path)
        factory, runtime_version = self._runtime()
        engine = factory(str(model_file), str(voices_file), num_threads)
        voices = tuple(
            voice
            for voice in (str(item) for item in engine.get_voices())
            if self._voice_language_supported(voice)
        )
        if not voices:
            raise RuntimeError("Kokoro runtime returned no direct-text English voices")

        self._engine = engine
        self._runtime_version = runtime_version
        self._model_path = model_file
        self._voices_path = voices_file
        self._model_dir = model_file.parent
        self._num_threads = num_threads
        self.available_voices = voices
        self.is_loaded = True

        # Discover (but do not load yet) the dedicated Chinese model so Mandarin
        # can bypass the tone-stripping espeak-ng 'cmn' path. Loading is deferred
        # to the first Chinese request to avoid a second ~340MB session at startup.
        self._zh_model_path, self._zh_voices_path, self._zh_config_path = (
            self._discover_zh_files(self._model_dir / "zh")
        )
        self._zh_available = self._zh_model_path is not None

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
        if not 0.5 <= speed <= 2.0:
            raise ValueError("speed must be between 0.5 and 2.0 for kokoro-onnx")
        if kwargs.get("seed") is not None:
            raise ArenaError(
                1003,
                "Kokoro does not support deterministic seed control",
                error_type="capability_conflict",
            )

        # An explicit voice that is not a known v1.0 voice may still be a
        # dedicated zh voice; only reject it outright when there is no zh model to
        # validate it against, so "unknown Kokoro voice" stays the reported error
        # instead of a downstream language-inference failure.
        voice_is_v1 = voice is None or voice in self.available_voices
        if not voice_is_v1 and not self._zh_available:
            raise ValueError(f"unknown Kokoro voice: {voice}")

        language = self._resolve_language(
            text=normalized, voice=voice if voice_is_v1 else None, kwargs=kwargs
        )

        # Mandarin prefers the dedicated v1.1-zh model + misaki ZHG2P. Only take it
        # when no explicit v1.0 voice was requested (an explicit v1.0 voice stays on
        # the v1.0 engine), and fall through to espeak-ng when the zh model is
        # missing or failed to load.
        if (
            language == "cmn"
            and self._zh_available
            and not self._zh_load_failed
            and (voice is None or voice not in self.available_voices)
        ):
            zh_engine = self._ensure_zh_engine()
            if zh_engine is not None:
                return self._infer_zh(
                    zh_engine, normalized, voice=voice, speed=speed, kwargs=kwargs
                )

        # Validate an explicit voice up front so an unknown one is reported as
        # "unknown Kokoro voice" instead of a downstream language-inference error.
        if voice is not None and voice not in self.available_voices:
            raise ValueError(f"unknown Kokoro voice: {voice}")
        selected_voice = voice or self._default_voice(language)
        if selected_voice not in self.available_voices:
            raise ValueError(f"unknown Kokoro voice: {selected_voice}")
        if language is None:
            language = self._language_for_voice(selected_voice)
        if language not in self.capabilities.languages:
            raise ValueError(f"unsupported Kokoro language: {language}")

        create_kwargs: dict[str, Any] = {
            "voice": selected_voice,
            "speed": speed,
            "lang": language,
        }
        for key in ("trim", "sentence_pause", "clause_pause", "continuous"):
            if key in kwargs and kwargs[key] is not None:
                create_kwargs[key] = kwargs[key]

        samples, sample_rate = engine.create(normalized, **create_kwargs)
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise RuntimeError("Kokoro returned invalid audio")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError("Kokoro returned an invalid sample rate")

        return TTSOutput(
            audio=audio,
            sample_rate=sample_rate,
            metadata={
                "adapter": self.id,
                "voice": selected_voice,
                "language": language,
                "runtime": "kokoro-onnx",
                "runtime_version": self._runtime_version or "unknown",
                "model_version": "v1.0",
                "quantization": self._quantization_hint(),
                "model_path": str(self._model_path) if self._model_path else None,
                "voices_path": str(self._voices_path) if self._voices_path else None,
                "threads": self._num_threads,
                "streaming": False,
            },
        )

    def unload_model(self) -> None:
        self._engine = None
        self._model_path = None
        self._voices_path = None
        self._model_dir = None
        self._zh_engine = None
        self._zh_g2p = None
        self._zh_voices = ()
        self._zh_load_failed = False
        self.available_voices = ()
        self.is_loaded = False
        gc.collect()

    def _runtime(self) -> tuple[KokoroFactory, str]:
        if self._kokoro_factory is not None:
            return self._kokoro_factory, self._runtime_version or "injected"
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise ArenaError(
                1002,
                'Kokoro runtime is not installed. Install with: pip install -e ".[kokoro]"',
                error_type="model_unavailable",
            ) from exc

        def factory(model_file: str, voices_file: str, num_threads: int) -> Any:
            options = ort.SessionOptions()
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model_file,
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            return Kokoro.from_session(session, voices_file)

        try:
            runtime_version = importlib_metadata.version("kokoro-onnx")
        except importlib_metadata.PackageNotFoundError:
            runtime_version = "unknown"
        return factory, runtime_version

    @staticmethod
    def _resolve_model_files(model_path: str) -> tuple[Path, Path]:
        raw = model_path.strip()
        if not raw:
            raise ValueError("model_path must not be empty for Kokoro")
        path = Path(raw).expanduser()
        if path.is_dir():
            models = sorted(path.glob("kokoro-v1*.onnx"))
            if len(models) != 1:
                raise ValueError(
                    f"Kokoro directory must contain exactly one kokoro-v1*.onnx; found {len(models)}"
                )
            model_file = models[0]
        elif path.is_file() and path.suffix.lower() == ".onnx":
            model_file = path
        else:
            raise FileNotFoundError(f"Kokoro model path does not exist or is not ONNX: {path}")

        candidates = [
            model_file.with_name("voices-v1.0.bin"),
            model_file.with_name("voices.bin"),
        ]
        voices_file = next((candidate for candidate in candidates if candidate.is_file()), None)
        if voices_file is None:
            raise FileNotFoundError(
                f"Kokoro voices file not found beside {model_file.name}; expected voices-v1.0.bin"
            )
        return model_file.resolve(), voices_file.resolve()

    @staticmethod
    def _discover_zh_files(zh_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
        """Locate the dedicated Chinese model beside the v1.0 files.

        Expects ``kokoro-v1*-zh.onnx``, ``voices-v1*-zh.bin`` and a vocab config
        (``config.json`` or ``v1.1_config.json``) inside ``zh_dir``. Returns
        ``(None, None, None)`` unless all three exist, so a partial download
        cleanly disables the Mandarin path instead of half-working.
        """
        if not zh_dir.is_dir():
            return (None, None, None)
        models = sorted(zh_dir.glob("kokoro-v1*-zh.onnx"))
        voices = sorted(zh_dir.glob("voices-v1*-zh.bin"))
        config = next(
            (c for c in (zh_dir / "config.json", zh_dir / "v1.1_config.json") if c.is_file()),
            None,
        )
        if len(models) == 1 and len(voices) == 1 and config is not None:
            return (models[0].resolve(), voices[0].resolve(), config.resolve())
        return (None, None, None)

    def _zh_runtime(self) -> ZhKokoroFactory:
        if self._zh_kokoro_factory is not None:
            return self._zh_kokoro_factory
        try:
            import onnxruntime as ort
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise ArenaError(
                1002,
                'Kokoro runtime is not installed. Install with: pip install -e ".[kokoro]"',
                error_type="model_unavailable",
            ) from exc

        def factory(model_file: str, voices_file: str, config_file: str, num_threads: int) -> Any:
            options = ort.SessionOptions()
            options.intra_op_num_threads = num_threads
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                model_file, sess_options=options, providers=["CPUExecutionProvider"]
            )
            # vocab_config is required: the zh vocab (Zhuyin + tone digits) differs
            # from the vocab embedded in this session.
            return Kokoro.from_session(session, voices_file, vocab_config=config_file)

        return factory

    def _zh_g2p_instance(self) -> Any:
        if self._zh_g2p_factory is not None:
            return self._zh_g2p_factory()
        try:
            from misaki import zh as misaki_zh
        except ImportError as exc:
            raise ArenaError(
                1002,
                'Mandarin G2P is not installed. Install with: pip install "misaki[zh]"',
                error_type="model_unavailable",
            ) from exc
        return misaki_zh.ZHG2P(version="1.1")

    def _ensure_zh_engine(self) -> Any | None:
        """Lazily build the Chinese engine + G2P; return ``None`` to trigger fallback."""
        if self._zh_engine is not None:
            return self._zh_engine
        if not self._zh_available or self._zh_load_failed:
            return None
        try:
            # Build the (cheap) G2P first so a missing misaki[zh] degrades to the
            # v1.0 espeak path without first loading the ~340MB zh session.
            g2p = self._zh_g2p_instance()
            factory = self._zh_runtime()
            engine = factory(
                str(self._zh_model_path),
                str(self._zh_voices_path),
                str(self._zh_config_path),
                self._num_threads,
            )
            zh_voices = tuple(str(item) for item in engine.get_voices())
        except Exception:
            # Any failure (missing runtime, corrupt model) degrades to the v1.0
            # espeak path rather than aborting the whole run.
            self._zh_load_failed = True
            self._zh_engine = None
            self._zh_g2p = None
            self._zh_voices = ()
            return None
        self._zh_engine = engine
        self._zh_g2p = g2p
        self._zh_voices = zh_voices
        return engine

    def _default_zh_voice(self) -> str:
        for candidate in ("zf_001", "zf_002", "zm_010"):
            if candidate in self._zh_voices:
                return candidate
        for voice in self._zh_voices:
            if voice.startswith("z"):
                return voice
        if not self._zh_voices:
            raise RuntimeError("Kokoro zh runtime returned no voices")
        return self._zh_voices[0]

    def _infer_zh(
        self,
        engine: Any,
        text: str,
        *,
        voice: str | None,
        speed: float,
        kwargs: dict[str, Any],
    ) -> TTSOutput:
        selected_voice = voice or self._default_zh_voice()
        if selected_voice not in self._zh_voices:
            raise ValueError(f"unknown Kokoro voice: {selected_voice}")
        # ZHG2P returns (phonemes, aux); feed the phonemes straight to the model so
        # the tone-bearing Zhuyin symbols bypass espeak-ng entirely.
        phonemes = self._zh_g2p(text)[0]
        create_kwargs: dict[str, Any] = {
            "voice": selected_voice,
            "speed": speed,
            "is_phonemes": True,
        }
        for key in ("trim", "sentence_pause", "clause_pause", "continuous"):
            if key in kwargs and kwargs[key] is not None:
                create_kwargs[key] = kwargs[key]

        samples, sample_rate = engine.create(phonemes, **create_kwargs)
        audio = np.asarray(samples, dtype=np.float32)
        if audio.ndim != 1 or audio.size == 0:
            raise RuntimeError("Kokoro returned invalid audio")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise RuntimeError("Kokoro returned an invalid sample rate")

        return TTSOutput(
            audio=audio,
            sample_rate=sample_rate,
            metadata={
                "adapter": self.id,
                "voice": selected_voice,
                "language": "cmn",
                "runtime": "kokoro-onnx",
                "runtime_version": self._runtime_version or "unknown",
                "model_version": "v1.1-zh",
                "phonemizer": "misaki-zh",
                "quantization": self._quantization_hint_for(self._zh_model_path),
                "model_path": str(self._zh_model_path) if self._zh_model_path else None,
                "voices_path": str(self._zh_voices_path) if self._zh_voices_path else None,
                "threads": self._num_threads,
                "streaming": False,
            },
        )

    @classmethod
    def _voice_language_supported(cls, voice: str) -> bool:
        try:
            return cls._language_for_voice(voice) in cls.capabilities.languages
        except ValueError:
            return False

    def _default_voice(self, language: str | None = None) -> str:
        """Pick a sensible default voice, honouring the target ``language``.

        Chinese (``cmn``) text must use a ``z*`` Mandarin voice; feeding Mandarin
        characters to the default English voice makes espeak-ng emit the literal
        placeholder ``chinese letter`` for every character.
        """
        preferred: tuple[str, ...] = ()
        if language == "cmn":
            preferred = ("zf_xiaoxiao", "zf_xiaobei", "zm_yunjian", "zm_yunxi")
        elif language in (None, "en-us", "en-gb"):
            preferred = ("af_heart", "af_sarah")
        for candidate in preferred:
            if candidate in self.available_voices:
                return candidate
        if language:
            for voice in self.available_voices:
                try:
                    if self._language_for_voice(voice) == language:
                        return voice
                except ValueError:
                    continue
        return self.available_voices[0]

    @staticmethod
    def _contains_cjk(text: str) -> bool:
        """True when ``text`` carries CJK ideographs (Chinese-dominant input)."""
        return any("\u4e00" <= char <= "\u9fff" for char in text)

    def _resolve_language(
        self, *, text: str, voice: str | None, kwargs: dict[str, Any]
    ) -> str | None:
        """Resolve the phonemization language for a request.

        Priority: explicit ``language``/``lang`` kwarg, then an explicit ``voice``
        (its prefix implies the language), then content detection. When the caller
        supplied neither a voice nor a language (the benchmark default) and the
        text contains Chinese, ``cmn`` is selected so espeak-ng phonemizes Mandarin
        correctly instead of emitting ``chinese letter`` placeholders. ``None``
        means "no opinion" and lets the default English voice decide.
        """
        explicit = kwargs.get("language") or kwargs.get("lang")
        if explicit:
            return str(explicit)
        if voice:
            return self._language_for_voice(voice)
        if self._contains_cjk(text):
            return "cmn"
        return None

    @staticmethod
    def _language_for_voice(voice: str) -> str:
        prefix = voice[:1].lower()
        mapping = {
            "a": "en-us",
            "b": "en-gb",
            "f": "fr-fr",
            "j": "ja",
            "k": "ko",
            "z": "cmn",
            "e": "es",
            "i": "it",
            "p": "pt-br",
            "h": "hi",
        }
        try:
            return mapping[prefix]
        except KeyError as exc:
            raise ValueError(f"cannot infer Kokoro language from voice: {voice}") from exc

    def _quantization_hint(self) -> str:
        return self._quantization_hint_for(self._model_path)

    @staticmethod
    def _quantization_hint_for(path: Path | None) -> str:
        name = path.name.lower() if path else ""
        for token in ("q4", "q8", "int8", "fp16"):
            if token in name:
                return token
        return "fp32/model-defined"

    def _require_loaded(self) -> Any:
        if not self.is_loaded or self._engine is None:
            raise ModelNotLoadedError(self.id)
        return self._engine
