from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable

import numpy as np
import soundfile as sf

from edgetts_arena.core.base_adapter import BaseTTSAdapter, TTSOutput
from edgetts_arena.core.capabilities import TTSCapabilities
from edgetts_arena.core.errors import ArenaError, ModelNotLoadedError

CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]

_OFFICIAL_SPEAKERS = (
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
)
_LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "ru": "Russian",
    "russian": "Russian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "es": "Spanish",
    "spanish": "Spanish",
    "it": "Italian",
    "italian": "Italian",
}
_QUANT_FLAGS = {"bf16": None, "int8": "--int8", "int4": "--int4"}


class Qwen3NativeTTSAdapter(BaseTTSAdapter):
    """Subprocess adapter for the pinned community pure-C Qwen3-TTS runtime.

    This adapter is deliberately separate from :class:`Qwen3TTSAdapter`, which
    remains the official ``qwen-tts`` FP32 compatibility baseline. The native
    runtime is optional and experimental; a local manifest pins the executable,
    model directory, runtime revision and quantization mode.
    """

    id = "qwen3-tts-0.6b-native"
    capabilities = TTSCapabilities(
        streaming=False,
        seed=True,
        speed=False,
        voices=True,
        voice_clone=False,
        languages=("zh", "en", "ja", "ko", "de", "fr", "ru", "pt", "es", "it"),
    )
    available_voices: tuple[str, ...] = _OFFICIAL_SPEAKERS

    def __init__(self, *, command_runner: CommandRunner | None = None) -> None:
        self._command_runner = command_runner
        self._manifest_path: Path | None = None
        self._binary: Path | None = None
        self._model_dir: Path | None = None
        self._quantization = "int8"
        self._runtime_revision = "unknown"
        self._runtime_caps = ""
        self._default_language = "English"
        self._default_voice = "Vivian"
        self._num_threads = 1
        self.is_loaded = False

    def load_model(
        self,
        model_path: str,
        *,
        device: str = "cpu",
        num_threads: int = 4,
    ) -> None:
        if device != "cpu":
            raise ValueError("Qwen3 native adapter currently supports CPU only")
        if num_threads < 1:
            raise ValueError("num_threads must be >= 1")

        manifest_path = Path(model_path).expanduser()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Qwen3 native manifest not found: {manifest_path}")
        manifest = self._read_json(manifest_path, "Qwen3 native manifest")
        if str(manifest.get("runtime") or "") != "qwen3-tts-c":
            raise ValueError("Qwen3 native manifest runtime must be 'qwen3-tts-c'")

        base = manifest_path.resolve().parent
        binary = self._resolve_manifest_path(base, manifest.get("binary"), "binary")
        model_dir = self._resolve_manifest_path(base, manifest.get("model_dir"), "model_dir")
        if not binary.is_file():
            raise FileNotFoundError(f"Qwen3 native binary not found: {binary}")
        if not model_dir.is_dir():
            raise FileNotFoundError(f"Qwen3 native model directory not found: {model_dir}")
        self._validate_checkpoint(model_dir)

        quantization = str(manifest.get("quantization") or "int8").strip().lower()
        if quantization not in _QUANT_FLAGS:
            raise ValueError("Qwen3 native quantization must be one of: bf16, int8, int4")
        default_language = self._normalize_language(manifest.get("default_language") or "English")
        default_voice = self._select_voice(str(manifest.get("default_voice") or "Vivian"))

        preflight = self._run_command([str(binary), "--caps"])
        if preflight.returncode != 0:
            raise ArenaError(
                1002,
                "Qwen3 native runtime preflight failed: " + self._diagnostic(preflight),
                error_type="model_unavailable",
            )

        self._manifest_path = manifest_path.resolve()
        self._binary = binary
        self._model_dir = model_dir
        self._quantization = quantization
        self._runtime_revision = str(manifest.get("runtime_revision") or "unknown")
        self._runtime_caps = (preflight.stdout or preflight.stderr or "").strip()[-512:]
        self._default_language = default_language
        self._default_voice = default_voice
        self._num_threads = num_threads
        self.is_loaded = True

    def infer(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> TTSOutput:
        binary, model_dir = self._require_loaded()
        normalized = text.strip()
        if not normalized:
            raise ValueError("text must not be empty")
        if speed != 1.0:
            raise ArenaError(
                1003,
                "Qwen3 native adapter does not expose semantic speed control",
                error_type="capability_conflict",
            )

        selected_voice = self._select_voice(voice or self._default_voice)
        language = self._normalize_language(kwargs.pop("language", self._default_language))
        seed = kwargs.pop("seed", None)
        if kwargs:
            unsupported = ", ".join(sorted(str(key) for key in kwargs))
            raise ValueError(f"unsupported Qwen3 native inference options: {unsupported}")

        with tempfile.TemporaryDirectory(prefix="edgetts-arena-qwen3-native-") as temp_dir:
            output_path = Path(temp_dir) / "output.wav"
            command = [
                str(binary),
                "-d", str(model_dir),
                "--text", normalized,
                "-o", str(output_path),
                "-s", selected_voice,
                "-l", language,
                "-j", str(self._num_threads),
                "--silent",
            ]
            quant_flag = _QUANT_FLAGS[self._quantization]
            if quant_flag is not None:
                command.append(quant_flag)
            if seed is not None:
                command.extend(["--seed", str(int(seed))])

            result = self._run_command(command)
            if result.returncode != 0:
                raise RuntimeError("Qwen3 native synthesis failed: " + self._diagnostic(result))
            if not output_path.is_file() or output_path.stat().st_size <= 44:
                raise RuntimeError("Qwen3 native runtime did not produce a valid WAV file")
            audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)

        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1, dtype=np.float32)
        audio = audio.reshape(-1)
        if audio.size == 0 or int(sample_rate) <= 0:
            raise RuntimeError("Qwen3 native runtime returned empty/invalid audio")
        if not np.isfinite(audio).all():
            raise RuntimeError("Qwen3 native runtime returned non-finite audio")

        return TTSOutput(
            audio=audio,
            sample_rate=int(sample_rate),
            metadata={
                "adapter": self.id,
                "voice": selected_voice,
                "language": language,
                "runtime": "gabriele-mastrapasqua/qwen3-tts",
                "runtime_revision": self._runtime_revision,
                "runtime_caps": self._runtime_caps,
                "model_version": model_dir.name,
                "quantization": self._quantization,
                "threads": self._num_threads,
                "streaming": False,
                "mode": "native_cli",
            },
        )

    def unload_model(self) -> None:
        self._manifest_path = None
        self._binary = None
        self._model_dir = None
        self._runtime_caps = ""
        self.is_loaded = False

    def _run_command(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if self._command_runner is not None:
            return self._command_runner(command)
        try:
            return subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            return subprocess.CompletedProcess(command, 127, stdout="", stderr=str(exc))

    @staticmethod
    def _read_json(path: Path, label: str) -> dict[str, Any]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid {label}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be a JSON object")
        return raw

    @classmethod
    def _validate_checkpoint(cls, model_dir: Path) -> None:
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Qwen3 native model config not found: {config_path}")
        config = cls._read_json(config_path, "Qwen3 model config")
        if str(config.get("tts_model_type") or "").lower() != "custom_voice":
            raise ValueError("Qwen3 native adapter requires a CustomVoice checkpoint")
        if str(config.get("tts_model_size") or "").lower() != "0b6":
            raise ValueError("Qwen3 native adapter requires the 0.6B (0b6) checkpoint")

    @staticmethod
    def _resolve_manifest_path(base: Path, value: Any, field: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError(f"Qwen3 native manifest field '{field}' must not be empty")
        expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
        return (base / expanded).resolve() if not expanded.is_absolute() else expanded.resolve()

    @staticmethod
    def _select_voice(requested: str) -> str:
        wanted = requested.strip().lower()
        for voice in _OFFICIAL_SPEAKERS:
            if voice.lower() == wanted:
                return voice
        raise ValueError(f"unknown Qwen3 native speaker: {requested}")

    @staticmethod
    def _normalize_language(language: Any) -> str:
        key = str(language or "").strip().lower()
        try:
            return _LANGUAGE_ALIASES[key]
        except KeyError as exc:
            raise ValueError(f"unsupported Qwen3 native language: {language}") from exc

    @staticmethod
    def _diagnostic(result: subprocess.CompletedProcess[str]) -> str:
        text = (result.stderr or result.stdout or f"exit code {result.returncode}").strip()
        return text[-2048:]

    def _require_loaded(self) -> tuple[Path, Path]:
        if not self.is_loaded or self._binary is None or self._model_dir is None:
            raise ModelNotLoadedError(self.id)
        return self._binary, self._model_dir
