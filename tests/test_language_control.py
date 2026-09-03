import pytest

from edgetts_arena.api.schemas import BenchmarkConfig
from edgetts_arena.core.benchmark_service import BenchmarkService
from edgetts_arena.core.errors import ArenaError


def test_benchmark_config_normalizes_language_code() -> None:
    config = BenchmarkConfig(language=" ZH ")
    assert config.language == "zh"


def test_benchmark_service_forwards_capability_gated_language() -> None:
    warnings: list[str] = []
    kwargs = BenchmarkService._normalize_infer_config(
        model_id="qwen",
        info={"voices": []},
        capabilities={
            "speed": False,
            "voices": True,
            "seed": False,
            "language_control": True,
            "languages": ["zh", "en", "ja"],
        },
        config={"speed": 1.0, "language": "ZH"},
        warnings=warnings,
    )
    assert kwargs == {"language": "zh"}
    assert warnings == []


def test_benchmark_service_rejects_language_for_fixed_language_model() -> None:
    with pytest.raises(ArenaError, match="does not support explicit language control"):
        BenchmarkService._normalize_infer_config(
            model_id="melotts-zh",
            info={"voices": []},
            capabilities={
                "speed": True,
                "voices": True,
                "seed": False,
                "language_control": False,
                "languages": ["zh", "en"],
            },
            config={"speed": 1.0, "language": "zh"},
            warnings=[],
        )


def test_benchmark_service_rejects_unsupported_language_code() -> None:
    with pytest.raises(ArenaError, match="language 'fr' is not available"):
        BenchmarkService._normalize_infer_config(
            model_id="kokoro",
            info={"voices": []},
            capabilities={
                "speed": True,
                "voices": True,
                "seed": False,
                "language_control": True,
                "languages": ["en-us", "en-gb"],
            },
            config={"speed": 1.0, "language": "fr"},
            warnings=[],
        )
