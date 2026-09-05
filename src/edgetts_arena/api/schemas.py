from __future__ import annotations

from typing import Any, Literal
import unicodedata

from pydantic import BaseModel, Field, field_validator


class ErrorDetail(BaseModel):
    type: str
    details: Any | None = None


class APIEnvelope(BaseModel):
    code: int = 200
    message: str = "success"
    data: Any | None = None
    error: ErrorDetail | None = None


class BenchmarkConfig(BaseModel):
    voice: str | None = None
    language: str | None = Field(default=None, min_length=2, max_length=32)
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    seed: int | None = None
    sample_rate: int | None = Field(default=None, ge=8_000, le=192_000)

    @field_validator("language")
    @classmethod
    def normalize_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("language must not be blank")
        return normalized


class BenchmarkRunRequest(BaseModel):
    text: str = Field(min_length=1, max_length=1000)
    models: list[str] = Field(min_length=1, max_length=4)
    execution_mode: Literal["sequential", "concurrent"] = "sequential"
    cpu_threads_per_model: int = Field(default=4, ge=1, le=64)
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value).strip()
        if not value:
            raise ValueError("text must not be blank")
        if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t" for ch in value):
            raise ValueError("text contains unsupported control characters")
        return value

    @field_validator("models")
    @classmethod
    def unique_models(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("model ids must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("model ids must be unique")
        return normalized


class BenchmarkSuiteRunRequest(BaseModel):
    models: list[str] = Field(min_length=1, max_length=4)
    case_ids: list[str] | None = Field(default=None, min_length=1, max_length=5)
    cpu_threads_per_model: int = Field(default=4, ge=1, le=64)
    warmup_runs: int | None = Field(default=None, ge=0, le=10)
    measured_runs: int | None = Field(default=None, ge=1, le=20)
    config: BenchmarkConfig = Field(default_factory=BenchmarkConfig)

    @field_validator("models")
    @classmethod
    def unique_suite_models(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("model ids must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("model ids must be unique")
        return normalized

    @field_validator("case_ids")
    @classmethod
    def unique_case_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("case ids must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("case ids must be unique")
        return normalized


class ResidencyConfigRequest(BaseModel):
    """Runtime residency policy controlled from the UI.

    ``mode`` selects the unload strategy (``eager`` = unload after every run;
    ``keep_warm`` = keep the last batch resident until a later run drops it).
    ``memory_aware`` gates multi-model residency on available memory, and
    ``resident_memory_budget_mb`` caps the combined footprint of warm models.
    """

    mode: Literal["eager", "keep_warm"] = "eager"
    memory_aware: bool = True
    resident_memory_budget_mb: int = Field(default=4096, ge=256, le=262144)


class StreamingStart(BaseModel):
    action: Literal["start"]
    text: str = Field(min_length=1, max_length=1000)
    voice: str | None = None
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        value = unicodedata.normalize("NFC", value).strip()
        if not value:
            raise ValueError("text must not be blank")
        if any(unicodedata.category(ch) == "Cc" and ch not in "\n\t" for ch in value):
            raise ValueError("text contains unsupported control characters")
        return value
