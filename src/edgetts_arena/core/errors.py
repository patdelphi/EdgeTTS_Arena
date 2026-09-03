from __future__ import annotations


class ArenaError(Exception):
    """Base application exception carrying a stable application error code."""

    def __init__(self, code: int, message: str, *, error_type: str = "arena_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.error_type = error_type


class ModelNotLoadedError(ArenaError):
    def __init__(self, model_id: str) -> None:
        super().__init__(
            1002,
            f"model '{model_id}' is not loaded",
            error_type="model_not_loaded",
        )
