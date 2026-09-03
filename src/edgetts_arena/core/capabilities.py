from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TTSCapabilities:
    """Features that an adapter actually supports at runtime."""

    streaming: bool = False
    seed: bool = False
    speed: bool = True
    voices: bool = True
    voice_clone: bool = False
    languages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["languages"] = list(self.languages)
        return data
