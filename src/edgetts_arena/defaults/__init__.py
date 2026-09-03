from __future__ import annotations

from importlib.resources import files


def read_default_text(name: str) -> str:
    """Read a configuration resource shipped inside the wheel."""
    return files(__package__).joinpath(name).read_text(encoding="utf-8")
