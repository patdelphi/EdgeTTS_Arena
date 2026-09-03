from __future__ import annotations

from copy import deepcopy
from secrets import SystemRandom
from typing import Any

from edgetts_arena.core.errors import ArenaError


BLIND_LABELS = ("A", "B", "C", "D")


def create_blind_session(
    run_id: str,
    results: list[dict[str, Any]],
    *,
    rng: Any | None = None,
) -> dict[str, Any]:
    successful = [
        item
        for item in results
        if item.get("status") == "success" and item.get("model_id") and item.get("audio_url")
    ]
    if len(successful) < 2:
        raise ArenaError(
            1001,
            "Blind AB requires at least two successful model results",
            error_type="blind_session_unavailable",
        )
    if len(successful) > len(BLIND_LABELS):
        raise ArenaError(1001, "Blind AB supports at most four samples", error_type="validation_error")

    shuffled = list(successful)
    shuffler = rng or SystemRandom()
    shuffler.shuffle(shuffled)
    samples = [
        {
            "label": BLIND_LABELS[index],
            "model_id": item["model_id"],
            "audio_url": item["audio_url"],
        }
        for index, item in enumerate(shuffled)
    ]
    return {
        "run_id": run_id,
        "samples": samples,
        "ratings": {},
        "revealed": False,
    }


def record_blind_rating(
    session: dict[str, Any],
    label: str,
    *,
    naturalness: int | float,
    intelligibility: int | float,
    prosody: int | float,
) -> dict[str, Any]:
    if session.get("revealed"):
        raise ArenaError(1001, "blind session is already revealed", error_type="blind_session_closed")
    valid_labels = {item["label"] for item in session.get("samples", [])}
    if label not in valid_labels:
        raise ArenaError(1001, f"unknown blind sample '{label}'", error_type="validation_error")

    values = {
        "naturalness": _validate_rating(naturalness),
        "intelligibility": _validate_rating(intelligibility),
        "prosody": _validate_rating(prosody),
    }
    updated = deepcopy(session)
    updated.setdefault("ratings", {})[label] = values
    return updated


def blind_session_complete(session: dict[str, Any]) -> bool:
    labels = {item["label"] for item in session.get("samples", [])}
    ratings = set((session.get("ratings") or {}).keys())
    return bool(labels) and labels <= ratings


def reveal_blind_session(session: dict[str, Any]) -> dict[str, Any]:
    if not blind_session_complete(session):
        raise ArenaError(
            1001,
            "rate every blind sample before reveal",
            error_type="blind_ratings_incomplete",
        )
    revealed = deepcopy(session)
    revealed["revealed"] = True
    revealed["mapping"] = {
        item["label"]: item["model_id"] for item in revealed.get("samples", [])
    }
    return revealed


def _validate_rating(value: int | float) -> int:
    numeric = int(value)
    if float(value) != numeric or not 1 <= numeric <= 5:
        raise ArenaError(1001, "ratings must be integer values from 1 to 5", error_type="validation_error")
    return numeric
