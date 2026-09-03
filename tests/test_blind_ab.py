import random

import pytest

from edgetts_arena.core.blind_ab import (
    blind_session_complete,
    create_blind_session,
    record_blind_rating,
    reveal_blind_session,
)
from edgetts_arena.core.errors import ArenaError


def _results():
    return [
        {"model_id": "alpha", "status": "success", "audio_url": "/a.wav"},
        {"model_id": "beta", "status": "success", "audio_url": "/b.wav"},
        {"model_id": "failed", "status": "error", "audio_url": None},
    ]


def test_blind_session_uses_only_successful_results_and_hides_mapping_until_reveal() -> None:
    session = create_blind_session("run_test", _results(), rng=random.Random(3))
    assert [item["label"] for item in session["samples"]] == ["A", "B"]
    assert {item["model_id"] for item in session["samples"]} == {"alpha", "beta"}
    assert "mapping" not in session
    assert session["revealed"] is False


def test_blind_scores_require_every_sample_before_reveal() -> None:
    session = create_blind_session("run_test", _results(), rng=random.Random(2))
    session = record_blind_rating(
        session,
        "A",
        naturalness=4,
        intelligibility=5,
        prosody=3,
    )
    assert blind_session_complete(session) is False
    with pytest.raises(ArenaError):
        reveal_blind_session(session)

    session = record_blind_rating(
        session,
        "B",
        naturalness=5,
        intelligibility=4,
        prosody=4,
    )
    assert blind_session_complete(session) is True
    revealed = reveal_blind_session(session)
    assert revealed["revealed"] is True
    assert set(revealed["mapping"].values()) == {"alpha", "beta"}


def test_blind_rating_range_is_enforced() -> None:
    session = create_blind_session("run_test", _results(), rng=random.Random(1))
    with pytest.raises(ArenaError):
        record_blind_rating(
            session,
            "A",
            naturalness=6,
            intelligibility=4,
            prosody=4,
        )


def test_blind_scores_can_be_persisted_into_run_export(tmp_path) -> None:
    import zipfile

    from edgetts_arena.core.artifacts import RunArtifactStore

    store = RunArtifactStore(tmp_path / "exports")
    store.create_run("run_blind")
    store.write_json("run_blind", "benchmark_report.json", {"data": {}})
    store.write_json("run_blind", "environment.json", {"os": "test"})
    store.write_json("run_blind", "blind_scores.json", {"run_id": "run_blind"})
    archive_path = store.build_export("run_blind")
    with zipfile.ZipFile(archive_path) as archive:
        assert "blind_scores.json" in archive.namelist()
