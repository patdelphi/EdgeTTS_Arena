from __future__ import annotations

import pytest

from edgetts_arena.core.artifacts import RunArtifactStore
from edgetts_arena.core.errors import ArenaError


def test_artifact_store_blocks_path_traversal(tmp_path) -> None:
    store = RunArtifactStore(tmp_path / "exports")
    store.create_run("run_safe")

    with pytest.raises(ArenaError) as run_exc:
        store.run_dir("../outside")
    assert run_exc.value.code == 1001

    with pytest.raises(ArenaError) as file_exc:
        store.audio_output_path("run_safe", "../escape.wav")
    assert file_exc.value.code == 1001

    with pytest.raises(ArenaError):
        store.audio_output_path("run_safe", "not-a-wave.txt")
