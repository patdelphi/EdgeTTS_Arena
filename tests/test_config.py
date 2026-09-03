from pathlib import Path

from edgetts_arena.core.config import load_settings


def test_load_settings_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "app.yaml"
    path.write_text(
        "host: 127.0.0.1\nport: 9001\ndefault_num_threads: 2\n"
        "resource_guard:\n  min_available_memory_mb_soft: 1000\n  min_available_memory_mb_hard: 500\n",
        encoding="utf-8",
    )
    settings = load_settings(path)
    assert settings.port == 9001
    assert settings.default_num_threads == 2
    assert settings.resource_guard.min_available_memory_mb_hard == 500
