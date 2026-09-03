from __future__ import annotations


def test_system_environment_uses_one_host_memory_snapshot(monkeypatch) -> None:
    import edgetts_arena.core.system_info as module

    gib = 1024**3

    class Memory:
        total = 16 * gib
        available = 8 * gib

    captured: dict[str, int | None] = {}

    def effective(*, host_available_bytes: int | None = None) -> int:
        captured["host_available_bytes"] = host_available_bytes
        assert host_available_bytes is not None
        return host_available_bytes - 512 * 1024 * 1024

    monkeypatch.setattr(module.psutil, "virtual_memory", lambda: Memory())
    monkeypatch.setattr(module, "effective_available_memory_bytes", effective)

    environment = module.collect_system_environment()
    assert captured["host_available_bytes"] == Memory.available
    assert environment["available_ram_gb"] == 8.0
    assert environment["available_ram_effective_gb"] == 7.5
    assert environment["available_ram_effective_gb"] <= environment["available_ram_gb"]
