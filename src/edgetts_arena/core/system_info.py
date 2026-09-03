from __future__ import annotations

import platform

import psutil


def collect_system_environment() -> dict[str, object]:
    memory = psutil.virtual_memory()
    cpu_brand = platform.processor().strip() or platform.uname().processor.strip() or "unknown"
    return {
        "os": platform.system().lower() or "unknown",
        "arch": platform.machine() or "unknown",
        "cpu_brand": cpu_brand,
        "cpu_logical_cores": int(psutil.cpu_count(logical=True) or 1),
        "cpu_physical_cores": int(psutil.cpu_count(logical=False) or 1),
        "total_ram_gb": round(memory.total / (1024**3), 3),
        "available_ram_gb": round(memory.available / (1024**3), 3),
    }
