from __future__ import annotations

import os
import platform
import sys
from importlib import metadata as importlib_metadata

import psutil

from edgetts_arena.core.resource_guard import effective_available_memory_bytes, effective_cpu_count


_PACKAGE_NAMES = (
    "edgetts-arena",
    "fastapi",
    "pydantic",
    "numpy",
    "onnxruntime",
    "piper-tts",
    "kokoro-onnx",
    "gradio",
)


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in _PACKAGE_NAMES:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def collect_system_environment(*, cpu_threads_per_model: int | None = None) -> dict[str, object]:
    memory = psutil.virtual_memory()
    cpu_brand = platform.processor().strip() or platform.uname().processor.strip() or "unknown"
    thread_settings = {
        "cpu_threads_per_model": cpu_threads_per_model,
        "omp_num_threads": os.getenv("OMP_NUM_THREADS"),
        "mkl_num_threads": os.getenv("MKL_NUM_THREADS"),
        "openblas_num_threads": os.getenv("OPENBLAS_NUM_THREADS"),
        "numexpr_num_threads": os.getenv("NUMEXPR_NUM_THREADS"),
    }
    return {
        "os": platform.system().lower() or "unknown",
        "os_release": platform.release() or "unknown",
        "arch": platform.machine() or "unknown",
        "cpu_brand": cpu_brand,
        "cpu_logical_cores": int(psutil.cpu_count(logical=True) or 1),
        "cpu_physical_cores": int(psutil.cpu_count(logical=False) or 1),
        "cpu_effective_cores": effective_cpu_count(),
        "total_ram_gb": round(memory.total / (1024**3), 3),
        "available_ram_gb": round(memory.available / (1024**3), 3),
        "available_ram_effective_gb": round(effective_available_memory_bytes() / (1024**3), 3),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_executable": sys.executable,
        "package_versions": _package_versions(),
        "thread_settings": thread_settings,
    }
