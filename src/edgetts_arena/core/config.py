from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from edgetts_arena.defaults import read_default_text

_DEFAULT_APP_CONFIG = Path("config/app_config.yaml")


@dataclass(frozen=True, slots=True)
class ResourceGuardSettings:
    min_available_memory_mb_soft: int = 1536
    min_available_memory_mb_hard: int = 768
    min_available_memory_mb_per_concurrent_model: int = 512
    max_concurrent_models: int = 4


@dataclass(frozen=True, slots=True)
class AppSettings:
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "INFO"
    default_num_threads: int = 4
    inference_timeout_sec: int = 60
    resource_guard: ResourceGuardSettings = ResourceGuardSettings()
    # 模型搜索路径（按优先级顺序），支持环境变量展开
    model_search_paths: tuple[str, ...] = (
        "${HF_HOME:-~/.cache}/huggingface/hub",
        "./models",
    )


def _expand_env_var(value: str) -> str:
    """展开环境变量，支持 ${VAR:-default} 语法"""
    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(3) if match.group(2) else ""
        return os.environ.get(var_name, os.path.expanduser(default) if default else "")
    
    # 匹配 ${VAR} 或 ${VAR:-default}
    pattern = r'\$\{([^}:]+)(:-([^}]*))?\}'
    return re.sub(pattern, replace_match, value)


def _resolve_search_paths(paths: list[str], project_root: Path) -> tuple[str, ...]:
    """解析搜索路径列表，展开环境变量并转换为绝对路径"""
    resolved = []
    for p in paths:
        expanded = _expand_env_var(p)
        path = Path(expanded)
        if not path.is_absolute():
            path = project_root / path
        resolved.append(str(path.resolve()))
    return tuple(resolved)


def _read_yaml(path: Path) -> dict[str, Any]:
    if path.exists():
        text = path.read_text(encoding="utf-8")
    elif path == _DEFAULT_APP_CONFIG:
        text = read_default_text("app_config.yaml")
    else:
        return {}
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError("application config root must be a mapping")
    return data


def load_settings(path: str | Path = _DEFAULT_APP_CONFIG) -> AppSettings:
    config_path = Path(path)
    raw = _read_yaml(config_path)
    guard_raw = raw.get("resource_guard", {}) or {}
    guard = ResourceGuardSettings(
        min_available_memory_mb_soft=int(
            os.getenv("EDGETTS_ARENA_MIN_MEMORY_SOFT_MB", guard_raw.get("min_available_memory_mb_soft", 1536))
        ),
        min_available_memory_mb_hard=int(
            os.getenv("EDGETTS_ARENA_MIN_MEMORY_HARD_MB", guard_raw.get("min_available_memory_mb_hard", 768))
        ),
        min_available_memory_mb_per_concurrent_model=int(
            os.getenv(
                "EDGETTS_ARENA_MIN_MEMORY_PER_CONCURRENT_MODEL_MB",
                guard_raw.get("min_available_memory_mb_per_concurrent_model", 512),
            )
        ),
        max_concurrent_models=int(
            os.getenv("EDGETTS_ARENA_MAX_CONCURRENT_MODELS", guard_raw.get("max_concurrent_models", 4))
        ),
    )
    
    # 解析模型搜索路径
    project_root = config_path.parent.parent if config_path.exists() else Path.cwd()
    search_paths_raw = raw.get("model_search_paths") or [
        "${HF_HOME:-~/.cache}/huggingface/hub",
        "./models",
    ]
    search_paths = _resolve_search_paths(search_paths_raw, project_root)
    
    return AppSettings(
        host=os.getenv("EDGETTS_ARENA_HOST", raw.get("host", "127.0.0.1")),
        port=int(os.getenv("EDGETTS_ARENA_PORT", raw.get("port", 8000))),
        log_level=os.getenv("EDGETTS_ARENA_LOG_LEVEL", raw.get("log_level", "INFO")),
        default_num_threads=int(os.getenv("EDGETTS_ARENA_DEFAULT_THREADS", raw.get("default_num_threads", 4))),
        inference_timeout_sec=int(
            os.getenv("EDGETTS_ARENA_INFERENCE_TIMEOUT_SEC", raw.get("inference_timeout_sec", 60))
        ),
        resource_guard=guard,
        model_search_paths=search_paths,
    )
