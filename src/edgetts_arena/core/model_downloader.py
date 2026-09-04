"""
模型下载器模块

支持从 Hugging Face Hub 和其他来源下载 TTS 模型。
下载路径遵循多级搜索路径配置。
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# 模型下载源配置
MODEL_SOURCES: dict[str, dict[str, Any]] = {
    "piper": {
        "type": "huggingface",
        "repo_id": "rhasspy/piper-voices",
        "files": [
            # 中文女声 - huayan
            {"source": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx", 
             "target": "zh_CN-huayan-medium.onnx"},
            {"source": "zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json",
             "target": "zh_CN-huayan-medium.onnx.json"},
        ],
        "description": "Piper TTS 中文女声 (huayan)",
        "size_mb": 60,
    },
    "kokoro": {
        "type": "github",
        "repo": "thewh1teagle/kokoro-onnx",
        "files": [
            # ONNX 模型和 voices 文件从 thewh1teagle/kokoro-onnx releases 下载
            {"url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", "target": "kokoro-v1.0.onnx"},
            {"url": "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", "target": "voices-v1.0.bin"},
        ],
        "description": "Kokoro TTS ONNX v1.0 (Apache-2.0)",
        "size_mb": 660,
    },
    "qwen3-tts-0.6b": {
        "type": "huggingface",
        "repo_id": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        "revision": "f3d1af06e4eaefac12b1ffa6726f9eef674a6f02",
        "description": "Qwen3-TTS 0.6B CustomVoice 模型",
        "size_mb": 1200,
    },
    "cosyvoice-300m-sft": {
        "type": "huggingface",
        "repo_id": "FunAudioLLM/CosyVoice-300M-SFT",
        "description": "CosyVoice 300M SFT 模型",
        "size_mb": 1500,
    },
    "melotts-zh": {
        "type": "huggingface",
        "repo_id": "myshell-ai/MeloTTS-Chinese",
        "revision": "082ca057e44f1e52ec47e1622a30286019e8a3ef",
        "files": [
            {"source": "config.json", "target": "config.json"},
            {"source": "checkpoint.pth", "target": "checkpoint.pth"},
        ],
        "description": "MeloTTS 中文模型",
        "size_mb": 200,
    },
}


@dataclass
class DownloadProgress:
    """下载进度信息"""
    model_id: str
    status: str  # "downloading", "extracting", "complete", "error"
    current_bytes: int = 0
    total_bytes: int = 0
    message: str = ""
    
    @property
    def progress_percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, self.current_bytes / self.total_bytes * 100)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "status": self.status,
            "current_bytes": self.current_bytes,
            "total_bytes": self.total_bytes,
            "progress_percent": self.progress_percent,
            "message": self.message,
        }


def get_model_source(model_id: str) -> dict[str, Any] | None:
    """获取模型的下载源配置"""
    return MODEL_SOURCES.get(model_id)


def list_downloadable_models() -> list[dict[str, Any]]:
    """列出所有可下载的模型"""
    result = []
    for model_id, source in MODEL_SOURCES.items():
        result.append({
            "id": model_id,
            "description": source.get("description", ""),
            "size_mb": source.get("size_mb", 0),
            "type": source.get("type", "unknown"),
            "repo_id": source.get("repo_id", ""),
        })
    return result


def resolve_download_path(
    model_id: str,
    search_paths: tuple[str, ...],
    project_root: Path,
) -> Path:
    """
    解析模型下载路径。
    下载到项目目录的 models 子目录，确保与模型配置一致。
    """
    source = MODEL_SOURCES.get(model_id)
    if not source:
        raise ValueError(f"未知的模型 ID: {model_id}")

    # 始终使用项目目录的 models 子目录作为下载目标
    # 这样与 models_config.yaml 中的路径一致
    base_path = project_root / "models"

    # 根据模型类型确定子目录
    if model_id == "piper":
        return base_path / "piper"
    elif model_id == "kokoro":
        return base_path / "kokoro"
    elif model_id == "qwen3-tts-0.6b":
        return base_path / "qwen3" / "Qwen3-TTS-12Hz-0.6B-CustomVoice"
    elif model_id == "cosyvoice-300m-sft":
        return base_path / "cosyvoice" / "CosyVoice-300M-SFT"
    elif model_id == "melotts-zh":
        return base_path / "melotts" / "zh"
    else:
        return base_path / model_id


def download_model(
    model_id: str,
    search_paths: tuple[str, ...],
    project_root: Path,
    progress_callback: Callable[[DownloadProgress], None] | None = None,
) -> dict[str, Any]:
    """
    下载指定模型。
    
    Args:
        model_id: 模型 ID
        search_paths: 模型搜索路径
        project_root: 项目根目录
        progress_callback: 进度回调函数
    
    Returns:
        下载结果字典
    """
    source = MODEL_SOURCES.get(model_id)
    if not source:
        raise ValueError(f"未知的模型 ID: {model_id}")
    
    def report(status: str, message: str, current: int = 0, total: int = 0) -> None:
        if progress_callback:
            progress_callback(DownloadProgress(
                model_id=model_id,
                status=status,
                current_bytes=current,
                total_bytes=total,
                message=message,
            ))
    
    download_path = resolve_download_path(model_id, search_paths, project_root)
    download_path.mkdir(parents=True, exist_ok=True)

    source_type = source.get("type", "huggingface")
    files = source.get("files")

    report("downloading", f"正在下载 {model_id}...", 0, 0)

    try:
        if source_type == "github":
            # 从 GitHub releases 下载
            import urllib.request
            for i, file_info in enumerate(files):
                url = file_info["url"]
                target_file = file_info.get("target", url.split("/")[-1])
                target_path = download_path / target_file
                report("downloading", f"下载 {target_file} ({i+1}/{len(files)})", i, len(files))
                urllib.request.urlretrieve(url, target_path)
        else:
            # 从 Hugging Face 下载
            try:
                from huggingface_hub import snapshot_download, hf_hub_download
            except ImportError as exc:
                raise RuntimeError("需要安装 huggingface_hub: pip install huggingface_hub") from exc

            repo_id = source.get("repo_id")
            revision = source.get("revision")

            report("downloading", f"正在从 {repo_id} 下载...", 0, 0)

            if files:
                # 下载指定文件列表
                for i, file_info in enumerate(files):
                    source_file = file_info["source"]
                    target_file = file_info.get("target", source_file)
                    report("downloading", f"下载 {source_file} ({i+1}/{len(files)})", i, len(files))

                    local_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=source_file,
                        revision=revision,
                        local_dir=str(download_path),
                    )

                    # 如果 target 与 source 不同，移动文件
                    if target_file != source_file:
                        target_path = download_path / target_file
                        shutil.move(local_path, target_path)
            else:
                # 下载整个仓库
                snapshot_download(
                    repo_id=repo_id,
                    revision=revision,
                    local_dir=str(download_path),
                )
        
        # 写入模型描述文件（根据模型类型生成不同格式）
        model_json = download_path / "model.json"
        if not model_json.exists():
            from datetime import datetime
            # 根据模型类型生成正确的 model.json 格式
            if model_id == "melotts-zh":
                descriptor = {
                    "language": "ZH",
                    "config_path": "config.json",
                    "ckpt_path": "checkpoint.pth",
                }
            elif model_id == "kokoro":
                descriptor = {
                    "source": "github",
                    "repo": source.get("repo", ""),
                    "downloaded_at": datetime.now().isoformat(),
                }
            elif model_id == "qwen3-tts-0.6b":
                descriptor = {
                    "source": "huggingface",
                    "repo_id": source.get("repo_id", ""),
                    "revision": source.get("revision", ""),
                    "downloaded_at": datetime.now().isoformat(),
                }
            elif model_id == "cosyvoice-300m-sft":
                descriptor = {
                    "source": "huggingface",
                    "repo_id": source.get("repo_id", ""),
                    "downloaded_at": datetime.now().isoformat(),
                }
            elif model_id == "piper":
                descriptor = {
                    "source": "huggingface",
                    "repo_id": source.get("repo_id", ""),
                    "downloaded_at": datetime.now().isoformat(),
                }
            else:
                descriptor = {
                    "source": source_type,
                    "downloaded_at": datetime.now().isoformat(),
                }
            model_json.write_text(json.dumps(descriptor, ensure_ascii=False, indent=2), encoding="utf-8")

        report("complete", f"模型已下载到 {download_path}", 1, 1)

        return {
            "success": True,
            "model_id": model_id,
            "path": str(download_path),
            "message": f"模型下载完成: {download_path}",
        }
        
    except Exception as exc:
        report("error", f"下载失败: {exc}", 0, 0)
        return {
            "success": False,
            "model_id": model_id,
            "path": str(download_path),
            "error": str(exc),
            "message": f"下载失败: {exc}",
        }


def check_model_downloaded(model_id: str, search_paths: tuple[str, ...], project_root: Path) -> bool:
    """检查模型是否已下载"""
    try:
        path = resolve_download_path(model_id, search_paths, project_root)
        return path.exists() and any(path.iterdir())
    except Exception:
        return False
