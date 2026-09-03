# CPU 推理与量化部署实操指南

> 版本：v0.3 ｜ 已纳入主 Arena 本地部署与 CosyVoice/MeloTTS 独立真实 CPU gate 结论

## 1. CPU 优化顺序

1. **限制线程**：先控制 OMP/MKL/PyTorch/ONNX Runtime 线程数。
2. **选择稳定 CPU runtime**：优先模型官方支持路径，其次再评估社区转换方案。
3. **减少内存带宽压力**：在模型支持时评估 INT8/低位量化。
4. **减少常驻模型数**：大模型默认 lazy load。
5. **Affinity / cgroup-aware budget**：仅在平台支持时启用，并以进程 affinity、cgroup quota、host CPU 中最保守值作为预算。

## 2. 本地部署拓扑

当前冻结为两类环境：

### 2.1 主 Arena 环境

用于 Dummy / Piper / Kokoro、FastAPI、Gradio UI、Standard Benchmark Suite。

推荐 Python 3.11：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,piper,kokoro]"
edgetts-arena doctor --ui --exports-root exports/doctor
```

主 CI 已在 Python 3.10/3.11/3.12 验证；Windows/macOS/Linux 均通过 Doctor、Dummy/API/spawn smoke。

### 2.2 Extended model 独立环境

CosyVoice/MeloTTS 虽已通过独立真实 CPU gate，但当前继续 `experimental + disabled`。**不要把官方 CosyVoice/MeloTTS 的完整依赖直接安装进主 UI venv。** CosyVoice pinned requirements 会带入旧版 Gradio/FastAPI/Pydantic，可能覆盖主 Arena 的 UI 依赖。

在正式 model-worker/venv 隔离完成前，扩展模型应在独立 venv 中执行 `scripts/real_model_smoke.py` 或等价验证。

## 3. 通用线程配置

在加载 Torch/ONNX Runtime 前设置：

```python
import os

threads = 4
os.environ["OMP_NUM_THREADS"] = str(threads)
os.environ["MKL_NUM_THREADS"] = str(threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
```

PyTorch：

```python
import torch
torch.set_num_threads(threads)
```

ONNX Runtime：

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = threads
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
```

实际 benchmark 仍以 ResourceGuard 生成的 effective thread budget 为准。

## 4. Piper 路线

Piper 是主 Arena 超轻量 baseline：

- Runtime：`piper-tts>=1.4,<2`。
- `.onnx` 与相邻 `.onnx.json` 必须同时存在。
- CI 使用官方 `en_US-lessac-low` 真模型完成 CPU synthesis gate。
- 可用 CLI：

```bash
python -m piper.download_voices en_US-lessac-low --data-dir models/piper
edgetts-arena piper --model models/piper/en_US-lessac-low.onnx --text "Piper test" --threads 2 --output exports/piper.wav
```

## 5. Kokoro 路线

主 Arena 采用 `kokoro-onnx` + ONNX Runtime，优先使用 v1.0 int8：

- `kokoro-v1.0.int8.onnx`
- `voices-v1.0.bin`
- CPU provider only
- Adapter 当前同步 streaming capability 为 false，因此 TTFB 为空，不伪造首包指标

CI 已完成真实 CPU gate。完整资产下载命令见 `12_本地部署与验收指南.md`。

## 6. Qwen3-TTS / 较大模型路线

Qwen3-TTS 0.6B 在本项目中仍属于 **experimental placeholder**：

1. 不批准未经本项目验证的 CPU runtime/量化格式。
2. 不生成假音频或伪 capability。
3. 后续比较官方 Python CPU、ncnn/C++ 与其他可复现路线。
4. 转正式前必须完成固定 revision、资产校验、RAM/RTF、license 与跨平台 gate。

## 7. CosyVoice 300M SFT 已验证路线

当前冻结实现：

- Upstream：QwenAudio/CosyVoice pinned source。
- 模型：CosyVoice-300M-SFT pinned model revision。
- CPU runtime：PyTorch CPU + ONNX Runtime CPU。
- Adapter 模式：SFT speaker，不把 zero-shot prompt audio/text 强塞进现有 voice-id schema。
- 真实 gate：Ubuntu 24.04 / Python 3.10 / 2 threads。
- 单次 gate 记录：4.098s 音频、17.875s inference、RTF 4.362、peak RSS 4181 MB；仅用于 gate 追溯。

### 7.1 WeText 离线前端

`wetext==0.0.4` 在没有显式路径时会尝试 `snapshot_download()`。项目侧已禁止依赖该隐式行为：

```bash
python scripts/prepare_cosyvoice_frontend.py --output models/cosyvoice/wetext
```

脚本只下载 5 个必需 FST：

```text
en/tn/tagger.fst
en/tn/verbalizer.fst
zh/tn/tagger.fst
zh/tn/verbalizer.fst
zh/tn/verbalizer_remove_erhua.fst
```

并生成 `asset_manifest.json`，记录 SHA-256。运行时通过 `EDGETTS_ARENA_COSYVOICE_WETEXT_DIR` 或默认 `models/cosyvoice/wetext` 注入本地路径。

真实 gate 的 offline preflight 会先把 WeText 的 `snapshot_download()` 替换为“调用即失败”，然后执行 `CosyVoiceTTSAdapter.load_model()`；该步骤已通过，证明 load/inference 前端不依赖隐式 ModelScope 网络访问。

## 8. MeloTTS 已验证路线

- 官方 MeloTTS source + pinned Chinese model。
- PyTorch CPU。
- 模型资产通过本地 `model.json` 显式指向 config/checkpoint，不允许 Adapter 推理时隐式 HuggingFace 下载。
- 独立 Ubuntu 24.04 / Python 3.10 / 2 threads real CPU gate 已通过。
- 历史单次 gate 记录：4.415s 音频、8.005s inference、RTF 1.813、peak RSS 2960.8 MB；仅用于 gate 追溯。

## 9. PyTorch CPU 安装

不要在 requirements 的单个 requirement 行中混写 `--index-url`。推荐由独立安装步骤完成，例如：

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

实际 Torch 版本与安装方式必须跟随对应模型的 pinned runtime；不要为了统一主 Arena 环境强行升级扩展模型依赖。

## 10. 性能测试最小规则

- warm-up 至少 1 次。
- 正式测量至少 3 次。
- 线程数固定并记录 requested/effective 值。
- 同一轮对比使用同一 execution mode。
- Sequential 与 Concurrent 不混排性能排名。
- 报告保留 CPU/OS/Python/runtime/model version。
- 真实设备结果与 GitHub-hosted gate 分开记录，不把 CI 单次 RTF 当成性能承诺。
