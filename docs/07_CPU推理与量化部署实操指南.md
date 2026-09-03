# CPU 推理与量化部署实操指南

> 版本：v0.2 ｜ 本文给出工程策略，不把未经实测的社区运行时作为既定事实

## 1. CPU 优化顺序

1. **限制线程**：先控制 OMP/MKL/PyTorch/ONNX Runtime 线程数。
2. **选择稳定 CPU runtime**：优先模型官方支持路径，其次再评估社区转换方案。
3. **减少内存带宽压力**：在模型支持时评估 INT8/低位量化。
4. **减少常驻模型数**：大模型默认 lazy load。
5. **Affinity**：仅在平台支持且确有收益时启用。

## 2. 通用线程配置

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

## 3. Kokoro 路线

MVP 优先评估 ONNX Runtime CPU 或模型当前官方支持的 CPU 推理方式。

验收重点：

- 模型与 voice 文件可定位。
- 中文/英文前处理链路可复现。
- 输出采样率由模型实际返回，不在业务层固定假设。
- 记录 runtime、模型版本和转换来源。

## 4. Piper 路线

Piper 作为轻量 baseline，应优先完成：

- CLI/runtime 能在 CPU 上生成 WAV。
- Adapter 统一转换输出数组。
- 记录 voice/model 文件版本。
- 不要求 streaming capability。

## 5. Qwen3-TTS / 较大模型路线

Qwen3-TTS 0.6B 在本项目中属于 **experimental adapter**。实际开发按以下顺序验证：

1. 确认模型当前官方 CPU 支持方式。
2. 建立最小 CPU inference POC。
3. 测量 FP/BF16/量化路径的实际 RAM 与 RTF。
4. 只有在兼容性得到验证后，才把 GGUF/ONNX/其他社区转换作为正式配置选项。
5. 若运行时依赖与主环境冲突，再提升为独立 worker/venv。

不得在验收标准中使用“音质无损”“固定提速百分比”等未经本项目实测的结论。

## 6. CosyVoice / Flow 类模型

第二批模型应通过 POC 确认：

- CPU runtime 是否稳定。
- 流式语义是否为真实可播放 chunk。
- NFE/步数调整对音质和速度的实际影响。

任何步数降低带来的性能提升都必须写入 benchmark report，而非作为固定事实。

## 7. PyTorch CPU 安装

不要在 requirements 的单个 requirement 行中混写 `--index-url`。

推荐由安装脚本或开发文档单独执行，例如：

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
```

实际 Torch 安装方式按操作系统/架构调整。

## 8. 性能测试最小规则

- warm-up 至少 1 次。
- 正式测量至少 3 次。
- 线程数固定。
- 同一轮对比使用同一 execution mode。
- 报告保留 CPU/OS/Python/runtime/model version。
