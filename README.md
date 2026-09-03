# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件。** Stage 0~5 MVP 已完成；Stage 6 已完成跨平台/ARM64/1-CPU smoke、Piper/Kokoro/MeloTTS/CosyVoice 真实 CPU gate、扩展模型 external Python worker、Qwen3 官方 FP32 CPU 功能路径，以及独立的 Qwen3 native INT8/INT4 hosted CPU 量化验证。Qwen3 仍默认 `experimental + disabled`：官方 FP32 保留为兼容基线，native INT8 是当前推荐的 CPU 优化候选；真实弱设备和量化音质 A/B 仍待验证。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 扩展模型与 worker：[`docs/12_第二批模型适配状态.md`](./docs/12_第二批模型适配状态.md)
- CPU/量化部署：[`docs/07_CPU推理与量化部署实操指南.md`](./docs/07_CPU推理与量化部署实操指南.md)
- 路线图：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 实现检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## 环境与启动

```text
Python >=3.10,<3.13
推荐本地首次部署：Python 3.11
```

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,piper,kokoro]"
pytest
edgetts-arena doctor --ui --exports-root exports/doctor
edgetts-arena serve --ui
```

默认 API：`http://127.0.0.1:8000`，Arena：`http://127.0.0.1:8000/arena/`，OpenAPI：`/docs`。

## Benchmark / UI

Arena 已实现 1~4 模型对比、Sequential/Concurrent、capability 驱动 controls、TC-01~05 preset、audio/metrics、ZIP、Blind AB、Standard Suite。Standard Suite 默认 warm-up 1 + measured 3。

API/Suite 的 `config.language` 与 CLI `suite --language` 已支持显式语言选择；只有 `language_control=true` 的模型才会接收该参数。Gradio 在单模型选择 Qwen/Kokoro 时提供 Language 下拉，多模型比较时使用各模型默认语言，避免跨模型强制不兼容语言。

```bash
edgetts-arena suite --models dummy --cases TC-01 TC-02 --warmup-runs 1 --measured-runs 3 --threads 2
```

## 扩展模型

默认全部 `experimental + disabled`：

- MeloTTS：hosted x86_64 real CPU gate ✅
- CosyVoice 300M SFT：hosted x86_64 real CPU gate + offline WeText ✅
- Qwen3-TTS 0.6B CustomVoice 官方 FP32：hosted real CPU synthesis ✅
- Qwen3-TTS 0.6B native INT8/INT4：pure-C pinned runtime hosted quant gate ✅

独立 Python worker 环境变量：

```text
EDGETTS_ARENA_QWEN3_PYTHON
EDGETTS_ARENA_COSYVOICE_PYTHON
EDGETTS_ARENA_MELOTTS_PYTHON
```

```bash
edgetts-arena doctor --workers
edgetts-arena doctor --worker qwen3-tts-0.6b
```

Doctor 只验证解释器、项目 source、基础依赖与 worker 协议，不代替真实模型 synthesis gate。

## Qwen3 CPU 路线

### 官方 FP32 功能基线

GitHub-hosted Ubuntu x86_64 / Python 3.11 / `qwen-tts==0.1.1` / 2 threads：9.04s 音频，48.05s inference，RTF 5.315，peak RSS 5317.8MB。该路径证明官方 CPU runtime 可运行，但不作为弱设备性能推荐。

### Native 量化候选

可选 backend 使用 pinned `gabriele-mastrapasqua/qwen3-tts` pure-C runtime，主 Arena 不需要安装 PyTorch。当前同文本、同 seed hosted 对照：

| 路线 | Threads | Audio | RTF | Peak RSS |
|---|---:|---:|---:|---:|
| native INT8 | 2 | 7.52s | **1.787** | **3124MB** |
| native INT8 | 4 | 7.52s | 1.984 | 3129MB |
| native INT4 | 4 | 8.48s | 2.432 | **2899MB** |

当前工程推荐：**native INT8 + 2 threads** 作为 CPU 优化候选。INT4 仅作为更低内存实验选项：它在该 hosted x86_64 gate 中更慢，且相同 seed/text 下输出时长与 INT8 不同，因此 Arena **不宣称 INT4 与 INT8 音质或生成行为等价**。

native 模型配置默认：

```yaml
- id: qwen3-tts-0.6b-native-int8
  enabled: false
  adapter: qwen3_native
  model_path: ./models/qwen3-native/int8/model.json
  keep_in_memory: false
  num_threads: 2
  experimental: true
  language_control: true
```

注意：native runtime 自行根据 `-j/--threads` 设置 OpenBLAS 线程。**不要预设 `OPENBLAS_NUM_THREADS`**，否则可能覆盖 runtime 的线程拓扑。

## Stage 6 剩余重点

- Qwen3 native INT8 / INT4 主观音质与 Blind AB 验证
- 真实 ARM/树莓派/低功耗弱设备实机验证
- Concurrent 真实设备资源校准
- 平台特定确定性 OOM attribution（可选）
- 扩展模型 venv/资产一键 bootstrap
