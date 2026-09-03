# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件。** Stage 0~5 MVP 已完成；Stage 6 已完成跨平台/ARM64/1-CPU smoke、Piper/Kokoro/MeloTTS/CosyVoice 真实 CPU gate、扩展模型 external Python worker、Qwen3 官方 FP32 CPU 功能路径，以及独立 Qwen3 native INT8/INT4 hosted CPU 量化验证。Qwen3 native INT8 是当前推荐 CPU 优化候选；INT8/INT4 Blind AB 工具链已实现，人工音质评分与真实弱设备验证仍待执行。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 扩展模型：[`docs/12_第二批模型适配状态.md`](./docs/12_第二批模型适配状态.md)
- CPU/量化部署：[`docs/07_CPU推理与量化部署实操指南.md`](./docs/07_CPU推理与量化部署实操指南.md)
- 路线图：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

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

默认 API：`http://127.0.0.1:8000`，Arena：`http://127.0.0.1:8000/arena/`。

## Benchmark / UI

Arena 支持 1~4 模型、Sequential/Concurrent、TC-01~05、warm-up/repeats、ZIP、Blind AB。`config.language`、CLI `suite --language` 与 Gradio Language control 均为 capability-aware。

多模型 Arena 会计算**共同 voice/language 交集**：只有所有所选模型都支持且存在共同值时才允许统一选择，否则各模型使用自身默认。这使同一模型的不同量化 backend 可以固定相同 voice/language/seed 做公平 Blind AB。

## Qwen3 hosted CPU 基线

| 路线 | Threads | Audio | RTF | Peak RSS |
|---|---:|---:|---:|---:|
| official FP32 | 2 | 9.04s | 5.315 | 5318MB |
| native INT8 | 2 | 7.52s | **1.787** | **3124MB** |
| native INT8 | 4 | 7.52s | 1.984 | 3129MB |
| native INT4 | 4 | 8.48s | 2.432 | **2899MB** |

官方 FP32 是兼容/功能基线；native INT8@2 是当前 CPU 优化候选。INT4 只作为更低内存实验项；相同 seed/text 下输出时长发生变化，因此不宣称与 INT8 音质等价。

## Qwen3 INT8 / INT4 Blind AB

一次生成成对 manifests，保证 runtime revision、模型 snapshot、默认 voice/language 一致：

```bash
python scripts/prepare_qwen3_native_variants.py \
  --binary runtime/qwen3-tts-c/qwen_tts \
  --model-dir models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output-root models/qwen3-native \
  --default-voice Vivian \
  --default-language Chinese
```

默认 catalog 已包含两个 disabled/experimental 项：

```text
qwen3-tts-0.6b-native-int8   # hosted 推荐基线：2 threads
qwen3-tts-0.6b-native-int4   # hosted 实验基线：4 threads
```

本地准备完成后将两者 `enabled: true`，启动 `edgetts-arena serve --ui`：

1. Arena 同时选择 INT8 + INT4。
2. 选择共同 Voice（如 Vivian）与 Language（如 zh），设置同一 seed。
3. Run benchmark。
4. Start Blind AB；在不知道 backend 的情况下分别评分 Naturalness / Intelligibility / Prosody。
5. 全部评分后 Reveal；结果写入 `blind_scores.json`，并随 ZIP 一起保留 benchmark metadata。

**A/B 工具链 ready ≠ 音质 gate passed。** 在真实人工评分产生前，INT4 继续保持实验路线。

注意：pinned native runtime 自己管理 OpenBLAS thread split；不要预设 `OPENBLAS_NUM_THREADS`。

## Stage 6 剩余重点

- 实际执行 Qwen native INT8/INT4 Blind AB 并积累人工评分
- 真实 ARM/树莓派/低功耗弱设备实机验证
- Concurrent 真实设备资源校准
- 平台特定确定性 OOM attribution（可选）
- 扩展模型 venv/资产一键 bootstrap
