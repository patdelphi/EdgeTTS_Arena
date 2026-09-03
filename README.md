# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件，可以开始本地部署测试。** Stage 0~5 MVP 已完成；Stage 6 已完成跨平台/ARM64/1-CPU smoke、Piper/Kokoro/MeloTTS/CosyVoice 真实 CPU gate、扩展模型 external Python worker、Qwen3 官方 FP32 CPU 功能路径、Qwen3 native INT8/INT4 hosted CPU 量化验证，以及真实目标设备验收包。Qwen3 native INT8 是当前推荐 CPU 优化候选；INT8/INT4 Blind AB 人工评分与真实弱设备实测仍属于扩展验证，不阻塞主 Arena 本地部署测试。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 扩展模型：[`docs/12_第二批模型适配状态.md`](./docs/12_第二批模型适配状态.md)
- CPU/量化部署：[`docs/07_CPU推理与量化部署实操指南.md`](./docs/07_CPU推理与量化部署实操指南.md)
- 路线图：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## 本地部署测试

推荐 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,piper,kokoro]"
pytest
edgetts-arena doctor --ui --exports-root exports/doctor
edgetts-arena serve --ui
```

Windows PowerShell 使用 `py -3.11 -m venv .venv` 与 `.\.venv\Scripts\Activate.ps1`。默认 API：`http://127.0.0.1:8000`，Arena：`http://127.0.0.1:8000/arena/`。

主 Arena 的本地验收顺序建议：Doctor → Dummy/API → Piper → Kokoro → UI → Sequential 对比 → Blind AB → Standard Suite → ZIP/report/environment。

## Benchmark / UI

Arena 支持 1~4 模型、Sequential/Concurrent、TC-01~05、warm-up/repeats、ZIP、Blind AB。`config.language`、CLI `suite --language` 与 Gradio Language control 均为 capability-aware。

多模型 Arena 会计算共同 voice/language 交集：只有所有所选模型都支持且存在共同值时才允许统一选择，否则各模型使用自身默认。这使同一模型的不同量化 backend 可以固定相同 voice/language/seed 做公平 Blind AB。

## Qwen3 hosted CPU 基线

| 路线 | Threads | Audio | RTF | Peak RSS |
|---|---:|---:|---:|---:|
| official FP32 | 2 | 9.04s | 5.315 | 5318MB |
| native INT8 | 2 | 7.52s | **1.787** | **3124MB** |
| native INT8 | 4 | 7.52s | 1.984 | 3129MB |
| native INT4 | 4 | 8.48s | 2.432 | **2899MB** |

官方 FP32 是兼容/功能基线；native INT8@2 是当前 CPU 优化候选。INT4 只作为更低内存实验项；相同 seed/text 下输出时长发生变化，因此不宣称与 INT8 音质等价。

## Qwen3 INT8 / INT4 Blind AB

一次生成成对 manifests：

```bash
python scripts/prepare_qwen3_native_variants.py \
  --binary runtime/qwen3-tts-c/qwen_tts \
  --model-dir models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output-root models/qwen3-native \
  --default-voice Vivian \
  --default-language Chinese
```

默认 catalog 包含两个 disabled/experimental 项：

```text
qwen3-tts-0.6b-native-int8   # hosted 推荐基线：2 threads
qwen3-tts-0.6b-native-int4   # hosted 实验基线：4 threads
```

A/B 工具链 ready 不等于音质 gate passed。在真实人工评分产生前，INT4 继续保持实验路线。

## 真实目标设备验收包

`scripts/target_device_acceptance.py` 用于树莓派、ARM、低功耗 x86 等实际设备。它连续运行真实 synthesis，保存每次 WAV/JSON、系统环境、聚合指标和 ZIP，并按**最差一次** RTF/RSS 做可选阈值判定。

示例：

```bash
unset OPENBLAS_NUM_THREADS
python scripts/target_device_acceptance.py qwen3-native \
  --model-path models/qwen3-native/int8/model.json \
  --text "你好，这是一条真实设备验收语音。" \
  --voice Vivian --language zh --seed 42 \
  --threads 2 --runs 3 \
  --require-arch aarch64,arm64 \
  --max-rtf 2.0 --max-peak-rss-mb 3500 \
  --output-dir exports/target-device/qwen3-int8
```

阈值由目标硬件/业务自行定义，项目不把 GitHub-hosted 数字冒充真机承诺。native runtime 不应预设 `OPENBLAS_NUM_THREADS`。

## Stage 6 剩余重点

- 实际执行 Qwen native INT8/INT4 Blind AB 并积累人工评分
- 用目标设备验收包完成真实 ARM/树莓派/低功耗弱设备实机验证
- Concurrent 真实设备资源校准
- 平台特定确定性 OOM attribution（可选）
- 扩展模型 venv/资产一键 bootstrap
