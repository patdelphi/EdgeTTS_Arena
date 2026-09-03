# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件。** Stage 0~5 MVP 已完成；Stage 6 已完成跨平台/ARM64/1-CPU smoke、Piper/Kokoro/MeloTTS/CosyVoice 真实 CPU gate、扩展模型 external Python worker，以及 Qwen3-TTS 0.6B CustomVoice 官方 `qwen-tts` CPU Adapter、本地资产准备与 manual heavy gate。Qwen3 real heavy CPU synthesis 尚未宣称通过。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 扩展模型与 worker：[`docs/12_第二批模型适配状态.md`](./docs/12_第二批模型适配状态.md)
- 开发规格：[`docs/README.md`](./docs/README.md)
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

核心接口：`/healthz`、`/api/v1/system/models`、`/api/v1/benchmark/run`、`/api/v1/benchmark/suite`、audio/export download 与 capability-gated streaming。

## Benchmark / UI

Arena 已实现 1~4 模型对比、Sequential/Concurrent、capability 驱动 controls、TC-01~05 preset、audio/metrics、ZIP、Blind AB、Standard Suite。Standard Suite 默认 warm-up 1 + measured 3，保留原始 metrics 并聚合 mean/median/min/max/P95/variance。

```bash
edgetts-arena suite --models dummy --cases TC-01 TC-02 --warmup-runs 1 --measured-runs 3 --threads 2
```

## 扩展模型与专用 Workers

默认全部 `experimental + disabled`：

- MeloTTS：hosted x86_64 real CPU gate ✅
- CosyVoice 300M SFT：hosted x86_64 real CPU gate + offline WeText ✅
- Qwen3-TTS 0.6B CustomVoice：官方 CPU Adapter + contract + local snapshot manifest + manual heavy gate definition ✅；real heavy CPU synthesis ⏳

独立解释器配置：

```text
EDGETTS_ARENA_QWEN3_PYTHON
EDGETTS_ARENA_COSYVOICE_PYTHON
EDGETTS_ARENA_MELOTTS_PYTHON
```

检查全部已声明 worker：

```bash
edgetts-arena doctor --workers
```

只检查一个已准备好的扩展模型，不要求另外两套 venv：

```bash
edgetts-arena doctor --worker qwen3-tts-0.6b
```

`--worker MODEL_ID` 可重复，并隐含开启 worker 检查。Doctor 只验证解释器、项目 source、基础依赖与 external JSON/WAV 协议，**不代替真实模型 synthesis gate**。

## Qwen3-TTS 0.6B CustomVoice

Arena 只接官方 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`，不把 Base voice-clone checkpoint 混入现有 `text + voice id` 请求合同。CPU fp32 是当前保守功能基线；Arena 暂不宣称 speed、seed、streaming 或 voice-clone capability。

```bash
python scripts/prepare_qwen3_model.py --output models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice
python scripts/real_model_smoke.py qwen3 \
  --model-path models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --voice Vivian --text "你好，这是一条 Qwen3 TTS CPU 验证语音。" \
  --threads 2 --output exports/qwen3.wav --report exports/qwen3.json
```

`.github/workflows/extended-model-gates.yml` 提供 `qwen3` manual-only heavy gate；普通 push 不自动下载约 2.5GB 模型资产。

## Stage 6 剩余重点

- Qwen3 manual real heavy CPU synthesis gate 实际通过
- Qwen3 quantization / low-memory CPU route
- 真实 ARM/弱算力目标设备实机验证
- 后续扩展模型 venv/资产一键 bootstrap
