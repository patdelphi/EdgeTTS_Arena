# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前实现状态：**Stage 0~5 已完成 MVP：Piper 与 Kokoro 已通过真实 CPU CI；FastAPI API、Gradio Arena UI、Blind AB、TC-01~05 标准 Benchmark Suite、warm-up/repeated benchmark、统计聚合与可复现 ZIP 均已接通；Qwen3-TTS 0.6B 继续保持 experimental/unavailable placeholder。**

- 开发规格：[`docs/README.md`](./docs/README.md)
- API 规范：[`docs/04_接口协议与数据规范.md`](./docs/04_接口协议与数据规范.md)
- Benchmark 规范：[`docs/05_评测基准与测试用例集.md`](./docs/05_评测基准与测试用例集.md)
- UI 规范：[`docs/08_前端UI交互原型与界面流程设计.md`](./docs/08_前端UI交互原型与界面流程设计.md)
- 路线图：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 实现检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## 环境

```text
Python >=3.10,<3.13
```

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## 启动 API

```bash
edgetts-arena serve
```

默认 `http://127.0.0.1:8000`，OpenAPI 为 `/docs`。

核心接口：

```text
GET  /healthz
GET  /api/v1/system/models
GET  /api/v1/benchmark/presets
POST /api/v1/benchmark/run
POST /api/v1/benchmark/suite
GET  /api/v1/audio/download/{run_id}/{filename}
GET  /api/v1/export/{run_id}
WS   /api/v1/tts/stream?model=<model_id>
```

## 启动 Arena UI

```bash
python -m pip install -e ".[ui]"
edgetts-arena serve --ui
```

打开 `http://127.0.0.1:8000/arena/`。

UI 已实现：

- 1~4 模型 Arena，Sequential 默认，Concurrent 可选
- capability 驱动 speed / seed / voice
- preset 下拉直接加载 TC-01~05 文本
- audio cards + Inference / Duration / RTF / RSS / CPU / TTFB 对比
- 非流式 `TTFB=null` 显示 `N/A (non-streaming)`
- ZIP 导出
- Blind AB：匿名随机、Naturalness / Intelligibility / Prosody 评分、全部评分后 Reveal
- `blind_scores.json` 与同一 `run_id` 归档
- Standard Benchmark Suite：选择 TC、模型、warm-up、measured repeats、线程数，并展示聚合统计

Gradio 是可选依赖，通过 `mount_gradio_app()` 挂载在同一个 FastAPI 进程；纯 API 模式不需要安装 Gradio。

## 标准 Benchmark Suite

标准题库位于 `config/benchmark_presets.json`：

- TC-01 日常短交互
- TC-02 数字、单位与符号
- TC-03 中英混读
- TC-04 多音字
- TC-05 300+ 字长文本稳定性

默认策略：**warm-up 1 次 + measured 3 次**。每次正式测量保留原始 metrics，同时聚合：

- mean
- median
- min / max
- P95
- variance

API：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/benchmark/suite \
  -H 'Content-Type: application/json' \
  -d '{"models":["dummy"],"case_ids":["TC-01","TC-02"],"warmup_runs":1,"measured_runs":3}'
```

CLI：

```bash
edgetts-arena suite \
  --models dummy \
  --cases TC-01 TC-02 \
  --warmup-runs 1 \
  --measured-runs 3 \
  --threads 2
```

Suite 在一个 `run_id` 下保存：

```text
exports/<run_id>/
  audio/<case_id>__<model_id>.wav
  benchmark_report.json
  environment.json
  <run_id>.zip
```

`environment.json` 记录 OS、CPU、内存、Python、关键 package versions 和线程设置；report 保存请求、标准语料、每次原始 measurement、aggregate statistics 以及模型 metadata。

## Dummy Smoke Test

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav
```

## Piper

```bash
python -m pip install -e ".[piper]"
python -m piper.download_voices en_US-lessac-low --data-dir models/piper
edgetts-arena piper --model models/piper/en_US-lessac-low.onnx --text "Piper test" --threads 2 --output exports/piper.wav
```

GitHub Actions 已用真实 Piper voice 完成 CPU gate。

## Kokoro

```bash
python -m pip install -e ".[kokoro]"
edgetts-arena kokoro --model models/kokoro/kokoro-v1.0.int8.onnx --voice af_heart --text "Kokoro test" --threads 2 --output exports/kokoro.wav
```

Kokoro 使用 `kokoro-onnx` + ONNX Runtime；GitHub Actions 已用 v1.0 int8 ONNX 完成真实 CPU gate。

## Qwen3-TTS experimental

Qwen3 当前仍为受控 placeholder：默认 disabled/unavailable，不绑定未验证社区 CPU runtime，不伪造 capability、音频或 benchmark。

## 下一阶段

**Stage 6 — Hardening**：跨平台 CI、并发压力与资源隔离、timeout/OOM watchdog、弱算力设备与第二批模型验证。
