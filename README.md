# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前实现状态：**Stage 0~3 已完成：Piper 与 Kokoro 已通过真实 CPU CI；FastAPI 同步 benchmark/download/export/streaming gate 已接通；Qwen3-TTS 0.6B 以 experimental/unavailable placeholder 接入。**

- 开发规格：[`docs/README.md`](./docs/README.md)
- 开发基线与冻结决策：[`docs/00_开发准备与文档审阅.md`](./docs/00_开发准备与文档审阅.md)
- 实施路线：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- API 规范：[`docs/04_接口协议与数据规范.md`](./docs/04_接口协议与数据规范.md)
- 实现检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## 开发环境

为保证 TTS 运行时兼容性，Python 支持范围冻结为：

```text
Python >=3.10,<3.13
```

安装开发依赖并执行测试：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## Stage 0~1 — 基础运行时

已实现：

- `pyproject.toml` + `src/edgetts_arena/` 包结构
- `BaseTTSAdapter` / `TTSCapabilities` / `TTSOutput`
- 配置、日志、错误类型
- `DummyTTSAdapter`
- `ModelRegistry` lazy load / unload / status
- `MetricsCollector`：Inference Time、Audio Duration、RTF、RSS、CPU；非流式 TTFB=`None`
- `ResourceGuard`：soft/hard memory guard、线程约束
- `ProcessRunner`：独立子进程、timeout terminate/kill
- GitHub Actions：Python 3.10 / 3.11 / 3.12

Dummy smoke test：

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav
```

## Stage 2 — Piper

安装：

```bash
python -m pip install -e ".[piper]"
python -m piper.download_voices en_US-lessac-low --data-dir models/piper
```

使用：

```bash
edgetts-arena piper \
  --model models/piper/en_US-lessac-low.onnx \
  --text "Edge TTS Arena Piper test." \
  --threads 2 \
  --output exports/piper.wav
```

Piper 已在 GitHub Actions 使用真实 voice 完成 CPU 合成 gate。当前维护的 Piper runtime（OHF-Voice/piper1-gpl）为 GPL-3.0；发行/捆绑时需单独处理许可证边界。

## Stage 2 — Kokoro

安装：

```bash
python -m pip install -e ".[kokoro]"
```

使用：

```bash
edgetts-arena kokoro \
  --model models/kokoro/kokoro-v1.0.int8.onnx \
  --voice af_heart \
  --text "Edge TTS Arena Kokoro test." \
  --threads 2 \
  --output exports/kokoro.wav
```

Kokoro 使用 `kokoro-onnx` + ONNX Runtime，当前直接文本能力冻结为 `en-us` / `en-gb`。GitHub Actions 已使用 v1.0 int8 ONNX 完成真实 CPU 合成 gate。

## Stage 2 — Qwen3-TTS experimental

`Qwen3TTSAdapter` 当前有意保持不可用：

- 默认 `enabled: false`、`experimental: true`
- 不绑定未冻结社区 CPU runtime
- 不伪造 capability、音频或 benchmark
- 只有通过可复现模型包、许可证、内存与 benchmark gate 后才会升级为正式 runtime

## Stage 3 — Local API

启动：

```bash
edgetts-arena serve
```

默认地址：`http://127.0.0.1:8000`，OpenAPI 为 `/docs`。

核心接口：

```text
GET  /healthz
GET  /api/v1/system/models
POST /api/v1/benchmark/run
GET  /api/v1/audio/download/{run_id}/{filename}
GET  /api/v1/export/{run_id}
WS   /api/v1/tts/stream?model=<model_id>
```

Stage 3 行为：

- benchmark API 保持**同步语义**，响应即最终结果
- 默认 Sequential；Concurrent 受 ResourceGuard 前置限制
- 单模型失败不会终止同轮其他模型
- 每轮生成独立 `run_id`
- 自动落盘 WAV、`benchmark_report.json`、`environment.json`
- ZIP export 基于已落盘结果构建
- download/export 使用 allow-list + resolve containment 防止路径穿越
- streaming 仅对 `capabilities.streaming=true` 的可用模型开放
- 流式二进制为 `pcm_s16le`，首块报告真实 TTFB
- 非流式 benchmark 的 TTFB 继续为 `null`

示例：

```bash
curl -X POST http://127.0.0.1:8000/api/v1/benchmark/run \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello","models":["dummy"],"execution_mode":"sequential"}'
```

## 下一阶段

**Stage 4 — Arena UI**：2~4 模型横向对比、capability 驱动参数、音频试听、Blind AB MVP，并直接复用 Stage 3 API。
