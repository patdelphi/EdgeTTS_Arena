# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前实现状态：**Stage 0~4 已完成 MVP：Piper 与 Kokoro 已通过真实 CPU CI；FastAPI benchmark/download/export/streaming API 已接通；Gradio Arena UI 与 Blind AB 已实现；Qwen3-TTS 0.6B 继续保持 experimental/unavailable placeholder。**

- 开发规格：[`docs/README.md`](./docs/README.md)
- API 规范：[`docs/04_接口协议与数据规范.md`](./docs/04_接口协议与数据规范.md)
- UI 规范：[`docs/08_前端UI交互原型与界面流程设计.md`](./docs/08_前端UI交互原型与界面流程设计.md)
- 路线图：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 实现检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## 环境

Python 支持范围：

```text
Python >=3.10,<3.13
```

基础开发环境：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

## 启动 API

```bash
edgetts-arena serve
```

默认地址 `http://127.0.0.1:8000`，OpenAPI 为 `/docs`。

核心接口：

```text
GET  /healthz
GET  /api/v1/system/models
POST /api/v1/benchmark/run
GET  /api/v1/audio/download/{run_id}/{filename}
GET  /api/v1/export/{run_id}
WS   /api/v1/tts/stream?model=<model_id>
```

## 启动 Arena UI

Gradio 是可选依赖，不会污染纯 API 安装：

```bash
python -m pip install -e ".[ui]"
edgetts-arena serve --ui
```

打开：

```text
http://127.0.0.1:8000/arena/
```

UI MVP 已实现：

- 1~4 模型选择（2~4 用于正式横向 Arena）
- Sequential 默认，Concurrent 可选且继续受 ResourceGuard 限制
- capability 驱动的 speed / seed / voice 控件
- 多模型模式自动使用各模型默认 voice，避免错误套用同一个 voice id
- 4 个结果卡片与音频播放器
- Inference Time / Audio Duration / RTF / Peak RSS / RSS Δ / CPU / TTFB 对比表
- 非流式模型 `TTFB=null` 显示 `N/A (non-streaming)`
- 单模型失败不影响其他结果卡片
- ZIP 一键导出
- Blind AB：匿名随机顺序、Naturalness / Intelligibility / Prosody 1~5 评分、全部评分后才能揭晓
- Blind 评分和匿名映射保存为 `blind_scores.json`，并写回同一 `run_id` ZIP

Gradio 通过 `mount_gradio_app()` 挂载在同一个 FastAPI 进程，UI 复用与 Stage 3 API 相同的 Registry、BenchmarkService 和 artifact store；基础 `edgetts-arena serve` 不需要安装 Gradio。当前 UI 运行时冻结为 Gradio 6.x（`>=6.5,<7`）。

## Dummy Smoke Test

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav
```

## Piper

```bash
python -m pip install -e ".[piper]"
python -m piper.download_voices en_US-lessac-low --data-dir models/piper
edgetts-arena piper \
  --model models/piper/en_US-lessac-low.onnx \
  --text "Edge TTS Arena Piper test." \
  --threads 2 \
  --output exports/piper.wav
```

GitHub Actions 已用真实 Piper voice 完成 CPU 合成 gate。

## Kokoro

```bash
python -m pip install -e ".[kokoro]"
edgetts-arena kokoro \
  --model models/kokoro/kokoro-v1.0.int8.onnx \
  --voice af_heart \
  --text "Edge TTS Arena Kokoro test." \
  --threads 2 \
  --output exports/kokoro.wav
```

Kokoro 使用 `kokoro-onnx` + ONNX Runtime，当前直接文本能力冻结为 `en-us` / `en-gb`；GitHub Actions 已用 v1.0 int8 ONNX 完成真实 CPU gate。

## Qwen3-TTS experimental

Qwen3 当前仍为受控 placeholder：默认 disabled/unavailable，不绑定未验证社区 CPU runtime，不伪造 capability、音频或 benchmark。

## CI

当前 CI 包括：

- Python 3.10 / 3.11 / 3.12 全量无模型测试
- Gradio UI build + FastAPI mount smoke
- Piper real CPU smoke
- Kokoro int8 real CPU smoke

## 下一阶段

**Stage 5 — Benchmark Suite**：实现 TC-01~05 preset、warm-up / repeated benchmark、统计汇总和更完整的可复现报告。
