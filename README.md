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

标准题库位于 `config/benchmark_presets.json`：TC-01~TC-05。默认策略：**warm-up 1 次 + measured 3 次**，保留 raw measurements，并聚合 mean / median / min / max / P95 / variance。

```bash
edgetts-arena suite --models dummy --cases TC-01 TC-02 --warmup-runs 1 --measured-runs 3 --threads 2
```

Suite 在一个 `run_id` 下保存代表 WAV、`benchmark_report.json`、`environment.json` 和 ZIP。

## Dummy / Piper / Kokoro

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav
```

Piper 与 Kokoro 均已有 GitHub Actions 真实 CPU gate。Qwen3 当前仍为受控 experimental/unavailable placeholder。

## Stage 6 Hardening（进行中）

第一批已实现：

- Ubuntu / Windows / macOS Python 3.11 Dummy WAV + FastAPI health/preset smoke CI。
- Concurrent 按逻辑核数公平分配 `threads_per_model`，避免 `models × threads` 超配 CPU。
- 默认最多 4 个并发模型。
- 默认按每并发模型至少 512 MB 可用内存预算，并继续遵守 soft/hard memory guard。
- Concurrent 返回 `execution_profile=pressure`；Sequential 为 `execution_profile=baseline`。
- 报告保存 requested/effective threads、total thread budget 和 resource warnings。

下一步：把现有 `ProcessRunner` 正式接入 benchmark 主链路，实现进程级 timeout/OOM/worker-crash watchdog。
