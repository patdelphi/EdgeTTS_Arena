# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件。** Stage 0~5 MVP 已完成；Stage 6 已完成 Windows/macOS/Linux、原生 ARM64、GitHub-hosted 1-CPU smoke，Python 3.10/3.11/3.12 主 CI 全绿，Piper/Kokoro 真实 CPU gate 通过；MeloTTS 与 CosyVoice 300M SFT 也已完成独立 x86_64 CPU synthesis gate。CosyVoice 的 WeText 前端资产已改为显式本地 FST，并通过“禁止 `snapshot_download()` 后仍可 load model”的 offline preflight。

> 部署边界：主 Arena 环境直接测试 Dummy/Piper/Kokoro。CosyVoice/MeloTTS 继续保持 `experimental + disabled`，建议放在独立 venv 中验证，避免官方旧依赖覆盖主 UI 的 Gradio/FastAPI/Pydantic 版本。真实 ARM/弱算力目标设备实机验证仍是 Stage 6 剩余项，但不阻塞桌面端本地部署测试。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 开发规格：[`docs/README.md`](./docs/README.md)
- API 规范：[`docs/04_接口协议与数据规范.md`](./docs/04_接口协议与数据规范.md)
- Benchmark 规范：[`docs/05_评测基准与测试用例集.md`](./docs/05_评测基准与测试用例集.md)
- UI 规范：[`docs/08_前端UI交互原型与界面流程设计.md`](./docs/08_前端UI交互原型与界面流程设计.md)
- 路线图：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 实现检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## 环境

```text
Python >=3.10,<3.13
推荐本地首次部署：Python 3.11
```

主 Arena 开发环境：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ui,piper,kokoro]"
pytest
edgetts-arena doctor --ui --exports-root exports/doctor
```

`doctor` 会检查 Python/config、导出目录、Dummy 真生成、FastAPI app，以及 `--ui` 时的 Gradio 挂载条件。

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

标准题库位于 `config/benchmark_presets.json`：TC-01 日常短交互、TC-02 数字/单位/符号、TC-03 中英混读、TC-04 多音字、TC-05 300+ 字长文本稳定性。

默认策略：**warm-up 1 次 + measured 3 次**。每次正式测量保留原始 metrics，同时聚合 mean、median、min/max、P95、variance。

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

## Dummy Smoke Test

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav
```

## Piper

```bash
python -m piper.download_voices en_US-lessac-low --data-dir models/piper
edgetts-arena piper --model models/piper/en_US-lessac-low.onnx --text "Piper test" --threads 2 --output exports/piper.wav
```

GitHub Actions 已用真实 Piper voice 完成 CPU gate。要让 Arena UI 使用 Piper，将 `config/models_config.yaml` 中 `piper.enabled` 改为 `true`。

## Kokoro

先下载 `kokoro-v1.0.int8.onnx` 与 `voices-v1.0.bin` 到 `models/kokoro/`，完整命令见本地部署指南，然后：

```bash
edgetts-arena kokoro --model models/kokoro/kokoro-v1.0.int8.onnx --voice af_heart --text "Kokoro test" --threads 2 --output exports/kokoro.wav
```

Kokoro 使用 `kokoro-onnx` + ONNX Runtime；GitHub Actions 已用 v1.0 int8 ONNX 完成真实 CPU gate。要让 Arena UI 使用 Kokoro，将 `config/models_config.yaml` 中 `kokoro.enabled` 改为 `true`。

## Qwen3-TTS experimental

Qwen3 当前仍为受控 placeholder：默认 disabled/unavailable，不绑定未验证社区 CPU runtime，不伪造 capability、音频或 benchmark。

## Stage 6 第二批模型

- **MeloTTS**：独立 Ubuntu x86_64 / Python 3.10 CPU synthesis gate 已通过；仍保持 `experimental + disabled`。
- **CosyVoice 300M SFT**：独立 Ubuntu x86_64 / Python 3.10 CPU gate 已通过。Gate 使用 2 threads，生成 4.098s WAV，单次 inference 17.875s、RTF 4.362、peak RSS 4181 MB；该记录只用于可复现 gate 追溯，不作为性能承诺。
- CosyVoice `wetext==0.0.4` 所需 5 个 FST 通过 `scripts/prepare_cosyvoice_frontend.py` 在安装准备阶段显式下载并写入 SHA-256 manifest；offline preflight 会把 `snapshot_download()` 替换为调用即失败，再验证 Adapter 仍可从本地 FST 加载。

第二批模型暂不建议直接安装进主 Arena UI venv。后续模型专用 worker/venv 隔离完成后，再考虑在同一 UI 中正式启用。

## Stage 6 Hardening

已完成：

- Windows/macOS/Linux 本地部署 Doctor + Dummy/API/spawn smoke
- Python 3.10/3.11/3.12 主测试矩阵
- 原生 ARM64 hosted smoke
- GitHub-hosted 1-CPU 资源预算 smoke
- Concurrent CPU/内存公平预算与 `pressure` profile
- 非持久模型进程级 watchdog、timeout/crash 回收、worker 退出/OOM 诊断
- Piper/Kokoro/MeloTTS/CosyVoice 真实 CPU gate（后两者为独立 extended gate）

剩余重点：真实 ARM/弱算力目标设备实机验证，以及扩展模型的正式独立 worker/venv 集成。
