# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件。** Stage 0~5 MVP 已完成；Stage 6 已完成 Windows/macOS/Linux、原生 ARM64、GitHub-hosted 1-CPU smoke，Python 3.10/3.11/3.12 主 CI 全绿，Piper/Kokoro 真实 CPU gate 通过；MeloTTS 与 CosyVoice 300M SFT 也已完成独立 x86_64 CPU synthesis gate。CosyVoice 的 WeText 前端资产已改为显式本地 FST，并通过 offline preflight。

扩展模型的专用 Python/venv worker 路由已经接入主 BenchmarkService、Standard Suite 与 UI 执行链路。CosyVoice/MeloTTS 继续保持 `experimental + disabled`，但可通过 `worker_python` 或环境变量指向独立 venv，不再要求把官方依赖合并进主 UI 环境。真实 ARM/弱算力目标设备实机验证仍是 Stage 6 剩余项。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 第二批模型与 worker：[`docs/12_第二批模型适配状态.md`](./docs/12_第二批模型适配状态.md)
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

`doctor` 检查 Python/config、导出目录、Dummy 真生成、FastAPI app；`--ui` 增加 Gradio 挂载检查；`--workers` 增加专用 Python worker 协议与 Dummy WAV probe。

## 启动 API / UI

```bash
edgetts-arena serve
edgetts-arena serve --ui
```

默认 API：`http://127.0.0.1:8000`，Arena：`http://127.0.0.1:8000/arena/`，OpenAPI：`/docs`。

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

## Arena UI

UI 已实现：

- 1~4 模型 Arena，Sequential 默认，Concurrent 可选
- capability 驱动 speed / seed / voice
- preset 下拉直接加载 TC-01~05
- audio cards + Inference / Duration / RTF / RSS / CPU / TTFB
- ZIP 导出
- Blind AB：匿名随机、Naturalness / Intelligibility / Prosody、Reveal
- Standard Benchmark Suite：TC、模型、warm-up、repeats、线程数与聚合统计

Gradio 是可选依赖，通过 `mount_gradio_app()` 挂在同一个 FastAPI 进程；纯 API 模式不需要安装 Gradio。

## 标准 Benchmark Suite

题库位于 `config/benchmark_presets.json`：TC-01 日常短交互、TC-02 数字/单位/符号、TC-03 中英混读、TC-04 多音字、TC-05 300+ 字长文本。

默认：**warm-up 1 次 + measured 3 次**，保留原始 metrics，并聚合 mean、median、min/max、P95、variance。

```bash
edgetts-arena suite \
  --models dummy \
  --cases TC-01 TC-02 \
  --warmup-runs 1 \
  --measured-runs 3 \
  --threads 2
```

## Dummy / Piper / Kokoro

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav

python -m piper.download_voices en_US-lessac-low --data-dir models/piper
edgetts-arena piper --model models/piper/en_US-lessac-low.onnx --text "Piper test" --threads 2 --output exports/piper.wav

edgetts-arena kokoro --model models/kokoro/kokoro-v1.0.int8.onnx --voice af_heart --text "Kokoro test" --threads 2 --output exports/kokoro.wav
```

Piper/Kokoro 都已有真实 CPU GitHub Actions gate。完整资产准备见本地部署指南。

## Stage 6 第二批模型与专用 Workers

- **MeloTTS**：Ubuntu x86_64 / Python 3.10 real CPU synthesis gate 已通过；默认 `experimental + disabled`。
- **CosyVoice 300M SFT**：Ubuntu x86_64 / Python 3.10 real CPU synthesis gate 已通过；WeText 本地 FST + offline preflight 已通过；默认 `experimental + disabled`。
- 两者均可通过独立 Python/venv worker 运行，主 UI venv 不需要安装它们的官方完整依赖。

默认可移植配置：

```yaml
worker_python_env: EDGETTS_ARENA_COSYVOICE_PYTHON
# 或 EDGETTS_ARENA_MELOTTS_PYTHON
```

Linux/macOS：

```bash
export EDGETTS_ARENA_COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python"
export EDGETTS_ARENA_MELOTTS_PYTHON="$PWD/.venv-melotts/bin/python"
edgetts-arena doctor --workers
```

Windows PowerShell：

```powershell
$env:EDGETTS_ARENA_COSYVOICE_PYTHON = (Resolve-Path .venv-cosyvoice\Scripts\python.exe)
$env:EDGETTS_ARENA_MELOTTS_PYTHON = (Resolve-Path .venv-melotts\Scripts\python.exe)
edgetts-arena doctor --workers
```

`doctor --workers` 只验证专用解释器、项目 source、基础 worker 依赖和 JSON/WAV 协议，不代替真实模型 synthesis gate。

`GET /api/v1/system/models` 会返回：

```json
{
  "worker_mode": "external",
  "worker_python_configured": true
}
```

解释器绝对路径不会通过 API 暴露。

## Qwen3-TTS experimental

Qwen3 当前仍为受控 placeholder：默认 disabled/unavailable，不绑定未验证社区 CPU runtime，不伪造 capability、音频或 benchmark。

## Stage 6 Hardening

已完成：跨平台 Doctor/smoke、Python 3.10~3.12、ARM64 hosted、1-CPU hosted、ResourceGuard cgroup-aware 预算、watchdog/timeout/crash/OOM diagnostics、Piper/Kokoro/MeloTTS/CosyVoice real CPU gate、CosyVoice offline frontend、扩展模型 external Python worker 路由与 Doctor probe。

剩余重点：**真实 ARM/弱算力目标设备实机验证**、Qwen3-TTS 正式 CPU runtime/量化路线，以及后续扩展模型 venv/资产一键 bootstrap 自动化。
