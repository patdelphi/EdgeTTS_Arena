# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件。** Stage 0~5 MVP 已完成；Stage 6 已完成 Windows/macOS/Linux、原生 ARM64、GitHub-hosted 1-CPU smoke，Python 3.10/3.11/3.12 主 CI 矩阵持续验证，Piper/Kokoro 真实 CPU gate 通过；MeloTTS 与 CosyVoice 300M SFT 也已完成独立 x86_64 CPU synthesis gate。CosyVoice 的 WeText 前端资产已改为显式本地 FST，并通过 offline preflight。

扩展模型的专用 Python/venv worker 路由已经接入主 BenchmarkService、Standard Suite 与 UI 执行链路。CosyVoice、MeloTTS、Qwen3-TTS 0.6B CustomVoice 均保持 `experimental + disabled`，可通过 `worker_python` 或环境变量指向独立 venv，不要求把官方重依赖合并进主 UI 环境。Qwen3 已从 placeholder 升级为官方 `qwen-tts` CPU Adapter，并具备本地资产准备脚本与手动重型 CPU gate；真实 Qwen3 CPU synthesis gate 尚未宣称通过。

- 本地部署与验收：[`docs/12_本地部署与验收指南.md`](./docs/12_本地部署与验收指南.md)
- 扩展模型与 worker：[`docs/12_第二批模型适配状态.md`](./docs/12_第二批模型适配状态.md)
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

`doctor` 检查 Python/config、导出目录、Dummy 真生成、FastAPI app；`--ui` 增加 Gradio 挂载检查；`--workers` 增加配置中专用 Python worker 的协议与 Dummy WAV probe。

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

UI 已实现：1~4 模型 Arena、Sequential/Concurrent、capability 驱动 speed/seed/voice、TC-01~05 preset、audio/metrics、ZIP 导出、Blind AB、Standard Benchmark Suite。

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

## Stage 6 扩展模型与专用 Workers

- **MeloTTS**：Ubuntu x86_64 / Python 3.10 real CPU synthesis gate 已通过；默认 `experimental + disabled`。
- **CosyVoice 300M SFT**：Ubuntu x86_64 / Python 3.10 real CPU synthesis gate 已通过；WeText 本地 FST + offline preflight 已通过；默认 `experimental + disabled`。
- **Qwen3-TTS 0.6B CustomVoice**：官方 `qwen-tts` CPU Adapter、合同测试、本地 snapshot/manifest 准备脚本、manual heavy CPU gate 已实现；真实 heavy gate 尚未实际通过，因此仍 `experimental + disabled`。
- 三者均使用独立 Python/venv worker，主 UI venv 不需要安装其完整官方依赖。

默认可移植配置：

```text
EDGETTS_ARENA_QWEN3_PYTHON
EDGETTS_ARENA_COSYVOICE_PYTHON
EDGETTS_ARENA_MELOTTS_PYTHON
```

Linux/macOS：

```bash
export EDGETTS_ARENA_QWEN3_PYTHON="$PWD/.venv-qwen3/bin/python"
export EDGETTS_ARENA_COSYVOICE_PYTHON="$PWD/.venv-cosyvoice/bin/python"
export EDGETTS_ARENA_MELOTTS_PYTHON="$PWD/.venv-melotts/bin/python"
edgetts-arena doctor --workers
```

Windows PowerShell：

```powershell
$env:EDGETTS_ARENA_QWEN3_PYTHON = (Resolve-Path .venv-qwen3\Scripts\python.exe)
$env:EDGETTS_ARENA_COSYVOICE_PYTHON = (Resolve-Path .venv-cosyvoice\Scripts\python.exe)
$env:EDGETTS_ARENA_MELOTTS_PYTHON = (Resolve-Path .venv-melotts\Scripts\python.exe)
edgetts-arena doctor --workers
```

`doctor --workers` 只验证专用解释器、项目 source、基础 worker 依赖和 JSON/WAV 协议，不代替真实模型 synthesis gate。

## Qwen3-TTS 0.6B CustomVoice

Arena 当前只接官方 `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`，不把 0.6B Base voice-clone checkpoint 混入相同请求合同。Adapter 使用 CPU + fp32 保守基线，支持内置 speaker 与 10 种主要语言；Arena 暂不宣称 speed、seed、streaming 或 voice-clone capability。

专用环境准备后：

```bash
python scripts/prepare_qwen3_model.py \
  --output models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice

python scripts/real_model_smoke.py qwen3 \
  --model-path models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --voice Vivian \
  --text "你好，这是一条 Qwen3 TTS CPU 验证语音。" \
  --threads 2 \
  --output exports/qwen3.wav \
  --report exports/qwen3.json
```

GitHub 的 `.github/workflows/extended-model-gates.yml` 提供 `qwen3` 手动重型 gate，不在普通 push 上自动下载约 2.5GB 的模型资产。只有真实 gate 成功后才会把该项标记为 verified。

## Stage 6 Hardening

已完成：跨平台 Doctor/smoke、Python 3.10~3.12、ARM64 hosted、1-CPU hosted、ResourceGuard cgroup-aware 预算、watchdog/timeout/crash/OOM diagnostics、Piper/Kokoro/MeloTTS/CosyVoice real CPU gate、CosyVoice offline frontend、扩展模型 external Python worker 路由、Qwen3 官方 CPU Adapter 与手动 gate 路径。

剩余重点：**真实 ARM/弱算力目标设备实机验证**、Qwen3-TTS 真实 heavy CPU synthesis gate、Qwen3 量化/低内存路线，以及后续扩展模型 venv/资产一键 bootstrap 自动化。
