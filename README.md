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

**Stage 6 — Hardening（进行中）**：已加入 Windows/macOS/Linux smoke、Concurrent CPU/内存预算、进程级 watchdog 与 worker 退出诊断。单轮 benchmark 与 Standard Suite 均可在 hard timeout 后终止 worker；显式 `MemoryError` → `2002`，worker crash/timeout → `3002` / `3001`。`SIGKILL` 只标记 `oom_suspected=true`，不在缺少内核证据时伪称确定 OOM。下一步继续第二批模型与弱算力设备验证。


## Stage 6 Hardening（第一批）

Concurrent 单轮 benchmark 现在会先生成资源计划：

- 按逻辑核数公平分配 `threads_per_model`，避免 `models × threads` 直接超配 CPU。
- 默认最多 4 个并发模型。
- 默认按每并发模型至少 512 MB 可用内存预算，并继续遵守 soft/hard memory guard。
- 返回 `execution_profile=pressure`、requested/effective threads、total thread budget 和 resource warnings。
- Sequential 保持 `execution_profile=baseline`。

GitHub CI 增加 Ubuntu / Windows / macOS 三平台 Python 3.11 Dummy WAV + FastAPI health/preset smoke，并执行真实 `spawn` worker Suite smoke。

## Stage 6 Watchdog（第二批）

`BenchmarkService` 与 `RepeatedBenchmarkService` 默认启用进程隔离；嵌入式调用或测试可显式传入 `isolate_model_processes=False` 关闭。对 `keep_in_memory=false` 的模型：

- 单轮 benchmark 在独立 `spawn` worker 中完成 load → infer → WAV write → unload。
- Standard Suite 以一个 `case/model` 为 worker 粒度，在同一子进程完成 warm-up + repeated measurements。
- WAV 直接写到主进程预先校验的 `exports/<run_id>/audio/` 路径，multiprocessing queue 只回传 metrics / metadata / error。
- 单轮 hard timeout 使用 `inference_timeout_sec`；Suite group timeout 为 `inference_timeout_sec × (warmup + measured)`。
- timeout 返回 `3001 inference_timeout`；异常退出返回 `3002 worker_exited`，主服务继续运行。
- 每个 isolated result 增加 `worker` 诊断：PID、exit code、elapsed、termination、signal、`oom_suspected`。
- worker 内显式 `MemoryError` 映射为 `2002 worker_memory_error`；无返回的 `SIGKILL` 只标记“possible OOM or external kill”。
- `keep_in_memory=true` 暂继续进程内执行并返回显式 warning，不伪装为已经具备 hard process timeout。



## Stage 6 第二批模型

第二批 Adapter 已完成 contract integration，但默认保持 disabled/experimental，尚未计为真实 CPU gate：

- **CosyVoice 300M SFT**：面向官方 QwenAudio/CosyVoice `AutoModel + inference_sft()`；支持固定 speaker、speed 与真实 chunk streaming。当前只接 SFT 模式，不把 CosyVoice2/3 需要 prompt audio/text 的 zero-shot 路径硬塞进现有 voice-id schema。运行时需单独安装官方 source checkout。
- **MeloTTS**：面向官方 `melo.api.TTS`；支持 EN/ES/FR/ZH/JP/KR、speaker id 与 speed。为保证 benchmark 可复现，Adapter 禁止隐式 HuggingFace 下载，要求本地 `model.json` 显式指向 config/checkpoint。

配置中新增 `cosyvoice-300m-sft` 与 `melotts-zh`，均 `enabled: false`、`experimental: true`。只有完成可复现真实 CPU synthesis CI 后才会把对应 real-model gate 标记完成。


## Extended Model Gates

CosyVoice/MeloTTS 的真实 CPU 验证不进入每次 push 的主 CI。仓库提供手动 `Extended Model Gates` workflow，可选择 `melotts`、`cosyvoice` 或 `both`。验证统一通过 `scripts/real_model_smoke.py` 输出 WAV 与 JSON metrics artifact。

该 workflow 只是可复现 gate；在实际 GitHub Actions run 成功前，两个模型仍保持 `experimental + disabled`，不标记为 real CPU verified。
