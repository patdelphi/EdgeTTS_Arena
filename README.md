# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前状态：**主 Arena 已达到本地部署测试条件，可以开始本地部署测试。** Stage 0~5 MVP 已完成；Stage 6 的主要工程工具链也已基本收口：跨平台/ARM64/1-CPU smoke、Piper/Kokoro/MeloTTS/CosyVoice 真实 CPU gate、扩展模型 external worker + pinned bootstrap、Qwen3 official FP32 与 native INT8/INT4 hosted CPU 路线、Blind AB、真实目标设备 acceptance、Concurrent 真机校准工具，以及分级 OOM evidence 均已实现。当前剩余主要是**真实人工评分与真实目标硬件数据**，不是主 Arena 启动 blocker。

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

Windows PowerShell：`py -3.11 -m venv .venv`，然后 `.\.venv\Scripts\Activate.ps1`。默认 API `http://127.0.0.1:8000`，Arena `http://127.0.0.1:8000/arena/`。

建议验收顺序：Doctor → Dummy/API → Piper → Kokoro → UI → Sequential 对比 → Blind AB → Standard Suite → ZIP/report/environment。

## 扩展模型 pinned bootstrap

Qwen3 official、MeloTTS、CosyVoice：

```bash
# 默认只打印计划，不联网、不下载模型
python scripts/bootstrap_extended_model.py qwen3 --python python3.11
python scripts/bootstrap_extended_model.py melotts --python python3.10
python scripts/bootstrap_extended_model.py cosyvoice --python python3.10

# 显式执行重操作
python scripts/bootstrap_extended_model.py qwen3 --python python3.11 --execute
```

成功后生成 `bootstrap_plan.json`、`env.sh`、`env.ps1`。bootstrap 固定与 hosted heavy gate 对齐的 runtime/source/model revision，并做 runtime preflight + targeted Worker Doctor。bootstrap success ≠ real-model synthesis success ≠ target-device performance pass。

### Qwen3 native INT8/INT4 推荐 CPU 路线

native pure-C 路线有独立 planner：

```bash
# 仅打印 plan
python scripts/bootstrap_qwen3_native.py --python python3.11

# Linux/macOS 系统依赖准备好后执行
python scripts/bootstrap_qwen3_native.py --python python3.11 --execute
```

它固定 native runtime revision 与官方 0.6B CustomVoice model revision，执行 `make blas`、`--caps`、`--self-test`，再生成匹配的 INT8/INT4 manifests 并做 Arena Adapter preflight。Linux 需要 C toolchain + OpenBLAS development library；macOS 使用 Accelerate。Windows 原生不是当前验证基线，建议 WSL/Linux。

## Qwen3 hosted CPU 基线

| 路线 | Threads | Audio | RTF | Peak RSS |
|---|---:|---:|---:|---:|
| official FP32 | 2 | 9.04s | 5.315 | 5318MB |
| native INT8 | 2 | 7.52s | **1.787** | **3124MB** |
| native INT8 | 4 | 7.52s | 1.984 | 3129MB |
| native INT4 | 4 | 8.48s | 2.432 | **2899MB** |

当前 CPU 优化候选是 **native INT8@2**。INT4 只作为更低内存实验项；相同 seed/text 下输出时长不同，因此不宣称与 INT8 音质等价。

## Blind AB

`qwen3-tts-0.6b-native-int8` 与 `qwen3-tts-0.6b-native-int4` 已作为 disabled/experimental 项进入 catalog。多模型 UI 使用共同 voice/language 交集，可固定相同 text/voice/language/seed 后匿名评分 Naturalness / Intelligibility / Prosody。**工具 ready 不等于人工质量 gate passed。**

## 真实目标设备验收

单模型连续真实 synthesis：

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

Sequential baseline 与 Concurrent pressure 成对校准：

```bash
python scripts/target_device_concurrent_calibration.py \
  --models piper kokoro \
  --text "Concurrent target-device calibration." \
  --threads 2 --runs 3 \
  --max-rtf-slowdown-ratio 1.8 \
  --output-dir exports/target-concurrent/piper-kokoro
```

Concurrent 报告保留每对 run_id、requested/effective thread budget、ResourceGuard warnings、每模型 Sequential/Concurrent RTF/RSS/CPU 与 `concurrent_rtf / sequential_rtf` slowdown。

两个目标设备工具都把每次 run、environment、aggregate/report 与 ZIP 保留为证据包。**默认拒绝复用非空 output-dir 或已有 ZIP**，避免旧结果混入新报告；确需重跑时显式加 `--overwrite`，脚本会先清理旧目录和 archive。

性能阈值由目标硬件/业务 SLA 定义；项目不把 GitHub-hosted 数字冒充真机承诺。

## Worker OOM evidence

Worker diagnostics 现在区分：

- `worker_memory_error`：显式 MemoryError / worker_memory_error；
- `cgroup_oom_kill_observed`：SIGKILL 且 Linux cgroup v2 `memory.events.oom_kill` 在执行窗口增加；
- `sigkill_suspected`：仅有 SIGKILL，证据不足；
- `none`：无 OOM 证据；Arena watchdog timeout 后自己的 terminate/kill 也保持 `none`。

cgroup 计数是 cgroup-wide，因此 `observed` 不写成“已经证明该 PID 被 OOM killer 杀死”。

## Stage 6 剩余

- 实际执行 Qwen native INT8/INT4 多文本/多语言人工 Blind AB；
- 在真实 ARM/树莓派/低功耗设备执行 target-device acceptance；
- 在真实目标设备执行 Concurrent calibration，并据此调整部署资源预算。

以上均不阻塞主 Arena 本地部署测试。
