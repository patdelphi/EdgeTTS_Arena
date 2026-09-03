# CPU 推理与量化部署实操指南

> 版本：v0.4 ｜ 已纳入 Qwen3 official FP32 与 native INT8/INT4 hosted CPU 实测

## 1. CPU 优化顺序

1. 限制线程并记录 effective budget。
2. 先验证官方 CPU runtime 作为兼容基线。
3. 对内存带宽敏感模型评估 INT8/低位量化。
4. 大模型 lazy load / process isolation。
5. affinity/cgroup-aware budget 与真实设备校准。

## 2. 主 Arena 与扩展 runtime

主 Arena 推荐 Python 3.11，Dummy/Piper/Kokoro 与 UI/API 在同一环境。Qwen official/CosyVoice/MeloTTS 使用独立 Python worker；Qwen native 是本地 C binary + manifest，可在普通 Python worker 中启动 C 子进程，不要求 PyTorch。

## 3. 线程规则

PyTorch 使用 `torch.set_num_threads()`；ONNX Runtime 使用 `intra_op_num_threads` + `inter_op_num_threads=1`。

通用 OMP/MKL 环境变量可按 runtime 文档设置，但**不要把 `OPENBLAS_NUM_THREADS` 当成全局固定模板**。当前 pinned Qwen native runtime 会根据自己的 `-j/--threads` 调 `openblas_set_num_threads()`；若环境中已存在 `OPENBLAS_NUM_THREADS`，会绕过该内部分配。因此运行 native Qwen 时应保证该变量未预设。

## 4. Piper / Kokoro

Piper 与 Kokoro 继续作为主 Arena 轻量真实模型 baseline。Kokoro 支持 capability-gated `language`（当前 en-us/en-gb）。

## 5. Qwen3 official FP32

模型：`Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`；runtime：官方 `qwen-tts==0.1.1` + CPU PyTorch。

首次 hosted gate：2 threads，9.04s audio / 48.05s inference / RTF 5.315 / peak RSS 5317.8MB。

结论：可工作，但不作为弱设备性能路线。

## 6. Qwen3 native quant route

可选 pure-C runtime：`gabriele-mastrapasqua/qwen3-tts`，当前固定 revision：

```text
e56ec7e6eabbed608b13bfbd3fba431708b2077f
```

Arena Adapter 通过本地 manifest 固定：binary、model_dir、runtime_revision、quantization、default voice、default language。load 时先执行 `--caps` preflight；CI 还执行 `--self-test`。

同文本、同 seed=42 hosted 对照：

| Quant | Threads | Audio | Inference | RTF | Peak RSS |
|---|---:|---:|---:|---:|---:|
| INT8 | 2 | 7.52s | 13.44s | **1.787** | 3124MB |
| INT8 | 4 | 7.52s | 14.92s | 1.984 | 3129MB |
| INT4 | 4 | 8.48s | 20.62s | 2.432 | **2899MB** |

### 当前选择

- 默认 CPU 候选：**INT8 + 2 threads**。
- INT4：低内存实验项；只节省约 225MB peak，但该 gate 中明显更慢。
- 相同 seed/text 下 INT4 音频时长发生变化，因此不能仅靠 WAV sanity 推断与 INT8 同质量。
- 质量判断必须使用 Blind AB / 主观自然度、可懂度、韵律评分。

准备示例：

```bash
python scripts/prepare_qwen3_model.py --output models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice
python scripts/prepare_qwen3_native_manifest.py \
  --binary runtime/qwen3-tts-c/qwen_tts \
  --model-dir models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --runtime-revision e56ec7e6eabbed608b13bfbd3fba431708b2077f \
  --quantization int8 --default-voice Vivian --default-language Chinese \
  --output models/qwen3-native/int8/model.json
```

## 7. 子进程资源指标

native C runtime 是 Python Adapter 的子进程。MetricsCollector 会将 Adapter metadata 中的 `subprocess_peak_rss_mb` / `subprocess_avg_cpu_usage_pct` 与 Python worker 指标合并，防止低估 C runtime 的真实 RSS/CPU。

## 8. CosyVoice / MeloTTS

两者 hosted real CPU gate 已通过，继续使用独立 Python worker 与本地资产；不要把完整官方依赖强塞进主 UI venv。

## 9. 性能测试规则

- warm-up ≥1，正式测量建议 ≥3。
- 固定文本/voice/language/seed/线程与 runtime revision。
- Sequential 与 Concurrent 分开报告。
- hosted CI 与真实设备结果分开记录。
- 性能 gate 与音质 gate 分开：RTF/RSS 通过不能代表语音质量通过。
