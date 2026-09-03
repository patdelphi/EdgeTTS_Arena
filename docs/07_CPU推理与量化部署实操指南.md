# CPU 推理与量化部署实操指南

> 版本：v0.5 ｜ Qwen3 native quant hosted 性能 gate + Blind AB 工具链已纳入

## 1. 当前 Qwen CPU 决策

| Route | Threads | RTF | Peak RSS | 定位 |
|---|---:|---:|---:|---|
| official FP32 | 2 | 5.315 | 5318MB | 官方兼容/功能基线 |
| native INT8 | 2 | **1.787** | 3124MB | 当前推荐 CPU 候选 |
| native INT8 | 4 | 1.984 | 3129MB | hosted 对照 |
| native INT4 | 4 | 2.432 | **2899MB** | 低内存实验项 |

hosted CI 数据不是具体设备性能承诺。

## 2. Native runtime

Pinned pure-C runtime revision：`e56ec7e6eabbed608b13bfbd3fba431708b2077f`。Adapter 通过 manifest 固定 binary/model_dir/revision/quant/default voice/default language；load 先执行 `--caps`，CI build gate 执行 `--self-test`。

native C 由 Python Adapter 启动为子进程；MetricsCollector 合并 C 子进程 peak RSS/CPU。

**不要预设 `OPENBLAS_NUM_THREADS`**：该 runtime 自己按照 `-j/--threads` 设置 OpenBLAS。

## 3. 一次准备 INT8 + INT4

```bash
python scripts/prepare_qwen3_native_variants.py \
  --binary runtime/qwen3-tts-c/qwen_tts \
  --model-dir models/qwen3/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --output-root models/qwen3-native \
  --runtime-revision e56ec7e6eabbed608b13bfbd3fba431708b2077f \
  --default-voice Vivian \
  --default-language Chinese
```

得到：

```text
models/qwen3-native/int8/model.json
models/qwen3-native/int4/model.json
```

两份 manifest 共用同一 runtime revision/model snapshot/default voice/language，只改变 quantization。

## 4. Catalog

默认配置已有：

```text
qwen3-tts-0.6b-native-int8  -> int8/model.json, 2 threads
qwen3-tts-0.6b-native-int4  -> int4/model.json, 4 threads
```

两者默认都 disabled + experimental。INT4 的 4 threads 是已验证 hosted baseline，不代表所有 CPU 的最优线程数。

## 5. 公平 Blind AB

启用两者并启动 UI。多模型 capability view 会计算共同 voice 与 language 交集，例如两套 Qwen native 都可共同选择 Vivian / zh。Seed 两者都支持，因此可固定相同 seed。

推荐流程：

1. 选 INT8 + INT4。
2. 固定同一 text、Vivian、zh、seed=42。
3. Sequential run。
4. Start Blind AB。
5. 每个匿名 Sample 独立评分 Naturalness / Intelligibility / Prosody（1~5）。
6. 所有 sample 完成后才 Reveal。
7. 保存 ZIP：`blind_scores.json` 与 `benchmark_report.json` 一起用于追溯 runtime/quant/metrics。

单一文本结果不足以做质量结论。至少应覆盖短交互、长句、数字/英文混读，以及目标语言文本；不同语言应分别运行一轮 Blind AB。

## 6. Gate 判定

- WAV sanity / RTF / RSS 属于功能与性能 gate。
- Blind AB 属于人工质量 gate。
- 真实 ARM/树莓派属于设备 gate。

当前：性能工具与质量工具都已完成；**人工质量 gate 尚未执行**。
