# 需求规格说明书（PRD）

> 版本：v0.2 ｜ 状态：MVP 基线 ｜ 更新：2026-09-03

## 1. 产品定位

**EdgeTTS-Arena** 是面向 **1B 参数量以下、CPU/端侧部署优先** 的多模型 TTS 对比、性能评测与试听工作台。核心目标不是提供生产级 TTS SaaS，而是帮助开发者在同一硬件、同一测试语料和统一指标口径下完成模型选型。

## 2. 目标用户

| 用户 | 核心任务 | 关注指标 |
|---|---|---|
| 端侧 AI / 嵌入式开发者 | 为 x86、ARM、轻薄本等设备选择 TTS | RTF、内存、线程占用、部署复杂度 |
| 语音算法 / 评测人员 | 横向比较自然度、发音与长文本稳定性 | MOS、清晰度、韵律、G2P、重复/漏字 |
| 全栈开发者 / PM | 快速验证 TTS 交互方案 | Web UI、流式体验、导出与复现实验 |

## 3. 产品范围

### 3.1 MVP 模型范围

| 级别 | 模型 | 说明 |
|---|---|---|
| 必做 | Dummy Adapter | CI / API / UI 联调，不依赖真实模型 |
| 必做 | Piper | CPU 超轻量 baseline |
| 必做 | Kokoro | 轻量高质量模型候选 |
| 条件接入 | Qwen3-TTS 0.6B | 作为较大模型 CPU/量化路线验证；允许 feature flag |
| 第二批 | CosyVoice 2、MeloTTS | 主链路稳定后接入 |
| Phase 2 | Fish Speech | 不阻塞 MVP |

> 具体运行时、量化格式和性能必须以实际集成验证为准，不把文档中的候选路线视为已验证事实。

### 3.2 模型生命周期

- Lazy Loading。
- 可选 Keep-in-Memory。
- 显式 Unload。
- 大模型可配置互斥常驻策略。
- 模型状态至少包含：`unavailable / unloaded / loading / ready / busy / error`。

## 4. 核心功能

### 4.1 文本输入与预设

- 自定义多行文本。
- 预设题库：短对话、数字/符号、中英混读、多音字、长文本。
- 文本最大长度默认 1000 字符，阈值可配置。
- 长文本按标点分句后逐段执行并拼接。

### 4.2 参数与 Capability

统一请求层可以包含：

- `voice`
- `speed`
- `seed`
- `sample_rate`
- `stream`

但模型必须通过 capability 声明实际支持能力。前端应禁用不支持选项；API 若收到不支持参数，应根据参数类型返回 validation warning 或 4xx 错误，禁止静默伪造支持。

### 4.3 Arena 对比

一次选择 2~4 个模型：

- **Sequential**：默认评测模式，用于获得较干净的单模型指标。
- **Concurrent**：压力/吞吐模式，用于观察资源竞争，不与 Sequential 结果直接排名。

### 4.4 Blind AB Test

MVP 支持：

- 隐藏模型名并随机映射为 Model A/B/...。
- 试听后对自然度、清晰度、韵律打分。
- 提交后揭晓模型身份。
- 本地持久化结果，支持基础胜率/均分统计。

### 4.5 性能指标

- `inference_time_ms`
- `audio_duration_ms`
- `rtf = inference_time / audio_duration`
- `peak_rss_mb`
- `rss_delta_mb`
- `avg_cpu_usage_pct`
- `ttfb_ms`

TTFB 仅适用于真实流式输出。非流式模型返回 `null`，不得把整段生成完成时间伪装成 TTFB。

### 4.6 音频与结果展示

- WAV 播放与下载。
- 波形图；频谱图可作为增强功能。
- 展示采样率、时长、声道等基本属性。
- 横向指标表。

### 4.7 导出

单次 Benchmark 生成：

```text
run_<id>/
├── audio/
│   ├── <model-a>.wav
│   └── <model-b>.wav
├── benchmark_report.json
└── environment.json
```

支持一键 ZIP 导出。

## 5. 非功能需求

### 5.1 稳定性

- 单 Adapter 崩溃不得导致 API/Web 主进程退出。
- 推理超时可终止对应 worker。
- 模型异常结果必须保留明确 error code/message。

### 5.2 CPU 资源保护

- 默认线程数必须受控，不允许模型自行占满全部逻辑核心。
- OMP/MKL/PyTorch/ONNX Runtime 线程参数统一配置。
- CPU affinity 仅做 best-effort，不作为跨平台必需条件。

### 5.3 跨平台

目标环境：

- Linux x86_64 / ARM64
- Windows 11 x86_64
- macOS Apple Silicon

MVP 首先保证代码路径可跨平台；特定模型是否支持所有平台以 Adapter 状态返回为准。

## 6. MVP 验收标准

1. 无真实模型时，Dummy Adapter 可完成从 API → Metrics → UI → Export 的全链路。
2. 至少 Piper + Kokoro 两个真实模型可在 CPU 模式生成 WAV。
3. `GET /api/v1/system/models` 可返回状态与 capabilities。
4. `POST /api/v1/benchmark/run` 可同步返回 1~4 个模型结果。
5. 单模型失败不影响同一轮其他模型结果。
6. 报告包含环境快照和模型配置，可复现实验。
7. UI 能根据 capability 自动禁用不支持的参数。
