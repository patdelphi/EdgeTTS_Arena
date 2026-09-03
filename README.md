# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前实现状态：**Stage 0~2 已完成：Piper 与 Kokoro 已通过真实 CPU CI；Qwen3-TTS 0.6B 以 experimental/unavailable placeholder 接入，不伪造未验证的 CPU runtime。**

- 开发规格：[`docs/README.md`](./docs/README.md)
- 开发基线与冻结决策：[`docs/00_开发准备与文档审阅.md`](./docs/00_开发准备与文档审阅.md)
- 实施路线：[`docs/06_项目实施计划与路线图.md`](./docs/06_项目实施计划与路线图.md)
- 实现检查清单：[`docs/11_实现检查清单.md`](./docs/11_实现检查清单.md)

## Stage 0 已实现

- `pyproject.toml` + `src/edgetts_arena/` 包结构
- `BaseTTSAdapter` / `TTSCapabilities` / `TTSOutput`
- 统一配置、日志和基础错误类型
- `DummyTTSAdapter`，支持 deterministic infer + simulated streaming
- WAV 输出工具
- `config/app_config.yaml` / `models_config.yaml` / benchmark presets
- pytest smoke tests
- GitHub Actions：Python 3.10 / 3.11 / 3.12

## 开发环境

当前为保证 TTS 运行时兼容性，Python 支持范围冻结为：

```text
Python >=3.10,<3.13
```

安装：

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

执行测试：

```bash
pytest
```

## Dummy Smoke Test

无需下载真实模型即可验证最小音频链路：

```bash
python -m edgetts_arena dummy \
  --text "EdgeTTS Arena smoke test" \
  --seed 7 \
  --output exports/dummy.wav
```

也可以使用安装后的命令：

```bash
edgetts-arena dummy --text "hello" --output exports/dummy.wav
```

## Stage 1 已实现

- `ModelRegistry`：配置加载、模型状态、lazy load / unload
- `MetricsCollector`：Inference Time、Audio Duration、RTF、RSS、CPU；非流式 TTFB=`None`
- `ResourceGuard`：soft/hard memory guard、线程数约束
- `ProcessRunner`：独立子进程执行、异常回传、timeout terminate/kill
- CLI subprocess 回归测试，防止 package-level circular import 再次漏过 CI

## Stage 2 — Piper

Piper 使用当前维护的 `piper-tts` Python API。Adapter 不在模块 import 阶段加载 Piper runtime，因此没有安装真实模型依赖时，Dummy/核心测试仍可正常运行。

安装 Piper 可选依赖：

```bash
python -m pip install -e ".[piper]"
```

下载一个官方 voice：

```bash
python -m piper.download_voices en_US-lessac-low --data-dir models/piper
```

使用 EdgeTTS-Arena 合成：

```bash
python -m edgetts_arena piper \
  --model models/piper/en_US-lessac-low.onnx \
  --text "Edge TTS Arena Piper test." \
  --threads 2 \
  --output exports/piper.wav
```

当前 Piper Adapter 支持：

- `.onnx` + 同目录 `.onnx.json` voice 加载
- 单 speaker / multi-speaker voice
- `speed` 映射到 Piper `length_scale`
- Piper `noise_scale` / `noise_w_scale` / `volume` / `normalize_audio` 参数透传
- `infer()` 合并多 sentence chunk
- `infer_stream()` 按 Piper sentence chunk 输出
- runtime/model/version/voice 等 metadata
- 不支持 `seed` 时返回 capability conflict，而不是静默忽略

CI 中另有真实 Piper CPU smoke job：安装 `piper-tts`、下载 `en_US-lessac-low`、调用本项目 CLI 并验证 WAV 非空。

> 第三方许可：当前维护的 Piper runtime（OHF-Voice/piper1-gpl）为 GPL-3.0；官方 `rhasspy/piper-voices` 中 `en_US-lessac-low` voice 标注为 MIT。若发行包内捆绑 Piper runtime，需要单独检查并遵守其许可证要求。

## Stage 2 — Kokoro

安装 Kokoro 可选依赖：

```bash
python -m pip install -e ".[kokoro]"
```

Adapter 使用 `kokoro-onnx>=0.6.1` + ONNX Runtime，并通过自建 `InferenceSession` 控制 CPU 线程。当前直接文本能力先冻结为 `en-us` / `en-gb`；其他语言在独立 G2P 路线验证前不宣称支持。CI 已使用官方 v1.0 int8 ONNX（约 88 MB）完成真实 CPU 合成验证。

```bash
python -m edgetts_arena kokoro \
  --model models/kokoro/kokoro-v1.0.int8.onnx \
  --voice af_heart \
  --text "Edge TTS Arena Kokoro test." \
  --threads 2 \
  --output exports/kokoro.wav
```

## Stage 2 — Qwen3-TTS experimental

`Qwen3TTSAdapter` 目前是有意保持不可用的占位实现：

- `config/models_config.yaml` 默认 `enabled: false`、`experimental: true`。
- 不绑定尚未冻结的社区 CPU conversion/runtime。
- capability 只反映“当前 Adapter 真正实现的能力”，因此全部保持 false/empty。
- `load_model()` 明确返回 `experimental_runtime_unavailable`，`infer()` 不生成伪音频。
- 社区 ncnn/C++ CPU 路线只作为后续 POC 候选，必须通过可复现模型包、许可证、内存和 benchmark gate 后才能转为正式 runtime。

## 下一阶段

**Stage 3 — API**：FastAPI app + schemas、统一模型状态接口、同步 benchmark API、导出下载安全和 streaming capability gate。
