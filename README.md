# EdgeTTS-Arena

EdgeTTS-Arena 是一个 **CPU/端侧优先** 的本地 TTS 多模型对比、性能评测与试听工作台。

当前实现状态：**Stage 0 + Stage 1 已实现，本地 15 tests passed；Stage 2 首批真实模型下一步。**

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

## 下一阶段

**Stage 2 — First Adapters**：开始真实 CPU TTS 接入，优先 Piper，然后 Kokoro；Qwen3-TTS 保持 experimental/feature flag。
