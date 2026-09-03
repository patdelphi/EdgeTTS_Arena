# EdgeTTS-Arena 文档

> 文档基线：v0.4（2026-09-03）  
> 当前状态：**Stage 0~5 已实现；Stage 6 已完成跨平台/ARM64/1-CPU hosted smoke、cgroup-aware CPU/内存预算、非持久模型 watchdog、Piper/Kokoro 主线真实 CPU gate、MeloTTS/CosyVoice 独立真实 CPU gate，以及扩展模型 external Python/venv worker 路由。主 Arena 已达到本地部署测试条件；真实 ARM/弱算力目标设备仍待实机验证。**

开发入口：

1. [00. 开发基线与决策记录](./00_开发准备与文档审阅.md)
2. [06. 项目实施计划与路线图](./06_项目实施计划与路线图.md)
3. [02. 系统架构与技术设计](./02_系统架构与技术设计方案.md)
4. [03. 模型适配与量化集成](./03_模型适配与量化集成规格.md)
5. [04. API 与数据规范](./04_接口协议与数据规范.md)
6. [05. Benchmark Suite](./05_评测基准与测试用例集.md)
7. [07. CPU 推理与量化部署](./07_CPU推理与量化部署实操指南.md)
8. [08. UI 与交互流程](./08_前端UI交互原型与界面流程设计.md)
9. [11. 实现检查清单](./11_实现检查清单.md)
10. [12. 本地部署与验收指南](./12_本地部署与验收指南.md)
11. [12. 第二批模型适配状态](./12_第二批模型适配状态.md)

当前冻结口径：

- `src/edgetts_arena/`
- `edgetts-arena serve` / optional `serve --ui`
- `edgetts-arena doctor` / `doctor --ui` / `doctor --workers`
- `GET /api/v1/system/models`
- `POST /api/v1/benchmark/run`
- `GET /api/v1/benchmark/presets`
- `POST /api/v1/benchmark/suite`
- `run_id` 统一归档
- Sequential 为标准基准；Concurrent 是压力模式
- Standard Suite 默认 warm-up 1 + measured 3
- 原始 measurement + aggregate statistics 同时保存
- TTFB 仅真流式有效
- Blind AB 写入 `blind_scores.json`
- 主 Arena 本地部署范围：Dummy + Piper + Kokoro
- Qwen3：experimental/unavailable placeholder
- Batch 2：MeloTTS 与 CosyVoice 300M SFT real CPU gate 已通过，继续默认 `experimental + disabled`
- Batch 2 runtime：通过 `worker_python` / `worker_python_env` 使用专用 external Python worker，避免污染主 UI venv
- 默认环境变量：`EDGETTS_ARENA_COSYVOICE_PYTHON`、`EDGETTS_ARENA_MELOTTS_PYTHON`
- `/system/models` 返回 `worker_mode` / `worker_python_configured`，不返回本机解释器绝对路径
- CosyVoice WeText：安装准备阶段显式下载本地 FST；推理/load 阶段不得隐式 `snapshot_download()`
