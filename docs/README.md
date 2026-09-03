# EdgeTTS-Arena 文档

> 文档基线：v0.2（2026-09-03）  
> 当前状态：**Stage 0~4 已实现；Piper/Kokoro real CPU gate、Stage 3 API、Gradio Arena UI 与 Blind AB 均已落地；Qwen3-TTS 为 experimental/unavailable placeholder**

EdgeTTS-Arena 是面向 CPU/端侧部署优先的 TTS 多模型评测、横向对比与试听工作台。

## 开发入口

建议按以下顺序阅读：

1. [00. 开发基线与决策记录](./00_开发准备与文档审阅.md)
2. [06. 项目实施计划与路线图](./06_项目实施计划与路线图.md)
3. [02. 系统架构与技术设计](./02_系统架构与技术设计方案.md)
4. [04. API 与数据规范](./04_接口协议与数据规范.md)
5. [08. UI 与交互流程](./08_前端UI交互原型与界面流程设计.md)
6. [10. 开发测试与环境配置](./10_开发测试用例与环境配置文件.md)
7. [11. 实现检查清单](./11_实现检查清单.md)

## 完整文档

- [01. PRD](./01_PRD_需求规格说明书.md)
- [02. 系统架构与技术设计](./02_系统架构与技术设计方案.md)
- [03. 模型适配与量化集成规格](./03_模型适配与量化集成规格.md)
- [04. 接口协议与数据规范](./04_接口协议与数据规范.md)
- [05. Benchmark Suite](./05_评测基准与测试用例集.md)
- [06. 项目实施计划与路线图](./06_项目实施计划与路线图.md)
- [07. CPU 推理与量化部署指南](./07_CPU推理与量化部署实操指南.md)
- [08. 前端 UI 与交互流程](./08_前端UI交互原型与界面流程设计.md)
- [09. 异常处理、容灾与安全](./09_异常处理容灾与安全设计规范.md)
- [10. 开发测试与环境配置](./10_开发测试用例与环境配置文件.md)
- [11. 实现检查清单](./11_实现检查清单.md)

## 当前冻结口径

- 包目录：`src/edgetts_arena/`
- API 启动：`edgetts-arena serve`
- UI 启动：安装 `.[ui]` 后 `edgetts-arena serve --ui`
- Arena UI：`/arena/`，Gradio 6.x 可选层，挂载到同一 FastAPI
- 模型状态：`GET /api/v1/system/models`
- Benchmark：同步 `POST /api/v1/benchmark/run`
- Audio：`GET /api/v1/audio/download/{run_id}/{filename}`
- Export：`GET /api/v1/export/{run_id}`
- Streaming：`WS /api/v1/tts/stream?model=<model_id>`
- 归档标识：`run_id`
- 模型能力：capability-driven
- TTFB：仅真流式模型有效；非流式 UI 显示 N/A
- 默认执行：Sequential
- Blind AB：匿名映射与评分在揭晓后写入 `blind_scores.json`
- MVP 模型：Dummy + Piper + Kokoro；Qwen3-TTS 为 experimental

## 下一阶段

Stage 5：Preset benchmark、TC-01~05、warm-up / repeated benchmark 与统计报告。
