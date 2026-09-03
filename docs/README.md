# EdgeTTS-Arena 文档

> 文档基线：v0.3（2026-09-03）  
> 当前状态：**Stage 0~5 已实现；Stage 6 已完成跨平台/ARM64/1-CPU hosted smoke、cgroup-aware CPU/内存并发预算、非持久模型 watchdog 与 worker 退出/OOM 诊断；MeloTTS 已通过可复现真实 CPU synthesis gate，CosyVoice 与真实弱算力设备继续验证**

开发入口：

1. [00. 开发基线与决策记录](./00_开发准备与文档审阅.md)
2. [06. 项目实施计划与路线图](./06_项目实施计划与路线图.md)
3. [02. 系统架构与技术设计](./02_系统架构与技术设计方案.md)
4. [04. API 与数据规范](./04_接口协议与数据规范.md)
5. [05. Benchmark Suite](./05_评测基准与测试用例集.md)
6. [08. UI 与交互流程](./08_前端UI交互原型与界面流程设计.md)
7. [11. 实现检查清单](./11_实现检查清单.md)

当前冻结口径：

- `src/edgetts_arena/`
- `edgetts-arena serve` / optional `serve --ui`
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
- MVP：Dummy + Piper + Kokoro；Qwen3 experimental placeholder
- Batch 2：MeloTTS Adapter 已通过 GitHub-hosted x86_64 real CPU gate但继续保持 experimental/disabled；CosyVoice SFT real CPU gate 待验证
