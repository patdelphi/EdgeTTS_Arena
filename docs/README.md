# EdgeTTS-Arena 文档

> 文档基线：v0.7（2026-09-03）  
> 当前状态：**Stage 0~5 已实现；Stage 6 已完成主 Arena 跨平台部署基线、扩展模型 worker 隔离、CosyVoice/MeloTTS/Qwen3 官方真实 CPU gate，以及 Qwen3 pure-C native INT8/INT4 hosted 量化 gate。native INT8@2 threads 是当前 CPU 优化候选；真实弱设备与量化音质 A/B 仍待验证。**

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

- 主 Arena 本地部署范围：Dummy + Piper + Kokoro。
- `POST /benchmark/run` 与 `/benchmark/suite` 使用同步语义；`run_id` 是归档 ID。
- Standard Suite：warm-up 1 + measured 3；Sequential 是基准，Concurrent 是压力模式。
- `config.language` 已进入 benchmark API；CLI 支持 `suite --language`。
- `language_control` 是模型配置级 capability：Qwen official/Qwen native/Kokoro=true；MeloTTS 当前由 descriptor 固定语言，不接受全局 language override。
- Gradio 单模型模式提供 capability-aware Language 下拉；多模型时使用各模型默认语言。
- Qwen official FP32 2-thread hosted baseline：RTF 5.315 / peak RSS 5317.8MB。
- Qwen native hosted quant baseline：INT8@2 RTF 1.787 / 3124MB；INT8@4 RTF 1.984 / 3129MB；INT4@4 RTF 2.432 / 2899MB。
- 当前推荐 native INT8@2；INT4 仅低内存实验选项，不宣称质量等价。
- native runtime 自己管理 OpenBLAS thread split；不要预设 `OPENBLAS_NUM_THREADS`。
- 全部扩展模型继续默认 `experimental + disabled`，直到对应目标设备验证完成。
