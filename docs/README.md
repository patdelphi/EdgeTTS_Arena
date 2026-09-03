# EdgeTTS-Arena 文档

> 文档基线：v1.2（2026-09-03）  
> **主 Arena 已达到本地部署测试条件。** Stage 6 工程工具链已基本收口：扩展模型 pinned bootstrap、Qwen native bootstrap、Qwen quant hosted gate、Blind AB、单模型 target acceptance、Concurrent target calibration、evidence-pack isolation 与 graded OOM diagnostics 均已实现。剩余主要是人工盲听与真实目标硬件结果。

开发入口：

1. [06. 项目实施计划与路线图](./06_项目实施计划与路线图.md)
2. [04. API 与数据规范](./04_接口协议与数据规范.md)
3. [07. CPU 推理与量化部署](./07_CPU推理与量化部署实操指南.md)
4. [11. 实现检查清单](./11_实现检查清单.md)
5. [12. 本地部署与验收指南](./12_本地部署与验收指南.md)
6. [12. 第二批模型适配状态](./12_第二批模型适配状态.md)

冻结口径：

- 主 Arena：Dummy + Piper + Kokoro + API/UI/Benchmark，local deploy ready。
- Qwen official FP32 hosted baseline：RTF 5.315 / 5318MB。
- native INT8@2：RTF 1.787 / 3124MB；当前 CPU 优化候选。
- native INT4@4：RTF 2.432 / 2899MB；低内存实验项。
- `bootstrap_extended_model.py`：Qwen3 official/MeloTTS/CosyVoice pinned venv/runtime/assets/Doctor。
- `bootstrap_qwen3_native.py`：pinned pure-C runtime build/caps/self-test + official model + INT8/INT4 manifests。
- bootstrap 默认 plan-only；`--execute` 才执行联网重操作。
- `target_device_acceptance.py`：单模型真实设备多次 synthesis + environment + thresholds + ZIP。
- `target_device_concurrent_calibration.py`：Sequential baseline 与 Concurrent pressure 成对真机校准。
- 两类 target evidence 工具默认拒绝 stale output；显式 `--overwrite` 才清理旧证据重跑。
- Worker diagnostics 使用 `oom_classification` + `oom_evidence` 分级，不将裸 SIGKILL 冒充确定 OOM。
- human Blind AB / real target-device / real concurrent calibration 尚未执行，不得虚假标记 verified。
