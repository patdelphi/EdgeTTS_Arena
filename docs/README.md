# EdgeTTS-Arena 文档

> 文档基线：v1.0（2026-09-03）  
> 当前状态：**Stage 0~5 已实现；主 Arena 已达到本地部署测试条件，可以开始本地部署测试。Stage 6 已完成扩展模型 worker、扩展模型 pinned bootstrap、Qwen3 official FP32、native INT8/INT4 hosted real CPU gate、量化 Blind AB 工具链与真实目标设备验收工具。INT8@2 是当前 CPU 优化候选；人工盲听结果和真实弱设备结果仍待产生。**

开发入口：

1. [06. 项目实施计划与路线图](./06_项目实施计划与路线图.md)
2. [04. API 与数据规范](./04_接口协议与数据规范.md)
3. [07. CPU 推理与量化部署](./07_CPU推理与量化部署实操指南.md)
4. [11. 实现检查清单](./11_实现检查清单.md)
5. [12. 本地部署与验收指南](./12_本地部署与验收指南.md)
6. [12. 第二批模型适配状态](./12_第二批模型适配状态.md)

当前冻结口径：

- **主 Arena ready：Dummy + Piper + Kokoro + API/UI/Benchmark；可开始本地部署测试。**
- Qwen official FP32 hosted baseline：RTF 5.315 / 5318MB。
- native INT8@2：RTF 1.787 / 3124MB；当前推荐 CPU 候选。
- native INT4@4：RTF 2.432 / 2899MB；低内存实验项。
- `qwen3-tts-0.6b-native-int8` 与 `qwen3-tts-0.6b-native-int4` 都在默认 catalog 中，但 disabled/experimental。
- `prepare_qwen3_native_variants.py` 可一次生成同 revision/model/default voice/language 的成对 manifest。
- 多模型 UI 对 voice/language 使用 capability intersection，可为 quant variants 固定相同条件。
- Blind AB 记录 Naturalness/Intelligibility/Prosody 并写入 `blind_scores.json`；工具 ready 不代表质量通过。
- `target_device_acceptance.py` 已提供真实设备多次 synthesis、环境记录、最差值阈值和 ZIP 验收包；工具 ready 不代表目标硬件已验证。
- `bootstrap_extended_model.py` 已提供 Qwen3/MeloTTS/CosyVoice pinned 专用 venv/runtime/assets/preflight/Worker Doctor；默认 plan-only，只有 `--execute` 执行联网重操作。
- bootstrap 会生成 `bootstrap_plan.json`、`env.sh`、`env.ps1`；bootstrap success 不代表 real-model synthesis 或目标设备性能已经通过。
- MeloTTS/CosyVoice heavy gate 已复用本地 preparation helper，降低 CI/local setup 漂移。
- Qwen3 standalone snapshot prep 默认 pin 到 Arena 已真实验证 revision。
- native runtime 不应预设 `OPENBLAS_NUM_THREADS`。
