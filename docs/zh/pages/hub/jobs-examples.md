<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 示例和教程

## 火车模型

Transformers、TRL、Unsloth 和 Axolotl 的启动命令位于 [Train Models on Jobs](./jobs-training)。每一篇都链接到图书馆自己的指南。

## 大规模处理数据

[DataTrove](https://github.com/huggingface/datatrove) 提供了一个实验性的 [⟦T1⟧](https://github.com/huggingface/datatrove#jobspipelineexecutor)，用于跨作业池分配数据处理管道。它支持并发限制、多阶段依赖、重试和可恢复运行——重新运行管道会跳过已经完成的任务，只运行剩余的任务。

请参阅准备运行的示例：

- [Filtering a Hub dataset](https://github.com/huggingface/datatrove/blob/main/examples/filter_hf_dataset_jobs.py)
- [Tokenizing and merging a Hub dataset](https://github.com/huggingface/datatrove/blob/main/examples/tokenize_hf_dataset_jobs.py)
- [Multi-stage MinHash deduplication](https://github.com/huggingface/datatrove/blob/main/examples/minhash_deduplication_jobs.py)

## UV 脚本

[uv-scripts](https://huggingface.co/uv-scripts) 组织维护着一组独立的 uv 脚本，这些脚本可以通过单个命令在作业上运行。脚本涵盖 OCR、批量推理、文本分类、对象检测、数据集统计、嵌入可视化等。

[Unsloth](https://huggingface.co/datasets/unsloth/jobs) 还提供了可立即运行的训练脚本，用于在 Jobs 上微调 LLM 和 VLM。

## 编码代理技能

Claude Code、Codex 和 Cursor 等编码代理可以为您提交和监控作业。安装从已安装的 CLI 生成的 `hf` CLI 技能，使其保持最新状态：

```bash
hf skills add
```

请参阅 [Hugging Face CLI for AI agents](./agents-cli) 了解每个代理的设置，并参阅 [Agent Skills](./agents-skills) 了解培训和其他工作流程技能。

## 沙箱Jobs 的 [expose ports](./jobs-configuration#expose-ports) 功能使其非常适合构建沙箱，即代理和 LLM 应用程序使用的临时独立环境。

## 社区教程和项目

- [Train on massive datasets without downloading](https://danielvanstrien.xyz/posts/2026/hf-streaming-unsloth/train-massive-datasets-without-downloading.html) - 使用 Unsloth 直接在作业上流式传输数据集，无需本地存储
- [Fine-tune a vision-language model with TRL](https://danielvanstrien.xyz/posts/2025/iconclass-vlm-sft/trl-vlm-fine-tuning-iconclass.html) - 使用 TRL 和 Jobs 微调 Qwen2.5-VL 以执行艺术史任务
- [FreeFlow](https://github.com/wjbmattingly/freeflow) - 具有内置 Jobs 集成的开源注释平台，用于训练 YOLOv11 对象检测模型
- [hfdask](https://github.com/Hanno-Labs/hfdask) - 从一个 YAML 集群定义跨 CPU 和 GPU 作业运行 Dask 程序，并在节点之间使用 mTLS 和自动清理

---

有使用 Jobs 的教程或项目吗？ [Open a PR](https://github.com/huggingface/hub-docs/edit/main/docs/hub/jobs-examples.md) 将其添加到此处。

### 许可证
https://huggingface.co/docs/hub/repositories-licenses.md