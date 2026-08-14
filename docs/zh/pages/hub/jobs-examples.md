<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 示例和教程

## 乔布斯培训指南

在作业中使用流行库的指南：

- [Training with TRL on Jobs](https://huggingface.co/docs/trl/jobs_training) - 使用 TRL 和 TRL 作业运行 SFT、GRPO、DPO 等
- [Fine-tune with Unsloth on Jobs](https://huggingface.co/blog/unsloth-jobs) - 使用 Unsloth 训练速度提高约 2 倍，VRAM 减少约 60%
- [Transformers example scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch) - 用于文本分类、摘要、图像分类、NER、语音识别等的 UV 兼容训练脚本 - 直接在作业上运行：

```bash
hf jobs uv run --flavor a10g-small --secrets HF_TOKEN \
  https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/image-classification/run_image_classification.py \
  --model_name_or_path google/vit-base-patch16-224-in21k \
  --dataset_name ethz/food101 \
  --output_dir vit-food101 \
  --push_to_hub
```

## UV 脚本

[uv-scripts](https://huggingface.co/uv-scripts) 组织维护着一组独立的 uv 脚本，这些脚本可以通过单个命令在作业上运行。脚本涵盖 OCR、批量推理、文本分类、对象检测、数据集统计、嵌入可视化等。

[Unsloth](https://huggingface.co/datasets/unsloth/jobs) 还提供了可立即运行的训练脚本，用于在 Jobs 上微调 LLM 和 VLM。

## 编码代理技能

[hugging-face-jobs skill](https://github.com/huggingface/skills/tree/main/skills/hugging-face-jobs) 允许 Claude Code 和 Cursor 等编码代理直接从编辑器提交和监控作业。

## 沙箱

Jobs 的 [expose ports](./jobs-configuration#expose-ports) 功能使其非常适合构建沙箱，即代理和 LLM 应用程序使用的临时独立环境。

## 社区教程和项目- [Train on massive datasets without downloading](https://danielvanstrien.xyz/posts/2026/hf-streaming-unsloth/train-massive-datasets-without-downloading.html) - 使用 Unsloth 直接在作业上流式传输数据集，无需本地存储
- [Fine-tune a vision-language model with TRL](https://danielvanstrien.xyz/posts/2025/iconclass-vlm-sft/trl-vlm-fine-tuning-iconclass.html) - 使用 TRL 和 Jobs 微调 Qwen2.5-VL 以执行艺术史任务
- [FreeFlow](https://github.com/wjbmattingly/freeflow) - 具有内置 Jobs 集成的开源注释平台，用于训练 YOLOv11 对象检测模型

---

有使用 Jobs 的教程或项目吗？ [Open a PR](https://github.com/huggingface/hub-docs/edit/main/docs/hub/jobs-examples.md) 将其添加到此处。

### 许可证
https://huggingface.co/docs/hub/repositories-licenses.md