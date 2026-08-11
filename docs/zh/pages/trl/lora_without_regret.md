<!-- huggingface-docs: machine-translated zh-CN from English source -->

#LoRA无悔

[Thinking Machines Lab](https://thinkingmachines.ai/blog/lora/) 团队的最新研究（Schulman 等人，2025）表明，如果配置正确，**LoRA 可以匹配完整的微调性能**，同时仅使用约 67% 的计算量。这些发现令 TRL 用户感到兴奋，因为它们易于实施，并且可以在较小的预算下提高模型性能。

本指南提供了在 TRL 中重现博客文章结果的简单说明。

> [!提示]
> 建议在遵循本指南之前阅读博客文章，或者同时查阅这两个资源以获得最佳结果。

## LoRA 相对于全面微调的优势

首先，让我们提醒自己[LoRA over full fine-tuning](https://huggingface.co/docs/trl/en/peft_integration)的好处。

LoRA 在基本模型之上添加了适配器层，其中包含的参数比基本模型本身少得多。这种设计降低了 GPU 内存需求并实现更高效的训练。正如[blog](https://thinkingmachines.ai/blog/lora/)中所述，这种方法最初被认为涉及性能权衡，尽管仔细的配置可以克服这种权衡并匹配完整的微调性能。  

## TRL 示例让我们根据博文的核心发现，在 TRL 脚本中实现和训练 LoRA 适配器。之后，我们将根据 TRL 结果重新审视每项发现。

### 监督微调（SFT）

该博文对 Hub 中的一系列模型和数据集执行 SFT，我们可以在 TRL 中重现。

|型号|数据集 |
| --- | --- |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [allenai/tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) |
| [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B) | [open-thoughts/OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) |

我们可以将这些发现与 TRL Python API 集成，如下所示：

```python

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

dataset = load_dataset("open-thoughts/OpenThoughts-114k", split="train")

peft_config = LoraConfig(r=256, lora_alpha=16, target_modules="all-linear")

training_args = SFTConfig(
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    report_to=["trackio"],
)

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    train_dataset=dataset,
    peft_config=peft_config,
    args=training_args,
)

trainer.train()

```

```bash

hf jobs uv run \
    --flavor a100-large \
    --timeout 8h \
    --secrets HF_TOKEN \
    "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name open-thoughts/OpenThoughts-114k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 16 \
    --lora_target_modules all-linear \
    --output_dir Qwen2.5-3B-OpenThoughts-LoRA \
    --report_to trackio \
    --push_to_hub

```

要使用拥抱脸部作业，请登录中心（`hf auth login`）并获得积极的[credit balance](https://huggingface.co/settings/billing)。详情请参阅[Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)。

```bash

uv run "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py" \
    --model_name_or_path Qwen/Qwen2.5-3B-Instruct \
    --dataset_name open-thoughts/OpenThoughts-114k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --eval_strategy no \
    --use_peft \
    --lora_r 256 \
    --lora_alpha 16 \
    --lora_target_modules all-linear \
    --output_dir Qwen2.5-3B-OpenThoughts-LoRA \
    --report_to trackio \
    --push_to_hub

```

要在本地运行脚本，您需要安装`uv`。查看[uv documentation](https://docs.astral.sh/uv/)了解更多详情。

训练开始后，您可以在 [Trackio](https://huggingface.co/trackio) 中监控进度，它将记录 URL。

### 强化学习（GRPO）

该博文对来自 Hub 的一系列模型和数据集执行 GRPO，我们可以再次在 TRL 中重现结果。

|型号|数据集 |
| --- | --- |
| [Llama-3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) | [GSM8k](https://huggingface.co/datasets/openai/gsm8k) |
| [Llama-3.1-8B-Base](https://huggingface.co/meta-llama/Llama-3.2-1B) | [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |
| [Qwen3-8b-base](https://huggingface.co/Qwen/Qwen3-8b-base) | [DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) |

对于强化学习，该博客使用了数学推理任务，我们可以将其重现为 Python 函数。我们可以使用 TRL Python API 来实现这些建议，如下所示：

```python

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from trl.rewards import reasoning_accuracy_reward

dataset = load_dataset("HuggingFaceH4/OpenR1-Math-220k-default-verified", split="train")

peft_config = LoraConfig(
    r=1,
    lora_alpha=32,
    target_modules="all-linear"
)

training_args = GRPOConfig(
    learning_rate=5e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    num_generations=8,
    generation_batch_size=8,
    report_to=["trackio"],
)

trainer = GRPOTrainer(
    model="Qwen/Qwen3-0.6B",
    reward_funcs=reasoning_accuracy_reward,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
)

trainer.train()

```

> [!警告]
> 此代码片段跳过上面定义的奖励函数，以保持示例简洁。

```bash

hf jobs uv run \
    --flavor a100-large \
    --timeout 4h \
    --secrets HF_TOKEN \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py" \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified \
    --output_dir grpo-full-qwen3-0.6b \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 0.0 \
    --max_grad_norm 1.0 \
    --beta 0.0 \
    --max_completion_length 4096 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules all-linear \
    --vllm_mode colocate \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_steps 200 \
    --report_to trackio
```

要使用拥抱脸部作业，请登录中心（`hf auth login`）并获得积极的[credit balance](https://huggingface.co/settings/billing)。详情请参阅[Jobs documentation](https://huggingface.co/docs/huggingface_hub/en/guides/jobs)。

```bash
uv run "https://huggingface.co/datasets/burtenshaw/lora-without-regrets/resolve/main/grpo.py" \
    --model_name_or_path Qwen/Qwen3-0.6B \
    --dataset_name HuggingFaceH4/OpenR1-Math-220k-default-verified \
    --output_dir grpo-full-qwen3-0.6b \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 0.0 \
    --max_grad_norm 1.0 \
    --beta 0.0 \
    --max_completion_length 4096 \
    --num_generations 16 \
    --generation_batch_size 16 \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 1 \
    --num_train_epochs 1 \
    --lora_r 1 \
    --lora_alpha 32 \
    --lora_dropout 0.0 \
    --lora_target_modules all-linear \
    --vllm_mode colocate \
    --save_strategy steps \
    --save_steps 50 \
    --save_total_limit 1 \
    --logging_steps 1 \
    --max_steps 200 \
    --report_to trackio
```

要在本地运行脚本，您需要安装`uv`。查看[uv documentation](https://docs.astral.sh/uv/)了解更多详情。

使用 GRPO 的强化学习脚本在 TRL 中作为自定义脚本实现，它使用上面所示的奖励函数。您可以在[⟦T12⟧](https://huggingface.co/datasets/burtenshaw/lora-without-regrets/blob/main/grpo.py) - 强化学习与 LoRA 最佳实践中查看

## LoRA 优化的主要发现

作者建议将 LoRA 应用于所有权重矩阵，而不是将其限制于注意力层，因为增加排名并不能弥补这一限制。在 TRL 中，可以使用 `--lora_target_modules all-linear` 进行配置，将 LoRA 应用于所有权重矩阵。我们能够使用 TRL 和 SmolLM3 模型重现博客文章的结果。我们使用上面的奖励函数和配置在[Math 220k dataset](https://huggingface.co/datasets/HuggingFaceH4/OpenR1-Math-220k-default-verified)上训练模型 500 步。如下图所示，LoRA 模型的平均训练奖励曲线与完整的微调曲线相匹配。

![train reward](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/5.png)

最重要的是，LoRA 模型使用的内存明显少于完整微调模型，如下图所示。

![memory usage](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/6.png)

这是我们用来训练上述模型的参数

|参数|洛拉 |完整金融时报 |
| --- | --- | --- |
| `--model_name_or_path` | HuggingFaceTB/SmolLM3-3B | HuggingFaceTB/SmolLM3-3B |
| `--dataset_name` | HuggingFaceH4/OpenR1-Math-220k-默认-已验证 | HuggingFaceH4/OpenR1-Math-220k-默认-已验证 |
| `--learning_rate` | 1.0e-5 | 1.0e-6 |
| `--max_completion_length` | 4096 | 4096 |
| `--lora_r` | 1 | - |
| `--lora_alpha` | 32 | 32 - |
| `--lora_dropout` | 0.0 | 0.0 - |
| `--lora_target_modules` |全线性| - |

让我们详细分析一下这篇博文的主要发现以及我们如何重现它们。

### 1. *LoRA 在应用于所有权重矩阵时表现更好*作者建议将 LoRA 应用于所有权重矩阵，而不是将其限制于注意力层，因为增加排名并不能弥补这一限制。

![all layers](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/1.png)

即使使用更高的等级来匹配参数计数，仅注意 LoRA 也表现不佳。在 TRL 中，可以使用 `--lora_target_modules all-linear` 进行配置，将 LoRA 应用于所有权重矩阵。  在Python中，我们可以这样做：

```python
from peft import LoraConfig  

peft_config = LoraConfig(target_modules="all-linear")  
```

### 2. *适配器需要足够的能力来从数据集中学习*

该博文建议使用足够的 LoRA 等级从数据集中学习。等级决定了 LoRA 适配器中可训练参数的数量。因此，“对于超过 LoRA 容量的数据集，LoRA 的性能低于 FullFT”。

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/3.png)

在 TRL 脚本中，我们可以使用 `--lora_r` 设置排名并根据我们正在训练的任务和数据集进行调整。该博客文章根据任务和数据集大小推荐以下排名：

强化学习任务通常需要较低的容量，因此可以使用较小的 LoRA 等级。这是因为策略梯度算法每集提取大约 1 位信息，需要最小的参数容量。该博客文章将 LoRA 匹配完整微调的理想数据集大小定义为“训练后规模”。我们可以用它来确定 SFT 和 RL LoRA 的推荐排名：

|任务类型|数据集大小 |推荐排名 |
| --- | --- | --- |
| **SFT** |培训后量表| 256 | 256
| **RL** |任何尺寸 | 1-32 | 1-32

### 3. *“FullFT 和高阶 LoRA 具有相似的学习曲线”*

与直觉相反，该博客文章建议使用比完全微调更高的学习率。在上表中，我们对 LoRA 使用 1.0e-5，对完全微调使用 1.0e-6。在TRL脚本中，我们可以使用`--learning_rate`来设置学习率。 LoRA 中的 \\( \frac{1}{r} \\) 缩放使得最佳学习率近似与排名无关。

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/2.png)

### 4. *“在某些情况下，LoRA 对大批量大小的容忍度低于完全微调。”*

该博客文章建议使用 < 32 的有效批量大小，因为作者发现 LoRA 对大批量大小的容忍度较低。这无法通过提高 LoRA 等级来缓解。在TRL脚本中，我们可以使用`--per_device_train_batch_size`和`--gradient_accumulation_steps`来设置批量大小。

![learning rate](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lora_without_regret/4.png)

## 要点使用 TRL，您可以高效地实施 LoRA 适配器来匹配完整的微调性能，应用核心见解（针对所有权重矩阵、选择正确的排名以及管理批量大小和学习率），而无需承担 FullFT 的大量计算成本。

## 引文

```bibtex
@article{schulman2025lora,  
    title        = {{LoRA Without Regret}},  
    author       = {John Schulman and Thinking Machines Lab},  
    year         = 2025,  
    journal      = {Thinking Machines Lab: Connectionism},  
    doi          = {10.64434/tml.20250929},  
    note         = {https://thinkingmachines.ai/blog/lora/}  
}  
```

### 数据集格式和类型
https://huggingface.co/docs/trl/v1.9.2/dataset_formats.md