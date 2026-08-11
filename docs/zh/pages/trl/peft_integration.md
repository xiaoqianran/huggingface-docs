<!-- huggingface-docs: machine-translated zh-CN from English source -->

#PEFT 集成

TRL 支持[PEFT](https://github.com/huggingface/peft)（参数高效微调）方法，以实现内存高效的模型训练。 PEFT 通过仅训练少量附加参数，同时保持基本模型冻结，可以对大型语言模型进行微调，从而显着降低计算成本和内存需求。

本指南介绍了如何将 PEFT 与不同的 TRL 训练器结合使用，包括 LoRA、QLoRA 和提示调整技术。

有关完整的工作示例，请参阅[SFT with LoRA/QLoRA notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb)。

## 安装

要将 PEFT 与 TRL 结合使用，请安装所需的依赖项：

```bash
pip install trl[peft]
```

对于 QLoRA 支持（4 位和 8 位量化），还需安装：

```bash
pip install bitsandbytes
```

## 快速入门

所有 TRL 培训师都通过 `peft_config` 论点支持 PEFT。启用 PEFT 的最简单方法是使用带有 `--use_peft` 标志的命令行界面：

```bash
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT-LoRA
```

或者，您可以直接在 Python 代码中传递 PEFT 配置：

```python
from peft import LoraConfig
from trl import SFTTrainer

# Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Configure training - note the higher learning rate for LoRA (10x base rate)
training_args = SFTConfig(
    learning_rate=2.0e-4,  # 10x the base rate (2.0e-5) for LoRA
    ...
)

# Create trainer with PEFT
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
)
```

## 配置 PEFT 的三种方法

TRL 提供了三种不同的方法来配置 PEFT，每种方法适合不同的用例：

### 1. 使用 CLI 标志（最简单）启用 PEFT 的最简单方法是在命令行界面中使用 `--use_peft` 标志。此方法非常适合快速实验和标准配置：

```bash
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --output_dir Qwen2-0.5B-SFT-LoRA
```

**优点**：快速设置，无需代码

**缺点**：仅限于 LoRA，定制选项较少

### 2.将peft_config传递给Trainer（推荐）

如需更多控制，请将 PEFT 配置直接传递给培训师。这是大多数用例的推荐方法：

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # Pass config here
)
```

**优点**：完全控制，支持所有 PEFT 方法（LoRA、Prompt Tuning 等）

**缺点**：需要Python代码

### 3.直接将 PEFT 应用于模型（高级）

为了获得最大的灵活性，您可以在将模型传递给训练器之前将 PEFT 应用于模型：

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer

# Load base model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Apply PEFT configuration
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, peft_config)

# Pass PEFT-wrapped model to trainer
trainer = SFTTrainer(
    model=model,  # Already has PEFT applied
    args=training_args,
    train_dataset=dataset,
    # Note: no peft_config needed here
)
```

**优点**：最大程度的控制，对于自定义模型架构或复杂的设置很有用

**缺点**：更冗长，需要了解 PEFT 内部结构

## 学习率注意事项

当使用 LoRA 或其他 PEFT 方法时，与完全微调相比，您通常需要使用**更高的学习率**（大约 10 倍）。这是因为 PEFT 方法仅训练一小部分参数，需要更大的学习率才能实现类似的参数更新。**推荐学习率：**

|培训师|全面微调 |与 LoRA (10x) |
|--------------------|--------------------------------|-----------------|
| **SFT** | `2.0e-5` | `2.0e-4` |
| **DPO** | `5.0e-7` | `5.0e-6` |
| **GRPO** | `1.0e-6` | `1.0e-5` |
| **及时调整** |不适用 | `1.0e-2` 至 `3.0e-2` |

> **为什么是 10x？** LoRA 适配器的可训练参数比完整模型少得多。更高的学习率可以补偿参数数量的减少，从而确保有效的训练。详细解释请参见[this blog post](https://thinkingmachines.ai/blog/lora/)。

有关有效使用 LoRA 的其他最佳实践，请参阅 [LoRA Without Regret](lora_without_regret) 文档。

## PEFT 与不同的培训师

TRL 的培训器支持各种培训范例的 PEFT 配置。以下是每个主要培训师的详细示例。

### 监督微调（SFT）

`SFTTrainer` 用于指令数据集的监督微调。

#### 与洛拉

```bash
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --learning_rate 2.0e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-SFT-LoRA
```

#### Python 示例

```python
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"],  # optional: specify target modules
)

# Configure training with higher learning rate for LoRA
training_args = SFTConfig(
    learning_rate=2.0e-4,  # 10x the base rate for LoRA
    ...
)

# Create trainer with PEFT config
trainer = SFTTrainer(
    model="Qwen/Qwen2-0.5B",  # can pass model name or loaded model
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # pass PEFT config here
)
trainer.train()
```

### 直接偏好优化 (DPO)

[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 根据人类反馈实现偏好学习。

#### 与洛拉

```bash
python trl/scripts/dpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --learning_rate 5.0e-6 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-DPO-LoRA
```

#### Python 示例

```python
from peft import LoraConfig
from trl import DPOConfig, DPOTrainer

# Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Configure training with higher learning rate for LoRA
training_args = DPOConfig(
    learning_rate=5.0e-6,  # 10x the base rate for DPO with LoRA
    ...
)

# Create trainer with PEFT config
trainer = DPOTrainer(
    model="Qwen/Qwen2-0.5B",  # can pass model name or loaded model
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # pass PEFT config here
)
trainer.train()
```**注意：** 将 PEFT 与 DPO 结合使用时，无需提供单独的参考模型 (`ref_model`)。训练器自动使用冻结的基础模型作为参考。

### 组相关策略优化 (GRPO)

`GRPOTrainer` 使用基于组的奖励来优化策略。

#### 与洛拉

```bash
python trl/scripts/grpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/math-reasoning \
    --learning_rate 1.0e-5 \
    --per_device_train_batch_size 2 \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --output_dir Qwen2-0.5B-GRPO-LoRA
```

#### Python 示例

```python
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

# Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Configure training with higher learning rate for LoRA
training_args = GRPOConfig(
    learning_rate=1.0e-5,  # 10x the base rate for GRPO with LoRA
    ...
)

# Create trainer with PEFT config
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B",  # can pass model name or loaded model
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # pass PEFT config here
)
trainer.train()
```

### 近端策略优化（PPO）

#### 多适配器强化学习训练

您可以将具有多个 PEFT 适配器的单个基本模型用于整个 PPO 算法 - 包括检索参考 logits、计算活动 logits 和计算奖励。这种方法对于内存高效的强化学习训练非常有用。

> [!警告]
> 此功能是实验性的，收敛性尚未经过广泛测试。我们鼓励社区分享反馈并报告任何问题。

**要求**

为 8 位模型安装 PEFT 和可选的 bitsandbytes：

```bash
pip install peft bitsandbytes
```

**培训工作流程**

多适配器方法需要三个阶段：1. **监督微调 (SFT)**：使用 `SFTTrainer` 在目标域（例如 IMDB 数据集）上训练基本模型
2. **奖励模型训练**：使用PEFT和`RewardTrainer`训练奖励模型适配器（参见[reward modeling example](https://github.com/huggingface/trl/tree/main/examples/scripts/reward_modeling.py)）
3. **PPO 培训**：使用 PPO 和奖励适配器微调新适配器

> [!重要]
> 对于阶段 2 和阶段 3 使用相同的基础模型（架构和权重）。

**基本用法**

训练您的奖励适配器并将其推送到 Hub 后：

```python
from peft import LoraConfig
from trl.experimental.ppo import PPOTrainer, AutoModelForCausalLMWithValueHead

model_name = "huggyllama/llama-7b"
rm_adapter_id = "trl-lib/llama-7b-hh-rm-adapter"

# Configure PPO adapter
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Load model with reward adapter
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    peft_config=lora_config,
    reward_adapter=rm_adapter_id,
)

trainer = PPOTrainer(model=model, ...)
```

在训练循环中，使用以下方法计算奖励：

```python
rewards = trainer.model.compute_reward_score(**inputs)
```

**高级功能**

**量化基础模型**

为了节省内存的训练，请以 8 位或 4 位加载基本模型，同时将适配器保留为 float32：

```python
from transformers import BitsAndBytesConfig

model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name,
    peft_config=lora_config,
    reward_adapter=rm_adapter_id,
    quantization_config=BitsAndBytesConfig(load_in_8bit=True),
)
```

## QLoRA：量化低阶适应

QLoRA 将 4 位量化与 LoRA 相结合，可以对消费类硬件上的大型模型进行微调。与标准 LoRA 相比，该技术可将内存需求减少多达 4 倍。

### QLoRA 的工作原理1. **4位量化**：使用`bitsandbytes`以4位精度加载基础模型
2. **冻结权重**：量化模型权重在训练期间保持冻结
3. **LoRA Adapters**：仅对LoRA适配器参数进行更高精度的训练
4. **内存效率**：能够在单个消费级 GPU 上对 Llama-70B 等模型进行微调

### 将 QLoRA 与 TRL 结合使用

只需将 `load_in_4bit=True` 与 PEFT 配置结合起来即可：

#### 命令行

```bash
python trl/scripts/sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --dataset_name trl-lib/Capybara \
    --load_in_4bit \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --output_dir Llama-2-7b-QLoRA
```

#### Python 示例

将 `quantization_config` 与 `peft_config` 一起直接传递给训练器 — 训练器为您加载并量化模型。相同的 `quantization_config` 参数可用于 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer)、[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer)、[GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) 和 [RLOOTrainer](/docs/trl/v1.9.2/en/rloo_trainer#trl.RLOOTrainer)。

```python
import torch

from peft import LoraConfig
from transformers import BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Configure LoRA
peft_config = LoraConfig(
    r=32,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# Configure training with higher learning rate for LoRA
training_args = SFTConfig(
    learning_rate=2.0e-4,  # 10x the base rate for QLoRA
    ...
)

# Create trainer with quantization and PEFT config
trainer = SFTTrainer(
    model="meta-llama/Llama-2-7b-hf",
    args=training_args,
    train_dataset=dataset,
    quantization_config=bnb_config,
    peft_config=peft_config,
)

trainer.train()
```

### QLoRA 配置选项

`BitsAndBytesConfig` 提供了多种选项来优化内存和性能：

```python
import torch

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # or "fp4"
    bnb_4bit_compute_dtype=torch.bfloat16,  # Compute dtype for 4-bit base models
    bnb_4bit_use_double_quant=True,  # Nested quantization for additional memory savings
)
```

**配置参数：**
- `bnb_4bit_quant_type`：量化数据类型（`"nf4"` 或 `"fp4"`）。推荐使用NF4。
- `bnb_4bit_compute_dtype`：用于计算的数据类型。使用`bfloat16`以获得更好的训练稳定性。
- `bnb_4bit_use_double_quant`：启用嵌套量化以节省每个参数额外的 ~0.4 位。

### 8 位量化

为了获得稍高的精度并减少内存节省，您可以使用 8 位量化：

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
)
```

或者通过命令行：

```bash
python trl/scripts/sft.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --load_in_8bit \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```## 及时调整

提示调整是另一种 PEFT 技术，它学习输入前的软提示（连续嵌入），同时保持整个模型冻结。这对于大型模型尤其有效。

### 快速调整的工作原理

1. **虚拟令牌**：向输入添加可学习的连续嵌入（虚拟令牌）
2. **冻结模型**：整个基础模型保持冻结状态
3. **任务特定提示**：每个任务都会学习自己的提示嵌入
4. **极高效率**：仅训练提示嵌入（通常为 8-20 个标记）

### 使用 TRL 进行提示调整

```python
from peft import PromptTuningConfig, PromptTuningInit, TaskType
from trl import SFTConfig, SFTTrainer

# Configure Prompt Tuning
peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=8,
    prompt_tuning_init_text="Classify if the tweet is a complaint or not:",
    tokenizer_name_or_path="Qwen/Qwen2-0.5B",
)

# Configure training with higher learning rate for Prompt Tuning
training_args = SFTConfig(
    learning_rate=2.0e-2,  # Prompt Tuning typically uses 1e-2 to 3e-2
    ...
)

# Create trainer with PEFT config
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,  # pass PEFT config here
)

trainer.train()
```

### 提示调整配置

```python
from peft import PromptTuningConfig, PromptTuningInit, TaskType

peft_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,  # Task type
    prompt_tuning_init=PromptTuningInit.TEXT,  # Initialize from text
    num_virtual_tokens=8,  # Number of virtual tokens
    prompt_tuning_init_text="Your initialization text here",
    tokenizer_name_or_path="model_name",
)
```

**配置参数：**
- `task_type`：任务类型（`TaskType.CAUSAL_LM`用于语言建模）
- `prompt_tuning_init`：初始化方法（`TEXT`、`RANDOM`）
- `num_virtual_tokens`：要添加的虚拟令牌数量（通常为 8-20）
- `prompt_tuning_init_text`：初始化虚拟令牌的文本（使用`TEXT` init 时）
- `tokenizer_name_or_path`：用于从文本初始化的标记器

### 快速调整与 LoRA|特色 |及时调整|洛拉 |
|--------|----------------|------|
| **参数训练** | ~0.001% | ~0.1-1% |
| **内存使用情况** |最小 |低|
| **训练速度** |最快|快|
| **模型修改** |无 |适配器层 |
| **最适合** |大模型，多任务 |一般微调|
| **学习率** |更高（1e-2 至 3e-2）|标准（1e-4 至 3e-4）|

## 高级 PEFT 配置

### LoRA 配置参数

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=16,  # LoRA rank
    lora_alpha=32,  # LoRA scaling factor
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # Bias training strategy
    task_type="CAUSAL_LM",  # Task type
    target_modules=["q_proj", "v_proj"],  # Modules to apply LoRA
    modules_to_save=None,  # Additional modules to train
)
```

**关键参数：**
- `r`：LoRA 等级（典型值：8、16、32、64）。更高的排名=更多的参数，但可能有更好的性能。
- `lora_alpha`：缩放因子（通常为等级的 2 倍）。控制 LoRA 更新的幅度。
- `lora_dropout`：LoRA 层的丢弃概率（典型值：0.05-0.1）。
- `target_modules`：应用 LoRA 的模块。常见选择：
  - `["q_proj", "v_proj"]`：注意力查询和值（内存高效）
  - `["q_proj", "k_proj", "v_proj", "o_proj"]`：所有注意力预测
  - `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`：所有线性层
- `modules_to_save`：用于全面训练的附加模块（例如，`["embed_tokens", "lm_head"]`）

### 目标模块选择

您可以指定要应用 LoRA 的模块。常见模式：

```python
# Minimal (most memory efficient)
target_modules=["q_proj", "v_proj"]

# Attention only
target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]

# All linear layers (best performance, more memory)
target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

### 使用命令行参数

TRL 脚本通过命令行接受 PEFT 参数：

```bash
python trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules q_proj v_proj \
    --output_dir output
```可用标志：
- `--use_peft`：启用 PEFT
- `--lora_r`：LoRA 等级（默认：16）
- `--lora_alpha`：LoRA alpha（默认值：32）
- `--lora_dropout`：LoRA 丢失（默认值：0.05）
- `--lora_target_modules`：目标模块（空格分隔）
- `--lora_modules_to_save`：需要训练的附加模块
- `--use_rslora`：启用排名稳定的 LoRA
- `--use_dora`：启用权重分解LoRA (DoRA)
- `--load_in_4bit`：启用 4 位量化 (QLoRA)
- `--load_in_8bit`：启用8位量化

## 保存和加载 PEFT 模型

### 保存

培训后，保存您的 PEFT 适配器：

```python
# Save the adapters
trainer.save_model("path/to/adapters")

# Or manually
model.save_pretrained("path/to/adapters")
```

这仅节省适配器权重（约几 MB），而不是完整模型（约几 GB）。

### 加载中

加载 PEFT 模型进行推理：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B")

# Load PEFT adapters
model = PeftModel.from_pretrained(base_model, "path/to/adapters")

# Optionally merge adapters into base model for faster inference
model = model.merge_and_unload()
```

### 推送到集线器

您可以在 Hugging Face Hub 上轻松共享您的 PEFT 适配器：

```python
# Push adapters to Hub
model.push_to_hub("username/model-name-lora")

# Load from Hub
from peft import PeftModel
model = PeftModel.from_pretrained(base_model, "username/model-name-lora")
```

## 多 GPU 训练

PEFT 通过 `accelerate` 与 TRL 的多 GPU 支持无缝协作：

```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch trl/scripts/sft.py \
    --model_name_or_path Qwen/Qwen2-0.5B \
    --dataset_name trl-lib/Capybara \
    --use_peft \
    --lora_r 32 \
    --lora_alpha 16
```

对于具有多个 GPU 的 QLoRA，基本模型会自动分片：

```bash
accelerate launch trl/scripts/sft.py \
    --model_name_or_path meta-llama/Llama-2-70b-hf \
    --load_in_4bit \
    --use_peft \
    --lora_r 32
```

### 大型模型的朴素管道并行性 (NPP)对于非常大的模型（>60B 参数），TRL 支持 Naive Pipeline Parallelism (NPP)，它将模型和适配器分布在多个 GPU 上。激活和梯度通过 GPU 进行通信，支持 `int8` 和其他数据类型。

![NPP](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/trl-npp.png)

**如何使用核电站**

使用自定义 `device_map` 加载模型以将其拆分到多个设备上：

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig

# Create custom device map (see accelerate documentation)
device_map = {
    "model.embed_tokens": 0,
    "model.layers.0": 0,
    # ... distribute layers across GPUs
    "lm_head": 0,  # Must be on GPU 0
}

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",
    device_map=device_map,
    peft_config=lora_config,
)
```

> [!重要]
> - 将`lm_head`模块保留在第一个GPU（设备0）上以避免错误
> - 请参阅此[tutorial on device maps](https://github.com/huggingface/blog/blob/main/accelerate-large-models.md)以了解正确的配置
> - 直接运行训练脚本（不使用`accelerate launch`）：`python script.py`
> - NPP 尚不支持数据并行

## 资源

### TRL 示例和笔记本

- **[SFT with LoRA/QLoRA Notebook](https://github.com/huggingface/trl/blob/main/examples/notebooks/sft_trl_lora_qlora.ipynb)** - 显示 LoRA 和 QLoRA 实现的完整工作示例
- **[TRL Examples Directory](https://github.com/huggingface/trl/tree/main/examples)** - 与不同培训师一起演示 PEFT 的培训脚本集合
- **[TRL Cookbook Recipes](https://github.com/huggingface/cookbook/tree/main/notebooks/transformers)** - 常见 PEFT 训练场景的分步指南

### 文档

- [PEFT Documentation](https://huggingface.co/docs/peft) - 官方 PEFT 库文档
- [TRL Documentation](https://huggingface.co/docs/trl) - 完整的 TRL 文档和培训师指南
- [LoRA Without Regret](lora_without_regret) - 有效使用 LoRA 的最佳实践

### 研究论文- [LoRA Paper](https://huggingface.co/papers/2106.09685) - 原始LoRA方法和结果
- [QLoRA Paper](https://huggingface.co/papers/2305.14314) - 通过 4 位量化进行高效微调
- [Prompt Tuning Paper](https://huggingface.co/papers/2104.08691) - 用于参数高效快速调整的规模力量

### 蒸馏训练器
https://huggingface.co/docs/trl/v1.9.2/distillation_trainer.md