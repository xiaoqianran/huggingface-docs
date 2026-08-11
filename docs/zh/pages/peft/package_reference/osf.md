<!-- huggingface-docs: machine-translated zh-CN from English source -->

# OSF（正交子空间微调）

正交子空间微调（[OSF](https://huggingface.co/papers/2504.07097)）是一种专为持续学习而设计的 PEFT 方法，它将参数更新限制为与之前重要的方向正交。这种方法可以实现全面的微调，同时防止灾难性遗忘，而不需要额外的参数或存储以前的梯度。

论文摘要是：

*大型语言模型（LLM）中的持续学习很容易出现灾难性遗忘，适应新任务会显着降低之前学习过的任务的性能。现有方法通常依赖于低秩、参数高效的更新，这限制了模型的表达能力并为每个任务引入额外的参数，从而导致可扩展性问题。为了解决这些限制，我们提出了一种利用自适应奇异值分解（SVD）的新颖的持续全面微调方法。我们的方法动态识别特定于任务的低秩参数子空间，并将更新限制为与先前任务相关的关键方向正交，从而有效地最小化干扰，而无需额外的参数开销或存储先前的任务梯度。我们评估使用编码器-解码器 (T5-Large) 和仅解码器 (LLaMA-2 7B) 模型在标准持续学习基准上广泛评估我们的方法，涵盖分类、生成和推理等各种任务。根据经验，我们的方法取得了最先进的结果，平均准确度比 O-LoRA 等最新基线高出 7%，并且通过将遗忘减少到几乎可以忽略不计的水平，在整个持续学习过程中显着保持了模型的一般语言能力、指令跟踪准确性和安全性。我们的自适应 SVD 框架有效地平衡了模型可塑性和知识保留，为大型语言模型中的持续学习场景提供了实用、有理论依据且计算可扩展的解决方案。*

## OSF 的工作原理

OSF 使用 SVD 将每个权重矩阵分解为高秩（冻结）和低秩（可训练）分量：

```
W = U_high * S_high * V_high^T + U_low * S_low * V_low^T
```

其中：
- `U_high, S_high, V_high`：保留之前任务的重要方向（冻结）
- `U_low, S_low, V_low`：允许适应新任务（可训练）

在训练期间，梯度被投影为与高秩子空间正交，确保更新不会干扰之前学到的知识。## 基本用法

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import OSFConfig, get_peft_model

# Load base model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Configure OSF
config = OSFConfig(
    target_modules=["c_attn", "c_proj"],  # Target attention layers
    effective_rank=8,                     # Default rank for decomposition
    rank_pattern={"c_attn": 16}          # Override rank for specific modules
)

# Apply OSF
model = get_peft_model(model, config)

# Train as usual
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
loss = model(**inputs, labels=inputs.input_ids).loss
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

## 配置选项

### 目标模块

您可以通过多种方式指定目标模块：

```python
# Specific module names
config = OSFConfig(target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# All linear layers
config = OSFConfig(target_modules="all-linear")

# Model-specific defaults (automatically detected)
config = OSFConfig()  # Uses model-appropriate defaults
```

### 有效排名配置

控制保留/可训练的子空间：

```python
# Global preserved rank (applies to all target modules)
config = OSFConfig(effective_rank=16)  # preserves top-16 singular directions; trains the rest

# Automatic preserved rank (50% of the smaller matrix dimension per target)
config = OSFConfig(effective_rank=None)

# Per-module preserved-rank overrides
config = OSFConfig(
    effective_rank=8,
    rank_pattern={
        "q_proj": 16,      # Higher rank for query projection
        "gate_proj": 4     # Lower rank for gate projection
    }
)

# Fractional preserved rank is supported (interpreted per-target as fraction * min_dim)
config = OSFConfig(effective_rank=0.8)  # preserve 80% of min_dim; train remaining 20%
config = OSFConfig(rank_pattern={"q_proj": 0.5})  # preserve 50% on q_proj, others use global/default
```

注意：OSF 的 `effective_rank` 是保留（冻结）的等级，而不是可训练的等级。可训练等级等于`min(weight.shape) - effective_rank`。这与LoRA的`r`不同，后者直接指定可训练的等级。

## 持续学习的培训建议

### 顺序任务学习

OSF 专为顺序学习任务而设计。在任务之间重新计算 SVD，以便保留的子空间反映最新的权重。一种简单的方法是再次使用 OSF 重新包装更新后的基础模型：

```python
# Task 1: train on domain A with initial preserved subspace
r = 8  # initial effective rank to preserve
model = get_peft_model(base_model, OSFConfig(effective_rank=r))
train_task(model, task_1_data)

# Task 2: recompute SVD on updated weights and increase preserved subspace
base_model = model.unload()  # unwrap base model without assuming internals
r += 4  # grow preserved subspace to include Task 1 knowledge
model = get_peft_model(base_model, OSFConfig(effective_rank=r))
train_task(model, task_2_data)

# Task 3: recompute again and expand preserved subspace further
base_model = model.unload()
r += 4
model = get_peft_model(base_model, OSFConfig(effective_rank=r))
train_task(model, task_3_data)
```

### 任务序列的预算分配

当对已知的 n 个任务序列进行训练时，一种有效的策略是逐步分配模型容量以平衡学习新任务，同时保留以前的知识：

- **任务 1**：充分利用能力（训练一切）
- **任务2**：冻结模型容量的1/n，训练剩余(n-1)/n容量
- **任务 3**：冻结 2/n 模型容量，训练剩余 (n-2)/n 容量
- **任务 n**：冻结 (n-1)/n 模型容量，将 1/n 容量用于最终任务这种方法确保每个任务获得足够的学习能力，同时逐步保留以前任务中的更多知识。

```python
# Example: 4-task sequence with progressive budget allocation
n_tasks = 4
max_preserved_rank = 512  # Upper bound for preserved rank per target (heuristic)

for task_id in range(n_tasks):
    # Freeze increases over time; trainable capacity shrinks
    preserved_fraction = (task_id + 1) / n_tasks
    preserved_rank = int(max_preserved_rank * preserved_fraction)

    config = OSFConfig(
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        effective_rank=preserved_rank,
    )

    print(
        f"Task {task_id + 1}: Preserving rank {preserved_rank} "
        f"({preserved_fraction:.1%} of max_preserved_rank - {max_preserved_rank} frozen); trainable rank = min_dim - preserved_rank"
    )

    model = get_peft_model(base_model, config)
    train_task(model, task_data[task_id])
```

### 最佳实践

1. **有效的排名选择**：从`effective_rank=None`开始（自动将排名设置为每个目标模块较小权重维度的50％）并根据任务复杂性进行调整
2. **学习率**：与标准微调相比，使用更小的学习率（1e-5 到 1e-4）
3. **任务重要性**：使用`rank_pattern`为关键模块分配更多容量
4. **模型架构**：OSF 最适合具有明确关注和 MLP 分离的变压器架构
5. **容量规划**：对于已知的任务序列，采用渐进式预算分配（1/n、2/n、...、(n-1)/n冻结）来平衡可塑性和稳定性

### 内存注意事项

OSF 就地修改权重并且不添加参数，从而提高内存效率：

```python
# Memory usage remains close to base model
print(f"Base model parameters: {base_model.num_parameters():,}")
print(f"OSF model parameters: {osf_model.num_parameters():,}")  # Similar count
```

## 高级用法

### 自定义目标模块

对于非标准架构的模型：

```python
config = OSFConfig(
    target_modules=["dense", "intermediate.dense"],  # Custom layer names
    effective_rank=12,
    rank_pattern={"dense": 8, "intermediate.dense": 16}
)
```

### 与其他方法集成

OSF 可以与其他技术结合使用：

```python
# Use with gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()

# Apply weight decay selectively (regularizes low-rank factors to limit drift/overfitting in continual updates; keep small)
optimizer = torch.optim.AdamW([
    {"params": [p for n, p in model.named_parameters() if "U_low" in n], "weight_decay": 0.01},
    {"params": [p for n, p in model.named_parameters() if "S_low" in n], "weight_decay": 0.001},
    {"params": [p for n, p in model.named_parameters() if "V_low" in n], "weight_decay": 0.01},
], lr=1e-4)
```

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=OSF"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## OSFConfig[[peft.OSFConfig]]

#### pft.OSFConfig[[peft.OSFConfig]]

```python
peft.OSFConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, effective_rank: Optional[Union[int, float]] = None, target_modules: Optional[Union[list[str], str]] = None, rank_pattern: Optional[dict[str, Union[int, float]]] = None, init_weights: Optional[bool] = None, modules_to_save: Optional[list[str]] = None, target_svd_config: Optional[dict[str, int]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/osf/config.py#L11)**参数：**

effective_rank（*int* 或 *float*，*可选*）：保留的 SVD 等级（“高”子空间）。 top-`effective_rank`奇异方向在任务中被冻结和保留；其余维度形成可训练的低秩子空间。如果*无*，则默认为每个目标模块较小重量尺寸的 50%。注意：这与 LoRA 的 *r*（可训练等级）不同。在 OSF 中，可训练排名为 *min(weight.shape) - effective_rank*。

target_modules (*Union[list[str], str]*, *可选*) ：要应用 OSF 的模块的名称。可以是模块名称列表或*“全线性”*。

rank_pattern (*dict[str, int|float]*, *可选*) ：用于覆盖特定模块的 * effective_rank * 的正则表达式模式字典。值可以是绝对整数或 (0, 1] 中的分数，解释为每个目标的较小矩阵维度的分数。

正交子空间微调 (OSF) 的配置。

## OSFModel[[peft.OSFModel]]

#### peft.OSFModel[[peft.OSFModel]]

```python
peft.OSFModel(model, config, adapter_name, low_cpu_mem_usage: bool = False, state_dict: dict[str, torch.Tensor] | None = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/osf/model.py#L14)

实现正交子空间微调的最小调谐器。

## 实用函数

### 权重分解[[peft.tuners.osf.utils.decompose_weight_matrix]]#### peft.tuners.osf.utils.decompose_weight_matrix[[peft.tuners.osf.utils.decompose_weight_matrix]]

```python
peft.tuners.osf.utils.decompose_weight_matrix(weight: torch.Tensor, top_k: int)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/osf/utils.py#L42)

执行 `weight` 的 SVD 并将其分成冻结和可训练的部分。

#### peft.tuners.osf.utils.reconstruct_weight_matrix[[peft.tuners.osf.utils.reconstruct_weight_matrix]]

```python
peft.tuners.osf.utils.reconstruct_weight_matrix(svd_dict: dict[str, torch.Tensor])
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/osf/utils.py#L62)

从 SVD 分量重建权重矩阵。

### 梯度投影[[peft.tuners.osf.utils.project_gradient_to_orthogonal_space]]

#### peft.tuners.osf.utils.project_gradient_to_orthogonal_space[[peft.tuners.osf.utils.project_gradient_to_orthogonal_space]]

```python
peft.tuners.osf.utils.project_gradient_to_orthogonal_space(svd_dict: dict[str, Any])
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/osf/utils.py#L84)

将 `U_low` 和 `V_low` 的梯度投影为与高阶空间正交。

### 广播电台
https://huggingface.co/docs/peft/v0.20.0/package_reference/boft.md