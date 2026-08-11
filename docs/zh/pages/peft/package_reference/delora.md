<!-- huggingface-docs: machine-translated zh-CN from English source -->

# DeLoRA：解耦低阶适应
[DeLoRA](https://huggingface.co/papers/2503.18225) 是一种参数高效的微调技术，通过标准化和缩放可学习的低秩矩阵，隐式地维护相对于预训练权重的 Frobenius 边界。这有效地解耦了权重更新的方向（BA 项）和幅度（边界项）的学习，避免了适应权重的灾难性变化并增强了超参数选择的鲁棒性。

注意：
- 使用比标准 LoRA 变体大 10-100 倍的学习率（典型值来自 1e-3/1e-2/..）
- 确保初始边界参数 lambda 不太小（典型值约为 10/15/..）。可以为不同层设置不同的 lambda

DeLoRA 目前有以下限制：
- 仅支持 nn.Linear 层。
- 不支持量化层。

如果这些约束不适用于您的用例，请考虑其他方法。

论文摘要是：> 由于大规模预训练模型的广泛使用，参数高效微调 (PEFT) 方法最近获得了广泛的欢迎。这些方法允许以最小的计算成本快速适应下游任务。然而，LoRA 等流行的微调方法在超参数选择或扩展训练方案方面表现出有限的稳健性，从而无法实现最佳的开箱即用性能。相比之下，有界方法（例如 ETHER）提供了更大的鲁棒性，但仅限于极低秩的适应和固定强度的变换，从而降低了它们的适应表达能力。在这项工作中，我们提出了解耦低秩适应（DeLoRA），这是一种新颖的微调方法，可以对可学习的低秩矩阵进行归一化和缩放。通过限制变换的距离，DeLoRA 有效地将角度学习与适应强度解耦，在不影响性能的情况下增强鲁棒性。通过对主题驱动的图像生成、自然语言理解和指令调整的评估，我们表明 DeLoRA 可以匹配或超越竞争性 PEFT 方法的性能，同时展现出强大的性能。更强的鲁棒性。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=DELORA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## DeloraConfig[[peft.DeloraConfig]]

#### peft.DeloraConfig[[peft.DeloraConfig]]

```python
peft.DeloraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, delora_lambda: int = 15, module_dropout: float = 0.0, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, bias: Literal['none', 'all', 'delora_only'] = 'none', init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, rank_pattern: Optional[dict] = <factory>, lambda_pattern: Optional[dict] = <factory>, modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/delora/config.py#L24)

**参数：**

r (`int`) ：DeLoRA 适配器的等级。

delora_lambda (`int`) ：DeLoRA 适配器边界的初始值。该变量设置了权重变化的 Frobenius 范数的上限，避免微调后的模型与原始模型偏差太大。

module_dropout (`float`) ：训练期间禁用 DeLoRA 模块的 dropout 概率。target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。偏差 (`str`) ：DeLoRA 的偏差类型。可以是“无”、“全部”或“delora_only”。如果是“all”或“delora_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

init_weights (`bool`) ：是否执行适配器权重的初始化。如果`True`（默认）：A使用kaiming统一初始化进行初始化，而B使用0进行初始化。如果`False`：A和B都使用kaiming Uniform初始化，立即贡献一个非零增量。对于正常使用来说，通常不鼓励这样做。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。rank_pattern (`dict`) ：从图层名称或正则表达式到与`r`指定的默认排名不同的排名的映射。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

lambda_pattern (`dict`) ：从层名称或正则表达式到 lambda 的映射，与 `delora_lambda` 指定的默认 lambda 不同。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

module_to_save (`Optional[List[str]]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

这是存储[DeloraModel](/docs/peft/v0.20.0/en/package_reference/delora#peft.DeloraModel)配置的配置类。

## DeloraModel[[peft.DeloraModel]]

#### peft.DeloraModel[[peft.DeloraModel]]

```python
peft.DeloraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/delora/model.py#L28)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([DeloraConfig](/docs/peft/v0.20.0/en/package_reference/delora#peft.DeloraConfig)) ：DeLoRA 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**返回：** `torch.nn.Module`

DeLoRA 模型。

从预训练的 Transformer 模型创建 DeLoRA 模型。

该方法在[DeLoRA: Decoupled Low-rank
Adaptation](https://huggingface.co/papers/2503.18225)中有详细描述。

**属性**：
- **模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([DeloraConfig](/docs/peft/v0.20.0/en/package_reference/delora#peft.DeloraConfig))：DeLoRA 模型的配置。### FRoD：带旋转度的全阶高效微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/frod.md