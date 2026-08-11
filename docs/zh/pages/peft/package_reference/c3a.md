<!-- huggingface-docs: machine-translated zh-CN from English source -->

# C3A：通过循环卷积进行参数高效微调

[C3A](https://huggingface.co/papers/2407.19342)是一种参数高效的微调技术，利用循环卷积在合理的资源限制内实现高秩自适应。

请注意，您应该为 C3A 使用比其他方法大得多的学习率 (LR)。例如，C3A 的 LR 为 1e-1 是一个很好的起点。此外，应该使用更小的权重衰减。您可以参考`method_comparison`文件夹了解更多详细信息。

对于`block_size`，它会影响可调参数和性能。首先，您可以选择 $\frac{\sqrt{d_1 \times d_2}}{r}$ 附近的 $\mathrm{gcd}(d_1,d_2)$，其中 $r$ 是您将用于此任务的 LoRA 的等级。

C3A目前有以下限制：

- 仅支持`nn.Linear`层。
- 不支持量化层。
- 块大小应该是目标层的输入和输出大小的公约数。

如果这些约束不适用于您的用例，请考虑其他方法。

论文摘要是：> 低秩适应 (LoRA) 在微调大型基础模型方面广受欢迎，利用低秩矩阵 $\mathbf{A}$ 和 $\mathbf{B}$ 来表示权重变化（即 $\Delta \mathbf{W} = \mathbf{B} \mathbf{A}$）。该方法通过按顺序将 $\mathbf{A}$ 和 $\mathbf{B}$ 与激活相乘来减少可训练参数并减轻与完整增量矩阵相关的大量内存消耗。尽管它取得了成功，但固有的低阶特性可能会限制其性能。尽管已经提出了几种变体来解决这个问题，但它们往往忽视了 LoRA 带来的关键计算和内存效率。在本文中，我们提出了循环卷积自适应（C3A），它不仅实现了具有增强性能的高秩自适应，而且在计算能力和内存利用率方面也表现出色。大量实验表明，C3A 在各种微调任务中始终优于 LoRA 及其变体。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=C3A"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## C3AConfig[[peft.C3AConfig]]

#### peft.C3AConfig[[peft.C3AConfig]]

```python
peft.C3AConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, block_size: int = 256, target_modules: Optional[Union[list[str], str]] = None, bias: str = 'none', modules_to_save: Optional[list[str]] = None, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, block_size_pattern: Optional[dict] = <factory>, init_weights: Optional[Union[bool, Literal['gaussian', 'kaiming_uniform', 'xavier_uniform']]] = 'xavier_uniform')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/c3a/config.py#L25)

**参数：**block_size (`int`) ：C3A的块大小，必须能被目标层的输入大小和输出大小整除。如果您不知道应该使用什么 block_size，请将其设置为目标层所有输入和输出大小的最大公约数。增加这个值会导致参数减少。

target_modules (`Union[list[str],str]`) ：要应用 C3A 的模块的名称。

bias (`str`) ：C3A 的偏置类型。可以是“无”、“全部”或“c3a_only”。如果是“all”或“c​​3a_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

module_to_save (`list[str]`) -- 除 C3A 层之外要设置为可训练并保存在最终检查点中的模块列表。

Layers_to_transform (`Union[list[int],int]`) ：要变换的图层索引，如果指定此参数，它将在此列表中指定的图层索引上应用 C3A。如果传递单个整数，它将在该索引处的图层上应用 C3A。Layers_pattern (`str`) ：图层模式名称，仅当`layers_to_transform`与`None`不同且图层模式不在公共图层模式中时使用。

block_size_pattern (`dict`) ：从图层名称或正则表达式到 block_size 的映射，与默认指定的不同。例如，`{"model.decoder.layers.0.encoder_attn.k_proj": 1280`}

init_weights (`Union[bool, Literal["gaussian", "kaiming_uniform", "xavier_uniform"]]`) ：默认为“xavier_uniform”。将其设置为 `False` 也会使用“xavier_uniform”。要将权重设置为零（从而使 C3A 成为无操作），请将值设置为 `True`。

这是存储[C3AModel](/docs/peft/v0.20.0/en/package_reference/c3a#peft.C3AModel)配置的配置类。

## C3AModel[[peft.C3AModel]]

#### peft.C3AModel[[peft.C3AModel]]

```python
peft.C3AModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/c3a/model.py#L29)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([C3AConfig](/docs/peft/v0.20.0/en/package_reference/c3a#peft.C3AConfig)) ：C3A模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**返回：** `torch.nn.Module`

C3A型号。

从预训练的 Transformer 模型创建 C3A 模型。

该方法在 https://huggingface.co/papers/2407.19342 中有详细描述。

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([C3AConfig](/docs/peft/v0.20.0/en/package_reference/c3a#peft.C3AConfig))：C3A模型的配置。### TinyLoRA：学习 13 个参数的推理
https://huggingface.co/docs/peft/v0.20.0/package_reference/tinylora.md