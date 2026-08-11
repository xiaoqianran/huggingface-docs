<!-- huggingface-docs: machine-translated zh-CN from English source -->

# VeRA：基于向量的随机矩阵适应

[VeRA](https://huggingface.co/papers/2310.11454) 是一种参数高效的微调技术，与 LoRA 类似，但需要更少的额外参数，同时承诺类似甚至更好的性能。因此，当参数预算非常有限时，例如，它特别有用。当缩放到非常大的模型时。通过在所有层之间共享相同的低秩矩阵，并且每层仅训练两个附加向量，可以减少可训练参数的数量。

## VerA 的工作原理

LoRA 通过为每个适应层学习两个小的低秩矩阵来更新冻结的基础模型。排名是
这些矩阵的内部尺寸，它控制适配器有多少容量。在LoRA中，增加`r`或
适应更多层可以快速增加可训练参数的数量，因为每个适应层都会学习自己的参数
一对低秩矩阵。VeRA 保持相同的低秩适应思想，但改变了训练的参数。而不是单独学习
每层都有低秩矩阵，VeRA 使用一对冻结的、随机初始化的低秩矩阵，`A` 和 `B`，
这是跨层共享的。每个适应层仅学习两个缩放向量，`vera_lambda_d`和`vera_lambda_b`，
它重新调整共享矩阵以产生特定于层的更新。由于 `vera_lambda_d` 的大小为 `r`，VeRA 的
可训练参数数量仍然随着 `r` 和适应层数的增加而增加。然而它却长得更多
比 LoRA 慢，因为大的 `A` 和 `B` 矩阵是共享和冻结的，而不是单独学习
层。

这就是为什么 VeRA 可以使用比 LoRA 更少的可训练参数。在论文中简化的参数计数中，LoRA
与`2 * L_tuned * d_model * r`缩放，其中`L_tuned`是适应层数，`d_model`是模型
维度，`r` 为秩。 VeRA 使用 `L_tuned * (d_model + r)` 进行缩放，因为大型低秩矩阵是
共享和冻结，而仅训练较小的每层向量。保存适配器参数时，可以通过在 `VeraConfig` 上设置 `save_projection=False` 来避免存储低秩矩阵。在这种情况下，这些矩阵将根据 `projection_prng_key` 参数中的固定随机种子进行恢复。这减少了检查点的大小，但我们不能保证所有设备和所有未来版本的 PyTorch 的可重复性。如果要确保再现性，请设置`save_projection=True`（这是默认值）。

为了处理不同形状的适应层，VeRA 使用每个维度所需的最大大小来初始化共享 A 和 B 矩阵。在前向传递过程中，给定层的子矩阵 A 和 B 被从这些共享矩阵中切出并按照论文中所述进行使用。例如，调整形状 (100, 20) 和 (80, 50) 的两个线性层将分别创建形状 (rank, 50) 和 (100, order) 的 A 和 B 矩阵。然后，为了适应形状 (100, 20) 的层，将提取形状 (rank, 20) 和 (100,rank) 的子矩阵 A 和 B。

VeRA 目前有以下限制：

- 仅支持`nn.Linear`层。

论文摘要是：> 低秩适应 (LoRA) 是一种流行的方法，可在微调大型语言模型时减少可训练参数的数量，但在扩展到更大的模型或部署大量每用户或每任务适应模型时仍然面临严峻的存储挑战。在这项工作中，我们提出了基于向量的随机矩阵自适应（VeRA），与 LoRA 相比，它显着减少了可训练参数的数量，但保持了相同的性能。它通过使用在所有层之间共享的一对低秩矩阵并学习小缩放向量来实现这一点。我们展示了它在 GLUE 和 E2E 基准、图像分类任务上的有效性，并展示了它在 7B 和 13B 语言模型的指令调优中的应用。

## 何时使用 VerRA

在以下情况下，VeRA 是一个不错的选择：

- 您希望最大限度地减少可训练参数的数量，同时保持与 LoRA 相当的性能。
- 您需要存储或部署许多特定于任务的适配器，其中较小的适配器检查点会降低存储要求。
- 您正在内存或参数预算紧张的情况下微调非常大的语言模型。

## 何时不使用 VeRA

在以下情况下，VeRA 可能不是最佳选择：- 您需要为每个适应层提供独立的低秩矩阵，从而为学习的适配器参数提供更大的灵活性。
- 您的模型需要调整 `nn.Linear` 以外的模块类型，因为 VeRA 目前仅支持线性层。
- 您的模型包含具有广泛不同的输入和输出维度的适应线性层。由于 VeRA 在所有适应层上共享一对投影矩阵，因此必须根据最大形状调整这些矩阵的大小。因此，层形状变化较大的模型（例如，变压器上投影层和下投影层）可能需要过度配置共享投影矩阵，从而降低了 VeRA 的一些参数效率优势。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=VERA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## VerAConfig[[peft.VeraConfig]]

#### peft.VeraConfig[[peft.VeraConfig]]

```python
peft.VeraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 256, target_modules: Optional[Union[list[str], str]] = None, projection_prng_key: int = 0, save_projection: bool = True, vera_dropout: float = 0.0, d_initial: float = 0.1, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/vera/config.py#L25)

**参数：**

r（`int`，*可选*，默认为`256`）：VeRA 参数维度（“rank”）。此处选择比 LoRA 等级更高的值，因为 VeRA 使用的参数比 LoRA 少得多（参见表 1）。

target_modules (`Union[List[str], str]`) ：要应用 Vera 的模块的名称。仅支持线性层。projection_prng_key (`int`) ：Vera PRNG 初始化密钥。用于初始化新模型的 vera_A 和 vera_B 或加载不包含这些投影的检查点时。默认为`0`。

save_projection (`bool`) ：是否将 vera_A / vera_B 投影与每层 lambda_b / lambda_d 权重一起保存在状态字典中。这将增加检查点的大小，但保证我们可以在所有系统配置上重新加载检查点。默认为 `True`。

vera_dropout (`float`) ：Vera 层的 dropout 概率。

d_initial (`float`，*可选*，默认为`0.1`)：初始化VeRA参数时使用的`vera_lambda_d`向量的初始值。建议使用较小的值（<=0.1）（参见论文中的表 6c）。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。bias (`str`) ：Vera 的偏置类型。可以是“无”、“全部”或“vera_only”。如果是“all”或“vera_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

module_to_save (`List[str]`) ：除 Vera 层之外的模块列表，要设置为可训练并保存在最终检查点中。

init_weights (`bool`) ：是否使用默认初始化来初始化 Vera 层的权重。除非您确切知道自己在做什么，否则请勿更改此设置。

Layers_to_transform (`Union[List[int],int]`) ：要变换的图层索引，如果指定此参数，它将在此列表中指定的图层索引上应用 Vera 变换。如果传递单个整数，它将在该索引处的图层上应用 Vera 变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

这是存储[VeraModel](/docs/peft/v0.20.0/en/package_reference/vera#peft.VeraModel)配置的配置类。论文：https://huggingface.co/papers/2310.11454。

## VerAModel[[peft.VeraModel]]

#### peft.VeraModel[[peft.VeraModel]]

```python
peft.VeraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/vera/model.py#L78)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([VeraConfig](/docs/peft/v0.20.0/en/package_reference/vera#peft.VeraConfig)) ：Vera 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

维拉模型。

从预训练的 Transformer 模型创建基于向量的随机矩阵适应 (Vera) 模型。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import VeraConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = VeraConfig(r=128)
>>> model = get_peft_model(base_model, config)
```

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([VeraConfig](/docs/peft/v0.20.0/en/package_reference/vera#peft.VeraConfig))：Vera 模型的配置。

### 道路
https://huggingface.co/docs/peft/v0.20.0/package_reference/road.md