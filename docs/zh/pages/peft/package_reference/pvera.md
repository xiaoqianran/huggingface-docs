<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PVeRA：基于概率向量的随机矩阵适应

[PVeRA](https://huggingface.co/papers/2512.07703) 是一种基于 VeRA 的参数高效微调技术，属于基于 LoRA 的适配器系列。它保持了 VeRA 非常低的参数预算，但通过学习潜在适应的分布来提高性能。这还使得采用 PVeRA 的模型能够通过在推理时从学习的分布中进行采样来生成蒙特卡洛置信区间估计。

保存适配器参数时，可以通过在 `PveraConfig` 上设置 `save_projection=False` 来避免存储低秩矩阵。在这种情况下，这些矩阵将根据 `projection_prng_key` 参数中的固定随机种子进行恢复。这减少了检查点的大小，但我们不能保证所有设备和所有未来版本的 PyTorch 的可重复性。如果要确保重现性，请设置`save_projection=True`（这是默认值）。为了处理不同形状的适应层，PVeRA 使用每个维度所需的最大大小来初始化共享 A 和 B 矩阵。在前向传递过程中，给定层的子矩阵 A 和 B 被从这些共享矩阵中切出并按照论文中所述进行使用。例如，调整形状 (100, 20) 和 (80, 50) 的两个线性层将分别创建形状 (rank, 50) 和 (100, order) 的 A 和 B 矩阵。然后，为了适应形状 (100, 20) 的层，将提取形状 (rank, 20) 和 (100,rank) 的子矩阵 A 和 B。

PVeRA 目前有以下限制：

- 仅支持`nn.Linear`层。
- 使用 KL 散度进行训练时，潜在表示不容易访问。

论文摘要是：> 大型基础模型在过去几年中出现，正在突破各种任务的性能极限。训练甚至微调此类模型需要大量数据集和计算资源，而这些数据集和计算资源通常稀缺且昂贵。适应方法提供了一种计算有效的解决方案，通过允许在少量数据和计算能力上对此类模型进行微调来解决这些限制。这是通过将新的可训练模块附加到仅具有一小部分可训练参数的冻结骨干网并仅将这些模块拟合到新任务来实现的。最近，VeRA 适配器通过利用跨所有层共享的一对冻结随机低秩矩阵，在参数高效自适应方面表现出色。在本文中，我们提出PVeRA，VeRA适配器的概率版本，它以概率方式修改VeRA的低秩矩阵。这种修改自然允许处理输入中固有的模糊性，并允许在训练和测试期间使用不同的采样配置。对 VTAB-1k 基准和七个适配器进行了全面评估，PVeRA 优于 VeRA 和其他适配器。## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PVERA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## PveraConfig[[peft.PveraConfig]]

#### peft.PveraConfig[[peft.PveraConfig]]

```python
peft.PveraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 256, target_modules: Optional[Union[list[str], str]] = None, projection_prng_key: int = 0, save_projection: bool = True, pvera_dropout: float = 0.0, d_initial: float = 0.1, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, sample_at_inference: bool = False, generator_seed: int = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/pvera/config.py#L25)

**参数：**

r（`int`，*可选*，默认为`256`）：PVeRA 参数维度（“排名”）。此处选择高于 LoRA 排名的值，因为 PVeRA 跨层共享参数，因此使用的参数比 LoRA 少得多。

target_modules (`Union[List[str], str]`) ：要应用 PVeRA 的模块的名称。仅支持线性层。传递字符串时，将执行正则表达式匹配。如果指定为“全线性”，则选择所有线性/Conv1D 模块。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误。

projection_prng_key (`int`) ：PVeRA PRNG 初始化密钥。用于初始化新模型的 pvera_A 和 pvera_B 或加载不包含这些投影的检查点时。默认为`0`。save_projection (`bool`) ：是否将 pvera_A / pvera_B 投影与每层 lambda_b / lambda_d 权重一起保存在状态字典中。这将增加检查点的大小，但保证我们可以在所有系统配置上重新加载检查点。默认为`True`。

pvera_dropout (`float`) ：PVeRA 层的 dropout 概率。

d_initial (`float`，*可选*，默认为`0.1`)：初始化PVeRA参数时使用的`pvera_lambda_d`向量的初始值。建议使用较小的值 (<=0.1)。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) ：PVeRA 的偏置类型。可以是“无”、“全部”或“pvera_only”。如果是“all”或“pvera_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

module_to_save (`List[str]`) ：除了 PVeRA 层之外的模块列表，要设置为可训练并保存在最终检查点中。init_weights (`bool`) ：是否使用默认初始化来初始化PVeRA层的权重。除非您确切知道自己在做什么，否则请勿更改此设置。

Layers_to_transform (`Union[List[int],int]`) ：要变换的图层索引，如果指定此参数，它将在此列表中指定的图层索引上应用 PVeRA 变换。如果传递单个整数，它将在该索引处的图层上应用 PVeRA 变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

Sample_at_inference (`bool` | `dict`，默认为 `False`) ：是否在推理时从学习到的 PVeRA 分布中进行采样。如果为 false，则使用学习到的平均值。默认值为 False（表示所有适配器均为 false）。如果提供 True，则该值对于所有适配器都将为 true。如果提供了字典，则可以为每个适配器指定一个特定值（对于未指定的适配器，默认情况下为 False）。例如，`sample_at_inference={'encoder.layer.0.attention.attention.query': True}`只会在推理时对一个特定适配器进行采样。Generator_seed (`int`，默认为 None) ：生成器的随机种子，用于从学习的分布中进行采样。

这是存储[PveraModel](/docs/peft/v0.20.0/en/package_reference/pvera#peft.PveraModel)配置的配置类。

论文：https://www.arxiv.org/abs/2512.07703。

## PveraModel[[peft.PveraModel]]

#### peft.PveraModel[[peft.PveraModel]]

```python
peft.PveraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/pvera/model.py#L35)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([PveraConfig](/docs/peft/v0.20.0/en/package_reference/pvera#peft.PveraConfig)) ：PVeRA 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

PVeRA 模型。

从预训练的 Transformer 模型创建基于概率向量的随机矩阵适应 (PVeRA) 模型。

示例：

```py
>>> from transformers import AutoModel
>>> from peft import PveraConfig, get_peft_model

>>> base_model = AutoModel.from_pretrained("facebook/dinov2-base")
>>> config = PveraConfig(r=128, sample_at_inference=False)
>>> model = get_peft_model(base_model, config)
```

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([PveraConfig](/docs/peft/v0.20.0/en/package_reference/pvera#peft.PveraConfig))：PVeRA 模型的配置。

### 格洛拉
https://huggingface.co/docs/peft/v0.20.0/package_reference/glora.md