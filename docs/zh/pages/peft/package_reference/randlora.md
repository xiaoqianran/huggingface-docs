<!-- huggingface-docs: machine-translated zh-CN from English source -->

# RandLora：大型模型的全秩参数高效微调
[RandLora](https://huggingface.co/papers/2502.00987) 是一种参数高效的微调技术，类似于 [LoRA](https://huggingface.co/papers/2106.09685) 和 [VeRA](https://huggingface.co/papers/2310.11454)，但执行全秩更新以提高性能。当使大型模型适应需要复杂更新的困难任务，同时保持 LoRA 的参数效率时，RandLora 特别有用。 RandLora 的满秩更新是通过线性缩放随机基数来实现的。随机基是多个低秩矩阵的集合，使得它们的秩的总和大于或等于参数矩阵的满秩。 RandLora 的可训练参数是两个对角矩阵（向量），与右侧低秩随机基相乘，与 VeRA 的更新方式类似。为了保持较低的内存使用量，RandLora 使用自定义函数来防止在内存中存储不必要的碱基以进行反向传播。RandLora 呈现出值得注意的差异，与其他类似 LoRA 的 PEFT 算法相反，增加 RandLora 的随机基等级会增加可训练参数的数量。由于 RandLora 中的碱基数量 x 碱基等级是恒定的，因此减少等级将增加随机碱基的数量，从而增加特定于碱基的可训练对角碱基的数量。

由于降低 RandLora 随机碱基的等级会增加其数量，因此对于非常小的等级（通常等级低于 4），RandLora 的训练速度可能会比 LoRA 慢，从而导致训练时间大幅增加。但这并不影响推理，因为 RandLora 适配器可以合并到预训练的权重矩阵中。RandLora还支持使用稀疏、三元随机基数（仅包含-1、0和1）进行训练。这些基数如[Bingham et al.](https://cs-people.bu.edu/evimaria/cs565/kdd-rp.pdf)和[Ping et al.](https://hastie.su.domains/Papers/Ping/KDD06_rp.pdf)中所述，理论上可用于通过执行聚合而不是矩阵乘法来创建权重更新来减少计算需求。目前不支持此功能。虽然它目前不会减少计算量，但在 RandLora 中使用稀疏随机基可以减少某些情况下的过度拟合。对于对使用稀疏三元基感兴趣的用户，建议使用 `sparse` 选项，而不是会降低性能的 `very_sparse` 选项。

与 VeRA 类似，在保存 RandLora 的参数时，可以通过在 `VeraConfig` 上设置 `save_projection=False` 来避免存储低秩矩阵。在这种情况下，这些矩阵将根据 `projection_prng_key` 参数中的固定随机种子进行恢复。这减少了检查点的大小，但我们不能保证所有设备和所有未来版本的 PyTorch 的可重复性。如果要确保重现性，请设置`save_projection=True`（这是默认值）。与 Vera 中一样，为了处理不同形状的适应层，RandLora 使用每个维度所需的最大大小来初始化共享 A 和 B 矩阵。在前向传递过程中，给定层的子矩阵 A 和 B 被从这些共享矩阵中切出并按照论文中所述进行使用。例如，调整形状 (100, 20) 和 (80, 50) 的两个线性层将分别创建形状 (rank, 50) 和 (100, order) 的 A 和 B 矩阵。然后，为了适应形状 (100, 20) 的层，将提取形状 (rank, 20) 和 (100,rank) 的子矩阵 A 和 B。

RandLora 目前有以下限制：

- 仅支持`nn.Linear`层。

论文摘要是：> 低阶适应 (LoRA) 及其变体在减少大型变压器网络的可训练参数数量和内存需求，同时保持微调性能方面显示出令人印象深刻的结果。然而，权重更新的低秩性质本质上限制了微调模型的表示能力，从而可能会影响复杂任务的性能。这就提出了一个关键问题：当观察到 LoRA 和标准微调之间的性能差距时，是否是由于可训练参数数量减少或等级不足所致？本文旨在通过引入 RandLora 来回答这个问题，这是一种参数有效的方法，它使用低秩、不可训练的随机矩阵的学习线性组合来执行全秩更新。我们的方法通过限制对应用于固定随机矩阵的对角缩放矩阵的优化来限制可训练参数的数量。这使我们能够有效克服低秩限制，同时在训练期间保持参数和内存效率。通过跨视觉、语言和视觉语言基准的广泛实验，我们系统地评估了 LoRA 和现有随机基础方法的局限性。我们的研究结果表明，满秩更新对于单独的视觉和语言任务是有益的，对于视觉语言任务更是如此，其中 RandLora 显着减少（有时甚至消除）标准微调和 LoRA 之间的性能差距，证明了其功效。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=RANDLORA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## RandLoraConfig[[peft.RandLoraConfig]]

#### peft.RandLoraConfig[[peft.RandLoraConfig]]

```python
peft.RandLoraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, target_modules: typing.Union[str, list[str], NoneType] = None, projection_prng_key: int = 0, save_projection: bool = True, sparse: bool = False, very_sparse: bool = False, randlora_dropout: float = 0.0, fan_in_fan_out: bool = False, randlora_alpha: int = 640, bias: str = 'none', modules_to_save: typing.Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: typing.Union[list[int], int, NoneType] = None, layers_pattern: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/randlora/config.py#L24)

**参数：**r (`int`，*可选*，默认为`32`)：RandLora 的随机基础排名维度。与 Lora 相反，该参数与可训练参数的数量成反比，因为减少该参数会增加可训练参数。

target_modules (`Union[list[str], str]`) ：要应用 RandLora 的模块的名称。仅支持线性层。

projection_prng_key (`int`) ：RandLora PRNG 初始化密钥。用于初始化新模型的 basic_A 和 basic_B 或加载不包含这些投影的检查点时。默认为`0`。

save_projection (`bool`) ：是否将全局 basic_A / basic_B 随机基础与每层 lambda / gamma 对角矩阵一起保存在状态字典中。这将增加检查点的大小，但保证我们可以在所有系统配置上重新加载检查点。默认为`True`。稀疏 (`bool`) ：是否使用 RandLora 论文中描述的稀疏随机基。基数是三元稀疏基数（仅包含 -1、0 和 1），其中 -1 和 1 的归因概率为 1/6，0 的归因概率为 2/3。这些稀疏矩阵旨在将来用于无 matmul 计算，请参阅 https://huggingface.co/papers/2406.02528v1 当前的实现是概念证明，但稀疏性不用于提高速度或内存使用率。使用稀疏矩阵通常不会降低性能，甚至可以帮助减少过度拟合。默认为`False`。

very_sparse (`bool`) ：是否使用 RandLora 论文中描述的高度稀疏随机基。非常稀疏的基是三元稀疏基（仅包含 -1、0 和 1），给定最小维度 d 的矩阵，-1 和 1 的归因概率为 1/√D，0 的归因概率为 1- 2/√D。使用这些稀疏矩阵可以进一步减少对 `sparse` 替代方案的过度拟合，但很可能会降低性能。小心使用。默认为`False`。

randlora_dropout (`float`) ：RandLora 层的 dropout 概率。randlora_alpha (`float`) ：RandLora 层的缩放系数，通常是排名的 20 倍。由于默认情况下`randlora_alpha`系数很大，因此可能会导致数值不稳定，尤其是在学习率较高时。如果训练不稳定，可以考虑降低学习率或`randlora_alpha`系数。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) ：偏置类型。可以是“无”、“全部”或“randlora_only”。如果是“all”或“randlora_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

module_to_save (`list[str]`) ：除了 RandLora 层之外的模块列表，要设置为可训练并保存在最终检查点中。

init_weights (`bool`) ：是否使用默认初始化来初始化 RandLora 层的权重。除非您确切知道自己在做什么，否则请勿更改此设置。Layers_to_transform (`Union[list[int],int]`) ：要变换的图层索引，如果指定此参数，它将在此列表中指定的图层索引上应用 RandLora 变换。如果传递单个整数，它将在该索引处的图层上应用 RandLora 变换。

Layers_pattern (`str`) ：图层模式名称，仅当`layers_to_transform`与`None`不同且图层模式不在公共图层模式中时使用。

这是存储[RandLoraModel](/docs/peft/v0.20.0/en/package_reference/randlora#peft.RandLoraModel)配置的配置类。

论文：https://huggingface.co/papers/2502.00987。

## RandLoraModel[[peft.RandLoraModel]]

#### peft.RandLoraModel[[peft.RandLoraModel]]

```python
peft.RandLoraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/randlora/model.py#L67)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([RandLoraConfig](/docs/peft/v0.20.0/en/package_reference/randlora#peft.RandLoraConfig)) ：RandLora 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

兰德洛拉模型。

从预训练的 Transformer 模型创建 RandLoRA 模型。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import RandLoraConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = RandLoraConfig(r=32)
>>> model = get_peft_model(base_model, config)
```**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([RandLoraConfig](/docs/peft/v0.20.0/en/package_reference/randlora#peft.RandLoraConfig))：RandLora 模型的配置。

### 墨盒
https://huggingface.co/docs/peft/v0.20.0/package_reference/cartridges.md