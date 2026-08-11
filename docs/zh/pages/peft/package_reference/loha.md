<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 乐哈

    

导航文本到图像定制：从 LyCORIS 微调到模型评估

低秩哈达玛积 ([LoHa](https://huggingface.co/papers/2108.06098)) 与 LoRA 类似，不同之处在于它用更多低秩矩阵来逼近大权重矩阵，并将它们与哈达玛积结合起来。该方法比 LoRA 的参数效率更高，并且实现了可比的性能。 LoHa 最初是为联邦学习 (FedPara) 提出的，但作为通用 PEFT 方法效果很好，并且特别适用于微调图像生成模型（例如稳定扩散）。

> **注：** LoHa 是 [LyCORIS](./adapter_utils) 适配器系列的一部分。它的近亲[LoKr](./lokr)使用克罗内克积而不是哈达玛积。

论文摘要是：*在这项工作中，我们提出了一种通信高效的参数化 FedPara，用于联邦学习 (FL)，以克服频繁模型上传和下载的负担。我们的方法使用低秩权重和哈达玛乘积重新参数化层的权重参数。与传统的低秩参数化相比，我们的 FedPara 方法不受低秩约束的限制，因此具有更大的容量。这一特性使得能够实现相当的性能，同时所需的通信成本比具有原始层的模型低 3 到 10 倍，这是传统的低秩方法无法实现的。通过与其他高效的 FL 优化器相结合，我们的方法的效率可以进一步提高。此外，我们将我们的方法扩展到个性化 FL 应用程序 pFedPara，它将参数分为全局参数和局部参数。我们证明，pFedPara 的参数数量比竞争性个性化 FL 方法要少三倍多。*低秩分解会影响性能，因为权重更新仅限于低秩空间，这会限制模型的表达能力。但是，您不一定要使用更大的等级，因为它会增加可训练参数的数量。为了解决这个问题，LoHa 被应用于扩散模型，其中生成不同图像的能力是一个重要的考虑因素。 LoHa 还应该适用于一般模型类型，但 PEFT 目前尚未实现对嵌入层的支持。

LoHa 使用 [Hadamard product](https://en.wikipedia.org/wiki/Hadamard_product_(matrices))（逐元素乘积）而不是矩阵乘积。 $\Delta W$ 由四个较小的矩阵表示，而不是像 LoRA 中那样由两个较小的矩阵表示，并且每对这些低秩矩阵都与 Hadamard 乘积相结合。因此，$\Delta W$ 可以具有相同数量的可训练参数，但具有更高的秩和表达能力。

## 何时使用 LoHa

在以下情况下，LoHa 是一个不错的选择：- 您正在微调**图像生成模型**（稳定扩散UNet或文本编码器），它的使用最广泛。
- 对于相同数量的可训练参数，您希望比 LoRA 具有**更高的有效秩**，因为两个低秩矩阵的 Hadamard 乘积比单个低秩乘积跨越更大的子空间。
- 您希望使用[⟦T4⟧](./peft_model#peft.PeftMixedModel)在推理时**组合不同的PEFT方法**，例如LoHa与LoKr。

LoHa 支持线性和 Conv2d 层。对于额外需要嵌入层自适应的任务，请考虑[LoRA](./lora)。

## 用法

```python
from diffusers import StableDiffusionPipeline
from peft import LoHaConfig, get_peft_model

config_unet = LoHaConfig(
    r=8,
    alpha=8,
    target_modules=[
        "to_k",
        "to_q",
        "to_v",
        "to_out.0",
        "proj_in",
        "proj_out",
    ],
    rank_dropout=0.0,
    module_dropout=0.0,
    use_effective_conv2d=True,
)

pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.unet = get_peft_model(pipeline.unet, config_unet)
pipeline.unet.print_trainable_parameters()
```

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LOHA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## LoHaConfig[[peft.LoHaConfig]]

#### peft.LoHaConfig[[peft.LoHaConfig]]

```python
peft.LoHaConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, rank_pattern: Optional[dict] = <factory>, alpha_pattern: Optional[dict] = <factory>, r: int = 8, alpha: int = 8, rank_dropout: float = 0.0, module_dropout: float = 0.0, use_effective_conv2d: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/loha/config.py#L24)

**参数：**

r (`int`) ：LoHa 等级。

alpha (`int`) ：LoHa 缩放的 alpha 参数。

rank_dropout (`float`) ：训练期间Rank维度的dropout概率。

module_dropout (`float`) ：训练期间禁用 LoHa 模块的 dropout 概率。

use_ effective_conv2d (`bool`) ：对 ksize > 1 的 Conv2d（和 Conv1d）使用参数有效分解（来自 FedPara 论文的“命题 3”）。target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

init_weights (`bool`) ：是否执行适配器权重的初始化。默认为 `True`，不鼓励传递 `False`。Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

rank_pattern (`dict`) ：从图层名称或正则表达式到与`r`指定的默认排名不同的排名的映射。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

alpha_pattern (`dict`) ：从图层名称或正则表达式到 alpha 的映射，与 `alpha` 指定的默认 alpha 不同。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

module_to_save (`Optional[List[str]]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

这是存储[LoHaModel](/docs/peft/v0.20.0/en/package_reference/loha#peft.LoHaModel)配置的配置类。

## LoHaModel[[peft.LoHaModel]]

#### peft.LoHaModel[[peft.LoHaModel]]

```python
peft.LoHaModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/loha/model.py#L27)

**参数：**

model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。config ([LoHaConfig](/docs/peft/v0.20.0/en/package_reference/loha#peft.LoHaConfig)) ：LoHa 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

LoHa 模型。

从预训练模型创建低秩 Hadamard 产品模型。该方法的部分描述见
https://huggingface.co/papers/2108.06098 当前的实现大量借鉴了
https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/loha.py

示例：
```py
>>> from diffusers import StableDiffusionPipeline
>>> from peft import LoHaModel, LoHaConfig

>>> config_te = LoHaConfig(
...     r=8,
...     lora_alpha=32,
...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
...     rank_dropout=0.0,
...     module_dropout=0.0,
...     init_weights=True,
... )
>>> config_unet = LoHaConfig(
...     r=8,
...     lora_alpha=32,
...     target_modules=[
...         "proj_in",
...         "proj_out",
...         "to_k",
...         "to_q",
...         "to_v",
...         "to_out.0",
...         "ff.net.0.proj",
...         "ff.net.2",
...     ],
...     rank_dropout=0.0,
...     module_dropout=0.0,
...     init_weights=True,
...     use_effective_conv2d=True,
... )

>>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> model.text_encoder = LoHaModel(model.text_encoder, config_te, "default")
>>> model.unet = LoHaModel(model.unet, config_unet, "default")
```

**属性**：
- **model** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([LoHaConfig](/docs/peft/v0.20.0/en/package_reference/loha#peft.LoHaConfig))：LoHa 模型的配置。

### 多肌球
https://huggingface.co/docs/peft/v0.20.0/package_reference/poly.md