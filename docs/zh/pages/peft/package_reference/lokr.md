<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 乐克尔

    

导航文本到图像定制：从 LyCORIS 微调到模型评估

低秩克罗内克积 ([LoKr](https://hf.co/papers/2309.14859)) 是一种 LoRA 变体方法，它用两个低秩矩阵近似大权重矩阵，并将它们与 [Kronecker product](https://en.wikipedia.org/wiki/Kronecker_product) 组合。 LoKr 还提供了可选的第三个低秩矩阵，以在微调过程中提供更好的控制。通过将权重更新矩阵表示为克罗内克乘积的分解，创建块矩阵，LoKr 能够保留原始权重矩阵的秩。较小矩阵的大小由其 *rank* 或 `r` 决定。克罗内克乘积的另一个好处是它可以通过堆叠矩阵列来矢量化。这可以加快该过程，因为您可以避免完全重建 ΔW。

论文摘要是：*文本到图像生成模型因其根据文本提示生成高保真图像的能力而受到极大关注。其中，Stable Diffusion 是这个快速发展领域中领先的开源模型。然而，微调这些模型的复杂性带来了从新方法整合到系统评估的多重挑战。为了解决这些问题，本文介绍了 LyCORIS [Lora beYond Conventional methods, Other Rank adaptation Implementations for Stable diffusion](https://github.com/KohakuBlueleaf/LyCORIS)，这是一个开源库，为稳定扩散提供了多种微调方法。此外，我们提出了一个全面的框架，用于系统评估各种微调技术。该框架采用了多种指标，并深入研究了微调的多个方面，包括超参数调整和跨不同概念类别的不同提示类型的评估。通过这种综合方法，我们的工作为微调参数的细微差别提供了重要的见解，弥合了最先进的研究和实际应用之间的差距。*

## 用法

```py
from peft import LoKrConfig, get_peft_model

config = LoKrConfig(
    r=16,
    alpha=16,
    target_modules=["query", "value"],
    module_dropout=0.1,
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 116,069 || all params: 87,172,042 || trainable%: 0.13314934162033282"
```

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LOKR"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## LoKrConfig[[peft.LoKrConfig]]#### peft.LoKrConfig[[peft.LoKrConfig]]

```python
peft.LoKrConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, rank_pattern: Optional[dict] = <factory>, alpha_pattern: Optional[dict] = <factory>, r: int = 8, alpha: int = 8, rank_dropout: float = 0.0, module_dropout: float = 0.0, use_effective_conv2d: bool = False, decompose_both: bool = False, decompose_factor: int = -1, rank_dropout_scale: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, init_weights: Union[bool, Literal['lycoris']] = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lokr/config.py#L24)

**参数：**

r (`int`) ：LoKr 排名。

alpha (`int`) ：LoKr 缩放的 alpha 参数。

rank_dropout (`float`) ：训练期间Rank维度的dropout概率。

module_dropout (`float`) ：训练期间禁用 LoKr 模块的 dropout 概率。

use_ effective_conv2d (`bool`) ：对 ksize > 1 的 Conv2d（和 Conv1d）使用参数有效分解（来自 FedPara 论文的“命题 3”）。

decompose_both (`bool`) ：对左克罗内克乘积矩阵进行秩分解。

decompose_factor (`int`) ：克罗内克乘积分解因子。

rank_dropout_scale ('bool) ：训练时是否缩放Rank dropout，默认为`False`。target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

init_weights (`bool`) ：是否执行适配器权重的初始化。默认为 `True`。使用“lycoris”以LYCORIS存储库的风格初始化权重。不鼓励通过`False`。Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

rank_pattern (`dict`) ：从图层名称或正则表达式到与`r`指定的默认排名不同的排名的映射。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

alpha_pattern (`dict`) ：从图层名称或正则表达式到 alpha 的映射，与 `alpha` 指定的默认 alpha 不同。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

module_to_save (`Optional[List[str]]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

[LoKrModel](/docs/peft/v0.20.0/en/package_reference/lokr#peft.LoKrModel)的配置类别。

## LoKrModel[[peft.LoKrModel]]

#### peft.LoKrModel[[peft.LoKrModel]]

```python
peft.LoKrModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lokr/model.py#L27)

**参数：**

model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。

config ([LoKrConfig](/docs/peft/v0.20.0/en/package_reference/lokr#peft.LoKrConfig)) ：LoKr 模型的配置。adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

LoKr 模型。

从预训练模型创建低秩克罗内克积模型。原始方法部分描述于
https://huggingface.co/papers/2108.06098 和 https://huggingface.co/papers/2309.14859 当前实施
大量借贷
https://github.com/KohakuBlueleaf/LyCORIS/blob/eb460098187f752a5d66406d3affade6f0a07ece/lycoris/modules/lokr.py

示例：
```py
>>> from diffusers import StableDiffusionPipeline
>>> from peft import LoKrModel, LoKrConfig

>>> config_te = LoKrConfig(
...     r=8,
...     lora_alpha=32,
...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
...     rank_dropout=0.0,
...     module_dropout=0.0,
...     init_weights=True,
... )
>>> config_unet = LoKrConfig(
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
>>> model.text_encoder = LoKrModel(model.text_encoder, config_te, "default")
>>> model.unet = LoKrModel(model.unet, config_unet, "default")
```

**属性**：
- **模型** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([LoKrConfig](/docs/peft/v0.20.0/en/package_reference/lokr#peft.LoKrConfig))：LoKr 模型的配置。

### DeLoRA：解耦低阶适应
https://huggingface.co/docs/peft/v0.20.0/package_reference/delora.md