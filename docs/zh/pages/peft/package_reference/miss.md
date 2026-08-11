<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 米SS

[MiSS (Matrix Shard Sharing)](https://arxiv.org/abs/2409.15371)是一种PEFT方法，在模型性能和计算效率之间取得了良好的平衡。它只需要一个可训练矩阵，并引入了与 LoRA 不同的分片共享机制。

论文摘要是：

*参数高效微调（PEFT）方法，特别是低秩适应（LoRA），可以有效减少大型语言模型（LLM）中可训练参数的数量。然而，随着模型规模的不断增长，对计算资源的需求仍然是一个重大挑战。现有的 LoRA 变体通常难以在适应性（模型性能和收敛速度）和效率（计算开销、内存使用和初始化时间）之间取得最佳平衡。本文介绍了 MiSS（矩阵碎片共享），这是一种新颖的 PEFT 方法，它通过简单的碎片共享机制解决了这种权衡问题。 MiSS 利用了这样的见解：通过将权重矩阵分解为多个片段矩阵并利用共享的、可训练的公共片段，可以实现低秩自适应。该方法通过复制这些共享的分区分片来构造低秩更新矩阵。我们还提出了一种硬件高效且广泛适用的 MiSS 实现。对一系列任务进行的大量实验以及对计算性能的系统分析证明了 MiSS 的优越性。结果表明，MiSS 在模型性能指标和计算效率（包括初始化速度和训练吞吐量）方面均显着优于标准 LoRA 及其主要变体。通过有效平衡表达能力和资源利用率，MiSS 为高效适应大型模型提供了引人注目的解决方案。*

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=MISS"
	frameborder="0"
	width="850"
	height="1000"
>

## 何时使用 MiSS

在以下情况下，MiSS 是一个不错的选择：

- 与使用昂贵设置的高级 LoRA 初始化方案（例如 PiSSA、LoRA-GA 或 OLoRA）相比，您需要更快的初始化和更高的训练吞吐量。
- 您想要一个 LoRA 的直接替代方案，且配置更改最少。

如果您需要以牺牲一些效率为代价获得更强的表达力，请考虑 `bat` 初始化变体（见下文）。

## init_weights 模式

MiSS通过`init_weights`参数支持三种初始化模式：- `True`（默认）：标准 MiSS 初始化。大多数用例的最佳起点。
- `"bat"`：启用跨不同分片的非线性更新。产生比标准 MiSS 更好的结果，但使用更多内存并且速度大约慢两倍。当性能优先于效率时使用此选项。
- `"mini"`：沿`out_features`维度使用较小的等级，由`mini_r`控制。这进一步减少了可训练参数。使用此模式时，必须设置`mini_r`且`out_features`必须能被`mini_r`整除。

## 快速开始

```python
import torch
from peft import MissConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

# Standard MiSS
config = MissConfig(
    r=64,
    miss_dropout=0.01,
    task_type="CAUSAL_LM"
)

# BAT variant — better performance, more memory
# config = MissConfig(
#     r=64,
#     init_weights="bat",
#     task_type="CAUSAL_LM"
# )

# Mini variant — fewer trainable parameters
# config = MissConfig(
#     r=64,
#     init_weights="mini",
#     mini_r=8,
#     task_type="CAUSAL_LM"
# )

model = get_peft_model(model, config)
model.print_trainable_parameters()
```

有关包括训练和推理的完整微调示例，请参阅[MiSS fine-tuning example](https://github.com/huggingface/peft/tree/main/examples/miss_finetuning)。

# API

## MissConfig[[peft.MissConfig]]

#### peft.MissConfig[[peft.MissConfig]]

```python
peft.MissConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 64, miss_dropout: float = 0.0, mini_r: int = 1, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, init_weights: bool | Literal['bat', 'mini'] = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[str] = None, bias: str = 'none', modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/miss/config.py#L25)

**参数：**

r (`int`) : MiSS 跨不同层的排名。最好将“r”设置为偶数；否则，默认的初始化方法将不起作用。 MiSS 的秩对应于沿 in_features 维度的低秩分解。

miss_dropout (`float`) ：MiSS 层的 dropout 概率。mini_r (`int`) ：MiSS 的秩对应于沿 out_features 维度的低秩分解。设置`init_weights=mini`时，还需要设置`mini_r`。请确保`out_features`能被`mini_r`整除。

target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果将其指定为“全线性”，则选择所有线性模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。init_weights (bool | Literal["bat", "mini"]) ：不同的初始化对应不同的 MiSS 变体。默认情况下（平衡），将使用 MiSS 中最有效、最通用的方法。 'bat'：在此模式下，您可以跨不同分片启用非线性更新。 'mini'：在此模式下，您可以设置较小的等级以使用较少的可训练参数，但建议保留`out_features % mini_r == 0`。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`str`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。

bias (`str`) ：MiSS 的偏置类型。可以是 `'none'`、`'all'` 或 `'MiSS_only'`。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

这是存储[MissModel](/docs/peft/v0.20.0/en/package_reference/miss#peft.MissModel)配置的配置类。

## MissModel[[peft.MissModel]]

#### peft.MissModel[[peft.MissModel]]

```python
peft.MissModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/miss/model.py#L38)

**参数：**

model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。config ([MissConfig](/docs/peft/v0.20.0/en/package_reference/miss#peft.MissConfig)) ：MiSS模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

MiSS 模型。

从预训练模型创建 Householder 反射适应 (MiSS) 模型。该方法描述于
https://huggingface.co/papers/2409.15371

示例：
```py
>>> from diffusers import StableDiffusionPipeline
>>> from peft import MissModel, MissConfig

>>> config_te = MissConfig(
...     r=8,
...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
...     init_weights=True,
... )
>>> config_unet = MissConfig(
...     r=8,
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
...     init_weights=True,
... )

>>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> model.text_encoder = MissModel(model.text_encoder, config_te, "default")
>>> model.unet = MissModel(model.unet, config_unet, "default")
```

**属性**：
- **model** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([MissConfig](/docs/peft/v0.20.0/en/package_reference/miss#peft.MissConfig))：MiSS 模型的配置。

### LoRA 转换
https://huggingface.co/docs/peft/v0.20.0/package_reference/lora_conversion.md