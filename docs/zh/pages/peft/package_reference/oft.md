<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 奥夫特

    

通过正交微调控制文本到图像的扩散

[Orthogonal Finetuning (OFT)](https://hf.co/papers/2306.07280)和[OFTv2](https://huggingface.co/papers/2506.19847)是一种为适应文本到图像扩散模型而开发的方法。它的工作原理是使用正交矩阵重新参数化预训练权重矩阵，以保留预训练模型中的信息。为了减少参数数量，OFT在正交矩阵中引入了块对角结构。该方法主要侧重于在微调模型中保留预训练模型的生成性能。它试图在一层中的所有成对神经元之间保持相同的余弦相似度（[hyperspherical energy](https://huggingface.co/papers/1805.09298)），因为这样可以更好地捕获神经元之间的语义信息。这意味着OFT更能保留主体，并且更适合可控生成（类似于[ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet)）。

论文摘要是：*大型文本到图像扩散模型在根据文本提示生成逼真图像方面具有令人印象深刻的功能。如何有效地引导或控制这些强大的模型来执行不同的下游任务成为一个重要的开放问题。为了应对这一挑战，我们引入了一种原则性的微调方法——正交微调（OFT），用于使文本到图像的扩散模型适应下游任务。与现有方法不同，OFT 可以证明保留超球面能量，该能量表征单位超球面上的成对神经元关系。我们发现这个属性对于保持文本到图像扩散模型的语义生成能力至关重要。为了提高微调稳定性，我们进一步提出约束正交微调（COFT），它对超球面施加额外的半径约束。具体来说，我们考虑两个重要的文本到图像的微调任务：主题驱动生成，其目标是在给定主题的一些图像和文本提示的情况下生成主题特定的图像；可控生成，其目标是使模型能够接收额外的控制信号。我们凭经验表明我们的 OFT 框架rk 在生成质量和收敛速度方面优于现有方法*。

OFT 通过学习神经元的正交变换来保留超球面能量，以保持神经元之间的余弦相似性不变，从而可能减少对先前学到的知识的遗忘。实际上，这意味着采用正交矩阵与预训练权重矩阵的矩阵乘积。然而，为了提高参数效率，正交矩阵被表示为具有秩 `r` 块的块对角矩阵。 LoRA 通过低秩结构减少了可训练参数的数量，而 OFT 通过稀疏块对角矩阵结构减少了可训练参数的数量。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=OFT"
	frameborder="0"
	width="850"
	height="1000"
>

## 将 OFT 权重合并到基础模型中

与LoRA类似，OFT学习到的权重可以使用[⟦T6⟧ function. This function merges the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.

## OFT Example Usage

For using OFT for quantized finetuning with [TRL](https://github.com/huggingface/trl)、`SFT`、`PPO`或`DPO`微调整合到预训练权重矩阵中，遵循以下概述：

```py
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer
from peft import OFTConfig

if use_quantization:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained("model_name")

# Configure OFT
peft_config = OFTConfig(
    oft_block_size=32,
    use_cayley_neumann=True,
    target_modules="all-linear",
    bias="none",
    task_type="CAUSAL_LM"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=ds['train'],
    peft_config=peft_config,
    processing_class=tokenizer,
    args=training_arguments,
    data_collator=collator,
)

trainer.train()
```

# API

## OFTConfig[[peft.OFTConfig]]

#### peft.OFTConfig[[peft.OFTConfig]]

```python
peft.OFTConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 0, oft_block_size: int = 32, module_dropout: float = 0.0, target_modules: Optional[Union[list[str], str]] = None, fan_in_fan_out: bool = False, bias: Literal['none', 'all', 'oft_only'] = 'none', exclude_modules: Optional[Union[list[str], str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None, coft: bool = False, eps: float = 6e-05, block_share: bool = False, use_cayley_neumann: bool = True, num_cayley_neumann_terms: int = 5)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/oft/config.py#L28)

**参数：**r (`int`)：OFT 等级，每个注入层的 OFT 块数。更大的`r`会导致更稀疏的更新矩阵和更少的可训练参数。您只能指定 `r` 或 `oft_block_size`，但不能同时指定两者，因为 `r` × `oft_block_size` = 图层尺寸。为简单起见，我们让您指定 `r` 或 `oft_block_size` 并推断另一个。默认设置为`r = 0`，建议用户设置为`oft_block_size`以获得更好的清晰度。

oft_block_size (`int`)：跨不同层的OFT块大小。更大的`oft_block_size`会导致更密集的更新矩阵和更多可训练的参数。选择 `oft_block_size` 可被图层的输入维度 (`in_features`) 整除，例如 4、8、16。您只能指定 `r` 或 `oft_block_size`，但不能同时指定两者，因为 `r` × `oft_block_size` = 图层维度。为简单起见，我们让您指定 `r` 或 `oft_block_size` 并推断另一个。默认设置为 `oft_block_size = 32`。use_cayley_neumann (bool) ：指定是否使用 Cayley-Neumann 参数化（高效但近似）或普通 Cayley 参数化（精确但由于矩阵逆而计算成本较高）。我们建议将其设置为`True`以获得更好的效率，但由于近似误差，性能可能会稍差。请根据您的需要测试这两个设置（`True` 和 `False`）。默认为`False`。

module_dropout (`float`) ：乘法 dropout 概率，通过在训练期间将 OFT 块设置为恒等，类似于 LoRA 中的 dropout 层。target_modules (`Optional[Union[list[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果将其指定为“全线性”，则选择所有线性模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。

bias (`str`) ：OFT 的偏置类型。可以是“无”、“全部”或“oft_only”。如果是“all”或“oft_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

init_weights (`bool`) ：是否执行OFT权重的初始化。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

coft (`bool`) ：是否使用 OFT 的约束变体，默认关闭。

eps (`float`) : COFT 的控制强度。旋转自由度。仅当 `coft` 设置为 True 时才有效。block_share (`bool`) : 是否在块之间共享OFT参数。默认为`False`。

这是存储[OFTModel](/docs/peft/v0.20.0/en/package_reference/oft#peft.OFTModel)配置的配置类。

#### check_kwargs[[peft.OFTConfig.check_kwargs]]

```python
check_kwargs(**kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/oft/config.py#L198)

**参数：**

kwargs（附加关键字参数，*可选*）：传递给子类初始化的附加关键字参数。

检查 kwargs 对于配置是否有效。

## OFTModel[[peft.OFTModel]]

#### peft.OFTModel[[peft.OFTModel]]

```python
peft.OFTModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/oft/model.py#L35)

**参数：**

model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。

config ([OFTConfig](/docs/peft/v0.20.0/en/package_reference/oft#peft.OFTConfig)) ：OFT模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

OFT 模型。

从预训练模型创建正交微调模型。该方法描述于
https://huggingface.co/papers/2306.07280

示例：
```py
>>> from diffusers import StableDiffusionPipeline
>>> from peft import OFTModel, OFTConfig

>>> config_te = OFTConfig(
...     r=8,
...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
...     module_dropout=0.0,
...     init_weights=True,
... )
>>> config_unet = OFTConfig(
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
...     module_dropout=0.0,
...     init_weights=True,
... )

>>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> model.text_encoder = OFTModel(model.text_encoder, config_te, "default")
>>> model.unet = OFTModel(model.unet, config_unet, "default")
```

**属性**：
- **model** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([OFTConfig](/docs/peft/v0.20.0/en/package_reference/oft#peft.OFTConfig))：OFT 模型的配置。### LayerNorm 调整
https://huggingface.co/docs/peft/v0.20.0/package_reference/layernorm_tuning.md