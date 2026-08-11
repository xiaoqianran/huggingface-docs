<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 格拉洛拉

[**Granular Low-Rank Adaptation (GraLoRA)**](https://huggingface.co/papers/2505.20355) 是一种 PEFT 方法，旨在增强低秩适应的**表现力**，同时提高**对异常值**激活的鲁棒性，基于对量化中众所周知的问题的见解。

![GraLoRA Overview](https://github.com/SqueezeBits/GraLoRA/raw/main/figure/gralora_overview.png)

与在整个特征空间中应用单个低秩适配器的标准 LoRA 不同，GraLoRA 引入了结构化且细粒度的自适应方案。它将适应空间划分为由 $𝑘^2$ 较小的独立适配器对组成的网格，每个适配器对输入和输出维度的局部子集负责。因此，每个适配器在两个维度上都比原始 LoRA 适配器小 $k$ 倍的子空间上运行。

这种粒度分解可以实现空间局部化和上下文感知更新，从而有效地提高表示能力，而无需额外的参数或计算成本。通过隔离较小子空间内极端激活的影响，GraLoRA 减轻了梯度失真并在适应过程中保持通道间平衡。

---

论文摘要是：*低秩适应（LoRA）是一种流行的参数高效精细方法
生成模型的调优（PEFT），因其简单性和有效性而受到重视。
尽管最近有所增强，LoRA 仍然存在一个根本性的限制：
当瓶颈变宽时过度拟合。它在排名 32-64 时表现最佳，但
准确性在较高级别上停滞或下降，仍达不到全面微调的水平
（FFT）性能。我们认为根本原因是 LoRA 的结构瓶颈，
它将梯度纠缠引入到不相关的输入通道并扭曲
梯度传播。为了解决这个问题，我们引入了一种新颖的结构，Granular
低秩适应（GraLoRA）将权重矩阵划分为子块，
每个都有自己的低级适配器。计算或存储成本可以忽略不计，
GraLoRA克服了LoRA的局限性，有效提高了代表性
容量，并且更接近 FFT 行为。代码实验
生成、常识推理、数学推理、通用语言
理解和图像生成基准表明 GraLoRA 始终如一优于 LoRA 和其他基线，在以下方面实现了高达 +8.5% 的绝对增益
在 HumanEval+ 上通过@1。这些改进适用于模型大小和等级
设置，使 GraLoRA 成为 PEFT 的可扩展且强大的解决方案。*

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=GRALORA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## GraloraConfig[[peft.GraloraConfig]]

#### peft.GraloraConfig[[peft.GraloraConfig]]

```python
peft.GraloraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, hybrid_r: int = 0, target_modules: typing.Union[str, list[str], NoneType] = None, alpha: int = 64, gralora_dropout: float = 0.0, gralora_k: int = 2, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: typing.Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: typing.Union[list[int], int, NoneType] = None, layers_pattern: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/gralora/config.py#L23)

**参数：**

r (`int`) ：GraLoRA 注意力维度决定 GraLoRA 适配器的等级。 GraLoRA适配器的总参数计数与具有相同等级r的LoRA相同，而表达能力乘以gralora_k。

Hybrid_r (`int`) ：混合 GraLoRA 等级决定使用混合 GraLoRA 方法时分配给普通 LoRA 方法的等级。混合 GraLoRA 是 GraLoRA 和 vanilla LoRA 的组合，当 Hybrid_r > 0 时可用。GraLoRA 适配器的参数计数为 r + Hybrid_r。target_modules (`Union[List[str], str]`) ：要替换为 GraLoRA 的模块名称列表或模块名称的正则表达式。 " 例如，['q', 'v'] 或 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'。 " 这也可以是通配符 'all-linear'，匹配所有线性/Conv1D " "（如果模型是 PreTrainedModel，则排除输出层）。如果未指定，将根据模型架构选择模块，如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。 " 为了避免定位任何模块（因为您想应用 `target_parameters`），请设置 " `target_modules=[]`。

alpha (`int`)：GraLoRA alpha。 GraLoRA alpha 是 GraLoRA 适配器的缩放因子。比例变为 alpha / (r + Hybrid_r)。

gralora_dropout (`float`) ：GraLoRA dropout 是 GraLoRA 适配器的 dropout 概率。它用于防止过度拟合并提高GraLoRA适配器的泛化能力。gralora_k (`int`) ：GraLoRA k 确定 GraLoRA 适配器中的子块数量。等级 r 必须能被 gralora_k 整除，GraLoRA 适配器才有效。无论 gralora_k 为何，总参数计数都会被保留。 GraLoRA适配器的整个rank增加grallora_k，而每个子块的rank减少grallora_k。对于等级 32 或更低，建议 gralora_k=2，对于等级 64 或更高，建议 gralora_k=4。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) : gralora 的偏置类型。可以是“无”、“全部”或“gralora_only”。如果是“all”或“gralora_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。module_to_save (`Optional[list[str]]`) ：除 gralora 层之外的模块列表，要设置为可训练并保存在最终检查点中。例如，在序列分类或令牌分类任务中，最后一层`classifier/score`是随机初始化的，因此需要可训练和保存。

init_weights (`bool`) ：是否使用默认初始化来初始化 GraLoRA 层的权重。除非您确切知道自己在做什么，否则请勿更改此设置。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引，如果指定了此参数，PEFT 将仅变换此列表中指定的图层索引。如果传递单个整数，PEFT 将仅转换该索引处的图层。仅当 target_modules 是 str 列表时才有效。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层模式名称，仅当 `layers_to_transform` 与 None 不同且图层模式不在公共图层模式中时使用。仅当 target_modules 是 str 列表时才有效。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

这是存储[GraloraModel](/docs/peft/v0.20.0/en/package_reference/gralora#peft.GraloraModel)配置的配置类。

## GraloraModel[[peft.GraloraModel]]#### peft.GraloraModel[[peft.GraloraModel]]

```python
peft.GraloraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/gralora/model.py#L28)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([GraloraConfig](/docs/peft/v0.20.0/en/package_reference/gralora#peft.GraloraConfig)) ：Gralora 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**返回：** `torch.nn.Module`

格拉洛拉模型。

从预训练的 Transformer 模型创建基于向量的随机矩阵适应 (Gralora) 模型。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import GraloraConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = GraloraConfig(r=128)
>>> model = get_peft_model(base_model, config)
```

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([GraloraConfig](/docs/peft/v0.20.0/en/package_reference/gralora#peft.GraloraConfig))：Gralora 模型的配置。

### DEFT：文本到图像模型的分解高效微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/deft.md