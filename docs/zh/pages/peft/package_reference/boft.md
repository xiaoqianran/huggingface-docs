<!-- huggingface-docs: machine-translated zh-CN from English source -->

#英国广播公司

[Orthogonal Butterfly (BOFT)](https://hf.co/papers/2311.06243)是一种专为微调基础模型而设计的通用方法。它从 Cooley-Tukey 快速傅里叶变换中汲取灵感，提高了微调范式——正交微调（OFT）的参数效率，在微调不同基础模型（包括大型视觉变换器、大型语言模型和文本到图像扩散模型）时显示出良好的结果。

论文摘要是：

*大型基础模型正变得无处不在，但从头开始训练它们的成本却高得令人望而却步。因此，有效地使这些强大的模型适应下游任务变得越来越重要。在本文中，我们研究了一种用于下游任务适应的原则性微调范式——正交微调（OFT）。尽管表现出良好的泛化性，但由于正交矩阵的高维性，OFT 仍然使用相当大量的可训练参数。为了解决这个问题，我们首先从信息传输的角度检查 OFT，然后确定一些能够实现更好参数效率的关键需求。受到 Cooley-Tukey 快速傅立叶变换算法如何实现高效信息传输的启发ssion，我们提出了使用蝴蝶结构的有效正交参数化。我们将此参数化应用于 OFT，创建了一种新颖的参数高效微调方法，称为正交蝴蝶 (BOFT)。通过将 OFT 作为特例，BOFT 引入了广义正交微调框架。最后，我们进行了广泛的实证研究，使大型视觉转换器、大型语言模型和文本到图像扩散模型适应视觉和语言*中的各种下游任务。

BOFT 专注于保留预训练模型的生成能力，同时参数效率显着高于标准[OFT](./oft)。与 OFT 一样，BOFT 通过对预训练权重矩阵应用正交变换来保持层中所有成对神经元之间相同的余弦相似性 ([hyperspherical energy](https://huggingface.co/papers/1805.09298))，确保保留神经元之间的语义关系。BOFT 没有使用块对角正交矩阵，而是将正交变换分解为 **稀疏蝴蝶矩阵** 的乘积（最初在[Cooley–Tukey FFT](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm)中引入）。与 OFT 的块对角旋转（仅混合每个块内的输入）不同，蝶形结构保证每个输入都可以影响每个输出，从而仅使用 `O(d log d)` 参数产生**密集连接**。与 OFT 相比，这种因式分解保留了表达能力，同时大大减少了参数数量（以计算时间为代价）。

在实践中，BOFT 将每个预训练的权重矩阵乘以一系列蝴蝶结构的正交因子，从而实现高效且富有表现力的神经元旋转。这使得 BOFT 非常适合可控生成和任务，其中维护预训练模型的主题表示至关重要，同时还可以扩展到具有较低内存和计算开销的更大模型。BOFT 可以应用于神经网络中权重矩阵的任何子集，以减少可训练参数的数量。给定注入 BOFT 参数的目标层，可训练参数的数量可以根据权重矩阵的大小确定。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=BOFT"
	frameborder="0"
	width="850"
	height="1000"
>

## 将 BOFT 权重合并到基础模型中

与LoRA类似，BOFT学习到的权重可以使用[⟦T5⟧ function. This function merges the adapter weights with the base model which allows you to effectively use the newly merged model as a standalone model.

    

This works because during training, the orthogonal weight matrix (R in the diagram above) and the pretrained weight matrices are separate. But once training is complete, these weights can actually be merged (multiplied) into a new weight matrix that is equivalent.

## BOFT Example Usage

For an example of the BOFT method application to various downstream tasks, please refer to the following guides:

Take a look at the following step-by-step guides on how to finetune a model with BOFT:
- [Dreambooth finetuning with BOFT](https://github.com/huggingface/peft/blob/main/examples/boft_dreambooth/boft_dreambooth.md)整合到预训练的权重矩阵中
- [Controllable generation finetuning with BOFT (ControlNet)](https://github.com/huggingface/peft/blob/main/examples/boft_controlnet/boft_controlnet.md)

对于图像分类任务，可以按如下方式初始化 DinoV2 模型的 BOFT 配置：

```py
import transformers
from transformers import AutoModelForSeq2SeqLM, BOFTConfig
from peft import BOFTConfig, get_peft_model

config = BOFTConfig(
    boft_block_size=4,
    boft_n_butterfly_factor=2,
    target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
    boft_dropout=0.1,
    bias="boft_only",
    modules_to_save=["classifier"],
)

model = transformers.Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-large",
    num_labels=100,
)

boft_model = get_peft_model(model, config)
```

# API

## BOFTConfig[[peft.BOFTConfig]]

#### pft.BOFTConfig[[peft.BOFTConfig]]

```python
peft.BOFTConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, boft_block_size: int = 4, boft_block_num: int = 0, boft_n_butterfly_factor: int = 1, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, boft_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/boft/config.py#L28)

**参数：**

boft_block_size (`int`) ：跨不同层的 BOFT 矩阵块大小，以`int`表示。更大的块大小会导致更密集的更新矩阵和更多可训练的参数。选择 `boft_block_size` 可以被大多数图层的输入维度 (`in_features`) 整除，例如 4, 8, 16。此外，请仅指定 `boft_block_size` 或 `boft_block_num`，但不能同时指定两者或将两者保留为 0，因为 `boft_block_size` x `boft_block_num` 必须等于图层的输入维度。boft_block_num (`int`) ：每个注入层的 BOFT 块数。更大的 `boft_block_num` 会导致更新矩阵更稀疏，可训练参数更少。 **注意**，请选择 `boft_block_num` 能被大多数图层的输入维度 (`in_features`) 整除，例如 4, 8, 16。仅指定 `boft_block_size` 或 `boft_block_num`，但不能同时指定两者或将两者保留为 0，因为 `boft_block_size` x `boft_block_num` 必须等于层的输入维度。

boft_n_butterfly_factor (`int`)：不同层的蝴蝶因子数量。对于`boft_n_butterfly_factor=1`，BOFT 与普通 OFT 相同，对于`boft_n_butterfly_factor=2`，OFT 的有效块大小变为两倍，块数变为一半。

target_modules (`Union[List[str],str]`) ：要应用适配器的模块的名称。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

boft_dropout (`float`) ：乘法 dropout 概率，通过在训练期间将 OFT 块设置为恒等，类似于 LoRA 中的 dropout 层。fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) ：BOFT 的偏置类型。可以是“无”、“全部”或“boft_only”。如果是“all”或“boft_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

module_to_save (`List[str]`) -- 除 BOFT 层之外要设置为可训练并保存在最终检查点中的模块列表。

Layers_to_transform (`Union[List[int],int]`) ：要变换的图层索引，如果指定此参数，它将在此列表中指定的图层索引上应用 BOFT 变换。如果传递单个整数，它将在该索引处的图层上应用 BOFT 变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层模式名称，仅当`layers_to_transform`与`None`不同且图层模式不在公共图层模式中时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。这是存储[BOFTModel](/docs/peft/v0.20.0/en/package_reference/boft#peft.BOFTModel)配置的配置类。

## BOFTModel[[peft.BOFTModel]]

#### peft.BOFTModel[[peft.BOFTModel]]

```python
peft.BOFTModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/boft/model.py#L47)

**参数：**

model ([transformers.PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([BOFTConfig](/docs/peft/v0.20.0/en/package_reference/boft#peft.BOFTConfig)) ：BOFT模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**退货：** `torch.nn.Module`

BOFT 模型。

从预训练的 Transformer 模型创建 BOFT 和 OFT 模型。论文：https://huggingface.co/papers/2311.06243
https://huggingface.co/papers/2306.07280

示例：
```py
>>> import transformers
>>> from peft import BOFTConfig, get_peft_model

>>> config = BOFTConfig(
...     boft_block_size=8,
...     boft_n_butterfly_factor=1,
...     target_modules=["query", "value", "key", "output.dense", "mlp.fc1", "mlp.fc2"],
...     boft_dropout=0.1,
...     bias="boft_only",
...     modules_to_save=["classifier"],
... )

>>> model = transformers.Dinov2ForImageClassification.from_pretrained(
...     "facebook/dinov2-large",
...     num_labels=100,
... )
>>> boft_model = get_peft_model(model, config)
```

**属性**：
- **model** ([transformers.PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([BOFTConfig](/docs/peft/v0.20.0/en/package_reference/boft#peft.BOFTConfig))：BOFT模型的配置。

### 乐克尔
https://huggingface.co/docs/peft/v0.20.0/package_reference/lokr.md