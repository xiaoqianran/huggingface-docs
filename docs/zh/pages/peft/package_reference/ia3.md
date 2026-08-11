<!-- huggingface-docs: machine-translated zh-CN from English source -->

#IA3

> [!提示]
> 当您需要尽可能最小的适配器大小时，IA3 是一个绝佳的选择 - 它使用学习向量而不是低秩矩阵，使其成为参数效率最高的可用方法之一。如果您的首要任务是最小化磁盘/内存占用和快速训练，那么 IA3 是一个强有力的候选者。

    

IA3 引入了三个向量 lv、lk 和 lff 来缩放值、键和前馈激活（图像源）。

Infused Adapter by Inhibiting and Amplifying Inner Activations，或[IA3](https://hf.co/papers/2205.05638)，是一种添加三个学习向量来重新调整自注意力和编码器-解码器注意力层的键和值以及位置前馈网络的中间激活的方法。

论文摘要是：*少镜头上下文学习 (ICL) 通过提供少量训练示例作为输入的一部分，使预先训练的语言模型能够执行以前未见过的任务，而无需任何基于梯度的训练。 ICL 会产生大量的计算、内存和存储成本，因为它涉及每次进行预测时处理所有训练示例。参数高效微调（PEFT）（例如适配器模块、提示调整、稀疏更新方法等）提供了一种替代范例，其中训练一小组参数以使模型能够执行新任务。在本文中，我们严格比较了少样本 ICL 和 PEFT，并证明后者提供了更好的精度以及显着降低的计算成本。在此过程中，我们引入了一种称为 (IA)^3 的新 PEFT 方法，该方法通过学习向量来缩放激活，从而获得更强的性能，同时仅引入相对少量的新参数。我们还提出了一个基于 T0 模型的简单方案，称为 T-Few，它可以应用于新任务，而无需针对特定任务进行调整或修改。我们通过将 T-Few 应用于 RA 来验证 T-Few 在完全未见过的任务上的有效性FT 基准测试，首次实现超人类性能，绝对优于最先进水平 6%。我们实验中使用的所有代码都是公开可用的*。

为了使微调更加高效，IA3（Infused Adapter by Inhibiting and Amplifying Inner Activations）
使用学习到的向量重新调整内部激活。这些学习到的向量被注入注意力和前馈模块中
在典型的基于变压器的架构中。这些学习到的向量是微调过程中唯一可训练的参数，因此原始向量
重量保持冻结状态。处理学习向量（与 LoRA 等权重矩阵的学习低秩更新相反）
使可训练参数的数量少得多。

与[LoRA](./lora)类似，IA3 具有许多相同的优点：* IA3 通过大幅减少可训练参数的数量，使微调更加高效。 （对于 T0，IA3 模型只有大约 0.01% 的可训练参数，而即使 LoRA 也有 > 0.1%）
* 原始的预训练权重保持冻结，这意味着您可以拥有多个轻量级便携式 IA3 模型，用于在其之上构建的各种下游任务。
* 使用 IA3 微调的模型的性能与完全微调的模型的性能相当。
* IA3 不会添加任何推理延迟，因为适配器权重可以与基本模型合并。

原则上，IA3 可以应用于神经网络中权重矩阵的任何子集，以减少可训练的数量
参数。根据作者的实现，IA3 权重被添加到键、值和前馈层
Transformer 模型的一部分。具体来说，对于 Transformer 模型，IA3 权重添加到键层和值层的输出以及第二前馈层的输入
在每个变压器块中。

给定注入 IA3 参数的目标层，可训练参数的数量
可以根据权重矩阵的大小来确定。

＃＃ 用法> [!提示]
> 对于自回归模型，目标为 `k_proj`、`v_proj` 和 `down_proj` — 这些是作者发现最有效的层。设置`feedforward_modules=["down_proj"]`，以便 IA3 知道哪些模块是前馈层并可以应用正确的缩放。

对于序列分类任务，可以按如下方式初始化 Llama 模型的 IA3 配置：

```py
from peft import IA3Config, TaskType, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M")
peft_config = IA3Config(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["k_proj", "v_proj", "down_proj"],
    feedforward_modules=["down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

### 何时使用 IA3 与 LoRA

当您需要尽可能最小的适配器来进行存储受限的部署时，或者当您希望使用更少的可训练参数进行快速收敛时，IA3 特别有效。当您需要更广泛的 LoRA 变体生态系统（DoRA、QLoRA 等）或想要拥有广泛社区支持且经过最广泛测试的方法时，LoRA 可能更可取。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=IA3"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## IA3Config[[peft.IA3Config]]

#### pft.IA3Config[[peft.IA3Config]]

```python
peft.IA3Config(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, feedforward_modules: Optional[Union[list[str], str]] = None, fan_in_fan_out: bool = False, modules_to_save: Optional[list[str]] = None, init_ia3_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/ia3/config.py#L25)

**参数：**target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。feedforward_modules (`Optional[Union[List[str], str]]`) ：被视为前馈模块的模块的名称，如原始论文中所示。这些模块将 (IA)3 向量乘以输入，而不是输出。 `feedforward_modules` 必须是`target_modules` 中存在的名称或名称子集。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

module_to_save (`Optional[List[str]]`) ：除了 (IA)3 层之外的模块列表，要设置为可训练并保存在最终检查点中。

init_ia3_weights (`bool`) ：是否初始化(IA)³层中的向量，默认为`True`。不鼓励将其设置为`False`。

这是存储[IA3Model](/docs/peft/v0.20.0/en/package_reference/ia3#peft.IA3Model)配置的配置类。

## IA3Model[[peft.IA3Model]]

#### pft.IA3Model[[peft.IA3Model]]

```python
peft.IA3Model(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/ia3/model.py#L36)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([IA3Config](/docs/peft/v0.20.0/en/package_reference/ia3#peft.IA3Config)) ：(IA)^3 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

(IA)^3 模型。

通过抑制和放大预训练的内部激活 ((IA)^3) 模型来创建注入适配器
变压器模型。该方法详细描述见https://huggingface.co/papers/2205.05638

示例：

```py
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import IA3Config, get_peft_model

>>> config = IA3Config(
...     peft_type="IA3",
...     task_type="SEQ_2_SEQ_LM",
...     target_modules=["k", "v", "w0"],
...     feedforward_modules=["w0"],
... )

>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> ia3_model = get_peft_model(model, config)
```

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([IA3Config](/docs/peft/v0.20.0/en/package_reference/ia3#peft.IA3Config))：(IA)^3 模型的配置。

#### add_weighted_adapter[[peft.IA3Model.add_weighted_adapter]]

```python
add_weighted_adapter(adapters: list[str], weights: list[float], adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/ia3/model.py#L273)

**参数：**

适配器 (`list`) ：要合并的适配器名称列表。

权重 (`list`) ：每个适配器的权重列表。

adapter_name (`str`) ：新适配器的名称。

此方法通过将给定的适配器与给定的权重合并来添加新的适配器。