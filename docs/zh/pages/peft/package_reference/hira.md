<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 希拉

高阶自适应 ([HiRA](https://openreview.net/pdf?id=TwJrTz9cRS)) 是一种 PEFT 方法，通过对原始权重矩阵应用逐元素调制来扩展 LoRA 方法。 HiRA 不是直接添加低秩更新，而是计算：

$$
W' = W_0 + W_0 \odot (B A)
$$

其中 $W_0$ 是基本权重，$A、B$ 是等级为 $r \ll \min( \text{in_features}, \text{out_features})$ 的低等级因子。这种公式允许 HiRA 通过乘法、依赖于输入的调制来调整现有权重，通常可以提高下游任务的微调效率。

HiRA 论文的摘要是：

> *我们提出了 Hadamard 高阶适应 (HiRA)，这是一种参数高效微调 (PEFT) 方法，可增强大型语言模型 (LLM) 的适应性。虽然低秩适应（LoRA）被广泛用于减少资源需求，但其低秩更新可能会限制其对新任务的表达能力。 HiRA 通过使用 Hadamard 产品来保留高阶更新参数来解决此问题，从而提高模型容量。根据经验，HiRA 在多项任务上优于 LoRA 及其变体，广泛的消融研究验证了其有效性。*

## 示例

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model
from peft.tuners.hira import HiraConfig

# Example 1: HiRA on opt-125m for causal language modeling
model_id = "facebook/opt-125m"
base_model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Define HiRA configuration: apply to the MLP dense layers in each transformer block
hira_config = HiraConfig(
    r=32,
    target_modules=["k_proj", "q_proj", "v_proj", "fc1", "fc2"],
    hira_dropout=0.0,
    init_weights=True,
)
peft_model = get_peft_model(base_model, hira_config)

peft_model.print_trainable_parameters()
# trainable params: 4,718,592 || all params: 129,957,888 || trainable%: 3.6309
```

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=HIRA"
	frameborder="0"
	width="850"
	height="1000"
>## 引用：

如果您发现 HiRA 有用，请将 HiRA 引用为：
```
@inproceedings{
huang2025hira,
title={Hi{RA}: Parameter-Efficient Hadamard High-Rank Adaptation for Large Language Models},
author={Qiushi Huang and Tom Ko and Zhan Zhuang and Lilian Tang and Yu Zhang},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=TwJrTz9cRS}
}
```

# API

## HiraConfig[[peft.HiraConfig]]

#### pft.HiraConfig[[peft.HiraConfig]]

```python
peft.HiraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, hira_dropout: float = 0.0, fan_in_fan_out: bool = False, modules_to_save: Optional[list[str]] = None, init_weights: bool | Literal['gaussian'] | None = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, rank_pattern: Optional[dict] = <factory>)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/config.py#L25)

**参数：**

r (`int`) ：HiRA 中低秩分量的秩。尽管 HiRA 通过 Hadamard 融合实现了高秩自适应，但该值定义了底层低秩分解（矩阵 A 和 B）的维度。

target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块（如果模型是 PreTrainedModel，则排除输出层）。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

hira_dropout (`float`) ：HiRA 层的丢弃概率。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。init_weights (`bool` | `Literal["gaussian"]`) ：如何初始化 HiRA 层的权重。传递 True（默认）会导致默认初始化，HiRA B 权重设置为 0。这意味着，如果没有进一步训练，HiRA 适配器将是无操作的。将初始化设置为 False 会导致 HiRA A 和 B 随机初始化，这意味着 HiRA 在训练前不是空操作；此设置用于调试目的。通过 `'gaussian'` 会导致按线性和层的 HiRA 等级缩放的高斯初始化。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

rank_pattern (`dict`) ：从层名称或正则表达式到排名的映射，与`r`指定的默认r不同。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。这是存储`HiraModel`配置的配置类。

## 核心层

### HiraLayer[[peft.tuners.hira.HiraLayer]]

#### peft.tuners.hira.HiraLayer[[peft.tuners.hira.HiraLayer]]

```python
peft.tuners.hira.HiraLayer(base_layer: nn.Module, ephemeral_gpu_offload: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L31)

### 线性适配器[[peft.tuners.hira.Linear]]

#### peft.tuners.hira.Linear[[peft.tuners.hira.Linear]]

```python
peft.tuners.hira.Linear(base_layer, adapter_name: str, config: HiraConfig, r: int = 0, fan_in_fan_out: bool = False, is_target_conv_1d_layer: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L168)

#### get_delta_weight[[peft.tuners.hira.Linear.get_delta_weight]]

```python
get_delta_weight(adapter)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L213)

**参数：**

适配器 (str) ：应计算增量权重的适配器的名称。

计算给定适配器的增量权重。

#### 合并[[peft.tuners.hira.Linear.merge]]

```python
merge(safe_merge: bool = False, adapter_names: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L188)

**参数：**

safe_merge (`bool`, *可选*) ：如果为 True，合并操作将在原始权重的副本中执行，并在合并权重之前检查 NaN。如果您想检查合并操作是否会产生 NaN，这很有用。默认为 `False`。

adapter_names (`list[str]`, *可选*) ：应合并的适配器名称列表。如果没有，所有活动适配器将被合并。默认为 `None`。

将活动适配器权重合并到基本权重中

#### 取消合并[[peft.tuners.hira.Linear.unmerge]]

```python
unmerge()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L207)此方法从基本权重中取消合并所有合并的适配器层。

### 嵌入适配器[[peft.tuners.hira.Embedding]]

#### peft.tuners.hira.Embedding[[peft.tuners.hira.Embedding]]

```python
peft.tuners.hira.Embedding(base_layer: nn.Module, adapter_name: str, config: HiraConfig, r: int = 0, fan_in_fan_out: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L289)

#### 转发[[peft.tuners.hira.Embedding.forward]]

```python
forward(x: torch.Tensor, *args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L403)

HiRA 转发嵌入层。支持每批次混合适配器或单个适配器。

#### get_delta_weight[[peft.tuners.hira.Embedding.get_delta_weight]]

```python
get_delta_weight(adapter)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L358)

**参数：**

适配器 (str) ：应计算增量权重的适配器的名称。

计算给定适配器的增量权重。

#### 合并[[peft.tuners.hira.Embedding.merge]]

```python
merge(safe_merge: bool = False, adapter_names: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L333)

**参数：**

safe_merge (`bool`, *可选*) ：如果为 True，合并操作将在原始权重的副本中执行，并在合并权重之前检查 NaN。如果您想检查合并操作是否会产生 NaN，这很有用。默认为 `False`。

adapter_names (`list[str]`, *可选*) ：应合并的适配器名称列表。如果没有，所有活动适配器将被合并。默认为`None`。

将活动适配器权重合并到基本权重中

#### 取消合并[[peft.tuners.hira.Embedding.unmerge]]

```python
unmerge()
```[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hira/layer.py#L352)

此方法从基本权重中取消合并所有合并的适配器层。

### 卷积适配器

[[autodoc]]tuners.hira.layer.Conv1d [[autodoc]]tuners.hira.layer.Conv2d [[autodoc]]tuners.hira.layer.ConvNd

### 权重分解低阶适应（DoRA）
https://huggingface.co/docs/peft/v0.20.0/package_reference/lora_variant_dora.md