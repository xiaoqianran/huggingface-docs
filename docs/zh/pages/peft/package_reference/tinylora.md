<!-- huggingface-docs: machine-translated zh-CN from English source -->

# TinyLoRA：学习 13 个参数的推理

[TinyLoRA](https://huggingface.co/papers/2602.04118) 是一种参数效率极高的微调技术，它建立在 [LoRA-XS](https://huggingface.co/papers/2405.17604) 方法的基础上，通过使用冻结权重的 SVD 分解并通过固定随机张量投影微小的可训练向量。当与 GRPO 等强化学习 (RL) 训练方法结合使用时，TinyLoRA 可以通过少至 1-13 个可训练参数实现有竞争力的性能。

TinyLoRA 的关键创新在于用固定随机投影矩阵的加权和替换可训练的低秩矩阵 R：`R = Σᵢ vᵢ Pᵢ`，其中`v ∈ R^u` 是尺寸为 `u` 的微小可训练向量，`Pᵢ` 是固定随机矩阵。这极大地减少了可训练参数的数量，同时保持了有竞争力的性能。

TinyLoRA 支持通过 `weight_tying` 参数进行权重绑定，这是一个介于 0.0 和 1.0 之间的比率，用于控制有多少模块共享相同的可训练向量 `v`。设置`weight_tying=0.0`（默认值）意味着不共享，而`weight_tying=1.0`意味着在所有目标模块之间完全共享——整个模型只需一个`u`可训练参数向量即可实现极高的参数效率。保存适配器参数时，可以通过在 `TinyLoraConfig` 上设置 `save_projection=False` 来避免存储随机投影矩阵。在这种情况下，这些矩阵将根据 `projection_seed` 参数中的固定随机种子进行恢复。这减少了检查点的大小，但我们不能保证所有设备和所有未来版本的 PyTorch 的可重复性。如果要确保再现性，请设置`save_projection=True`（这是默认值）。

TinyLoRA 目前有以下限制：

- 仅支持 `nn.Linear`、`nn.Embedding` 和 `transformers.pytorch_utils.Conv1D` 图层。

论文摘要是：> 最近的研究表明，语言模型通常可以通过强化学习来学习推理。有些工作甚至训练低秩参数化进行推理，但传统的 LoRA 无法扩展到模型维度以下。我们质疑即使rank=1 LoRA对于学习推理是否是必要的，并提出了TinyLoRA，一种将低秩适配器缩放到小至一个参数的大小的方法。在我们的新参数化中，我们能够在 GSM8K 上将 Qwen2.5 的 8B 参数大小训练到 91% 的准确度，而 bf16 中仅需要 13 个经过训练的参数（总共 26 个字节）。我们发现这种趋势总体上是成立的：我们能够恢复 90% 的性能改进，同时在一系列更困难的推理学习基准（例如 AIME、AMC 和 MATH500）中训练的参数减少了 1000 倍。值得注意的是，我们只能通过 RL 实现如此强大的性能：使用 SFT 训练的模型需要 100-1000 倍的更新才能达到相同的性能。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=TINYLORA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## TinyLoraConfig[[peft.TinyLoraConfig]]

#### peft.TinyLoraConfig[[peft.TinyLoraConfig]]

```python
peft.TinyLoraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 2, u: int = 64, weight_tying: float = 0.0, projection_seed: int = 42, save_projection: bool = True, init_v_bound: float = 0.02, target_modules: Optional[Union[list[str], str]] = None, tinylora_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: Optional[list[str]] = None, init_weights: Union[bool, str] = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tinylora/config.py#L25)

**参数：**r（`int`，*可选*，默认为`2`）：冻结 U、Sigma、V 分解的 SVD 等级。论文推荐r=2。

u (`int`，*可选*，默认为`64`)：每组可训练向量维度。这控制了适应的表现力。可以低至 1-13，以实现极高的参数效率。

weight_tying（`float`，*可选*，默认为`0.0`）：目标模块之间的权重绑定程度，以 0.0 和 1.0 之间的比率表示。控制有多少个模块共享相同的可训练向量 v。0.0 表示不共享（每个模块都有自己的 v）。 1.0表示完全共享（所有模块共享一个v）。介于两者之间的值提供部分共享。

projection_seed (`int`，*可选*，默认为`42`)：用于生成固定投影矩阵 P 的随机种子。

save_projection (`bool`, *可选*, 默认为`True`) : 是否将投影张量 P 保存在状态字典中。如果为 False，则加载时它们将从种子中重新生成。

init_v_bound (`float`，*可选*，默认为`0.02`)：可训练向量 v 的统一初始化边界。值在 [-init_v_bound, init_v_bound] 中初始化。target_modules (`Union[List[str], str]`, *可选*) ：要应用 TinyLoRA 的模块的名称。这可以是模块名称列表（例如`['q_proj', 'v_proj']`）、正则表达式模式（例如`'.*decoder.*(q|v)_proj$'`）或用于定位所有线性模块的特殊关键字`"all-linear"`。仅支持 `nn.Linear`、`nn.Embedding` 和 `transformers.pytorch_utils.Conv1D` 层。

tinylora_dropout（`float`，*可选*，默认为`0.0`）：TinyLoRA 层的 dropout 概率。

fan_in_fan_out（`bool`，*可选*，默认为`False`）：如果要替换的图层存储像（fan_in，fan_out）这样的权重，则将此设置为True。例如，gpt-2 使用 `Conv1D` 来存储像 (fan_in, fan_out) 这样的权重。

bias (`str`，*可选*，默认为`"none"`)：TinyLoRA 的偏置类型。可以是“无”、“全部”或“tinylora_only”。

module_to_save (`List[str]`, *可选*) ：除了 TinyLoRA 层之外要设置为可训练和保存的模块列表。

init_weights (`bool` | `Literal["uniform"]`，*可选*，默认为`True`)：如何初始化可训练向量v。传递`True`（默认）将v初始化为零，使适配器成为无操作（身份操作）。传递 `"uniform"` 使用 `[-init_v_bound, init_v_bound]` 中的统一随机值初始化 v。传递 `False` 会使 v 未初始化（对于高级用例）。Layers_to_transform (`Union[List[int], int]`, *可选*) ：要变换的图层索引。如果指定，则仅调整这些层。

Layers_pattern (`Optional[Union[List[str], str]]`, *可选*) ：图层图案名称，仅在`layers_to_transform`与`None`不同时使用。

这是存储[TinyLoraModel](/docs/peft/v0.20.0/en/package_reference/tinylora#peft.TinyLoraModel)配置的配置类。

TinyLoRA 是一种参数极其高效的微调方法，基于论文《Learning to Reason in 13》
参数”（arXiv：2602.04118）。它使用冻结权重的 SVD 分解并投影一个微小的可训练向量
通过固定的随机张量。

论文：https://arxiv.org/abs/2602.04118

示例：
```python
from peft import get_peft_model, TinyLoraConfig

config = TinyLoraConfig(
    r=2,  # SVD rank (paper recommends 2)
    u=64,  # Trainable vector dimension
    weight_tying=0.0,  # No weight tying (0.0 = none, 1.0 = full)
    target_modules=["q_proj", "v_proj"],
    projection_seed=42,
)
model = get_peft_model(base_model, config)
```

## TinyLoraModel[[peft.TinyLoraModel]]

#### peft.TinyLoraModel[[peft.TinyLoraModel]]

```python
peft.TinyLoraModel(model, config, adapter_name, low_cpu_mem_usage = False, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tinylora/model.py#L33)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([TinyLoraConfig](/docs/peft/v0.20.0/en/package_reference/tinylora#peft.TinyLoraConfig)) ：TinyLoRA 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，*可选*，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

TinyLoRA 模型。

从预训练的 Transformer 模型创建 TinyLoRA 模型。TinyLoRA 是一种参数效率极高的微调方法，它使用冻结权重的 SVD 分解和
通过固定的随机张量投影一个微小的可训练向量。基于论文“13年学习推理”
参数”（arXiv：2602.04118）。

示例：
```python
>>> from transformers import AutoModelForCausalLM
>>> from peft import TinyLoraConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = TinyLoraConfig(r=2, u=64, target_modules=["q_proj", "v_proj"])
>>> model = get_peft_model(base_model, config)
```

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([TinyLoraConfig](/docs/peft/v0.20.0/en/package_reference/tinylora#peft.TinyLoraConfig))：TinyLoRA 模型的配置。

#### 删除适配器[[peft.TinyLoraModel.delete_adapter]]

```python
delete_adapter(adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tinylora/model.py#L284)

删除适配器并清理模型级共享 v 参数。

### 配置
https://huggingface.co/docs/peft/v0.20.0/package_reference/config.md