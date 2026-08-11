<!-- huggingface-docs: machine-translated zh-CN from English source -->

# VB-LoRA：使用向量库进行极端参数高效微调

## 概述

[VB-LoRA](https://huggingface.co/papers/2405.15179)是一种参数高效的微调技术，通过学习子向量级别的细粒度参数共享方案来扩展LoRA，从而实现显着更高的参数效率。这使得 VB-LoRA 在存储和传输成本至关重要的场景中特别有用。它的工作原理是将来自不同层和模块（例如 K、Q、V 和 FFN）的低秩矩阵分解为子向量，然后通过向量库全局共享。

论文摘要是：*随着大型语言模型的采用增加以及对每个用户或每个任务模型定制的需求的增长，参数高效微调（PEFT）方法，例如低秩适应（LoRA）及其变体，会产生大量的存储和传输成本。为了进一步减少存储的参数，我们引入了“分而共享”范例，通过向量库全局共享参数，打破了跨矩阵维度、模块和层的低秩分解的障碍。作为 LoRA 范式的实例，我们提出的 VB-LoRA 将共享向量库中的 LoRA 的所有低秩矩阵与可微分的 top-k 混合模块组合在一起。与最先进的 PEFT 方法相比，VB-LoRA 实现了极高的参数效率，同时保持了可比或更好的性能。大量实验证明了 VB-LoRA 在自然语言理解、自然语言生成和指令调优任务方面的有效性。在微调 Llama2-13B 模型时，VB-LoRA 仅使用 LoRA 存储参数的 0.4%，却取得了优异的结果。*

## 使用提示- VB-LoRA 利用稀疏 top-k 模块来学习共享机制。保存适配器参数时，可以通过在`VBLoRAConfig`中设置`save_only_topk_weights = True`来仅保存前k个权重及其索引，也可以通过将其设置为`False`来保存所有可训练的logits。启用`save_only_topk_weights = True`可大幅减少存储空间；例如，在Llama2-7B中，存储文件大小从308MB减少到2.5MB。请注意，使用`save_only_topk_weights = True`保存的模型仅用于合并或推理，不能用于恢复训练。

- VB-LoRA有两组训练参数：向量库参数和logit参数。在实践中，我们发现logit参数需要较高的学习率，而向量库参数需要较低的学习率。使用 AdamW 优化器时，logits 的典型学习率为 0.01，矢量库参数的典型学习率为 0.001。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=VBLORA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## VBLoRAConfig[[peft.VBLoRAConfig]]

#### pft.VBLoRAConfig[[peft.VBLoRAConfig]]

```python
peft.VBLoRAConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 4, num_vectors: int = 256, vector_length: int = 256, topk: int = 2, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, save_only_topk_weights: bool = False, vblora_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: Optional[list[str]] = None, init_vector_bank_bound: float = 0.02, init_logits_std: float = 0.1, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/vblora/config.py#L25)

**参数：**

r (`int`) ：增量矩阵的秩。

num_vectors (`int`) ：向量库中向量的数量。当模型尺寸增加时，使用更高的值。vector_length (`int`) ：向量库中向量的长度。向量的长度应该可以被模型的隐藏维度整除。

topk (`int`) ：top-K 选择的 K 值。 K 值越大，保存模型的大小越大。实际上，设置 K=2 通常可以提供最佳性能和参数效率。更多详细信息，请参阅论文中的讨论。

target_modules (`Union[List[str], str]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

save_only_topk_weights (`bool`) : 是否只保存topk权重。设置`save_only_topk_weights = True`可显着减少存储空间。但是，在此模式下保存的模型只能用于合并或推理，不能用于恢复训练。

vblora_dropout (`float`)：VBLoRA 层的 dropout 概率。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) ：VBLoRA 的偏置类型。可以是“无”、“全部”或“vblora_only”。如果是“all”或“vblora_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。module_to_save (`List[str]`) ：除了 VBLoRA 层之外的模块列表，要设置为可训练并保存在最终检查点中。

init_vector_bank_bound (`float`) ：向量库以 -init_vector_bank_bound 和 init_vector_bank_bound 之间的均匀分布进行初始化。避免用全零初始化向量库以防止零梯度。较小的值（例如 0.02）通常是有效的。使用较大的值进行初始化可能会导致训练不稳定。

init_logits_std (`float`) ：使用标准差为 init_logits_std 的正态分布初始化 logits。默认值为 0.1。

Layers_to_transform (`Union[List[int],int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

这是存储[VBLoRAModel](/docs/peft/v0.20.0/en/package_reference/vblora#peft.VBLoRAModel)配置的配置类。

论文：https://huggingface.co/papers/2405.15179## VBLoRAModel[[peft.VBLoRAModel]]

#### peft.VBLoRAModel[[peft.VBLoRAModel]]

```python
peft.VBLoRAModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/vblora/model.py#L29)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([VBLoRAConfig](/docs/peft/v0.20.0/en/package_reference/vblora#peft.VBLoRAConfig)) ：VBLoRA 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

VBLoRA 模型。

从预训练的 Transformer 模型创建 VBLoRA 模型。

该方法在 https://huggingface.co/papers/2405.15179 中有详细描述。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import VBLoRAConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = VBLoRAConfig(
...     task_type="SEQ_CLS",
...     r=4,
...     target_modules=["fc1", "fc2", "k_proj", "out_proj", "q_proj", "v_proj"],
...     num_vectors=60,
...     vector_length=256,
...     save_only_topk_weights=True,
... )
>>> model = get_peft_model(base_model, config)
```

**属性**：
- **模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([VBLoRAConfig](/docs/peft/v0.20.0/en/package_reference/vblora#peft.VBLoRAConfig))：VBLoRAConfig 模型的配置。

#### get_nb_savable_parameters[[peft.VBLoRAModel.get_nb_savable_parameters]]

```python
get_nb_savable_parameters(adapter = 'default')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/vblora/model.py#L157)

返回可保存的 VB-LoRA 参数的数量和其他可保存的参数。

#### print_savable_parameters[[peft.VBLoRAModel.print_savable_parameters]]

```python
print_savable_parameters()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/vblora/model.py#L193)

打印可保存的 VB-LoRA 参数数量和可保存参数总数。

### PEFT 集成功能
https://huggingface.co/docs/peft/v0.20.0/package_reference/function.md