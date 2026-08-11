<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 型号

[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 是基础模型类，用于指定要应用 PEFT 方法的基础 Transformer 模型和配置。基础 `PeftModel` 包含从 Hub 加载和保存模型的方法。

## PeftModel[[peft.PeftModel]]

#### peft.PeftModel[[peft.PeftModel]]

```python
peft.PeftModel(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = 'default', autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L107)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：Peft 使用的基本变压器模型。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 模型的配置。

adapter_name (`str`, *可选*) ：适配器的名称，默认为`"default"`。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为 `True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应层的数据类型保持相同。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。有助于加快加载加载过程。  > [!TIP] > 创建新的 PEFT 适配器进行训练时，请勿使用 `low_cpu_mem_usage=True`。

包含各种 Peft 方法的基础模型。**属性**：
- **base_model** (`torch.nn.Module`) -- 用于 Peft 的基本变压器模型。
- **peft_config** ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) -- Peft 模型的配置。
- **modules_to_save** (`list` of `str`) -- 保存时要保存的子模块名称列表
  保存模型。
- **prompt_encoder** ([PromptEncoder](/docs/peft/v0.20.0/en/package_reference/p_tuning#peft.PromptEncoder)) -- 用于 Peft if 的提示编码器
  使用[PromptLearningConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PromptLearningConfig)。
- **prompt_tokens** (`torch.Tensor`) -- 用于 Peft 的虚拟提示标记，如果
  使用[PromptLearningConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PromptLearningConfig)。
- **transformer_backbone_name** (`str`) -- 变压器的名称
  如果使用[PromptLearningConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PromptLearningConfig)，则在基本模型中骨干。
- **word_embeddings** (`torch.nn.Embedding`) -- Transformer 主干的词嵌入
  如果使用[PromptLearningConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PromptLearningConfig)，则在基本模型中。

#### add_adapter[[peft.PeftModel.add_adapter]]

```python
add_adapter(adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False, autocast_adapter_dtype: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1105)

**参数：**

adapter_name (`str`) ：要添加的适配器的名称。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) ：要添加的适配器的配置。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。有助于加快加载已保存适配器的过程。创建新的 PEFT 适配器用于训练时，请勿使用此选项。autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为 `True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应层的数据类型保持相同。

根据传递的配置将适配器添加到模型。

该适配器未经培训。要加载经过训练的适配器，请查看[PeftModel.load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)。

新适配器的名称应该是唯一的。

新适配器不会自动设置为活动适配器。使用[PeftModel.set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)设置激活
适配器。

#### create_or_update_model_card[[peft.PeftModel.create_or_update_model_card]]

```python
create_or_update_model_card(output_dir: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1686)

更新或创建模型卡以包含有关 peft 的信息：
1.添加`peft`库标签
2.增加peft版本
3.添加基础型号信息
4. 添加量化信息（如果使用）

#### 删除适配器[[peft.PeftModel.delete_adapter]]

```python
delete_adapter(adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1183)

**参数：**

adapter_name (str) ：要删除的适配器的名称。

删除现有适配器。

#### 禁用_适配器[[peft.PeftModel.disable_adapter]]

```python
disable_adapter()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1045)禁用适配器模块的上下文管理器。使用它在基本模型上运行推理。

示例：

```py
>>> with model.disable_adapter():
...     model(inputs)
```

#### 转发[[peft.PeftModel.forward]]

```python
forward(*args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1024)

模型的前向传递。

#### from_pretrained[[peft.PeftModel.from_pretrained]]

```python
from_pretrained(model: torch.nn.Module, model_id: Union[str, os.PathLike], adapter_name: str = 'default', is_trainable: bool = False, config: Optional[PeftConfig] = None, autocast_adapter_dtype: bool = True, ephemeral_gpu_offload: bool = False, low_cpu_mem_usage: bool = False, key_mapping: Optional[dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L435)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。对于🤗 Transformers 模型，模型应该使用 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) 进行初始化。

model_id（`str` 或 `os.PathLike`）：要使用的 PEFT 配置的名称。可以是： - 字符串，即 Hugging Face Hub 上模型存储库内托管的 PEFT 配置的 `model id`。 - 包含使用 `save_pretrained` 方法 (`./my_peft_config_directory/`) 保存的 PEFT 配置文件的目录路径。

适配器名称（`str`，*可选*，默认为`"default"`）：要加载的适配器的名称。这对于加载多个适配器非常有用。

is_trainable (`bool`，*可选*，默认为`False`)：适配器是否应该可训练。如果`False`，适配器将被冻结，只能用于推理。config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig), *可选*) ：要使用的配置对象，而不是自动加载的配置。该配置对象与`model_id`和`kwargs`互斥。当在调用 `from_pretrained` 之前已经加载配置时，这非常有用。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

ephemeral_gpu_offload (`bool`, *可选*) ：是否对部分加载的模块使用临时 GPU 卸载。默认为 `False`。当模型和/或组件（例如适配器）的一部分保留在 CPU 内存中直到需要时，这非常有用。数据不是对小数据执行昂贵的操作，而是按需传输到 GPU，执行操作，然后将结果移回 CPU 内存。这会带来轻微的瞬时 VRAM 开销，但在某些情况下会带来数量级的加速。low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在加载保存的权重之前在元设备上创建空适配器权重。对于加快进程很有用。

torch_device（`str`，*可选*，默认为 None）：加载适配器的设备。如果`None`，将推断该设备。

key_mapping (dict, *可选*, 默认为 None) : 在加载 `state_dict` 之前应用 PEFT `state_dict` 键的额外映射。应用此映射时，PEFT 特定的`"base_model.model"`前缀将被预先删除，并且适配器名称（例如`"default"`）尚未插入。仅当您知道自己在做什么时才通过此论证。

kwargs : (`optional`)：传递给特定 PEFT 配置类的附加关键字参数。

从预训练模型和加载的 PEFT 权重实例化 PEFT 模型。

请注意，传递的`model`可以就地修改。

#### get_base_model[[peft.PeftModel.get_base_model]]

```python
get_base_model()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1099)

返回基本模型。

#### get_layer_status[[peft.PeftModel.get_layer_status]]

```python
get_layer_status()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1215)

**返回：**列表`peft.peft_model.TunerLayerStatus`

数据类列表，每个数据类包含相应适配器层的状态。

获取模型中每个适配器层的状态。该方法返回`TunerLayerStatus`数据类实例的列表，每个实例包含以下内容
属性：

- `name` (`str`):
  适配器层的名称，例如`model.encoder.block.0.layer.0.SelfAttention.q`。
- `module_type` (`str`):
  适配器层的类型，例如`lora.Linear`。
- `enabled`（`bool`）：
  适配器层是否启用。
- `active_adapters` (`list[str]`):
  活动适配器的名称（如果有），例如`["default"]`。
- `merged_adapters`（`list[str]`）：
  合并适配器的名称（如果有），例如`["default"]`。
- `available_adapters`（`list[str]`）：
  可用适配器的名称，例如`["default"]`。
- `quantization_backend`（`str` 或 `None`）：
  量化后端的名称，例如`"bnb 4bit"`，或`None`（如果未量化）。

#### get_model_status[[peft.PeftModel.get_model_status]]

```python
get_model_status()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1243)

**返回：** `peft.peft_model.TunerModelStatus`

包含模型状态的数据类。

获取模型调谐器的状态。

该方法返回一个`TunerModelStatus`数据类实例，其中包含以下属性：- `base_model_type`（`str`）：
  基本模型的类型，例如`T5Model`。
- `adapter_model_type` (`str`):
  适配器型号的类型，例如`LoraModel`。
- `peft_types`（`dict[str, str]`）：
  适配器名称到适配器类型的映射，例如`{"default": "LORA"}`。
- `trainable_params`（`int`）：
  模型中可训练参数的数量。
- `total_params`（`int`）：
  模型中参数的总数。
- `num_adapter_layers` (`int`):
  模型中适配器层的数量。
- `enabled`（`bool`、`Literal["irregular"]`）：
  是否启用所有适配器层。如果有些启用而有些未启用，则为`"irregular"`。
  这意味着您的模型处于不一致的状态，并且可能无法按预期工作。
- `active_adapters`（`list[str]`、`Literal["irregular"]`）：
  活动适配器的名称。如果活动适配器在所有层上不一致，这将是
  `"irregular"`，这意味着您的模型处于不一致状态，可能无法按预期工作。
- `merged_adapters`（`list[str]`、`Literal["irregular"]`）：
  合并的适配器的名称。如果合并的适配器在所有层上不一致，这将是
  `"irregular"`，这意味着您的模型处于不一致状态，可能无法按预期工作。
- `available_adapters`（`list[str]`）：
  可用适配器的名称，例如`["default"]`。
- `quantization_backend`（`str`、`None`、`Literal["irregular"]`）：量化后端的名称，例如`"bnb 4bit"`，或`None`（如果未量化）。如果后端没有
  所有层都一致，这将是`"irregular"`。

#### get_nb_trainable_parameters[[peft.PeftModel.get_nb_trainable_parameters]]

```python
get_nb_trainable_parameters()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L955)

返回模型中可训练参数的数量和所有参数的数量。

#### get_prompt[[peft.PeftModel.get_prompt]]

```python
get_prompt(batch_size: int, task_ids: Optional[torch.Tensor] = None, max_cache_len: Optional[int] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L788)

返回用于 Peft 的虚拟提示。仅适用于使用即时学习方法时。

#### get_prompt_embedding_to_save[[peft.PeftModel.get_prompt_embedding_to_save]]

```python
get_prompt_embedding_to_save(adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L767)

返回保存模型时要保存的提示嵌入。仅适用于使用提示学习时
方法。

#### load_adapter[[peft.PeftModel.load_adapter]]

```python
load_adapter(model_id: Union[str, os.PathLike], adapter_name: str, is_trainable: bool = False, torch_device: Optional[str] = None, autocast_adapter_dtype: bool = True, ephemeral_gpu_offload: bool = False, low_cpu_mem_usage: bool = False, key_mapping: Optional[dict[str, str]] = None, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1406)

**参数：**

model_id（`str` 或 `os.PathLike`）：要使用的 PEFT 配置的名称。可以是： - 字符串，即 Hugging Face Hub 上模型存储库内托管的 PEFT 配置的 `model id`。 - 包含使用 `save_pretrained` 方法 (`./my_peft_config_directory/`) 保存的 PEFT 配置文件的目录路径。

adapter_name (`str`) ：要添加的适配器的名称。is_trainable (`bool`，*可选*，默认为`False`)：适配器是否应该可训练。如果`False`，适配器将被冻结，只能用于推理。

torch_device（`str`，*可选*，默认为 None）：加载适配器的设备。如果`None`，将推断该设备。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

ephemeral_gpu_offload (`bool`，*可选*，默认为`False`)：是否对部分加载的模块使用临时 GPU 卸载。默认为`False`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在加载保存的权重之前在元设备上创建空适配器权重。对于加快进程很有用。key_mapping (dict, *可选*, 默认为 None) : 在加载 `state_dict` 之前应用 PEFT `state_dict` 键的额外映射。应用此映射时，PEFT 特定的 `"base_model.model"` 前缀将被预先删除，并且适配器名称（例如 `"default"`）尚未插入。仅当您知道自己在做什么时才通过此论证。

kwargs : (`optional`): 用于修改适配器加载方式的附加参数，例如Hugging Face Hub 的代币。

将经过训练的适配器加载到模型中。

新适配器的名称应该是唯一的。

新适配器不会自动设置为活动适配器。使用[PeftModel.set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)设置激活
适配器。

#### 准备模型_for_gradient_checkpointing[[peft.PeftModel.prepare_model_for_gradient_checkpointing]]

```python
prepare_model_for_gradient_checkpointing(model: PreTrainedModel)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L745)

如有必要，为梯度检查点准备模型

#### print_trainable_parameters[[peft.PeftModel.print_trainable_parameters]]

```python
print_trainable_parameters()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L985)

打印模型中可训练参数的数量。注意： print_trainable_parameters() 使用 get_nb_trainable_parameters() ，这与
来自huggingface/transformers 的num_parameters(only_trainable=True)。 get_nb_trainable_parameters() 返回
Peft 模型（可训练参数，所有参数），其中包括修改后的骨干变压器模型。
对于 LoRA 等技术，主干变压器模型会使用 LoRA 模块进行适当修改。然而，对于
及时调整，主干变压器模型未修改。 num_parameters(only_trainable=True) 返回数字
骨干变压器模型的可训练参数可以不同。

#### save_pretrained[[peft.PeftModel.save_pretrained]]

```python
save_pretrained(save_directory: str, safe_serialization: bool = True, selected_adapters: Optional[list[str]] = None, save_embedding_layers: Union[str, bool] = 'auto', is_main_process: bool = True, path_initial_model_for_weight_conversion: Optional[str] = None, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L225)

**参数：**

save_directory (`str`) ：保存适配器模型和配置文件的目录（如果不存在则创建）。

safe_serialization (`bool`, *可选*) : 是否以safetensors格式保存适配器文件，默认为`True`。

selected_adapters (`List[str]`, *可选*) ：要保存的适配器列表。如果`None`，将默认为所有适配器。save_embedding_layers（`Union[bool, str]`，*可选*，默认为`"auto"`）：如果`True`，除了适配器权重之外还保存嵌入层。如果是`auto`，则检查配置的`target_modules`中的公共嵌入层`peft.utils.other.EMBEDDING_LAYER_NAMES`（如果可用）。并自动设置布尔标志。这仅适用于 🤗 变形金刚模型。

is_main_process (`bool`, *可选*) ：调用此进程的进程是否为主进程。默认为`True`。如果不在主进程上，则不会保存检查点，这对于多设备设置（例如 DDP）很重要。path_initial_model_for_weight_conversion (`str`, *可选*) ：初始化适配器的路径，在使用 PiSSA/CorDA/OLoRA 初始化模型之后和执行任何训练之前获得。当`path_initial_model_for_weight_conversion`不为None时，计算微调前后adapter的差异。这种差异可以表示为标准 LoRA 适配器的参数。与 PiSSA 及其朋友相比，使用此转换后的适配器不需要更改基本模型，因此可以方便地允许将多个 PiSSA/CorDA/OLoRA 适配器与 LoRA 适配器一起使用，以及激活或停用任何适配器。请注意，如果 `rslora` 与 `rank_pattern` 或 `alpha_pattern` 结合使用，则不支持此转换。请参阅[peft.tuners.lora.LoraModel.subtract_mutated_init()](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraModel.subtract_mutated_init)了解更多信息。

kwargs（附加关键字参数，*可选*）：传递给 `push_to_hub` 方法的附加关键字参数。

该函数将适配器型号和适配器配置文件保存到一个目录中，以便于调用
使用[PeftModel.from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained)类方法重新加载，也由`PeftModel.push_to_hub()`使用
方法。

#### set_adapter[[peft.PeftModel.set_adapter]]

```python
set_adapter(adapter_name: str, inference_mode: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1591)

**参数：**adapter_name (`str`) ：要设置为活动的适配器的名称。必须首先加载适配器。

inference_mode（`bool`，可选）：是否应冻结激活的适配器（即`requires_grad=False`）。默认值为 False。

设置活动适配器。

一次只能有一个适配器处于活动状态。

此外，此函数会将指定的适配器设置为可训练（即，requires_grad=True），除非
inference_mode 为 True。

#### set_requires_grad[[peft.PeftModel.set_requires_grad]]

```python
set_requires_grad(adapter_names: str | Sequence[str], requires_grad: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1616)

**参数：**

适配器名称（`str`或`Sequence[str]`）：应启用/禁用其渐变的适配器的名称。

require_grad (`bool`, *可选*) ：是否启用（`True`，默认）或禁用（`False`）。

启用或禁用给定适配器上的梯度。

注意：不支持提示调整等提示学习方法。

####支持_lora_conversion[[peft.PeftModel.supports_lora_conversion]]

```python
supports_lora_conversion(adapter_name: str = 'default')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1751)

该型号的适配器是否可以转为LoRA。

通常，如果 PEFT 方法是可加的，即 W' = W_base + delta_weight，则此方法有效。

## PeftModelForSequenceClassification[[peft.PeftModelForSequenceClassification]]用于序列分类任务的`PeftModel`。

#### peft.PeftModelForSequenceClassification[[peft.PeftModelForSequenceClassification]]

```python
peft.PeftModelForSequenceClassification(model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = 'default', **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L1767)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：基础变压器模型。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 配置。

adapter_name (`str`, *可选*) ：适配器的名称，默认为`"default"`。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

用于序列分类任务的 Pft 模型。

**属性**：
- **config** (`PretrainedConfig`) -- 基础模型的配置对象。
- **cls_layer_name** (`str`) -- 分类层的名称。

示例：

```py
>>> from transformers import AutoModelForSequenceClassification
>>> from peft import PeftModelForSequenceClassification, get_peft_config

>>> config = {
...     "peft_type": "PREFIX_TUNING",
...     "task_type": "SEQ_CLS",
...     "inference_mode": False,
...     "num_virtual_tokens": 20,
...     "token_dim": 768,
...     "num_transformer_submodules": 1,
...     "num_attention_heads": 12,
...     "num_layers": 12,
...     "encoder_hidden_size": 768,
...     "prefix_projection": False,
...     "postprocess_past_key_value_function": None,
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForSequenceClassification(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
```

## PeftModelForTokenClassification[[peft.PeftModelForTokenClassification]]

用于标记分类任务的`PeftModel`。

#### peft.PeftModelForTokenClassification[[peft.PeftModelForTokenClassification]]

```python
peft.PeftModelForTokenClassification(model: torch.nn.Module, peft_config: PeftConfig = None, adapter_name: str = 'default', **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L2625)

**参数：**model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：基础变压器模型。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 配置。

adapter_name (`str`, *可选*) ：适配器的名称，默认为`"default"`。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

用于令牌分类任务的 Pft 模型。

**属性**：
- **config** (`PretrainedConfig`) -- 基础模型的配置对象。
- **cls_layer_name** (`str`) -- 分类层的名称。

示例：

```py
>>> from transformers import AutoModelForSequenceClassification
>>> from peft import PeftModelForTokenClassification, get_peft_config

>>> config = {
...     "peft_type": "PREFIX_TUNING",
...     "task_type": "TOKEN_CLS",
...     "inference_mode": False,
...     "num_virtual_tokens": 20,
...     "token_dim": 768,
...     "num_transformer_submodules": 1,
...     "num_attention_heads": 12,
...     "num_layers": 12,
...     "encoder_hidden_size": 768,
...     "prefix_projection": False,
...     "postprocess_past_key_value_function": None,
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForTokenClassification.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForTokenClassification(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 370178 || all params: 108680450 || trainable%: 0.3406113979101117
```

#### add_adapter[[peft.PeftModelForTokenClassification.add_adapter]]

```python
add_adapter(adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False, autocast_adapter_dtype: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L2695)

**参数：**

adapter_name (`str`) ：要添加的适配器的名称。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) ：要添加的适配器的配置。low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。有助于加快加载已保存适配器的过程。创建新的 PEFT 适配器用于训练时，请勿使用此选项。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

根据传递的配置将适配器添加到模型。

该适配器未经培训。要加载经过训练的适配器，请查看[PeftModel.load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)。

新适配器的名称应该是唯一的。

新适配器不会自动设置为活动适配器。使用[PeftModel.set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)设置激活
适配器。

## PeftModelForCausalLM[[peft.PeftModelForCausalLM]]

用于因果语言建模的`PeftModel`。

#### peft.PeftModelForCausalLM[[peft.PeftModelForCausalLM]]

```python
peft.PeftModelForCausalLM(model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = 'default', **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L2019)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：基础变压器型号。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 配置。adapter_name (`str`, *可选*) ：适配器的名称，默认为`"default"`。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为 `True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

用于因果语言建模的 Peft 模型。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import PeftModelForCausalLM, get_peft_config

>>> config = {
...     "peft_type": "PREFIX_TUNING",
...     "task_type": "CAUSAL_LM",
...     "inference_mode": False,
...     "num_virtual_tokens": 20,
...     "token_dim": 1280,
...     "num_transformer_submodules": 1,
...     "num_attention_heads": 20,
...     "num_layers": 36,
...     "encoder_hidden_size": 1280,
...     "prefix_projection": False,
...     "postprocess_past_key_value_function": None,
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForCausalLM.from_pretrained("gpt2-large")
>>> peft_model = PeftModelForCausalLM(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 1843200 || all params: 775873280 || trainable%: 0.23756456724479544
```

## PeftModelForSeq2SeqLM[[peft.PeftModelForSeq2SeqLM]]

用于序列到序列语言建模的`PeftModel`。

#### peft.PeftModelForSeq2SeqLM[[peft.PeftModelForSeq2SeqLM]]

```python
peft.PeftModelForSeq2SeqLM(model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = 'default', **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L2361)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：基础变压器型号。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 配置。

adapter_name (`str`, *可选*) ：适配器的名称，默认为`"default"`。autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

用于序列到序列语言建模的 Peft 模型。

示例：

```py
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import PeftModelForSeq2SeqLM, get_peft_config

>>> config = {
...     "peft_type": "LORA",
...     "task_type": "SEQ_2_SEQ_LM",
...     "inference_mode": False,
...     "r": 8,
...     "target_modules": ["q", "v"],
...     "lora_alpha": 32,
...     "lora_dropout": 0.1,
...     "fan_in_fan_out": False,
...     "enable_lora": None,
...     "bias": "none",
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> peft_model = PeftModelForSeq2SeqLM(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 884736 || all params: 223843584 || trainable%: 0.3952474242013566
```

## PeftModelForQuestionAnswering[[peft.PeftModelForQuestionAnswering]]

用于回答问题的`PeftModel`。

#### peft.PeftModelForQuestionAnswering[[peft.PeftModelForQuestionAnswering]]

```python
peft.PeftModelForQuestionAnswering(model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = 'default', **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L2857)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：基础变压器型号。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 配置。

adapter_name (`str`, *可选*) : 适配器的名称，默认为 `"default"`。autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

用于提取式问答的 Peft 模型。

**属性**：
- **config** (`PretrainedConfig`) -- 基础模型的配置对象。
- **cls_layer_name** (`str`) -- 分类层的名称。

示例：

```py
>>> from transformers import AutoModelForQuestionAnswering
>>> from peft import PeftModelForQuestionAnswering, get_peft_config

>>> config = {
...     "peft_type": "LORA",
...     "task_type": "QUESTION_ANS",
...     "inference_mode": False,
...     "r": 16,
...     "target_modules": ["query", "value"],
...     "lora_alpha": 32,
...     "lora_dropout": 0.05,
...     "fan_in_fan_out": False,
...     "bias": "none",
... }

>>> peft_config = get_peft_config(config)
>>> model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForQuestionAnswering(model, peft_config)
>>> peft_model.print_trainable_parameters()
trainable params: 592900 || all params: 108312580 || trainable%: 0.5473971721475013
```

#### add_adapter[[peft.PeftModelForQuestionAnswering.add_adapter]]

```python
add_adapter(adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False, autocast_adapter_dtype: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L2925)

**参数：**

adapter_name (`str`) ：要添加的适配器的名称。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) ：要添加的适配器的配置。

low_cpu_mem_usage (`bool`, `optional`, 默认为`False`) : 在元设备上创建空适配器权重。有助于加快加载已保存适配器的过程。创建新的 PEFT 适配器用于训练时，请勿使用此选项。autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

根据传递的配置将适配器添加到模型。

该适配器未经培训。要加载经过训练的适配器，请查看[PeftModel.load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)。

新适配器的名称应该是唯一的。

新适配器不会自动设置为活动适配器。使用[PeftModel.set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)设置活动
适配器。

## PeftModelForFeatureExtraction[[peft.PeftModelForFeatureExtraction]]

用于从变压器模型中提取特征/嵌入的`PeftModel`。

#### peft.PeftModelForFeatureExtraction[[peft.PeftModelForFeatureExtraction]]

```python
peft.PeftModelForFeatureExtraction(model: torch.nn.Module, peft_config: PeftConfig, adapter_name: str = 'default', **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L3110)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：基础变压器模型。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) : Peft 配置。

adapter_name (`str`, *可选*) ：适配器的名称，默认为`"default"`。autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。

用于从 Transformer 模型中提取特征/嵌入的 Peft 模型

**属性**：
- **config** (`PretrainedConfig`) -- 基础模型的配置对象。

示例：

```py
>>> from transformers import AutoModel
>>> from peft import PeftModelForFeatureExtraction, get_peft_config

>>> config = {
...     "peft_type": "LORA",
...     "task_type": "FEATURE_EXTRACTION",
...     "inference_mode": False,
...     "r": 16,
...     "target_modules": ["query", "value"],
...     "lora_alpha": 32,
...     "lora_dropout": 0.05,
...     "fan_in_fan_out": False,
...     "bias": "none",
... }
>>> peft_config = get_peft_config(config)
>>> model = AutoModel.from_pretrained("bert-base-cased")
>>> peft_model = PeftModelForFeatureExtraction(model, peft_config)
>>> peft_model.print_trainable_parameters()
```

## PeftMixedModel[[peft.PeftMixedModel]]

用于混合不同适配器类型（例如 LoRA 和 LoHa）的`PeftModel`。

#### peft.PeftMixedModel[[peft.PeftMixedModel]]

```python
peft.PeftMixedModel(model: nn.Module, peft_config: PeftConfig, adapter_name: str = 'default')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L67)

**参数：**

model (`torch.nn.Module`) ：要调整的模型。

config (`PeftConfig`) ：要调整的模型的配置。适配器类型必须兼容。

适配器名称（`str`，`optional`，默认为`"default"`）：第一个适配器的名称。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

PeftMixedModel 用于加载混合不同类型的适配器以进行推理。该类不支持加载/保存，并且通常不应直接初始化。相反，使用
`get_peft_model` 与参数 `mixed=True`。

> [!TIP] > 阅读 [Mixed adapter types](https://huggingface.co/docs/peft/en/developer_guides/mixed_models) 指南
了解 > 有关使用不同适配器类型的更多信息。

示例：

```py
>>> base_model = ...  # load the base model, e.g. from transformers
>>> peft_model = PeftMixedModel.from_pretrained(base_model, path_to_adapter1, "adapter1").eval()
>>> peft_model.load_adapter(path_to_adapter2, "adapter2")
>>> peft_model.set_adapter(["adapter1", "adapter2"])  # activate both adapters
>>> peft_model(data)  # forward pass using both adapters
```

#### add_adapter[[peft.PeftMixedModel.add_adapter]]

```python
add_adapter(adapter_name: str, peft_config: PeftConfig, low_cpu_mem_usage: bool = False, autocast_adapter_dtype: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L203)

**参数：**

adapter_name (`str`) ：要添加的适配器的名称。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) ：要添加的适配器的配置。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。有助于加快加载已保存适配器的过程。  > [!TIP] > 在创建新的 PEFT 适配器进行训练时不要使用 `low_cpu_mem_usage=True`（训练未经测试 > 并且一般不鼓励用于 PeftMixedModel）。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。如果设置为`False`，则数据类型将与相应图层的数据类型保持相同。根据传递的配置将适配器添加到模型。

该适配器未经培训。要加载经过训练的适配器，请查看[PeftModel.load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)。

新适配器的名称应该是唯一的。

新适配器不会自动设置为活动适配器。使用[PeftModel.set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)设置激活
适配器。

#### 禁用适配器[[peft.PefMixedModel.disable_adapter]]

```python
disable_adapter()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L192)

禁用适配器模块。

#### 转发[[peft.PeftMixedModel.forward]]

```python
forward(*args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L180)

模型的前向传递。

#### from_pretrained[[peft.PefMixedModel.from_pretrained]]

```python
from_pretrained(model: nn.Module, model_id: str | os.PathLike, adapter_name: str = 'default', is_trainable: bool = False, config: Optional[PeftConfig] = None, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L394)

**参数：**

model (`nn.Module`) ：要适配的模型。

model_id（`str` 或 `os.PathLike`）：要使用的 PEFT 配置的名称。可以是： - 字符串，即 Hugging Face Hub 上模型存储库内托管的 PEFT 配置的 `model id`。 - 包含使用 `save_pretrained` 方法 (`./my_peft_config_directory/`) 保存的 PEFT 配置文件的目录路径。

适配器名称（`str`，*可选*，默认为`"default"`）：要加载的适配器的名称。这对于加载多个适配器非常有用。is_trainable (`bool`，*可选*，默认为`False`)：适配器是否应该可训练。如果`False`，适配器将被冻结并用于推理

config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig), *可选*) ：要使用的配置对象，而不是自动加载的配置。该配置对象与`model_id`和`kwargs`互斥。当在调用 `from_pretrained` 之前已经加载配置时，这非常有用。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在加载保存的权重之前在元设备上创建空适配器权重。对于加快进程很有用。

kwargs : (`optional`)：传递给特定 PEFT 配置类的附加关键字参数。

从预训练模型和加载的 PEFT 权重实例化 PEFT 混合模型。

请注意，传递的 `model` 可以就地修改。

#### 生成[[peft.PefMixedModel.generate]]

```python
generate(*args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L186)

生成输出。

#### get_nb_trainable_parameters[[peft.PeftMixedModel.get_nb_trainable_parameters]]

```python
get_nb_trainable_parameters()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L126)

返回模型中可训练参数的数量和所有参数的数量。

#### load_adapter[[peft.PeftMixedModel.load_adapter]]

```python
load_adapter(model_id: str, adapter_name: str, *args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L345)

**参数：**adapter_name (`str`) ：要添加的适配器的名称。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) ：要添加的适配器的配置。

is_trainable (`bool`，*可选*，默认为`False`)：适配器是否可训练。如果`False`，适配器将被冻结，只能用于推理。

torch_device（`str`，*可选*，默认为 None）：加载适配器的设备。如果`None`，将推断该设备。

autocast_adapter_dtype (`bool`，*可选*，默认为`True`)：是否自动转换适配器数据类型。默认为`True`。目前，这只会将使用 float16 和 bfloat16 的适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。

ephemeral_gpu_offload (`bool`，*可选*，默认为`False`)：是否对部分加载的模块使用临时GPU卸载。默认为 `False`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在加载保存的权重之前在元设备上创建空适配器权重。对于加快进程很有用。

kwargs : (`optional`): 用于修改适配器加载方式的附加参数，例如Hugging Face Hub 的代币。将经过训练的适配器加载到模型中。

新适配器的名称应该是唯一的。

新适配器不会自动设置为活动适配器。使用[PeftModel.set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)设置激活
适配器。

#### merge_and_unload[[peft.PefMixedModel.merge_and_unload]]

```python
merge_and_unload(*args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L308)

**参数：**

Progressbar (`bool`) : 是否显示进度条指示卸载和合并过程

safe_merge (`bool`) : 是否激活安全合并检查以检查适配器权重中是否存在潜在的 Nan

adapter_names (`List[str]`, *可选*) ：应合并的适配器名称列表。如果没有，所有活动适配器将被合并。默认为`None`。

此方法将适配器层合并到基础模型中。如果有人想使用基地，这是需要的
模型作为独立模型。

#### print_trainable_parameters[[peft.PeftMixedModel.print_trainable_parameters]]

```python
print_trainable_parameters()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L151)

打印模型中可训练参数的数量。注意： print_trainable_parameters() 使用 get_nb_trainable_parameters() ，这与
来自huggingface/transformers 的num_parameters(only_trainable=True)。 get_nb_trainable_parameters() 返回
Peft 模型（可训练参数，所有参数），其中包括修改后的骨干变压器模型。
对于 LoRA 等技术，主干变压器模型会使用 LoRA 模块进行适当修改。然而，对于
及时调整，主干变压器模型未修改。 num_parameters(only_trainable=True) 返回数字
骨干变压器模型的可训练参数可以不同。

#### set_adapter[[peft.PeftMixedModel.set_adapter]]

```python
set_adapter(adapter_name: Union[str, list[str]], inference_mode: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L266)

**参数：**

adapter_name (str, list[str]) ：要设置为活动的适配器的名称

inference_mode (bool, 可选) ：是否应冻结激活的适配器（即`requires_grad=False`）。默认值为 False。

设置模型的活动适配器。请注意，在前向传递过程中应用适配器的顺序可能与
它们被传递给这个函数。相反，前向传播期间的顺序由
适配器加载到模型中的顺序。活动适配器仅确定哪些适配器是
在前向传递期间有效，但不按照它们的应用顺序起作用。

此外，此函数会将指定的适配器设置为可训练（即，requires_grad=True），除非
inference_mode 为 True。

#### 卸载[[peft.PefMixedModel.unload]]

```python
unload(*args: Any, **kwargs: Any)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mixed_model.py#L325)

通过删除所有适配器模块而不合并来恢复基本模型。这又恢复了原来的基础
模型。

## 实用程序[[peft.cast_mixed_ precision_params]]

#### peft.cast_mixed_ precision_params[[peft.cast_mixed_ precision_params]]

```python
peft.cast_mixed_precision_params(model: torch.nn.Module, dtype: torch.dtype)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py#L1441)

**参数：**

model (`torch.nn.Module`) ：投射不可训练参数的模型。

dtype (`torch.dtype`) ：将不可训练参数转换为的 dtype。 `dtype` 可以是 `torch.float16` 或将模型的所有不可训练参数转换为给定的 `dtype`。 `dtype` 可以是 `torch.float16` 或
`torch.bfloat16` 根据您正在执行的混合精度训练。可训练参数被转换为完整
精度。这是为了通过使用半精度 dtype 来减少使用 PEFT 方法时的 GPU 内存使用量
不可训练的参数。拥有全精度的可训练参数可以在使用时保持训练稳定性
自动混合精度训练。

`torch.bfloat16` 根据您正在执行的混合精度训练。

#### peft.get_peft_model[[peft.get_peft_model]]

```python
peft.get_peft_model(model: PreTrainedModel, peft_config: PeftConfig, adapter_name: str = 'default', mixed: bool = False, autocast_adapter_dtype: bool = True, revision: Optional[str] = None, low_cpu_mem_usage: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mapping_func.py#L30)

**参数：**

model ([transformers.PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要包装的模型。

peft_config ([PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)) ：包含 Peft 模型参数的配置对象。

adapter_name (`str`, `optional`, 默认为`"default"`) ：要注入的适配器名称，如果未提供，则使用默认适配器名称（“default”）。

mix (`bool`, `optional`, 默认为`False`) ：是否允许混合不同（兼容）的适配器类型。autocast_adapter_dtype (`bool`, *可选*) ：是否自动转换适配器数据类型。默认为 `True`。目前，这只会使用 float16 或 bfloat16 将适配器权重投射到 float32，因为这通常是稳定训练所必需的，并且仅影响选定的 PEFT 调谐器。

revision (`str`, `optional`, 默认为 `main`) ：基础模型的修订版本。如果未设置，保存的 peft 模型将加载基础模型的 `main` 修订版

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。如果您打算训练模型，请将此设置保留为 False，除非在训练开始之前适配器权重将被不同的权重替换。

从模型和配置返回 Peft 模型对象，其中模型将就地修改。

#### peft.inject_adapter_in_model[[peft.inject_adapter_in_model]]

```python
peft.inject_adapter_in_model(peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = 'default', low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mapping.py#L47)

**参数：**

peft_config (`PeftConfig`) ：包含 PEFT 模型参数的配置对象。

model (`torch.nn.Module`) ：将注入适配器的输入模型。adapter_name (`str`, `optional`, 默认为`"default"`) ：要注入的适配器名称，如果未提供，则使用默认适配器名称（“default”）。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

state_dict（`dict`，*可选*，默认为`None`）：如果此处传递`state_dict`，则将根据state_dict的条目注入适配器。当 PEFT 方法的确切 `target_modules` 未知时，这可能很有用，例如因为检查点是在没有元数据的情况下创建的。请注意，不使用 `state_dict` 中的值，仅使用键来确定应调整的正确层。

创建 PEFT 层并将其注入模型中。

目前API不支持提示学习方法和适配提示。

该函数与[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)类似，但它不返回[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)实例。相反，它返回
传递模型的原始变异实例。

#### peft.get_peft_model_state_dict[[peft.get_peft_model_state_dict]]

```python
peft.get_peft_model_state_dict(model, state_dict = None, adapter_name: str = 'default', unwrap_compiled: bool = False, save_embedding_layers: bool | Literal['auto'] = 'auto')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/save_and_load.py#L94)

**参数：**型号 ([PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel))：Peft 模型。当使用 torch.nn.DistributedDataParallel、DeepSpeed 或 FSDP 时，模型应该是底层模型/展开模型（即 model.module）。

state_dict (`dict`，*可选*，默认为`None`)：模型的状态字典。如果未提供，将使用传递模型的状态字典。

适配器名称（`str`，*可选*，默认为`"default"`）：应返回其状态字典的适配器的名称。

unwrap_compiled (`bool`，*可选*，默认为`False`)：如果使用torch.compile，是否解开模型。

save_embedding_layers（`Union[bool, str]`，，*可选*，默认为`auto`）：如果`True`，除了适配器权重之外还保存嵌入层。如果是`auto`​​，则检查配置的`target_modules`中的公共嵌入层`peft.utils.other.EMBEDDING_LAYER_NAMES`（如果可用）。基于它设置布尔标志。这仅适用于 🤗 变形金刚模型。

获取 PEFT 模型给定适配器的状态字典。

这仅包括 PEFT 参数，不包括基础模型的参数。因此返回的`state_dict`是
与完整模型尺寸相比通常较小。要检索完整的`state_dict`，只需调用`model.state_dict()`。请注意，适配器名称已从 `state_dict` 中删除，因为这只是一个可以更改的任意名称
加载适配器时。所以例如如果适配器名称是`'default'`并且原始密钥是
`'model.q_proj.lora_A.default.weight'`，返回的密钥将为`'model.q_proj.lora_A.weight'`。使用此功能
与 [set_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.set_peft_model_state_dict) 结合使用，在加载重量时处理适配器名称。

#### peft.prepare_model_for_kbit_training[[peft.prepare_model_for_kbit_training]]

```python
peft.prepare_model_for_kbit_training(model, use_gradient_checkpointing = True, gradient_checkpointing_kwargs = None, auto_clear_cache = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/other.py#L151)

**参数：**

model (`transformers.PreTrainedModel`) : 从`transformers`加载的模型

use_gradient_checkpointing (`bool`，*可选*，默认为`True`)：如果为 True，则使用梯度检查点来节省内存，但代价是向后传递速度较慢。

gradient_checkpointing_kwargs (`dict`，*可选*，默认为`None`)：传递给梯度检查点函数的关键字参数，请参阅`torch.utils.checkpoint.checkpoint`的文档以获取有关可以传递给该方法的参数的更多详细信息。请注意，这仅适用于最新的 Transformer 版本 (> 4.34.1)。auto_clear_cache (`bool`, *可选*, 默认为`True`) : 将参数向上转换为 fp32 后是否清空加速器缓存。这会释放缓存分配器持有的内存，这对于共享主机和加速器内存的设备特别有用。设置为`False`可跳过此步骤。

请注意，此方法仅适用于`transformers`型号。

该方法包装了在运行训练之前准备模型的整个协议。这包括：
1- 在 fp32 中投射层范数 2- 使输出嵌入层需要梯度 3- 添加 lm 的向上投射
前往 fp32 4- 冻结基础模型层以确保它们在训练期间不会更新

#### peft.get_layer_status[[peft.get_layer_status]]

```python
peft.get_layer_status(model: torch.nn.Module)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L3225)

**参数：**

model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]) ：从中获取适配器层状态的模型。

**返回：**列表`peft.peft_model.TunerLayerStatus`

数据类列表，每个数据类包含相应适配器层的状态。

获取模型中每个适配器层的状态。

该函数返回`TunerLayerStatus`数据类实例的列表，每个实例包含以下内容
属性：- `name`（`str`）：
  适配器层的名称，例如`model.encoder.block.0.layer.0.SelfAttention.q`。
- `module_type`（`str`）：
  适配器层的类型，例如`lora.Linear`。
- `enabled`（`bool`）：
  适配器层是否启用。
- `active_adapters`（`list[str]`）：
  活动适配器的名称（如果有），例如`["default"]`。
- `merged_adapters`（`list[str]`）：
  合并适配器的名称（如果有），例如`["default"]`。
- 需要_grad：字典[str，bool |字面意思[“不规则”]]
  每个适配器模块的参数的requires_grad状态。理想情况下，它应该是 `True` 或
  `False`。如果所有参数的requires_grad状态不一致，则该值将设置为
  `"irregular"`。
- `available_adapters`（`list[str]`）：
  可用适配器的名称，例如`["default"]`。
- `devices`（`dict[str, list[str]]`）：
  存储给定适配器参数的设备，例如`["cuda"]`。
- `quantization_backend`（`str`或`None`）：
  量化后端的名称，例如`"bnb 4bit"`，或`None`（如果未量化）。

#### peft.get_model_status[[peft.get_model_status]]

```python
peft.get_model_status(model: torch.nn.Module)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/peft_model.py#L3359)

**参数：**

model ([Union[`~PeftModel`, `~transformers.PreTrainedModel`, `nn.Module`]]) ：从中获取适配器层状态的模型。

**退货：** `peft.peft_model.TunerModelStatus`

包含模型状态的数据类。

获取模型调谐器的状态。该函数返回一个`TunerModelStatus`数据类实例，其中包含以下属性：

- `base_model_type`（`str`）：
  基本模型的类型，例如`T5Model`。
- `adapter_model_type`（`str`）：
  适配器型号的类型，例如`LoraModel`。
- `peft_types`（`dict[str, str]`）：
  适配器名称到适配器类型的映射，例如`{"default": "LORA"}`。
- `trainable_params`（`int`）：
  模型中可训练参数的数量。
- `total_params`（`int`）：
  模型中参数的总数。
- `num_adapter_layers`（`int`）：
  模型中适配器层的数量。
- `enabled`（`bool`、`Literal["irregular"]`）：
  是否启用所有适配器层。如果有些启用而有些未启用，则这将是 `"irregular"`。这个
  意味着您的模型处于不一致状态，可能无法按预期工作。
- `active_adapters`（`list[str]`、`Literal["irregular"]`）：
  活动适配器的名称。如果活动适配器在所有层上不一致，这将是
  `"irregular"`，这意味着您的模型处于不一致状态，可能无法按预期工作。
- `merged_adapters`（`list[str]`、`Literal["irregular"]`）：
  合并的适配器的名称。如果合并的适配器在所有层上不一致，这将是
  `"irregular"`，这意味着您的模型处于不一致状态，可能无法按预期工作。- `requires_grad`（`dict[str, bool | Literal["irregular"]]`）：
  无论对于给定的适配器，所有适配器层都将 `requires_grad` 设置为 `True` 或 `False`。如果有一个
  mix，这将被设置为`"irregular"`，这意味着你的模型处于不一致的状态，并且可能不会
  按预期工作。
- `available_adapters`（`list[str]`）：
  可用适配器的名称，例如`["default"]`。
- `devices`（`dict[str, list[str]]`）：
  存储给定适配器参数的设备，例如`["cuda"]`。
- `quantization_backend`（`str`、`None`、`Literal["irregular"]`）：
  量化后端的名称，例如`"bnb 4bit"`，或`None`（如果未量化）。如果后端没有
  所有层都一致，这将是`"irregular"`。

### FourierFT：离散傅里叶变换微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/fourierft.md