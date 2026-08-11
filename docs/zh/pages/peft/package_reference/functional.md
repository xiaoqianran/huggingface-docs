<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PEFT 集成函数

对非 PeftModel 模型有用的函数集合，例如变压器或扩散器集成

这里提供的函数可以被认为是 PEFT 的“公共 API”，因此可以安全地被提供 PEFT 集成的包使用。

## 转换适配器权重 dtypes[[peft.tuners.tuners_utils.cast_adapter_dtype]]

#### peft.tuners.tuners_utils.cast_adapter_dtype[[peft.tuners.tuners_utils.cast_adapter_dtype]]

```python
peft.tuners.tuners_utils.cast_adapter_dtype(model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L2243)

**参数：**

adapter_name (`str`) ：适配器名称。

autocast_adapter_dtype (`bool`, *可选*) ：是否自动转换适配器数据类型。默认为`True`。

将适配器权重转换为正确的数据类型的辅助方法。

目前，这仅将 float 数据类型向上转换为 float32。

## 从模型中删除 PEFT 适配器[[peft.tuners.tuners_utils.delete_adapter]]

#### peft.tuners.tuners_utils.delete_adapter[[peft.tuners.tuners_utils.delete_adapter]]

```python
peft.tuners.tuners_utils.delete_adapter(model: nn.Module, adapter_name: str, prefix: str, layer_cls: type[BaseTunerLayer] = <class 'peft.tuners.tuners_utils.BaseTunerLayer'>)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L2204)

**参数：**

model (`nn.Module`) ：应从中删除适配器的模型。

adapter_name (str) ：要删除的适配器的名称。

prefix (str) : PEFT 方法的前缀，例如“lora_”代表 LoRA。layer_cls（type，可选）：适配器层的类。默认为`BaseTunerLayer`。

**返回：** new_adapter (list[str] | None)

删除后剩余适配器的名称，如果没有剩余的活动适配器，则为`None`。用这个
如有必要，设置模型的新活动适配器。

删除现有的 PEFT 适配器。

注意：此功能不会删除模型上的 PEFT 配置（如果有）。也不会完全
如果最后一个 PEFT 适配器被删除，则清除 PEFT 层。为此，如果使用
PEFT 模型实例，或者只是重新加载基础模型。

## 获取 PEFT 适配器的状态字典[[peft.get_peft_model_state_dict]]

#### peft.get_peft_model_state_dict[[peft.get_peft_model_state_dict]]

```python
peft.get_peft_model_state_dict(model, state_dict = None, adapter_name: str = 'default', unwrap_compiled: bool = False, save_embedding_layers: bool | Literal['auto'] = 'auto')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/save_and_load.py#L94)

**参数：**

型号 ([PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel))：Peft 模型。当使用 torch.nn.DistributedDataParallel、DeepSpeed 或 FSDP 时，模型应该是底层模型/展开模型（即 model.module）。

state_dict (`dict`，*可选*，默认为`None`)：模型的状态字典。如果未提供，将使用传递模型的状态字典。适配器名称（`str`，*可选*，默认为`"default"`）：应返回其状态字典的适配器的名称。

unwrap_compiled (`bool`，*可选*，默认为`False`)：如果使用了torch.compile，是否解开模型。

save_embedding_layers（`Union[bool, str]`，，*可选*，默认为`auto`）：如果`True`，除了适配器权重之外还保存嵌入层。如果是`auto`，则检查配置的`target_modules`中的公共嵌入层`peft.utils.other.EMBEDDING_LAYER_NAMES`（如果可用）。基于它设置布尔标志。这仅适用于 🤗 变形金刚模型。

获取 PEFT 模型给定适配器的状态字典。

这仅包括 PEFT 参数，不包括基础模型的参数。因此返回的`state_dict`是
与完整模型尺寸相比通常较小。要检索完整的`state_dict`，只需调用`model.state_dict()`。

请注意，适配器名称已从 `state_dict` 中删除，因为这只是一个可以更改的任意名称
加载适配器时。所以例如如果适配器名称是`'default'`并且原始密钥是
`'model.q_proj.lora_A.default.weight'`，返回的密钥将为`'model.q_proj.lora_A.weight'`。使用此功能
与 [set_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.set_peft_model_state_dict) 结合使用，在加载重量时处理适配器名称。## 根据 PEFT 配置将 PEFT 适配器注入模型[[peft.inject_adapter_in_model]]

#### peft.inject_adapter_in_model[[peft.inject_adapter_in_model]]

```python
peft.inject_adapter_in_model(peft_config: PeftConfig, model: torch.nn.Module, adapter_name: str = 'default', low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/mapping.py#L47)

**参数：**

peft_config (`PeftConfig`) ：包含 PEFT 模型参数的配置对象。

model (`torch.nn.Module`) ：适配器将被注入的输入模型。

adapter_name (`str`, `optional`, 默认为`"default"`) ：要注入的适配器名称，如果未提供，则使用默认适配器名称（“default”）。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

state_dict（`dict`，*可选*，默认为`None`）：如果此处传递`state_dict`，则将根据state_dict的条目注入适配器。当 PEFT 方法的确切 `target_modules` 未知时，这可能很有用，例如因为检查点是在没有元数据的情况下创建的。请注意，不使用 `state_dict` 中的值，仅使用键来确定应调整的正确层。

创建 PEFT 层并将其注入模型中。目前API不支持提示学习方法和适配提示。

该函数与[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)类似，但它不返回[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)实例。相反，它返回
传递模型的原始变异实例。

## 设置模型的活动 PEFT 适配器[[peft.tuners.tuners_utils.set_adapter]]

#### peft.tuners.tuners_utils.set_adapter[[peft.tuners.tuners_utils.set_adapter]]

```python
peft.tuners.tuners_utils.set_adapter(model, adapter_name: str | list[str], inference_mode: bool = False, layer_cls: type[BaseTunerLayer] = <class 'peft.tuners.tuners_utils.BaseTunerLayer'>)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L2168)

**参数：**

model (`nn.Module`) ：应设置适配器的型号。

adapter_name (str, list[str]) ：要设置为活动的适配器的名称

inference_mode (bool, 可选) ：是否应冻结激活的适配器（即`requires_grad=False`）。默认值为 False。

layer_cls（type，可选）：适配器层的类。默认为`BaseTunerLayer`。

设置模型的活动 PEFT 适配器。

活动适配器是那些参与前向传递的适配器。如果您想切换，请使用此功能
多个 PEFT 适配器之间。

## 设置指定适配器的`requires_grad`属性[[peft.tuners.tuners_utils.set_requires_grad]]

#### peft.tuners.tuners_utils.set_requires_grad[[peft.tuners.tuners_utils.set_requires_grad]]

```python
peft.tuners.tuners_utils.set_requires_grad(model, adapter_names: str | Sequence[str], requires_grad: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L2291)

**参数：**model (`nn.Module`) ：应更新适配器梯度要求的模型。

适配器名称（`str`或`Sequence[str]`）：应启用/禁用其渐变的适配器的名称。

require_grad (`bool`, *可选*) ：是否启用（`True`，默认）或禁用（`False`）。

启用或禁用给定适配器上的梯度。

## 将 PEFT 状态字典的权重加载到模型中[[peft.set_peft_model_state_dict]]

#### peft.set_peft_model_state_dict[[peft.set_peft_model_state_dict]]

```python
peft.set_peft_model_state_dict(model, peft_model_state_dict, adapter_name = 'default', ignore_mismatched_sizes: bool = False, low_cpu_mem_usage: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/save_and_load.py#L676)

**参数：**

型号 ([PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel))：Peft 模型。

peft_model_state_dict (`dict`) ：Peft 模型的状态字典。

适配器名称（`str`，*可选*，默认为`"default"`）：应设置其状态字典的适配器的名称。

ignore_mismatched_sizes (`bool`，*可选*，默认为`False`)：是否忽略状态字典中的不匹配。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：如果`model`在元设备上加载了适配器权重，则此参数必须为`True`，例如使用 `low_cpu_mem_usage=True` 调用 `inject_adapter_in_model` 后。否则，将其保留为 `False`。

**退货：**

加载结果 (`_IncompatibleKeys`)
具有 `missing_keys` 和 `unexpected_keys` 字段的命名元组。

设置 PEFT 模型的状态字典。给定 PEFT `state_dict`（由 [get_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.get_peft_model_state_dict) 返回），将权重插入模型中。的
模型需要已安装 PEFT 适配器（例如通过 [inject_adapter_in_model()](/docs/peft/v0.20.0/en/package_reference/functional#peft.inject_adapter_in_model)）。

设置适配器权重还需要重新插入适配器名称。该名称可能是不同的名称
比最初用于训练适配器的那个要好。

### 石蒜
https://huggingface.co/docs/peft/v0.20.0/package_reference/adapter_utils.md