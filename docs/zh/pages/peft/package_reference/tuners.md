<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 调音器

调谐器（或适配器）是可以插入`torch.nn.Module`的模块。 `BaseTuner` 其他调谐器的基类，并提供共享方法和属性来准备适配器配置并用适配器模块替换目标模块。 `BaseTunerLayer`是适配器层的基类。它提供了用于管理适配器的方法和属性，例如激活和禁用适配器。

## BaseTuner[[peft.tuners.tuners_utils.BaseTuner]]

#### peft.tuners.tuners_utils.BaseTuner[[peft.tuners.tuners_utils.BaseTuner]]

```python
peft.tuners.tuners_utils.BaseTuner(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L236)

**参数：**

model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。

forward (`Callable`) ：模型的前向方法。

peft_config (`Union[`PeftConfig`, dict[str, PeftConfig]]`) ：适配器配置对象，它应该是`str`到`PeftConfig`对象的字典。还可以传递 PeftConfig 对象，然后将使用默认名称 `adapter` 创建一个新适配器，或者使用键 `adapter_name` 和该 peft 配置的值创建一个新字典。

config (`dict[str, Any]`) ：模型配置对象，它应该是`str`到`Any`对象的字典。Targeted_module_names (`list[str]`) ：实际适配的模块名称列表。如果您想快速仔细检查 `config.target_modules` 是否指定正确，这对于检查非常有用。

Targeted_pa​​rameter_names (`list[str]`) ：实际适配的参数名称列表。如果您想快速仔细检查 `config.target_parameters` 是否指定正确，这对于检查非常有用。

prefix (`str`) ：PEFT 方法特定的唯一前缀。例如。 `"lora_"` 适用于 LoRA。

基本调谐器模型，为可注入到的所有调谐器提供通用方法和属性
torch.nn.模块

要添加新的 Tuner 类，需要重写以下方法：

- **_prepare_adapter_config**：
  最终准备适配器配置的私有方法，例如字段 `target_modules` 的情况
  失踪了。
- **_创建并_替换**：
  用于创建目标模块并将其替换为适配器模块的私有方法。
- **_check_target_module_exists**：
  一个私有帮助器方法，用于检查传递的模块的密钥名称是否与文件中的任何目标模块匹配
  适配器配置。

最简单的是检查 `peft.tuners.lora.LoraModel` 类中做了什么。#### 删除适配器[[peft.tuners.tuners_utils.BaseTuner.delete_adapter]]

```python
delete_adapter(adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L537)

**参数：**

adapter_name (str) ：要删除的适配器的名称。

删除现有适配器。

####disable_adapter_layers[[peft.tuners.tuners_utils.BaseTuner.disable_adapter_layers]]

```python
disable_adapter_layers()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L513)

就地禁用所有适配器。

禁用所有适配器时，模型输出对应于基本模型的输出。

####启用_适配器_层[[peft.tuners.tuners_utils.BaseTuner.enable_adapter_layers]]

```python
enable_adapter_layers()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L530)

就地启用所有适配器

#### get_model_config[[peft.tuners.tuners_utils.BaseTuner.get_model_config]]

```python
get_model_config(model: nn.Module)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1279)

**参数：**

model (`nn.Module`) ：从中获取配置的模型。

default (`dict|None`, *可选*) --：如果模型没有配置属性，则返回什么。

此方法以字典形式从模型中获取配置。如果模型没有属性配置，那么这个
方法返回默认配置。

####注入适配器[[peft.tuners.tuners_utils.BaseTuner.inject_adapter]]

```python
inject_adapter(model: nn.Module, adapter_name: str, autocast_adapter_dtype: bool = True, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L753)

**参数：**

model (`nn.Module`) ：要调整的模型。

adapter_name (`str`) ：适配器名称。autocast_adapter_dtype (`bool`, *可选*) ：是否自动转换适配器数据类型。默认为 `True`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

state_dict（`dict`，*可选*，默认为`None`）：如果此处传递state_dict，则将根据state_dict的条目注入适配器。当 PEFT 方法的确切 `target_modules` 未知时，这可能很有用，例如因为检查点是在没有元数据的情况下创建的。请注意，不使用 state_dict 中的值，仅使用键来确定应调整的正确层。

创建适配器层并用适配器层替换目标模块。该方法在下面调用
如果通过了非提示调整适配器类，则通过 `peft.mapping.get_peft_model` 进行遮罩。

相应的 PEFT 配置直接从 BaseTuner 类的 `peft_config` 属性中检索。

#### merge_adapter[[peft.tuners.tuners_utils.BaseTuner.merge_adapter]]

```python
merge_adapter(adapter_names: Optional[list[str]] = None, safe_merge: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1227)

**参数：**adapter_names (`list[str]`, *可选*) ：应合并的适配器名称列表。如果`None`，所有活动适配器将被合并。默认为`None`。

safe_merge (`bool`, *可选*) ：如果`True`，合并操作将在原始权重的副本中执行，并在合并权重之前检查 NaN。如果您想检查合并操作是否会产生 NaN，这很有用。默认为 `False`。

此方法将适配器层合并到基础模型中。

合并适配器可以加快前向传播的速度。适配器权重的副本仍保留在
内存，这是取消适配器合并所需的。为了合并适配器权重而不保留它们
记忆中请拨打`merge_and_unload`。

#### merge_and_unload[[peft.tuners.tuners_utils.BaseTuner.merge_and_unload]]

```python
merge_and_unload(progressbar: bool = False, safe_merge: bool = False, adapter_names: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L696)

**参数：**

Progressbar (`bool`) ：是否显示指示卸载和合并过程的进度条（默认值：False）。

safe_merge (`bool`) ：是否激活安全合并检查以检查适配器权重中是否存在潜在的 Nan。adapter_names (`List[str]`, *可选*) ：应合并的适配器名称列表。如果没有，所有活动适配器将被合并。默认为 `None`。

此方法将适配器层合并到基础模型中。

如果有人想将基本模型用作独立模型，则需要这样做。返回的模型具有相同的
架构作为原始基础模型。

将返回的模型分配给变量并使用它很重要，这不是就地操作！

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import PeftModel

>>> model_id = ...
>>> base_model = AutoModelForCausalLM.from_pretrained(model_id)
>>> peft_model_id = ...
>>> model = PeftModel.from_pretrained(base_model, peft_model_id)
>>> merged_model = model.merge_and_unload()
```

#### set_adapter[[peft.tuners.tuners_utils.BaseTuner.set_adapter]]

```python
set_adapter(adapter_name: str | list[str], inference_mode: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1265)

**参数：**

adapter_name (str, list[str]) ：要设置为活动的适配器的名称

inference_mode (bool, 可选) ：是否应冻结激活的适配器（即`requires_grad=False`）。默认值为 False。

设置活动适配器。

#### set_requires_grad[[peft.tuners.tuners_utils.BaseTuner.set_requires_grad]]

```python
set_requires_grad(adapter_names: str | Sequence[str], requires_grad: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L553)

**参数：**

适配器名称（`str`或`Sequence[str]`）：应启用/禁用其渐变的适配器的名称。

require_grad (`bool`, *可选*) ：是否启用（`True`，默认）或禁用（`False`）。

启用或禁用给定适配器上的梯度。####支持_lora_conversion[[peft.tuners.tuners_utils.BaseTuner.supports_lora_conversion]]

```python
supports_lora_conversion(adapter_name: str = 'default')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1388)

该型号的适配器是否可以转为LoRA。

通常，如果 PEFT 方法是可加的，即 W' = W_base + delta_weight，则此方法有效。

#### 卸载[[peft.tuners.tuners_utils.BaseTuner.unload]]

```python
unload()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L734)

通过删除所有 PEFT 模块返回基本模型。

将返回的模型分配给变量并使用它很重要，这不是就地操作！

#### unmerge_adapter[[peft.tuners.tuners_utils.BaseTuner.unmerge_adapter]]

```python
unmerge_adapter()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1256)

此方法从基础模型中取消合并所有合并的适配器层。

## BaseTunerLayer[[peft.tuners.tuners_utils.BaseTunerLayer]]

#### peft.tuners.tuners_utils.BaseTunerLayer[[peft.tuners.tuners_utils.BaseTunerLayer]]

```python
peft.tuners.tuners_utils.BaseTunerLayer()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1408)

**参数：**

is_pluggable (`bool`, *可选*) : 适配器层是否可以插入任何pytorch模块

active_adapters (Union[List`str`, `str`], *可选*) ：活动适配器的名称。

一个调谐器层 mixin，为所有调谐器提供通用方法和属性。#### 删除适配器[[peft.tuners.tuners_utils.BaseTunerLayer.delete_adapter]]

```python
delete_adapter(adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1645)

**参数：**

adapter_name (`str`) : 要删除的适配器的名称

从层中删除适配器

这应该在所有适配器层上调用，否则我们将得到不一致的状态。

如果删除的适配器是活动适配器，此方法还将设置一个新的活动适配器。这很重要
以确定性方式选择新适配器，以便在所有层上选择相同的适配器。

#### 启用_适配器[[peft.tuners.tuners_utils.BaseTunerLayer.enable_adapters]]

```python
enable_adapters(enabled: bool)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1564)

**参数：**

enabled (bool) : True 表示启用适配器，False 表示禁用适配器

切换适配器的启用和禁用

负责为适配器权重设置 require_grad 标志。

#### get_base_layer[[peft.tuners.tuners_utils.BaseTunerLayer.get_base_layer]]

```python
get_base_layer()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1438)

（递归地）获取base_layer。

这对于调谐器层包裹另一个调谐器层的情况是必要的。

#### get_base_weight[[peft.tuners.tuners_utils.BaseTunerLayer.get_base_weight]]

```python
get_base_weight()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1450)

返回基础层的权重。如果权重被量化的话，这可以解决潜在的反量化问题。

#### set_adapter[[peft.tuners.tuners_utils.BaseTunerLayer.set_adapter]]

```python
set_adapter(adapter_names: str | list[str], inference_mode: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1609)

**参数：**

适配器名称（`str`或`list[str]`）：要设置为活动的适配器的名称。

inference_mode (bool, 可选) ：是否应冻结激活的适配器（即`requires_grad=False`）。默认值为 False。

设置活动适配器。

此外，此函数会将指定的适配器设置为可训练（即，requires_grad=True），除非
inference_mode 为 True。

#### set_base_weight[[peft.tuners.tuners_utils.BaseTunerLayer.set_base_weight]]

```python
set_base_weight(weight_data: torch.Tensor)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1459)

将基础层的基础权重设置为新张量

这也适用于量化权重。

#### set_requires_grad[[peft.tuners.tuners_utils.BaseTunerLayer.set_requires_grad]]

```python
set_requires_grad(adapter_names: str | Sequence[str], requires_grad: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1689)

**参数：**

适配器名称（`str`或`Sequence[str]`）：应启用/禁用其渐变的适配器的名称。

require_grad (`bool`, *可选*) ：是否启用（`True`，默认）或禁用（`False`）。

启用或禁用给定适配器上的梯度。

####supports_lora_conversion[[peft.tuners.tuners_utils.BaseTunerLayer.supports_lora_conversion]]```python
supports_lora_conversion(adapter_name: str = 'default')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/tuners_utils.py#L1794)

该层类型是否可以转换为LoRA。

通常，如果 PEFT 方法是可加的，即 W' = W_base + delta_weight，则此方法有效。

### 情境感知提示调整：利用对抗性方法推进情境学习
https://huggingface.co/docs/peft/v0.20.0/package_reference/cpt.md