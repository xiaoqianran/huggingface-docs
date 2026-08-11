<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 热插拔适配器

热插拔适配器的想法如下：我们已经可以加载多个适配器，例如同时使用两个 LoRA。但有时，我们希望加载一个 LoRA，然后将其权重替换为另一个适配器的 LoRA 权重。现在可以使用 `hotswap_adapter` 功能。

一般来说，这应该比删除一个适配器并在其位置加载该适配器更快，这将是如何在不进行热插拔的情况下实现相同的最终结果。热交换的另一个优点是，如果 PEFT 模型已经使用 `torch.compile` 编译，它可以防止重新编译。这可以节省很多时间。

## 没有 `torch.compile` 的示例

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter

model_id = ...
inputs = ...
device = ...
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, <path-adapter-0>)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

## `torch.compile` 的示例

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel
from peft.utils.hotswap import hotswap_adapter, prepare_model_for_compiled_hotswap

model_id = ...
inputs = ...
device = ...
max_rank = ...  # maximum rank among all LoRA adapters that will be used
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# load lora 0
model = PeftModel.from_pretrained(model, <path-adapter-0>)
# Prepare the model to allow hotswapping even if ranks/scalings of 2nd adapter differ.
# You can skip this step if all ranks and scalings are identical.
prepare_model_for_compiled_hotswap(model, target_rank=max_rank)
model = torch.compile(model)
with torch.inference_mode():
    output_adapter_0 = model(inputs)

# replace the "default" lora adapter with the new one
hotswap_adapter(model, <path-adapter-1>, adapter_name="default", torch_device=device)
with torch.inference_mode():
    output_adapter_1 = model(inputs).logits
```

请注意，如果您想热交换通过`target_parameters`添加的权重，即直接针对`nn.Parameter`，则无法防止重新编译和/或图形中断。因此，建议避免将`target_parameters`与编译模型和热插拔一起使用。

## 注意事项

热插拔适用于变压器模型和扩散器模型。但是，有一些注意事项：- 目前，仅正确支持 LoRA。
- 它仅适用于相同的 PEFT 方法，因此不能交换 LoRA 和 LoHa 等。
- 正在换入的适配器必须面向与前一个适配器相同的层或这些层的子集。它无法定位新层。因此，如果可能，请从针对大多数层的适配器开始。

## API[[peft.utils.hotswap.hotswap_adapter]]

#### peft.utils.hotswap.hotswap_adapter[[peft.utils.hotswap.hotswap_adapter]]

```python
peft.utils.hotswap.hotswap_adapter(model, model_name_or_path, adapter_name, torch_device = None, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/hotswap.py#L613)

**参数：**

model ([~PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)) ：带有加载适配器的 PEFT 模型。

model_name_or_path (`str`) ：从中加载新适配器的模型的名称或路径。

adapter_name (`str`) ：要交换的适配器的名称，例如`"default"`。交换后名称将保持不变。

torch_device ：（`str`，*可选*，默认为 None）：要加载新适配器的设备。

- ****kwargs** (`optional`) ：用于加载配置和权重的附加关键字参数。

用新适配器数据替换旧适配器数据，保持其余部分不变。

目前仅支持 LoRA。当您想要用新适配器替换已加载的适配器时，此功能非常有用。适配器名称将
保持不变，但权重和其他参数将被交换。

如果适配器不兼容，例如针对不同的图层或具有不同的 alpha 值，将会出现错误
被提高。

示例：

```py
>>> import torch
>>> from transformers import AutoModelForCausalLM
>>> from peft import PeftModel
>>> from peft.utils.hotswap import hotswap_adapter

>>> model_id = ...
>>> inputs = ...
>>> device = ...
>>> model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

>>> # load lora 0
>>> model = PeftModel.from_pretrained(model, "path-adapter-0")
>>> model = torch.compile(model)  # optionally compile the model
>>> with torch.inference_mode():
...     output_adapter_0 = model(inputs)

>>> # replace the "default" lora adapter with the new one
>>> hotswap_adapter(model, "path-adapter-1", adapter_name="default", torch_device=device)
>>> with torch.inference_mode():
...     output_adapter_1 = model(inputs).logits
```

#### peft.utils.hotswap.hotswap_adapter_from_state_dict[[peft.utils.hotswap.hotswap_adapter_from_state_dict]]

```python
peft.utils.hotswap.hotswap_adapter_from_state_dict(model: torch.nn.Module, state_dict: dict[str, torch.Tensor], adapter_name: str, config: LoraConfig, parameter_prefix: str = 'lora_')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/hotswap.py#L419)

**参数：**

model (`nn.Module`) ：已加载适配器的模型。

state_dict (`dict[str, torch.Tensor]`) ：新适配器的状态字典，需要兼容（针对相同的模块等）。

adapter_name (`str`) ：应该热插拔的适配器的名称，例如`"default"`。交换后名称将保持不变。

config (`LoraConfig`) ：LoRA 适配器的配置。这用于确定适配器的缩放和等级。

parameter_prefix（`str`，*可选*，默认为`"lora_"`）：用于标识状态字典中适配器键的前缀。对于 LoRA，这将是 `"lora_"` （默认值）。

**加薪：** ``RuntimeError``

- ``RuntimeError`` -- 
  如果新旧适配器不兼容，则会引发运行时错误。将模型中的适配器权重替换为 state_dict 中的权重。

目前仅支持 LoRA。

这是一个低级函数，假设已检查适配器的兼容性并且
state_dict 已正确映射以与 PEFT 一起使用。对于为您执行此工作的高级功能，
使用 `hotswap_adapter` 代替。

### IA3
https://huggingface.co/docs/peft/v0.20.0/package_reference/ia3.md