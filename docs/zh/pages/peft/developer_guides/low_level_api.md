<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 适配器注入

使用 PEFT，您可以将可训练适配器注入任何 `torch` 模块，这允许您使用适配器方法，而无需依赖 PEFT 中的建模类。这适用于所有适配器，除了那些基于即时学习的适配器（例如前缀调整或 p 调整）。

检查下表以了解何时应该注入适配器。

|优点 |缺点 |
|---|---|
|模型就地修改，保留所有原始属性和方法 |手动编写 Hugging Face 中的 `from_pretrained` 和 `save_pretrained` 实用函数来保存和加载适配器 |
|适用于任何 `torch` 模块和模式 |不适用于 `PeftModel` 提供的任何实用方法，例如禁用和合并适配器 |

## 创建新的 PEFT 模型

要执行适配器注入，请使用[inject_adapter_in_model()](/docs/peft/v0.20.0/en/package_reference/functional#peft.inject_adapter_in_model)方法。此方法采用 3 个参数：PEFT 配置、模型和可选的适配器名称。如果您使用不同的适配器名称多次调用 [inject_adapter_in_model()](/docs/peft/v0.20.0/en/package_reference/functional#peft.inject_adapter_in_model)，您还可以将多个适配器附加到模型。

例如，要将 LoRA 适配器注入到 `DummyModel` 模块的 `linear` 子模块中：

```python
import torch
from peft import inject_adapter_in_model, LoraConfig

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(10, 10)
        self.linear = torch.nn.Linear(10, 10)
        self.lm_head = torch.nn.Linear(10, 10)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x = self.linear(x)
        x = self.lm_head(x)
        return x

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    target_modules=["linear"],
)

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)

dummy_inputs = torch.LongTensor([[0, 1, 2, 3, 4, 5, 6, 7]])
dummy_outputs = model(dummy_inputs)
```

打印模型以查看适配器是否已正确注入。

```bash
DummyModel(
  (embedding): Embedding(10, 10)
  (linear): Linear(
    in_features=10, out_features=10, bias=True
    (lora_dropout): ModuleDict(
      (default): Dropout(p=0.1, inplace=False)
    )
    (lora_A): ModuleDict(
      (default): Linear(in_features=10, out_features=64, bias=False)
    )
    (lora_B): ModuleDict(
      (default): Linear(in_features=64, out_features=10, bias=False)
    )
    (lora_embedding_A): ParameterDict()
    (lora_embedding_B): ParameterDict()
  )
  (lm_head): Linear(in_features=10, out_features=10, bias=True)
)
```

### 基于`state_dict`的注入有时，可能存在 PEFT 适配器检查点，但由于某种原因不知道相应的 PEFT 配置。要为此检查点注入 PEFT 层，您通常必须根据检查点中的 `state_dict` 对相应的 PEFT 配置进行逆向工程，尤其是 `target_modules` 参数。这可能很麻烦并且容易出错。为了避免这种情况，也可以调用 [inject_adapter_in_model()](/docs/peft/v0.20.0/en/package_reference/functional#peft.inject_adapter_in_model) 并将加载的 `state_dict` 作为参数传递：

```python
from safetensors.torch import load_file

model = ...
state_dict = load_file(<path-to-safetensors-file>)
lora_config = LoraConfig(...)
model = inject_adapter_in_model(lora_config, model, state_dict=state_dict)
```

在这种情况下，PEFT 将使用 `state_dict` 作为目标层的参考，而不是使用 PEFT 配置。作为用户，您不必设置 PEFT 配置的确切 `target_modules` 即可使其工作。但是，您仍然应该传递正确类型的 PEFT 配置，在本例中为 `LoraConfig`，您可以将 `target_modules` 保留为 `None`。

请注意，这仍然只会创建未初始化的 PEFT 层，`state_dict` 中的值不会用于填充模型权重。要填充权重，请继续调用 [set_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.set_peft_model_state_dict)，如下所述。⚠️ 请注意，如果 PEFT 配置中的配置与 `state_dict` 中的配置不匹配，PEFT 将向您发出警告。如果您知道未正确指定 PEFT 配置，则可以忽略该警告。

> [!警告]
> 如果原始 PEFT 适配器使用 `target_parameters` 而不是 `target_modules`，则从 `state_dict` 进样将无法正常工作。在这种情况下，必须使用正确的 PEFT 配置进行注入。

## 保存模型

要仅保存适配器，请使用 [get_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.get_peft_model_state_dict) 函数：

```python
from peft import get_peft_model_state_dict

peft_state_dict = get_peft_model_state_dict(model)
print(peft_state_dict)
```

否则，`model.state_dict()`返回模型的完整状态字典。

## 加载模型

加载保存的`state_dict`后，可以使用[set_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.set_peft_model_state_dict)函数应用：

```python
from peft import set_peft_model_state_dict

model = DummyModel()
model = inject_adapter_in_model(lora_config, model)
outcome = set_peft_model_state_dict(model, peft_state_dict)
# check that there were no wrong keys
print(outcome.unexpected_keys)
```

如果注入适配器很慢或者您需要加载大量适配器，您可以使用一种优化，允许在元设备上创建一个“空”适配器，并且仅在调用 [set_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.set_peft_model_state_dict) 时用真实权重填充权重。为此，请将 `low_cpu_mem_usage=True` 传递给 [inject_adapter_in_model()](/docs/peft/v0.20.0/en/package_reference/functional#peft.inject_adapter_in_model) 和 [set_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.set_peft_model_state_dict)。

```python
model = DummyModel()
model = inject_adapter_in_model(lora_config, model, low_cpu_mem_usage=True)

print(model.linear.lora_A["default"].weight.device.type == "meta")  # should be True
set_peft_model_state_dict(model, peft_state_dict, low_cpu_mem_usage=True)
print(model.linear.lora_A["default"].weight.device.type == "cpu")  # should be True
```

## 设置和加载基础权重上面的函数处理*适配器*的状态字典。还有一些情况需要通过 PEFT 包装器读取或写入“基本模型”权重。这并非完全微不足道，因为 PEFT 重命名了目标模块的参数（例如，`q_proj.weight` 变为 `q_proj.base_layer.weight`）并添加了基础模型中不存在的适配器参数。使用 `get_base_model_state_dict()` 和 `set_base_model_state_dict()` 在两个命名之间进行转换：

```python
from peft import LoraConfig, get_peft_model, get_base_model_state_dict, set_base_model_state_dict
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
model = get_peft_model(model, LoraConfig(target_modules="all-linear"))

# state dict with the original (pre-PEFT) key names, adapter parameters excluded
base_state_dict = get_base_model_state_dict(model)

# load a state dict with original key names into the PEFT-wrapped model
outcome = set_base_model_state_dict(model, base_state_dict, strict=False)
# check that there were no wrong keys
print(outcome.missing_keys, outcome.unexpected_keys)
```

其主要用例是在模型已被 PEFT 包裹后*加载基本重量。例如，TorchTitan 等 FSDP 训练设置会在元设备上初始化模型、应用 PEFT 并对结果进行分片。真实内存仅在分片后才存在，因此其键使用原始名称的检查点必须加载到已经包装的模型中。请注意，`get_base_model_state_dict()` 返回模型的实时张量，而不是副本。

### PEFT 检查点格式
https://huggingface.co/docs/peft/v0.20.0/developer_guides/checkpoint.md