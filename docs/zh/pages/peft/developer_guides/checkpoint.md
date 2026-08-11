<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PEFT 检查点格式

本文档描述了 PEFT 的检查点文件的结构以及如何在 PEFT 格式和其他格式之间进行转换。

## PEFT 文件

PEFT（参数高效微调）方法仅更新模型参数的一小部分，而不是全部。这很好，因为检查点文件通常比原始模型文件小得多，并且更容易存储和共享。然而，这也意味着要加载 PEFT 模型，您还需要拥有可用的原始模型。

当您在 PEFT 模型上调用 [save_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.save_pretrained) 时，PEFT 模型会保存三个文件，如下所述：

1. `adapter_model.safetensors`或`adapter_model.bin`

默认情况下，模型以 `safetensors` 格式保存，这是 `bin` 格式的安全替代方案，众所周知，该格式容易受到 [security vulnerabilities](https://huggingface.co/docs/hub/security-pickle) 的影响，因为它在底层使用了 pickle 实用程序。不过，两种格式存储相同的`state_dict`，并且可以互换。

`state_dict`仅包含适配器模块的参数，不包含基础模型的参数。为了说明大小差异，普通 BERT 模型需要约 420MB 的磁盘空间，而此 BERT 模型之上的 IA³ 适配器仅需要约 260KB。

2.`adapter_config.json``adapter_config.json` 文件包含适配器模块的配置，这是加载模型所必需的。以下是 IA3 适配器的 `adapter_config.json` 示例，其标准设置应用于 BERT 模型：

```json
{
  "auto_mapping": {
    "base_model_class": "BertModel",
    "parent_library": "transformers.models.bert.modeling_bert"
  },
  "base_model_name_or_path": "bert-base-uncased",
  "fan_in_fan_out": false,
  "feedforward_modules": [
    "output.dense"
  ],
  "inference_mode": true,
  "init_ia3_weights": true,
  "modules_to_save": null,
  "peft_type": "IA3",
  "revision": null,
  "target_modules": [
    "key",
    "value",
    "output.dense"
  ],
  "task_type": null
}
```

配置文件包含：

- 存储的适配器模块类型，`"peft_type": "IA3"`
- 有关基本型号的信息，如`"base_model_name_or_path": "bert-base-uncased"`
- 模型的修订（如果有），`"revision": null`

如果基础模型不是预训练的 Transformers 模型，则后两个条目将为`null`。除此之外，所有设置都与用于微调模型的特定 IA³ 适配器相关。

3.`README.md`

生成的`README.md`是PEFT模型的模型卡，包含一些预填充的条目。这样做的目的是为了更轻松地与其他人共享模型并提供有关模型的一些基本信息。加载模型不需要此文件。

## 转换为PEFT格式

当从其他格式转换为 PEFT 格式时，我们需要`adapter_model.safetensors`（或`adapter_model.bin`）文件和`adapter_config.json`文件。

### 适配器模型对于模型权重，使用从参数名称到值的正确映射对于 PEFT 加载文件非常重要。正确获取此映射需要检查实现细节，因为 PEFT 适配器没有普遍认可的格式。

幸运的是，对于常见的基本情况来说，计算出这种映射并不太复杂。让我们看一个具体的例子，[⟦T25⟧](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)：

```python
# showing only part of the code

class LoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.use_dora: dict[str, bool] = {}
        self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
        self._caches: dict[str, Any] = {}
        self.kwargs = kwargs
```

在PEFT中所有`LoraLayer`类使用的`__init__`代码中，有一堆用于初始化模型的参数，但只有少数参数与检查点文件相关：`lora_A`、`lora_B`、`lora_embedding_A`和`lora_embedding_B`。这些参数列在类属性`adapter_layer_names`中，并且包含可学习的参数，因此它们必须包含在检查点文件中。所有其他参数，如排名 `r`，均源自 `adapter_config.json`，并且必须包含在那里（除非使用默认值）。

我们来检查一下应用于 BERT 的 PEFT LoRA 模型的`state_dict`。当使用默认 LoRA 设置打印前五个键时（其余键相同，只是层数不同），我们得到：

- `base_model.model.encoder.layer.0.attention.self.query.lora_A.weight` 
- `base_model.model.encoder.layer.0.attention.self.query.lora_B.weight` 
- `base_model.model.encoder.layer.0.attention.self.value.lora_A.weight` 
- `base_model.model.encoder.layer.0.attention.self.value.lora_B.weight` 
- `base_model.model.encoder.layer.1.attention.self.query.lora_A.weight`
- 等

让我们来分解一下：- 默认情况下，对于 BERT 模型，LoRA 应用于注意力模块的`query` 和 `value` 层。这就是为什么您会在每层的键名称中看到 `attention.self.query` 和 `attention.self.value`。
- LoRA 将权重分解为两个低秩矩阵，`lora_A` 和 `lora_B`。这就是键名称中 `lora_A` 和 `lora_B` 的来源。
- 这些LoRA矩阵被实现为`nn.Linear`层，因此参数存储在`.weight`属性中（`lora_A.weight`，`lora_B.weight`）。
- 默认情况下，LoRA 不应用于 BERT 的嵌入层，因此 `lora_A_embedding` 和 `lora_B_embedding` 没有条目。
- `state_dict` 的按键始终以 `"base_model.model."` 开头。原因是，在 PEFT 中，我们将基础模型包装在特定于调谐器的模型中（在本例中为`LoraModel`），而该模型本身又包装在通用 PEFT 模型中（`PeftModel`）。因此，这两个前缀被添加到键中。转换为PEFT格式时，需要添加这些前缀。

> [!提示]
> 最后一点对于诸如提示调整之类的前缀调整技术来说并不正确。在那里，额外的嵌入直接存储在`state_dict`中，而不向键添加任何前缀。检查加载模型中的参数名称时，您可能会惊讶地发现它们看起来有点不同，例如`base_model.model.encoder.layer.0.attention.self.query.lora_A.default.weight`。不同之处在于倒数第二段中的 *`.default`* 部分。这部分的存在是因为 PEFT 通常允许一次添加多个适配器（使用 `nn.ModuleDict` 或 `nn.ParameterDict` 来存储它们）。例如，如果您添加另一个名为“other”的适配器，则该适配器的键将为 `base_model.model.encoder.layer.0.attention.self.query.lora_A.other.weight`。

当您调用 [save_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.save_pretrained) 时，适配器名称将从按键中删除。原因是适配器名称不是模型架构的重要组成部分；这只是一个任意的名称。加载适配器时，您可以选择完全不同的名称，并且模型仍将以相同的方式工作。这就是适配器名称不存储在检查点文件中的原因。

> [!提示]
> 如果调用`save_pretrained("some/path")`且适配器名称不是`"default"`，则适配器存储在与适配器同名的子目录中。因此，如果名称是“other”，它将存储在`some/path/other`内部。在某些情况下，决定将哪些值添加到检查点文件可能会变得更加复杂。例如，在 PEFT 中，DoRA 是作为 LoRA 的特例实现的。如果要将 DoRA 模型转换为 PEFT，您应该创建一个包含 DoRA 额外条目的 LoRA 检查点。您可以在之前的 `LoraLayer` 代码的 `__init__` 中看到这一点：

```python
self.lora_magnitude_vector: Optional[torch.nn.ParameterDict] = None  # for DoRA
```

这表明 DoRA 每层都有一个可选的额外参数。

### 适配器配置

加载 PEFT 模型所需的所有其他信息都包含在 `adapter_config.json` 文件中。让我们检查此文件中是否有应用于 BERT 的 LoRA 模型：

```json
{
  "alpha_pattern": {},
  "auto_mapping": {
    "base_model_class": "BertModel",
    "parent_library": "transformers.models.bert.modeling_bert"
  },
  "base_model_name_or_path": "bert-base-uncased",
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 8,
  "lora_dropout": 0.0,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 8,
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "query",
    "value"
  ],
  "task_type": null,
  "use_dora": false,
  "use_rslora": false
}
```其中包含很多条目，乍一看，找出所有正确的值放入其中可能会让人感到不知所措。然而，大多数条目对于加载模型来说并不是必需的。这要么是因为它们使用默认值并且不需要添加，要么是因为它们只影响 LoRA 权重的初始化，这与加载模型无关。如果您发现您不知道某个特定参数的作用，例如 `"use_rslora",`，请不要添加它，应该没问题。另请注意，随着添加更多选项，该文件将来将获得更多条目，但它应该向后兼容。

您至少应包含以下条目：

```json
{
  "target_modules": ["query", "value"],
  "peft_type": "LORA"
}
```

但是，建议添加尽可能多的条目，例如排名 `r` 或 `base_model_name_or_path`（如果是变形金刚模型）。这些信息可以帮助其他人更好地理解模型并更轻松地共享。要检查需要哪些键和值，请查看 PEFT 源代码中的 [config.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/config.py) 文件（例如，这是 LoRA 的配置文件）。

## 模型存储在某些情况下，您可能想要存储整个 PEFT 模型，包括基本权重。例如，如果尝试加载 PEFT 模型的用户无法使用基本模型，则这可能是必要的。您可以先合并权重或将其转换为 Transformer 模型。

### 合并权重

存储整个 PEFT 模型最直接的方法是将适配器权重合并到基本权重中：

```python
merged_model = model.merge_and_unload()
merged_model.save_pretrained(...)
```

但这种方法也有一些缺点：

- 一旦调用[merge_and_unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.merge_and_unload)，您将获得一个没有任何 PEFT 特定功能的基本模型。这意味着您不能再使用任何特定于 PEFT 的方法。
- 您无法取消合并权重、一次加载多个适配器、禁用适配器等。
- 并非所有 PEFT 方法都支持合并权重。
- 某些 PEFT 方法通常允许合并，但不允许进行特定设置（例如，当使用某些量化技术时）。
- 整个模型将比 PEFT 模型大得多，因为它也包含所有基本权重。

但使用合并模型进行推理应该会更快一些。

### 转换为 Transformers 模型保存整个模型的另一种方法（假设基础模型是 Transformers 模型）是使用这种 hacky 方法将 PEFT 权重直接插入基础模型中并保存它，只有当您“欺骗”Transformers 相信 PEFT 模型不是 PEFT 模型时，这种方法才有效。这仅适用于 LoRA，因为 Transformers 中未实现其他适配器。

```python
model = ...  # the PEFT model
...
# after you finish training the model, save it in a temporary location
model.save_pretrained(<temp_location>)
# now load this model directly into a transformers model, without the PEFT wrapper
# the PEFT weights are directly injected into the base model
model_loaded = AutoModel.from_pretrained(<temp_location>)
# now make the loaded model believe that it is _not_ a PEFT model
model_loaded._hf_peft_config_loaded = False
# now when we save it, it will save the whole model
model_loaded.save_pretrained(<final_location>)
# or upload to Hugging Face Hub
model_loaded.push_to_hub(<final_location>)
```

### 模型合并
https://huggingface.co/docs/peft/v0.20.0/developer_guides/model_merging.md