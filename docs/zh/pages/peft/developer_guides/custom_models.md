<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 自定义模型

一些微调技术（例如提示调整）特定于语言模型。这意味着在 🤗 PEFT 中，它是
假设正在使用 🤗 Transformers 模型。然而，其他微调技术 - 例如
[LoRA](../package_reference/lora) - 不限于特定型号。

在本指南中，我们将了解如何将 LoRA 应用于多层感知器、[timm](https://huggingface.co/docs/timm/index) 库中的计算机视觉模型或新的 🤗 Transformers 架构。

## 多层感知器

假设我们想要使用 LoRA 微调多层感知器。这是定义：

```python
from torch import nn

class MLP(nn.Module):
    def __init__(self, num_units_hidden=2000):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(20, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, num_units_hidden),
            nn.ReLU(),
            nn.Linear(num_units_hidden, 2),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, X):
        return self.seq(X)
```

这是一个简单的多层感知器，具有输入层、隐藏层和输出层。

> [!提示]
> 对于这个玩具示例，我们选择了大量的隐藏单元来突出效率增益
> 来自 PEFT，但这些收益与更现实的例子相符。

该模型中有一些线性层可以使用 LoRA 进行调整。使用常见的 🤗 变形金刚时
模型中，PEFT 会知道将 LoRA 应用到哪些层，但在这种情况下，由我们作为用户来选择层。
要确定要调整的层的名称：

```python
print([(n, type(m)) for n, m in MLP().named_modules()])
```

这应该打印：

```
[('', __main__.MLP),
 ('seq', torch.nn.modules.container.Sequential),
 ('seq.0', torch.nn.modules.linear.Linear),
 ('seq.1', torch.nn.modules.activation.ReLU),
 ('seq.2', torch.nn.modules.linear.Linear),
 ('seq.3', torch.nn.modules.activation.ReLU),
 ('seq.4', torch.nn.modules.linear.Linear),
 ('seq.5', torch.nn.modules.activation.LogSoftmax)]
```假设我们要将 LoRA 应用于输入层和隐藏层，即 `'seq.0'` 和 `'seq.2'`。而且，
假设我们想要在没有 LoRA 的情况下更新输出层，那就是`'seq.4'`。相应的配置将
是：

```python
from peft import LoraConfig

config = LoraConfig(
    target_modules=["seq.0", "seq.2"],
    modules_to_save=["seq.4"],
)
```

这样，我们就可以创建 PEFT 模型并检查训练参数的比例：

```python
from peft import get_peft_model

model = MLP()
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# prints trainable params: 56,164 || all params: 4,100,164 || trainable%: 1.369798866581922
```

最后，我们可以使用任何我们喜欢的训练框架，或者编写我们自己的拟合循环来训练`peft_model`。

如需完整示例，请查看[this notebook](https://github.com/huggingface/peft/blob/main/examples/multilayer_perceptron/multilayer_perceptron_lora.ipynb)。

## 蒂姆模型

[timm](https://huggingface.co/docs/timm/index)库包含大量预训练的计算机视觉模型。
这些也可以通过 PEFT 进行微调。让我们看看这在实践中是如何运作的。

首先，确保 Python 环境中安装了 timm：

```bash
python -m pip install -U timm
```

接下来我们为图像分类任务加载 timm 模型：

```python
import timm

num_classes = ...
model_id = "timm/poolformer_m36.sail_in1k"
model = timm.create_model(model_id, pretrained=True, num_classes=num_classes)
```

同样，我们需要决定将 LoRA 应用到哪些层。由于 LoRA 支持 2D 转换层，并且由于
这些是该模型的主要构建块，我们应该将 LoRA 应用于 2D 转换层。来识别姓名
这些图层，让我们看看所有图层名称：

```python
print([(n, type(m)) for n, m in model.named_modules()])
```

这将打印一个很长的列表，我们只显示前几个：

```
[('', timm.models.metaformer.MetaFormer),
 ('stem', timm.models.metaformer.Stem),
 ('stem.conv', torch.nn.modules.conv.Conv2d),
 ('stem.norm', torch.nn.modules.linear.Identity),
 ('stages', torch.nn.modules.container.Sequential),
 ('stages.0', timm.models.metaformer.MetaFormerStage),
 ('stages.0.downsample', torch.nn.modules.linear.Identity),
 ('stages.0.blocks', torch.nn.modules.container.Sequential),
 ('stages.0.blocks.0', timm.models.metaformer.MetaFormerBlock),
 ('stages.0.blocks.0.norm1', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.0.token_mixer', timm.models.metaformer.Pooling),
 ('stages.0.blocks.0.token_mixer.pool', torch.nn.modules.pooling.AvgPool2d),
 ('stages.0.blocks.0.drop_path1', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.layer_scale1', timm.models.metaformer.Scale),
 ('stages.0.blocks.0.res_scale1', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.norm2', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.0.mlp', timm.layers.mlp.Mlp),
 ('stages.0.blocks.0.mlp.fc1', torch.nn.modules.conv.Conv2d),
 ('stages.0.blocks.0.mlp.act', torch.nn.modules.activation.GELU),
 ('stages.0.blocks.0.mlp.drop1', torch.nn.modules.dropout.Dropout),
 ('stages.0.blocks.0.mlp.norm', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.mlp.fc2', torch.nn.modules.conv.Conv2d),
 ('stages.0.blocks.0.mlp.drop2', torch.nn.modules.dropout.Dropout),
 ('stages.0.blocks.0.drop_path2', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.0.layer_scale2', timm.models.metaformer.Scale),
 ('stages.0.blocks.0.res_scale2', torch.nn.modules.linear.Identity),
 ('stages.0.blocks.1', timm.models.metaformer.MetaFormerBlock),
 ('stages.0.blocks.1.norm1', timm.layers.norm.GroupNorm1),
 ('stages.0.blocks.1.token_mixer', timm.models.metaformer.Pooling),
 ('stages.0.blocks.1.token_mixer.pool', torch.nn.modules.pooling.AvgPool2d),
 ...
 ('head.global_pool.flatten', torch.nn.modules.linear.Identity),
 ('head.norm', timm.layers.norm.LayerNorm2d),
 ('head.flatten', torch.nn.modules.flatten.Flatten),
 ('head.drop', torch.nn.modules.linear.Identity),
 ('head.fc', torch.nn.modules.linear.Linear)]
 ]
```经过仔细检查，我们发现 2D 卷积层的名称如 `"stages.0.blocks.0.mlp.fc1"` 和
`"stages.0.blocks.0.mlp.fc2"`。我们如何具体匹配这些图层名称？您可以编写 [regular
expressions](https://docs.python.org/3/library/re.html) 来匹配图层名称。对于我们的例子，正则表达式
`r".*\.mlp\.fc\d"` 应该可以完成这项工作。

此外，与第一个示例一样，我们应该确保输出层（在本例中为分类头）是
也更新了。查看上面打印的列表的末尾，我们可以看到它的名称为`'head.fc'`。考虑到这一点，
这是我们的 LoRA 配置：

```python
config = LoraConfig(target_modules=r".*\.mlp\.fc\d", modules_to_save=["head.fc"])
```

然后我们只需要将基础模型和配置传递给 `get_peft_model` 来创建 PEFT 模型：

```python
peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
# prints trainable params: 1,064,454 || all params: 56,467,974 || trainable%: 1.88505789139876
```

这表明我们只需要训练不到 2% 的参数，这是一个巨大的效率增益。

如需完整示例，请查看[this notebook](https://github.com/huggingface/peft/blob/main/examples/image_classification/image_classification_timm_peft_lora.ipynb)。

## 新的 Transformer 架构当新的流行 Transformer 架构发布时，我们会尽力将它们快速添加到 PEFT 中。如果您遇到不支持开箱即用的变压器模型，请不要担心，如果配置设置正确，它很可能仍然可以工作。具体来说，您必须在初始化相应的配置类时确定应调整的层并正确设置它们，例如`LoraConfig`。以下是一些有助于解决此问题的提示。

作为第一步，最好检查现有模型以获取灵感。您可以在 PEFT 存储库的[constants.py](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py) 中找到它们。通常，您会发现使用相同名称的类似架构。例如，如果新模型架构是“mistral”模型的变体，并且您想要应用 LoRA，则可以看到 `REDACTED` 中的“mistral”条目包含 `["q_proj", "v_proj"]`。这告诉您，对于“mistral”模型，LoRA 的 `target_modules` 应该是 `["q_proj", "v_proj"]`：

```python
from peft import LoraConfig, get_peft_model

my_mistral_model = ...
config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    ...,  # other LoRA arguments
)
peft_model = get_peft_model(my_mistral_model, config)
```如果这没有帮助，请使用 `named_modules` 方法检查模型架构中的现有模块，并尝试识别注意层，尤其是键、查询和值层。这些通常具有诸如 `c_attn`、`query`、`q_proj` 等名称。关键层并不总是适用的，理想情况下，您应该检查包含它是否会带来更好的性能。

此外，线性层是需要调整的常见目标（例如，在[QLoRA paper](https://huggingface.co/papers/2305.14314)中，作者也建议调整它们）。它们的名称通常包含字符串 `fc` 或 `dense`。

如果您想向 PEFT 添加新模型，请在 [constants.py](https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py) 中创建条目并在 [repository](https://github.com/huggingface/peft/pulls) 上打开拉取请求。不要忘记也更新[README](https://github.com/huggingface/peft#models-support-matrix)。

## 验证参数和层

您可以通过多种方式验证是否已将 PEFT 方法正确应用于模型。* 检查可使用 [print_trainable_parameters()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.print_trainable_parameters) 方法训练的参数比例。如果该数字低于或高于预期，请通过打印型号来检查型号`repr`。这显示了模型中所有图层类型的名称。确保仅预期的目标层被适配器层替换。例如，如果 LoRA 应用于 `nn.Linear` 层，那么您应该只会看到正在使用的 `lora.Linear` 层。

```py
peft_model.print_trainable_parameters()
```

* 查看适配层的另一种方法是使用 `targeted_module_names` 属性列出每个适配模块的名称。

```python
print(peft_model.targeted_module_names)
```

## 不支持的模块类型

LoRA 等方法仅在目标模块受 PEFT 支持时才有效。例如，可以将 LoRA 应用于 `nn.Linear` 和 `nn.Conv2d` 层，但不能应用于 `nn.LSTM`。如果您发现要应用 PEFT 的图层类不受支持，您可以：

 - 定义自定义映射以动态调度 LoRA 中的自定义模块
 - 打开一个[issue](https://github.com/huggingface/peft/issues)并请求维护人员将实现它的功能，或者如果对此模块类型的需求足够高，则指导您如何自己实现它

### LoRA 中自定义模块动态调度的实验支持> [!警告]
> 此功能是实验性的，可能会发生变化，具体取决于社区的接受程度。如果有大量需求，我们将推出公共且稳定的 API。

PEFT 支持 LoRA 自定义模块类型的实验性 API。假设您有一个 LSTM 的 LoRA 实现。通常，您无法告诉 PEFT 使用它，即使理论上它可以与 PEFT 一起使用。然而，这可以通过动态调度自定义层来实现。

目前实验性的 API 如下所示：

```python
class MyLoraLSTMLayer:
    ...

base_model = ...  # load the base model that uses LSTMs

# add the LSTM layer names to target_modules
config = LoraConfig(..., target_modules=["lstm"])
# define a mapping from base layer type to LoRA layer type
custom_module_mapping = {nn.LSTM: MyLoraLSTMLayer}
# register the new mapping
config._register_custom_module(custom_module_mapping)
# after registration, create the PEFT model
peft_model = get_peft_model(base_model, config)
# do training
```

> [!提示]
> 当您调用[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)时，您将看到一条警告，因为PEFT无法识别目标模块类型。在这种情况下，您可以忽略此警告。

通过提供自定义映射，PEFT 首先根据自定义映射检查基本模型的层，如果存在匹配，则分派到自定义 LoRA 层类型。如果不匹配，PEFT 将检查内置 LoRA 层类型是否匹配。

因此，此功能还可以用于覆盖现有的调度逻辑，例如如果您想使用自己的 LoRA 层来实现`nn.Linear`，而不是使用 PEFT 提供的层。创建自定义 LoRA 模块时，请遵循与[existing LoRA modules](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)相同的规则。需要考虑的一些重要限制：

- 自定义模块应继承自`nn.Module`和`peft.tuners.lora.layer.LoraLayer`。
- 自定义模块的`__init__`方法应具有位置参数`base_layer`和`adapter_name`。之后，还有额外的`**kwargs`，您可以自由使用或忽略。
- 可学习参数应存储在`nn.ModuleDict`或`nn.ParameterDict`中，其中键对应于特定适配器的名称（请记住，一个模型一次可以有多个适配器）。
- 这些可学习参数属性的名称应以`"lora_"`开头，例如`self.lora_new_param = ...`。
- 有些方法是可选的，例如如果你想支持权重合并，只需要实现`merge`和`unmerge`即可。

目前，保存模型时，有关自定义模块的信息不会保留。加载模型时，您必须再次注册自定义模块。

```python
# saving works as always and includes the parameters of the custom modules
peft_model.save_pretrained(<model-path>)

# loading the model later:
base_model = ...
# load the LoRA config that you saved earlier
config = LoraConfig.from_pretrained(<model-path>)
# register the custom module again, the same way as the first time
custom_module_mapping = {nn.LSTM: MyLoraLSTMLayer}
config._register_custom_module(custom_module_mapping)
# pass the config instance to from_pretrained:
peft_model = PeftModel.from_pretrained(model, tmp_path / "lora-custom-module", config=config)
```

如果您使用此功能并发现它很有用，或者遇到问题，请通过在 GitHub 上创建问题或讨论来告知我们。这使我们能够估计对此功能的需求，并在需求足够高时添加公共 API。### 深速
https://huggingface.co/docs/peft/v0.20.0/accelerate/deepspeed.md