<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 故障排除

如果您在使用 PEFT 时遇到任何问题，请查看以下常见问题列表及其解决方案。

## 示例不起作用

示例通常依赖于最新的软件包版本，因此请确保它们是最新的。特别是检查以下软件包版本：

- `peft`
- `transformers`
- `accelerate`
- `torch`

一般来说，您可以通过在 Python 环境中运行以下命令来更新软件包版本：

```bash
python -m pip install -U <package_name>
```

从源代码安装 PEFT 对于跟上最新发展很有用：

```bash
python -m pip install git+https://github.com/huggingface/peft
```

## Dtype相关问题

### ValueError：尝试取消缩放 FP16 梯度

发生此错误的原因可能是模型加载了`dtype=torch.float16`，然后在自动混合精度（AMP）上下文中使用，例如通过在🤗变形金刚的[Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer)类中设置`fp16=True`。原因是使用 AMP 时，可训练权重不应该使用 fp16。要在不加载 fp32 中整个模型的情况下完成此操作，请将以下内容添加到您的代码中：

```python
peft_model = get_peft_model(...)

# add this:
for param in model.parameters():
    if param.requires_grad:
        param.data = param.data.float()

# proceed as usual
trainer = Trainer(model=peft_model, fp16=True, ...)
trainer.train()
```

或者，您可以使用 [cast_mixed_precision_params()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.cast_mixed_precision_params) 函数来正确投射权重：

```python
from peft import cast_mixed_precision_params

peft_model = get_peft_model(...)
cast_mixed_precision_params(peft_model, dtype=torch.float16)

# proceed as usual
trainer = Trainer(model=peft_model, fp16=True, ...)
trainer.train()
```> [!提示]
> 从 PEFT 版本 v0.12.0 开始，PEFT 在适当的情况下自动将适配器权重的 dtype 从 `torch.float16` 和 `torch.bfloat16` 提升为 `torch.float32`。为了防止这种行为，您可以将 `autocast_adapter_dtype=False` 传递给 [~get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)、[from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained) 和 [load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)。

### 选择适配器的数据类型

大多数 PEFT 方法（如 LoRA）通过添加可训练的适配器权重来工作。默认情况下，这些权重存储在 float32 dtype (fp32) 中，即以相对较高的精度存储。因此，即使基础模型以 float16 (fp16) 或 bfloat16 (bf16) 加载，适配器权重也为 float32。在前向传递过程中计算适配器结果时，输入通常采用基本模型的 dtype，因此如有必要，它将向上转换为 float32，然后转换回原始 dtype。

如果您希望适配器权重采用基础模型的较低精度，即 float16 或 bfloat16，则可以在创建模型 ([~get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)) 或加载模型 ([from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained)) 时传递 `autocast_adapter_dtype=False`。这样做有一些优点和缺点：

半精密适配器的优点：
- 计算速度稍快一些
- 内存稍少
- 检查点的文件大小较小（大小的一半）半精度适配器的缺点：
- 损失稍微严重一些
- 上溢或下溢的风险较高

请注意，对于大多数用例，总体运行时间和内存成本将由基础模型的大小和数据集决定，而 PEFT 适配器的数据类型只会产生很小的影响。

## 训练运行但模型没有改进

如果训练看起来很健康（损失下降，检查点保存无错误），但结果模型的性能并不比基础模型好，请检查可训练参数在训练期间是否实际发生了变化：

```python
# before training
sample_param = next(p for p in model.parameters() if p.requires_grad)
before = sample_param.detach().clone()

# ... train for at least three optimizer steps ...

after = next(p for p in model.parameters() if p.requires_grad)
print("weights updated:", not torch.allclose(before, after.detach().cpu()))
```如果权重没有移动，可能的原因是在参数在目标设备上具体化之前捕获了参数引用。当基本模型被 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 包装而其参数尚未移动到目标设备时，并且第三方库随后注册前向/后向钩子（例如，来自差分隐私框架的每个样本梯度钩子）或者在第一次前向调用之前创建优化器时，就会发生这种情况。当参数稍后具体化时，这些钩子和优化器引用指向陈旧的张量，并且每次更新都变成静默无操作。

为了避免这种情况，请使用以下顺序：

1. 加载基础模型并将其移动到目标设备（例如`model.to(device)`）。
2. 用[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)包裹起来。
3. 然后才注册钩子或创建优化器。

## 加载的 PEFT 模型产生的不良结果

加载的 PEFT 模型得到较差结果可能有多种原因，如下所列。如果您仍然无法解决问题，请查看 GitHub 上是否有其他人有类似的 [issue](https://github.com/huggingface/peft/issues)，如果找不到，请打开一个新问题。当打开一个问题时，如果您提供一个重现该问题的最小代码示例，将会有很大帮助。另外，请报告加载的模型的性能是否与微调前的模型相同，是否在随机级别上执行，或者是否仅比预期稍差。这些信息可以帮助我们更快地识别问题。

### 随机偏差

如果您的模型输出与之前的运行不完全相同，则可能存在随机元素问题。例如：

1. 请确保它处于`.eval()`模式，这很重要，例如，如果模型使用dropout
2. 如果在语言模型上使用[generate](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/text_generation#transformers.GenerationMixin.generate)，可能会出现随机采样，因此要获得相同的结果需要设置随机种子
3. 如果您使用量化并合并权重，则由于舍入误差，预计会出现小偏差

### 模型加载不正确

请确保正确加载模型。一个常见的错误是尝试使用 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 加载_trained_模型，这是不正确的。相反，加载代码应该如下所示：

```python
from peft import PeftModel, PeftConfig

base_model = ...  # to load the base model, use the same code as when you trained it
config = PeftConfig.from_pretrained(peft_model_id)
peft_model = PeftModel.from_pretrained(base_model, peft_model_id)
```

### 随机初始化层对于某些任务，在配置中正确配置`modules_to_save`以考虑随机初始化的层非常重要。

举个例子，如果您使用 LoRA 微调序列分类的语言模型，这是必要的，因为🤗 Transformers 在模型顶部添加了一个随机初始化的分类头。如果不将此层添加到`modules_to_save`，则不会保存分类头。下次加载模型时，您将获得_不同_随机初始化的分类头，从而产生完全不同的结果。

如果您在配置中提供 `task_type` 参数，PEFT 会尝试正确猜测 `modules_to_save`。这应该适用于遵循标准命名方案的变压器模型。不过，仔细检查总是一个好主意，因为我们不能保证所有模型都遵循命名方案。

当您加载具有随机初始化层的 Transformer 模型时，您应该看到如下警告：

```
Some weights of <MODEL> were not initialized from the model checkpoint at <ID> and are newly initialized: [<LAYER_NAMES>].
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
```

应将提到的层添加到配置中的`modules_to_save`以避免所描述的问题。> [!提示]
> 举个例子，当加载使用 DeBERTa 架构进行序列分类的模型时，您会看到一条警告，指出以下权重是新初始化的：`['classifier.bias', 'classifier.weight', 'pooler.dense.bias', 'pooler.dense.weight']`。由此可见，`classifier`和`pooler`层应添加到：`modules_to_save=["classifier", "pooler"]`。

### 扩展词汇量

对于许多语言微调任务来说，由于引入了新的标记，因此扩展模型的词汇量是必要的。这需要扩展嵌入层以考虑新的标记，并且根据微调方法，在保存适配器时除了适配器权重之外还存储嵌入层。有几种方法可以实现按参数有效性排序：

- [trainable tokens](../package_reference/trainable_tokens)，仅训练指定的标记，可选择仅存储更新的值
- 在嵌入矩阵上训练适配器，可选择仅存储更新的值
- 嵌入层的全面微调

#### 使用可训练令牌

让我们从可训练的令牌开始，在本例中是[LoRA integration](../package_reference/lora#efficiently-train-tokens-alongside-lora)。  如果您只想训练新的嵌入而不感兴趣，请参阅[standalone documentation](../package_reference/trainable_tokens)。要启用嵌入层的选择性令牌训练，您需要通过 `trainable_token_indices` 参数提供新添加的令牌的令牌 ID。  如果有多个嵌入层，您可以选择指定要定位的层。对于米斯特拉尔模型，这可能如下所示：

```python
new_tokens = ['<think>', '</think>']
tokenizer.add_tokens(new_tokens)
base_model.resize_token_embeddings(len(tokenizer))

lora_config = LoraConfig(
    ...,
    trainable_token_indices={'embed_tokens': tokenizer.convert_tokens_to_ids(new_tokens)},
)
```

如果您的模型使用绑定权重（例如`lm_head`），可训练令牌将尝试解决这些问题并保持更新，因此在这种情况下，不需要添加`modules_to_save=["lm_head"]`。仅当模型使用 Transformers 约定来绑定权重时，这才有效。

使用 `model.save_pretrained` 保存模型可能会保存完整的嵌入矩阵，而不是
仅将差异作为预防措施，因为嵌入矩阵已调整大小。为了节省空间，您可以通过在调用 `save_pretrained` 时设置 `save_embedding_layers=False` 来禁用此行为。只要您不通过其他方式修改嵌入矩阵，这样做就是安全的，因为可训练令牌不会跟踪此类更改。

#### 使用适配器，例如洛拉

通过将嵌入层添加到适配器配置的 `target_modules` 来准备嵌入层。例如，Mistral 配置可能如下所示：

```python
config = LoraConfig(..., target_modules=["embed_tokens", "lm_head", "q_proj", "v_proj"])
```一旦添加到`target_modules`，如果模型具有`get_input_embeddings`和`get_output_embeddings`，PEFT会在保存适配器时自动存储嵌入层。变形金刚模型通常就是这种情况。

如果模型的嵌入层不遵循 Transformer 的命名方案，但仍然实现了`get_input_embeddings`，您仍然可以在保存适配器时通过手动传递`save_embedding_layers=True`来保存它：

```python
model = get_peft_model(...)
# train the model
model.save_pretrained("my_adapter", save_embedding_layers=True)
```

为了进行推理，首先加载基本模型并按照训练模型之前的方式调整其大小。调整基本模型的大小后，您可以加载 PEFT 检查点。

如需完整示例，请查看[this notebook](https://github.com/huggingface/peft/blob/main/examples/causal_language_modeling/peft_lora_clm_with_additional_tokens.ipynb)。

#### 全面微调

就 VRAM 或存储空间而言，全面微调的成本更高，但如果其他方法都失败，您可以退回到此方法，看看它是否适合您。通过将嵌入层的名称添加到`modules_to_save`来实现。请注意，您还需要添加绑定层，例如`lm_head`。采用 LoRA 的 Mistral 模型示例：

```python
config = LoraConfig(..., modules_to_save=["embed_tokens", "lm_head"], target_modules=["q_proj", "v_proj"])
```

### 收到有关“权重未从模型检查点初始化”的警告

当您加载已针对任务（例如分类）进行训练的 PEFT 模型时，您可能会收到如下警告：> LlamaForSequenceClassification 的一些权重未从 meta-llama/Llama-3.2-1B 的模型检查点初始化，而是新初始化的：['score.weight']。您可能应该在下游任务上训练该模型，以便能够将其用于预测和推理。

虽然这看起来很可怕，但很可能没什么可担心的。此警告来自 Transformers，并非 PEFT 特定警告。它让您知道随机初始化的分类头 (`score`) 附加到基本模型，并且必须训练该头才能产生合理的预测。

当您在训练模型之前收到此警告时，如果您正确地将 `task_type` 参数传递给 PEFT 配置，PEFT 会自动使分类头可训练。

```python
from peft import LoraConfig, TaskType

lora_config = LoraConfig(..., task_type=TaskType.SEQ_CLS)
```

如果您的分类头不遵循 Transformers 的通常命名约定（这种情况很少见），您必须在 `modules_to_save` 中明确告诉 PEFT 头的名称。

```python
lora_config = LoraConfig(..., modules_to_save=["name-of-classification-head"])
```

要检查分类头的名称，打印模型，它应该是最后一个模块。如果您从推理代码中收到此警告，即在训练模型之后，当您加载 PEFT 模型时，您始终必须首先加载 Transformers 模型。由于 Transformers 不知道您之后会加载 PEFT 权重，因此它仍然会发出警告。

与往常一样，最佳实践是通过对其运行一些验证来确保模型正确地进行推理。

### 检查图层和模型状态

有时，PEFT 模型可能会处于不良状态，尤其是在处理多个适配器时。对于存在哪些适配器、哪个适配器处于活动状态、哪个适配器已合并等，可能会出现一些混淆。为了帮助调查此问题，请调用 [get_layer_status()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.get_layer_status) 和 [get_model_status()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.get_model_status) 方法。

[get_layer_status()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.get_layer_status) 方法为您提供每个目标层的活动、合并和可用适配器的详细概述。

```python
>>> from transformers import AutoModel
>>> from peft import get_peft_model, LoraConfig

>>> model_id = "google/flan-t5-small"
>>> model = AutoModel.from_pretrained(model_id)
>>> model = get_peft_model(model, LoraConfig())

>>> model.get_layer_status()
[TunerLayerStatus(name='model.encoder.block.0.layer.0.SelfAttention.q',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['default'],
                  merged_adapters=[],
                  requires_grad={'default': True},
                  available_adapters=['default']),
 TunerLayerStatus(name='model.encoder.block.0.layer.0.SelfAttention.v',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['default'],
                  merged_adapters=[],
                  requires_grad={'default': True},
                  available_adapters=['default']),
...]

>>> model.get_model_status()
TunerModelStatus(
    base_model_type='T5Model',
    adapter_model_type='LoraModel',
    peft_types={'default': 'LORA'},
    trainable_params=344064,
    total_params=60855680,
    num_adapter_layers=48,
    enabled=True,
    active_adapters=['default'],
    merged_adapters=[],
    requires_grad={'default': True},
    available_adapters=['default'],
)
```

在模型状态输出中，您应该查找显示 `"irregular"` 的条目。这意味着 PEFT 检测到模型中存在不一致的状态。例如，如果`merged_adapters="irregular"`，则意味着对于至少一个适配器，它已合并到某些目标模块上，但未合并到其他目标模块上。因此，推理结果很可能是不正确的。解决此问题的最佳方法是重新加载整个模型和适配器检查点。确保您不对模型执行任何错误操作，例如手动合并某些模块上的适配器，但不合并其他模块上的适配器。

将图层状态转换为 pandas `DataFrame`，以便于目视检查。

```python
from dataclasses import asdict
import pandas as pd

df = pd.DataFrame(asdict(layer) for layer in model.get_layer_status())
```

如果非 PEFT 模型在底层使用 PEFT 层，则可以获得此信息，但在这种情况下无法确定一些信息，如 `base_model_type` 或 `peft_types`。例如，您可以在 [diffusers](https://huggingface.co/docs/diffusers/index) 模型上调用它，如下所示：

```python
>>> import torch
>>> from diffusers import StableDiffusionPipeline
>>> from peft import get_model_status, get_layer_status

>>> path = "runwayml/stable-diffusion-v1-5"
>>> lora_id = "takuma104/lora-test-text-encoder-lora-target"
>>> pipe = StableDiffusionPipeline.from_pretrained(path, dtype=torch.float16)
>>> pipe.load_lora_weights(lora_id, adapter_name="adapter-1")
>>> pipe.load_lora_weights(lora_id, adapter_name="adapter-2")
>>> pipe.set_lora_device(["adapter-2"], "cuda")
>>> get_layer_status(pipe.text_encoder)
[TunerLayerStatus(name='text_model.encoder.layers.0.self_attn.k_proj',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['adapter-2'],
                  merged_adapters=[],
                  requires_grad={'adapter-1': False, 'adapter-2': True},
                  available_adapters=['adapter-1', 'adapter-2'],
                  devices={'adapter-1': ['cpu'], 'adapter-2': ['cuda']}),
 TunerLayerStatus(name='text_model.encoder.layers.0.self_attn.v_proj',
                  module_type='lora.Linear',
                  enabled=True,
                  active_adapters=['adapter-2'],
                  merged_adapters=[],
                  requires_grad={'adapter-1': False, 'adapter-2': True},
                  devices={'adapter-1': ['cpu'], 'adapter-2': ['cuda']}),
...]

>>> get_model_status(pipe.unet)
TunerModelStatus(
    base_model_type='other',
    adapter_model_type='None',
    peft_types={},
    trainable_params=797184,
    total_params=861115332,
    num_adapter_layers=128,
    enabled=True,
    active_adapters=['adapter-2'],
    merged_adapters=[],
    requires_grad={'adapter-1': False, 'adapter-2': True},
    available_adapters=['adapter-1', 'adapter-2'],
    devices={'adapter-1': ['cpu'], 'adapter-2': ['cuda']},
)
```

## 速度

### 加载适配器权重很慢

与加载基本模型相比，加载 LoRA 权重等适配器通常应该更快。然而，在某些用例中，适配器的权重非常大，或者用户需要加载大量适配器——在这种情况下，加载时间可能会增加。这样做的原因是适配器权重首先被初始化，然后被加载的权重覆盖，这是浪费的。为了加快加载时间，您可以将 `low_cpu_mem_usage=True` 参数传递给 [from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained) 和 [load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)。> [!提示]
> 如果此选项在不同的用例中运行良好，它可能会成为将来适配器加载的默认选项。

## 再现性

### 使用批量归一化的模型

当加载基本模型使用批量规范（例如`torch.nn.BatchNorm1d`或`torch.nn.BatchNorm2d`）的经过训练的PEFT模型时，您可能会发现无法重现完全相同的输出。这是因为批标准化层在训练期间跟踪运行统计数据，但这些统计数据不是 PEFT 检查点的一部分。因此，当您加载 PEFT 模型时，将使用基础模型的运行统计数据（即使用 PEFT 训练之前的统计数据）。

根据您的用例，这可能不是什么大问题。但是，如果您需要输出 100% 可重复，则可以通过将批量规范层添加到 `modules_to_save` 来实现此目的。下面是使用 resnet 和 LoRA 的示例。请注意，我们设置了`modules_to_save=["classifier", "normalization"]`。我们需要 `"classifier"` 参数，因为我们的任务是图像分类，并且我们添加 `"normalization"` 参数以确保批量规范层保存在 PEFT 检查点中。

```python
from transformers import AutoModelForImageClassification
from peft import LoraConfig, get_peft_model

model_id = "microsoft/resnet-18"
base_model = AutoModelForImageClassification.from_pretrained(self.model_id)
config = LoraConfig(
    target_modules=["convolution"],
    modules_to_save=["classifier", "normalization"],
),
```

根据您使用的模型类型，批量归一化层的名称可能与`"normalization"`不同，因此请确保该名称与您的模型架构相匹配。## 版本不匹配

### 由于意外的关键字参数而加载配置时出错

当您遇到如下所示的错误时，这意味着您尝试加载的适配器是使用比您系统上安装的版本更新的 PEFT 版本进行训练的。

```
TypeError: LoraConfig.__init__() got an unexpected keyword argument <argument-name>
```

解决此问题的最佳方法是安装最新的 PEFT 版本：

```sh
python -m pip install -U PEFT
```

如果适配器是通过 PEFT 源安装（PEFT 的未发布版本）进行训练的，那么您还需要从源安装 PEFT。

```sh
python -m pip install -U git+https://github.com/huggingface/peft.git
```

如果您无法升级 PEFT，您可以尝试一种解决方法。

假设错误消息显示未知关键字参数名为 `foobar`。在此 PEFT 适配器的 `adapter_config.json` 内部搜索 `foobar` 条目并将其从文件中删除。然后保存文件并尝试再次加载模型。

该解决方案在大多数情况下都有效。只要是`foobar`的默认值，就可以忽略。但是，当它设置为其他值时，您将得到不正确的结果。升级 PEFT 是推荐的解决方案。

## 适配器处理

### 同时使用多个适配器PEFT 允许您在同一模型上创建多个适配器。这在很多情况下都很有用。例如，为了进行推理，您可能希望从同一基本模型提供两个微调模型，而不是为每个微调模型加载一次基本模型，这会消耗更多内存。但是，可以同时激活多个适配器。这样，模型可以同时利用所有这些适配器的学习成果。举例来说，如果您有一个扩散模型，您可能需要使用一个 LoRA 适配器来更改风格，并使用另一个适配器来更改主题。

除了快速学习方法（p-tuning、前缀调谐等）之外，所有 PEFT 方法（LoRA、LoHa、IA³ 等）通常都可以同时激活多个适配器。以下示例说明了如何实现此目的：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

model_id = ...
base_model = AutoModelForCausalLM.from_pretrained(model_id)
model = PeftModel.from_pretrained(base_model, lora_path_0)  # default adapter_name is 'default'
model.load_adapter(lora_path_1, adapter_name="other")
# the 'other' adapter was loaded but it's not active yet, so to activate both adapters:
model.base_model.set_adapter(["default", "other"])
```

> [!提示]
> 在上面的例子中，你可以看到我们需要调用`model.base_model.set_adapter(["default", "other"])`。为什么我们不能拨打`model.set_adapter(["default", "other"])`？遗憾的是，这是不可能的，因为如前所述，某些 PEFT 方法不支持一次激活多个适配器。也可以同时训练两个适配器，但您应该小心确保优化器知道两个适配器的权重。否则，只有一个适配器会收到更新。

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

model_id = ...
base_model = AutoModelForCausalLM.from_pretrained(model_id)
lora_config_0 = LoraConfig(...)
lora_config_1 = LoraConfig(...)
model = get_peft_model(base_model, lora_config_0)
model.add_adapter(adapter_name="other", peft_config=lora_config_1)
```

如果我们现在调用：

```python
from transformers import Trainer

trainer = Trainer(model=model,  ...)
trainer.train()
```

或

```python
optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], ...)
```

那么第二个 LoRA 适配器 (`"other"`) 将不会被训练。这是因为此时它处于非活动状态，这意味着其参数上的`requires_grad`属性被设置为`False`，优化器将忽略它。因此，请确保在初始化优化器之前激活应训练的所有适配器：

```python
# activate all adapters
model.base_model.set_adapter(["default", "other"])
trainer = Trainer(model=model,  ...)
trainer.train()
```

> [!提示]
> 本节讨论在同一模型上使用_相同类型_的多个适配器，例如，同时使用多个 LoRA 适配器。它不适用于在同一型号上使用_不同类型_的适配器，例如一个 LoRA 适配器和一个 LoHa 适配器。为此，请检查[⟦T86⟧](https://huggingface.co/docs/peft/developer_guides/mixed_models)。

### 混合适配器类型
https://huggingface.co/docs/peft/v0.20.0/developer_guides/mixed_models.md