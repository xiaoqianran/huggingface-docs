<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PEFT 配置和模型

当今大型预训练模型的庞大规模（通常具有数十亿个参数）带来了巨大的训练挑战，因为它们需要更多的存储空间和更多的计算能力来处理所有这些计算。您需要使用强大的 GPU 或 TPU 来训练这些大型预训练模型，但这些模型价格昂贵、并非每个人都能广泛使用、不环保且不太实用。 PEFT 方法解决了其中许多挑战。 PEFT 方法有多种类型（软提示、矩阵分解、适配器），但它们都专注于同一件事，减少可训练参数的数量。这使得在消费类硬件上训练和存储大型模型变得更容易。

PEFT 库旨在帮助您在免费或低成本 GPU 上快速训练大型模型，在本教程中，您将学习如何设置配置以将 PEFT 方法应用于预训练的基础模型进行训练。设置 PEFT 配置后，您可以使用您喜欢的任何训练框架（Transformer 的 [Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer) 类、[Accelerate](https://hf.co/docs/accelerate)、自定义 PyTorch 训练循环）。

## PEFT 配置> [!提示]
> 在各自的 API 参考页面中了解有关可为每个 PEFT 方法配置的参数的更多信息。

配置存储指定如何应用特定 PEFT 方法的重要参数。

例如，看一下以下用于应用LoRA的`LoraConfig`和用于应用p-tuning的`PromptEncoderConfig`（这些配置文件已经是JSON序列化的）。每当加载 PEFT 适配器时，最好检查它是否具有所需的关联 `adapter_config.json` 文件。

```json
{
  "base_model_name_or_path": "facebook/opt-350m", #base model to apply LoRA to
  "bias": "none",
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layers_pattern": null,
  "layers_to_transform": null,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "modules_to_save": null,
  "peft_type": "LORA", #PEFT method type
  "r": 16,
  "revision": null,
  "target_modules": [
    "q_proj", #model modules to apply LoRA to (query and value projection layers)
    "v_proj"
  ],
  "task_type": "CAUSAL_LM" #type of task to train model on
}
```

您可以通过初始化[LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)来创建自己的训练配置。

```py
from peft import LoraConfig, TaskType

lora_config = LoraConfig(
    r=16,
    target_modules=["q_proj", "v_proj"],
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=32,
    lora_dropout=0.05
)
```

```json
{
  "base_model_name_or_path": "roberta-large", #base model to apply p-tuning to
  "encoder_dropout": 0.0,
  "encoder_hidden_size": 128,
  "encoder_num_layers": 2,
  "encoder_reparameterization_type": "MLP",
  "inference_mode": true,
  "num_attention_heads": 16,
  "num_layers": 24,
  "num_transformer_submodules": 1,
  "num_virtual_tokens": 20,
  "peft_type": "P_TUNING", #PEFT method type
  "task_type": "SEQ_CLS", #type of task to train model on
  "token_dim": 1024
}
```

您可以通过初始化[PromptEncoderConfig](/docs/peft/v0.20.0/en/package_reference/p_tuning#peft.PromptEncoderConfig)来创建自己的训练配置。

```py
from peft import PromptEncoderConfig, TaskType

p_tuning_config = PromptEncoderConfig(
    encoder_reparameterization_type="MLP",
    encoder_hidden_size=128,
    num_attention_heads=16,
    num_layers=24,
    num_transformer_submodules=1,
    num_virtual_tokens=20,
    token_dim=1024,
    task_type=TaskType.SEQ_CLS
)
```

## PEFT 模型

有了 PEFT 配置，您现在可以将其应用于任何预训练模型以创建 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)。从 [Transformers](https://hf.co/docs/transformers) 库中的任何最先进的模型、自定义模型，甚至是新的且不受支持的变压器架构中进行选择。

在本教程中，加载基本 [facebook/opt-350m](https://huggingface.co/facebook/opt-350m) 模型进行微调。

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```

使用 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 函数从基本 facebook/opt-350m 模型和您之前创建的 `lora_config` 创建一个 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)。

```py
from peft import get_peft_model

lora_model = get_peft_model(model, lora_config)
lora_model.print_trainable_parameters()
"trainable params: 1,572,864 || all params: 332,769,280 || trainable%: 0.472659014678278"
```> [!警告]
> 当调用[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)时，基础模型将被*就地*修改。这意味着，当在之前已经以相同方式修改过的模型上调用[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)时，该模型将进一步变异。因此，如果您想在调用`get_peft_model()`之后修改PEFT配置，则首先必须使用[unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.unload)卸载模型，然后使用新配置调用`get_peft_model()`。或者，您可以在应用新的 PEFT 配置之前重新初始化模型，以确保处于全新的、未修改的状态。

现在您可以使用您喜欢的训练框架来训练[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)！训练完成后，您可以使用[save_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.save_pretrained)将模型保存在本地，或者使用[push_to_hub](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)方法将其上传到Hub。

```py
# save locally
lora_model.save_pretrained("your-name/opt-350m-lora")

# push to Hub
lora_model.push_to_hub("your-name/opt-350m-lora")
```

要加载 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 进行推理，您需要提供用于创建它的 [PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig) 以及训练它的基础模型。

```py
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

> [!提示]
> 默认情况下，[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 设置为推理，但如果您想进一步训练适配器，可以设置 `is_trainable=True`。
>
> ```py
> lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora", is_trainable=True)
> ```[PeftModel.from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained) 方法是加载 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 最灵活的方法，因为使用什么模型框架（Transformers、timm、通用 PyTorch 模型）并不重要。其他类，如 [AutoPeftModel](/docs/peft/v0.20.0/en/package_reference/auto_class#peft.AutoPeftModel)，只是基础 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 的方便包装，并且可以更轻松地直接从 Hub 或存储 PEFT 权重的本地加载 PEFT 模型。

```py
from peft import AutoPeftModelForCausalLM

lora_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
```

查看 [AutoPeftModel](../package_reference/auto_class) API 参考以了解有关 [AutoPeftModel](/docs/peft/v0.20.0/en/package_reference/auto_class#peft.AutoPeftModel) 类的更多信息。

## 后续步骤

借助适当的 [PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)，您可以将其应用于任何预训练模型来创建 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)，并在免费提供的 GPU 上更快地训练大型强大模型！要了解有关 PEFT 配置和模型的更多信息，以下指南可能会有所帮助：

* 在 [Working with custom models](../developer_guides/custom_models) 指南中了解如何为非 Transformers 的模型配置 PEFT 方法。

### 火炬.编译
https://huggingface.co/docs/peft/v0.20.0/developer_guides/torch_compile.md