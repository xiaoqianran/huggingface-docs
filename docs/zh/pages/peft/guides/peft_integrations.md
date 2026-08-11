<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PEFT 集成

PEFT 的实际好处扩展到其他 Hugging Face 库，例如 [Diffusers](https://hf.co/docs/diffusers) 和 [Transformers](https://hf.co/docs/transformers)。 PEFT 的主要优点之一是 PEFT 方法生成的适配器文件比原始模型小很多，这使得管理和使用多个适配器变得非常容易。您只需加载针对您正在解决的任务进行微调的新适配器，即可将一个预训练的基本模型用于多项任务。或者，您可以将多个适配器与文本到图像扩散模型结合起来以创建新的效果。

本教程将向您展示 PEFT 如何帮助您管理扩散器和变压器中的适配器。

## 扩散器

Diffusers 是一个生成式 AI 库，用于使用扩散模型从文本或图像创建图像和视频。 LoRA 是一种特别流行的扩散模型训练方法，因为您可以非常快速地训练和共享扩散模型以生成新风格的图像。为了更轻松地使用和尝试多个 LoRA 模型，Diffusers 使用 PEFT 库来帮助管理不同的适配器以进行推理。例如，加载基础模型，然后加载[artificialguybr/3DRedmond-V1](https://huggingface.co/artificialguybr/3DRedmond-V1)适配器以使用[⟦T11⟧](https://huggingface.co/docs/diffusers/v0.24.0/en/api/loaders/lora#diffusers.loaders.LoraLoaderMixin.load_lora_weights)方法进行推理。加载方法中的 `adapter_name` 参数由 PEFT 启用，并允许您为适配器设置名称，以便更容易引用。

```py
import torch
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "peft-internal-testing/artificialguybr__3DRedmond-V1", 
    weight_name="3DRedmond-3DRenderStyle-3DRenderAF.safetensors", 
    adapter_name="3d"
)
image = pipeline("sushi rolls shaped like kawaii cat faces").images[0]
image
```

    

现在让我们尝试另一个很酷的 LoRA 模型，[ostris/super-cereal-sdxl-lora](https://huggingface.co/ostris/super-cereal-sdxl-lora)。您需要做的就是加载这个新适配器并将其命名为 `adapter_name`，然后使用 [⟦T14⟧](https://huggingface.co/docs/diffusers/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.set_adapters) 方法将其设置为当前活动适配器。

```py
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora", 
    weight_name="cereal_box_sdxl_v1.safetensors", 
    adapter_name="cereal"
)
pipeline.set_adapters("cereal")
image = pipeline("sushi rolls shaped like kawaii cat faces").images[0]
image
```

    

最后可以调用[⟦T15⟧](https://huggingface.co/docs/diffusers/api/loaders/unet#diffusers.loaders.UNet2DConditionLoadersMixin.disable_lora)方法恢复基础模型。

```py
pipeline.disable_lora()
```

在 [Inference with PEFT](https://huggingface.co/docs/diffusers/tutorials/using_peft_for_inference) 教程中了解有关 PEFT 如何支持扩散器的更多信息。

## 变形金刚

🤗 [Transformers](https://hf.co/docs/transformers) 是适用于所有模式的所有类型任务的预训练模型的集合。您可以加载这些模型进行训练或推理。许多模型都是大型语言模型 (LLM)，因此将 PEFT 与 Transformer 集成来管理和训练适配器是有意义的。

加载基本预训练模型进行训练。

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
```

接下来，添加适配器配置以指定如何调整模型参数。调用[add_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.add_adapter)方法将配置添加到基础模型中。

```py
from peft import LoraConfig

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)
model.add_adapter(peft_config)
```

现在您可以使用 Transformer 的 [Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer) 类或您喜欢的任何训练框架来训练模型。为了使用新训练的模型进行推理，[AutoModel](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModel)类在后端使用 PEFT 将适配器权重和配置文件加载到基础预训练模型中。

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("peft-internal-testing/opt-350m-lora")
```

或者，您可以使用变压器[Pipelines](https://huggingface.co/docs/transformers/en/main_classes/pipelines)来加载模型以方便运行推理：

```py
from transformers import pipeline

model = pipeline("text-generation", "peft-internal-testing/opt-350m-lora")
print(model("Hello World"))
```

如果您有兴趣比较或使用多个适配器，可以调用 [add_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.add_adapter) 方法将适配器配置添加到基础模型中。唯一的要求是适配器类型必须相同（不能混合使用 LoRA 和 LoHa 适配器）。

```py
from transformers import AutoModelForCausalLM
from peft import LoraConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
model.add_adapter(lora_config_1, adapter_name="adapter_1")
```

再次调用 [add_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.add_adapter) 将新适配器连接到基础模型。

```py
model.add_adapter(lora_config_2, adapter_name="adapter_2")
```

然后你可以使用[set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.set_adapter)来设置当前活动的适配器。

```py
model.set_adapter("adapter_1")
output = model.generate(**inputs)
print(tokenizer.decode(output_disabled[0], skip_special_tokens=True))
```

要禁用适配器，请调用 [disable_adapters](https://github.com/huggingface/transformers/blob/4e3490f79b40248c53ee54365a9662611e880892/src/transformers/integrations/peft.py#L313) 方法。

```py
model.disable_adapters()
```

[enable_adapters](https://github.com/huggingface/transformers/blob/4e3490f79b40248c53ee54365a9662611e880892/src/transformers/integrations/peft.py#L336) 可用于再次启用适配器。

如果您好奇，请查看 [Load and train adapters with PEFT](https://huggingface.co/docs/transformers/main/peft) 教程以了解更多信息。

### PEFT 配置和模型
https://huggingface.co/docs/peft/v0.20.0/guides/peft_model_config.md