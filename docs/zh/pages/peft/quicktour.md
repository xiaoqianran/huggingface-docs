<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 快速游览

PEFT 提供参数有效的方法来微调大型预训练模型。传统的范例是针对每个下游任务微调模型的所有参数，但由于当今模型中的参数数量巨大，这变得非常昂贵且不切实际。相反，训练较少数量的提示参数或使用低秩自适应（LoRA）等重新参数化方法来减少可训练参数的数量会更有效。

  
    PEFT 可以被认为是向现有模型（“基础模型”）中的任意位置添加可训练参数的框架。特定的 PEFT 方法以某种方式排列可训练参数或修改训练过程，以达到与训练基础模型的所有参数相当的微调性能。
  
  

本快速浏览将向您展示 PEFT 的主要功能，以及如何在消费设备上通常无法访问的大型模型上训练或运行推理。

## PEFT配置和模型对于任何 PEFT 方法，您需要创建一个配置，其中包含指定如何应用 PEFT 方法的所有参数，最重要的是使用可训练参数定位现有模型的哪些层。设置配置后，将其与基本模型一起传递给 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 函数，以创建可训练的 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)。

让我们使用 [LoRA](./package_reference/lora) 作为示例，但仅讨论常用参数 - 您可能想使用 [many other PEFT methods](./methods/overview) 之一。
配置通常需要这样：

- `target_modules`：适配基础模型的哪些模块
- `task_type`（默认：`None`，参见[available ⟦T14⟧s](package_reference/peft_types#peft.TaskType)）：训练任务的性质；如果提供，可能有助于自动保存相关层以及适配器权重，或警告您不兼容的情况
- `inference_mode`（默认：`False`）：是否使用模型进行推理

根据您选择的 PEFT 方法，您将添加特定参数，例如确定更新矩阵的大小。
以下是您可能在野外遇到的配置示例：

```python
from peft import LoraConfig, TaskType

peft_config = LoraConfig(target_modules=["q_proj"], task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
```

> [!提示]
> 有关 PEFT 配置如何在后台工作的更多详细信息，请参阅 [configuration guide](guides/peft_model_config)。设置 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 后，使用 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 函数创建 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)。它需要一个基本模型 - 您可以（但不必）从 Transformers 库加载 - 以及包含如何配置模型以使用 LoRA 进行训练的参数的 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)。

加载您想要微调的基本模型。

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
```

现在使用 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 函数包装基本模型和 `peft_config` 以创建一个 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)。

  
    
    包装意味着 PEFT 将目标层（此处：所有 q_proj 层）替换为目标层类型的适配器特定层。  由于我们正在处理线性层，因此在本例中它将是 lora.Linear 层。请注意，这些更改是就地完成的，以节省内存，因此您的基本模型现在已修改。
    
    
    请注意，我们只指定了 q_proj，但实际上我们的目标是所有 model.layers[:].self_attn.q_proj 层。这是因为 PEFT 默认情况下搜索匹配的后缀。如果您想定位更复杂的图层模式，请传递带有正则表达式的字符串。基础模型的层将被包装、保留且不进行训练，同时添加并组合新的可训练权重。  这些新权重的构造方式以及如何与基本模型的权重相结合是不同 PEFT 方法的独特之处。
    
  

要了解模型中可训练参数的数量，请使用 `print_trainable_parameters` 方法。

```python
from peft import get_peft_model

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters()
"output: trainable params: 524,288 || all params: 1,236,338,688 || trainable%: 0.0424"
```

在 [meta-llama/Llama-3.2-1B's](https://huggingface.co/meta-llama/Llama-3.2-1B) 1B 个参数中，您只训练了其中的 0.04%！

就是这样🎉！现在，您可以使用 Transformers [Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer)、Accelerate 或任何自定义 PyTorch 训练循环来训练模型。

例如，要使用 [Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer) 类进行训练，请设置带有一些训练超参数的 [TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments) 类。

```py
training_args = TrainingArguments(
    output_dir="your-name/meta-llama/my-llama3.2-adapter",
    learning_rate=1e-3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=2,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)
```

将模型、训练参数、数据集、分词器和任何其他必要组件传递给[Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer)，并调用[train](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer.train)开始训练。

```py
trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
```

### 保存模型

模型训练完成后，您可以使用[save_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.save_pretrained)功能将模型保存到目录中。

```py
peft_model.save_pretrained("output_dir")
```

您还可以使用 [push_to_hub](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub) 功能将模型保存到 Hub（确保您先登录 Hugging Face 帐户）。

```python
from huggingface_hub import notebook_login

notebook_login()
peft_model.push_to_hub("your-name/my-llama3.2-adapter")
```两种方法都只保存训练过的额外 PEFT 权重，这意味着存储、传输和加载非常高效。例如，这个用LoRA训练的[facebook/opt-350m](https://huggingface.co/ybelkada/opt-350m-lora)模型只包含两个文件：`adapter_config.json`和`adapter_model.safetensors`。 `adapter_model.safetensors` 文件只有 6.3MB！

  
  与模型权重的完整大小（约 700MB）相比，存储在集线器上的 opt-350m 模型的适配器权重仅为约 6MB。

## 推论

> [!提示]
> 查看 [AutoPeftModel](package_reference/auto_class) API 参考，获取可用 `AutoPeftModel` 类的完整列表。

使用 [AutoPeftModel](/docs/peft/v0.20.0/en/package_reference/auto_class#peft.AutoPeftModel) 类和 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) 方法轻松加载任何经过 PEFT 训练的模型进行推理：

```py
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch

peft_model = AutoPeftModelForCausalLM.from_pretrained("ybelkada/opt-350m-lora")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

peft_model = peft_model.to("cuda")
peft_model.eval()
inputs = tokenizer("Preheat the oven to 350 degrees and place the cookie dough", return_tensors="pt")

outputs = peft_model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=50)
print(tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0])

"Preheat the oven to 350 degrees and place the cookie dough in the center of the oven. In a large bowl, combine the flour, baking powder, baking soda, salt, and cinnamon. In a separate bowl, combine the egg yolks, sugar, and vanilla."
```

对于 `AutoPeftModelFor` 类未明确支持的其他任务（例如自动语音识别），您仍然可以使用基本 [AutoPeftModel](/docs/peft/v0.20.0/en/package_reference/auto_class#peft.AutoPeftModel) 类来加载该任务的模型。

```py
from peft import AutoPeftModel

peft_model = AutoPeftModel.from_pretrained("smangrul/openai-whisper-large-v2-LORA-colab")
```

将经过训练的 PEFT 适配器加载到模型上的最通用方法是使用 [from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained)：

```py
from transformers import AutoPeftModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/llama3.2-1B")
peft_model = PeftModel.from_pretrained(base_model, "my-user/my-llama-adapter")  # you can also pass a directory instead of a hub path
```

## 多个适配器

PEFT 支持在基本模型之上安装多个适配器（同类适配器，在本文档中为 LoRA）。当您调用 `get_peft_model` 时，只有一个名为 `"default"` 的适配器，但您可以通过调用 `peft_model.add_adapter(adapter_name=...)` 添加任意数量的其他适配器。这是有效的，因为包装层实际上为每个适配器名称都有一组唯一的可训练权重。默认情况下，并非每个适配器都是活动的且可训练的。  您必须在适配器处于活动状态之前按名称显式启用它们。这使您可以在需要特定任务知识的适配器之间快速切换，或者在一个模型之上为不同的用例提供服务。
    
  
  

请记住先调用`peft_model.set_adapter(<adapter_name>)`来启用适配器。

快速示例：

```py
peft_model.add_adapter(adapter_name='new_adapter')
peft_model.set_adapter('new_adapter')
```

## 后续步骤

现在您已经了解了如何使用其中一种 PEFT 方法训练模型，我们鼓励您尝试其他一些方法，例如提示调整。这些步骤与快速教程中显示的步骤非常相似：

1. 准备用于 PEFT 方法的[PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)，例如[LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 或其他一些配置（请参阅[method overview](methods/overview)）
2.使用[get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model)方法从配置和基础模型创建[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)

然后你就可以随心所欲地训练它了！要加载 PEFT 模型进行推理，您可以使用 [AutoPeftModel](/docs/peft/v0.20.0/en/package_reference/auto_class#peft.AutoPeftModel) 类。如果您有兴趣使用另一种 PEFT 方法训练模型以执行特定任务（例如语义分割、多语言自动语音识别、DreamBooth、标记分类等），请随意查看任务指南。

### PEFT
https://huggingface.co/docs/peft/v0.20.0/index.md