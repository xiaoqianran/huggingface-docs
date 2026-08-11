<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型合并

为每项任务训练模型可能成本高昂，占用存储空间，并且模型无法学习新信息来提高其性能。多任务学习可以通过训练模型来学习多个任务来克服其中的一些限制，但训练成本很高，并且为此设计数据集具有挑战性。 *模型合并*通过将多个预训练模型组合成一个模型，为这些挑战提供了一种解决方案，无需任何额外的训练即可赋予每个模型的综合能力。

PEFT 提供了多种合并模型的方法，例如线性或 SVD 组合。本指南重点介绍两种通过消除冗余参数来更有效地合并 LoRA 适配器的方法：* [TIES](https://hf.co/papers/2306.01708) - TrIm、Elect 和 Merge (TIES) 是一种合并模型的三步方法。首先，修剪冗余参数，然后将冲突符号分解为聚合向量，最后对符号与聚合符号相同的参数进行平均。此方法考虑到某些值（冗余和符号不一致）可能会降低合并模型的性能。
* [DARE](https://hf.co/papers/2311.03099) - Drop And REscale 是一种可用于为 TIES 等其他模型合并方法做准备的方法。它的工作原理是根据丢弃率随机丢弃参数并重新调整剩余参数。这有助于减少多个模型之间冗余和潜在干扰参数的数量。

模型合并采用[add_weighted_adapter()](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraModel.add_weighted_adapter)方法，具体模型合并方法在`combination_type`参数中指定。

## 合并方法

对于 TIES 和 DARE，通过将 `combination_type` 和 `density` 设置为远离各个模型的权重值来启用合并。例如，让我们合并三个微调的 [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) 模型：[tinyllama_lora_nobots](https://huggingface.co/smangrul/tinyllama_lora_norobots)、[tinyllama_lora_sql](https://huggingface.co/smangrul/tinyllama_lora_sql) 和 [tinyllama_lora_adcopy](https://huggingface.co/smangrul/tinyllama_lora_adcopy)。当您尝试将完全训练的模型与 TIES 合并时，您应该注意每个模型可能添加到嵌入层的任何特殊标记，这些标记不属于原始检查点词汇表的一部分。这可能会导致问题，因为每个模型可能都在同一嵌入位置添加了特殊标记。如果是这种情况，您应该使用 [resize_token_embeddings](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings) 方法来避免合并同一嵌入索引处的特殊标记。

如果您只是合并从同一基本模型训练的 LoRA 适配器，这应该不是问题。

加载基础模型，并可以使用 [load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter) 方法加载每个适配器并为其分配名称：

```py
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

config = PeftConfig.from_pretrained("smangrul/tinyllama_lora_norobots")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, load_in_4bit=True, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained("smangrul/tinyllama_lora_norobots")

model.config.vocab_size = 32005
model.resize_token_embeddings(32005)

model = PeftModel.from_pretrained(model, "smangrul/tinyllama_lora_norobots", adapter_name="norobots")
_ = model.load_adapter("smangrul/tinyllama_lora_sql", adapter_name="sql")
_ = model.load_adapter("smangrul/tinyllama_lora_adcopy", adapter_name="adcopy")
```

使用 [add_weighted_adapter()](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraModel.add_weighted_adapter) 方法设置适配器、权重、`adapter_name`、`combination_type` 和 `density`。

大于 `1.0` 的权重值通常会产生更好的结果，因为它们保留了正确的比例。权重的一个很好的默认起始值​​是将所有值设置为 `1.0`。

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 1.0, 1.0]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="ties", density=density)
```

```py
adapters = ["norobots", "adcopy", "sql"]
weights = [2.0, 0.3, 0.7]
adapter_name = "merge"
density = 0.2
model.add_weighted_adapter(adapters, weights, adapter_name, combination_type="dare_ties", density=density)
```

使用 [set_adapter()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.set_adapter) 方法将新合并的模型设置为活动模型。

```py
model.set_adapter("merge")
```

现在，您可以使用合并模型作为指令调整模型来编写广告文案或 SQL 查询！

```py
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
messages = [
    {"role": "user", "content": "Write an essay about Generative AI."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

```py
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
messages = [
    {"role": "system", "content": "Create a text ad given the following product and description."},
    {"role": "user", "content": "Product: Sony PS5 PlayStation Console\nDescription: The PS5 console unleashes new gaming possibilities that you never anticipated."},
]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=128, do_sample=True, top_p=0.95, temperature=0.2, repetition_penalty=1.2, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

```py
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

text = """Table: 2-11365528-2
Columns: ['Team', 'Head Coach', 'President', 'Home Ground', 'Location']
Natural Query: Who is the Head Coach of the team whose President is Mario Volarevic?
SQL Query:"""

inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1, eos_token_id=tokenizer("</s>").input_ids[-1])
print(tokenizer.decode(outputs[0]))
```## 合并 (IA)³ 模型
(IA)³ 模型有助于适配器的线性合并。要合并 (IA)³ 模型中的适配器，请使用 `IA3Model` 类中的 `add_weighted_adapter` 方法。此方法类似于 `LoraModel` 中使用的 `add_weighted_adapter` 方法，主要区别在于缺少 `combination_type` 参数。例如，要将三个 (IA)³ 适配器合并到 PEFT 模型中，您可以按以下步骤操作：

```py
adapters = ["adapter1", "adapter2", "adapter3"]
weights = [0.4, 0.3, 0.3]
adapter_name = "merge"
model.add_weighted_adapter(adapters, weights, adapter_name)
```

建议权重总和为 1.0 以保持模型的规模。然后可以使用 `set_adapter` 方法将合并的模型设置为活动模型：

```py
model.set_adapter("merge")
```

### 定制模型
https://huggingface.co/docs/peft/v0.20.0/developer_guides/custom_models.md