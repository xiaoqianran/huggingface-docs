<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 混合适配器类型

通常，在🤗 PEFT 中混合不同类型的适配器是不可能的。您可以使用两个不同的 LoRA 适配器（可以有不同的配置选项）创建 PEFT 模型，但无法组合 LoRA 和 LoHa 适配器。然而，对于 [PeftMixedModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftMixedModel)，只要适配器类型兼容，这就可以工作。允许混合适配器类型的主要目的是组合经过训练的适配器进行推理。虽然可以训练混合适配器模型，但尚未经过测试，因此不建议这样做。

要将不同的适配器类型加载到 PEFT 模型中，请使用 [PeftMixedModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftMixedModel) 而不是 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)：

```py
from peft import PeftMixedModel

base_model = ...  # load the base model, e.g. from transformers
# load first adapter, which will be called "default"
peft_model = PeftMixedModel.from_pretrained(base_model, <path_to_adapter1>)
peft_model.load_adapter(<path_to_adapter2>, adapter_name="other")
peft_model.set_adapter(["default", "other"])
```

[set_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftMixedModel.set_adapter) 方法对于激活两个适配器是必需的，否则只有第一个适配器会被激活。您可以通过重复调用[add_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.add_adapter)来不断添加更多适配器。

[PeftMixedModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftMixedModel)不支持保存和加载混合适配器。适配器应该已经经过训练，并且加载模型需要每次运行一个脚本。

＃＃ 尖端- 并非所有适配器类型都可以组合。有关兼容类型的列表，请参阅[⟦T1⟧](https://github.com/huggingface/peft/blob/1c1c7fdaa6e6abaa53939b865dee1eded82ad032/src/peft/tuners/mixed/model.py#L35)。如果您尝试组合不兼容的适配器类型，将会引发错误。
- 可以混合相同类型的多个适配器，这对于组合具有不同配置的适配器非常有用。
- 如果您想要组合许多不同的适配器，最有效的方法是连续添加相同的适配器类型。例如，按此顺序添加 LoRA1、LoRA2、LoHa1、LoHa2，而不是 LoRA1、LoHa1、LoRA2 和 LoHa2。虽然顺序会影响输出，但本质上不存在“最佳”顺序，因此最好选择最快的顺序。

### 记忆高效训练
https://huggingface.co/docs/peft/v0.20.0/developer_guides/memory_efficient_training.md