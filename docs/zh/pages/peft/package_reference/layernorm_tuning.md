<!-- huggingface-docs: machine-translated zh-CN from English source -->

# LayerNorm 调整

LayerNorm Tuning ([LN Tuning](https://huggingface.co/papers/2312.11420)) 是一种 PEFT 方法，仅微调模型中 LayerNorm 层的参数。
论文在大型语言模型上测试了该方法的性能，并表明该方法可以通过显着减少可训练参数的数量和 GPU 内存使用来实现强大的性能。
然而，该方法不限于语言模型，并且可以应用于使用LayerNorm层的任何模型。
在此实现中，默认情况下模型内的所有 Layernorm 层都会进行微调，但它可以用于定位其他层类型，例如 `MLP` 或 `Attention` 层，这可以通过在 `LNTuningConfig` 中指定 `target_modules` 来完成。

论文摘要是：*本文介绍了一种将大型语言模型（LLM）转换为多模态大型语言模型（MLLM）的有效策略。通过将这种转变概念化为领域适应过程，即从文本理解过渡到拥抱多种模式，我们有趣地注意到，在每个注意力块内，调整 LayerNorm 足以产生强大的性能。此外，当与全参数微调或 LoRA 等其他调整方法进行基准测试时，它对效率的好处是巨大的。例如，与 13B 模型规模的 LoRA 相比，在 5 个多模态任务中，性能平均提升超过 20%，同时可训练参数显着减少 41.9%，GPU 内存使用量减少 17.6%。在此 LayerNorm 策略之上，我们展示了仅使用会话数据进行选择性调整可以进一步提高效率。除了这些实证结果之外，我们还提供了全面的分析，以探索 LayerNorm 在使 LLM 适应多模态领域并提高模型表达能力方面的作用。*

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LN_TUNING"
	frameborder="0"
	width="850"
	height="1000"
>

# API## LNTuningConfig[[peft.LNTuningConfig]]

#### pft.LNTuningConfig[[peft.LNTuningConfig]]

```python
peft.LNTuningConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, modules_to_save: Optional[Union[list[str], str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/ln_tuning/config.py#L24)

**参数：**

target_modules (*可选[Union[List[str], str]]*) ：要替换为 LNTuning 的模块名称列表或模块名称的正则表达式。例如，“.*解码器。*”或“.*编码器。*”。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (*Optional[Union[List[str], str]]*) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

module_to_save (*可选[Union[List[str], str]]*) ：要设置为可训练并保存在最终检查点中的模块列表。例如，在序列分类或令牌分类任务中，最后一层*分类器/分数*是随机初始化的，因此需要可训练和保存。这是存储[LNTuningModel](/docs/peft/v0.20.0/en/package_reference/layernorm_tuning#peft.LNTuningModel)配置的配置类。

## LNTuningModel[[peft.LNTuningModel]]

#### peft.LNTuningModel[[peft.LNTuningModel]]

```python
peft.LNTuningModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/ln_tuning/model.py#L28)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([LNTuningConfig](/docs/peft/v0.20.0/en/package_reference/layernorm_tuning#peft.LNTuningConfig)) : LN 调优模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`、`optional`，默认为`False`）：此选项对 LN 调整没有影响，但存在是为了与其他 PEFT 方法保持一致。

**返回：** `'torch.nn.Module'`

调整后的模型已启用 LayerNorm。

从预训练的变压器模型创建 LayerNorm 调整。

该方法在 https://huggingface.co/papers/2312.11420 中有详细描述。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import get_peft_model, TaskType, LNTuningConfig

>>> peft_config = LNTuningConfig(
...     task_type=TaskType.CAUSAL_LM,
... )

>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
>>> model = get_peft_model(model, peft_config)
>>> model.print_trainable_parameters()
```

**属性**：
- **模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适应的模型。
- **peft_config** ([LNTuningConfig](/docs/peft/v0.20.0/en/package_reference/layernorm_tuning#peft.LNTuningConfig))：LN 调整模型的配置。

### C3A：通过循环卷积进行参数高效微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/c3a.md