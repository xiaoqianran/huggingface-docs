<!-- huggingface-docs: machine-translated zh-CN from English source -->

# BEFT：低数据机制中语言模型的偏置高效微调

[BEFT](https://arxiv.org/abs/2509.15974) 是一种参数高效微调算法 (PEFT)，仅微调预训练 Transformer 模型的值投影的添加偏差项。 BEFT 证明，与微调查询/关键投影的附加偏差项相比，微调预训练 Transformer 的值投影的附加偏差项通常会在低数据情况下带来更高的下游性能。

BEFT 目前有以下权衡：

优点：
- BEFT 需要的参数比 LoRA 少得多，同时在低数据条件下的任务中保持竞争性或卓越的性能。

缺点：
- 在高数据情况下，与 LoRA 和全参数微调相比，BEFT 的有效性可能有限。

如果您的用例属于高数据范围，请考虑其他 PEFT 方法，例如 LoRA。

论文摘要是：*微调大型语言模型 (LLM) 的偏差项有可能实现前所未有的参数效率，同时保持竞争性能，特别是在低数据情况下。然而，微调不同偏差项（即查询、键或值预测中的 **b**q、**b**k 和 **b**v）与下游性能之间的联系迄今为止仍不清楚。在本文中，我们研究了微调 **b**q、**b**k 和 **b**v 与下游任务性能之间的联系。我们的主要发现是，与 **b**q 和 **b**k 相比，直接微调 **b**v 通常会在低数据情况下带来更高的下游性能。我们在各种 LLM 中广泛评估了这一独特属性，涵盖仅编码器和仅解码器架构，最高可达 6.7B 参数（包括无偏差 LLM）。我们的结果为跨各种下游任务*直接微调 **b**v 的有效性提供了强有力的证据。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=BEFT"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## BeftConfig[[peft.BeftConfig]]

#### peft.BeftConfig[[peft.BeftConfig]]

```python
peft.BeftConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, target_modules: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None, init_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/beft/config.py#L25)

**参数：**target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

module_to_save (`Optional[List[str]]`) ：除 BEFT 层之外的模块列表，要设置为可训练并保存在最终检查点中。

init_weights (`bool`) ：是否初始化BEFT层中的向量，默认为`True`。不鼓励将其设置为 `False`。

这是存储[BeftModel](/docs/peft/v0.20.0/en/package_reference/beft#peft.BeftModel)配置的配置类。

## BeftModel[[peft.BeftModel]]

#### peft.BeftModel[[peft.BeftModel]]

```python
peft.BeftModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/beft/model.py#L26)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([BeftConfig](/docs/peft/v0.20.0/en/package_reference/beft#peft.BeftConfig)) ：(BEFT) 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

(BEFT) 模型。

通过仅微调预训练的值投影的附加偏差项来创建注入适配器
低训练数据状态下的 Transformer 模型 (BEFT)。该方法详细描述于
https://arxiv.org/abs/2509.15974

示例：

```py
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import BeftModel, BeftConfig

>>> config = BeftConfig(
...     peft_type="Beft",
...     task_type="SEQ_2_SEQ_LM",
...     target_modules=["v"],
... )

>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> beft_model = BeftModel(model, config, adapter_name="default")
```

**属性**：
- **模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([BeftConfig](/docs/peft/v0.20.0/en/package_reference/beft#peft.BeftConfig))：(BEFT) 模型的配置。

### PVeRA：基于概率向量的随机矩阵适应
https://huggingface.co/docs/peft/v0.20.0/package_reference/pvera.md