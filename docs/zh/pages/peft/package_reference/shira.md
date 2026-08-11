<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 稀疏高等级适配器

稀疏高等级适配器或[SHiRA](https://huggingface.co/papers/2406.13175)是适配器的替代类型，并且已被发现比低等级适配器具有显着优势。具体来说，对于各种视觉和语言任务，SHiRA 比 LoRA 取得了更好的准确性。它还通过显着减少概念损失（低等级适配器面临的常见问题）来提供更简单和更高质量的多适配器融合。 SHiRA 直接微调少量基础模型的参数，以在任何适应任务上微调模型。

SHiRA 目前有以下限制：

- 仅支持`nn.Linear`层。

论文摘要是：> 低秩适应（LoRA）在最近的生成式人工智能研究中引起了广泛关注。 LoRA 的主要优点之一是它能够与预训练模型融合，在推理过程中不会增加任何开销。然而，从移动部署的角度来看，我们可以避免融合模式下的推理开销，但失去快速切换适配器的能力，或者在非融合模式下实现快速切换时遭受显着的推理延迟（高达 30% 以上）。当同时使用多个适配器时，LoRA 还会表现出概念丢失。在本文中，我们提出了稀疏高阶适配器（SHiRA），这是一种新的范式，不会产生推理开销，能够实现快速切换，并显着减少概念损失。具体来说，SHiRA 可以通过直接调整 1-2% 的基础模型权重而保持其他不变来进行训练。这导致了高度稀疏的适配器，可以直接在融合模式下切换。我们进一步提供了关于 SHiRA 中的高稀疏性如何通过减少概念丢失来帮助多适配器融合的理论和经验见解。我们对 LVM 和 LLM 的大量实验表明，仅微调一小部分参数基本模型的性能显着优于 LoRA，同时支持快速切换和多适配器融合。最后，我们提供基于参数高效微调 (PEFT) 库的延迟和内存高效 SHiRA 实现，其训练速度几乎与 LoRA 相同，同时消耗峰值 GPU 内存高达 16%，从而使 SHiRA 易于在实际用例中采用。为了证明推理过程中快速切换的优势，我们证明在基本模型上加载 SHiRA 的速度比在 CPU 上加载 LoRA 融合快 5 倍到 16 倍。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=SHIRA"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## ShiraConfig[[peft.ShiraConfig]]

#### peft.ShiraConfig[[peft.ShiraConfig]]

```python
peft.ShiraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, mask_type: Literal['random'] = 'random', random_seed: Optional[int] = None, target_modules: Optional[Union[list[str], str]] = None, fan_in_fan_out: bool = False, init_weights: bool = True, modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/shira/config.py#L28)

**参数：**

r（`int`，*可选*，默认为`32`）：对于给定的目标模块，SHiRA 参数的数量计算为 r(m+n)，其中原始张量维度为 m x n。这意味着 SHiRA 参数的数量与 LoRA 适配器的参数数量相同。 SHiRA 是一个高级适配器。设置此 r 参数不会将排名限制为此值。mask_type (`str`, 默认为`random`) : mask 函数的类型。默认为随机稀疏掩码。还可以通过实例化 `config = ShiraConfig(...)` 然后设置 `config.mask_fn = <your custom mask function>` 来提供可选的用户定义 mask_fn 来计算掩码值。对于形状为 m x n 的预训练权重，自定义掩码函数必须仅返回一个掩码（形状：m x n），该掩码必须为二进制 0 或 1，且线性层的 num_shira_parameters = r(m + n)。掩码的设备和数据类型必须与基础层权重的设备和数据类型相同。请参阅 mask_functions.py 了解更多详细信息并查看默认的随机稀疏掩码实现。

random_seed (`int`，*可选*，默认为`None`)：random_mask 的火炬生成器的随机种子。

target_modules (`Union[List[str], str]`) ：要替换为 SHiRA 的模块名称列表或模块名称的正则表达式。例如，['q', 'v'] 或 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'。仅支持线性层。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。init_weights (`bool`，默认为 `True`) ：初始化 SHiRA 权重为零值。如果设置为 False，SHiRA 权重将初始化为 randn 值而不是零，并且仅用于测试。

module_to_save (`List[str]`) ：除 SHiRA 层之外的模块列表，要设置为可训练并保存在最终检查点中。

这是存储[ShiraModel](/docs/peft/v0.20.0/en/package_reference/shira#peft.ShiraModel)配置的配置类。

## ShiraModel[[peft.ShiraModel]]

#### peft.ShiraModel[[peft.ShiraModel]]

```python
peft.ShiraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/shira/model.py#L41)

**参数：**

model ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([ShiraConfig](/docs/peft/v0.20.0/en/package_reference/shira#peft.ShiraConfig)) ：SHiRA 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**返回：** `torch.nn.Module`

希拉模型。

从预训练模型创建稀疏高阶适配器 (SHiRA) 模型。

示例：

```py
>>> from transformers import AutoModelForCausalLM
>>> from peft import ShiraConfig, get_peft_model

>>> base_model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
>>> config = ShiraConfig(r=32)
>>> model = get_peft_model(base_model, config)
```

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([ShiraConfig](/docs/peft/v0.20.0/en/package_reference/shira#peft.ShiraConfig))：SHiRA 模型的配置。

### UniLoRA
https://huggingface.co/docs/peft/v0.20.0/package_reference/unilora.md