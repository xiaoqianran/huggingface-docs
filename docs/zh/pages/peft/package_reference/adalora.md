<!-- huggingface-docs: machine-translated zh-CN from English source -->

#阿达洛拉

[AdaLoRA](https://hf.co/papers/2303.10512)（自适应 LoRA）是一种优化分配给权重矩阵和层的可训练参数数量的方法，与 LoRA 不同，LoRA 在所有模块中均匀分配参数。为重要的权重矩阵和层预算更多的参数，而不太重要的权重矩阵和层则接收更少的参数。您可以控制矩阵的平均期望 *rank* 或 `r`，以及使用 `target_modules` 将 AdaLoRA 应用到哪些模块。其他需要设置的重要参数是`lora_alpha`（缩放因子）和`modules_to_save`（除了要训练和保存的 AdaLoRA 层之外的模块）。所有这些参数以及更多参数都可以在 [AdaLoraConfig](/docs/peft/v0.20.0/en/package_reference/adalora#peft.AdaLoraConfig) 中找到。

论文摘要是：*在下游任务上微调大型预训练语言模型已成为 NLP 的重要范例。然而，常见的做法是对预训练模型中的所有参数进行微调，当存在大量下游任务时，这会变得令人望而却步。因此，提出了许多微调方法，以参数有效的方式学习预训练权重的增量更新，例如低秩增量。这些方法通常在所有预训练的权重矩阵中均匀分配增量更新的预算，并忽略不同权重参数的不同重要性。因此，微调性能不是最佳的。为了弥补这一差距，我们提出了 AdaLoRA，它根据权重矩阵的重要性得分自适应地分配权重矩阵之间的参数预算。特别是，AdaLoRA 以奇异值分解的形式参数化增量更新。这种新颖的方法使我们能够有效地修剪不重要更新的奇异值，这本质上是为了减少参数预算，但避免密集的精确 SVD 计算。我们使用几个预训练模型对自然语言进行了广泛的实验语言处理、问答和自然语言生成来验证 AdaLoRA 的有效性。结果表明，AdaLoRA 较基线有显着改善，尤其是在低预算环境中。我们的代码可在 https://github.com/QingruZhang/AdaLoRA* 上公开获取。

> [!警告]
> AdaLoRA 有一个 [update_and_allocate()](/docs/peft/v0.20.0/en/package_reference/adalora#peft.AdaLoraModel.update_and_allocate) 方法，应在每个训练步骤调用该方法来更新参数预算和掩码，否则不执行适应步骤。这需要编写自定义训练循环或对 [Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer) 进行子类化以合并此方法。举个例子，看看这个[custom training loop](https://github.com/huggingface/peft/blob/912ad41e96e03652cabf47522cd876076f7a0c4f/examples/conditional_generation/peft_adalora_seq2seq.py#L120)。AdaLoRA 通过为更适合任务的重要权重矩阵分配更多参数（换句话说，更高的等级`r`）并修剪不太重要的参数来管理从 LoRA 引入的参数预算。等级由类似于奇异值分解（SVD）的方法控制。 $\Delta W$ 用两个正交矩阵和一个包含奇异值的对角矩阵进行参数化。这种参数化方法避免了计算量大的迭代应用 SVD。基于该方法，$\Delta W$的排名根据重要性得分进行调整。 $\Delta W$ 分为三元组，每个三元组根据其对模型性能的贡献进行评分。具有低重要性分数的三元组被修剪，具有高重要性分数的三元组被保留以进行微调。AdaLoRA 训练分为三个阶段：初始阶段、预算阶段和最终阶段。在初始阶段，不应用预算，因此排名不会受到影响。在预算阶段，应用上述过程并根据预算重新分配排名，旨在为更重要的适配器提供更多排名，为不太重要的层提供更少的排名。到了最后阶段，预算已经结束，职级会重新分配，但我们可能会用重新分配的职级继续训练一段时间，以进一步提高绩效。

> [!注意]
> **欢迎贡献**：本节需要澄清。
>
> 目前还不清楚重要性是如何衡量的。这些解释也有点多余，可以从整合中受益。
> 请参阅[here](../developer_guides/contributing#documentation-improvements)了解如何贡献。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=ADALORA"
	frameborder="0"
	width="850"
	height="1000"
>

## 用法

```py
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    r=8,
    init_r=12,
    tinit=200,
    tfinal=1000,
    deltaT=10,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 520,325 || all params: 87,614,722 || trainable%: 0.5938785036606062"

[... training code ...]

model.update_and_allocate(step_idx)
```

# API

## AdaLoraConfig[[peft.AdaLoraConfig]]

#### peft.AdaLoraConfig[[peft.AdaLoraConfig]]

```python
peft.AdaLoraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, lora_alpha: int = 8, lora_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: Literal['none', 'all', 'lora_only'] = 'none', use_rslora: bool = False, modules_to_save: Optional[list[str]] = None, init_lora_weights: bool | Literal['gaussian', 'eva', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'corda', 'loftq', 'orthogonal', 'mica'] = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, rank_pattern: typing.Optional[dict] = None, alpha_pattern: Optional[dict] = <factory>, megatron_config: Optional[dict] = None, megatron_core: Optional[str] = 'megatron.core', trainable_token_indices: Optional[Union[list[int], dict[str, list[int]]]] = None, loftq_config: Union[LoftQConfig, dict] = <factory>, eva_config: Optional[EvaConfig] = None, corda_config: Optional[CordaConfig] = None, lora_ga_config: Optional[LoraGAConfig] = None, use_dora: bool = False, velora_config: Optional[Union[VeloraConfig, dict]] = None, alora_invocation_tokens: Optional[list[int]] = None, use_qalora: bool = False, qalora_group_size: int = 16, monteclora_config: Optional[MontecloraConfig] = None, layer_replication: Optional[list[tuple[int, int]]] = None, runtime_config: LoraRuntimeConfig = <factory>, lora_bias: bool = False, target_parameters: Optional[list[str]] = None, use_bdlora: Optional[BdLoraConfig] = None, arrow_config: Optional[ArrowConfig] = None, ensure_weight_tying: bool = False, target_r: int = 8, init_r: int = 12, tinit: int = 0, tfinal: int = 0, deltaT: int = 1, beta1: float = 0.85, beta2: float = 0.85, orth_reg_weight: float = 0.5, total_step: typing.Optional[int] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adalora/config.py#L24)

**参数：**

target_r (`int`) ：增量矩阵的目标平均秩。

init_r (`int`) ：每个增量矩阵的初始等级。

tinit (`int`) ：初始微调预热的步骤。tfinal (`int`) ：最终微调的步数。

deltaT (`int`) ：两次预算分配之间的时间间隔。

beta1 (`float`) ：用于敏感性平滑的 EMA 超参数。

beta2 (`float`) ：用于模糊量化的 EMA 超参数。

orth_reg_weight (`float`) ：正交正则化的系数。

Total_step (`int`) ：训练前应指定的总训练步数。

rank_pattern (`list`) ：RankAllocator为每个权重矩阵分配的排名。

这是存储[AdaLoraModel](/docs/peft/v0.20.0/en/package_reference/adalora#peft.AdaLoraModel)配置的配置类。

AdaLoRA 具有由`tinit`、`tfinal` 和`total_step` 定义的三个阶段。

初始阶段可以理解为预训练适配器的步骤，以便在降低其等级时，
已经是一些可以减少的信息编码，而不是随机矩阵。该阶段定义为
供应`tinit`。

在初始阶段结束（`tinit`步骤已通过）且最终阶段尚未开始后，AdaLoRA 减少了
每一步允许每层有多少排名的预算。这就是降级的地方
正在发生。这一直持续到达到 `total_step - tfinal` 步为止。最后一个阶段，一旦达到 `total_step - tfinal` 步数就开始，不再改变层等级，但是
微调前一阶段产生的降级层。

一个实际的例子：`tinit`是10，`tfinal`是20，`total_step`是100。我们花了10步进行预训练
没有降级，因为我们的预算是恒定的（初始阶段），然后我们在
削减阶段，我们的预算逐步减少，最后在最后的微调阶段有 20 个步骤，没有
减少。

## AdaLoraModel[[peft.AdaLoraModel]]

#### peft.AdaLoraModel[[peft.AdaLoraModel]]

```python
peft.AdaLoraModel(model, config, adapter_name, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adalora/model.py#L36)

**参数：**

model ([transformers.PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) ：要适配的模型。

config ([AdaLoraConfig](/docs/peft/v0.20.0/en/package_reference/adalora#peft.AdaLoraConfig)) ：AdaLora 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

阿达洛拉模型。

从预训练的 Transformer 模型创建 AdaLoRA（自适应 LoRA）模型。纸张：
https://openreview.net/forum?id=lq62uWRJjiY

示例：
```py
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import AdaLoraConfig, get_peft_model

>>> config = AdaLoraConfig(
...     peft_type="ADALORA",
...     task_type="SEQ_2_SEQ_LM",
...     init_r=12,
...     lora_alpha=32,
...     target_modules=["q", "v"],
...     lora_dropout=0.01,
...     total_step=1000,
... )
>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> adalora_model = get_peft_model(model, config)
```**属性**：
- **model** ([transformers.PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([AdaLoraConfig](/docs/peft/v0.20.0/en/package_reference/adalora#peft.AdaLoraConfig))：AdaLora 模型的配置。

#### add_weighted_adapter[[peft.AdaLoraModel.add_weighted_adapter]]

```python
add_weighted_adapter(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adalora/model.py#L347)

AdaLoRA 不支持此方法，请改用 LoRA。

#### update_and_allocate[[peft.AdaLoraModel.update_and_allocate]]

```python
update_and_allocate(global_step)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adalora/model.py#L305)

**参数：**

global_step (`int`) ：当前训练步骤，用于计算adalora预算。

此方法更新 Adalora 预算和掩码。

这应该在`loss.backward()`之后和`zero_grad()`之前的每个训练步骤中调用。

`tinit`、`tfinal`、`deltaT` 在方法中处理。

示例：

```python
>>> loss = model(**input).loss
>>> loss.backward()
>>> optimizer.step()
>>> model.base_model.update_and_allocate(i_step)
>>> optimizer.zero_grad()
```

### 阿达MSS
https://huggingface.co/docs/peft/v0.20.0/package_reference/adamss.md