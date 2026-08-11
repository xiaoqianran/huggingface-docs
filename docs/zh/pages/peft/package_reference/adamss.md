<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 阿达MSS

[AdaMSS](https://openreview.net/forum?id=8ZdWmpYxT0)（AdaMSS：参数高效微调的自适应多子空间方法）是一种参数高效微调方法，它使用 SVD 分解权重矩阵，并将分解的空间聚类为多个可训练子空间。每个子空间学习独立的低秩更新，同时原始权重保持冻结。 AdaMSS 还支持自适应子空间分配 (ASA)，它在训练期间根据梯度信息动态修剪不太重要的子空间。

论文摘要是：> 我们提出了 AdaMSS，一种自适应多子空间方法，用于大型模型的参数高效微调。与在网络权重的大型单个子空间内运行的传统参数高效微调方法不同，AdaMSS 利用子空间分割来获得多个较小的子空间，并在训练期间自适应地减少可训练参数的数量，最终仅更新与目标下游任务最相关的子空间小子集相关的参数。通过使用最低秩表示，AdaMSS 实现了更紧凑的表达能力和更精细的模型参数调整。理论分析表明，AdaMSS 比 LoRA、PiSSA 等单子空间低秩方法具有更好的泛化保证。图像分类、自然语言理解和自然语言生成任务的大量实验表明，AdaMSS 实现了与完全微调相当的性能，并且在大多数情况下优于其他参数高效的微调方法，同时需要更少的可训练参数。值得注意的是，在 ViT-Large 模型上，AdaMSS 在 7 t 时间段内的平均准确率比 LoRA 高 4.7%问道，仅使用 15.4% 的可训练参数。在 RoBERTa-Large 上，AdaMSS 在六项任务中的平均准确度比 PiSSA 高出 7%，同时将可训练参数的数量减少了约 94.4%。这些结果证明了 AdaMSS 在参数高效微调方面的有效性。 AdaMSS 的代码可在 https://github.com/jzheng20/AdaMSS 获取。

AdaMSS 目前有以下限制：
- 仅支持`nn.Linear`层。
- KMeans 聚类步骤需要 scikit-learn。

如果这些约束不适用于您的用例，请考虑其他方法。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=ADAMSS"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## AdamssConfig[[peft.AdamssConfig]]

#### peft.AdamssConfig[[peft.AdamssConfig]]

```python
peft.AdamssConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 100, num_subspaces: int = 5, subspace_rank: int = 1, target_modules: Optional[Union[list[str], str]] = None, init_weights: Optional[Literal['orthogonal']] = 'orthogonal', modules_to_save: Optional[list[str]] = None, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, use_asa: bool = False, asa_target_subspaces: int = 50, init_warmup: int = 50, final_warmup: int = 1000, mask_interval: int = 100, asa_importance_beta: float = 0.85, asa_uncertainty_beta: float = 0.85, asa_schedule_exponent: float = 3.0, use_dynamic_rank: bool = False, svd_threshold: float = 0.1, random_seed: int = 0)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adamss/config.py#L25)

**参数：**

r (`int`) ：SVD 分解的总秩（在论文中表示为 R）。这决定了在聚类之前使用多少个奇异向量来表示权重矩阵。较高的值从原始权重中捕获更多信息，但需要更多计算和内存。较低的值提供更强的正则化。典型值范围为 50 到 500。默认值为 100。num_subspaces (`int`) ：将 SVD 分解空间聚类到的子空间数 (K)。每个子空间学习独立的低秩更新。增加该值可以实现更细粒度的适应，但会成比例地增加可训练参数的数量。当使用 ASA（自适应子空间分配）时，这确定了修剪之前的初始子空间数量。典型值范围为 3 到 10。默认值为 5。 

subspace_rank (`int`) ：每个可训练子空间的排名 (r_i)。这控制了每个子空间学习适应的能力。较高的值会增加表现力，但也会增加可训练参数。总可训练参数规模为 O(num_subspaces * subspace_rank * (in_dim + out_dim) / num_subspaces)。对于大多数任务，1-4 的值效果很好。默认值为 1。 

target_modules (`Optional[Union[list[str], str]]`) ：要应用 AdaMSS 的模块的名称。如果指定，则仅调整这些模块。可以是精确模块名称的列表或正则表达式。例如，`['q_proj', 'v_proj']`用于注意层，或`'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`用于正则表达式匹配。module_to_save (`Optional[list[str]]`) ：除了 AdaMSS 层之外的模块列表，要设置为可训练并保存在最终检查点中。这些模块将进行全面微调（不仅仅是低等级）。分类任务中随机初始化的头（例如`classifier`或`score`）是必需的。 

init_weights (`Literal["orthogonal"]`) ：AdaMSS 可训练权重的初始化方法。目前仅支持“正交”，它对 B 矩阵（输出投影）使用正交初始化。 A 矩阵初始化为零，以确保模型从预训练的权重开始。设置为 None 可在从检查点加载时跳过初始化。默认为“正交”。 

Layers_to_transform (`Optional[Union[list[int], int]]`) ：应用 AdaMSS 的特定层索引。如果指定，则仅对这些层进行调整，这对于试验哪些层从调整中受益最大非常有用。可以是单个整数或整数列表。 

Layers_pattern (`Optional[Union[list[str], str]]`) ：指定`layers_to_transform`时匹配图层名称的模式。用于从不遵循通用模式的模块名称中提取层索引。use_asa (`bool`) ：是否启用自适应子空间分配（ASA）。启用后，ASA 会在训练期间根据梯度信息动态修剪不太重要的子空间，从而在保持性能的同时减少有效参数数量。需要与训练回调集成。默认值为 False。 

asa_target_subspaces (`int`) ：启用 ASA 时所有层的目标活动子空间总数。 ASA 将逐步修剪子空间，直到达到此目标。较低的值会导致更积极的修剪和更少的可训练参数。应小于`num_subspaces * num_target_modules`。典型值范围为 20 到 100，具体取决于模型大小。默认为 50。当 `use_asa=True` 时必须为正整数。 

init_warmup (`int`) ：开始 ASA 修剪之前等待的训练步骤数。在热身期间，所有子空间都保持活动状态，以使重要性分数稳定。较高的值可以提供更多时间进行准确的重要性估计，但会延迟修剪。典型值范围为 50 到 200。默认值为 50。当`use_asa=True` 时，必须小于`final_warmup`。Final_warmup (`int`) ：ASA 完成修剪并达到 `asa_target_subspaces` 活动子空间的训练步骤。剪枝分布在`init_warmup`和`final_warmup`之间。应根据总训练步数设置；通常是总训练步骤的 1/3 到 1/2。默认为1000。当`use_asa=True`时必须大于`init_warmup`。 

mask_interval (`int`) ：ASA 掩码更新之间的训练步骤数。较低的值允许更频繁的适应，但会增加开销。值越高，更新之间的重要性估计就越稳定。典型值范围为 50 到 200。默认值为 100。当`use_asa=True` 时必须为正整数。 

asa_importance_beta (`float`)：用于平滑子空间重要性分数的指数移动平均 (EMA) 系数。值越高（接近 1.0），历史重要性就越重要，从而提供稳定性。较低的值使得重要性对最近的梯度更加敏感。典型值范围为 0.8 至 0.95。默认值为 0.85。asa_uncertainty_beta (`float`)：用于平滑重要性不确定性估计的 EMA 系数。控制不确定性适应梯度方差的速度。与 asa_importance_beta 类似，较高的值提供更稳定的估计。典型值范围为 0.8 至 0.95。默认值为 0.85。 

asa_schedule_exponent (`float`) ：控制 ASA 预热期间从总子空间到`asa_target_subspaces` 的衰减率的调度指数。较高的值会导致更快的初始修剪（更积极的早期减少），而较低的值会提供更渐进的、类似线性的衰减。公式为：current_active_subspaces = asa_target_subspaces + (asa_total_subspaces - asa_target_subspaces) * (进度 ** 指数)。典型值范围从 1.0（线性）到 5.0（激进）。默认值为 3.0。当`use_asa=True`时必须为正数（零或负指数要么使调度退化为永久无操作，要么一旦`progress`正好达到0.0，则引发`ZeroDivisionError`）。use_dynamic_rank (`bool`) ：是否根据奇异值大小动态确定子空间等级。当为 True 时，每个子空间的等级是通过对阈值以上的奇异值进行计数来确定的，从而允许不同的子空间具有不同的有效等级。当为 False 时，所有子空间都使用固定的 `subspace_rank`。默认值为 False。 

svd_threshold (`float`) ：动态排名选择的阈值比例，仅在`use_dynamic_rank=True`时使用。如果奇异值超过`threshold * max_singular_value`，则被认为是显着的。值越高，有效排名越低（更激进的截断）。典型值范围为 0.05 至 0.2。默认值为 0.1（最大值的 10%）。

Adamss（自适应多子空间）方法的配置类。

AdaMSS 是一种参数高效的微调方法，它使用 SVD 分解权重矩阵并对权重矩阵进行聚类
将空间分解为多个可训练的子空间。它学习这些子空间内的低等级更新，同时保持
原始重量被冻结。

## AdamssModel[[peft.AdamssModel]]

#### peft.AdamssModel[[peft.AdamssModel]]

```python
peft.AdamssModel(model, config, adapter_name, low_cpu_mem_usage: bool = False, state_dict: Optional[dict] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adamss/model.py#L31)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config (`AdamssConfig`) ：Adamss 模型的配置。adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**回报：** `torch.nn.Module`

亚当斯模型。

从预训练模型创建 Adamss（自适应多子空间）模型。

该方法使用 SVD 分解权重矩阵，并将分解空间聚类为多个可训练子空间
用于参数高效的微调。

示例：
```python
>>> from transformers import AutoModelForImageClassification
>>> from peft import AdamssConfig, get_peft_model

>>> config = AdamssConfig(
...     r=500,
...     num_subspaces=5,
...     target_modules=["query", "value"],
... )

>>> model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224")
>>> adamss_model = get_peft_model(model, config)
```

**属性**：
- **model** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([AdamssConfig](/docs/peft/v0.20.0/en/package_reference/adamss#peft.AdamssConfig))：Adamss 模型的配置。

#### update_and_allocate[[peft.AdamssModel.update_and_allocate]]

```python
update_and_allocate(global_step: int)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adamss/model.py#L157)

**参数：**

global_step (*int*) ：当前训练步骤。

更新重要性分数并应用 ASA 屏蔽（如果启用）。

应在`loss.backward()`之后和之前的**每个**训练步骤中调用此方法
`optimizer.zero_grad()` 启用 ASA 时。其内部：

1. 在预热期间通过 EMA 每一步累积重要性分数。
2. 在掩码间隔，应用全局 top-K 掩码并重置重要性。

这是 ASA 的单一入口点 – 使用 `AdamssAsaCallback` 和 HuggingFace `Trainer`
只需委托给此方法即可。对于自定义训练循环，请直接调用此函数而不是回调。

示例：

```python
for step, batch in enumerate(dataloader):
loss = model(**batch).loss loss.backward() optimizer.step() model.base_model.update_and_allocate(step)
optimizer.zero_grad()
```### 乐哈
https://huggingface.co/docs/peft/v0.20.0/package_reference/loha.md