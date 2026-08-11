<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Lily：跨层低阶互联适应

[Lily](https://huggingface.co/papers/2407.09946)是一种参数高效的微调技术，为适配器矩阵引入了跨层权重共享。 Lily 没有像 LoRA 那样每层学习一个独立的 AB 对，而是使用**本地共享 A 适配器**（每个 A 在`stride_A` 连续层块中共享）和**全局共享 B 专家**（一小组 `num_B` B 适配器在所有层之间共享）。在每次前向传递时，轻量级数据相关路由器都会计算 B 专家的 softmax 加权组合，以生成该层和输入的有效 B。

这种共享可以将适配器矩阵的总数从`2N`（标准LoRA）减少到`N / stride_A + num_B`，从而释放参数预算以使用**大得多的等级`r`** - 通常是`2×`–`4×`您将在LoRA中使用的矩阵。更高的等级和更好的互连性增加了权重更新`ΔW = A × combined_B`的有效等级，从而带来更好的适应性能。因为 B 组合是**数据相关**（路由器权重取决于运行时的输入激活），所以 **不支持 `merge` 和 `unmerge`**。如果您的部署需要权重合并，请考虑其他方法，例如 LoRA。

Lily 目前有以下额外限制：
- 仅支持`nn.Linear`层。
- 不支持量化层。

如果这些约束不适用于您的用例，请考虑其他方法。

论文摘要是：> 低秩自适应（LoRA）是一种广泛使用的参数高效微调（PEFT）方法，通过低秩适配器 A 和 B 学习预训练权重 W 的权重更新 ΔW = AB。虽然 LoRA 保证了硬件效率，但其低秩权重更新限制了自适应性能。在本文中，我们提出了跨层低秩互连自适应（Lily），这是一种新颖的 PEFT 方法，引入了具有本地共享 A 和全局共享 B 专家的互连框架。这种结构消除了冗余的每层 AB 对，从而可以使用相同或更少的参数实现更高阶的 ΔW。为了增强表现力，我们使用数据相关的路由器来确定 A-B 互连，防止 B 专家收敛到相同的行为，并提高跨域的表现力。跨模式、架构和模型大小的实验证明了 Lily 的卓越性能和效率。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LILY"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## LilyConfig[[peft.LilyConfig]]

#### pft.LilyConfig[[peft.LilyConfig]]

```python
peft.LilyConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, stride_A: int = 1, num_B: int = 2, scaling: float = 1.0, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, init_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lily/config.py#L24)

**参数：**r (`int`) : Lily 的等级。确定每个适配器的内部隐藏维度和权重更新`A @ B`的等级。在 Lily 中，由于适配器的数量通常比 LoRA 中的要少，因此每个适配器需要承载更多的容量，因此建议使用比 LoRA 中更大的 `r` — 通常是您通常使用的 LoRA 等级 `2x`、`3x` 或 `4x`。可训练参数的总数与`r * (total_layers / stride_A + num_B)`成比例，因此增加`r`同时保持`stride_A`大和`num_B`小是建议的权衡。stride_A (`int`) ：共享一个 A 适配器的连续层数。例如，如果`stride_A=4`，则每 4 个相邻层共享相同的 A 适配器，总共产生 `total_layers / stride_A` 不同的 A 适配器。 A 适配器将输入压缩为大小为 `r` 的低秩表示。 `stride_A` 应不小于 1。建议值：`2`、`3` 或 `4`（即每 2、3 或 4 层共享）。相反，保持 `stride_A` 大（更少的不同 A 适配器）并增加 `r` 会比相反的权衡（小 `stride_A`，小 `r`）带来更好的性能。设置`stride_A=1`意味着每一层都有自己的A适配器。注意：A 共享发生在每个目标内（具有相同目标后缀的层）。例如，如果您的 target_modules 为 `['q_proj', 'v_proj']` 并且您设置了 `stride_A=2`，则每 2 个相邻的 q_proj 层将共享一个 A 适配器，每 2 个相邻的 v_proj 层将共享另一个 A 适配器，但 q_proj 和 v_proj 层不会彼此共享 A 适配器，因为它们具有不同的后缀。num_B (`int`) ：共享 B 适配器的数量。与 A 适配器（按层分组）不同，所有 B 适配器在每一层全局共享。对于每个前向传递，路由器计算所有 `num_B` B 适配器的加权组合（使用 softmax 归一化权重）以生成单个组合 B 适配器，然后将低秩表示投影回原始维度。建议将`num_B`设置为与`total_layers / stride_A`类似的顺序。建议值：`total_layers / 2`、`total_layers / 3` 或 `total_layers / 4`。与`stride_A`类似，更喜欢较小的`num_B`和较大的`r`，而不是较大的`num_B`和较小的`r`。注意：要训练路由器，您至少需要 2 个 B 适配器（即 `num_B >= 2`），因为路由器会学习计算 B 适配器的加权组合。注意：B 共享发生在每个目标内（具有相同目标后缀的层）。例如，如果您的 target_modules 是 `['q_proj', 'v_proj']` 并且您设置了 `num_B=4`，那么将有 4 个 B 适配器在所有 q_proj 层之间共享，另外 4 个 B 适配器在所有 v_proj 层之间共享，但 q_proj 和 v_proj 层不会彼此共享 B 适配器，因为它们具有不同的后缀。target_modules (`Union[List[str], str]`, *可选*) ：要应用 Lily 的模块的名称。可以是模块名称字符串列表（例如`['q_proj', 'v_proj']`）或正则表达式模式（例如`'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`）。如果未指定，Lily 将应用于所有支持的线性图层。

缩放 (`float`) ：在将其添加到冻结权重的前向传递之前，应用于组合适配器输出 (`scaling * A @ combined_B`) 的标量乘法器。与 LoRA 不同，Lily 不使用 `alpha / r` 配方；相反，`scaling`是一个直接乘数。这种设计使得扫描对数刻度上的值变得简单（例如`0.01`、`0.1`、`1.0`、`10.0`）。最佳值取决于任务，应将其视为超参数。我们建议从`1.0`开始。

module_to_save (`List[str]`, *可选*) ：除了 Lily 层之外，要设置为可训练并保存在最终检查点中的模块列表。例如，在序列分类或令牌分类任务中，最后一层`classifier/score`是随机初始化的，因此需要可训练和保存。except_modules (`Union[List[str], str]`, *可选*) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

Layers_to_transform (`Union[list[int], int]`, *可选*) ：要转换的图层索引，如果指定此参数，PEFT 将仅转换此列表中指定的图层索引。如果传递单个整数，PEFT 将仅转换该索引处的图层。

Layers_pattern (`Optional[Union[List[str], str]]`, *可选*) ：图层模式名称，仅在 `layers_to_transform` 与 None 不同且图层模式不在公共图层模式中时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。init_weights (`bool`) : 是否使用默认初始化方案初始化Lily适配器权重：A矩阵使用Kaiminguniform初始化，B矩阵初始化为零，确保适配器输出在训练开始时为零，并且不会干扰预训练模型。强烈建议将其保留为`True`，除非您有特定原因要更改它。

这是存储[LilyModel](/docs/peft/v0.20.0/en/package_reference/lily#peft.LilyModel)配置的配置类。

## LilyModel[[peft.LilyModel]]

#### peft.LilyModel[[peft.LilyModel]]

```python
peft.LilyModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lily/model.py#L30)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([LilyConfig](/docs/peft/v0.20.0/en/package_reference/lily#peft.LilyConfig)) ：Lily 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**返回：** `torch.nn.Module`

莉莉 PEFT 模型。

从预训练的 Transformer 模型创建低阶跨层互连适应 (Lily) 模型。

该方法在 https://arxiv.org/abs/2407.09946 中有详细描述。

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([LilyConfig](/docs/peft/v0.20.0/en/package_reference/lily#peft.LilyConfig))：Lily 模型的配置。

### WaveFT：小波微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/waveft.md