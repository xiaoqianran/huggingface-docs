<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 洛拉

> [!提示]
> LoRA 是最流行的 PEFT 方法之一，如果您刚刚开始使用 PEFT，这是一个很好的起点。它最初是为大型语言模型开发的，但由于其效率和有效性，它是扩散模型的一种非常流行的训练方法。

低秩适应（[LoRA](https://huggingface.co/papers/2106.09685)）是一种 PEFT 方法，可将大矩阵分解为两个较小的低秩矩阵。这大大减少了需要微调的参数数量。

论文摘要是：*自然语言处理的一个重要范例包括对通用领域数据的大规模预训练以及对特定任务或领域的适应。当我们预训练更大的模型时，重新训练所有模型参数的完全微调变得不太可行。以 GPT-3 175B 为例——部署微调模型的独立实例（每个实例都有 175B 参数）的成本极其昂贵。我们提出了低秩适应（LoRA），它冻结了预训练的模型权重，并将可训练的秩分解矩阵注入到 Transformer 架构的每一层中，大大减少了下游任务的可训练参数的数量。与使用 Adam 微调的 GPT-3 175B 相比，LoRA 可以减少 10,000 倍的可训练参数数量，以及 3 倍的 GPU 内存需求。 LoRA 在 RoBERTa、DeBERTa、GPT-2 和 GPT-3 上的模型质量表现与微调相当或更好，尽管可训练参数较少、训练吞吐量较高，并且与适配器不同，没有额外的推理延迟。我们还对语言模型适应中的排名缺陷进行了实证研究，这揭示了 LoRA 的功效。我们发布一个软件包，促进 LoRA 与 PyTorch 模型的集成，并在 [this https URL](https://github.com/microsoft/LoRA) 处提供 RoBERTa、DeBERTa 和 GPT-2 的实现和模型检查点。*

LoRA 通过低秩分解用两个较小的矩阵（称为*更新矩阵*）表示权重更新 $\Delta W$。可以训练这些新矩阵以适应新数据，同时保持较低的参数总数。原始权重矩阵保持冻结状态，不会收到任何进一步的更新。为了产生最终结果，将原始权重和额外调整的权重相结合。您还可以将适配器权重与基本模型合并，以消除推理延迟。

    

这种方法有很多优点：* LoRA 通过大幅减少可训练参数的数量，使微调更加高效。
* 原始预训练权重保持冻结状态，这意味着您可以拥有多个轻量级便携式 LoRA 模型，用于在其之上构建的各种下游任务。
* LoRA 与其他参数高效方法正交，并且可以与其中许多方法组合。
* 使用 LoRA 微调的模型的性能与完全微调的模型的性能相当。

原则上，LoRA 可以应用于神经网络中权重矩阵的任何子集，以减少可训练参数的数量。然而，为了简单性和进一步的参数效率，LoRA 通常仅应用于 Transformer 模型中的注意力块 - 它也可能值得针对其他层。 LoRA 模型中可训练参数的数量取决于更新矩阵的大小，而更新矩阵的大小主要由秩`r` 和原始权重矩阵的形状决定。您可以根据不同的用例来初始化低秩矩阵 - 任务感知（CoRDA、EVA）、更快的收敛（PiSSA）、减轻量化（LoftQ） - 仅举几个用例。了解不同的初始化[below](#initialization)。 LoRA 的默认初始化是无操作，逐步学习新行为，而不会对现有模型造成太大干扰。

## 用法

低秩更新矩阵的大小由 *rank* 或 `r` 确定。更高的排名意味着模型有更多的参数需要训练，但也意味着模型有更多的学习能力。在以下示例中，您将定位注意力块的 *query* 和 *value* 矩阵。其他需要设置的重要参数是`lora_alpha`（缩放因子）、`bias`（是否应该训练`none`、`all`或仅训练LoRA偏差参数）和`modules_to_save`（除了要训练和保存LoRA层之外的模块）。所有这些参数以及更多参数都可以在 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 中找到。

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
"trainable params: 667,493 || all params: 86,543,818 || trainable%: 0.7712775047664294"
```

## 基准概述

<iframe
    src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=LORA"
	frameborder="0"
	width="850"
	height="1000"
>

## 初始化LoRA权重的初始化由[LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)中的参数`init_lora_weights`控制。默认情况下，PEFT 使用权重 A 的 Kaiming-uniform 和权重 B 的零来初始化 LoRA 权重，从而产生恒等变换（与参考[implementation](https://github.com/microsoft/LoRA)相同）。

也可以通过`init_lora_weights="gaussian"`。顾名思义，这用高斯分布初始化权重 A，权重 B 为零（这就是 [Diffusers](https://huggingface.co/docs/diffusers/index) 初始化 LoRA 权重的方式）。

```py
from peft import LoraConfig

config = LoraConfig(init_lora_weights="gaussian", ...)
```

还有一个设置`init_lora_weights=False`的选项，这对于调试和测试很有用。这应该是您唯一一次使用此选项。选择此选项时，LoRA 权重将被初始化，这样它们*不会*导致恒等变换。

```py
from peft import LoraConfig

config = LoraConfig(init_lora_weights=False, ...)
```

[PiSSA](https://huggingface.co/papers/2404.02948) 使用主奇异值和奇异向量初始化 LoRA 适配器。这种简单的修改使 PiSSA 能够比 LoRA 更快地收敛，并最终获得卓越的性能。此外，与 QLoRA 相比，PiSSA 减少了量化误差，从而实现了进一步的增强。将初始化方法配置为“pissa”，这可能需要几分钟才能在预训练模型上执行 SVD：
```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="pissa", ...)
```
或者，执行快速 SVD，只需几秒钟。迭代次数决定了误差和计算时间之间的权衡：
```python
lora_config = LoraConfig(init_lora_weights="pissa_niter_[number of iters]", ...)
```
有关 PiSSA 的详细使用说明，请关注[these instructions](https://github.com/huggingface/peft/tree/main/examples/pissa_finetuning)。

[MiCA](https://arxiv.org/abs/2604.01694)（次要组件适应）是 PiSSA 的补充：MiCA 使用*次要*组件，而不是从*主要*奇异组件进行初始化。具体来说，通过`W = U Σ V^T`，MiCA 设置`B = U[:, -r:]`（与最小奇异值相关的`r`左奇异向量）和`A = 0`。训练时，仅更新`A`； `B` 被冻结了。直觉是，预训练任务很大程度上未使用次奇异方向，因此提供了一个更“可塑”的子空间，用于注入新知识，同时保留预训练能力。

因为初始化时的`A == 0`，适配器贡献`B · A == 0`和模型输出在步骤0处被精确保留——不需要对基本权重进行剩余减法（与PiSSA不同）。由于只有`A`是可训练的，因此匹配`r`的可训练参数数量大约是LoRA的一半。

```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="mica", r=16, target_modules=["q_proj", "v_proj"], ...)
```MiCA目前支持`nn.Linear`和`nn.Embedding`目标模块。所选的等级必须满足线性层的`r <= min(in_features, out_features)`和嵌入层的`r <= min(num_embeddings, embedding_dim)`。详细使用方法请参见[these instructions](https://github.com/huggingface/peft/tree/main/examples/mica_finetuning)。

MiCA 主要用于持续预训练/领域自适应预训练。在该设置中，使用基本模型（而不是指令或聊天调整的检查点）进行 SVD 初始化和 MiCA 训练。训练后，将适配器合并到基本模型权重中，并使用生成的适应基本模型作为后续指令/聊天调整的起点。

[CorDA](https://huggingface.co/papers/2406.05223) 根据下游任务上下文的权重分解构建任务感知的 LoRA 适配器，以学习（指令预览模式，IPM）或维护世界知识（知识保留模式，KPM）。  KPM 不仅在微调任务上比 LoRA 取得了更好的性能，而且还减轻了预先训练的世界知识的灾难性遗忘。  当不关心保留预训练知识时，IPM 受到青睐，因为它可以进一步加速收敛并增强微调性能。您需要将初始化方法配置为“corda”，并指定IPM或KPM的模式以及收集协方差矩阵的数据集。

```py
@torch.no_grad()
def run_model():
    # Assume `model` and `dataset` is in context...
    model.eval()
    for batch in dataset:
        model(**batch)

corda_config = CordaConfig(
    corda_method="kpm",
)
lora_config = LoraConfig(
    init_lora_weights="corda",
    corda_config=corda_config,
)
preprocess_corda(model, lora_config, run_model=run_model)
peft_model = get_peft_model(model, lora_config)
```

有关 CorDA 的详细使用说明，请参阅[these instructions](https://github.com/huggingface/peft/tree/main/examples/corda_finetuning)。

[OLoRA](https://huggingface.co/papers/2406.01775) 利用 QR 分解来初始化 LoRA 适配器。 OLoRA 通过 QR 分解的因子来转换模型的基本权重，即，它在对权重进行任何训练之前改变权重。这种方法显着提高了稳定性，加快了收敛速度，最终实现了优越的性能。

您只需传递一个附加选项即可使用 OLoRA：
```python
from peft import LoraConfig
config = LoraConfig(init_lora_weights="olora", ...)
```
更多高级用法请参考我们的[documentation](https://github.com/huggingface/peft/tree/main/examples/olora_finetuning)。

[EVA](https://huggingface.co/papers/2410.07170) 对每一层的输入激活执行 SVD，并使用右奇异向量来初始化 LoRA 权重。因此，它是一种数据驱动的初始化方案。此外，EVA 根据“解释方差比”（从 SVD 分析得出的指标）自适应地跨层分配排名。您可以通过设置`init_lora_weights="eva"`并在[LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)中定义[EvaConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.EvaConfig)来使用EVA：
```python
from peft import LoraConfig, EvaConfig
peft_config = LoraConfig(
    init_lora_weights = "eva",
    eva_config = EvaConfig(rho = 2.0),
    ...
)
```
参数`rho` (≥ 1.0) 决定允许多少重新分配。当`rho=1.0`和`r=16`时，LoRA适配器仅限于16个等级，以防止发生任何重新分配。具有重新分布功能的 EVA 的建议值为 2.0，这意味着层允许的最大等级为 2r。

建议在加速器（例如 CUDA GPU、Intel XPU）上执行 EVA 初始化，因为它的速度要快得多。要优化 EVA 的可用内存量，您可以在 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 中使用 `low_cpu_mem_usage` 标志：
```python
peft_model = get_peft_model(model, peft_config, low_cpu_mem_usage=True)
```
然后，调用[initialize_lora_eva_weights()](/docs/peft/v0.20.0/en/package_reference/lora#peft.initialize_lora_eva_weights)初始化EVA权重（大多数情况下，用于eva初始化的数据加载器可以与用于微调的数据加载器相同）：
```python
initialize_lora_eva_weights(peft_model, dataloader)
```
EVA 使用位和字节开箱即用。只需使用 `quantization_config` 初始化模型并像往常一样调用 [initialize_lora_eva_weights()](/docs/peft/v0.20.0/en/package_reference/lora#peft.initialize_lora_eva_weights) 即可。

> [!提示]
> 有关使用 EVA 的更多说明，请参阅我们的[documentation](https://github.com/huggingface/peft/tree/main/examples/eva_finetuning)。在量化 QLoRA 训练的基本模型时，请考虑使用 [LoftQ initialization](https://huggingface.co/papers/2310.08659)，事实证明，它可以提高训练量化模型时的性能。这个想法是初始化 LoRA 权重，以使量化误差最小化。要使用LoftQ，请遵循[these instructions](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning)。

> [!提示]
> 在 [Quantization](../developer_guides/quantization) 指南中详细了解 PEFT 如何与量化配合使用以及如何使用 LoftQ。

初始化 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 的另一种方法是使用 [rank-stabilized LoRA (rsLoRA)](https://huggingface.co/papers/2312.03732) 方法。 LoRA 架构在每次前向传递期间按固定标量缩放每个适配器，该标量在初始化时设置并取决于等级`r`。在原始实现中，标量由 `lora_alpha/r` 给出，但 rsLoRA 使用 `lora_alpha/math.sqrt(r)` 来稳定适配器并通过使用更高的 `r` 提高性能潜力。

```py
from peft import LoraConfig

config = LoraConfig(use_rslora=True, ...)
```

[LoRA-GA](https://hf.co/papers/2407.05000)（具有梯度近似的低阶自适应）初始化适配器
通过对估计梯度执行 SVD 来调整权重，使权重更接近完全微调，从而更快
收敛。

该方法需要一个初始化函数来估计梯度
在开始实际训练之前：

```python
from peft.tuners.lora import preprocess_loraga

def train_step():
    """Run forward and backward passes for gradient estimation."""
    dataloader_iter = iter(grad_dataloader)
    for _ in range(N):
        batch = next(dataloader_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

preprocess_loraga(model, lora_config, train_step)
```

**使用技巧**- **梯度估计**：LoRA-GA 在模型初始化之前需要梯度估计阶段。使用 `preprocess_loraga()` 和 `train_step` 回调来计算少量训练批次（通常为 64-128 批次）的梯度。

- **初始化策略**：LoRA-GA 支持四种方向策略（`direction`）：`"ArBr"`、`"A2rBr"`、`"ArB2r"`（默认）和`"random"`，以及四种缩放策略（`scale`）：`"stable"`（默认）， `"weight_svd"`、`"gd_scale"`、`"unit"`。默认组合提供了收敛速度和稳定性的最佳平衡。

- **基本权重修改**：与标准 LoRA 不同，LoRA-GA 在初始化期间通过减去低秩近似的缩放版本来修改基本模型权重。这可以更好地与完全微调梯度对齐。由于基本权重已修改，请使用 `save_pretrained()` 与 `save_embedding_layers` 参数或 `save_mutated_as_lora` 模式来正确保存适配器。

- **计算开销**：梯度估计在初始化期间增加了少量开销（通常 64 个批次需要 1-2 分钟），但这很快就会被训练期间更快的收敛所摊销。- **兼容性**：LoRA-GA 需要全精度权重，不支持量化模型。可以与 DoRA 等其他 LoRA 变体结合使用。

## 培训

本节展示如何处理更复杂的训练场景，而不是仅将低秩适配器应用于模型并提供数据。

### QLoRA 式培训

PEFT 中的默认 LoRA 设置将可训练权重添加到每个注意块的查询层和值层。但是[QLoRA](https://hf.co/papers/2305.14314)为变压器模型的所有线性层添加了可训练权重，可以提供与完全微调模型相同的性能。要将 LoRA 应用于所有线性层，就像在 QLoRA 中一样，设置`target_modules="all-linear"`（比按名称指定各个模块更容易，名称可能因架构而异）。

```py
config = LoraConfig(target_modules="all-linear", ...)
```

有关如何将量化应用于 PEFT 适配器的更多信息，请参阅[quantization guide](../developer_guides/quantization)。

### 使用 LoRA 进行内存高效层复制用于提高模型性能的一种方法是通过复制模型中的层来扩展模型，以从给定大小的预训练模型构建更大的模型。例如，将 7B 模型增加到 10B 模型，如 [SOLAR](https://huggingface.co/papers/2312.15166) 论文中所述。 PEFT LoRA 以内存高效的方式支持这种扩展，支持使用附加到层复制后的层的 LoRA 适配器进行进一步微调。复制层不占用额外的内存，因为它们共享底层权重，因此唯一需要的额外内存是适配器权重的内存。要使用此功能，您需要使用 `layer_replication` 参数创建一个配置。

```py
config = LoraConfig(layer_replication=[[0,4], [2,5]], ...)
```

假设原始模型有 5 层`[0, 1, 2 ,3, 4]`，这将创建一个具有 7 层排列为 `[0, 1, 2, 3, 2, 3, 4]` 的模型。这遵循 [mergekit](https://github.com/arcee-ai/mergekit) 传递合并约定，其中指定为起始包含和结束排除元组的层序列被堆叠以构建最终模型。最终模型中的每一层都有自己独特的 LoRA 适配器集。[Fewshot-Metamath-OrcaVicuna-Mistral-10B](https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B) 是在 Mistral-7B 上使用此方法训练并扩展到 10B 的模型示例。的
[adapter_config.json](https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B/blob/main/adapter_config.json) 显示了应用此方法进行微调的示例 LoRA 适配器配置。

### 对排名和 alpha（缩放）的细粒度控制

默认情况下，所有以 LoRA 为目标的层都将具有相同的等级 `r` 和相同的 `lora_alpha`（决定 LoRA 缩放），具体取决于 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 中指定的内容。然而，在某些情况下，您可能希望为不同的图层指定不同的值。这可以通过将 `rank_pattern` 和 `alpha_pattern` 参数传递给 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 来实现。这些参数应该是字典，键是图层名称，值是排名/alpha 值。键可以是[regular expressions](https://docs.python.org/3/library/re.html)（正则表达式）。 `rank_pattern`和`alpha_pattern`中未明确提及的所有LoRA层都将采用默认的`r`和`lora_alpha`值。

举个例子，假设我们有一个具有以下结构的模型：

```python
>>> print(model)
Outer(
  (foo): Linear(...)
  (module): Middle(
    (foo): Linear(...)
    (foobar): Linear(...)
    (module): Inner(
      (foo): Linear(...)
      (barfoo): Linear(...)
    )
  )
)
```- `rank_pattern={"foo": 42}` 将匹配所有 3 个 `foo` 层。 `foobar` 和 `barfoo` 均不匹配。
- `rank_pattern={"^foo": 42}`将仅匹配模型的`foo`层，但`module.foo`和`module.module.foo`都不匹配。这是因为使用正则表达式时，`^`表示“字符串的开头”，只有`foo`以`"foo"`开头，其他图层名称都有前缀。
- 出于同样的原因，`rank_pattern={"^module.foo": 42}`仅匹配`module.foo`，但不匹配`module.module.foo`。
- `rank_pattern={"module.foo": 42}` 匹配 `module.foo` 和 `module.module.foo`，但不匹配 `foo`。
- `rank_pattern={"^foo": 42, "^module.module.foo": 55}` 分别匹配 `foo` 和 `module.module.foo`，但不匹配 `module.foo`。
- 无需指示`$`来标记比赛结束，因为这是由PEFT自动添加的。

同样的逻辑也适用于`alpha_pattern`。如果您有疑问，请不要尝试使用正则表达式 - 只需传递具有不同等级/alpha 的每个模块的全名，前面加上 `^` 前缀，您应该就可以了。

### 自动检测可行的目标模块[peft.helpers.KappaTuneSelector](/docs/peft/v0.20.0/en/package_reference/helpers#peft.helpers.KappaTuneSelector) 实现了[KappaTune paper](https://arxiv.org/abs/2506.16289) 的基于条件数的目标选择策略。它扫描每个 `nn.Linear` 模块，对于 MoE 专家权重存储为融合 3D `nn.Parameter` 张量的模型（例如 Llama-4、Qwen3-MoE）以及这些参数，计算每个模块的矩阵条件数 κ = σ_max / σ_min，并选择最各向同性层（最低 κ）。这些各向同性层是微调的理想候选者，因为它们的高熵性质使它们能够更容易地吸收新信息，使专门的各向异性层保持完整，以减轻持续学习期间的灾难性遗忘。

使用 [peft.helpers.find_kappa_target_modules()](/docs/peft/v0.20.0/en/package_reference/helpers#peft.find_kappa_target_modules) 作为单行以获得 `LoraConfig` 的最佳 `target_modules`：

```python
from peft import LoraConfig, get_peft_model
from peft.helpers import find_kappa_target_modules

model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")

targets = find_kappa_target_modules(model, top_p=0.2)
config = LoraConfig(
    target_modules=targets["target_modules"],
    target_parameters=targets["target_parameters"] if stable_modules_dic["target_parameters"] else None,
    r=64,
    lora_alpha=32,
    task_type="CAUSAL_LM",
)
peft_model = get_peft_model(model, config)
```

请参阅完整示例[here](https://github.com/huggingface/peft/blob/main/examples/KappaTune/experiments_kappatune_peft.py)。

### 直接针对`nn.Parameter`通常，您应该使用 `target_modules` 来定位模块（例如 `nn.Linear`）。然而，在某些情况下，这是不可能的。例如，在 HF Transformers 中的许多专家 (MoE) 层混合中，不使用 `nn.Linear`，而是使用 `nn.Parameter`。 PEFT 通常会覆盖 LoRA 的 `forward` 方法，但对于 `nn.Parameter` 则没有。因此，要将LoRA应用于该参数，需要以`target_parameters`为目标。例如，对于 [Llama4](https://huggingface.co/collections/meta-llama/llama-4-67f0c30d9fe03840bc9d0164)，您可以传递：`target_parameters=['feed_forward.experts.gate_up_proj', 'feed_forward.experts.down_proj]`。

请注意，当针对专家参数时，PEFT 可能会增加大量的运行时开销。原因是 PEFT 总是为_每个专家_实现 LoRA 贡献，即使只需要少量专家。在训练期间，这一点不太相关，因为在序列过程中，通常大部分专家至少被激活一次。然而，在推理过程中，通常会使用 KV 缓存，因此我们只需要计算最后一个 token，这意味着只有少量的专家被激活。因此，在 MoE 层上使用 LoRA 可能会导致推理时间大幅减慢。因此，建议合并权重（`model.merge_adapter()`或`model = model.merge_and_unload()`）。这消除了 PEFT 开销。关于这个问题的更详细的调查可以在这个[pull request on MoE optimization](https://github.com/huggingface/peft/pull/3139)上找到。

#### 注意事项

- 目前，此参数允许定位 2-dim 或 3-dim `nn.Parameter`s。假设在 3 维参数的情况下，第 0 维是专家维度。
- 支持使用 `target_parameters` 的多个 LoRA 适配器（通过 `model.add_adapter` 或 `model.load_adapter` 添加），但所有适配器都必须针对同一组参数。

#### MoE 专家参数和 vLLM

Transformers 中的一些 MoE 模型将专家权重存储为 `nn.Parameter` 张量（通常是 3D），而不是 `nn.Linear` 模块。
要将 LoRA 应用于这些专家，请使用 `target_parameters` 并使用 `rank_pattern` 设置每层排名：

```python
num_experts = getattr(model.config, "num_local_experts", None) or model.config.num_experts
effective_r = max(1, r // num_experts)
config = LoraConfig(
    r=r,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    target_parameters=[
        # Mixtral / Qwen3-MoE / GPT-OSS
        "mlp.experts.gate_up_proj",
        "mlp.experts.down_proj",
        # Llama4
        # "feed_forward.experts.gate_up_proj",
        # "feed_forward.experts.down_proj",
    ],
    rank_pattern={
        "experts.gate_up_proj": effective_r,
        "experts.down_proj": effective_r,
    },
)
```

这使得总 LoRA 参数预算与密集层相似（参见 Schulman 等人的[LoRA Without Regret](https://thinkingmachines.ai/blog/lora/)）。  非专家模块使用默认等级`r`。

例如，使用 [vLLM](https://vllm.ai/) 可以使用微调模型进行加速推理，它自 v0.11.2 起支持融合的 MoE 专家层。

### 与 LoRA 一起高效训练代币PEFT LoRA 适配器支持使用 `trainable_token_indices` 参数添加新令牌。这允许在微调特定层的同时调整其他令牌。仅训练指定的令牌，所有其他令牌均保持不变。与训练整个嵌入矩阵不同，它可以节省内存，并且不会丢弃从现有令牌嵌入中学习到的上下文。在底层，该方法使用[TrainableTokensModel](/docs/peft/v0.20.0/en/package_reference/trainable_tokens#peft.TrainableTokensModel)层。

```py
# for layer 'embed_tokens'
config = LoraConfig(trainable_token_indices=[idx_1, idx_2, ...], ...)

# specific embedding layer
config = LoraConfig(trainable_token_indices={'emb_tokens': [idx_1, idx_2, ...]}, ...)
```

在下面的代码片段中，我们展示了如何向模型添加新令牌以及如何与模型中的其他层一起训练它。

```py
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# we define our new tokens and add them to the tokenizer as special tokens
special_tokens = ['<|start_think|>', '<|stop_think|>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# make room for new tokens in the embedding matrix if it isn't big enough already
base_model.resize_token_embeddings(max(len(tokenizer), base_model.model.embed_tokens.num_embeddings))

# typical LoRA config with `trainable_token_indices` targeting embedding layer `embed_tokens`
# and specifically our new tokens we just added
lora_config = LoraConfig(
    target_modules='all-linear',
    trainable_token_indices={'embed_tokens': tokenizer.convert_tokens_to_ids(special_tokens)},
)
peft_model = get_peft_model(base_model, lora_config)

# proceed to train the model like normal
[...]
```

令牌权重与 LoRA 权重一起保存为适配器状态字典的一部分。完全微调和保存嵌入矩阵将存储一个更大的文件。

为了稍微说明可以节省多少 VRAM，我们对完全训练嵌入矩阵 (`modules_to_save=["embed_tokens"]`)、使用 LoRA 进行嵌入矩阵 (`target_modules=[..., "embed_tokens"]`，等级 32) 和可训练令牌 (`trainable_token_indices=[...]`，6 个令牌) 之间进行了基本比较：|           |可训练代币 |       洛拉 |全面微调 |
| --------： | :--------------: | :--------: | :--------------: |
|显存 |        15,562 MB |   15,581MB |     〜16,500MB |
|影响力 |         6 代币 |所有代币 |       所有代币 |

### 重量绑定

许多因果 LM 使用**权重绑定**，其中两个或多个权重共享相同的基础参数。在最常见的情况下，输入嵌入权重 (`embed_tokens`) 和输出投影权重 (`lm_head`) 共享相同的张量。这是因为它减少了参数并且通常保持模型质量。

当 PEFT 作为微调目标时，如何处理这些绑定权重并不总是很明显。对于 LoRA，[LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 上的`ensure_weight_tying` 控制 PEFT 是否应显式地将适配器端更新与这些层相关联。实际上，这可能会影响 `modules_to_save`、`target_modules` 和 `trainable_token_indices`。请注意，在命名层时，此逻辑部分依赖于约定（`"embed_tokens"`、`"lm_head"`），如果不使用这些约定，则无法保证正常工作。

下表总结了预期行为。

#### `modules_to_save`|基本模型权重绑定| `ensure_weight_tying` | `LoraConfig` 形状 |行为 |
|------------------------------------|------------------------------------|--------------------------------------------------------------------------------|----------------------------------------------------------------------------|
|没有 | `False` | `modules_to_save=["embed_tokens"]` 或 `["lm_head"]` |仅在选定图层上添加 `ModulesToSaveWrapper` |
|没有 | `True` | `modules_to_save=["embed_tokens"]` 或 `["lm_head"]` |警告，然后仅在选定图层上添加 `ModulesToSaveWrapper` |
|是的 | `False` | `modules_to_save=["embed_tokens"]` 或 `["lm_head"]` |视为单独 |
|是的 | `True` | `modules_to_save=["embed_tokens"]` 或 `["lm_head"]` |将绑好的层包裹起来并保持包装纸绑好|
|没有 | `False` | `modules_to_save=["embed_tokens", "lm_head"]` |视为单独 |
|没有 | `True` | `modules_to_save=["embed_tokens", "lm_head"]` |警告，然后单独对待 ||是的 | `False` | `modules_to_save=["embed_tokens", "lm_head"]` |警告，然后单独对待 |
|是的 | `True` | `modules_to_save=["embed_tokens", "lm_head"]` |保持`ModulesToSaveWrapper`系好 |

#### `target_modules`

|基本模型权重绑定| `ensure_weight_tying` | `LoraConfig` 形状 |行为 |
|------------------------------------|------------------------------------|----------------------------------------------------------------|--------------------------------------------------------|
|没有 | `False` | `target_modules=["embed_tokens"]` 或 `["lm_head"]` |仅在选定层上添加 LoRA |
|没有 | `True` | `target_modules=["embed_tokens"]` 或 `["lm_head"]` |警告，然后仅在选定层上添加 LoRA |
|是的 | `False` | `target_modules=["embed_tokens"]` 或 `["lm_head"]` |视为单独 |
|是的 | `True` | `target_modules=["embed_tokens"]` 或 `["lm_head"]` |保持 LoRA 适配器连接 |
|没有 | `False` | `target_modules=["embed_tokens", "lm_head"]` |视为单独 ||没有 | `True` | `target_modules=["embed_tokens", "lm_head"]` |警告，然后单独对待 |
|是的 | `False` | `target_modules=["embed_tokens", "lm_head"]` |警告，然后单独对待 |
|是的 | `True` | `target_modules=["embed_tokens", "lm_head"]` |保持 LoRA 适配器连接 |

#### `trainable_token_indices`

对于可训练的令牌，我们有额外的复杂性，即使 LM 头和嵌入是捆绑在一起的，作为用户我可能想在它们上微调*不同的*令牌。因此，在下面的示例表中，我们区分了微调相同标记和微调不同标记。|基本模型权重绑定| `ensure_weight_tying` | `LoraConfig` 形状 |行为 |
|------------------------------------|------------------------------------|------------------------------------------------------------------------------------------------|------------------------------------------------|
|没有 | `False` | `trainable_token_indices=[1, 2, 3]` |仅嵌入嵌入的可训练令牌 |
|没有 | `True` | `trainable_token_indices=[1, 2, 3]` |警告，然后仅在嵌入上训练令牌 |
|是的 | `False` | `trainable_token_indices=[1, 2, 3]` |绑定可训练代币 |
|是的 | `True` | `trainable_token_indices=[1, 2, 3]` |绑定可训练代币 |
|没有 | `False` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [1, 2]}` |视为单独 |
|没有 | `True` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [1, 2]}` |警告，然后单独对待 ||是的 | `False` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [1, 2]}` |绑定可训练代币 |
|是的 | `True` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [1, 2]}` |绑定可训练代币 |
|没有 | `False` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [3, 4]}` |视为单独 |
|没有 | `True` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [3, 4]}` |警告，然后单独对待 |
|是的 | `False` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [3, 4]}` |视为单独 |
|是的 | `True` | `trainable_token_indices={"lm_head": [1, 2], "embed_tokens": [3, 4]}` |错误 |

对于用户来说，这意味着：

- 一般来说，如果你想微调绑定的权重并希望保持它们绑定，请通过`ensure_weight_tying=True`。
- 如果你的基础模型的权重被解开，`ensure_weight_tying=True`不能强制解绑，只会发出警告。
- 对于`trainable_token_indices`，绑定层必须使用与`ensure_weight_tying=True`相同的令牌索引。

## 优化器

LoRA 训练可以选择包含特殊用途的优化器。目前PEFT支持LoRA-FA和LoRA+。

### LoRA-FA 优化器使用 LoRA-FA 可以使 LoRA 训练更加有效和高效，如[LoRA-FA](https://huggingface.co/papers/2308.03303) 中所述。 LoRA-FA通过固定矩阵A，仅调整矩阵B来减少激活内存消耗。在训练过程中，优化B的梯度以逼近全参数微调梯度。而且，LoRA-FA的内存消耗对rank不敏感（因为它消除了$A$的激活），因此可以在不增加内存消耗的情况下通过扩大lora的rank来提高性能。

```py
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_lorafa_optimizer
from transformers import Trainer, get_cosine_schedule_with_warmup

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

config = LoraConfig(...)
model = get_peft_model(base_model, config)

optimizer = create_lorafa_optimizer(
    model=model,
    r=128,
    lora_alpha=32,
    lr=7e-5,
)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=100,
    num_training_steps=1000,
)

trainer = Trainer(
    ...,
    optimizers=(optimizer, scheduler),
)
```

### LoRA+ 优化 LoRA

LoRA 训练可以使用 [LoRA+](https://huggingface.co/papers/2402.12354) 进行优化，它对适配器矩阵 A 和 B 使用不同的学习率，可将微调速度提高多达 2 倍，将性能提高 1-2%。

```py
from peft import LoraConfig, get_peft_model
from peft.optimizers import create_loraplus_optimizer
from transformers import Trainer
import bitsandbytes as bnb

base_model = ...
config = LoraConfig(...)
model = get_peft_model(base_model, config)

optimizer = create_loraplus_optimizer(
    model=model,
    optimizer_cls=bnb.optim.Adam8bit,
    lr=5e-5,
    loraplus_lr_ratio=16,
)
scheduler = None

...
trainer = Trainer(
    ...,
    optimizers=(optimizer, scheduler),
)
```

## 训练后

本节展示了经过训练的适配器的潜在后处理方法。

### 将 LoRA 权重合并到基本模型中虽然 LoRA 明显更小且训练速度更快，但由于单独加载基础模型和 LoRA 适配器，您可能会在推理过程中遇到延迟问题。为了消除延迟，请使用 [merge_and_unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.merge_and_unload) 函数将适配器权重与基本模型合并。这允许您将新合并的模型用作独立模型。 [merge_and_unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.merge_and_unload) 函数不会将适配器权重保留在内存中。

下图解释了 LoRA 适配器合并的直觉：

    

我们在下面的代码片段中展示了如何使用 PEFT 运行它。

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()
```

将返回的模型分配给变量并使用它很重要，[merge_and_unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.merge_and_unload)不是就地操作。如果您需要保留权重的副本，以便以后可以取消合并适配器或删除并加载不同的权重，则应该使用 [merge_adapter()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.merge_adapter) 函数。现在您可以选择使用 [unmerge_adapter()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.unmerge_adapter) 返回基本模型。

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model.merge_adapter()

# unmerge the LoRA layers from the base model
model.unmerge_adapter()
```

[add_weighted_adapter()](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraModel.add_weighted_adapter) 函数可用于根据用户在 `weights` 参数中提供的加权方案将多个 LoRA 合并到新适配器中。下面是一个端到端的示例。

首先加载基础模型：

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1", dtype=torch.float16, device_map="auto"
)
```

然后我们加载第一个适配器：

```python
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id, adapter_name="sft")
```

然后加载不同的适配器并将其与第一个适配器合并：```python
weighted_adapter_name = "sft-dpo"
model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")
model.add_weighted_adapter(
    adapters=["sft", "dpo"],
    weights=[0.7, 0.3],
    adapter_name=weighted_adapter_name,
    combination_type="linear"
)
model.set_adapter(weighted_adapter_name)
```

> [!提示]
> `combination_type` 有多种支持的方法。更多详情请参阅[documentation](../package_reference/lora#peft.LoraModel.add_weighted_adapter)。请注意，当使用 `torch.float16` 或 `torch.bfloat16` 作为数据类型时，不支持“svd”作为 `combination_type`。

现在，进行推理：

```python
device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    generate_ids = model.generate(**inputs, max_length=30)
outputs = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(outputs)
```

### 通过入侵者降维恢复基本模型性能

论文[LoRA vs Full Fine-tuning: An Illusion of Equivalence](https://huggingface.co/papers/2410.21228)认为，LoRA 训练在权重中引入了额外的维度，这些维度与已经学习的权重几乎没有共同点，并导致忘记已经学习的信息。 PEFT 实施[peft.tuners.lora.intruders.reduce_intruder_dimension()](/docs/peft/v0.20.0/en/package_reference/lora#peft.tuners.lora.intruders.reduce_intruder_dimension) 中建议的缓解措施。

缓解措施将采用加载了 LoRA 的 PEFT 模型，并创建一个新的、经过修改的适配器，该适配器与现有适配器和现在的活动适配器一起加载。

用法示例：

```python
from peft.tuners.lora.intruders import reduce_intruder_dimension

peft_model = AutoPeftModelForCausalLM.from_pretrained('hubnemo/llama-3.2b-metamathqa-lora64')

reduce_intruder_dimension(
    peft_model,
    mitigation_lambda=0.75,
)

peft_model.generate(...)
```有一些超参数可用于调整缓解措施的有效性，但正如本文图 8 所示，它始终是适配器学习的任务准确性和忘记基本模型知识之间的权衡。缓解措施将从适配器中删除信息，以减少忘记先前知识的影响，但这也意味着适配器学到的任务的一些信息也会丢失。

虽然默认值设置为在两个因素之间提供良好的权衡，但不能保证默认值适用于您的适配器、模型和数据，因此明智的做法是准备一个基准来衡量效果。

## 加载适配器

可以使用[load_adapter()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.load_adapter)将适配器加载到预训练模型上，这对于尝试权重未合并的不同适配器非常有用。使用 [set_adapter()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.set_adapter) 功能设置活动适配器重量。

```py
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
peft_model_id = "alignment-handbook/zephyr-7b-sft-lora"
model = PeftModel.from_pretrained(base_model, peft_model_id)

# load different adapter
model.load_adapter("alignment-handbook/zephyr-7b-dpo-lora", adapter_name="dpo")

# set adapter as active
model.set_adapter("dpo")
```

要返回基本模型，您可以使用 [unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.unload) 卸载所有 LoRA 模块或 [delete_adapter()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.delete_adapter) 完全删除适配器。 [unload()](/docs/peft/v0.20.0/en/package_reference/tuners#peft.tuners.tuners_utils.BaseTuner.unload) 不是就地操作，请记住将返回的模型分配给变量并使用它。

```py
# unload adapter
model = model.unload()

# delete adapter
model.delete_adapter("dpo")
```

## 张量并行性LoRA 支持 Transformers 提供的[Tensor Parallelism (TP)](https://huggingface.co/docs/transformers/main/en/perf_train_gpu_many#tensor-parallelism)。当基础模型加载`tp_plan`时，PEFT会自动检测每个目标模块的TP配置，并向LoRA适配器权重添加适当的钩子，以便它们正确参与张量并行计算。

> [!警告]
> LoRA 的张量并行支持需要 `transformers >= 5.4.0`。

用法与标准 LoRA 工作流程相同 - 只需使用 `tp_plan` 加载基本模型，然后用 PEFT 包装它：

```py
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", tp_plan="auto")
lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
```

通过 `save_pretrained` / `from_pretrained` 照常保存和加载工作。 PEFT 在保存之前将分片适配器权重收集回完整张量，因此检查点是可移植的并且独立于训练期间使用的设备数量。

## 推论

本部分展示了您在 LoRA 推理期间可以执行的操作，例如解耦适配器。

### 激活 LoRA (aLoRA)激活的 LoRA (aLoRA) 是一种用于因果 LM 的低等级适配器架构，它重用现有的基本模型 KV 缓存以实现更高效的推理。这种方法最适合大多数任务/代依赖于基本模型的推理管道，但使用 LoRA 适配器来执行链内的专门任务。例如，检查或纠正基本模型生成的输出。在这些设置中，推理时间可以加快一个数量级或更多。有关 aLoRA 的更多信息和许多示例用例，请参阅 aLoRA [paper](https://huggingface.co/papers/2504.12397)。该技术扫描每个输入中最后一次出现的调用序列 (`alora_invocation_tokens`)（可以短至 1 个令牌）。它从调用序列开始时激活令牌上的适配器权重。调用序列之后的任何输入也会被调整，并且所有生成的令牌将使用调整后的权重。先前令牌的权重保持不变，由于因果 LM 中的因果注意掩码，使得这些令牌的缓存可以与基本模型缓存互换。用法与标准 LoRA 非常相似。主要区别在于创建适配器时必须指定调用顺序：

```py
from peft import LoraConfig

config = LoraConfig(alora_invocation_tokens=alora_invocation_tokens, task_type="CAUSAL_LM", ...)
```

alora_inspiration_tokens` 是整数令牌 id 的列表。给定所需的调用字符串，可以通过以下方式获得：
```py
invocation_string = "placeholder"
alora_invocation_tokens = tokenizer.encode(invocation_string, add_special_tokens=False).
```
分词器是基本模型的分词器。使用 `add_special_tokens=False` 可以避免在搜索字符串中添加 `SOS`/`EOS` 标记（这很可能会导致搜索失败）。**注释**
* aLoRA 仅支持 `task_type=CAUSAL_LM` 任务，因为它专注于缓存重用。
* 由于权重适应较少的代币，通常（并非总是）aLoRA 需要比 LoRA 更高的排名 (`r`)。 `r=32` 可以是一个很好的起点。
* 根据定义，aLoRA 权重无法合并到基本模型中，因为适配器权重有选择地应用于令牌子集。尝试合并会引发错误。
* 尚不支持束搜索。
* 通常不建议将基本模型中不存在的新标记添加到标记生成器中。这可能会使在重叠上下文上运行的基本模型和适配器模型的目标用例变得复杂。您可以通过在训练适配器之前将 [trainable tokens](../package_reference/trainable_tokens) 添加到基础模型来解决此问题。

#### 调用顺序和SFT设计的选择

您必须添加 `alora_invocation_tokens` 序列，因为它不会自动添加。我们建议尽早激活适配器权重（在任何特定于适配器的提示开始时），但在任何长输入之后，以最大限度地提高模型性能而不影响缓存重用。与任何模型一样，
训练和测试之间的格式应该一致。考虑以下示例，其中基本模型有一个聊天模板，
目标是训练适配器生成所需的输出。

* 选项 1：如果没有特定于任务的提示，即输入是带有 `assistant` 提示的聊天记录，则聊天模板的 `assistant` 提示（例如 `<|start_of_role|>assistant<|end_of_role|>`）是调用字符串的自然选择。请参阅该模特的聊天模板以查找该模特的提示。
* 选项 2：如果适配器有一个特定于任务的提示来描述适配器正在学习的任务，并且该提示在生成之前被放置为 `user` 轮，则聊天模板的 `user` 提示（例如 `<|start_of_role|>user<|end_of_role|>`）是调用字符串的自然选择。

确定调用字符串后，获取模型分词器并获得`alora_invocation_tokens`：
```py
alora_invocation_tokens = tokenizer.encode(invocation_string, add_special_tokens=False).
```

推理设置示例位于 [alora finetuning](https://github.com/huggingface/peft/blob/main/examples/alora_finetuning/alora_finetuning.py)。

> [!注意]
> 如果对调用字符串使用自定义字符串，请确保字符串的开头和结尾是特殊标记，以避免边界处的标记化问题。要了解原因，请假设“a”、“b”、“c”和“ab”是标记生成器中的标记（分别为数字 1、2、3、4）。假设您的 alora_inspiration_tokens = [2, 3]。现在假设您的输入字符串是“abc”。因为“ab”是一个标记，所以它将被标记为 [4,3]。因此，尽管有字符串“bc”，但 alora_invocation_tokens 将无法找到。然而，如果调用字符串的开头和结尾是特殊标记，则这种失败情况永远不会发生，因为特殊标记永远不会与其他字符标记为相同的标记。#### 使用（和重用）缓存进行生成
aLoRA 的主要目的是使 KV 缓存在调用序列之前**在基本模型和 aLoRA 适配器模型之间可互换，因为基本 KV 值和适配 KV 值不兼容。具体来说，在一个模型生成期间存储的键和值可以在后续生成中使用，以避免上下文令牌的昂贵的预填充操作。当基础模型和 aLoRA 适配器之间共享缓存时，有两种主要模式：
1.基础模型生成了一些东西，然后调用aLoRA适配器进行后续生成。例如，基础模型回答一个问题，经过训练来检测幻觉的 aLoRA 检查基础模型的响应。2. aLoRA 适配器已生成某些内容，并且调用基本模型或不同的 aLoRA 适配器来进行后续生成，其中与原始 aLoRA 存在部分上下文重叠。例如，用户提供查询，aLoRA 重写查询以使其更加独立并改进 RAG 系统中的检索。然后，检索文档并将其加载到上下文中，aLoRA 检查这些文档是否与问题相关，然后基础模型生成答案。

为了演示使用缓存时的上述行为，我们使用 `transformers` 中的 [DynamicCache](https://huggingface.co/docs/transformers/en/kv_cache)。请注意确保调整后的缓存值不会与基本缓存值混合。特别是，当存在部分上下文重叠时（模式 2），需要额外的步骤来共享缓存。

**模式 1：基础模型后跟 aLoRA** 这里，基础模型的整个输入和生成连同调用序列一起输入到 aLoRA 适配器中：
```
from transformers import DynamicCache
...
cache = DynamicCache()
inputs_base = tokenizer(prompt_base, return_tensors="pt")
# Generate from base model and save cache
with model_alora.disable_adapter():
    output = model_alora.generate(inputs_base["input_ids"].to(device),attention_mask=inputs_base["attention_mask"].to(device),past_key_values = cache,return_dict_in_generate=True)
output_text_base = tokenizer.decode(output.sequences[0])
cache = output.past_key_values

# Generate with aLoRA adapter from cache
prompt_alora = output_text + INVOCATION_STRING
inputs_alora = tokenizer(prompt_alora, return_tensors="pt").to(device)
output = model_alora.generate(**inputs_alora, past_key_values=cache)
output_text_alora = tokenizer.decode(output[0])

# Note: cache is now tainted with adapter values and cannot be used in base model from here on!
```

**模式 2：aLoRA 生成，然后是具有部分上下文重叠的基础模型（或另一个 aLoRA）** 在这里，我们使用基础模型预填充共享上下文，然后生成。

```
from transformers import DynamicCache
import copy
...
cache = DynamicCache()
inputs_shared = tokenizer(prompt_shared, return_tensors="pt").to(device)

# Prefill from base model and save cache
with model_alora.disable_adapter():
    with torch.no_grad():
        model_alora(**inputs_shared, past_key_values=cache)
cache_copy = copy.deepcopy(cache)

# Generate from aLoRA using prefilled cache
prompt_alora = prompt_shared + INVOCATION_STRING
inputs_alora = tokenizer(prompt_alora, return_tensors="pt").to(device)
output = model_alora.generate(**inputs_alora, past_key_values=cache)
output_text_alora = tokenizer.decode(output[0])

# Generate from base model using saved cache not tainted by aLoRA KV values
prompt_base = prompt_shared
inputs_base = tokenizer(prompt_base, return_tensors="pt").to(device)
with model_alora.disable_adapter():
    output = model_alora.generate(**inputs_base, past_key_values=cache_copy)
output_text_base = tokenizer.decode(output[0])
```### 同一批中不同 LoRA 适配器的推理

通常，每个推理批次必须在 PEFT 中使用相同的适配器。这有时会很烦人，因为我们可能有一些批次包含要与不同 LoRA 适配器一起使用的样本。例如，我们可以拥有一个适用于英语的基本模型和另外两个 LoRA 适配器，一个用于法语，一个用于德语。通常，我们必须拆分批次，使每个批次仅包含一种语言的样本，我们不能在同一批次中组合不同的语言。

值得庆幸的是，可以使用 `adapter_name` 参数在同一批次中混合不同的 LoRA 适配器。下面，我们通过一个示例来展示其在实践中的运作方式。首先，让我们加载基本模型（英语）和两个适配器（法语和德语），如下所示：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_id = ...
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id)
# load the LoRA adapter for French
peft_model = PeftModel.from_pretrained(model, <path>, adapter_name="adapter_fr")
# next, load the LoRA adapter for German
peft_model.load_adapter(<path>, adapter_name="adapter_de")
```现在，我们想要在包含所有三种语言的示例上生成文本：前三个示例是英语，接下来的三个示例是法语，最后三个示例是德语。我们可以使用 `adapter_names` 参数来指定每个样本使用哪个适配器。由于我们的基本模型用于英语，因此我们对这些样本使用特殊字符串 `"__base__"`。对于接下来的三个示例，我们指出了法国 LoRA 微调的适配器名称，在本例中为`"adapter_fr"`。对于最后三个示例，我们指出了德国 LoRA 微调的适配器名称，在本例中为`"adapter_de"`。这样，我们就可以在一个批次中使用基本模型和两个适配器。

```python
inputs = tokenizer(
    [
        "Hello, my dog is cute",
        "Hello, my cat is awesome",
        "Hello, my fish is great",
        "Salut, mon chien est mignon",
        "Salut, mon chat est génial",
        "Salut, mon poisson est super",
        "Hallo, mein Hund ist süß",
        "Hallo, meine Katze ist toll",
        "Hallo, mein Fisch ist großartig",
    ],
    return_tensors="pt",
    padding=True,
)

adapter_names = [
    "__base__", "__base__", "__base__",
    "adapter_fr", "adapter_fr", "adapter_fr",
    "adapter_de", "adapter_de", "adapter_de",
]
output = peft_model.generate(**inputs, adapter_names=adapter_names, max_new_tokens=20)
```

请注意，此处的顺序并不重要，即批次中的样本不需要按适配器进行分组，如上例所示。我们只需要确保 `adapter_names` 参数与样本正确对齐。

此外，同样的方法还适用于 `modules_to_save` 功能，该功能允许在不同的 LoRA 适配器上保存和重用特定的神经网络层，例如用于分类任务的自定义头。

#### 注意事项

使用此功能有一些缺点，即：- 它仅适用于推理，不适用于训练。
- 使用`with model.disable_adapter()`上下文禁用适配器优先于`adapter_names`。
- 当使用 `merge_adapter` 方法将某些适配器权重与基本权重合并时，您无法通过 `adapter_names`。请先通过调用`model.unmerge_adapter()`取消合并所有适配器。
- 由于显而易见的原因，在调用`merge_and_unload()`后不能使用此功能，因为在这种情况下所有LoRA适配器都将合并到基本权重中。
- 此功能目前不适用于 DoRA，因此如果您想使用它，请在您的 `LoraConfig` 中设置 `use_dora=False`。
- `modules_to_save`功能目前仅支持`Linear`、`Embedding`、`Conv2d`和`Conv1d`类型的图层。
- 使用 `adapter_names` 进行推理会产生预期开销，特别是当批次中不同适配器的数量很高时。这是因为批量大小有效地减少到每个适配器的样本数量。如果运行时性能是您的首要任务，请尝试以下操作：
  - 增加批量大小。
  - 尽量避免同一批次中有大量不同的适配器，最好是同质批次。这可以通过使用相同的适配器缓冲样本并仅使用少数不同的适配器执行推理来实现。- 查看替代实现，例如 [LoRAX](https://github.com/predibase/lorax)、[punica](https://github.com/punica-ai/punica) 或 [S-LoRA](https://github.com/S-LoRA/S-LoRA)，它们专门用于处理大量不同的适配器。

### 组合和重用 LoRA 适配器
#### 箭头
[Arrow](https://huggingface.co/papers/2405.11157)是一种模块化路由算法，旨在结合多个预先训练的特定任务LoRA适配器来解决给定任务，类似于[Polytropon](poly)，但不需要微调。 Arrow 并没有简单地合并所有适配器，而是引入了**无梯度、令牌明智的专家混合 (MoE) 路由机制**。在推理时，它首先通过从 SVD 分解中提取右上角的奇异向量来计算每个 LoRA 的_prototype_。然后通过余弦相似度将每个令牌表示与这些原型进行比较以获得路由系数。令牌被分配给前 k 个最相关的 LoRA 适配器，系数通过 softmax 归一化，并且它们的输出线性组合。这允许有效地重用现有的 LoRA 模块来执行新任务，并带来更强的零样本泛化能力。在 PEFT 中，Arrow 是通过在路由之前纯化特定于任务的 LoRA 适配器来增强 Arrow 的[⟦T332⟧ and ⟦T333⟧. You can also configure parameters such as ⟦T334⟧ (the number of LoRA adapters combined per token), ⟦T335⟧ (the softmax temperature applied to the routing coefficients), and ⟦T336⟧ (for reproducibility).

⟦T39⟧

Furthermore, you can add or remove adapters after calling ⟦T40⟧—for example, to fine-tune a new adapter or discard an unnecessary one. Once the adapters are in place, you can activate the ⟦T41⟧ for inference to use Arrow. Note that if you add a new LoRA adapter after ⟦T42⟧ and want to fine-tune it, you must explicitly set the new adapter as active, since ⟦T43⟧ is activated by default in ⟦T44⟧.

⟦T45⟧

#### GenKnowSub
[GenKnowSub](https://aclanthology.org/2025.acl-short.54/) 来启用的。关键思想是基于 [forgetting-via-negation principle](https://huggingface.co/papers/2212.04089) 减去 LoRA 空间中编码的一般知识，以便任务适配器变得更加孤立并专注于任务相关信号。具体来说，GenKnowSub 从一组通用（非特定任务）LoRA 适配器中估计一个低维“通用”子空间，并在 Arrow 的令牌明智路由之前从每个任务适配器的 LoRA 更新中删除该组件。这通常会提高组合性并减少组合许多任务适配器时的干扰。

在 PEFT 中，通过在 ArrowConfig 中设置 ```use_gks=True``` 并在 ```create_arrow_model``` 中提供 ```general_adapter_paths``` 来启用 GenKnowSub：

```py
from peft import create_arrow_model, ArrowConfig
from transformers import AutoModelForCausalLM

# Loading the model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Creating the Arrow config
arrow_config = ArrowConfig(
    top_k=3,
    router_temperature=1.0,
    use_gks=True,
    rng_seed=42,
)

# Path to task-specific, trained on flan clustered dataset (as we explained before.)
task_specific_adapter_paths = [
        f"TahaBa/phi3-mini-clustered-flan/ts_expert_{i}" for i in range(10)
    ]
# These general adapters are trained on English, German, and French Wikipedia dataset,
# with causal language modelling objective, each pair like: (507 token tsentence, 5 token completion), and the loss computed on the completion
general_adapter_paths = [
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langen/checkpoint-17",
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langfr/checkpoint-35",
        "TahaBa/phi3-mini-general-adapters/cluster0_batch16_prop1.0_langger/checkpoint-17"
    ]

# Creating the Arrow model
model = create_arrow_model(
        base_model=base_model,
        task_specific_adapter_paths=task_specific_adapter_paths,
        general_adapter_paths=general_adapter_paths,
        arrow_config=arrow_config,
    )

# Now the forward path could be called on this model, like a normal PeftModel.
```
为了对常识进行编码，GenKnowSub 在路由开始之前从每个特定于任务的适配器中减去所提供的通用适配器的平均值一次。此外，在这种情况下，仍然支持在调用 ```create_arrow_model``` 后添加或删除适配器的功能（如箭头部分所述）。> [!提示]
> **使用 Arrow + GenKnowSub 时要记住的事项：**
>
> - 所有 LoRA 适配器（特定任务和通用）必须共享相同的 ```rank``` 和 ```target_modules```。
>
> - 这些设置中的任何不一致都会在 ```create_arrow_model``` 中引发错误。
>
> - 支持跨任务适配器具有不同的缩放因子 (```lora_alpha```) - Arrow 会自动处理它们。
>
> - 由于其动态路由行为，不支持合并 ```"arrow_router"```。
>
> - 在create_arrow_model中，任务适配器加载为```task_i```，通用适配器加载为```gks_j```（其中```i```和```j```是索引）。该功能确保```target_modules```、```rank```的一致性，以及适配器是否应用于```Linear```或```Linear4bit```层。然后它添加 ```"arrow_router"``` 模块并激活它。此过程的任何自定义都需要覆盖```create_arrow_model```。
>
> - 此实现与 4 位量化兼容（通过 bitsandbytes）：
>
> ```py
>     from transformers import AutoModelForCausalLM, BitsAndBytesConfig
>     import torch
>
>     # Quantisation config
>     bnb_config = BitsAndBytesConfig(
>             load_in_4bit=True,
>             bnb_4bit_quant_type="nf4",
>             bnb_4bit_compute_dtype=torch.bfloat16,
>             bnb_4bit_use_double_quant=False,
>         )
>
>     # Loading the model
>     base_model = AutoModelForCausalLM.from_pretrained(
>         "microsoft/Phi-3-mini-4k-instruct",
>         dtype=torch.bfloat16,
>         device_map="auto",
>         quantization_config=bnb_config,
>     )
>
>     # Now call create_arrow_model() as we explained before.
>     ```

# API

## LoraConfig[[peft.LoraConfig]]

#### peft.LoraConfig[[peft.LoraConfig]]

```python
peft.LoraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, lora_alpha: int = 8, lora_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: Literal['none', 'all', 'lora_only'] = 'none', use_rslora: bool = False, modules_to_save: Optional[list[str]] = None, init_lora_weights: bool | Literal['gaussian', 'eva', 'olora', 'pissa', 'pissa_niter_[number of iters]', 'corda', 'loftq', 'orthogonal', 'mica'] = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, rank_pattern: Optional[dict] = <factory>, alpha_pattern: Optional[dict] = <factory>, megatron_config: Optional[dict] = None, megatron_core: Optional[str] = 'megatron.core', trainable_token_indices: Optional[Union[list[int], dict[str, list[int]]]] = None, loftq_config: Union[LoftQConfig, dict] = <factory>, eva_config: Optional[EvaConfig] = None, corda_config: Optional[CordaConfig] = None, lora_ga_config: Optional[LoraGAConfig] = None, use_dora: bool = False, velora_config: Optional[Union[VeloraConfig, dict]] = None, alora_invocation_tokens: Optional[list[int]] = None, use_qalora: bool = False, qalora_group_size: int = 16, monteclora_config: Optional[MontecloraConfig] = None, layer_replication: Optional[list[tuple[int, int]]] = None, runtime_config: LoraRuntimeConfig = <factory>, lora_bias: bool = False, target_parameters: Optional[list[str]] = None, use_bdlora: Optional[BdLoraConfig] = None, arrow_config: Optional[ArrowConfig] = None, ensure_weight_tying: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/config.py#L371)

**参数：**

r (`int`)：Lora 注意力维度（“排名”）。target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块（如果模型是 PreTrainedModel，则排除输出层）。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。为了避免定位任何模块（因为您想应用`target_parameters`），请设置`target_modules=[]`。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

lora_alpha (`int`) ：Lora 缩放的 alpha 参数。lora_dropout (`float`) ：Lora 层的 dropout 概率。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) ：LoRA 的偏置类型。可以是“无”、“全部”或“lora_only”。如果是“all”或“lora_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

use_rslora (`bool`) ：设置为 True 时，使用 [Rank-Stabilized LoRA](https://huggingface.co/papers/2312.03732) 将适配器缩放因子设置为 `lora_alpha/math.sqrt(r)`，因为事实证明它效果更好。否则，将使用原来的默认值`lora_alpha/r`。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。init_lora_weights (`bool` | `Literal["gaussian", "eva", "olora", "pissa", "pissa_niter_[number of iters]", "corda", "loftq", "orthogonal", "mica"]`) ：如何初始化适配器层的权重。传递 True（默认值）会导致 Microsoft 的参考实现进行默认初始化，LoRA B 权重设置为 0。这意味着，如果没有进一步训练，LoRA 适配器将是无操作的。将初始化设置为 False 会导致 LoRA A 和 B 随机初始化，这意味着 LoRA 在训练前不是空操作；此设置用于调试目的。传递“gaussian”会导致按线性和层的 LoRA 等级缩放的高斯初始化。传递`'loftq'`以使用LoftQ初始化。通过 `'eva'` 会导致解释方差适应的数据驱动初始化。 EVA 基于层输入激活的 SVD 初始化 LoRA，并由于其适应微调数据的能力而实现了 SOTA 性能。通过 `'olora'` 使用 OLoRA 初始化。通过`'pissa'`会导致主奇异值和奇异向量自适应（PiSSA）的初始化，其收敛速度比LoRA更快，最终实现卓越的性能。此外，与 QLoRA 相比，PiSSA 降低了量化误差，从而实现了其他增强功能。传递`'pissa_niter_[number of iters]'`启动基于Fast-SVD的PiSSA初始化，其中`[number of iters]`表示执行FSVD的子空间迭代次数，并且必须是非负整数。当`[number of iters]`设置为16时，可以在几秒内完成7B模型的初始化，训练效果与使用SVD大致相当。通过`'corda'`会导致面向上下文的分解适应的初始化，其收敛速度比指令预览模式下的PiSSA更快，并且比知识保存模式下的LoRA更好地保存世界知识。通过`"orthogonal"`导致LoRA A和B正交初始化；在此，它类似于 `"olora"`，但基本权重保持不变（要求 `r` 为偶数，目前仅支持线性层）。传递`"mica"`会导致小分量自适应（MiCA）的初始化，它从与最小奇异值相关的基础权重的r个左奇异向量初始化B，将A设置为零，并在训练期间冻结B；仅 A 被更新。目前支持线性层和嵌入层。Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

rank_pattern (`dict`) ：从图层名称或正则表达式到与`r`指定的默认排名不同的排名的映射。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

alpha_pattern (`dict`) ：从图层名称或正则表达式到 alpha 的映射，与 `lora_alpha` 指定的默认 alpha 不同。例如，`{'^model.decoder.layers.0.encoder_attn.k_proj': 16}`。

megatron_config (`Optional[dict]`) ：Megatron 的 TransformerConfig 参数。它用于创建 LoRA 的并行线性层。你可以这样得到，`core_transformer_config_from_args(get_args())`，这两个函数来自威震天。这些参数将用于初始化 Megatron 的 TransformerConfig。当您要将LoRA应用到megatron的ColumnParallelLinear和RowParallelLinear层时，需要指定此参数。megatron_core (`Optional[str]`) ：使用威震天的核心模块，默认为`"megatron.core"`。

trainable_token_indices (`Optional[Union[List[int], dict[str, List[int]]]]`) ：允许您指定有选择地微调哪些标记索引，而无需使用 `peft.TrainableTokensModel` 方法重新训练整个嵌入矩阵。您可以通过两种方式指定令牌索引。您可以指定一个索引列表，然后将其定位到模型的输入嵌入层（或者，如果未找到，则为`embed_tokens`）。或者，您可以指定一个字典，其中键是嵌入模块的名称，值是标记索引列表，例如`{'embed_tokens': [0, 1, ...]}`。请注意，使用 FSDP 进行训练需要`use_orig_params=True`，以避免`requires_grad` 不均匀的问题。

loftq_config (`Optional[LoftQConfig]`) : LoftQ 的配置。如果这不是 None，则 LoftQ 将用于量化主干权重并初始化 Lora 层。还通过`init_lora_weights='loftq'`。请注意，在这种情况下您不应传递量化模型，因为 LoftQ 会量化模型本身。

eva_config (`Optional[EvaConfig]`) : EVA 的配置。至少需要设置数据集参数（使用与微调相同的数据集）。corda_config (`Optional[CordaConfig]`) : CorDA 的配置。如果这不是 None，则 CorDA 将用于构建适配器层。还通过`init_lora_weights='corda'`。

lora_ga_config (`Optional[LoraGAConfig]`) : LoRA-GA的配置。如果通过，那么 LoRA-GA 将用于初始化适配器层。在这种情况下还要设置`init_lora_weights='lora_ga'`。

use_dora (`bool`) ：启用“权重分解低阶适应”(DoRA)。该技术将权重的更新分解为两个部分：大小和方向。方向由普通 LoRA 处理，而幅度由单独的可学习参数处理。这可以提高 LoRA 的性能，尤其是在低级别时。目前，DoRA 仅支持线性和 Conv2D 层。 DoRA 比纯 LoRA 引入了更大的开销，因此建议合并权重进行推理。有关更多信息，请参阅 https://huggingface.co/papers/2402.09353。

velora_config (`Optional[VeloraConfig]`) ：通过提供 VeloraConfig 来启用 VeLoRA。 VeLoRA 将自定义反向传递替换为 LoRA A 投影，该投影存储压缩激活而不是完整输入激活。alora_invocation_tokens (`List[int]`) ：如果不是“无”，则启用“激活的 LoRA”(aLoRA)，其中 alora_invocation_tokens 是适配器的标记化调用字符串（必须出现在所有模型输入字符串中）。该技术仅在 alora_inspiration_tokens 期间和之后有选择地激活令牌上的适配器权重。当在 CausalLM 中使用时，这意味着调用之前的 KV 缓存可以与基本模型（以及以这种方式操作的其他 aLoRA 适配器）的缓存互换。因此，在涉及基础模型推理和适配器推理之间切换的推理管道中（例如代理管道，请参阅论文中的示例），通过节省预填充操作可以实现显着的节省（相对于 LoRA）。 vLLM 上的整体适配器推理加速可能会出现一个数量级或更多，具体取决于共享上下文的长度。请注意，由于权重的选择性应用，合并是不可能的。use_qalora (`bool`) ：目前仅在 GPTQ 中实现。启用量化感知低阶适应 (QALoRA)。该技术将量化感知训练与 LoRA 相结合，以提高量化模型的性能。这可以提高 LoRA 的性能，尤其是在低级别时。目前，QALoRA 仅支持线性层。

qalora_group_size (`int`​​) ：QALoRA池化的组大小参数，控制降维因子。输入维度被汇集到该大小的组中，从而降低了计算成本。较高的值提供更多的压缩，但可能会降低模型质量。此参数确定将多少个原始特征平均在一起以创建一个池化特征。仅当`use_qalora=True`时使用。

monteclora_config (`Optional[MontecloraConfig]`) ：Monteclora（蒙特卡罗低阶适应）的配置。如果通过，Monteclora 将用于在 LoRA 适配器之上添加变分蒙特卡洛采样。有关各个超参数的详细信息，请参阅`MontecloraConfig`。layer_replication (`List[Tuple[int, int]]`) ：通过根据指定的范围堆叠原始模型层来构建新的层堆栈。这允许扩展（或缩小）模型，而无需复制基本模型权重。新层都将附加单独的 LoRA 适配器。

runtime_config (`LoraRuntimeConfig`) ：运行时配置（不保存或恢复）。

lora_bias (`bool`) ：默认为`False`。是否启用 LoRA B 参数的偏置项。通常，应禁用此功能。主要用例是从完全微调的参数中提取 LoRA 权重，因此可以考虑这些参数的偏差。target_parameters (`List[str]`, *可选*) ：要替换为 LoRA 的参数名称的参数名称列表。该参数的行为与 `target_modules` 类似，只是应该传递参数名称。通常，您应该使用 `target_modules` 来定位模块（例如 `nn.Linear`）。然而，在某些情况下，这是不可能的。例如，在 HF Transformer 中的许多专家 (MoE) 层混合中，不使用 `nn.Linear`，而是使用 `nn.Parameter`。 PEFT 通常会覆盖 LoRA 的 `forward` 方法，但对于 `nn.Parameter` 则没有。因此，要将LoRA应用于该参数，需要以`target_parameters`为目标。例如，对于 Llama4，您可以传递：`target_parameters=['feed_forward.experts.gate_up_proj', 'feed_forward.experts.down_proj]`。传递字符串进行正则表达式匹配尚未实现。请注意，当模型编译并使用 `target_parameters` 时，预计会出现重新编译和/或图形中断。建议不要同时使用两者。

use_bdlora (`Optional[BdLoraConfig]`) ：通过提供 BdLoraConfig 来启用 BD-LoRA（块对角 LoRA）。该技术使用 LoRA-A 或 LoRA-B 因子的块对角矩阵，通过消除分布式设置中的通信开销来实现更快的多 LoRA 服务。arrow_config (`Optional[ArrowConfig]`) ：在模型上应用箭头路由的必要配置。

Ensure_weight_tying (`bool`, *可选*) ：peft 初始化后是否绑定权重。这将确保添加到绑定层的适配器也被绑定。这仅适用于通过`modules_to_save`和`target_modules`传递的层。

这是存储[LoraModel](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraModel)配置的配置类。

#### to_dict[[peft.LoraConfig.to_dict]]

```python
to_dict()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/config.py#L902)

以字典形式返回适配器模型的配置。删除运行时配置。

## LoraModel[[peft.LoraModel]]

#### peft.LoraModel[[peft.LoraModel]]

```python
peft.LoraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/model.py#L88)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)) ：Lora 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**退货：** `torch.nn.Module`

劳拉模型。

从预训练的 Transformer 模型创建低阶适配器 (LoRA) 模型。

该方法在 https://huggingface.co/papers/2106.09685 中有详细描述。

示例：

```py
>>> from transformers import AutoModelForSeq2SeqLM
>>> from peft import LoraModel, LoraConfig

>>> config = LoraConfig(
...     task_type="SEQ_2_SEQ_LM",
...     r=8,
...     lora_alpha=32,
...     target_modules=["q", "v"],
...     lora_dropout=0.01,
... )

>>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
>>> lora_model = LoraModel(model, config, "default")
```

```py
>>> import torch
>>> import transformers
>>> from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

>>> rank = ...
>>> target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc_in", "fc_out", "wte"]
>>> config = LoraConfig(
...     r=4, lora_alpha=16, target_modules=target_modules, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM"
... )
>>> quantization_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

>>> tokenizer = transformers.AutoTokenizer.from_pretrained(
...     "kakaobrain/kogpt",
...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
...     bos_token="[BOS]",
...     eos_token="[EOS]",
...     unk_token="[UNK]",
...     pad_token="[PAD]",
...     mask_token="[MASK]",
... )
>>> model = transformers.GPTJForCausalLM.from_pretrained(
...     "kakaobrain/kogpt",
...     revision="KoGPT6B-ryan1.5b-float16",  # or float32 version: revision=KoGPT6B-ryan1.5b
...     pad_token_id=tokenizer.eos_token_id,
...     use_cache=False,
...     device_map={"": rank},
...     torch_dtype=torch.float16,
...     quantization_config=quantization_config,
... )
>>> model = prepare_model_for_kbit_training(model)
>>> lora_model = get_peft_model(model, config)
```**属性**：
- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的型号。
- **peft_config** ([LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig))：Lora 模型的配置。

#### add_weighted_adapter[[peft.LoraModel.add_weighted_adapter]]

```python
add_weighted_adapter(adapters: list[str], weights: list[float], adapter_name: str, combination_type: str = 'svd', svd_rank: int | None = None, svd_clamp: int | None = None, svd_full_matrices: bool = True, svd_driver: str | None = None, density: float | None = None, majority_sign_method: Literal['total', 'frequency'] = 'total')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/model.py#L664)

**参数：**

适配器 (`list`) ：要合并的适配器名称列表。

权重 (`list`) ：每个适配器的权重列表。权重可以是正数或负数，允许添加或减去适配器效应。

adapter_name (`str`) ：新适配器的名称。

组合类型（`str`）：合并类型可以是[`svd`，`linear`，`cat`，`ties`，`ties_svd`，`dare_ties`，`dare_linear`之一， `dare_ties_svd`、`dare_linear_svd`、`magnitude_prune`、`magnitude_prune_svd`]。当使用`cat`combination_type时，生成的适配器的等级等于所有适配器等级的总和（混合​​适配器可能太大并导致OOM错误）。

svd_rank (`int`, *可选*) : svd 输出适配器的等级。如果未提供，将使用合并适配器的最大等级。

svd_clamp (`float`，*可选*)：用于钳位 SVD 分解输出的分位数阈值。如果未提供，则不执行夹紧。默认为无。svd_full_matrices (`bool`, *可选*) ：控制是计算完整的还是简化的 SVD，从而控制返回的张量 U 和 Vh 的形状。默认为 True。

svd_driver (`str`, *可选*) ：要使用的 cuSOLVER 方法的名称。此关键字参数仅在 CUDA 上合并时有效。可以是 [无、`gesvd`、`gesvdj`、`gesvda`] 之一。有关更多信息，请参阅`torch.linalg.svd`文档。默认为无。

密度（`float`，*可选*）：0到1之间的值。0表示所有值都被修剪，1表示没有值被修剪。应与 [`ties`、`ties_svd`、`dare_ties`、`dare_linear`、`dare_ties_svd`、`dare_linear_svd`、`magnintude_prune`、`magnitude_prune_svd`] 一起使用

Majority_sign_method (`str`) ：该方法应该是 ["total", "Frequency"] 之一，用于获取符号值的大小。应与 [`ties`、`ties_svd`、`dare_ties`、`dare_ties_svd`] 一起使用

此方法通过将给定的适配器与给定的权重合并来添加新的适配器。

当使用 `cat`组合类型时，你应该知道生成的适配器的等级将等于
所有适配器等级的总和。因此混合适配器可能会变得太大并导致 OOM
错误。

#### subtract_mutated_init[[peft.LoraModel.subtract_mutated_init]]```python
subtract_mutated_init(output_state_dict: dict[str, torch.Tensor], adapter_name: str, kwargs = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/model.py#L921)

该函数可以通过比较PiSSA/CorDA/OLoRA的参数来计算更新
`output_state_dict` 中的 PiSSA/CorDA/OLoRA 适配器，PiSSA/CorDA/OLoRA 的初始值在
`adapter_name`，从而将 PiSSA/CorDA/OLoRA 转换为 LoRA。

像 PiSSA 这样的方法会删除模型基础权重的一部分进行训练，因此并不容易
交换了。但是，如果您知道初始训练前权重和训练后权重，则可以计算
\Delta W 并将其用作 LoRA 适配器。

计算步骤：

- $W = W_{res} + A_0 B_0$ (PiSSA 初始化)
- $W + \Delta W = W_{res} + A B$ （训练后的 PiSSA）
- $\Delta W = W_{res} + AB - W = W_{res} + AB - W_{res} - A_0 B_0$（为 LoRA 导出 dW）
- $\Delta W = AB - A_0 B_0$

## 实用程序

### ArrowConfig[[peft.ArrowConfig]]

#### pft.ArrowConfig[[peft.ArrowConfig]]

```python
peft.ArrowConfig(top_k: int = 3, router_temperature: float = 1.0, use_gks: bool = False, rng_seed: Optional[int] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/config.py#L120)这是存储 Arrow 和 GenKnowSub 算法配置的子配置类。箭头是一个
结合经过训练的 LoRA 模块来解决新任务的路由算法，在
'https://huggingface.co/papers/2405.11157'。 GenKnowSub 是在组合之前对经过训练的模块进行的细化
通过 Arrow，在“https://aclanthology.org/2025.acl-short.54/”中介绍

### LoftQ[[peft.replace_lora_weights_loftq]]

#### peft.replace_lora_weights_loftq[[peft.replace_lora_weights_loftq]]

```python
peft.replace_lora_weights_loftq(peft_model, model_path: Optional[str] = None, adapter_name: str = 'default', callback: Optional[Callable[[torch.nn.Module, str], bool]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/loftq_utils.py#L193)

**参数：**

peft_model (`PeftModel`) ：要替换权重的模型。必须是具有 LoRA 层的量化 PEFT 模型。

model_path (`Optional[str]`) ：模型安全张量文件的路径。如果模型是 Hugging Face 模型，这将从模型的配置中推断出来。否则，必须提供。

adapter_name (`str`) ：要替换权重的适配器的名称。默认适配器名称是“default”。callback (`Optional[Callable[[PeftModel, str], bool]]`) ：每个模块替换后都会调用的回调函数。回调函数应将模型和当前模块的名称作为输入，并返回一个布尔值，指示是否应保留替换。如果回调返回 False，则替换将被回滚。这对于确认 LoftQ 初始化实际上减少了模型的量化误差非常有用。例如，此回调可以为给定输入生成 logits，并将其与具有相同输入的原始非量化模型的 logits 进行比较，并且仅在有改进时返回 `True`。由于这是一种贪婪优化，因此多次调用此函数可能会产生增量改进。

使用 LoftQ 技术，用位和字节量化模型的 LoRA 权重。

通过从本地存储的安全张量模型加载非量化权重来动态完成替换
文件并初始化 LoRA 权重，以便原始权重和量化权重之间的量化误差
被最小化。由于 pickle 无法实现延迟加载，因此无法支持普通的 PyTorch 检查点文件。

根据模型大小，调用此函数可能需要一些时间才能完成。

### 伊娃

#### EvaConfig[[peft.EvaConfig]]

#### peft.EvaConfig[[peft.EvaConfig]]

```python
peft.EvaConfig(rho: float = 2.0, tau: float = 0.99, use_label_mask: bool = True, label_mask_value: int = -100, whiten: bool = False, adjust_scaling_factors: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/config.py#L244)

**参数：**

rho (`float`) ：EVA 重新分配的 Rho 值 (>= 1.0)。层的最大等级是 lora_r * rho。默认值为 2.0，表示层允许的最大等级为 2r。增加 rho 将允许跨层更高程度的排名重新分配。一些预先训练的模型可能对排名重新分配更敏感。因此，如果性能低于预期，尝试 rho=1.0（不重新分配）可能会有所帮助。

tau (`float`)：提前停止的余弦相似度阈值。比较两个连续 SVD 步骤之间右奇异向量的余弦相似度。如果余弦相似度高于此阈值，则停止 SVD 迭代。默认值为 0.99。use_label_mask (`bool`) : 使用标签掩码进行 EVA 初始化。这意味着 SVD 计算将忽略 labels=label_mask_value 的位置。在大多数情况下，设置 use_label_mask=True 是首选，并且对于多轮对话特别有益。默认值为 True。基于标签掩码过滤掉项目有时会导致批量大小较小，从而导致 SVD 计算不稳定。对于要过滤掉大量批次项目的情况，请设置 use_label_mask=False。

label_mask_value (`int`) ：如果 use_label_mask=True 则查找用于屏蔽被忽略的标记的值。默认值为-100。

Whiten (`bool`) ：对奇异向量应用白化。默认值为 False。事实证明，美白对视觉领域的 EVA 有益。

adjustment_scaling_factors (`bool`) ：在排名重新分配后调整 LoRA 缩放因子。将其设置为 True 意味着调整缩放因子，以便所有 LoRA 梯度具有相同的缩放比例，无论其等级如何。默认为 True。这是子配置类，用于存储通过 EVA 进行数据驱动初始化的配置。 EVA 是
在解释方差适应中引入。

#### 初始化lora_eva_weights[[peft.initialize_lora_eva_weights]]

#### peft.initialize_lora_eva_weights[[peft.initialize_lora_eva_weights]]

```python
peft.initialize_lora_eva_weights(model: Module, dataloader: typing.Optional[collections.abc.Iterable] = None, eva_state_dict: typing.Optional[dict] = None, forward_fn: typing.Optional[collections.abc.Callable] = <function forward_fn_dict at 0x7f29b0f34550>, prepare_model_inputs_fn: typing.Optional[collections.abc.Callable] = <function prepare_model_inputs_fn_language_modeling at 0x7f29b0f34430>, prepare_layer_inputs_fn: typing.Union[collections.abc.Callable, dict[str, collections.abc.Callable], NoneType] = <function prepare_layer_inputs_fn_language_modeling at 0x7f29b0f344c0>, adapter_name: str = 'default', gather_distributed_inputs: bool = True, show_progress_bar: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/eva.py#L654)

**参数：**

model (PeftModel) ：计算 SVD 的 peft 模型。

dataloader (可选[Iterable]) ：用于前向传递的数据加载器。如果没有，则需要提供 eva_state_dict。

eva_state_dict（可选[dict]）：要加载到模型中的state_dict。如果没有，则需要提供数据加载器，并且将使用`get_eva_state_dict`计算state_dict。

forward_fn (Callable) ：用于前向传递的前向函数。采用两个参数：`model` 和 `inputs`。默认行为是`return model(**inputs)`prepare_model_inputs_fn（可选[可调用]）：此函数接收模型输入和peft_config，并将输出传递给`prepare_layer_inputs_fn`。可用于根据原始模型输入修改 SVD 计算的输入。例如，对于语言建模，注意力掩码用于确定哪些索引是填充标记，并且不应用于 SVD。此处定义的任何函数都需要两个参数：`model_input`和`peft_config`。默认使用`peft.tuners.lora.eva.prepare_model_inputs_fn_language_modeling`。

prepare_layer_inputs_fn (Union[Callable, Dict[str, Callable], None]) ：此函数接收层输入、模型输入（可能由`prepare_model_inputs_fn`修改）和层名称，并返回应用于该特定层的 SVD 的输入。此处定义的任何自定义函数都需要三个参数：`layer_input`、`model_input` 和 `layer_name`，并且应返回 2d 张量。默认逻辑可以在 peft.tuners.lora.eva.prepare_layer_inputs_fn_language_modeling 中找到，适用于语言建模。在这种情况下 model_inputs 是用于确定哪些索引应用于 SVD 的掩码（由 `prepare_model_inputs_fn_language_modeling` 创建）。

adapter_name (str) ：要为其初始化权重的适配器的名称。Gather_distributed_inputs (bool) ：是否收集所有等级的层输入。默认值为 True，这意味着在分布式设置中，将从所有等级收集层输入以进行 SVD 计算。对于非分布式设置，该参数将被忽略。如果您在分布式设置中使用非分布式数据加载器，请设置为 False。

show_progress_bar (bool) : 是否显示进度条。默认为 True。

**返回：** `model (torch.nn.Module)`

具有初始化 LoRA 权重的模型。

使用 EVA 方法初始化 LoRA 层的权重。

该函数使用 EVA 方法初始化 LoRA 层的权重。它计算每个适配器的 SVD
层并相应地更新权重。

#### get_eva_state_dict[[peft.get_eva_state_dict]]

#### peft.get_eva_state_dict[[peft.get_eva_state_dict]]

```python
peft.get_eva_state_dict(model: Module, dataloader: Iterable, peft_config: typing.Optional[peft.tuners.lora.config.LoraConfig] = None, forward_fn: typing.Optional[collections.abc.Callable] = <function forward_fn_dict at 0x7f29b0f34550>, prepare_model_inputs_fn: typing.Optional[collections.abc.Callable] = <function prepare_model_inputs_fn_language_modeling at 0x7f29b0f34430>, prepare_layer_inputs_fn: typing.Union[collections.abc.Callable, dict[str, collections.abc.Callable], NoneType] = <function prepare_layer_inputs_fn_language_modeling at 0x7f29b0f344c0>, adapter_name: str = 'default', gather_distributed_inputs: bool = True, show_progress_bar: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/eva.py#L556)

**参数：**

model (torch.nn.Module) ：计算 SVD 的模型。不需要是 PeftModel。

dataloader (Iterable) ：用于前向传递的数据加载器。

peft_config （可选[LoraConfig]）：LoRA 层的配置。仅当 `model` 不是 PeftModel 时才需要。forward_fn (Callable) ：用于前向传递的前向函数。采用两个参数：`model` 和 `inputs`。默认行为是`return model(**inputs)`

prepare_model_inputs_fn（可选[可调用]）：该函数接收模型输入和peft_config，并将输出传递给`prepare_layer_inputs_fn`。可用于根据原始模型输入修改 SVD 计算的输入。例如，对于语言建模，注意力掩码用于确定哪些索引是填充标记，并且不应用于 SVD。此处定义的任何函数都需要两个参数：`model_input`和`peft_config`。默认使用`peft.tuners.lora.eva.prepare_model_inputs_fn_language_modeling`。prepare_layer_inputs_fn (Union[Callable, Dict[str, Callable], None]) ：此函数接收层输入、模型输入（可能由`prepare_model_inputs_fn`修改）和层名称，并返回应用于该特定层的 SVD 的输入。此处定义的任何自定义函数都需要三个参数：`layer_input`、`model_input` 和 `layer_name`，并且应返回 2d 张量。默认逻辑可以在 peft.tuners.lora.eva.prepare_layer_inputs_fn_language_modeling 中找到，适用于语言建模。在这种情况下 model_inputs 是用于确定哪些索引应用于 SVD 的掩码（由 `prepare_model_inputs_fn_language_modeling` 创建）。

adapter_name (str) ：要为其计算 SVD 的适配器的名称。

Gather_distributed_inputs (bool) ：是否收集所有等级的层输入。默认值为 True，这意味着在分布式设置中，将从所有等级收集层输入以进行 SVD 计算。对于非分布式设置，该参数将被忽略。如果您在分布式设置中使用非分布式数据加载器，请设置为 False。

show_progress_bar (bool) : 是否显示进度条。默认为 True。

**退货：** `eva_state_dict (dict)`状态字典包含每层的 SVD 组件。

计算模型中每一层的 SVD。

此函数计算模型中每一层的奇异值分解 (SVD)。它使用增量
计算 SVD 分量的 PCA 方法。该函数还使用以下方法检查计算组件的收敛性
余弦相似度。每层的排名分布是根据解释的方差比确定的。

### LoraGAConfig[[peft.LoraGAConfig]]

#### peft.LoraGAConfig[[peft.LoraGAConfig]]

```python
peft.LoraGAConfig(direction: Literal['ArBr', 'A2rBr', 'ArB2r', 'random'] = 'ArB2r', scale: Literal['stable', 'weight_svd', 'gd_scale', 'unit'] = 'stable', stable_gamma: int = 16)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/config.py#L1033)

**参数：**

Direction (`Literal["ArBr", "A2rBr", "ArB2r", "random"]`) ：将梯度 SVD 分量分配到 lora_A 和 lora_B 矩阵的策略。 - “ArBr”：交替索引（A 采用奇数，B 采用偶数） - “A2rBr”：A 采用索引 [r:2r]，B 采用索引 [:r] - “ArB2r”：A 采用索引 [:r]，B 采用索引 [r:2r]（推荐） - “random”：随机选择索引 默认值：“ArB2r”

scale (`Literal["stable", "weight_svd", "gd_scale", "unit"]`) ：适配器初始化的缩放策略。 - “stable”：使用 gamma 参数进行稳定缩放 - “weight_svd”：基于权重矩阵奇异值进行缩放 - “gd_scale”：基于梯度下降的缩放 - “unit”：无额外缩放 默认值：“stable”stable_gamma (`int`) ：稳定缩放方法的伽玛参数。默认值：16

这是存储 LoRA-GA 初始化配置的子配置类。

LoRA-GA（具有梯度近似的低秩自适应）在初始化期间使用梯度信息来
通过将初始适配器权重与全方向对齐，实现更快的收敛（2-4 倍加速）
微调梯度。

参考：https://arxiv.org/abs/2407.05000

#### 实用程序[[peft.tuners.lora.loraga.estimate_gradients]]

#### peft.tuners.lora.loraga.estimate_gradients[[peft.tuners.lora.loraga.estimate_gradients]]

```python
peft.tuners.lora.loraga.estimate_gradients(model: Module, lora_config: LoraConfig, train_step: Callable)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/loraga.py#L118)

估计 LoRA-GA 初始化的梯度。

此函数仅启用目标模块权重的梯度计算并运行 train_step 回调。这是
比全局启用梯度更节省内存。

#### peft.preprocess_loraga[[peft.preprocess_loraga]]

```python
peft.preprocess_loraga(model: Module, lora_config: LoraConfig, train_step: Callable, cache_file: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/loraga.py#L46)

**参数：**

model (`nn.Module`) ：要预处理的模型。

lora_config (`LoraConfig`) ：模型的Lora配置。应设置`lora_config.lora_ga_config`。train_step (`Callable[[], None]`) ：运行梯度估计的回调。通常，您应该在此回调中运行模型前向和后向传递。梯度将在此回调中的所有调用中累积。

cache_file (`Optional[str]`) ：用于保存/加载渐变的缓存文件的可选路径。如果提供并且文件存在，将从缓存加载渐变。否则，将估计梯度并将其保存到该路径。

通过估计梯度为模型构建必要的 LoRA-GA 字段。

对于每个线性层，将通过运行提供的 train_step 回调来估计梯度。这些梯度是
然后附加到模块并在初始化期间使用。

完成后，将为每个目标模块设置以下字段：
_peft_loraga_grad（`torch.Tensor`）：
权重矩阵的累积梯度。

## 入侵者降维[[peft.tuners.lora.intruders.reduce_intruder_dimension]]

#### peft.tuners.lora.intruders.reduce_intruder_dimension[[peft.tuners.lora.intruders.reduce_intruder_dimension]]

```python
peft.tuners.lora.intruders.reduce_intruder_dimension(peft_model, old_adapter_name = 'default', new_adapter_name = 'intruder_reduced', top_k = 10, threshold_epsilon = 0.5, mitigation_lambda = 0.75, logging_sink = <built-in function print>)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/intruders.py#L20)

**参数：**

peft_model ：已加载 LoRA 适配器的 PEFT 模型，其名称在 `old_adapter_name` 中提供。目前不支持混合模型。top_k（默认值：10）考虑入侵者检测的 top-k 维度。值越大，入侵者检测分析将考虑的维度越多（误报就越多）。对基本权重和适配器权重之间的余弦相似度进行操作，粗略地按维度影响排序（由奇异值分解确定），因此 10 的 top-k 将查看 10 个最“重要”的维度。 

Threshold_epsilon（默认值：0.5）将基本权重和适配器权重之间的余弦相似度视为入侵者的阈值。根据该论文，入侵者维度与预训练的奇异向量显示出接近零的绝对余弦相似度。该值越低，识别出的潜在入侵者尺寸就越少。值越高，被视为入侵者的潜在误报就越多。 

Mitigation_lambda（默认值：0.75）从适配器的增量权重中减去入侵者尺寸的相对部分。值越高，减去的入侵者维度越多，但丢失的信息也越多。请参阅论文中的图 8 进行权衡分析。logging_sink（默认：print）打印有关缓解过程的信息的函数。如果您不需要任何输出，请设置为 None。

基于 https://huggingface.co/papers/2410.21228 的入侵者维度缓解（“LoRA vs Full Fine-tuning：An
等价幻觉”）。

该方法可以通过对已经训练好的低秩进行后处理来恢复先前的知识（即减轻遗忘）
适配器。这是以任务准确性为代价的 - 调整 `migration_lambda` 值可用于在
这两个因素。

缓解完成后，将出现一个新适配器，其名称在 `new_adapter_name` 中设置，也设置为
是当前活动的适配器。因此，对缓解模型的推断将使用修改后的适配器。至
换回原来的适配器即可使用`peft_model.set_adapter(<old_adapter_name>)`。

目前仅支持 LoRA，因为尚不清楚该方法是否可以推广到其他增量权重方法。

### 格拉洛拉
https://huggingface.co/docs/peft/v0.20.0/package_reference/gralora.md