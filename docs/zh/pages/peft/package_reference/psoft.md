<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 普软件

[PSOFT](https://hf.co/papers/2505.11235)是一种基于正交微调（OFT）的参数高效微调方法，它保留预训练权重列向量的几何关系，同时实现性能和多维效率（包括参数数量、内存使用和计算成本）之间的平衡权衡。通过将正交变换限制为从预训练权重导出的低秩主子空间，PSOFT 弥合了 LoRA 和 OFT 之间的差距，提供了理论保证和实际适应性。其有效性通过对各种基准的广泛评估得到验证，包括 GLUE、VTAB-1K、GSM8K、MATH 和常识推理基准。

- 仅支持`nn.Linear`层。
- 不支持量化层。

论文摘要是：*在模型参数快速增长的推动下，参数高效微调（PEFT）对于在计算资源有限的情况下使大型模型适应不同的下游任务变得至关重要。在这种范式中，正交微调及其变体保留了预训练模型的语义表示，但在参数计数、内存和计算方面难以实现表达性和效率。为了克服这一限制，我们提出了带有主子空间自适应的高效正交微调（PSOFT），它将正交变换限制在预训练权重的主子空间中。具体来说，PSOFT 通过矩阵分解构造该子空间以实现兼容变换，建立严格维护该子空间的几何形状以保留基本语义的理论条件，并引入有效的可调谐向量，在训练过程中逐渐放松正交性以增强适应性。对 4 个代表性模型的 35 个 NLP 和 CV 任务进行的广泛实验表明，PSOFT 提供了一种实用且可扩展的解决方案，可以同时实现语义保存，表示PEFT 的有效性和多维效率。*

## PSOFT 的工作原理

PSOFT 使用 SVD 将每个权重矩阵 $W_{pre}$ 分解为 $W_{pri}$ 和 $W_{res}$：
$W_{\text{pre}} = US V^\top$

主子空间 $W_{\text{pri}} = U_r S_r V_r^\top = AB$ 由 top-$r$ 奇异分量构造：

$W_{\text{pre}} = W_{\text{pri}} + W_{\text{res}} = AB + W_{\text{res}},$

$W_{\text{ps-tuned}} = ARB + W_{\text{res}}.$（PSOFT-SO：具有严格正交性的 PSOFT）

$W_{\text{ps-tuned}} = A \, \mathrm{diag}(\alpha) \, R \, \mathrm{diag}(\beta) \, B + W_{\text{res}}.$ （PSOFT-RO：具有松弛正交性的 PSOFT）

在训练期间，$A$、$B$ 和 $W_{\text{res}}$ 被冻结，只有 $R$（或带有 $\alpha$ 和 $\beta$ 的 $R$）可以训练。

为了与 PEFT 框架（期望附加权重更新）兼容，PSOFT 以以下附加形式实现：
$W_{\text{ps-tuned}} = W_{\text{pre}} + A (R - I_r) B$

## 可训练参数

应用 PSOFT 后：

- 原始模型权重（$A$、$B$ 和 $W_{\text{res}}$）被冻结。
- 只有正交矩阵 $R$ （以及可选的 $\alpha$、$\beta$）是可训练的。
- 没有引入额外的偏置参数。## 基本用法
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PsoftConfig, get_peft_model

# Load base model
model_id = "facebook/opt-125m"
model = AutoModelForCausalLM.from_pretrained(model_id)

# Configure PSOFT
config = PsoftConfig(
    r=32,                                   # the dimension of trainable matrix R,
    psoft_alpha=32,                         # scaling factor (typically set to r in PSOFT),
    target_modules=["q_proj", "v_proj"],    # target attention projection layers
    ab_svd_init="psoft_init",        # principal subspace initialization
    psoft_svd="full",                       # SVD method
    psoft_orth=True,                        # enable orthogonal R (Cayley parameterization)
    psoft_mag_a=True,                       # enable tunable vector alpha
    psoft_mag_b=True,                       # enable tunable vector beta
    use_cayley_neumann=False,               # disable Cayley–Neumann approximation
    num_cayley_neumann_terms=5,             # number of Neumann series terms
    cayley_neumann_eps=None,                # improve numerical stability
)

# Apply PSOFT
model = get_peft_model(model, config)
model.train()

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# Train
inputs = tokenizer("Hello world", return_tensors="pt", padding=True)
loss = model(**inputs, labels=inputs["input_ids"]).loss
loss.backward()

trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, lr=5e-4)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```
## 配置选项

### 不同模式

（PSOFT-SO：具有严格正交性的PSOFT）

```python
config = PsoftConfig(psoft_orth=True,psoft_mag_a=False,psoft_mag_b=False)
```

（PSOFT-RO：具有松弛正交性的 PSOFT）
```python
config = PsoftConfig(psoft_orth=True,psoft_mag_a=True,psoft_mag_b=True)
```

### 最佳实践
1. **排名选择**：较小的排名（例如，`32–128`）适合更简单的任务，而较大的排名（例如，`64–256`）可以为更复杂的任务提供更好的表现力，但代价是增加参数和计算量。
2. **缩放因子**：缩放因子在 PSOFT 中通常设置为 $r$。
3. **学习率**：使用标准学习率（例如`1e-4`到`5e-3`）进行稳定训练。
4. **SVD初始化**：`lowrank`选项比`full`更具内存和计算效率，使其更适合大型模型。
5. **凯莱-诺依曼近似**：当秩较大时，启用凯莱-诺依曼近似可以显着提高计算效率，而对于较小的秩，好处不太明显。在实践中，少量的诺依曼级数项（通常是`5`）通常可以在准确性和效率之间提供良好的平衡。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PSOFT"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## PsoftConfig[[peft.PsoftConfig]]

#### peft.PsoftConfig[[peft.PsoftConfig]]

```python
peft.PsoftConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, psoft_alpha: int = 32, psoft_dropout: float = 0.0, fan_in_fan_out: bool = False, ab_svd_init: Literal['psoft_init', 'pissa_init'] = 'psoft_init', psoft_svd: Literal['full', 'lowrank'] = 'full', psoft_svd_lowrank_niter: int = 10, random_seed: int = 0, psoft_orth: bool = True, psoft_mag_b: bool = True, psoft_mag_a: bool = True, use_cayley_neumann: bool = False, num_cayley_neumann_terms: int = 5, cayley_neumann_eps: Optional[float] = None, modules_to_save: Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/psoft/config.py#L26)

**参数：**r (`int`) ：默认为 32。PSOFT 等级 (r) 通过 r*r 变换 R 控制适配器容量。较小的等级 32-128 通常足以满足简单任务，更复杂的任务可能会受益于 64-256，以额外参数和计算为代价提高表现力。请参阅论文以了解经过经验验证的设置：https://openreview.net/forum?id=FSHrinMArK。

target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块（如果模型是 PreTrainedModel，则排除输出层）。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

psoft_alpha (`int`) ：默认为 32。它控制 PSOFT 缩放因子。与 LoRA alpha 语义相同。

psoft_dropout (`float`) ：默认为 0.0。 PSOFT 路径的丢失。与 LoRA dropout 的语义相同。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

ab_svd_init (`Literal["psoft_init", "pissa_init"]`) ：默认为“psoft_init”。 A 和 B 的初始化策略用于构造 PSOFT 中的主子空间。 'psoft_init'：使用行正交 A 进行基于 SVD 的初始化，确保严格正交性 (PSOFT)。 'pissa_init'：基于 SVD 的初始化，具有对称 A 和 B（标准 PiSSA）。

psoft_svd (`Literal["full", "lowrank"]`) ：默认为“完整”。用于初始化的 SVD 后端：“full”使用 torch.linalg.svd； 'lowrank' 使用 torch.svd_lowrank。psoft_svd_lowrank_niter (`int`) ：仅当 psoft_svd='lowrank' 时使用。默认为 10。当 psoft_svd='lowrank' 时 torch.svd_lowrank 使用的幂迭代次数。

psoft_orth (`bool`) ：默认为“True”。如果为 True，则通过 Cayley 参数化将 R 约束为正交，从而保留预训练权重向量列之间的几何关系。如果为 False，则 R 是没有正交性约束的自由矩阵。

psoft_mag_b (`bool`) ：默认为“True”。如果为 True，则在 R 的“输出”侧学习对角缩放向量。通常与 psoft_mag_a 配对以提高任务适应性，并对预训练的几何图形造成轻微扭曲。

psoft_mag_a (`bool`) ：默认为“True”。如果为 True，则在 R 的“输入”侧学习对角缩放向量。通常与 psoft_mag_b 配对以提高任务适应性，并对预训练的几何图形造成轻微扭曲。

use_cayley_neumann (`bool`) ：默认为“False”。是否使用 PSOFT 的 Cayley-Neumann 公式。设置为 True 可提高计算效率，但代价是正交性的近似误差更大。num_cayley_neumann_terms (`int`) ：默认为 5。仅当 use_cayley_neumann=True 时使用。要使用的凯莱-诺依曼术语的数量。数字越大，正交性的近似误差越小。

cayley_neumann_eps (`optional[float]`) ：默认为“无”。仅当 use_cayley_neumann=True 时使用。 Cayley-Neumann 近似中生成矩阵 Q 的可选 Frobenius 范数界。如果无（默认），则不应用重新缩放。如果设置为 (0, 1) 中的值（例如 0.9），则只要 ||Q||_F 超过阈值，就会重新缩放 Q，以提高数值稳定性。详情请参阅 https://spherelab.ai/oftv2/。

init_weights (`bool`) ：默认为“True”。是否使用默认初始化来初始化 PSOFT 层的权重。除非您确切知道自己在做什么，否则请勿更改此设置。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

PSOFT（具有主子空间自适应的高效正交微调）的配置。

PSOFT 在低秩矩阵 A 和 B 之间插入一个 r*r 正交变换 R，因此低秩更新为 ΔW =
B @ (R-I) @ A。仅训练 R（和可选的可调向量）； A和B用psoft_init初始化
（基于 SVD，行正交 A）并冻结。

## PsoftModel[[peft.PsoftModel]]

#### peft.PsoftModel[[peft.PsoftModel]]

```python
peft.PsoftModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/psoft/model.py#L28)

**参数：**

model ：要适应的模型。

配置：PsoftConfig。

adapter_name ：适配器名称，默认“default”。

low_cpu_mem_usage ：在元设备上创建空适配器权重。

PSOFT（具有主子空间自适应的高效正交微调）模型。

在低秩 A 和 B 之间插入 r*r 正交（或缩放）变换 R： ΔW = B @ (R-I) @ A。使用
ab_svd_init="psoft_init" 从 SVD 初始化 A/B 并冻结它们，仅训练 R（以及可选的幅度
向量）。### 热插拔适配器
https://huggingface.co/docs/peft/v0.20.0/package_reference/hotswap.md