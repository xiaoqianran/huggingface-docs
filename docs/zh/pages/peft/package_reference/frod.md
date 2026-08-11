<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FRoD：带旋转度的全阶高效微调

FRoD 是一种参数高效的微调方法，它将共享满秩基础与稀疏可学习相结合
旋转度数。适配器更新通过固定投影张量和可训练系数来表达，其中
允许 FRoD 应用满秩更新，同时保持训练参数的数量较少。

纸张：[Full-Rank Efficient Fine-Tuning with Rotational Degrees](https://doi.org/10.1609/aaai.v40i31.39813)。

保存适配器参数时，可以通过设置来避免存储投影张量
`save_projection=False` 于 `FrodConfig`。在这种情况下，预测是从基本模型权重恢复的，
来自`projection_prng_key`的固定随机种子。这会减少检查点的大小，但默认值是
`save_projection=True` 使检查点加载独立于再生细节。

与 LoRA 相比，FRoD 可以在每个适应的线性层中表达满秩更新，同时仅训练对角线
系数和一组稀疏的非对角旋转系数。当低等级更新太频繁时，这可能很有用
限制性的。权衡是 FRoD 在适配器期间根据基本权重计算固定投影张量
注入，这使得设置成本更高，并且实现比 LoRA 更不受支持。在大型模型上，投影初始化可能会很慢，因为 FRoD 在目标模块上运行矩阵分解
注入适配器之前的类别。默认情况下显示进度条，可以使用以下命令禁用
`FrodConfig(progressbar=False)`。

对于内存受限的训练，`runtime_offload_base_weight=True` 在活动时将目标基本权重保留在 CPU 上
FRoD 路径不需要它们。这是可选的，因为 PEFT 方法通常将所有基本参数保留在加速器上
移动模型后和向前传递后。

FRoD 目前有以下限制：

- 仅支持`nn.Linear`和`transformers.pytorch_utils.Conv1D`层。

## 快速入门

```python
from transformers import AutoModelForSequenceClassification

from peft import FrodConfig, TaskType, get_peft_model

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

peft_config = FrodConfig(
    task_type=TaskType.SEQ_CLS,
    target_modules=["query", "value"],
    modules_to_save=["classifier"],
    sparse_rate=0.02,
    frod_dropout=0.0,
    runtime_offload_base_weight=True,
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
```

## FrodConfig[[peft.FrodConfig]]

#### pft.FrodConfig[[peft.FrodConfig]]

```python
peft.FrodConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, target_modules: Optional[Union[list[str], str]] = None, projection_prng_key: int = 0, save_projection: bool = True, frod_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: Optional[list[str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, sparse_rate: float = 0.01, regularization_alpha: float = 0.001, progressbar: bool = True, runtime_offload_base_weight: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/frod/config.py#L24)

**参数：**

target_modules (`Union[List[str], str]`) ：要应用 FRoD 的模块的名称。仅支持线性层。

projection_prng_key (`int`) ：初始化稀疏 FRoD COO 模式时使用的随机种子。

save_projection (`bool`) : 是否将 FRoD 投影张量保存在状态字典中。这会增加检查点大小，但会使适配器重新加载独立于本地缓存重新生成。默认为`True`。

frod_dropout (`float`) ：FRoD 层的 dropout 概率。fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

bias (`str`) ：FRoD 的偏置类型。可以是“无”、“全部”或“frod_only”。如果是“all”或“frod_only”，则相应的偏差将在训练期间更新。请注意，这意味着即使禁用适配器，模型也不会产生与没有适应的基本模型相同的输出。

module_to_save (`List[str]`) ：除 FRoD 层之外的模块列表，要设置为可训练并保存在最终检查点中。

init_weights (`bool`) ：是否使用默认初始化来初始化 FRoD 层的权重。除非您确切知道自己在做什么，否则请勿更改此设置。

Layers_to_transform (`Union[List[int],int]`) ：要变换的图层索引，如果指定此参数，它将在此列表中指定的图层索引上应用 FRoD 变换。如果传递单个整数，它将在该索引处的图层上应用 FRoD 变换。Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

稀疏率（`float`）：稀疏可训练旋转矩阵中非对角线条目的分数。较高的值会增加容量和可训练参数；较低的值更便宜。默认为 `0.01`。

regularization_alpha (`float`)：从基本权重构建共享基础时使用的小正值。当同一类别中的层具有相关权重时，它可以稳定矩阵逆。默认为`1e-3`。

Progressbar (`bool`) ：构建 FRoD 投影时是否显示进度条。在大型模型上，投影初始化可能会很慢，因为它对目标模块类别运行矩阵分解。默认为 `True`。

runtime_offload_base_weight (`bool`) ：当活动 FRoD 路径不需要目标基本权重时，是否将目标基本权重保留在 CPU 上。这可以减少 GPU 内存，因为 FRoD 直接重建适应的权重，但它改变了通常的 PEFT 约定，即在移动模型或向前运行后所有基本参数都保留在加速器上。默认为 `False`。这是存储[FrodModel](/docs/peft/v0.20.0/en/package_reference/frod#peft.FrodModel)配置的配置类。

论文：https://doi.org/10.1609/aaai.v40i31.39813。

## FrodModel[[peft.FrodModel]]

#### peft.FrodModel[[peft.FrodModel]]

```python
peft.FrodModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/frod/model.py#L101)

### 模型
https://huggingface.co/docs/peft/v0.20.0/package_reference/peft_model.md