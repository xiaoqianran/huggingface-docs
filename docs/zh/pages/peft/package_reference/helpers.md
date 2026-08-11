<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 辅助方法

PEFT 辅助函数的集合。

## 检查模型是否为 PEFT 模型[[peft.helpers.check_if_peft_model]]

#### peft.helpers.check_if_peft_model[[peft.helpers.check_if_peft_model]]

```python
peft.helpers.check_if_peft_model(model_name_or_path: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/helpers.py#L148)

**参数：**

model_name_or_path (`str`) ：要检查的模型 ID，可以是本地的或 Hugging Face Hub 上的。

**返回：** `bool`

如果模型是 PEFT 模型，则为 True，否则为 False。

检查模型是否为 PEFT 模型。

## 暂时重新调整 LoraLayer 模块中的适配器比例[[peft.helpers.rescale_adapter_scale]]

#### peft.helpers.rescale_adapter_scale[[peft.helpers.rescale_adapter_scale]]

```python
peft.helpers.rescale_adapter_scale(model, multiplier)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/helpers.py#L169)

**参数：**

model ：包含需要调整缩放比例的`LoraLayer`模块的模型。

multiplier (float 或 int) : 重新缩放 `scaling` 属性的乘数。必须是 float 或 int 类型。

**加薪：** ``ValueError``

- ``ValueError`` -- If the model does not contain any `LoraLayer`
  实例，表明该模型不支持缩放。

上下文管理器，用于临时重新调整模型中 LoRA 适配器的缩放比例。当上下文管理器退出时，原始缩放值将恢复。该上下文管理器与
直接加载 LoRA 适配器的变压器和扩散器模型。

对于 LoRA，应用此上下文管理器与 [0, 1] 中的乘数严格等同于应用
[wise-ft](https://huggingface.co/papers/2109.01903)（参见[#1940](https://github.com/huggingface/peft/issues/1940)
了解详情）。如果训练之间存在分布变化，则可以提高模型的性能
用于微调的数据，以及推理时使用的测试数据。

警告：据报道，当使用Apple的MPS后端进行PyTorch时，需要添加短暂的睡眠
退出上下文后，尺度完全恢复之前的时间。

示例：

```python
>>> model = ModelWithLoraLayer()
>>> multiplier = 0.5
>>> with rescale_adapter_scale(model, multiplier):
...     outputs = model(**inputs)  # Perform operations with the scaled model
>>> outputs = model(**inputs)  # The original scaling values are restored here
```

## 上下文管理器在 LoRA 层的 `forward` 方法中禁用输入数据类型转换[[peft.helpers.disable_input_dtype_casting]]

#### peft.helpers.disable_input_dtype_casting[[peft.helpers.disable_input_dtype_casting]]

```python
peft.helpers.disable_input_dtype_casting(model: Module, active: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/helpers.py#L230)

**参数：**

model (nn.Module) ：包含要调整其输入 dtype 转换的 PEFT 模块的模型。

active (bool) ：上下文管理器是活动的（默认）还是非活动的。

上下文管理器禁用输入数据类型转换为权重的数据类型。## 用于启用 DoRA 缓存的上下文管理器（推理时间更快，但需要更多内存）[[peft.helpers.DoraCaching]]

#### peft.helpers.DoraCaching[[peft.helpers.DoraCaching]]

```python
peft.helpers.DoraCaching(enabled: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/helpers.py#L348)

用于启用 DoRA 缓存的上下文管理器，这会以内存为代价提高 DoRA 推理的速度。

使用主动缓存，物化 LoRA 权重 (B @ A) 和权重范数（基本权重 + LoRA 权重）为
缓存。

即使在缓存上下文中，如果模型处于训练模式，缓存也会被禁用。当模型切换到
训练模式下，缓存将被清除。

示例：

```py
>>> from peft.helpers import DoraCaching

>>> model.eval()  # put in eval model for caching to work

>>> with DoraCaching():  # use as a context manager
...     output = model(inputs)

>>> dora_caching = DoraCaching()
>>> dora_caching(enabled=True)  # permanently enable caching
>>> output = model(inputs)
>>> dora_caching(enabled=False)  # permanently disable caching
>>> output = model(inputs)
```

## KappaTune 目标选择[[peft.helpers.KappaTuneSelector]]

`KappaTuneSelector`和`find_kappa_target_modules`实现了[KappaTune paper](https://arxiv.org/abs/2506.16289)的通用目标选择过程。 

该方法识别具有较高灵活性（较高输出微分熵）和较低专业化（对特定输入方向较低敏感度）的模块。

这些特性使得所选模块成为减轻任何添加可训练参数的适应方法中灾难性遗忘的良好候选者，包括 LoRA、DoRA、LoHa、AdaLoRA，甚至直接微调原始权重。#### peft.helpers.KappaTuneSelector[[peft.helpers.KappaTuneSelector]]

```python
peft.helpers.KappaTuneSelector(model: Module, max_dim_size_to_analyze: int = 16384, moe_param_suffixes: typing.Optional[tuple[str, ...]] = None, show_progress: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/helpers.py#L392)

用于计算每个模块/每个参数条件数并返回最佳 LoRA 目标的轻量级实用程序。

支持：
- 经典 nn.Linear 模块（LoraConfig 中的 target_modules）
- 现代融合 MoE 权重存储为 3D nn.Parameter（gate_up_proj / down_proj、gate_proj / up_proj 等），用于
  Llama-4、Qwen2_MoE、Qwen3_MoE、Mixtral、OLMoE 和类似模型。这些是通过 target_parameters 返回的。

注意事项：
- 条件数计算需要运行 SVD，并且在非常大的模型上可能需要几分钟的时间。一个进步
可以通过`show_progress`显示/禁用栏。

#### peft.find_kappa_target_modules[[peft.find_kappa_target_modules]]

```python
peft.find_kappa_target_modules(model: Module, top_p: float = 0.2, max_dim_size_to_analyze: int = 16384, moe_param_suffixes: typing.Optional[tuple[str, ...]] = None, show_progress: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/helpers.py#L536)

**参数：**

model (nn.Module) ：将分析其权重以获取条件数的基本模型。

top_p（float，可选）：选择条件编号最低的候选模块/参数的顶部部分。

max_dim_size_to_analyze（int，可选）：通过 SVD 分析的最大矩阵维度的上限。默认为 16384。moe_param_suffixes （可选[tuple[str, ...]]，可选）：参数名称后缀，用于识别应通过`target_parameters`返回的融合MoE张量。如果没有，则使用合理的默认值。

show_progress (bool, 可选) ：在跨候选张量/模块计算条件数（基于 SVD）时是否显示进度条。在 CI 或其他非交互式环境中禁用。默认为 True。

用于 KappaTune 目标选择的一行便捷功能。返回 target_modules 和 target_parameters。

### 维洛拉
https://huggingface.co/docs/peft/v0.20.0/package_reference/lora_variant_velora.md

### 维洛拉

> [!注意]
> 这是 LoRA 的一个变体，因此，除非本页另有说明，LoRA 的所有功能都适用于此方法。

[VeLoRA](https://huggingface.co/papers/2405.17991) 是一种 LoRA 变体，它通过压缩前向传递中为 LoRA 保存的激活，然后在后向传递中重建它们以实现更新规则来减少训练内存。在 PEFT 中，VeLoRA 通过 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig) 上的`velora_config` 参数配置为 LoRA 变体。

```py
from peft import LoraConfig, VeloraConfig

config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    velora_config=VeloraConfig(
        num_groups=64,
        scale=0.2,
        init_type="batch_average",
    ),
)
```VeLoRA 应用于`target_modules` 选择的每个 LoRA 层。 `num_groups` 控制压缩前输入激活深度的分割方式。如果激活深度不能被 `num_groups` 整除，VeLoRA 会在内部填充分组表示，并在重建后删除填充。 `scale` 在向后传递过程中重新调整重建的激活，`init_type` 选择如何初始化投影。

使用 `batch_average_once` 从第一个训练批次初始化投影，使用 `batch_average` 从每次训练前向传递更新投影，或使用 `random` 立即从随机归一化向量初始化投影。

以下是 [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) 的一些结果。

|变体 |训练损失|最大内存 (GiB) |令牌/秒 |
|---|---:|---:|---:|
|洛拉 | 0.5427 | 0.5427 27.69 | 27.69 2366.2 | 2366.2
| LoRA + GC | 0.5426 | 0.5426 13.17 | 13.17 1671.8 | 1671.8
| LoRA+VeLoRA | 0.5427 | 0.5427 19.94 | 19.94 2057.6 | 2057.6

#### 注意事项

- VeLoRA 目前仅在标准 LoRA 线性层上受支持。

### X-LoRA
https://huggingface.co/docs/peft/v0.20.0/package_reference/xlora.md