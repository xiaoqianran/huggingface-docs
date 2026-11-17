<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FP8

以下是与底层 FP8 实现相关的函数和类

## TERecipeKwargs[[accelerate.utils.TERecipeKwargs]]

####加速.utils.TERecipeKwargs[[accelerate.utils.TERecipeKwargs]]

```python
accelerate.utils.TERecipeKwargs(use_autocast_during_eval: bool | None = None, margin: int | None = None, interval: int | None = None, fp8_format: typing.Literal['HYBRID', 'E4M3', 'E5M2'] = None, amax_history_len: int | None = None, amax_compute_algo: typing.Literal['max', 'most_recent'] = None, override_linear_precision: tuple[bool, bool, bool] = None, use_mxfp8_block_scaling: bool | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L333)

**参数：**

use_autocast_during_eval (`bool`, *可选*, 默认为 `False`) : 是否在 eval 模式下使用 FP8 自动转换。通常，当这是 `False` 时，会找到更好的指标。

margin (`int`，*可选*，默认为 0) ：用于渐变缩放的边距。

间隔（`int`，*可选*，默认为 1）：重新计算缩放因子的频率的间隔。

fp8_format（`str`，*可选*，默认为“HYBRID”）：用于 FP8 配方的格式。必须是 `HYBRID`、`E4M3` 或 `E5M2` 之一。 （一般训练用`HYBRID`，评估用`E4M3`或`E5M2`）

amax_history_len (`int`，*可选*，默认为 1024) ：用于缩放因子计算的历史记录的长度

amax_compute_algo（`str`，*可选*，默认为“most_recent”）：用于缩放因子计算的算法。必须是 `max` 或 `most_recent` 之一。override_线性_精度（三个`bool`中的`tuple`，*可选*，默认为`(False, False, False)`）：是否以更高精度执行`fprop`，`dgrad`和`wgrad`GEMMS。

在 [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义 FP8 混合精度配方的初始化
使用`transformer-engine`进行训练。

有关args的更多信息，请参阅API
[documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html)。

```python
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs

kwargs = TERecipeKwargs(fp8_format="HYBRID")
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
```

## AORecipeKwargs[[accelerate.utils.AORecipeKwargs]]

####加速.utils.AORecipeKwargs[[accelerate.utils.AORecipeKwargs]]

```python
accelerate.utils.AORecipeKwargs(config: ForwardRef('Float8LinearConfig') | None = None, module_filter_func: typing.Callable | None = None, pad_inner_dim: bool | None = None, enable_fsdp_float8_all_gather: bool | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L285)

**参数：**

config (`torchao.float8.Float8LinearConfig`，*可选*，默认为`None`)：FP8 训练的配置。如果`None`，将为大多数用例创建一个具有合理默认值的默认配置： - `pad_inner_dim=True`：将矩阵维度填充为可被 16 整除，这是 `torch._scaled_mm` 操作所必需的，以防止运行时错误。 - `enable_fsdp_float8_all_gather=True`：为 FSDP2 启用 FP8 全收集。这通过在全收集操作之前转换参数来节省内存带宽，与 BF16 相比节省了 50% 的带宽。  您可以通过提供自己的 `Float8LinearConfig` 实例来覆盖这些默认值。module_filter_func (`Callable`，*可选*，默认为`None`)：可选函数，必须接受模块和层名称，并返回一个布尔值，指示模块是否应转换为 FP8。默认为`accelerate.utils.ao.filter_linear_layers`。请参阅示例。

在您的 [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义 FP8 混合精度配方的初始化
使用 `torchao` FP8 进行训练。

## 转换模型[[accelerate.utils.convert_model]]

####加速.utils.convert_model[[accelerate.utils.convert_model]]

```python
accelerate.utils.convert_model(model, to_transformer_engine = True, _convert_linear = True, _convert_ln = True)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L26)

递归地将模型的线性层和层范数层转换为其对应的 `transformers_engine` 层。

## has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

####加速.utils.has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

```python
accelerate.utils.has_transformer_engine_layers(model)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L95)

返回给定模型是否具有某些 `transformer_engine` 层。

## contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

####加速.utils.contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

```python
accelerate.utils.contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval = False)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L118)用于应用 FP8 自动转换的模型前向方法的包装器。是上下文感知的，这意味着默认情况下它会
在评估模式下禁用 FP8 自动转换，这通常对于更准确的指标更好。

## apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

####加速.utils.apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

```python
accelerate.utils.apply_fp8_autowrap(model, fp8_recipe_handler)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L142)

将 FP8 上下文管理器应用于模型的前向方法

### 加速器
https://huggingface.co/docs/accelerate/v1.16.0/package_reference/accelerator.md