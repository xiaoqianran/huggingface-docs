# FP8

Below are functions and classes relative to the underlying FP8 implementation

## TERecipeKwargs[[accelerate.utils.TERecipeKwargs]]

#### accelerate.utils.TERecipeKwargs[[accelerate.utils.TERecipeKwargs]]

```python
accelerate.utils.TERecipeKwargs(use_autocast_during_eval: bool | None = None, margin: int | None = None, interval: int | None = None, fp8_format: typing.Literal['HYBRID', 'E4M3', 'E5M2'] = None, amax_history_len: int | None = None, amax_compute_algo: typing.Literal['max', 'most_recent'] = None, override_linear_precision: tuple[bool, bool, bool] = None, use_mxfp8_block_scaling: bool | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L333)

**Parameters:**

use_autocast_during_eval (`bool`, *optional*, default to `False`) : Whether to use FP8 autocast during eval mode. Generally better metrics are found when this is `False`.

margin (`int`, *optional*, default to 0) : The margin to use for the gradient scaling.

interval (`int`, *optional*, default to 1) : The interval to use for how often the scaling factor is recomputed.

fp8_format (`str`, *optional*, default to "HYBRID") : The format to use for the FP8 recipe. Must be one of `HYBRID`, `E4M3` or `E5M2`. (Generally `HYBRID` for training, `E4M3` or `E5M2` for evaluation)

amax_history_len (`int`, *optional*, default to 1024) : The length of the history to use for the scaling factor computation

amax_compute_algo (`str`, *optional*, default to "most_recent") : The algorithm to use for the scaling factor computation. Must be one of `max` or `most_recent`.

override_linear_precision (`tuple` of three `bool`, *optional*, default to `(False, False, False)`) : Whether or not to execute `fprop`, `dgrad`, and `wgrad` GEMMS in higher precision.

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize the initialization of the recipe for FP8 mixed precision
training with `transformer-engine`.

For more information on the args, please refer to the API
[documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/common.html).

```python
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs

kwargs = TERecipeKwargs(fp8_format="HYBRID")
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=[kwargs])
```

## AORecipeKwargs[[accelerate.utils.AORecipeKwargs]]

#### accelerate.utils.AORecipeKwargs[[accelerate.utils.AORecipeKwargs]]

```python
accelerate.utils.AORecipeKwargs(config: ForwardRef('Float8LinearConfig') | None = None, module_filter_func: typing.Callable | None = None, pad_inner_dim: bool | None = None, enable_fsdp_float8_all_gather: bool | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L285)

**Parameters:**

config (`torchao.float8.Float8LinearConfig`, *optional*, default to `None`) : The configuration for the FP8 training. If `None`, a default config will be created with sensible defaults for most use cases: - `pad_inner_dim=True`: Pads matrix dimensions to be divisible by 16, required for `torch._scaled_mm` operations to prevent runtime errors. - `enable_fsdp_float8_all_gather=True`: Enables FP8 all-gather for FSDP2. This provides memory bandwidth savings by casting parameters before the all-gather operation, saving 50% bandwidth compared to BF16.  You can override these defaults by providing your own `Float8LinearConfig` instance.

module_filter_func (`Callable`, *optional*, default to `None`) : Optional function that must take in a module and layer name, and returns a boolean indicating whether the module should be converted to FP8. Defaults to `accelerate.utils.ao.filter_linear_layers`. See it for an example.

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize the initialization of the recipe for FP8 mixed precision
training with `torchao` FP8.

## convert_model[[accelerate.utils.convert_model]]

#### accelerate.utils.convert_model[[accelerate.utils.convert_model]]

```python
accelerate.utils.convert_model(model, to_transformer_engine = True, _convert_linear = True, _convert_ln = True)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L26)

Recursively converts the linear and layernorm layers of a model to their `transformers_engine` counterpart.

## has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

#### accelerate.utils.has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

```python
accelerate.utils.has_transformer_engine_layers(model)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L95)

Returns whether a given model has some `transformer_engine` layer or not.

## contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

#### accelerate.utils.contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

```python
accelerate.utils.contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval = False)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L118)

Wrapper for a model's forward method to apply FP8 autocast. Is context aware, meaning that by default it will
disable FP8 autocast during eval mode, which is generally better for more accurate metrics.

## apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

#### accelerate.utils.apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

```python
accelerate.utils.apply_fp8_autowrap(model, fp8_recipe_handler)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/transformer_engine.py#L142)

Applies FP8 context manager to the model's forward method

### Accelerator
https://huggingface.co/docs/accelerate/v1.16.0/package_reference/accelerator.md
