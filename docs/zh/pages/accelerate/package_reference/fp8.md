<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FP8

以下是与底层 FP8 实现相关的函数和类

## FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

####加速.utils.FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

```python
accelerate.utils.FP8RecipeKwargs(opt_level: typing.Literal['O1', 'O2'] = None, use_autocast_during_eval: typing.Optional[bool] = None, margin: typing.Optional[int] = None, interval: typing.Optional[int] = None, fp8_format: typing.Literal['HYBRID', 'E4M3', 'E5M2'] = None, amax_history_len: typing.Optional[int] = None, amax_compute_algo: typing.Literal['max', 'most_recent'] = None, override_linear_precision: tuple = None, use_mxfp8_block_scaling: typing.Optional[bool] = None, backend: typing.Literal['MSAMP', 'TE'] = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L457)

已弃用。请使用正确的 FP8 配方 kwargs 类之一，例如 `TERecipeKwargs` 或 `MSAMPRecipeKwargs`
相反。

## 转换模型[[accelerate.utils.convert_model]]

####加速.utils.convert_model[[accelerate.utils.convert_model]]

```python
accelerate.utils.convert_model(model, to_transformer_engine = True, _convert_linear = True, _convert_ln = True)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/transformer_engine.py#L26)

递归地将模型的线性层和层范数层转换为其对应的 `transformers_engine` 层。

## has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

####加速.utils.has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

```python
accelerate.utils.has_transformer_engine_layers(model)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/transformer_engine.py#L95)

返回给定模型是否具有某些 `transformer_engine` 层。

## contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

####加速.utils.contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

```python
accelerate.utils.contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval = False)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/transformer_engine.py#L118)

用于应用 FP8 自动转换的模型前向方法的包装器。是上下文感知的，这意味着默认情况下它会
在评估模式下禁用 FP8 自动转换，这通常对于更准确的指标更好。## apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

####加速.utils.apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

```python
accelerate.utils.apply_fp8_autowrap(model, fp8_recipe_handler)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/transformer_engine.py#L142)

将 FP8 上下文管理器应用于模型的前向方法

### 日志记录[[accelerate.logging.get_logger]]
https://huggingface.co/docs/accelerate/v1.15.0/package_reference/logging.md