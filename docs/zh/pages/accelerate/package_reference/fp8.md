<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FP8

以下是与底层 FP8 实现相关的函数和类

## FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

####加速.utils.FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/dataclasses.py#L457)

已弃用。请使用正确的 FP8 配方 kwargs 类之一，例如 `TERecipeKwargs` 或 `MSAMPRecipeKwargs`
相反。

## 转换模型[[accelerate.utils.convert_model]]

####加速.utils.convert_model[[accelerate.utils.convert_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/transformer_engine.py#L26)

递归地将模型的线性层和层范数层转换为其对应的 `transformers_engine` 层。

## has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

####加速.utils.has_transformer_engine_layers[[accelerate.utils.has_transformer_engine_layers]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/transformer_engine.py#L95)

返回给定模型是否具有某些 `transformer_engine` 层。

## contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

####加速.utils.contextual_fp8_autocast[[accelerate.utils.contextual_fp8_autocast]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/transformer_engine.py#L118)

用于应用 FP8 自动转换的模型前向方法的包装器。是上下文感知的，这意味着默认情况下它会
在评估模式期间禁用 FP8 自动转换，这通常对于更准确的指标更好。## apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

####加速.utils.apply_fp8_autowrap[[accelerate.utils.apply_fp8_autowrap]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/transformer_engine.py#L142)

将 FP8 上下文管理器应用于模型的前向方法

### Kwargs 处理程序
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/kwargs.md