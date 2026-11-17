<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 低精度训练方法

新型硬件的发布导致了更好地利用它们的新训练范例的出现。目前，这是以培训的形式
使用 [TransformersEngine](https://github.com/NVIDIA/TransformerEngine) (TE) 或 [torchao](https://github.com/pytorch/ao)（本机 PyTorch FP8）等软件包以 8 位精度实现。

对于今天讨论的主题的介绍，我们建议查看 [low-precision usage guide](../usage_guides/low_precision_training)，因为本文档将定期引用它。 

## 快速图表

下面是一个快速图表，显示了训练期间每个解决方案的不同位精度：

优化级别|计算(GEMM) |通讯 |重量 |大师体重|权重梯度|优化器状态
--| --| --| --| --| --| --
FP16 放大器 | FP16 | FP32 | FP32 |不适用 | FP32 | FP32+FP32
Nvidia TE | FP8 | FP32 | FP32 |不适用 | FP32 | FP32+FP32

## `TransformersEngine`

`TransformersEngine`是尝试训练8位浮点的第一个解决方案。它的工作原理是对模型中的某些层使用直接替换层，利用 FP8 引擎来减少位数（例如 32 至 8），而不会降低模型的最终精度。 

具体来说，Accelerate 将查找以下层并将其替换为 `TransformersEngine` 版本：

* `nn.LayerNorm` 为 `te.LayerNorm`
* `nn.Linear` 为 `te.Linear`因此，我们最终得到的模型的大部分层都在 BF16 中，而某些层在 FP8 中，从而减少了一些内存。 

有趣的是，我们注意到使用 `TransformerEngine` 时性能提升并没有真正开始显现，直到绝大多数层
模型中是由这两层来替换的。因此，当参数数量约为数十亿或更多时，只有较大的模型才会显示出性能改进。 

`TransformerEngine` 可以接收许多不同的参数，这些参数可自定义其执行 FP8 计算的方式及其用途。下面提供了完整的参数列表：

* `margin`：用于渐变缩放的边距。
* `interval`：重新计算缩放因子的间隔。
* `fp8_format``: The format to use for the FP8 recipe. Must be one of `HYBRID` or `E4M3`. (Generally `HYBRID` for training, `E4M3`用于评估）
* `amax_history_len`：用于缩放因子计算的历史长度
* `amax_compute_algo`：用于计算缩放因子的算法。必须是 `max` 或 `most_recent` 之一。
* `override_linear_precision`：是否以更高精度执行`fprop`、`dgrad`、`wgrad` GEMMS。

您可以将其中的每一个自定义为 [utils.TERecipeKwargs](/docs/accelerate/v1.16.0/en/package_reference/fp8#accelerate.utils.TERecipeKwargs) 的一部分，以帮助优化模型的性能。如果我们注意到前面提到的图表，TE 只是将计算层转换为 FP8，而其他所有内容都在 FP32 中。因此，这最终会利用最多的内存，但这样做的好处是保证训练期间最终精度的损失最小。

### 分析器
https://huggingface.co/docs/accelerate/v1.16.0/usage_guides/profiler.md