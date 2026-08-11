<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 低精度训练方法

新型硬件的发布导致了更好地利用它们的新训练范例的出现。目前，这是以培训的形式
使用 [TransformersEngine](https://github.com/NVIDIA/TransformerEngine) (TE)、[torchao](https://github.com/pytorch/ao)（本机 PyTorch FP8）或旧版 [MS-AMP](https://github.com/Azure/MS-AMP/tree/main)（不再维护，请参阅下面的警告）等软件包以 8 位精度实现。

对于今天讨论的主题的介绍，我们建议查看[low-precision usage guide](../usage_guides/low_precision_training)，因为本文档将定期引用它。 

## 快速图表

下面是 MS-AMP 文档中的快速图表，显示了训练期间每个解决方案的不同位精度：

优化级别|计算(GEMM) |通讯 |重量 |大师体重|权重梯度|优化器状态
--| --| --| --| --| --| --
FP16 放大器 | FP16 | FP32 | FP32 |不适用 | FP32 | FP32+FP32
Nvidia TE | FP8 | FP32 | FP32 |不适用 | FP32 | FP32+FP32
MS-AMP O1 | FP8 | FP8 | FP16 |不适用 | FP8 | FP32+FP32
MS-AMP O2 | FP8 | FP8 | FP16 |不适用 | FP8 | FP8+FP16
MS-AMP O3 | FP8 | FP8 | FP8 | FP16 | FP8 | FP8+FP16

## `TransformersEngine``TransformersEngine`是尝试训练8位浮点的第一个解决方案。它的工作原理是对模型中的某些层使用直接替换层，利用 FP8 引擎来减少位数（例如 32 至 8），而不会降低模型的最终精度。 

具体来说，Accelerate 将查找以下层并将其替换为 `TransformersEngine` 版本：

* `nn.LayerNorm` 为 `te.LayerNorm`
* `nn.Linear` 为 `te.Linear`

因此，我们最终得到的模型的大部分层都在 BF16 中，而某些层在 FP8 中，从而减少了一些内存。 

有趣的是，我们注意到使用 `TransformerEngine` 时性能提升并没有真正开始显现，直到绝大多数层
模型中是由这两层来替换的。因此，当参数数量约为数十亿或更多时，只有较大的模型才会显示出性能改进。 

`TransformerEngine` 可以接收许多不同的参数，这些参数可自定义其执行 FP8 计算的方式及其用途。下面提供了完整的参数列表：* `margin`：用于渐变缩放的边距。
* `interval`：重新计算缩放因子的频率间隔。
* `fp8_format``: The format to use for the FP8 recipe. Must be one of `HYBRID` or `E4M3`. (Generally `HYBRID` for training, `E4M3`用于评估）
* `amax_history_len`：用于缩放因子计算的历史长度
* `amax_compute_algo`：用于计算缩放因子的算法。必须是 `max` 或 `most_recent` 之一。
* `override_linear_precision`：是否以更高精度执行`fprop`、`dgrad`、`wgrad` GEMMS。

您可以将其中的每一个自定义为 [utils.FP8RecipeKwargs](/docs/accelerate/v1.14.0/en/package_reference/fp8#accelerate.utils.FP8RecipeKwargs) 的一部分，以帮助优化模型的性能。

如果我们注意到前面提到的图表，TE 只是将计算层转换为 FP8，而其他所有内容都在 FP32 中。因此，这最终会利用最多的内存，但这样做的好处是保证训练期间最终精度的损失最小。 

## `MS-AMP`

**⚠️ 已弃用/不再维护：** Microsoft 不再积极维护 MS-AMP。该存储库自 2023 年以来一直没有更新，并且已知与 CUDA 12.x+、现代 NCCL 版本和最新 PyTorch 版本 (2.2+) 的兼容性问题。 **我们强烈建议使用 `TransformersEngine` 或 `torchao`。** 请参阅 [usage guide](../usage_guides/low_precision_training) 了解迁移说明。MS-AMP 对`TransformersEngine`采取了不同的方法，通过提供三种不同的优化级别来转换 FP8 或 FP16 中的更多操作。

* 基础优化级别（`O1`），在 FP8 中传递权重（例如在 DDP 中）的通信，将模型的权重存储在 FP16 中，并将优化器状态保留在 FP32 中。这种优化级别的主要好处是我们可以将通信带宽减少一半。此外，由于 1/2 的所有内容都转换为 FP8，并且权重转换为 FP16，因此节省了更多 GPU 内存。值得注意的是，两个优化器状态都保留在 FP32 中。

* 第二个优化级别 (`O2`) 通过降低优化器状态的精度对此进行了改进。一个是 FP8，另一个是 FP16。一般来说，事实证明，这只会提供不降低最终精度、提高训练速度和减少内存的净增益，因为现在每个状态都处于 FP16 或 FP8 中。* 最后，MS-AMP 具有第三个优化级别 (`O3`)，这在 DeepSpeed 等 DDP 场景中很有帮助。内存中模型的权重完全转换为 FP8，主权重现在存储在 FP16 中。这会最大程度地减少内存，因为现在不仅 FP8 中几乎包含所有内容，而且 FP16 中只剩下两个状态。目前，仅支持 DeepSpeed 0.9.2 及以上版本，因此此功能不包含在 Accelerate 集成中

## 将两者结合起来

由于 MS-AMP 不再维护，因此不建议在新项目中使用此组合。

需要进行更多实验，但值得注意的是，通过依赖 NVIDIA 优化的 FP8 运算符并利用 MS-AMP 减少内存开销的方式，结合 MS-AMP 和 TransformersEngine 可以实现最高吞吐量。

### 检查点
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/checkpoint.md