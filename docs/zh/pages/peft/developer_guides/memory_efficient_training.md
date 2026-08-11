<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 内存高效训练

🤗 PEFT 使微调参数变得高效，但不能自动提高内存效率。本概述收集了减少训练记忆的技巧以及详细指南的链接。

> [!提示]
> 始终考虑选择较小的基础模型、较小的批量大小或较短的序列长度的基础知识，以降低内存使用量。

## 训练记忆概述

让我们剖析训练记忆的分布，以便我们可以推断出潜在的对策。我们将使用使用 Adam 优化器训练的大型语言模型作为示例。在进行全量微调时，我们会有以下位置占用内存：

1. 基本模型参数：内存消耗很大程度上取决于所选的`dtype`。每个参数的位数越少（float16 为 16），占用的内存就越少。 float16 格式的 1B 模型（每个参数 16 位/2 字节）将大致占用 `1e9 × 2 byte = 1.863 GiB` 的内存。
2. 所有可训练的基本模型参数×3用于梯度（1×）和Adam优化器状态（2×），因此需要5.59GB内存
3.层之间中间激活的内存，这些很难预测，但主要取决于所使用的计算数据类型和序列长度/批量大小。较小的基础模型或使用较小的计算数据类型将减少所有点，而使用较短的序列或较小的批次主要影响梯度和激活内存。采用 PEFT 方法将减少可训练参数的数量，从而显着减少梯度和优化器状态，​​从而节省大量内存。

## 选择正确的方法

并非每种 PEFT 方法的构建都是相同的，并且某些公式更容易以内存有效的方式构建。如果您的内存预算有限，那么检查 [PEFT method comparison suite](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison) 并筛选**最大**加速器内存使用量是有意义的。不同方法之间的平均加速器内存使用量可能相当相等，但并非每种方法都会随着激活和序列长度而平等扩展；有些方法比其他方法更容易出现内存峰值。

当针对语言建模头或嵌入层等大型层来微调特定标记时，请考虑[using trainable tokens](troubleshooting#using-trainable-tokens)。

## 量化量化是减少“基本模型”内存消耗的最佳方法之一，并且根据所采用的量化，还会减少激活内存。由于 PEFT 方法仅占用参数总数的一小部分，因此 PEFT 默认使用比基础模型更高的精度。这还可以产生适配器可以减轻量化方法引起的一些质量损失的效果。阅读[PEFT quantization guide](quantization)。

## 编译

我们训练的模型由矩阵乘法、求和和赋值等运算组成，其中每个运算都会产生一个新结果，并且随后需要占用内存。如果不需要这些中间结果，我们可以融合这些操作并节省内存。这只是`torch.compile`可以为您做的众多优化之一，所以请查看[PEFT torch.compile guide](torch_compile)。

## 梯度检查点

您可以通过仅保存层之间的每个第 n 个梯度并动态计算其余梯度来用计算来交换内存。查看 Transformers 的 [gradient checkpointing](https://huggingface.co/docs/transformers/grad_checkpointing) 文档以了解更多信息。> [!注意]
> 当不使用 Diffusers 或 Transformers 时，您可能需要实现自己的梯度检查点逻辑，具体取决于您使用的训练框架。

## 分块 NLL 损失

在训练语言模型（或分类任务）时，使用[⟦T3⟧](https://docs.pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)非常常见。您分配一个大小为 `batch × sequence × vocabulary` 的矩阵。对于特别长的序列或词汇表，这可能会很快变得昂贵。

使用[TRL](https://huggingface.co/docs/trl)时，您可以使用[Liger kernel integration](https://huggingface.co/docs/trl/liger_kernel_integration)或使用[Chunked NLLLoss](https://huggingface.co/docs/trl/v1.5.1/en/reducing_memory_usage#chunked-cross-entropy-for-reducing-peak-memory-usage)。后者会将序列分割成大小为 256 的块，以保持最大内存消耗恒定。

![NLL vs. Chunked NLL comparison](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/chunked_nll.png)

如果默认块大小不适合您的设置，请查看 [original TRL PR](https://github.com/huggingface/trl/pull/5575) 了解有关如何调整块大小的更多信息。

### 适配器注入
https://huggingface.co/docs/peft/v0.20.0/developer_guides/low_level_api.md