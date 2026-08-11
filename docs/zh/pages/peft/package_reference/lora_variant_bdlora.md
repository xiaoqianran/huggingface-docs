<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 用于消除张量并行 LoRA 服务中的通信开销的块对角 LoRA

块对角 LoRA (BD-LoRA) 是一种 LoRA 变体，其中一些 LoRA 因子被限制为块对角。这样可以消除通信开销，从而实现更快的服务
在多个 GPU 上运行推理时。尽管存在块对角线限制，BD-LoRA 在参数数量相似的情况下与普通 LoRA 的性能相似。

BD-LoRA 设计用于张量并行，这意味着将模型的权重分片到多个 GPU 上。一种流行的分片策略是[Megatron Sharding Strategy](https://arxiv.org/abs/1909.08053)。对于两个相互跟随的线性层 $W_1$、$W_2$（例如 Transformer MLP 模块中的上下投影），我们将以列并行方式分片第一层（这要求 LoRA B 是块对角线），以行并行方式分片第二层（这要求 LoRA A 是块对角线）。对于注意力模块，这可以通过将 Q、K 和 V 投影一起作为 $W_1$ 并将输出投影作为 $W_2$ 来实现，并相应地进行分片。这种分片允许兼容的推理引擎将每个块对角分片分布在不同的 GPU 上，需要在 GPU 之间传递部分结果。在下图中，您可以看到确切的分片策略以及它如何节省计算量。

论文：https://hf.co/papers/2510.23346

### 性能、排名和参数计数
在相同的参数数量下，BD-LoRA 实现了与 LoRA 类似的性能（见下图，或 peft 存储库根目录中的 `method_comparison` 文件夹）。然而，由于 BD-LoRA 中的所有其他因素都是块对角的，因此 BD-LoRA 适配器的参数将少于相同等级的 LoRA 适配器。只有当等级相应提高时，BD-LoRA的性能才具有竞争力。我们在此示例笔记本的末尾提供了用于排名匹配的示例代码。

### PEFT 类型
https://huggingface.co/docs/peft/v0.20.0/package_reference/peft_types.md