<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 🤗`accelerate` 中的上下文并行

本指南将介绍在 🤗`accelerate` 中使用上下文并行的基础知识，对于更好奇的读者，我们还将在后面的部分中介绍一些技术细节。

另请参阅非常相关的[Guide to Sequence Parallelism](./sequence_parallelism)。

## 为什么要上下文并行？

随着大型语言模型和最近推理模型的出现，序列长度迅速增长。这与注意力的二次记忆复杂性相结合，导致需要更有效的方法来训练长序列模型。
序列长度为 128k，对于 `bf16` 精度，注意矩阵的内存需求为 `128k * 128k * 2 bytes * num_heads = ~32 GB * num_heads`，考虑到普通的注意实现。诚然，使用 `flash attention` 或 `SDPA` 不会实现这些注意力权重，这会急剧下降，但内存需求的增长仍然相当可观。

上下文并行性允许我们沿着序列维度分割注意力计算的输入，并在多个 GPU 上并行计算注意力。有了这个，我们可以训练具有长序列的模型，有可能扩展到 1M+ 序列长度。

## 如何使用上下文并行？

```diff
from accelerate.utils import ParallelismConfig, TorchContextParallelConfig

+ cp_config = TorchContextParallelConfig(
+       cp_comm_strategy="alltoall", # no need to use cp_config at all, if you want to use the default "allgather"
+ )

+ parallelism_config = ParallelismConfig(
+     cp_size=8,
+     cp_handler=cp_config,  # or just cp_size=8, if you want to use the default "allgather"
+ )

accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```与 🤗`accelerate` 中的任何其他功能一样，您也可以通过将相应的标志传递给 `accelerate launch` 来启用上下文并行性。
在这种情况下，没有什么不同：

```bash
accelerate launch --parallelism-config-cp-size 8 --parallelism-config-cp-comm-strategy [allgather|alltoall] ...
```

> [!提示]
> 您还可以在`accelerate config`命令中设置`cp_size`和`cp_comm_strategy`，这会将它们保存在`accelerate`配置文件中，这样您就不必在每次启动脚本时都传递它们。

> [!提示]
> 上下文并行与其他并行策略兼容，例如数据并行、张量并行和 FSDP2。
> 您可以通过将并行度大小设置为所需的值来简单地组合它们，例如`--parallelism-config-dp-size 8 --parallelism-config-tp-size 2 --parallelism-config-cp-size 8`。或者您可以使用 `ParallelismConfig` 类以编程方式设置它们。

> [!警告]
> 上下文并行性与`FSDP2`紧密结合，您可以在[FSDP2 introduction](fsdp1_vs_fsdp2)中了解更多信息。意思是，上下文并行仅在您使用 `FullyShardedDataParallelPlugin` 或 `--use-fsdp` 并将版本设置为 2 时才有效。
> 计划。如果没有使用`FSDP2`，将会引发错误。> [!警告]
> 上下文并行仅适用于 [SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 并且仅适用于没有掩码或因果掩码的情况。我们无法为您正确检测到这一点，因此您有责任确保您使用不带面罩或因果面罩的`SDPA`。如果您使用任何其他注意实现，则会引发错误。

使用上述方法启用上下文并行性后，您可以将其应用到您的训练循环中。我们提供了一个围绕 [⟦T25⟧](https://docs.pytorch.org/docs/stable/distributed.tensor.html#torch.distributed.tensor.experimental.context_parallel) 的薄包装器，您可以在训练循环中使用它，它抽象了使用它的一些复杂性（稍后会详细介绍）。为了最大限度地减少您在训练循环中必须进行的更改，我们提供了一个上下文管理器，如果未启用上下文并行性，则该管理器是`noop`；如果启用了上下文并行性，则应用上下文并行性。这样，您就可以在训练循环中使用它，而无需根据并行配置更改任何代码。
您可以按如下方式使用它：

```python
for batch in dataloader:
    with accelerator.maybe_context_parallel(
        buffers=[batch["input_ids"], batch["attention_mask"]],
        buffer_seq_dims=[1, 1],
        no_restore_buffers={batch["input_ids"], batch["labels"]},
    ):
        outputs = model(**batch)
        ...
```

> [!警告]
> 这个上下文管理器必须在每个训练步骤中重新创建，如上例所示。这样做至关重要。这可能会将您的上下文大小扩展到 1M+ 序列长度。下面，我们展示了高达 256k 上下文大小的上下文并行的速度和内存使用情况。我们可以看到，当我们将上下文大小和 GPU 数量加倍时，我们可以实现一致的内存使用，从而有可能实现无限的上下文长度缩放。

  
  
  图 1：高达 256k 上下文大小的内存使用情况和上下文并行速度。

> [!提示]
> 这些示例是使用脚本创建的，您可以找到 [in the examples folder](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py)。要在 8 个 H100 GPU（128k 序列长度）上运行该示例，您可以使用以下命令：
> ```bash
> accelerate launch --use-fsdp --fsdp-activation-checkpointing=TRUE examples/fsdp2/nd_parallel.py --cp-size=8 --sequence-length=128000
> ```

## 加速的界面

上下文管理器采用一些参数，用于配置上下文并行性。- `buffers`：这是要在序列维度上分片的张量列表。这些张量通常是输入 ID、标签和注意力掩码。
- `buffer_seq_dims`：这是一个整数列表，按照`buffers`列表的顺序指定缓冲区的序列维度。如果您通过`buffers=[input_ids, shift_labels]`且两者都具有形状`[batch_size, sequence_length]`，则您将通过`buffer_seq_dims=[1, 1]`。
                     因为序列维度是张量的第二个维度。这是正确计算模型输出所必需的。
- `no_restore_buffers`：上下文并行性的实现就地修改缓冲区，将它们转换为`torch.distributed.tensor.Dtensor`。上下文管理器退出后，需要启动通信内核以将缓冲区恢复到原始状态（通常是全收集）。这需要一些时间，因此建议传递与 `buffers` 参数中相同的张量，以避免不必要的通信，除非您确定需要在上下文管理器退出后使用缓冲区。> [!警告]
> 上下文并行性与 `labels` 不兼容，`input_ids` 是 `input_ids` 的副本，🤗 Transformer 中的模型可以转变以启用因果语言建模本身。
> 想象一下这个案例：
> 标签 = [l1, l2, l3, l4, ... li]
> 如果我们应用上下文并行性，每个排名最终都会有一部分标签，例如：
> labels_rank_0 = [l1, l2], labels_rank_1 = [l3, l4], ...
> 在 Transformers 建模代码移动标签之后，最终会得到：
> labels_rank_0 = [l2, PAD], labels_rank_1 = [l3, PAD], ...
> 其中 `PAD` 是填充标记。这将导致错误的损失计算，因为标签不再与输入对齐。
> 因此，您需要在将标签传递到模型之前手动移动标签

## 可配置选项
Accelerate 仅提供一个选项来配置上下文并行性（`cp_size` 除外）

- `cp_comm_strategy`：分片使用的旋转方法。我们强烈建议将其保留为`"allgather"`，因为在大多数情况下它很可能会优于`"alltoall"`。上下文并行大小是不言自明的，它是输入要分片的等级数。
上下文并行分片轮换定义了输入分片如何跨等级轮换。我们将在下一节中更详细地介绍这两个选项。

您可以在 [ND parallel example](https://github.com/huggingface/accelerate/blob/main/examples/fsdp2/nd_parallel.py) 文件中看到一个端到端示例，其中您可以在单个 8xH100 节点上训练具有高达 128k 上下文长度的 8B 模型。使用多节点训练，您可以在多个 GPU 上将其扩展到 1M+ 序列长度。您还可以将其与其他并行策略无缝结合，以满足您的需求。

## 技术细节

> [!提示]
> 这一部分相当技术性，所以如果你不需要学习上下文并行的内部原理，你可以跳过它并开始构建 🚀

我们将在接下来的章节中广泛使用单词`shard`，所以让我们先定义它。如果我们将跨 `Dth` 维度、跨 `N` 等级的张量称为 `sharded`，我们的意思是该张量被分成 `N` 部分，其中张量的每个部分都具有形状 `[..., D//N, ...]`。

## 那么它是如何工作的呢？上下文并行性致力于跨序列维度对 `Q, K and V` 矩阵进行分片。每个等级都有其分配的`Q`分片，我们称之为`Q_i`。在整个计算过程中，该矩阵仅保持在该等级上。同样，每个等级都有自己的`K`和`V`分片，我们称它们为`K_i`和`V_i`。然后，每个排名用自己的分片`Q_i`、`K_i`和`V_i`计算注意力，我们称之为`attn_i`。在此计算过程中，启动通信内核来收集所有其他等级的`Ks`和`Vs`。使用什么通信原语取决于`context_parallel_shard_rotation`选项。
这样，每个等级都可以计算局部注意力，首先使用`Q_i`、`K_i`和`V_i`，然后使用所有其他等级的`K_j`和`V_j`。由于每个等级都包含在序列维度上分片的 `Q, K and V` 矩阵，因此生成的矩阵更小并且可以适合单个 GPU。

我们可以用下面的伪代码形式化它：
```python
comm_kernel = {"allgather": allgather, "alltoall": alltoall}[context_parallel_shard_rotation]
Qi, Ki, Vi = shard(Q, K, V, seq_dim)
attn[i] = attn(Qi, Ki, Vi)
for j in range(context_parallel_size):
    Kj, Vj = comm_kernel()
    attn[j] = attn(Qi, Kj, Vj) # [batch, num_heads, seq_len // context_parallel_size, head_dim]

final_attn = combine(attn)
```

## 全部到全部 vs 全部聚集###全员齐聚
那么all-to-all和all-gather有什么区别呢？通过all-gather，沟通变得非常简单。之后（好吧，之前，因为通常需要更长的时间）我们计算本地注意力`attn_i`，我们启动一个全收集来收集所有其他等级的所有其他`Ks`和`Vs`。当这种通信完成时，每个等级都拥有所有其他等级的所有`Ks`和`Vs`，并且可以用它们顺序计算注意力。
在理想情况下，所有聚集在 `attn_i` 计算完成的同时完成。然而，这在实践中从未发生过，所以当完整的`attn_i`与通信的一部分重叠时，就实现了理想的真实重叠，然后用`K_j`和`V_j`开始计算，我们等待全收集完成。### 全部到全部
All-to-all（有时称为`ring-rotation`）采用环形通信模式。完成`attn_i`计算后，启动全对所有，将`K_i`和`V_i`发送到相邻的队列。然后，我们重复此`context_parallel_size-1`次，以便每个等级都可以看到所有其他等级的`K`和`V`的所有分片。在理想情况下，我们从相邻等级预取分片`K_i+1`和`V_i+1`，并且此通信与当前`attn_i`的计算完全重叠。再说一次，实际上，这种完美的重叠永远不会发生。考虑到这种方法的性质，如果我们没有实现完美的重叠，那么惩罚会比全聚集要大得多。

## 如何选择正确的旋转方式？
理论上来说，all-to-all应该是更好的选择。尽管在实践中，这种情况很少发生。因此，我们默认为全聚集，因为它更有可能获得更好的性能。来自 `torchtitan` 团队的广泛 [benchmarks](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082) 也表明，all-to-all 很少能优于 all-gather。不过，我们仍然提供这两种选项，因为您可能会发现其中一种更适合您的用例。

您可以在下图中的探查器输出中直接看到此问题：图 1：在红色部分，您可以看到空闲时间，同时我们等待 all-to-all 内核完成。在第一个蓝色条中突出显示，您可以看到大约需要 250us 才能完成，每个注意力调用都会重复 N-1 次，其中 N 是上下文并行大小。

## 为什么只有 FSDP2？

我们仅支持`FSDP2`的上下文并行，因为我们创建`context_parallel_size`和`dp_shard_size`的联合网格来
充分发挥其潜力。
它的工作原理是：我们将模型分割到大小为 `cp_size*dp_shard_size` 的联合网格上，从而最大限度地节省内存。
这可以说是“免费午餐”，因为`FSDP`通信与注意力计算完全重叠，如下图所示。

  
  
  图 2：在蓝色矩形（流 23）中，您可以看到 `FSDP` 分片的预取与注意力计算（流 7）完全重叠，而在红色矩形（流 24）中，您可以看到全收集内核导致空闲时间泡沫，其中我们的计算流（7）处于空闲状态。在上图中，您还可以注意到 all-to-all 和 all-gather 之间的区别。在 all-to-all 中（图 1），我们为每个注意力调用启动通信内核 N-1 次，而在 all-gather 中（图 2），我们仅启动一次通信内核。这会产生更大的泡沫，但每次注意力调用只会发生一次，而在所有情况下，它会发生 N-1 次。

## 联合网格中的数据调度

我们确保将同一批数据发送到整个`cp`子组，以便结果是正确的。 （意味着`cp`子组中的每个等级都获得相同批次的数据。）但是，我们也向`dp_shard`组的每个等级分派不同的批次。
想象一下它是这样的：
```
# 8 GPUS, --dp_shard_size 4, --cp_size 2
# mesh = [[0, 1], [2, 3], [4, 5], [6, 7]]
# model is sharded across the whole mesh (each GPU holds 1/8 of the model)
# GPUs 0,1 = batch 0
# GPUs 2,3 = batch 1
... and so on.
```

### 比较分布式设置的性能
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/performance.md