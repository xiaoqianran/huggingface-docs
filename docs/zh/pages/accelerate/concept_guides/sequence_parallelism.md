<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 🤗`accelerate` 中的序列并行

本指南将介绍在 🤗`accelerate` 中使用序列并行性的基础知识。

另请参阅非常相关的[Context Parallelism](./context_parallelism)。

## 为什么要进行序列并行？

随着大型语言模型和最近推理模型的出现，序列长度迅速增长。这与注意力的二次记忆复杂性相结合，导致需要更有效的方法来训练长序列模型。
序列长度为 128k，对于 `bf16` 精度，注意矩阵的内存需求为 `128k * 128k * 2 bytes * num_heads = ~32 GB * num_heads`，考虑到普通的注意实现。诚然，使用 `flash attention` 或 `SDPA` 不会实现这些注意力权重，这会急剧下降，但内存需求的增长仍然相当可观。Ulysses 序列并行性允许我们沿着序列维度对注意力计算的输入进行分片并正常计算注意力，但在每个 GPU 上仅使用一部分注意力头。有了这个，我们可以使用更多工具训练具有长序列的模型，扩展到 15M+ 序列长度。要了解如何使用 TiledMLP、Liger-Kernel、激活检查点卸载到 cpu 以及其他一些技巧来增强 Ulysses SP，请参阅论文：[Arctic Long Sequence Training: Scalable And Efficient Training For Multi-Million Token Sequences](https://arxiv.org/abs/2506.13996)。

## Ulysses SP 与 FSDP CP 有何不同

在文档[Context Parallelism](./context_parallelism)中，您可以了解如何部署另一种称为上下文并行性的技术，该技术也在序列维度上进行切片，但使用环形注意力而不是在头部维度上进行切片。

以下文章非常详细地解释了这两种技术之间的差异：
- https://insujang.github.io/2024-01-11/tensor-parallelism-and-sequence-parallelism-detailed-analysis/
- https://huggingface.co/blog/exploding-gradients/ulysses-ring-attention改编自其中一篇文章的快速摘要：
- Ulysses SP 的通信开销相对较低，但受到 Attention Head 数量的限制，因此对网络拓扑有一定的要求（对于单个副本来说，Attention Head 的数量必须能被参与 GPU 的数量整除）。全方位通信对延迟很敏感，并且需要 Deepspeed。
- FSDP CP Ring-Attention的P2P环通信没有前述的可分性要求，但通信量较高。

最后，应该可以按照论文[USP: A Unified Sequence Parallelism Approach for Long Context Generative AI](https://arxiv.org/abs/2405.07719)中的说明结合 SP + CP 以支持更长的序列长度，尽管这尚未集成到🤗`accelerate`中。

## 支持的序列并行后端

目前唯一的序列并行后端是`deepspeed`，它来自现代化的Ulysses SP，它是[Arctic Long Sequence Training technology](https://arxiv.org/abs/2506.13996)的一部分。如果您想将其直接集成到您自己的代码中，还有一个[tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)。

## 如何使用序列并行性？

```diff
from accelerate.utils import ParallelismConfig, DeepSpeedSequenceParallelConfig

+# Example: 4 GPUs with sp_size=4, dp_shard_size=1
+# Ensure: dp_replicate_size × dp_shard_size × sp_size = 1 × 1 × 4 = 4 GPUs
parallelism_config = ParallelismConfig(
+     sp_backend="deepspeed",
+     sp_size=4,
+     dp_shard_size=1,  # Explicit: no data parallelism
+     sp_handler=DeepSpeedSequenceParallelConfig(
+         sp_seq_length_is_variable: true,
+         sp_attn_implementation="sdpa",
+     ),
+ )

accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```

与 🤗`accelerate` 中的任何其他功能一样，您也可以通过将相应的标志传递给 `accelerate launch` 来启用序列并行性。在这种情况下，没有什么不同：

```bash
accelerate launch --parallelism-config-sp-size 8  ...
```> [!提示]
> 您还可以在 `accelerate config` 命令中设置 `sp_size` 和其他配置，这会将它们保存在您的 `accelerate` 配置文件中，因此您不必每次启动脚本时都传递它们。

> [!提示]
> 序列并行性与数据并行性相结合。它不需要额外的 GPU。
> 因此，如果您有 8 个 GPU，您可以执行：`--parallelism-config-dp-shard-size 8 --parallelism-config-sp-size 8`。或者您可以使用 `ParallelismConfig` 类以编程方式设置它们。
>
> **重要**：您必须确保`dp_replicate_size × dp_shard_size × sp_size = num_processes`。例如，如果有 8 个 GPU 和 `sp_size=8`，则需要 `dp_shard_size=1`（因为 1 × 1 × 8 = 8）。使用 4 个 GPU 和 `sp_size=2`，您可以使用 `dp_shard_size=2`（因为 1 × 2 × 2 = 4）实现 2D 并行性。

## ALST/Ulysses SP 后端配置

ALST/UlyssesSP 使用注意力头并行性实现序列并行性，如 [this paper](https://arxiv.org/abs/2506.13996) 中所述。为简单起见，我们重用了序列并行性的概念和设置，从用户的角度来看，这是相同的：使用多个 GPU 来处理单个批次。为了让大家了解 ALST 的潜力 - 它允许我们在单个 H100 GPU 上使用 500K 令牌进行 bf16 训练，在单个节点上使用 3.7M 令牌进行训练，仅使用四个节点在 Llama-8B 上进行 15M 令牌训练。 HF Accelerate 的这一功能仅启用 3 个 ALST 组件中的 1 个，因此可实现的序列长度会更小。您需要 TiledMLP、激活检查点卸载到 CPU 以及启用其他一些功能以获得 ALST 的全部功能。详情请参阅[this tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)。

配置`deepspeed`后端：

```python
# Example: 4 GPUs with sp_size=4, dp_shard_size=1
# Ensure: dp_replicate_size × dp_shard_size × sp_size = 1 × 1 × 4 = 4 GPUs
parallelism_config = ParallelismConfig(
    sp_backend="deepspeed",
    sp_size=4,
    dp_shard_size=1,  # Explicit: no data parallelism
    sp_handler=DeepSpeedSequenceParallelConfig(
        sp_seq_length=256,
        sp_seq_length_is_variable=True,
        sp_attn_implementation="sdpa",
    ),
)
accelerator = Accelerator(
    ...,
    parallelism_config=parallelism_config,
)
```- `sp_backend`：此处设置为`deepspeed`
- `sp_size` 是序列并行度 - 在上面的示例中为 4，因此将使用 4 个 GPU 来处理单个批次（同时在同一 GPU 上执行 DP=4）
- `sp_seq_length`和`sp_seq_length_is_variable`用于处理序列长度。如果`sp_seq_length_is_variable=True`后端将使用可能在批次之间变化的序列长度，在这种情况下`sp_seq_length`值可以设置为可被序列并行度整除的任何值或根本不设置。在这种情况下，在每个`forward`上，序列变量将从输入中导出。如果`False`，则`seq_length`需要匹配批次的序列长度维度，然后必须对其进行填充以使其始终相同。默认为`True`。
- `sp_attn_implementation` 是 `sdpa`、`flash_attention_2` 或 `flash_attention_3` 之一。此序列并行实现使用`position_ids`而不是`attention_mask`，因此，`eager`不能在这里工作，直到它支持与`position_ids`一起使用。另请注意，`sdpa` 无法正确处理多个样本合并为一个的情况；它将把整个样本作为一个整体来处理。如果未合并样本，`sdpa` 将正常工作。因此，Flash Attention 应该是理想的选择，因为它始终有效。您也可以使用环境变量来完成相同的操作，而不是在 `DeepSpeedSequenceParallelConfig` 对象中设置这些值 - 这里它们对应于上面列表的末尾。
- `PARALLELISM_CONFIG_SP_BACKEND`
- `PARALLELISM_CONFIG_SP_SEQ_LENGTH`
- `PARALLELISM_CONFIG_SP_SEQ_LENGTH_IS_VARIABLE`
- `PARALLELISM_CONFIG_SP_ATTN_IMPLEMENTATION`

如果未在代码中传递，则可以通过 `--parallelism_config_sp_size` CLI 参数设置`sp_size`。其他参数也一样。您还可以进行加速配置文件样式配置，例如，对于 2 个 GPU：

```yaml
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: path/to/ds_config.json
machine_rank: 0
num_machines: 1
num_processes: 2
parallelism_config:
  parallelism_config_dp_replicate_size: 1
  parallelism_config_dp_shard_size: 1  # Must satisfy: 1 × 1 × 2 = 2 num_processes
  parallelism_config_sp_size: 2
  parallelism_config_sp_backend: deepspeed
  parallelism_config_sp_seq_length_is_variable: true
  parallelism_config_sp_attn_implementation: sdpa

```

如前所述，Ulysses 序列并行性通常与数据并行性重叠 - 相同的等级用于提供唯一的数据流并执行 Ulysses 序列并行性。但您也可以像这样创建副本：

```python
# Example: 4 GPUs with 2D parallelism (SP=2, DP=2)
# Ensure: dp_replicate_size × dp_shard_size × sp_size = 2 × 1 × 2 = 4 GPUs
parallelism_config = ParallelismConfig(
    dp_replicate_size=2,
    dp_shard_size=1,  # Explicit: no sharding within replicas
    sp_size=2,
    sp_backend="deepspeed",
    sp_handler=DeepSpeedSequenceParallelConfig(...),
)
```
这里我们使用 4 个 GPU，带有 2 个序列并行副本。 Deepspeed-ZeRO 是驱动数据并行性的因素。

请注意，[UlyssesSPDataLoaderAdapter](https://github.com/deepspeedai/DeepSpeed/blob/64c0052fa08438b4ecf4cae30af15091a92d2108/deepspeed/runtime/sequence_parallel/ulysses_sp.py#L442)内部隐藏着很多魔法。它在幕后使用，包装您的原始 DataLoader 对象，但如果您遇到任何问题，您应该意识到它。在批次在参与的排名之间进行分片之前，它还会自动将正确的`shift_labels`注入批次字典中。现在，开始使用 ALST/UlyssesSP 的唯一剩余部分是使用可微分 `all_gather` 来汇总各个等级的损失，以获得正确的梯度。下面的代码可以做到这一点，同时还排除任何用 `-100` 标记屏蔽的内容，以获得正确的平均值：

```python
sp_size = parallelism_config.sp_size if parallelism_config is not None else 1
if sp_size > 1:
    sp_group = accelerator.torch_device_mesh["sp"].get_group()
    sp_world_size = parallelism_config.sp_size

# Normal training loop
for iter, batch in enumerate(dl):
    optimizer.zero_grad()

    batch = move_to_device(batch, model.device)

    # The model automatically receives shift_labels via **kwargs and uses it for loss computation.
    # Both standard transformers models and Liger-patched models handle this correctly.
    outputs = model(**batch)
    loss = outputs.loss
    shift_labels = batch["shift_labels"]

    if sp_size > 1:
        # differentiable weighted per-shard-loss aggregation across ranks
        losses_per_rank = torch.distributed.nn.functional.all_gather(loss, group=sp_group)
        # special dealing with SFT that has prompt tokens that aren't used in loss computation
        good_tokens = (shift_labels != -100).view(-1).sum()
        good_tokens_per_rank = torch.distributed.nn.functional.all_gather(
            good_tokens, group=sp_group
        )
        # Skip ranks with zero valid tokens to avoid NaN contamination (NaN * 0 = NaN)
        total_loss = sum(
            losses_per_rank[rank] * good_tokens_per_rank[rank]
            for rank in range(sp_world_size)
            if good_tokens_per_rank[rank] > 0
        )
        total_good_tokens = sum(good_tokens_per_rank)
        loss = total_loss / max(total_good_tokens, 1)

    if rank == 0: accelerator.print(f"{iter}: {loss=}")
    accelerator.log(dict(train_loss=loss, step=iter))

    accelerator.backward(loss)
    optimizer.step()
```

请注意，当批次中存在 `shift_labels` 时，模型会自动处理它。模型的前向传递通过 `**kwargs` 接收 `shift_labels` 并将其传递给损失函数，该函数正确计算序列并行性的损失。如果您使用[Liger Kernel](https://github.com/linkedin/Liger-Kernel)，它还可以无缝处理`shift_labels`，并以非常节省内存的方式计算损失。对于长序列长度，强烈建议使用 Liger，因为它通过使用融合操作（例如，永远不会在内存中实现完整 Logits 张量的融合 Logit 损失计算）来释放 GPU 内存。

如果您想了解 HF Accelerate 在幕后做了什么，请阅读[this full integration tutorial](https://www.deepspeed.ai/tutorials/ulysses-alst-sequence-parallelism/)。

有关启用 ALST/UlyssesSP 的加速训练循环的示例，请参阅[examples/alst_ulysses_sequence_parallelism](https://github.com/huggingface/accelerate/blob/main/examples/alst_ulysses_sequence_parallelism)。

[！警告]
> 这个 API 相当新，仍处于实验阶段。虽然我们努力提供稳定的 API，但公共 API 的一些小部分将来可能会发生变化。由于这是 Deepspeed 后端，因此应用通常的 Deepspeed 配置，因此您可以将序列并行性与优化器状态和/或权重卸载相结合，以释放更多 GPU 内存并实现更长的序列长度。该技术已经过测试，可与 DeepSpeed ZeRO 第 2 阶段和第 3 阶段配合使用。

### 🤗`accelerate` 中的上下文并行
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/context_parallelism.md