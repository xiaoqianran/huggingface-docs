<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FSDP1 与 FSDP2

本指南解释了 `FSDP1` 和 `FSDP2` 之间的主要区别，并帮助您以最小的更改将现有代码迁移到使用 `FSDP2`。

## FSDP2 比 FSDP1 有何优势？

首先，我们要了解`FSDP1`和`FSDP2`内部是如何工作的，以了解它们之间的差异。这也有助于我们理解`FSDP1`的局限性以及`FSDP2`如何解决它们。

我们将讨论这样一个场景：我们有一个 `Layer`，其中包含 3 个 `Linear` 层，并使用 `FSDP` 进行包装，以便在 2 个 GPU 上进行分片。

  

### FSDP1
首先，我们要了解原始的`FSDP1`以及它带来的局限性。它将每个`FSDP`模块表示为单个`FlatParameter`，它是包含所有模块参数的单个一维张量，然后将其跨等级分片。 IE。如果你用 `FSDP1` 包裹 `Layer`，你会得到这样的结果：

  

您可能会注意到一个问题。整个`Layer`被扁平化为单个`FlatParameter`，然后被跨等级分片。但如果它是单个`FlatParameter`对象，我们如何存储元数据？这是限制之一。如果不进行一些丑陋的黑客攻击，就不可能正确存储每个参数的元数据，例如 `dtype`、`requires_grad` 等。### FSDP2
这就是引入`FSDP2`的原因。它不使用`FlatParameter`，而是使用`DTensor`，它是“分布式张量”的缩写。每个 `DTensor` 基本上代表了一个普通的 `torch.Tensor`，它已被跨等级分片。它包含有关原始 `torch.Tensor` 及其分片方式、[placement type](https://pytorch.org/docs/stable/distributed.tensor.html#module-torch.distributed.tensor.placement_types) 是什么的元数据等等。这就是为什么它被称为`per-parameter sharding`。下图显示了差异：

  

原始 `Layer` 的每个参数都在第 0 维度上进行分片，并在 2 个 GPU 之间分配。现在，每个`Linear`层都是一个单独的`DTensor`，并且可以简单地存储每个参数的元数据。

> [!提示] 
> 在上图中，为了使图像适合屏幕，张量在第 1 维上进行了分片，实际上，它们在第 0 维上进行了分片，如上所述

## FSDP2 提供什么？`FSDP2`是PyTorch全分片数据并行训练API的新改进版本。它的主要优点是使用`DTensor`来表示分片参数。与`FSDP1`相比，它提供：
- 更简单的内部实现，其中每个`Parameter`是一个单独的`DTensor`
- 由于上述原因，启用简单的部分参数冻结，这使得 [⟦T37⟧](https://huggingface.co/papers/2106.09685) 等方法开箱即用
- 与`DTensor`一起，`FSDP2`支持在同一模型中混合使用`fp8`和其他参数类型，开箱即用
- 使用`SHARDED_STATE_DICT`和[⟦T42⟧](https://pytorch.org/docs/stable/distributed.checkpoint.html)更快、更简单的检查点，无需跨等级进行额外的通信，这样，每个等级仅保存自己的分片和相应的元数据
- 加载时，使用分片模型的`state_dict`直接加载分片参数
- 支持异步检查点，其中参数首先复制到CPU内存，之后主线程继续训练，而另一个线程将参数存储在磁盘上
- 内存效率和确定性内存使用，`FSDP2`不再使用`recordStream`，而是使用流到流同步（更多技术细节请参阅[this forum post](https://dev-discuss.pytorch.org/t/fsdp-cudacachingallocator-an-outsider-newb-perspective/1486)和[this issue](https://github.com/pytorch/pytorch/issues/114299)）- 未来计划优化`torch.compile`的通信模式，进一步提高性能和内存效率

## API 差异

我们已经讨论了内部差异，现在让我们讨论您作为用户需要知道的差异。 

以下是通过 `accelerate` CLI 使用 `FSDP2` 时配置选项的主要变化：

上一页 (`FSDP1`) |新 (`FSDP2`) |发生了什么变化
--| --| --
`--fsdp_sharding_strategy` | `--fsdp_reshard_after_forward` |取代 `--fsdp_sharding_strategy`，更改为 `true`（之前为 `FULL_SHARD`）或 `false`（之前为 `SHARD_GRAD_OP`）
`--fsdp_backward_prefetch` | \*\***已删除**\*\* | `FSDP2` 默认使用之前的 `BACKWARD_PRE` 选项，因为只有这样才允许通信和计算重叠
`--fsdp_forward_prefetch` | \*\***尚未实施**\*\* |如何实现正在积极讨论中，目前`FSDP2`不支持
`--fsdp_sync_module_states` | \*\***已删除**\*\* |使用`FSDP2`，此参数变得多余
`--fsdp_cpu_ram_efficient_loading` | `--fsdp_cpu_ram_efficient_loading` |如果`true`、`FSDP2`同样只会在Rank 0上加载模型，然后参数会同步到其他Rank，这与`FSDP1`的行为相同，但是不再需要设置`--fsdp_sync_module_states``--fsdp_state_dict_type` | `--fsdp_state_dict_type` | `LOCAL_STATE_DICT` 已过时，`FSDP2` `SHARDED_STATE_DICT` 是默认选项，这会导致没有额外的通信，并且每个等级都会保存自己的分片，其他可能的选项是`FULL_STATE_DICT`，它会导致额外的通信和内存使用量激增，但会保存等级 0 的完整模型。
`--fsdp_use_orig_params` | \*\***已删除**\*\* | `FSDP2`在后台使用`DTensor`类，这意味着它*总是*默认使用原始参数
\*\***新**\*\* | `--fsdp_version` | `1`是默认选项，为了不破坏现有代码，设置为`2`以使用`FSDP2`

对于所有其他保持不变的选项，请参阅[⟦T84⟧ documentation](../usage_guides/fsdp)。

## 如何切换到 FSDP2

### 如果使用 Python 代码：
只需在创建插件时设置 `fsdp_version=2` 并根据上表替换选项即可。

```python
from accelerate import FullyShardedDataParallelPlugin, Accelerator

fsdp_plugin = FullyShardedDataParallelPlugin(
    fsdp_version=2
    # other options...
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
```

### 如果使用 YAML 配置：
使用我们的转换工具：
```bash
accelerate to-fsdp2 --config_file config.yaml --output_file new_config.yaml
```

这将自动将所有 FSDP1 设置转换为其 FSDP2 等效设置。使用 `--overwrite` 更新现有文件而不是创建新文件。

### 执行和推迟作业
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/deferring_execution.md