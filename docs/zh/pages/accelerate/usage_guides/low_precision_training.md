<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 低精度训练方法

Accelerate 提供集成，通过 `TransformersEngine` 和 `torchao` 包使用指定支持的硬件来训练较低精度的方法。本文档将帮助指导您了解支持哪些硬件、如何配置 [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) 以利用低精度方法，以及训练时可以期待什么。 

## FP8 训练意味着什么

要探索使用 PyTorch 和 Accelerate 进行 FP8 训练的更多细节，请查看 [concept_guide](../concept_guides/low_precision_training) 了解为什么这可能很困难。但本质上，训练模型的某些（或全部）方面可以使用 8 位而不是 16 位来执行，而不是在 BF16 中进行训练。挑战在于这样做不会降低最终性能。 

这仅在特定 NVIDIA 硬件上启用，即：

* 3000 系列消费级显卡之后的任何产品（例如 4090）
* 基于Hopper的GPU架构（例如`H100`和`H200`）

这将导致使用的内存有所减少（因为我们已将训练某些部分所需的内存减少了一半），并且对于可以用启用 FP8 的层替换某些层的较大模型来说，吞吐量也应该会有所增加。## 配置加速器

目前支持 FP8 的两个主动维护的后端（`TransformersEngine` 和 `torchao`），每个后端都有不同的功能和配置。

要使用其中任何一个，都使用相同的核心 API。只需在 `accelerate config` 提示混合精度时将 `mixed_precision="fp8"` 传递给 [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator)，或者作为 `config.yaml` 文件的 `mixed_precision` 键中的一部分：

```{python}
from accelerate import Accelerator
accelerator = Accelerator(mixed_precision="fp8")
```

要指定后端（并自定义 FP8 混合精度设置的其他部分），您可以利用 `RecipeKwargs` 数据类之一，例如 [utils.AORecipeKwargs](/docs/accelerate/v1.16.0/en/package_reference/fp8#accelerate.utils.AORecipeKwargs) 或 [utils.TERecipeKwargs](/docs/accelerate/v1.16.0/en/package_reference/fp8#accelerate.utils.TERecipeKwargs)；您还可以在配置`yaml`/期间`accelerate launch`中澄清它。我们建议对新项目使用`TransformersEngine`或`torchao`：

```{python}
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs, AORecipeKwargs
# Use TransformersEngine
kwargs = [TERecipeKwargs()]
# Or to use torchao
# kwargs = [AORecipeKwargs()]
accelerator = Accelerator(mixed_precision="fp8", kwarg_handlers=kwargs)
```

```{yaml}
mixed_precision: fp8
fp8_config:
  amax_compute_algo: max
  amax_history_len: 1024
  backend: TE
  fp8_format: HYBRID
  interval: 1
  margin: 0
  override_linear_precision: (false, false, false)
  use_autocast_during_eval: false
```

## 配置 TransformersEngine

TransformersEngine 有许多选项可用于自定义 FP8 计算的执行方式和内容。支持的参数及其含义的完整列表可在 [NVIDIA's documentation](https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html) 中找到，但为了方便起见，它们被重新表述为 `FP8KwargsHandler` 文档字符串的一部分。 

Accelerate 尝试设置合理的默认值，但自己探索和调整各种参数可能会带来更好的性能。

要使用它，请指定 `backend="te"` 并修改您想要作为 kwarg 处理程序一部分的任何参数：

```{python}
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs
kwargs = [TERecipeKwargs(...)]
accelerator = Accelerator(mixed_precision="fp8", kwargs_handlers=kwargs)
```或者在`accelerate launch`期间经`--fp8_backend=te ...`。使用`accelerate launch --fp8_backend=te -h`查看相关参数。

同样，这可以在您的 `config.yaml` 中设置：

```{yaml}
mixed_precision: fp8
fp8_config:
    amax_compute_algo: max
    amax_history_len: 1024
    backend: TE
    fp8_format: HYBRID
    interval: 1
    margin: 0
    override_linear_precision: (false, false, false)
    use_autocast_during_eval: false
```

## 配置`torchao`

`torchao` 是一个 [PyTorch-driven](https://github.com/pytorch/ao/tree/main/torchao/float8) 可破解的 FP8 后端，旨在比前两个引擎更易于使用。与前两者相比，`ao` 的核心区别之一是，为了数值稳定性，通常最好将模型中的第一层和最后一层保持在常规精度（无论是 FP32 还是 BF16），然后将其他层量化为 FP8。因此，`ao`的配置看起来有点不同：

> 注意：此 API 是实验性的，可能会发生变化

```{python}
from accelerate import Accelerator
from accelerate.utils import AORecipeKwargs, TorchDynamoPlugin, FullyShardedDataParallelPlugin
from torchao.float8 import Float8LinearConfig

fsdp2_plugin = FullyShardedDataParallelPlugin(
  fsdp_version=2,
  cpu_ram_efficient_loading=False, # CPU RAM efficient loading CANNOT work with fp8 torchao
  fsdp_auto_wrap_policy="TRANSFORMER_BASED_WRAP",
)
dynamo_plugin = TorchDynamoPlugin(
  backend="inductor",
  use_regional_compilation=True,
)
fp8_config = Float8LinearConfig(
  enable_fsdp_float8_all_gather=True, # Use FP8 all_gather in FSDP2
  pad_inner_dim=True,
)
kwargs = [AORecipeKwargs(
  config=fp8_config
)]
accelerator = Accelerator(
  mixed_precision="fp8",
  fsdp_plugin=fsdp2_plugin,
  dynamo_plugin=dynamo_plugin,
  kwarg_handlers=kwargs,
)
```

或者在`accelerate launch`期间经`--fp8_backend=ao ...`。使用`accelerate launch --fp8_backend=ao -h`查看相关参数。

同样，可以在`config.yaml`中设置：

```{yaml}
mixed_precision: fp8
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_cpu_ram_efficient_loading: false
  fsdp_version: 2
fp8_config:
  backend: AO
  pad_inner_dim: true
  enable_fsdp_float8_all_gather: true
dynamo_config:
  dynamo_backend: INDUCTOR
  dynamo_use_regional_compilation: true
```

要了解更多具体使用的参数，请参阅官方`torchao` repo。

## 动物园示例

我们有一些示例展示了 FP8 的训练，包括加速及其底层实现，可在加速存储库中找到。
目前我们支持展示以下脚本：

* 单GPU
* 分布式数据并行（多GPU）
* 完全分片的数据并行性
* DeepSpeed ZeRO 1 至 3

了解更多[here](https://github.com/huggingface/accelerate/tree/main/benchmarks/fp8)

## 进一步阅读要了解有关 FP8 培训的更多信息，请查看以下资源：

* [Our concept guide](../concept_guides/low_precision_training) 详细介绍 TransformersEngine 和 torchao
* [The ⟦T37⟧ documentation](https://docs.nvidia.com/deeplearning/transformer-engine/api/common.html)
* [The ⟦T38⟧ documentation](https://github.com/pytorch/ao/tree/main/torchao/float8)

### 英特尔高迪
https://huggingface.co/docs/accelerate/v1.16.0/usage_guides/gaudi.md