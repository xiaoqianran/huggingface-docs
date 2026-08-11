<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 编译

## 概述

Pytorch 2.0 引入了`torch.compile`，这是一项强大的功能，通过将 PyTorch 代码 JIT 编译到优化的内核中，使 PyTorch 代码运行得更快。 `torch.compile`的主要特点包括：

- **性能改进**：通过优化计算图，显着加快模型执行速度。
- **易于使用**：只需最少的代码更改即可实现，使其易于访问。
- **兼容性**：与现有 PyTorch 代码和模型无缝协作。

与 Accelerate 一起使用时，`torch.compile` 可以顺利集成到分布式训练工作流程中，让您同时受益于分布式执行和编译优化。

编译代码的第一次执行通常需要更长的时间，因为它包括编译时间，但后续运行速度明显更快。为了在不同场景下获得最佳性能，`torch.compile`提供了各种模式，例如`"default"`、`"reduce-overhead"`（使用 CUDA 图来进一步减少开销）和 `"max-autotune"`（执行广泛的自动调整以找到最适合您的模型的内核）。

## 将 `torch.compile` 与 Accelerate 结合使用

Accelerate 提供 `TorchDynamoPlugin`，以便将 `torch.compile` 轻松无缝地集成到您的训练脚本中。

```python
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin

# Configure the compilation backend
dynamo_plugin = TorchDynamoPlugin(
    backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
    mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
    fullgraph=True,
    dynamic=False
)

# Initialize accelerator with the plugin
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
# This will apply torch.compile to your model
model = accelerator.prepare(model)
```它兼容 Accelerate 的所有其他功能和插件，包括混合精度、分布式训练（DDP、FSDP、Deepspeed）等。

## 地区编译

而不是尝试编译整个模型，这通常有很大的优化问题空间。区域编译以同一类的重复块为目标，并按顺序编译它们以命中编译器的缓存。例如，在`GPT2LMHeadModel`中，重复的块/类是`GPT2Block`，并且可以作为`model.transformer.h[0]`进行访问。模型的其余部分（例如 model.lm_head）是单独编译的。

这使我们能够加快 LLM 和 Transformers 等模型的编译开销/冷启动速度。
请参阅了解更多详情。

### 如何使用区域编译

可以通过在 `TorchDynamoPlugin` 配置中设置 `use_regional_compilation=True` 来启用它：

```python
# Configure the compilation backend
dynamo_plugin = TorchDynamoPlugin(
    use_regional_compilation=True,
    ... # other parameters
)
# Initialize accelerator with the plugin
accelerator = Accelerator(dynamo_plugin=dynamo_plugin)
# This will apply compile_regions to your model
model = accelerator.prepare(model)
```

您还可以像使用 `torch.compile` 一样直接使用 `accelerate.utils.compile_regions` 实用程序。

### 区域编译的好处

我们使用 PyTorch 中的 `torch.compile` 功能进行了广泛的基准测试，比较完整编译和区域编译。完整结果可在 [accelerate repository](https://github.com/huggingface/accelerate/tree/main/benchmarks/torch.compile/regional_compilation) 中找到。我们的基准测试的主要发现是：1. **可比较的性能**：区域编译可提供与完整编译类似的性能加速，特别是对于较大的模型。
2. **更快的编译**：区域编译显着减少了编译模型的时间，使其成为更高效的部署选择。
3. **批大小影响**：编译策略之间的性能差异随着批大小的增大而减小，表明在这些情况下编译开销的影响较小。
4. **模型大小考虑**：区域编译的好处在较大的模型中更为明显，可以节省大量的编译时间。
5. **实际应用**：对于实际应用，区域编译是优化训练冷启动时间的实用选择，特别是在处理大型模型时。

## 结论

完整编译和区域编译都可以显着加快模型速度。区域编译在编译时间和运行时性能之间提供了实际的平衡，特别是对于训练具有大量批量大小的大型模型。

### 深速
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/deepspeed.md