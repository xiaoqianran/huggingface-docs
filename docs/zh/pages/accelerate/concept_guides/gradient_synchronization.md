<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 梯度同步

PyTorch 的分布式模块通过在系统中的所有 GPU 之间来回通信来运行。
这种通信需要时间，并确保所有进程都知道彼此的状态发生在特定的触发点
使用`ddp`模块时。 

这些触发点被添加到 PyTorch 模型中，特别是它们的 `forward()` 和 `backward()` 方法。 
当模型用 `DistributedDataParallel` 包装时会发生这种情况：
```python
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

model = nn.Linear(10, 10)
ddp_model = DistributedDataParallel(model)
```
在 Accelerate 中，当调用 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 并传入模型时，此转换会自动发生。

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()
  import torch.nn as nn
- from torch.nn.parallel import DistributedDataParallel

  model = nn.Linear(10,10)
+ model = accelerator.prepare(model)
```

## 梯度积累减慢

您现在了解了，当以下情况发生时，PyTorch 会向 PyTorch 模型的 `forward` 和 `backward` 方法添加钩子： 
在分布式设置中进行训练。但是这种风险会降低你的代码速度吗？

在 DDP（分布式数据并行）中，期望执行和运行进程的特定顺序
在特定的点上，并且这些也必须在继续之前大致在同一时间发生。最直接的例子是当你通过更新模型参数时
`optimizer.step()`。
如果没有梯度累积，模型的所有实例都需要更新
在继续下一个之前，对它们的梯度进行计算、整理和更新
一批数据。
当执行梯度累积时，你累积`n`损失梯度并且
跳过`optimizer.step()`，直到达到`n`批次。正如所有训练
进程只需要在调用`optimizer.step()`时进行同步，
无需对您的训练步骤进行任何修改，这种不必要的进程间
通信可能会导致速度显着减慢。

 如何避免这种开销？

## 解决速度变慢的问题

由于在训练这些批次时跳过模型参数更新，因此在实际调用 `optimizer.step()` 之前不需要同步它们的梯度。 
PyTorch 无法自动告诉您何时需要执行此操作，但它们确实提供了一个工具来通过 [⟦T17⟧](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel.no_sync) 上下文管理器提供帮助
将其转换为 DDP 后添加到您的模型中。在此上下文管理器下，PyTorch 将在以下情况下跳过同步渐变：
`.backward()`被调用，并且在此之外第一次调用`.backward()` 
上下文管理器将触发同步。请参阅下面的示例：
```python
ddp_model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

for index, batch in enumerate(dataloader):
    inputs, targets = batch
    # Trigger gradient synchronization on the last batch
    if index != (len(dataloader) - 1):
        with ddp_model.no_sync():
            # Gradients only accumulate
            outputs = ddp_model(inputs)
            loss = loss_func(outputs)
            accelerator.backward(loss)
    else:
        # Gradients finally sync
        outputs = ddp_model(inputs)
        loss = loss_func(outputs)
        accelerator.backward(loss)
        optimizer.step()
```

在 Accelerate 中，使其成为一个无论训练设备如何都可以调用的 API（尽管如果您不在分布式系统中，它可能不会执行任何操作！），
`ddp_model.no_sync` 被 [no_sync()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.no_sync) 取代，操作方式相同：

```diff
  ddp_model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

  for index, batch in enumerate(dataloader):
      inputs, targets = batch
      # Trigger gradient synchronization on the last batch
      if index != (len(dataloader)-1):
-         with ddp_model.no_sync():
+         with accelerator.no_sync(model):
              # Gradients only accumulate
              outputs = ddp_model(inputs)
              loss = loss_func(outputs, targets)
              accelerator.backward(loss)
      else:
          # Gradients finally sync
          outputs = ddp_model(inputs)
          loss = loss_func(outputs)
          accelerator.backward(loss)
          optimizer.step()
          optimizer.zero_grad()
```

正如您所期望的，[accumulate()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.accumulate) 函数通过跟踪当前批次号来围绕此条件检查，为您留下最终的
梯度累积API：

```python
ddp_model, dataloader, optimizer = accelerator.prepare(model, dataloader, optimizer)

for batch in dataloader:
    with accelerator.accumulate(model):
        optimizer.zero_grad()
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

因此，在选择 API 时，您应该使用 *`accelerator.accumulate` 或 `accelerator.no_sync`*。 

## 速度到底有多慢，以及您容易犯的错误

要设置一个实际示例，请考虑以下设置：

* 2 个单 GPU T4 节点和 1 个带 2 个 GPU 的节点
* 每个 GPU 都是 T4，托管在 GCP 上
* 使用的脚本是[NLP Example](https://github.com/muellerzr/timing_experiments/blob/main/baseline.py)脚本的修改
* 每个GPU的batch size为16，梯度每4步累加一次

所有脚本都可以在 [this repository](https://github.com/muellerzr/timing_experiments) 中找到。如果不小心梯度同步和 GPU 通信，可能会浪费“大量”时间 
当这些 GPU 在不必要的时间段内相互通信时。

多少？

参考：
- 基线：不使用此处讨论的同步实践
- `no_sync` 不正确：`no_sync` 仅围绕 `backward` 调用，而不是 `forward`
- `no_sync`：正确使用`no_sync`模式
- `accumulate`：正确使用[accumulate()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.accumulate)

以下是单节点和双节点设置上每个设置迭代 29 批数据的每批平均秒数：

|             |基线| `no_sync` 不当 | `no_sync` | `accumulate`| 
| :---------: | :-----: | :------------------: | :-----: | :---------: |
|多节点| 2±0.01秒| 2.13±0.08秒| **0.91±0.11s** | **0.91±0.11s** |
|单节点 | 0.50±0.01秒| 0.50±0.01秒| **0.41±0.015s** | **0.41±0.015s** |

正如您所看到的，如果您不小心设置梯度同步，您在训练期间可能会出现超过 2 倍的减速！如果您担心确保一切都正确完成，我们强烈建议使用 [accumulate()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.accumulate) 函数并传入
`gradient_accumulation_steps` 或 `gradient_accumulation_plugin` 到 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 对象，以便 Accelerate 可以为您处理此问题。

### 使用 FSDP 时，`no_sync` 需要额外的 GPU 内存

请注意，在执行 FSDP 训练时不同步梯度可能会产生不利影响。正如`torch`中所警告的那样，[⟦T37⟧ context manager for FSDP](https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.no_sync)将需要额外的内存。

因此，在使用 FSDP 时，在内存密集型情况下，我们建议将 [GradientAccumulationPlugin](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.GradientAccumulationPlugin) 中的 `sync_each_batch` 设置为 `True` 以禁用 `no_sync`。

请参阅下面的示例，其中我们在 8 个 A100-80GB GPU 上微调 Mixtral（47B 参数）。我们发现，即使对于适度的 `gradient_accumulation_steps=2`，如果启用 `no_sync`，我们也会很快出现内存不足 (OOM)。同样，这是由于 FSDP 的 `no_sync` 造成的额外内存开销。但是，如果通过 `sync_each_batch=True` 禁用 `no_sync`，则 `gradient_accumulation_steps=16` 的内存消耗将恢复为 `gradient_accumulation_steps=1` 的内存消耗。

|型号| `no_sync`（累加=1）| `no_sync`（累加=2）| `no_sync` 禁用（累加=16）
| :-------------: | :-----------------: | :-----------------: | :-----------------: 
混合 8x7B | 69G| OOM | 69G> [!警告] 
> 禁用 `no_sync` 意味着由于额外的数据同步，速度会变慢，如本指南前面部分所述。

### 🤗`accelerate` 中的序列并行
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/sequence_parallelism.md