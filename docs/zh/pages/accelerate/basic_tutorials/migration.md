<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 将加速添加到您的代码中

每个分布式训练框架都有自己的工作方式，这可能需要编写大量自定义代码以使其适应您的 PyTorch 训练代码和训练环境。 Accelerate 提供了一种与这些分布式训练框架交互的友好方式，而无需了解每个框架的具体细节。 Accelerate 会为您处理这些细节，因此您可以专注于训练代码并将其扩展到任何分布式训练环境。

在本教程中，您将学习如何使用 Accelerate 调整现有的 PyTorch 代码，并让您轻松地在分布式系统上进行培训！您将从基本的 PyTorch 训练循环开始（它假设已经设置了所有训练对象，如 `model` 和 `optimizer`），并逐步将 Accelerate 集成到其中。

```python
device = "cuda"
model.to(device)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

## 加速器[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 是用于调整代码以与 Accelerate 配合使用的主类。它了解您正在使用的分布式设置，例如不同进程的数量和您的硬件类型。此类还提供对许多必要方法的访问，使您的 PyTorch 代码能够在任何分布式训练环境中工作以及跨设备管理和执行流程。

这就是为什么您应该始终从在脚本中导入和创建 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 实例开始。

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 还知道将 PyTorch 对象移动到哪个设备，因此建议让 Accelerate 为您处理此问题。

```diff
- device = "cuda"
+ device = accelerator.device
  model.to(device)
```

## 准备 PyTorch 对象

接下来，您需要为分布式训练准备 PyTorch 对象（模型、优化器、调度器等）。 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 方法负责将模型放置在适合训练设置的适当容器（例如单 GPU 或多 GPU）中，调整优化器和调度程序以使用 Accelerate 的 [AcceleratedOptimizer](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.optimizer.AcceleratedOptimizer) 和 [AcceleratedScheduler](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.scheduler.AcceleratedScheduler)，并创建一个可以跨进程分片的新数据加载器。

> [!提示]
> Accelerate 仅准备从各自的 PyTorch 类继承的对象，例如 `torch.optim.Optimizer`。PyTorch 对象按照发送的顺序返回。

```py
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
```

## 训练循环

最后，删除训练循环中对输入和目标的 `to(device)` 调用，因为 Accelerate 的 DataLoader 类会自动将它们放置在正确的设备上。您还应该将通常的 `backward()` 通道替换为 Accelerate 的 [backward()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.backward) 方法，该方法会为您缩放梯度，并根据您的分布式设置（例如 DeepSpeed 或 Megatron）使用适当的 `backward()` 方法。

```diff
-   inputs = inputs.to(device)
-   targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
-   loss.backward()
+   accelerator.backward(loss)
```

将所有内容放在一起，您的新加速训练循环现在应该如下所示！

```python
from accelerate import Accelerator
accelerator = Accelerator()

device = accelerator.device
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

for batch in training_dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
```

## 训练特点

Accelerate 提供了额外的功能，例如梯度累积、梯度裁剪、混合精度训练等，您可以将其添加到脚本中以改进训练运行。让我们来探讨一下这三个功能。

### 梯度累积

梯度累积使您能够在更新权重之前通过累积多个批次的梯度来训练更大的批次大小。这对于解决内存限制很有用。要在 Accelerate 中启用此功能，请在 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 类中指定 `gradient_accumulation_steps` 参数，并将 [accumulate()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.accumulate) 上下文管理器添加到脚本中。

```diff
+ accelerator = Accelerator(gradient_accumulation_steps=2)
  model, optimizer, training_dataloader = accelerator.prepare(model, optimizer, training_dataloader)

  for input, label in training_dataloader:
+     with accelerator.accumulate(model):
          predictions = model(input)
          loss = loss_function(predictions, label)
          accelerator.backward(loss)
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```### 渐变裁剪

梯度裁剪是一种防止“梯度爆炸”的技术，Accelerate 提供：

* [clip_grad_value_()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_value_) 将渐变裁剪为最小值和最大值
* [clip_grad_norm_()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_) 用于将梯度标准化为某个值

### 混合精度

混合精度通过使用较低精度的数据类型（如 fp16（半精度））来计算梯度来加速训练。为了获得 Accelerate 的最佳性能，应该在模型内部计算损失（就像在 Transformers 模型中一样），因为模型外部的计算是以全精度计算的。

设置要在 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用的混合精度类型，然后使用 [autocast()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.autocast) 上下文管理器自动将值转换为指定的数据类型。

> [!警告]
> Accelerate 启用自动混合精度，因此仅当除了已经处理缩放的 [backward()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.backward) 对损失执行的操作之外还有其他混合精度操作时才需要[autocast()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.autocast)。

```diff
+ accelerator = Accelerator(mixed_precision="fp16")
+ with accelerator.autocast():
      loss = complex_loss_function(outputs, target)
```

## 保存并加载

训练完成后，加速还可以保存和加载*模型*，或者您也可以保存模型和优化器*状态*，这对于恢复训练可能很有用。

＃＃＃ 模型所有过程完成后，在保存之前使用 [unwrap_model()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.unwrap_model) 方法解开模型，因为 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 方法将模型包装到正确的接口中以进行分布式训练。如果您不解开模型，保存模型状态字典还会保存较大模型中任何潜在的额外层，并且您将无法将权重加载回基本模型中。

您应该使用 [save_model()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_model) 方法来解包并保存模型状态字典。此方法还可以将模型保存到分片检查点或[safetensors](https://hf.co/docs/safetensors/index)格式中。

```py
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory)
```

对于[Transformers](https://hf.co/docs/transformers/index)库中的模型，使用[save_pretrained](https://huggingface.co/docs/transformers/v5.11.0/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)方法保存模型，以便可以使用[from_pretrained](https://huggingface.co/docs/transformers/v5.11.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)方法重新加载。

```py
from transformers import AutoModel

unwrapped_model = accelerator.unwrap_model(model)
unwrapped_model.save_pretrained(
    "path/to/my_model_directory",
    is_main_process=accelerator.is_main_process,
    save_function=accelerator.save,
)

model = AutoModel.from_pretrained("path/to/my_model_directory")
```

要加载权重，请先使用 [unwrap_model()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.unwrap_model) 方法打开模型，然后再加载权重。所有模型参数都是对张量的引用，因此这会将您的权重加载到 `model` 中。

```py
unwrapped_model = accelerator.unwrap_model(model)
path_to_checkpoint = os.path.join(save_directory,"pytorch_model.bin")
unwrapped_model.load_state_dict(torch.load(path_to_checkpoint))
```

设置 `safe_serialization=True` 以 safetensor 格式保存模型。

```py
accelerator.wait_for_everyone()
accelerator.save_model(model, save_directory, max_shard_size="1GB", safe_serialization=True)
```

要加载分片检查点或 safetensor 格式的检查点，请使用 [load_checkpoint_in_model()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.load_checkpoint_in_model) 方法。此方法允许您将检查点加载到特定设备上。

```py
load_checkpoint_in_model(unwrapped_model, save_directory, device_map={"":device})
```

### 状态在训练期间，您可能希望保存模型、优化器、随机生成器和潜在的学习率调度程序的当前状态，以便可以在*同一脚本*中恢复它们。您应该将 [save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) 和 [load_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.load_state) 方法添加到脚本中以保存和加载状态。

要进一步自定义通过 [save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) 保存状态的位置和方式，请使用 [ProjectConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.ProjectConfiguration) 类。例如，如果启用`automatic_checkpoint_naming`，则每个保存的检查点都存储在`Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`。

要存储的任何其他有状态项目应使用 [register_for_checkpointing()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.register_for_checkpointing) 方法注册，以便可以保存和加载它们。传递给此方法要存储的每个对象都必须具有 `load_state_dict` 和 `state_dict` 函数。

> [!提示]
> 如果您安装了[⟦T26⟧](https://github.com/pytorch/data/tree/main)，您还可以将`use_stateful_dataloader=True`传递到您的[DataLoaderConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.DataLoaderConfiguration)。这使用 `load_state_dict` 和 `state_dict` 函数扩展了 Accelerate 的 DataLoader 类，并且使得 `Accelerator.save_state` 和 `Accelerator.load_state` 还可以跟踪在持久化模型时它已读取训练数据集的程度。

### TPU 训练
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/tpu.md