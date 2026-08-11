<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 快速游览

根据您的训练环境（[torchrun](https://pytorch.org/docs/stable/elastic/run.html)、[DeepSpeed](https://www.deepspeed.ai/) 等）和可用硬件，启动和运行代码的方法有多种。 Accelerate 提供了一个统一的界面，用于在不同的分布式设置上启动和训练，使您能够专注于 PyTorch 训练代码，而不是使代码适应这些不同的设置的复杂问题。这使您可以轻松扩展 PyTorch 代码，以便使用 GPU 和 TPU 等硬件在分布式设置上进行训练和推理。 Accelerate 还提供大模型推理，使通常不适合内存的大型模型的加载和运行推理变得更容易。

本快速浏览介绍了 Accelerate 的三个主要功能：

* 分布式训练脚本的统一命令行启动界面
* 一个训练库，用于调整 PyTorch 训练代码以在不同的分布式设置上运行
* 大模型推理

## 统一启动界面Accelerate 通过 [⟦T12⟧](package_reference/cli#accelerate-config) 命令生成的统一配置文件，自动为任何给定的分布式训练框架（DeepSpeed、FSDP 等）选择适当的配置值。您还可以将配置值显式传递到命令行，这在某些情况下很有帮助，例如您使用 SLURM。

但在大多数情况下，您应该始终先运行 [⟦T13⟧](package_reference/cli#accelerate-config) 来帮助 Accelerate 了解您的训练设置。

```bash
accelerate config
```

[⟦T14⟧](package_reference/cli#accelerate-config) 命令会在 Accelerate 的缓存文件夹中创建并保存 default_config.yaml 文件。该文件存储您的训练环境的配置，这有助于 Accelerate 根据您的机器正确启动您的训练脚本。

配置环境后，您可以使用 [⟦T15⟧](package_reference/cli#accelerate-test) 测试您的设置，它会启动一个简短的脚本来测试分布式环境。

```bash
accelerate test
```

> [!提示]
> 在`accelerate test`或`accelerate launch`命令中添加`--config_file`，用于指定配置文件保存在缓存等非默认位置时的位置。

设置好环境后，使用 [⟦T19⟧](package_reference/cli#accelerate-launch) 启动训练脚本！

```bash
accelerate launch path_to_script.py --args_for_the_script
```

要了解更多信息，请查看 [Launch distributed code](basic_tutorials/launch) 教程，了解有关启动脚本的更多信息。我们还有一个 [configuration zoo](https://github.com/huggingface/accelerate/blob/main/examples/config_yaml_templates)，它展示了许多预制的**最小**示例配置，适用于您可以运行的各种设置。

## 调整训练代码

Accelerate 的下一个主要功能是 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 类，它可以调整您的 PyTorch 代码以在不同的分布式设置上运行。

您只需在训练脚本中添加几行代码即可使其在多个 GPU 或 TPU 上运行。

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ device = accelerator.device
+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for batch in training_dataloader:
      optimizer.zero_grad()
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
+     accelerator.backward(loss)
      optimizer.step()
      scheduler.step()
```

1. 在训练脚本开头导入并实例化 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 类。 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)类初始化分布式训练所需的一切，并根据代码的启动方式自动检测您的训练环境（具有GPU的单台机器、具有多个GPU的机器、具有多个GPU或TPU的多台机器等）。

```python
from accelerate import Accelerator

accelerator = Accelerator()
```

2. 删除模型上的 `.cuda()` 等调用并输入数据。 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 类会自动将这些对象放置在适合您的设备上。> [!警告]
> 此步骤是*可选*，但允许 Accelerate 处理设备放置被认为是最佳实践。您还可以在初始化 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 时通过传递 `device_placement=False` 来停用自动设备放置。如果您想使用 `.to(device)` 在设备上显式放置对象，请确保使用 `accelerator.device`。例如，如果您在将模型放置在 `accelerator.device` 上之前创建优化器，则 TPU 上的训练会失败。

> [!警告]
> Accelerate 默认情况下不使用非阻塞传输来进行自动设备放置，这可能会导致潜在不需要的 CUDA 同步。  您可以通过在初始化[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)时传递[DataLoaderConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.DataLoaderConfiguration)并将`non_blocking=True`设置为`dataloader_config`来启用非阻塞传输。  像往常一样，只有当数据加载器也设置了 `pin_memory=True` 时，非阻塞传输才会起作用。  请注意，如果使用从 GPU 到 CPU 的非阻塞传输导致 CPU 操作在未就绪的张量上执行，则可能会导致不正确的结果。

```py
device = accelerator.device
```3. 创建后立即将所有相关的 PyTorch 训练对象（优化器、模型、数据加载器、学习率调度程序）传递给 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 方法。此方法将模型包装在针对分布式设置进行优化的容器中，使用优化器和调度程序的加速版本，并创建数据加载器的分片版本以跨 GPU 或 TPU 进行分发。

```python
model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)
```

4. 将 `loss.backward()` 替换为 [backward()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.backward)，以便为您的训练设置使用正确的 `backward()` 方法。

```py
accelerator.backward(loss)
```

阅读 [Accelerate’s internal mechanisms](concept_guides/internal_mechanism) 指南，了解有关 Accelerate 如何调整代码的更多详细信息。

### 分布式评估

要执行分布式评估，请将验证数据加载器传递给 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 方法：

```python
validation_dataloader = accelerator.prepare(validation_dataloader)
```分布式设置中的每个设备仅接收一部分评估数据，这意味着您应该使用 [gather_for_metrics()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics) 方法将预测分组在一起。此方法要求每个进程上的所有张量都具有相同的大小，因此如果每个进程上的张量具有不同的大小（例如，当动态填充到批次中的最大长度时），您应该使用 [pad_across_processes()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.pad_across_processes) 方法将张量跨进程填充到最大大小。请注意，张量必须是一维的，并且我们沿着第一维连接张量。 

```python
for inputs, targets in validation_dataloader:
    predictions = model(inputs)
    # Gather all predictions and targets
    all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
    # Example of use with a *Datasets.Metric*
    metric.add_batch(all_predictions, all_targets)
```

对于更复杂的情况（例如 2D 张量、不想连接张量、3D 张量的字典），您可以在 `gather_for_metrics` 中传递 `use_gather_object=True`。这将在收集后返回对象列表。请注意，将其与 GPU 张量一起使用并没有得到很好的支持并且效率低下。

> [!提示]
> 数据集末尾的数据可能会重复，因此批次可以在所有工作人员之间平均分配。 [gather_for_metrics()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics)方法会自动删除重复数据以计算更准确的指标。

## 大模型推理

Accelerate 的大模型推理有两个主要功能：[init_empty_weights()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.init_empty_weights) 和 [load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)，用于加载通常无法装入内存的大型模型进行推理。> [!提示]
> 查看 [Handling big models for inference](concept_guides/big_model_inference) 指南，以更好地了解大模型推理的底层工作原理。

### 空权重初始化

[init_empty_weights()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.init_empty_weights) 上下文管理器通过创建*模型骨架*并在每次创建参数时移动和放置参数到 PyTorch 的 [**meta**](https://pytorch.org/docs/main/meta.html) 设备来初始化任何大小的模型。这样，并非所有权重都会立即加载，并且一次仅将模型的一小部分加载到内存中。

例如，加载空的 [Mixtral-8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1) 模型比完全加载模型和权重到 CPU 上占用的内存要少得多。

```py
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config)
```

### 装载和调度重量

[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch) 函数将完整或分片的检查点加载到空模型中，并自动在所有可用设备上分配权重。

`device_map` 参数确定每个模型层的放置位置，指定 `"auto"` 首先将它们放置在 GPU 上，然后是 CPU，如果内存仍然不够，最后将硬盘驱动器作为内存映射张量。使用 `no_split_module_classes` 参数来指示哪些模块不应跨设备拆分（通常是那些具有剩余连接的模块）。

```py
from accelerate import load_checkpoint_and_dispatch

model_checkpoint = "your-local-model-folder"
model = load_checkpoint_and_dispatch(
    model, checkpoint=model_checkpoint, device_map="auto", no_split_module_classes=['Block']
)
```

## 后续步骤现在您已经了解了 Accelerate 的主要功能，接下来的步骤可能包括：

* 查看 [tutorials](basic_tutorials/overview) 了解 Accelerate 的温和演练。如果您不熟悉分布式训练和库，这尤其有用。
* 深入研究 [guides](usage_guides/explore) 了解如何针对特定用例使用 Accelerate。
* 通过阅读[concept guides](concept_guides/internal_mechanism)，加深对 Accelerate 内部工作原理的概念理解。
* 在[API reference](package_reference/accelerator)中查找类和命令以查看可用的参数和选项。

### FP8
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/fp8.md