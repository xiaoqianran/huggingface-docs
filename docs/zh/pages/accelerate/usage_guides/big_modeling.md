<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 大模型推理

Accelerate 提供的最大进步之一是[Big Model Inference](../concept_guides/big_model_inference)，它允许您使用不完全适合您的显卡的模型执行推理。

本教程将向您展示如何在 Accelerate 和 Hugging Face 生态系统中使用大模型推理。

## 加速

加载 PyTorch 模型的典型工作流程如下所示。 `ModelClass` 是超出设备 GPU 内存（mps 或 cuda 或 xpu）的模型。

```py
import torch

my_model = ModelClass(...)
state_dict = torch.load(checkpoint_file)
my_model.load_state_dict(state_dict)
```

使用大模型推理，第一步是使用 `init_empty_weights` 上下文管理器初始化模型的空骨架。这不需要任何内存，因为 `my_model` 是“无参数”的。

```py
from accelerate import init_empty_weights
with init_empty_weights():
    my_model = ModelClass(...)
```

接下来，将权重加载到模型中进行推理。

[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch) 方法在空模型内加载一个检查点，并在所有可用设备上分配每一层的权重，首先从最快的设备（GPU、MPS、XPU、NPU、MLU、SDAA、MUSA）开始，然后再转移到较慢的设备（CPU 和硬盘驱动器）。

设置 `device_map="auto"` 会首先自动填充 GPU 上的所有可用空间，然后是 CPU，最后是硬盘驱动器（绝对最慢的选项）（如果内存仍然不足）。> [!提示]
> 有关如何设计自己的设备映射的更多详细信息，请参阅[Designing a device map](../concept_guides/big_model_inference#designing-a-device-map)指南。

```py
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto"
)
```

如果某些层“块”不应该被分割，请将它们传递给`no_split_module_classes`（有关更多详细信息，请参阅[here](../concept_guides/big_model_inference#loading-weights)）。

模型权重还可以分为多个检查点以节省内存，例如当 `state_dict` 不适合内存时（请参阅 [here](../concept_guides/big_model_inference#sharded-checkpoints) 了解更多详细信息）。

现在模型已完全调度，您可以执行推理。

```py
input = torch.randn(2,3)
device_type = next(iter(model.parameters())).device.type
input = input.to(device_type)
output = model(input)
```

每次输入通过一个层时，它都会从 CPU 发送到 GPU（或从磁盘到 CPU 再到 GPU），计算输出，然后从 GPU 中删除该层，然后返回。虽然这会增加一些推理开销，但只要最大的层适合您的 GPU，您就可以在系统上运行任何大小的模型。

可以利用多个 GPU 或“模型并行性”，但在任何给定时刻只有一个 GPU 处于活动状态。这会强制 GPU 等待前一个 GPU 向其发送输出。您应该使用 Python 正常启动脚本，而不是使用 torchrun 和加速启动等其他工具。> [!提示]
> 您可能还对“管道并行性”感兴趣，它一次利用所有可用的 GPU，而不是一次只有一个 GPU 处于活动状态。但这种方法不太灵活。欲了解更多详情，请参阅[Memory-efficient pipeline parallelism](./distributed_inference#memory-efficient-pipeline-parallelism-experimental)指南。

请看下面的大模型推理的完整示例。

```py
import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

with init_empty_weights():
    model = MyModel(...)

model = load_checkpoint_and_dispatch(
    model, checkpoint=checkpoint_file, device_map="auto"
)

input = torch.randn(2,3)
device_type = next(iter(model.parameters())).device.type
input = input.to(device_type)
output = model(input)
```

## Hugging Face 生态系统

Hugging Face 生态系统中的其他库（例如 Transformers 或 Diffusers）在其 [from_pretrained](https://huggingface.co/docs/transformers/v5.11.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) 构造函数中支持大模型推理。

您只需在[from_pretrained](https://huggingface.co/docs/transformers/v5.11.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)中添加`device_map="auto"`即可启用大模型推理。

例如，使用大模型推理加载 Big Sciences T0pp 110 亿参数模型。

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto")
```

加载模型后，将执行之前的空初始化和智能调度步骤，并且模型已完全准备好使用计算机中的所有资源。通过这些构造函数，您还可以通过指定 `torch_dtype` 参数以较低精度加载模型来节省更多内存。

```py
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", device_map="auto", torch_dtype=torch.float16)
```

## 后续步骤

有关大模型推理的更详细说明，请务必查看[conceptual guide](../concept_guides/big_model_inference)！

### 动物园示例
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/training_zoo.md