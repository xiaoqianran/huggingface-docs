<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 分布式推理

分布式推理可以分为两类：

1. 将整个模型加载到每个 GPU 上，并一次通过每个 GPU 的模型副本发送一批数据块
2. 将模型的各个部分加载到每个 GPU 上并一次处理单个输入

本指南涵盖第一个支架。对于第二个，请参见[Big Model Inference](../concept_guides/big_model_inference)。

## 自动将批次的块发送到每个加载的模型

这是内存最密集的解决方案，因为它要求每个 GPU 在给定时间在内存中保存模型的完整副本。 

通常，执行此操作时，用户将模型发送到特定设备以从 CPU 加载它，然后将每个提示移动到不同的设备。 

使用 `diffusers` 库的基本管道可能如下所示：

```python
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
```
然后根据具体提示进行推理：

```python
def run_inference(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    pipe.to(rank)

    if torch.distributed.get_rank() == 0:
        prompt = "a dog"
    elif torch.distributed.get_rank() == 1:
        prompt = "a cat"

    result = pipe(prompt).images[0]
    result.save(f"result_{rank}.png")
```
人们会注意到我们必须检查排名才能知道要发送什么提示，这可能有点乏味。

用户可能还会认为，通过 Accelerate，使用 `Accelerator` 为此类任务准备数据加载器也可能是 
一个简单的方法来管理这个。 （要了解更多信息，请查看[Quick Tour](../quicktour#distributed-evaluation)中的相关部分）能管得着吗？是的。但是它是否添加了不需要的额外代码：也是的。

通过 Accelerate，我们可以通过使用 [Accelerator.split_between_processes()](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator.split_between_processes) 上下文管理器（`PartialState` 和 `AcceleratorState` 中也存在）来简化此过程。 
该函数将自动将您传递给它的任何数据（无论是提示、一组张量、先前数据的字典等）分割到所有进程（有可能
待填充）供您立即使用。

让我们使用上下文管理器重写上面的示例：

```python
import torch
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
distributed_state = PartialState()
pipe.to(distributed_state.device)

# Assume two processes
with distributed_state.split_between_processes(["a dog", "a cat"]) as prompt:
    result = pipe(prompt).images[0]
    result.save(f"result_{distributed_state.process_index}.png")
```

然后要启动代码，我们可以使用 Accelerate：

如果您已生成要使用`accelerate config`使用的配置文件：

```bash
accelerate launch distributed_inference.py
```

如果您有要使用的特定配置文件：

```bash
accelerate launch --config_file my_config.json distributed_inference.py
```

或者，如果不想创建任何配置文件并在两个 GPU 上启动：

> 注意：您将收到一些关于根据您的系统猜测值的警告。要删除这些，您可以执行`accelerate config default`或通过`accelerate config`创建配置文件。

```bash
accelerate launch --num_processes 2 distributed_inference.py
```

现在，我们已经减少了将这些数据轻松拆分为几行代码所需的样板代码。

但是如果 GPU 的提示分布很奇怪怎么办？例如，如果我们有 3 个提示，但只有 2 个 GPU，该怎么办？在上下文管理器下，第一个 GPU 将接收前两个提示，第二个 GPU 将接收第三个提示，确保 
所有提示都是分开的，不需要任何开销。

*但是*，如果我们随后想要对*所有 GPU* 的结果执行某些操作该怎么办？ （假设将它们全部收集并执行某种后处理）
您可以传入 `apply_padding=True` 以确保提示列表填充到相同的长度，并获取额外的数据 
从最后一个样本。这样，所有 GPU 将具有相同数量的提示，然后您可以收集结果。

仅当尝试执行收集结果等操作时才需要，其中每个设备上的数据 
需要具有相同的长度。基本推理不需要这个。

例如：

```python
import torch
from accelerate import PartialState  # Can also be Accelerator or AcceleratorState
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
distributed_state = PartialState()
pipe.to(distributed_state.device)

# Assume two processes
with distributed_state.split_between_processes(["a dog", "a cat", "a chicken"], apply_padding=True) as prompt:
    result = pipe(prompt).images
```

在第一个 GPU 上，提示将为 `["a dog", "a cat"]`，在第二个 GPU 上，提示将为 `["a chicken", "a chicken"]`。
确保丢弃最终样本，因为它将与前一个样本重复。

您可以找到更复杂的示例[here](https://github.com/huggingface/accelerate/tree/main/examples/inference/distributed)，例如如何将其与法学硕士一起使用。

### 低精度训练方法
https://huggingface.co/docs/accelerate/v1.16.0/usage_guides/low_ precision_training.md