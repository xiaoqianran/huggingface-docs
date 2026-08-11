<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 启动加速脚本

在上一教程中，向您介绍了如何修改当前的训练脚本以使用 Accelerate。
该代码的最终版本如下所示：

```python
from accelerate import Accelerator

accelerator = Accelerator()

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

但是如何运行这段代码并让它利用可用的特殊硬件呢？

首先，您应该将上述代码重写为函数，并使其可作为脚本调用。例如：

```diff
  from accelerate import Accelerator
  
+ def main():
      accelerator = Accelerator()

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

+ if __name__ == "__main__":
+     main()
```

接下来，您需要使用`accelerate launch`启动它。 

  建议您在使用 `accelerate launch` 之前运行 `accelerate config` 以根据您的喜好配置环境。 
  否则，加速将根据您的系统设置使用非常基本的默认值。

## 使用加速启动

Accelerate 有一个特殊的 CLI 命令，可以帮助您通过 `accelerate launch` 在系统中启动代码。
该命令包含在各种平台上启动脚本所需的所有不同命令，而您不必记住每个命令是什么。

  如果您熟悉在 PyTorch 中启动脚本（例如使用 `torchrun`），您仍然可以执行此操作。不需要使用`accelerate launch`。

您可以使用以下命令快速启动脚本：

```bash
accelerate launch {script_name.py} --arg1 --arg2 ...
```只需将 `accelerate launch` 放在命令的开头，然后像平常一样将其他参数传递给脚本即可！

由于这会运行各种火炬生成方法，因此也可以在此处修改所有预期的环境变量。
例如，以下是如何在单个 GPU 上使用 `accelerate launch`：

```bash
# for cuda device:
CUDA_VISIBLE_DEVICES="0" accelerate launch {script_name.py} --arg1 --arg2 ...
# for xpu device:
ZE_AFFINITY_MASK="0" accelerate launch {script_name.py} --arg1 --arg2 ...
```

您也可以使用`accelerate launch`而不先执行`accelerate config`，但您可能需要手动传入正确的配置参数。
在这种情况下，Accelerate 将为您做出一些超参数决策，例如，如果 GPU 可用，它将默认使用所有 GPU，而不使用混合精度。
以下是如何使用所有 GPU 并在禁用混合精度的情况下进行训练：

```bash
accelerate launch --multi_gpu {script_name.py} {--arg1} {--arg2} ...
```

或者通过指定要使用的 GPU 数量：

```bash
accelerate launch --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

为了获得更具体的信息，您应该自己传入所需的参数。例如，您可以这样 
还将使用混合精度在两个 GPU 上启动相同的脚本，同时避免所有警告： 

```bash
accelerate launch --multi_gpu --mixed_precision=fp16 --num_processes=2 {script_name.py} {--arg1} {--arg2} ...
```

要获取可以传入的参数的完整列表，请运行：

```bash
accelerate launch -h
```

  即使您没有在代码中使用 Accelerate，您仍然可以使用启动器来启动脚本！为了直观地显示这种差异，多 GPU 上的早期 `accelerate launch` 与 `torchrun` 看起来像这样：

```bash
MIXED_PRECISION="fp16" torchrun --nproc_per_node=2 --nnodes=1 {script_name.py} {--arg1} {--arg2} ...
```

您还可以利用启动 CLI 作为 python 模块本身来启动脚本，从而能够传递其他特定于 python 的脚本
启动行为。为此，请使用 `accelerate.commands.launch` 而不是 `accelerate launch`：

```bash
python -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```

如果你想使用任何其他Python标志来执行脚本，你也可以像`-m`一样将它们传入，例如 
下面的示例启用无缓冲的 stdout 和 stderr：

```bash
python -u -m accelerate.commands.launch --num_processes=2 {script_name.py} {--arg1} {--arg2}
```

  您也可以在 CPU 上运行您的代码！这对于玩具模型和数据集的调试和测试很有帮助。 

```bash
accelerate launch --cpu {script_name.py} {--arg1} {--arg2}
```  

## 为什么你应该始终使用`accelerate config`

为什么它有用到你应该**总是**运行`accelerate config`？ 

还记得之前对 `accelerate launch` 和 `torchrun` 的调用吗？
配置后，要使用所需部分运行该脚本，您只需直接使用 `accelerate launch` ，无需传递任何其他内容：

```bash
accelerate launch {script_name.py} {--arg1} {--arg2} ...
```

## 自定义配置正如前面简单提到的，`accelerate launch`主要应该通过组合集合配置来使用 
使用 `accelerate config` 命令制作。这些配置将保存到缓存文件夹中的 `default_config.yaml` 文件中以供加速。 
此缓存文件夹位于（按优先级降序排列）：

- 环境变量`HF_HOME`的内容，后缀为`accelerate`。
- 如果不存在，则你的环境变量`XDG_CACHE_HOME`的内容后缀为
  `huggingface/accelerate`。
- 如果也不存在，则文件夹`~/.cache/huggingface/accelerate`。

要具有多种配置，可以将标志`--config_file`传递给配对的`accelerate launch`命令 
以及自定义 yaml 的位置。 

对于使用 `fp16` 实现混合精度的单台机器上的两个 GPU，示例 yaml 可能如下所示：
```yaml
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: MULTI_GPU
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
use_cpu: false
```

从该自定义 yaml 文件的位置启动脚本如下所示：
```bash
accelerate launch --config_file {path/to/config/my_config_file.yaml} {script_name.py} {--arg1} {--arg2} ...
```

## 多节点训练
使用 Accelerate 进行多节点训练类似于[multi-node training with torchrun](https://pytorch.org/tutorials/intermediate/ddp_series_multinode.html)。启动多节点训练运行的最简单方法是执行以下操作：- 将您的代码库和数据复制到所有节点。 （或将它们放在共享文件系统上）
- 在所有节点上设置 python 包。
- 首先在主单节点上运行`accelerate config`。指定节点数量后，系统会要求您指定每个节点的等级（主节点为 0），以及主进程的 IP 地址和端口。这是工作节点与主进程通信所必需的。之后，您可以在所有节点上复制或发送此配置文件，将`machine_rank`更改为1、2、3等，以避免运行命令（或者直接按照他们的指示使用`torchrun`启动）

完成此操作后，您可以通过在所有节点上运行`accelerate launch`（或`torchrun`）来开始多节点训练运行。

    需要在所有节点上运行该命令才能启动所有内容，而不仅仅是从主节点运行它。您可以使用 SLURM 或不同的流程执行器之类的工具来满足此要求并通过单个命令调用所有内容。建议使用主节点的内网 IP，而不是公网 IP，以获得更好的延迟。这是您在主节点上运行 `hostname -I` 时看到的 `192.168.x.x` 或 `172.x.x.x` 地址。

要更好地了解多节点训练，请查看我们的 [multi-node training with FSDP](https://huggingface.co/blog/ram-efficient-pytorch-fsdp) 示例。

### 从 Jupyter Notebooks 启动分布式训练
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/notebook.md