<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 疑难解答

本指南提供了使用 Accelerate 时可能遇到的一些问题的解决方案。并未涵盖所有错误，因为 Accelerate 是一个不断发展的活跃库，并且有许多不同的用例和分布式训练设置。如果此处描述的解决方案无法帮助解决您的特定错误，请查看 [Ask for help](#ask-for-help) 部分，了解在哪里以及如何获得帮助。

## 日志记录

日志记录可以帮助您确定错误来自何处。在具有多个进程的分布式设置中，日志记录可能是一个挑战，但 Accelerate 提供了 `logging()` 实用程序来确保日志同步。

要解决问题，请使用 `logging()` 而不是标准 Python [⟦T11⟧](https://docs.python.org/3/library/logging.html#module-logging) 模块。使用 `log_level` 参数设置详细级别（`INFO`、`DEBUG`、`WARNING`、`ERROR`、`CRITICAL`），然后您可以：

1. 将`log_level`导出为`ACCELERATE_LOG_LEVEL`环境变量。
2. 将`log_level`直接转至`get_logger`。

例如，设置`log_level="INFO"`：

```py
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")
```

默认情况下，仅在主进程上调用日志。要在所有进程上调用它，请传递 `main_process_only=False`。
如果应该在所有进程上按顺序调用日志，还可以传递 `in_order=True`。

```py
from accelerate.logging import get_logger

logger = get_logger(__name__, log_level="DEBUG")
# log all processes
logger.debug("thing_to_log", main_process_only=False)
# log all processes in order
logger.debug("thing_to_log", main_process_only=False, in_order=True)
```

## 挂起代码和超时错误代码挂起的原因可能有很多。让我们看一下如何解决一些可能导致代码挂起的最常见问题。

### 不匹配的张量形状

张量形状不匹配是一个常见问题，可能会导致您的代码在分布式设置上挂起很长时间。

在分布式设置中运行脚本时，需要使用 [Accelerator.gather()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.gather) 和 [Accelerator.reduce()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.reduce) 等函数来跨设备抓取张量，以便共同对它们执行操作。这些（和其他）函数依赖于 `torch.distributed` 来执行 `gather` 操作，这要求张量在所有进程中具有**完全相同的形状**。当张量形状不匹配时，您的代码将挂起，最终会遇到超时异常。

您可以使用 Accelerate 的操作调试模式立即捕获此问题。我们建议在 `accelerate config` 设置期间启用此模式，但您也可以从 CLI、作为环境变量或通过手动编辑 `config.yaml` 文件来启用它。

```bash
accelerate launch --debug {my_script.py} --arg1 --arg2
```

如果将调试模式作为环境变量启用，则无需调用`accelerate launch`。

```bash
ACCELERATE_DEBUG_MODE="1" torchrun {my_script.py} --arg1 --arg2
```

将 `debug: true` 添加到您的 `config.yaml` 文件中。

```yaml
compute_environment: LOCAL_MACHINE
debug: true
```启用调试模式后，您应该获得指向张量形状不匹配问题的回溯。

```py
Traceback (most recent call last):
  File "/home/zach_mueller_huggingface_co/test.py", line 18, in <module>
    main()
  File "/home/zach_mueller_huggingface_co/test.py", line 15, in main
    broadcast_tensor = broadcast(tensor)
  File "/home/zach_mueller_huggingface_co/accelerate/src/accelerate/utils/operations.py", line 303, in wrapper
accelerate.utils.operations.DistributedOperationException:

Cannot apply desired operation due to shape mismatches. All shapes across devices must be valid.

Operation: `accelerate.utils.operations.broadcast`
Input shapes:
  - Process 0: [1, 5]
  - Process 1: [1, 2, 5]
```

### 提前停止

对于分布式训练中的早期停止，如果每个进程都有特定的停止条件（例如验证损失），则可能不会在所有进程之间同步。因此，中断可能发生在进程 0 上，但不会发生在进程 1 上，这将导致代码无限期挂起，直到发生超时。

如果您有提前停止条件，请使用 `set_trigger` 和 `check_trigger` 方法来确保所有进程
已正确结束。

```py
# Assume `should_do_breakpoint` is a custom-defined function that returns a conditional, 
# and that conditional might be true only on process 1
if should_do_breakpoint(loss):
    accelerator.set_trigger()

# Later in the training script when we need to check for the breakpoint
if accelerator.check_trigger():
    break
```

### Linux 上的低内核版本

在内核版本 < 5.5 的 Linux 上，已报告挂起进程。要避免此问题，请将系统升级到更高的内核版本。

### MPI

如果使用 MPI 的分布式 CPU 训练作业挂起，请确保您有
[passwordless SSH](https://www.open-mpi.org/faq/?category=rsh#ssh-keys) 节点之间的设置（使用密钥）。这意味着
对于主机文件中的所有节点，您应该能够从一个节点通过 SSH 连接到另一个节点，而不会提示输入密码。

接下来，尝试运行 `mpirun` 命令作为健全性检查。例如，下面的命令应该打印出
每个节点的主机名。

```bash
mpirun -f hostfile -n {number of nodes} -ppn 1 hostname
```

## 内存不足运行训练脚本时最令人沮丧的错误之一是在 CUDA、XPU 或 CPU 等设备上遇到“内存不足”。整个脚本需要重新启动，任何进度都会丢失。

为了解决这个问题，Accelerate 提供了很大程度上基于 [toma](https://github.com/BlackHC/toma) 的[find_executable_batch_size()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.find_executable_batch_size) 实用程序。
该实用程序会重试因 OOM（内存不足）情况而失败的代码，并自动降低批处理大小。对于每个 OOM 条件，算法会将批大小减少一半并重试代码，直到成功。

要使用 [find_executable_batch_size()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.find_executable_batch_size)，请重构您的训练函数以包含带有 `find_executable_batch_size` 的内部函数，并在其中构建数据加载器。至少，这只需要 4 行新代码。

 

内部函数**必须**将批量大小作为第一个参数，但我们在调用时不会向它传递一个参数。包装器将为您处理这个问题。任何消耗设备内存并传递给 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 的对象（模型、优化器）也**必须**在内部函数内声明。

```diff
def training_function(args):
    accelerator = Accelerator()

+   @find_executable_batch_size(starting_batch_size=args.batch_size)
+   def inner_training_loop(batch_size):
+       nonlocal accelerator # Ensure they can be used in our context
+       accelerator.free_memory() # Free all lingering references
        model = get_model()
        model.to(accelerator.device)
        optimizer = get_optimizer()
        train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)
        lr_scheduler = get_scheduler(
            optimizer, 
            num_training_steps=len(train_dataloader)*num_epochs
        )
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )
        train(model, optimizer, train_dataloader, lr_scheduler)
        validate(model, eval_dataloader)
+   inner_training_loop()
```

## 设备设置之间的不可重现结果如果您更改了设备设置并观察到不同的模型性能，则在从一种设置转移到另一种设置时，您很可能没有更新脚本。即使您使用具有相同批处理大小的相同脚本，TPU、多 GPU 和单 GPU 上的结果仍然会有所不同。

例如，如果您在批量大小为 16 的单 GPU 上进行训练，然后转移到双 GPU 设置，则需要将批量大小更改为 8，以获得相同的有效批量大小。这是因为在使用 Accelerate 进行训练时，传递给数据加载器的批量大小是**每个 GPU 的批量大小**。

为了确保您可以在设置之间重现结果，请确保使用相同的种子，相应地调整批量大小，并考虑缩放学习率。

有关批量大小的更多详细信息和快速参考，请查看 [Comparing performance between different device setups](../concept_guides/performance) 指南。

## 不同GPU上的性能问题

如果您的多 GPU 设置由不同的 GPU 组成，您可能会遇到一些性能问题：- GPU 之间的 GPU 内存可能不平衡。在这种情况下，具有较小内存的 GPU 将限制批量大小或可以加载到 GPU 上的模型的大小。
- 如果您使用具有不同性能配置文件的 GPU，则性能将由您使用的最慢的 GPU 驱动，因为其他 GPU 必须等待它完成其工作负载。

同一设置中截然不同的 GPU 可能会导致性能瓶颈。

## 寻求帮助

如果此处的解决方案和建议都无法帮助解决您的问题，您可以随时向社区和 Accelerate 团队寻求帮助。

- 通过在 [Accelerate category](https://discuss.huggingface.co/c/accelerate/18) 中发布您的问题，在 Hugging Face 论坛上寻求帮助。确保写一篇描述性文章，其中包含有关您的设置和可重现代码的相关上下文，以最大程度地解决您的问题！

- 在[Discord](http://hf.co/join/discord)上发布问题，让团队和社区帮助您。

- 如果您认为发现了与该库相关的错误，请在 Accelerate [GitHub repository](https://github.com/huggingface/accelerate/issues) 上创建问题。包括有关错误的上下文和有关分布式设置的详细信息，以帮助我们更好地找出问题所在以及如何修复它。### 概述
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/overview.md