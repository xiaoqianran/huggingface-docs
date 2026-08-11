<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 执行过程

使用分布式训练系统时，管理跨 GPU 执行进程的方式和时间非常重要。有些流程比其他流程完成得更快，而有些流程如果其他流程尚未完成则不应开始。 Accelerate 提供了在执行流程时进行编排的工具，以确保所有内容在所有设备上保持同步。

本教程将教您如何仅在一台机器上执行进程以及如何延迟执行直到所有进程都达到某一点。

## 在一个进程上执行

某些代码只需要在给定机器上运行一次，例如打印一条日志语句或仅在本地主进程上显示一个进度条。

您应该使用 `accelerator.is_local_main_process` 来指示只应执行一次的代码。

```py
from tqdm.auto import tqdm

progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
```

您还可以使用 `accelerator.is_local_main_process` 来包装语句。

> [!提示]
> 对于未包装在 `accelerator.is_local_main_process` 中的独立 `print` 语句，请将 `print` 替换为 Accelerate 的 [print()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.print) 方法，以便每个进程仅打印一次。

```py
if accelerator.is_local_main_process:
    print("Accelerate is the best")
```

对于只应执行一次的函数，请使用[on_local_main_process()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.on_local_main_process)。

```py
@accelerator.on_local_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once_per_server()
```您还可以指示 Accelerate 在“所有进程”中执行一次代码，无论机器数量如何。如果您要将最终模型上传到 Hub，这非常有用。

您应该使用 `accelerator.is_main_process` 来指示只应在所有进程中执行一次的代码。

```py
if accelerator.is_main_process:
    repo.push_to_hub()
```

对于只应在所有进程中执行一次的函数，请使用[on_main_process()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.on_main_process)。

```py
@accelerator.on_main_process
def do_my_thing():
    "Something done once per server"
    do_thing_once()
```

## 在特定进程上执行

Accelerate 还可以帮助您执行只应在特定进程或本地进程索引上执行的函数。

使用 [on_process()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.on_process) 方法并指定要执行函数的进程索引。

```py
@accelerator.on_process(process_index=0)
def do_my_thing():
    "Something done on process index 0"
    do_thing_on_index_zero()
```

使用 [on_local_process()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.on_local_process) 方法并指定要执行函数的本地进程索引。

```py
@accelerator.on_local_process(local_process_idx=0)
def do_my_thing():
    "Something done on process index 0 on each server"
    do_thing_on_index_zero_on_each_server()
```

## 推迟执行

当您同时在多个 GPU 上运行脚本时，某些代码的执行速度可能会比其他代码快。您可能需要等待所有进程到达某个点，然后再执行下一组指令。例如，在确保每个过程都经过训练之前，您不应该保存模型。为此，请在代码中添加 [wait_for_everyone()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.wait_for_everyone)。这会阻止所有先完成的进程继续运行，直到所有剩余进程都达到同一点（如果您在单个 GPU 或 CPU 上运行，这不会产生任何影响）。

```py
accelerator.wait_for_everyone()
```

### 故障排除
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/troubleshooting.md