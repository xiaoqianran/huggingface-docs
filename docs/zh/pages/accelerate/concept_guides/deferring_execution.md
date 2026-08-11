<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 执行和推迟作业

当您运行常用脚本时，指令将按顺序执行。使用 Accelerate 将脚本部署到多个
GPU 同时引入了一个复杂性：虽然每个进程按顺序执行所有指令，但有些指令可能会被执行。
比其他人更快。

在执行给定指令之前，您可能需要等待所有进程都达到某个点。对于
例如，在确保每个过程都通过训练完成之前，您不应该保存模型，而且您也不希望 
在加载所有模型权重之前继续训练。为此，只需在代码中编写以下行：

```
accelerator.wait_for_everyone()
```

该指令将阻塞所有首先到达的进程，直到所有其他进程都到达该指令
一点（如果您只在一个 GPU 或 CPU 上运行脚本，这不会执行任何操作）。

下面列出了何时使用此实用程序的一些示例：

    其中一些与[main_process_first()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.main_process_first)上下文管理器一起使用，它利用[wait_for_everyone()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.wait_for_everyone)来 
    在触发和启动其他进程之前，预先在主进程上运行一组特定的代码

## 下载数据集下载数据集时，应先在主进程中下载，然后再加载缓存的数据集

    `load_dataset` 将在后台执行锁定以阻止同时发生多个下载，但如果您正在下载某些内容 
    不使用这个库，你应该使用这个方法。
    

```python
with accelerator.main_process_first():
    datasets = load_dataset("glue", "mrpc")
```

在底层，这与调用相同： 

```python
# First do something on the main process
if accelerator.is_main_process:
    datasets = load_dataset("glue", "mrpc")
else:
    accelerator.wait_for_everyone()

# And then send it to the rest of them
if not accelerator.is_main_process:
    datasets = load_dataset("glue", "mrpc")
else:
    accelerator.wait_for_everyone()
```

## 保存`state_dict`

保存模型的 `state_dict` 时，因为您通常只在主进程上保存一个文件
您应该指定：

```python
if accelerator.is_main_process:
    model = accelerator.unwrap_model(model)
    torch.save(model.state_dict(), "weights.pth")
```

## 加载到`state_dict`

当将 `state_dict` 加载到模型、优化器或调度器时，您应该等待 
让所有工人在开始训练之前加载重量

```python
with accelerator.main_process_first():
    state = torch.load("weights.pth")
    model.load_state_dict(state)
```

## 应用多worker CPU 操作 

对多个工作人员应用 `map()` 操作，例如应在 
首先是主进程，然后传播到每个进程。 

```python
datasets = load_dataset("glue", "mrpc")

with accelerator.main_process_first():
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["idx", "sentence1", "sentence2"],
    )
```

## 应用提前停止等检查要对特定进程设置的标志进行检查，应使用 `set_trigger` 和 `check_trigger` API。有用的例子
为此，可以包括使用提前停止和监控损失等情况（因为每个过程中的每个损失都略有不同）。

当满足条件时调用[Accelerator.set_trigger()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.set_trigger)，并在检查任何过程中是否满足该条件时调用[Accelerator.check_trigger()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.check_trigger)：

```python
for (x,y) in data_loader:
    logits = model(x)
    loss = loss_func(logits, y)
    # Assume `should_do_early_stopping` is a custom defined function that returns a conditional
    if should_do_early_stopping(loss):
        accelerator.set_trigger()

    # Later in the training script when we need to check for the breakpoint
    if accelerator.check_trigger():
        break
```

###加速内部机制
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/internal_mechanism.md