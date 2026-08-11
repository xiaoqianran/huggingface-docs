<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 加速器

[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 是用于在任何类型的训练设置上启用分布式训练的主类。阅读 [Add Accelerator to your code](../basic_tutorials/migration) 教程，了解有关如何将 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 添加到脚本的更多信息。

## 加速器[[api]][[accelerate.Accelerator]]

####加速.加速器[[加速.加速器]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L184)

创建用于分布式训练或混合精度训练的加速器实例。

**可用属性：**

- **设备** (`torch.device`) -- 要使用的设备。
- **distributed_type** ([DistributedType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.DistributedType)) -- 分布式训练配置。
- **local_process_index** (`int`) -- 当前机器上的进程索引。
- **mixed_ precision** (`str`) -- 配置的混合精度模式。
- **num_processes** (`int`) -- 用于训练的进程总数。
- **optimizer_step_was_skipped** (`bool`) -- 优化器更新是否被跳过（因为
  混合精度的梯度溢出），其中
在这种情况下，学习率不应改变。
- **process_index** (`int`) -- 当前进程在所有进程中的总体索引。
- **状态** ([AcceleratorState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.state.AcceleratorState)) -- 分布式设置状态。- **sync_gradients** (`bool`) -- 渐变当前是否在所有进程之间同步。
- **use_distributed** (`bool`) -- 当前配置是否用于分布式训练。

accelerate.Accelerator.accumulatehttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1254[{"name": "*models", "val": ""}]- ***models** (`torch.nn.Module`列表) --
  使用 `Accelerator.prepare` 准备的 PyTorch 模块。传递到`accumulate()`的模型将
  在分布式训练中向后传递期间跳过梯度同步0

一个上下文管理器，将轻轻环绕并自动执行梯度累积

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=1)
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

>>> for input, output in dataloader:
...     with accelerator.accumulate(model):
...         outputs = model(input)
...         loss = loss_func(outputs)
...         loss.backward()
...         optimizer.step()
...         scheduler.step()
...         optimizer.zero_grad()
```

**参数：**

device_placement（`bool`，*可选*，默认为`True`）：加速器是否应将对象放置在设备上（由数据加载器、模型等生成的张量）。混合精度（`str`，*可选*）：是否使用混合精度训练。从“no”、“fp16”、“bf16”或“fp8”中选择。将默认为环境变量`ACCELERATE_MIXED_PRECISION`中的值，这将使用当前系统的加速配置中的默认值或通过`accelerate.launch`命令传递的标志。 “fp8”需要安装变压器引擎。

gradient_accumulation_steps (`int`，*可选*，默认为 1) ：累积梯度之前应经过的步数。 > 1 的数字应与 `Accelerator.accumulate` 组合。如果不传递，将默认为环境变量`ACCELERATE_GRADIENT_ACCUMULATION_STEPS`中的值。也可以通过`GradientAccumulationPlugin`进行配置。

cpu (`bool`, *可选*) : 是否强制脚本在CP​​U上执行。如果设置为`True`，将忽略可用的 GPU，并强制仅在一个进程上执行。

dataloader_config（`DataLoaderConfiguration`，*可选*）：有关如何在分布式场景中处理数据加载器的配置。deepspeed_plugin（[DeepSpeedPlugin](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin) 或 `str` 的字典：[DeepSpeedPlugin](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin)，*可选*）：使用此参数调整与 DeepSpeed 相关的参数。该参数是可选的，可以使用 *accelerate config* 直接配置。如果使用多个插件，请使用每个插件配置的 `key` 属性从 `accelerator.state.get_deepspeed_plugin(key)` 访问它们。 `deepspeed_plugins` 的别名。

fsdp_plugin ([FullyShardedDataParallelPlugin](/docs/accelerate/v1.14.0/en/package_reference/fsdp#accelerate.FullyShardedDataParallelPlugin), *可选*) ：使用此参数调整 FSDP 相关参数。该参数是可选的，可以使用 *accelerate config* 直接配置

torch_tp_plugin (`TorchTensorParallelPlugin`, *可选*) ：已弃用：使用 `parallelism_config` 和 `tp_size` 代替。

megatron_lm_plugin ([MegatronLMPlugin](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.MegatronLMPlugin), *可选*) ：使用此参数调整与 MegatronLM 相关的参数。该参数是可选的，可以使用 *accelerate config* 直接配置rng_types（`str`或[RNGType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.RNGType)列表）：在准备好的数据加载器中每次迭代开始时同步的随机数生成器列表。应该是以下一项或多项： - `"torch"`：基本 Torch 随机数生成器 - `"cuda"`：CUDA 随机数生成器（仅限 GPU） - `"xla"`：XLA 随机数生成器（仅限 TPU） - `"generator"`：采样器（或批量采样器，如果数据加载器中没有采样器）或可迭代数据集（如果存在）（如果基础数据集属于该类型）。  对于 PyTorch 版本 <=1.5.1 and ⟦T82⟧ for PyTorch versions >= 1.6，默认为 `["torch"]`。

log_with（`str`、[LoggerType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.LoggerType)或[GeneralTracker](/docs/accelerate/v1.14.0/en/package_reference/tracking#accelerate.tracking.GeneralTracker)的列表，*可选*）：要设置用于实验跟踪的记录器列表。应该是以下一项或多项： - `"all"` - `"tensorboard"` - `"wandb"` - `"trackio"` - `"aim"` - `"comet_ml"` - `"mlflow"` - `"dvclive"` - `"swanlab"` 如果`"all"`被选中，将拾取环境中所有可用的跟踪器并初始化它们。还可以接受自定义跟踪器的 `GeneralTracker` 实现，并且可以与 `"all"` 组合。

project_config ([ProjectConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.ProjectConfiguration), *可选*) ：如何处理保存状态的配置。project_dir (`str`, `os.PathLike`, *可选*) ：用于存储数据的目录路径，例如本地兼容记录器的日志和可能保存的检查点。

step_scheduler_with_optimizer (`bool`，*可选*，默认为`True`)：如果学习率调度程序与优化器同时步进，则设置`True`，如果仅在某些情况下执行（例如在每个时期结束时），则设置`False`。

kwargs_handlers（[KwargsHandler](/docs/accelerate/v1.14.0/en/package_reference/kwargs#accelerate.utils.KwargsHandler)列表，*可选*）：[KwargsHandler](/docs/accelerate/v1.14.0/en/package_reference/kwargs#accelerate.utils.KwargsHandler)列表，用于自定义如何创建与分布式训练、分析或混合精度相关的对象。请参阅[kwargs](kwargs)了解更多信息。

dynamo_backend（`str`或[DynamoBackend](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.DynamoBackend)，*可选*，默认为`"no"`）：设置为可能的发电机后端之一，以优化您使用火炬发电机的训练。

dynamo_plugin（[TorchDynamoPlugin](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.TorchDynamoPlugin)，*可选*）：如果需要比`backend`或`mode`更多的调整，则应如何处理火炬发电机的配置。

gradient_accumulation_plugin（[GradientAccumulationPlugin](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.GradientAccumulationPlugin)，*可选*）：如果需要比`gradient_accumulation_steps`更多的调整，则应如何处理梯度累积的配置。
#### 自动施法[[accelerate.Accelerator.autocast]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4177)如果启用的话，将在此上下文管理器内的块内应用自动混合精度。什么都没有
否则会发生不同的情况。

可以传入不同的 `autocast_handler` 来覆盖 `Accelerator` 对象中设置的一组。这是
在 `autocast` 下想要恢复到 fp32 的块中很有用。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(mixed_precision="fp16")
>>> with accelerator.autocast():
...     train()
```
#### 向后[[accelerate.Accelerator.backward]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2818)

根据 `GradientAccumulationPlugin` 缩放梯度并调用正确的 `backward()`
关于配置。

应该用来代替 `loss.backward()`。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=2)
>>> outputs = model(inputs)
>>> loss = loss_fn(outputs, labels)
>>> accelerator.backward(loss)
```
#### check_trigger[[accelerate.Accelerator.check_trigger]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2878)

检查任何进程中内部触发张量是否已设置为 1。如果是这样，将返回`True`并且
将触发张量重置为 0。

注意：
不需要`wait_for_everyone()`

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume later in the training script
>>> # `should_do_breakpoint` is a custom function to monitor when to break,
>>> # e.g. when the loss is NaN
>>> if should_do_breakpoint(loss):
...     accelerator.set_trigger()
>>> # Assume later in the training script
>>> if accelerator.check_trigger():
...     break
```
#### 清除[[加速.加速器.清除]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3931)

`Accelerate.free_memory`的别名，释放对存储的内部对象的所有引用并调用
垃圾收集器。您应该在使用不同模型/优化器的两次训练之间调用此方法。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, scheduler = ...
>>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
>>> model, optimizer, scheduler = accelerator.clear(model, optimizer, scheduler)
```
#### Clip_grad_norm_[[accelerate.Accelerator.clip_grad_norm_]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2946)

应该用来代替`torch.nn.utils.clip_grad_norm_`。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=2)
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

>>> for input, target in dataloader:
...     optimizer.zero_grad()
...     output = model(input)
...     loss = loss_func(output, target)
...     accelerator.backward(loss)
...     if accelerator.sync_gradients:
...         accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
...     optimizer.step()
```

**退货：**

``torch.Tensor``参数梯度的总范数（视为单个向量）。
####clip_grad_value_[[accelerate.Accelerator.clip_grad_value_]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3009)

应该用来代替`torch.nn.utils.clip_grad_value_`。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(gradient_accumulation_steps=2)
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)

>>> for input, target in dataloader:
...     optimizer.zero_grad()
...     output = model(input)
...     loss = loss_func(output, target)
...     accelerator.backward(loss)
...     if accelerator.sync_gradients:
...         accelerator.clip_grad_value_(model.parameters(), clip_value)
...     optimizer.step()
```
#### deepspeed_ulysses_dl_adapter[[accelerate.Accelerator.deepspeed_ulysses_dl_adapter]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2486)

这通常作为`prepare`的一部分被调用，但是当数据加载器与模型分开准备时（对于外部加速器.prepare调用），需要在prepare(model)之后进行这个额外的调用（请参阅HF Trainer作为用例）
#### end_training[[accelerate.Accelerator.end_training]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3388)

运行任何特殊的最终训练行为，例如仅在主进程上停止跟踪器或销毁
进程组。如果使用实验跟踪，则应始终在脚本末尾调用。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(log_with="tensorboard")
>>> accelerator.init_trackers("my_project")
>>> # Do training
>>> accelerator.end_training()
```
#### 空闲内存[[accelerate.Accelerator.free_memory]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3902)

将释放对存储的内部对象的所有引用并调用垃圾收集器。你应该这样称呼
使用不同模型/优化器的两次训练之间的方法。还将`Accelerator.step`重置为0。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, scheduler = ...
>>> model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
>>> model, optimizer, scheduler = accelerator.free_memory(model, optimizer, scheduler)
```
#### 聚集[[accelerate.Accelerator.gather]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3036)跨所有进程收集 *tensor* 中的值并将它们连接到第一个维度。有用到
在进行评估时重新组合所有流程的预测。

注意：
这种聚集发生在所有进程中。

示例：

```python
>>> # Assuming four processes
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> process_tensor = torch.tensor([accelerator.process_index], device=accelerator.device)
>>> gathered_tensor = accelerator.gather(process_tensor)
>>> gathered_tensor
tensor([0, 1, 2, 3])
```

**参数：**

张量（`torch.Tensor`，或`torch.Tensor`的嵌套元组/列表/字典）：跨所有进程收集的张量。

**退货：**

``torch.Tensor`，或`torch.Tensor`的嵌套元组/列表/字典`

聚集的张量。请注意，
结果的第一个维度是 *num_processes* 乘以输入张量的第一个维度。
#### Gather_for_metrics[[accelerate.Accelerator.gather_for_metrics]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3068)

如果在分布式系统上，则收集 `input_data` 并可能删除最后一批中的重复项。应该是
用于收集度量计算的输入和目标。

示例：

```python
>>> # Assuming two processes, with a batch size of 5 on a dataset with 9 samples
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader = torch.utils.data.DataLoader(range(9), batch_size=5)
>>> dataloader = accelerator.prepare(dataloader)
>>> batch = next(iter(dataloader))
>>> gathered_items = accelerator.gather_for_metrics(batch)
>>> len(gathered_items)
9
```

**参数：**

input (`torch.Tensor`、`object`、`torch.Tensor`的嵌套元组/列表/字典或`object`的嵌套元组/列表/字典)：用于计算所有进程的度量的张量或对象use_gather_object(`bool`) ：是否强制使用gather_object而不是gather（如果传递的所有对象不包含张量，则已经完成）。该标志对于收集不同大小的张量非常有用，我们不想沿着第一维填充和连接这些张量。将其与 GPU 张量一起使用并没有得到很好的支持并且效率低下，因为它会导致 GPU -> CPU 传输，因为张量会被 pickle。
#### get_state_dict[[accelerate.Accelerator.get_state_dict]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4002)

返回通过 [Accelerator.prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 发送的模型的状态字典，可能没有完整的
精度。

示例：

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> net = torch.nn.Linear(2, 2)
>>> net = accelerator.prepare(net)
>>> state_dict = accelerator.get_state_dict(net)
```

**参数：**

model (`torch.nn.Module`) : 通过[Accelerator.prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)发送的PyTorch模型

unwrap (`bool`, *可选*, 默认为`True`) : 是否返回`model`的原始底层state_dict或返回包装后的state_dict

**退货：**

``dict``

模型的状态字典可能没有完全精度。
#### get_tracker[[accelerate.Accelerator.get_tracker]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3324)

仅在主进程上基于 `name` 从 `self.trackers` 返回 `tracker`。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(log_with="tensorboard")
>>> accelerator.init_trackers("my_project")
>>> tensorboard_tracker = accelerator.get_tracker("tensorboard")
```

**参数：**

name (`str`) ：跟踪器的名称，对应`.name`属性。unwrap (`bool`) ：是否返回内部跟踪机制或返回包装的跟踪器（推荐）。

**退货：**

``GeneralTracker``

`name`对应的跟踪器（如果存在）。
#### join_uneven_inputs[[accelerate.Accelerator.join_uneven_inputs]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1299)

上下文管理器，有助于对不均匀输入进行分布式训练或评估，充当包装器
`torch.distributed.algorithms.join` 左右。当总批量大小不能均匀划分时，这很有用
数据集的长度。

`join_uneven_inputs` 仅支持多个 GPU 上的分布式数据并行训练。对于任何其他
配置后，此方法将不起作用。

重写 `even_batches` 不会影响可迭代风格的数据加载器。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator(even_batches=True)
>>> ddp_model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

>>> with accelerator.join_uneven_inputs([ddp_model], even_batches=False):
...     for input, output in dataloader:
...         outputs = model(input)
...         loss = loss_func(outputs)
...         loss.backward()
...         optimizer.step()
...         optimizer.zero_grad()
```

**参数：**

joinables (`list[torch.distributed.algorithms.Joinable]`)：子类`torch.distributed.algorithms.Joinable`的模型或优化器列表。最常见的是使用 `Accelerator.prepare` 准备的 PyTorch 模块，用于分布式数据并行训练。

Even_batches (`bool`, *可选*) ：如果设置，这将覆盖`Accelerator`中设置的`even_batches`的值。如果未提供，将使用默认的 `Accelerator` 值。
#### load_state[[accelerate.Accelerator.load_state]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3750)加载模型、优化器、缩放器、RNG 生成器和注册对象的当前状态。

只能与 [Accelerator.save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) 一起使用。如果文件未注册
检查点，如果存储在目录中，则不会加载。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, lr_scheduler = ...
>>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
>>> accelerator.load_state("my_checkpoint")
```

**参数：**

input_dir（`str`或`os.PathLike`）：保存所有相关权重和状态的文件夹的名称。如果使用`automatic_checkpoint_naming`，则可以是`None`，并且将从最新的检查点获取。

load_kwargs (`dict`, *可选*) ：底层 `load` 函数的附加关键字参数，例如 state_dict 和优化器的可选参数。

load_model_func_kwargs (`dict`, *可选*) ：用于加载模型的附加关键字参数，可以传递给底层加载函数，例如 DeepSpeed 的 `load_checkpoint` 函数的可选参数或用于加载模型和优化器的 `map_location`。
#### local_main_process_first[[accelerate.Accelerator.local_main_process_first]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1109)

让本地主进程进入 with 块。

主进程退出后，其他进程将进入with块。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> with accelerator.local_main_process_first():
...     # This will be printed first by local process 0 then in a seemingly
...     # random order by the other processes.
...     print(f"This will be printed by process {accelerator.local_process_index}")
```
#### lomo_backward[[accelerate.Accelerator.lomo_backward]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4320)在 LOMO 优化器上运行反向传递。
#### main_process_first[[accelerate.Accelerator.main_process_first]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1087)

让主进程首先进入 with 块。

主进程退出后，其他进程将进入with块。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> with accelerator.main_process_first():
...     # This will be printed first by process 0 then in a seemingly
...     # random order by the other processes.
...     print(f"This will be printed by process {accelerator.process_index}")
```
#### Maybe_context_parallel[[accelerate.Accelerator.maybe_context_parallel]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4110)

支持上下文并行训练的上下文管理器。

`context_parallel` 目前受 FSDP2 支持，并且需要 `parallelism_config.cp_size` >
1. 如果这些条件中的任何一个不满足，则该上下文管理器将不起作用，尽管可以启用更少的
代码更改不会引发异常。

必须在每个训练步骤中重新创建此上下文管理器，如下例所示。

示例：

```python
>>> for batch in dataloader:
...     with accelerator.maybe_context_parallel(
...         buffers=[batch["input_ids"], batch["attention_mask"]],
...         buffer_seq_dims=[1, 1],
...         no_restore_buffers={batch["input_ids"]},
...     ):
...         outputs = model(batch)
...         ...
```

**参数：**

buffers (`list[torch.Tensor]`, `optional`) ：缓冲区，将沿着序列维度进行分片。常见的例子是输入、标签或位置嵌入缓冲区。该上下文管理器将就地修改这些缓冲区，退出上下文后，缓冲区将恢复到原始状态。为了避免不必要的恢复，您可以使用`no_restore_buffers`来指定哪些缓冲区不需要恢复。buffer_seq_dims (`list[int]`, `optional`) ：`buffers`的序列维度。

no_restore_buffers (`set[torch.Tensor]`, `optional`) ：该集合必须是`buffers`的子集。指定上下文退出后不会恢复 `buffers` 参数中的哪些缓冲区。这些缓冲区将保持分片状态。
#### no_sync[[accelerate.Accelerator.no_sync]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1131)

上下文管理器，通过调用禁用跨 DDP 进程的梯度同步
`torch.nn.parallel.DistributedDataParallel.no_sync`。

如果 `model` 不在 DDP 中，则该上下文管理器不执行任何操作

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)
>>> input_a = next(iter(dataloader))
>>> input_b = next(iter(dataloader))

>>> with accelerator.no_sync():
...     outputs = model(input_a)
...     loss = loss_func(outputs)
...     accelerator.backward(loss)
...     # No synchronization across processes, only accumulate gradients
>>> outputs = model(input_b)
>>> accelerator.backward(loss)
>>> # Synchronization across all processes
>>> optimizer.step()
>>> optimizer.zero_grad()
```

**参数：**

model (`torch.nn.Module`) : 使用`Accelerator.prepare`准备的PyTorch模块
#### on_last_process[[accelerate.Accelerator.on_last_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L955)

一个装饰器，仅在最后一个进程上运行装饰函数。也可以使用
`PartialState`类。

示例：
```python
# Assume we have 4 processes.
from accelerate import Accelerator

accelerator = Accelerator()

@accelerator.on_last_process
def print_something():
    print(f"Printed on process {accelerator.process_index}")

print_something()
"Printed on process 3"
```

**参数：**

function (`Callable`) ：要装饰的函数。
#### on_local_main_process[[accelerate.Accelerator.on_local_main_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L913)

一个装饰器，仅在本地主进程上运行装饰函数。也可以使用
`PartialState`类。

示例：
```python
# Assume we have 2 servers with 4 processes each.
from accelerate import Accelerator

accelerator = Accelerator()

@accelerator.on_local_main_process
def print_something():
    print("This will be printed by process 0 only on each server.")

print_something()
# On server 1:
"This will be printed by process 0 only"
# On server 2:
"This will be printed by process 0 only"
```

**参数：**function (`Callable`) ：要装饰的函数。
#### on_local_process[[accelerate.Accelerator.on_local_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1039)

一个装饰器，仅在给定的本地进程索引上运行装饰函数。也可以使用调用
`PartialState` 类。

示例：
```python
# Assume we have 2 servers with 4 processes each.
from accelerate import Accelerator

accelerator = Accelerator()

@accelerator.on_local_process(local_process_index=2)
def print_something():
    print(f"Printed on process {accelerator.local_process_index}")

print_something()
# On server 1:
"Printed on process 2"
# On server 2:
"Printed on process 2"
```

**参数：**

function (`Callable`, *可选*) ：要装饰的函数。

local_process_index (`int`, *可选*) ：运行函数的本地进程的索引。
#### on_main_process[[accelerate.Accelerator.on_main_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L874)

一个装饰器，仅在主进程上运行装饰函数。也可以使用
`PartialState`类。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()

>>> @accelerator.on_main_process
... def print_something():
...     print("This will be printed by process 0 only.")

>>> print_something()
"This will be printed by process 0 only"
```

**参数：**

function (`Callable`) ：要装饰的函数。
#### on_process[[accelerate.Accelerator.on_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L994)

一个装饰器，仅在给定的进程索引上运行装饰函数。也可以使用
`PartialState`类。

示例：
```python
# Assume we have 4 processes.
from accelerate import Accelerator

accelerator = Accelerator()

@accelerator.on_process(process_index=2)
def print_something():
    print(f"Printed on process {accelerator.process_index}")

print_something()
"Printed on process 2"
```

**参数：**

function (`Callable`, `optional`) ：要装饰的函数。

process_index (`int`, `optional`) ：运行函数的进程的索引。
#### pad_across_processes[[accelerate.Accelerator.pad_across_processes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3178)将所有设备中张量的嵌套列表/元组/字典中的张量递归填充到相同的大小，以便
他们可以安全地被聚集。

示例：

```python
>>> # Assuming two processes, with the first processes having a tensor of size 1 and the second of size 2
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> process_tensor = torch.arange(accelerator.process_index + 1).to(accelerator.device)
>>> padded_tensor = accelerator.pad_across_processes(process_tensor)
>>> padded_tensor.shape
torch.Size([2])
```

**参数：**

张量（`torch.Tensor`的嵌套列表/元组/字典）：要收集的数据。

暗淡（`int`，*可选*，默认为 0）：要填充的尺寸。

pad_index (`int`，*可选*，默认为 0) ：要填充的值。

pad_first (`bool`，*可选*，默认为`False`) ：是否在开头或结尾填充。

**退货：**

``torch.Tensor`，或`torch.Tensor`的嵌套元组/列表/字典`

填充的张量。
#### 准备[[加速.加速器.准备]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1414)

准备传入`args`的所有对象以进行分布式训练和混合精度，然后以相同的方式返回它们
订单。

如果您仅将模型用于推理而没有任何类型的混合精度，则无需准备模型

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume a model, optimizer, data_loader and scheduler are defined
>>> model, optimizer, data_loader, scheduler = accelerator.prepare(model, optimizer, data_loader, scheduler)
```

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume a model, optimizer, data_loader and scheduler are defined
>>> device_placement = [True, True, False, False]
>>> # Will place the first two items passed in automatically to the right device but not the last two.
>>> model, optimizer, data_loader, scheduler = accelerator.prepare(
...     model, optimizer, data_loader, scheduler, device_placement=device_placement
... )
```

**参数：**

- ***args**（对象列表）：以下任何类型的对象： - `torch.utils.data.DataLoader`：PyTorch 数据加载器 - `torch.nn.Module`：PyTorch 模块 - `torch.optim.Optimizer`：PyTorch 优化器 - `torch.optim.lr_scheduler.LRScheduler`：PyTorch LR 调度程序device_placement (`list[bool]`, *可选*) ：用于自定义是否应对每个传递的对象执行自动设备放置。需要是与`args`长度相同的列表。与 DeepSpeed 或 FSDP 不兼容。
#### 准备数据加载器[[accelerate.Accelerator.prepare_data_loader]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2674)

准备 PyTorch DataLoader 以在任何分布式设置中进行训练。推荐使用
[Accelerator.prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 代替。

示例：

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> data_loader = torch.utils.data.DataLoader(...)
>>> data_loader = accelerator.prepare_data_loader(data_loader, device_placement=True)
```

**参数：**

data_loader (`torch.utils.data.DataLoader`)：准备一个普通的 PyTorch DataLoader

device_placement (`bool`, *可选*) ：是否将批次放置在准备好的数据加载器中的正确设备上。默认为`self.device_placement`。

slice_fn_for_dispatch (`Callable`, *可选*`) : If passed, this function will be used to slice tensors across `num_processes`. Will default to [slice_tensors()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.slice_tensors). This argument is used only when `dispatch_batches` is set to `True`，否则将被忽略。
#### 准备模型[[accelerate.Accelerator.prepare_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1769)

准备 PyTorch 模型以在任何分布式设置中进行训练。推荐使用
[Accelerator.prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 代替。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume a model is defined
>>> model = accelerator.prepare_model(model)
```

**参数：**

model (`torch.nn.Module`) ：要准备的 PyTorch 模型。如果模型仅用于推理而没有任何类型的混合精度，则无需准备模型device_placement (`bool`, *可选*) ：是否将模型放置在正确的设备上。默认为`self.device_placement`。

评估模式（`bool`，*可选*，默认为`False`）：是否仅通过应用混合精度和`torch.compile`（如果在`Accelerator`对象中配置）来设置模型仅用于评估。
#### 准备优化器[[accelerate.Accelerator.prepare_optimizer]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2733)

准备 PyTorch 优化器以在任何分布式设置中进行训练。推荐使用
[Accelerator.prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 代替。

示例：

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> optimizer = torch.optim.Adam(...)
>>> optimizer = accelerator.prepare_optimizer(optimizer, device_placement=True)
```

**参数：**

优化器 (`torch.optim.Optimizer`) ：准备一个普通的 PyTorch 优化器

device_placement (`bool`, *可选*) ：是否将优化器放置在正确的设备上。默认为`self.device_placement`。
#### 准备调度器[[accelerate.Accelerator.prepare_scheduler]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2777)

准备 PyTorch Scheduler 以在任何分布式设置中进行训练。推荐使用
[Accelerator.prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 代替。

示例：

```python
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> optimizer = torch.optim.Adam(...)
>>> scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, ...)
>>> scheduler = accelerator.prepare_scheduler(scheduler)
```

**参数：**

调度程序（`torch.optim.lr_scheduler.LRScheduler`）：一个普通的 PyTorch 调度程序来准备
#### 打印[[加速.加速器.打印]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1382)

替换`print()`，每个服务器仅打印一次。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> accelerator.print("Hello world!")
```
#### 配置文件[[accelerate.Accelerator.profile]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4202)将分析上下文管理器内的代码。如果出现以下情况，配置文件将保存到 Chrome 跟踪文件中：
`profile_handler.output_trace_dir` 已设置。

可以传入不同的 `profile_handler` 来覆盖 `Accelerator` 对象中设置的一组。

示例：

```python
# Profile with default settings
from accelerate import Accelerator
from accelerate.utils import ProfileKwargs

accelerator = Accelerator()
with accelerator.profile() as prof:
    train()
accelerator.print(prof.key_averages().table())

# Profile with the custom handler
def custom_handler(prof):
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

kwargs = ProfileKwargs(schedule_option=dict(wait=1, warmup=1, active=1), on_trace_ready=custom_handler)
accelerator = Accelerator(kwarg_handler=[kwargs])
with accelerator.profile() as prof:
    for _ in range(10):
        train_iteration()
        prof.step()

# Profile and export to Chrome Trace
kwargs = ProfileKwargs(output_trace_dir="output_trace")
accelerator = Accelerator(kwarg_handler=[kwargs])
with accelerator.profile():
    train()
```

**参数：**

profile_handler (`ProfileKwargs`, *可选*) ：用于此上下文管理器的配置文件处理程序。如果不通过，将使用`Accelerator`对象中设置的一组。
#### 减少[[加速.加速器.减少]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3141)

基于*减少*减少所有进程中*张量*中的值。

注意：
所有流程都会获得减少的价值。

示例：

```python
>>> # Assuming two processes
>>> import torch
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> process_tensor = torch.arange(accelerator.num_processes) + 1 + (2 * accelerator.process_index)
>>> process_tensor = process_tensor.to(accelerator.device)
>>> reduced_tensor = accelerator.reduce(process_tensor, reduction="sum")
>>> reduced_tensor
tensor([4, 6])
```

**参数：**

张量（`torch.Tensor`，或`torch.Tensor`的嵌套元组/列表/字典）：要在所有进程中减少的张量。

归约（`str`，*可选*，默认为“sum”）：归约类型，可以是“sum”、“mean”、“max”或“none”之一。如果为“none”，则不会执行任何操作。

scale (`float`，*可选*，默认为1.0)：reduce后应用的默认缩放值，仅在XLA上有效。

**退货：**

``torch.Tensor`，或`torch.Tensor`的嵌套元组/列表/字典`

减少的张量。
#### register_for_checkpointing[[accelerate.Accelerator.register_for_checkpointing]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4074)记下`objects`，并将在`save_state`或`load_state`期间保存或加载它们。

当在同一脚本中加载或保存状态时应该使用这些。它的设计初衷并不是为了
用于不同的脚本中。

每个`object`必须有一个`load_state_dict`和`state_dict`函数来存储。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume `CustomObject` has a `state_dict` and `load_state_dict` function.
>>> obj = CustomObject()
>>> accelerator.register_for_checkpointing(obj)
>>> accelerator.save_state("checkpoint.pt")
```
#### register_load_state_pre_hook[[accelerate.Accelerator.register_load_state_pre_hook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3719)

注册一个要在 [Accelerator.load_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.load_state) 中调用 `load_checkpoint` 之前运行的预挂钩。

该钩子应具有以下签名：

`hook(models: list[torch.nn.Module], input_dir: str) -> None`

`models`参数是在`accelerator._models`下加速器状态下保存的模型，并且
`input_dir` 参数是传递给 [Accelerator.load_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.load_state) 的 `input_dir` 参数。

只能与 [Accelerator.register_save_state_pre_hook()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.register_save_state_pre_hook) 一起使用。加载可能有用
除了模型重量之外的配置。还可以用于使用自定义覆盖模型加载
方法。在这种情况下，请确保从模型列表中删除已加载的模型。

**参数：**

hook (`Callable`) ：在 [Accelerator.load_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.load_state) `load_checkpoint` 之前调用的函数。

**退货：**

``torch.utils.hooks.RemovableHandle``

一个句柄，可用于通过调用删除添加的钩子
`handle.remove()`
#### register_save_state_pre_hook[[accelerate.Accelerator.register_save_state_pre_hook]][Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3552)

注册一个要在 [Accelerator.save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) 中调用 `save_checkpoint` 之前运行的预挂钩。

该钩子应具有以下签名：

`hook(models: list[torch.nn.Module], weights: list[dict[str, torch.Tensor]], input_dir: str) -> None`

`models`参数是在`accelerator._models`、`weights`下加速器状态下保存的模型
参数是`models`的状态字典，`input_dir`参数是传递的`input_dir`参数
至[Accelerator.load_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.load_state)。

只能与 [Accelerator.register_load_state_pre_hook()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.register_load_state_pre_hook) 一起使用。保存起来很有用
除了模型重量之外的配置。也可用于使用自定义覆盖模型保存
方法。在这种情况下，请确保从权重列表中删除已加载的权重。

**参数：**

hook (`Callable`) ：在 [Accelerator.save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) `save_checkpoint` 之前调用的函数。

**退货：**

``torch.utils.hooks.RemovableHandle``

一个句柄，可用于通过调用删除添加的钩子
`handle.remove()`
#### 保存[[加速.加速器.保存]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3409)

每台机器将传递到磁盘的对象保存一次。代替`torch.save`使用。

注意：
如果 `save_on_each_node` 作为 `ProjectConfiguration` 传入，将为每个节点保存一次对象，
而不是仅在主节点上一次。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> arr = [0, 1, 2, 3]
>>> accelerator.save(arr, "array.pkl")
```

**参数：**

obj (`object`) ：要保存的对象。

f (`str` 或 `os.PathLike`) : `obj` 的内容保存在哪里。safe_serialization (`bool`, *可选*, 默认为`False`) : 是否使用`safetensors`保存`obj`
#### save_model[[accelerate.Accelerator.save_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3439)

保存模型，以便可以使用 load_checkpoint_in_model 重新加载它

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model = ...
>>> accelerator.save_model(model, save_directory)
```

**参数：**

model : (`torch.nn.Module`)：要保存的模型。该模型可以被包裹或展开。

save_directory（`str`或`os.PathLike`）：要保存的目录。如果不存在则将被创建。

max_shard_size（`int`或`str`，*可选*，默认为`"10GB"`）：分片之前检查点的最大大小。检查点分片的大小将小于此大小。如果表示为字符串，则需要是数字后跟单位（如`"5MB"`）。    如果模型的单个权重大于`max_shard_size`，它将位于自己的检查点分片中，该检查点分片将大于`max_shard_size`。   

safe_serialization（`bool`，*可选*，默认为`True`）：是否使用`safetensors`或传统的PyTorch方式（使用`pickle`）保存模型。
#### save_state[[accelerate.Accelerator.save_state]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3584)

将模型、优化器、缩放器、RNG 生成器和注册对象的当前状态保存到文件夹中。如果在启用 `automatic_checkpoint_naming` 的情况下将 `ProjectConfiguration` 传递给 `Accelerator` 对象
那么检查点将被保存到`self.project_dir/checkpoints`。如果当前保存的数量更大
超过`total_limit`，则最旧的保存将被删除。每个检查点都保存在名为
`checkpoint_`。

否则它们只会保存到`output_dir`。

仅当想要在训练期间保存检查点并在同一时间恢复状态时才应使用
环境。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer, lr_scheduler = ...
>>> model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
>>> accelerator.save_state(output_dir="my_checkpoint")
```

**参数：**

output_dir（`str`或`os.PathLike`）：保存所有相关权重和状态的文件夹名称。

safe_serialization（`bool`，*可选*，默认为`True`）：是否使用`safetensors`或传统的PyTorch方式（使用`pickle`）保存模型。

save_model_func_kwargs (`dict`, *可选*) ：用于保存模型的附加关键字参数，可以传递给底层保存函数，例如 DeepSpeed 的 `save_checkpoint` 函数的可选参数。
#### set_trigger[[accelerate.Accelerator.set_trigger]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2852)

将当前进程的内部触发张量设置为 1。后面的检查应该使用这个
将检查所有进程。

注意：
不需要`wait_for_everyone()`

示例：```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> # Assume later in the training script
>>> # `should_do_breakpoint` is a custom function to monitor when to break,
>>> # e.g. when the loss is NaN
>>> if should_do_breakpoint(loss):
...     accelerator.set_trigger()
>>> # Assume later in the training script
>>> if accelerator.check_breakpoint():
...     break
```
####skip_first_batches[[accelerate.Accelerator.skip_first_batches]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4273)

创建一个新的`torch.utils.data.DataLoader`，它将有效地跳过第一个`num_batches`。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
>>> skipped_dataloader = accelerator.skip_first_batches(dataloader, num_batches=2)
>>> # for the first epoch only
>>> for input, target in skipped_dataloader:
...     optimizer.zero_grad()
...     output = model(input)
...     loss = loss_func(output, target)
...     accelerator.backward(loss)
...     optimizer.step()

>>> # subsequent epochs
>>> for input, target in dataloader:
...     optimizer.zero_grad()
...     ...
```

**参数：**

dataloader (`torch.utils.data.DataLoader`) ：要跳过批次的数据加载器。

num_batches (`int`, *可选*, 默认为 0) : 要跳过的批次数
#### split_ Between_processes[[accelerate.Accelerator.split_ Between_processes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L832)

在 `self.num_processes` 之间快速拆分 `input`，然后可以在该流程中使用。做的时候有用
分布式推理，例如使用不同的提示。

请注意，使用 `dict` 时，所有键都需要具有相同数量的元素。

示例：

```python
# Assume there are two processes
from accelerate import Accelerator

accelerator = Accelerator()
with accelerator.split_between_processes(["A", "B", "C"]) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C"]

with accelerator.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C", "C"]
```

**参数：**

输入（`list`、`tuple`、`torch.Tensor` 或 `list`/`tuple`/`torch.Tensor`）：要在进程之间拆分的输入。apply_padding (`bool`, `optional`, 默认为`False`) : 是否通过重复输入的最后一个元素来应用填充，以便所有进程具有相同数量的元素。当尝试在输出上执行诸如 `Accelerator.gather()` 之类的操作或传入比进程少的输入时很有用。如果是这样，请记住随后删除填充的元素。
####trigger_sync_in_backward[[accelerate.Accelerator.trigger_sync_in_backward]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L1180)

在多次前向传递后，在模型的下一次反向传递中触发梯度同步
`Accelerator.no_sync`（仅适用于多GPU场景）。

如果脚本不是以分布式模式启动的，则此上下文管理器不执行任何操作。

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> dataloader, model, optimizer = accelerator.prepare(dataloader, model, optimizer)

>>> with accelerator.no_sync():
...     loss_a = loss_func(model(input_a))  # first forward pass
...     loss_b = loss_func(model(input_b))  # second forward pass
>>> accelerator.backward(loss_a)  # No synchronization across processes, only accumulate gradients
>>> with accelerator.trigger_sync_in_backward(model):
...     accelerator.backward(loss_b)  # Synchronization across all processes
>>> optimizer.step()
>>> optimizer.zero_grad()
```

**参数：**

model (`torch.nn.Module`) ：触发梯度同步的模型。
#### unscale_gradients[[accelerate.Accelerator.unscale_gradients]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L2911)

使用 AMP 取消缩放混合精度训练中的梯度。这在所有其他设置中都是无用的。

可能应该通过 [Accelerator.clip_grad_norm_()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_norm_) 或 [Accelerator.clip_grad_value_()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.clip_grad_value_) 调用

示例：

```python
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model, optimizer = accelerator.prepare(model, optimizer)
>>> outputs = model(inputs)
>>> loss = loss_fn(outputs, labels)
>>> accelerator.backward(loss)
>>> accelerator.unscale_gradients(optimizer=optimizer)
```

**参数：**优化器（`torch.optim.Optimizer`或`list[torch.optim.Optimizer]`，*可选*）：取消缩放梯度的优化器。如果未设置，将取消传递到 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 的所有优化器上的梯度。
#### unwrap_model[[accelerate.Accelerator.unwrap_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3213)

从 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 可能添加的附加层中解开 `model`。保存前有用
模型。

示例：

```python
>>> # Assuming two GPU processes
>>> from torch.nn.parallel import DistributedDataParallel
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> model = accelerator.prepare(MyModel())
>>> print(model.__class__.__name__)
DistributedDataParallel

>>> model = accelerator.unwrap_model(model)
>>> print(model.__class__.__name__)
MyModel
```

**参数：**

model (`torch.nn.Module`) ：要展开的模型。

keep_fp32_wrapper (`bool`, *可选*, 默认为`True`) : 如果添加了混合精度钩子，是否不删除。

keep_torch_compile (`bool`，*可选*，默认为`True`)：如果编译，是否不解开已编译的模型。

**退货：**

``torch.nn.Module``

未包装的模型。
#### verify_device_map[[accelerate.Accelerator.verify_device_map]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L4309)

验证 `model` 尚未准备好使用类似于 `auto` 的设备映射的大模型推理。
#### wait_for_everyone[[accelerate.Accelerator.wait_for_everyone]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py#L3247)

将停止当前进程的执行，直到所有其他进程都到达该点（所以这
当脚本仅在一个进程中运行时什么也没有）。在保存模型之前执行此操作很有用。

示例：

```python
>>> # Assuming two GPU processes
>>> import time
>>> from accelerate import Accelerator

>>> accelerator = Accelerator()
>>> if accelerator.is_main_process:
...     time.sleep(2)
>>> else:
...     print("I'm waiting for the main process to finish its sleep...")
>>> accelerator.wait_for_everyone()
>>> # Should print on every process at the same time
>>> print("Everyone is here")
```## 实用程序[[accelerate.utils.gather_object]]

####加速.utils.gather_object[[accelerate.utils.gather_object]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/operations.py#L505)

从所有设备递归地将对象收集到对象的嵌套列表/元组/字典中。

**参数：**

object（可挑选对象的嵌套列表/元组/字典）：要收集的数据。

**退货：**

与`object`相同的数据结构，所有对象都发送到每个设备。

### 启动器
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/launchers.md