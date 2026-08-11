<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据加载器、优化器和调度器

Accelerate 用于准备分布式训练对象的内部类
当拨打[prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)时。

## DataLoader 实用程序[[accelerate.data_loader.prepare_data_loader]]

####加速.data_loader.prepare_data_loader[[accelerate.data_loader.prepare_data_loader]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py#L1014)

包装 PyTorch `DataLoader` 以仅为其中一个进程生成批次。

根据传递的 `dataloader` 的 `drop_last` 属性的值，它将停止迭代
在第一批中，该批次太小/不存在于所有进程中或从一开始就使用索引进行循环。

默认情况下不启用具有不同批量大小的`BatchSampler`。要启用此行为，请设置 `even_batches`
等于`False`

**参数：**

dataloader (`torch.utils.data.dataloader.DataLoader`) ：跨多个设备拆分的数据加载器。

device (`torch.device`) ：返回的`DataLoader`的目标设备。

num_processes (`int`, *可选*) ：同时运行的进程数。将默认为[PartialState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.PartialState)给出的值。

process_index (`int`, *可选*) ：当前进程的索引。将默认为[PartialState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.PartialState)给出的值。split_batches（`bool`，*可选*，默认为`False`）：生成的`DataLoader`是否应该跨设备分割原始数据加载器的批次或生成完整批次（在这种情况下，它将生成从第`process_index`开始的批次，并在每次迭代时前进`num_processes`批次）。  另一种观察方法是，如果此选项设置为 `True`，则观察到的批量大小将与初始 `dataloader` 相同，否则初始 `dataloader` 的批量大小乘以 `num_processes`。  将此选项设置为`True`要求`dataloader`的批量大小是`batch_size`的整数倍。

put_on_device (`bool`，*可选*，默认为`False`)：是否将批次放在`device`上（仅当批次是嵌套列表、元组或张量字典时才有效）。rng_types（`str`或[RNGType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.RNGType)列表）：在每次迭代开始时同步的随机数生成器列表。应该是以下一项或多项： - `"torch"`：基本 Torch 随机数生成器 - `"cuda"`：CUDA 随机数生成器（仅限 GPU） - `"xla"`：XLA 随机数生成器（仅限 TPU） - `"generator"`：采样器（或批量采样器，如果数据加载器中没有采样器）或可迭代数据集（如果存在）（如果基础数据集属于该类型）。 

dispatch_batches (`bool`, *可选*) ：如果设置为`True`，则准备好的数据加载器仅在主进程上迭代，然后将批次拆分并广播到每个进程。当基础数据集是 `IterableDataset` 时，将默认为 `True`，否则为 `False`。

Even_batches（`bool`，*可选*，默认为`True`）：如果设置为`True`，在所有进程的总批次大小不能完全划分数据集的情况下，数据集开头的样本将被复制，以便批次可以在所有工作人员之间平均分配。

slice_fn_for_dispatch (`Callable`, *可选*`) : If passed, this function will be used to slice tensors across `num_processes`. Will default to [slice_tensors()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.slice_tensors). This argument is used only when `dispatch_batches` is set to `True`，否则将被忽略。use_seedable_sampler（`bool`，*可选*，默认为`False`）：是否使用`SeedableRandomSampler`代替`RandomSampler`以获得更好的重现性。由于不同的洗牌算法，代价可能是不同的性能，但确保结果“完全”相同。每次 `self.set_epoch` 时应与 `set_seed()` 配对

data_seed (`int`，*可选*，默认为`None`)：使用`use_seedable_sampler`时用于底层生成器的种子。如果`None`，生成器将使用来自torch的当前默认种子。

non_blocking（`bool`，*可选*，默认为`False`）：如果设置为`True`，数据加载器将利用非阻塞主机到设备传输。如果数据加载器将`pin_memory`设置为`True`，这将有助于增加数据传输和计算之间的重叠。

use_stateful_dataloader (`bool`，*可选*，默认为`False`) : "如果设置为 true，则 Accelerator 准备的 dataloader 将由 " "[torchdata.StatefulDataLoader](https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader) 支持。这需要安装支持 StatefulDataLoader 的 `torchdata` 0.8.0 或更高版本。"

torch_device_mesh (`torch.distributed.DeviceMesh`，*可选*，默认为`None`)：PyTorch 设备网格。

**退货：**

``torch.utils.data.dataloader.DataLoader``

一个新的数据加载器将产生部分批次####加速.skip_first_batches[[accelerate.skip_first_batches]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py#L1393)

创建一个`torch.utils.data.DataLoader`，它将有效地跳过第一个`num_batches`。不应使用，如果
原始数据加载器是`StatefulDataLoader`。

## BatchSamplerShard[[accelerate.data_loader.BatchSamplerShard]]

####加速.data_loader.BatchSamplerShard[[accelerate.data_loader.BatchSamplerShard]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py#L110)

包装 PyTorch `BatchSampler` 以仅为其中一个进程生成批次。该类的实例将
总是产生多个批次，其数量是 `num_processes` 的整数倍，并且全部具有相同的大小。
根据传递的批量采样器的`drop_last`属性的值，它将停止迭代
在第一批中，该批次太小/不存在于所有进程中或从一开始就使用索引进行循环。

默认情况下不启用具有不同批量大小的`BatchSampler`。要启用此行为，请设置 `even_batches`
等于`False`

**参数：**

batch_sampler (`torch.utils.data.sampler.BatchSampler`) ：批量采样器分为几个分片。

num_processes (`int`，*可选*，默认为 1) ：同时运行的进程数。

process_index (`int`, *可选*, 默认为 0) : 当前进程的索引。split_batches（`bool`，*可选*，默认为`False`）：是否应通过拆分批次以在每个进程上提供一部分来创建分片，或者通过在每个进程上生成不同的完整批次来创建分片。  在采样器为 `[[0, 1, 2, 3], [4, 5, 6, 7]]` 的两个进程上，这将导致： - 如果此参数设置为 `False`，进程 0 上的采样器将产生 `[0, 1, 2, 3]`，进程 1 上的采样器将产生 `[4, 5, 6, 7]`。 - 如果此参数设置为 `True`，则进程 0 上的采样器将产生 `[0, 1]`，然后产生 `[4, 5]`，进程 1 上的采样器将产生`[2, 3]`，然后产生 `[6, 7]`。

Even_batches (`bool`，*可选*，默认为`True`) ：当样本数不是（原始批量大小/进程数）的整数倍时是否在采样器的开头循环。

## IterableDatasetShard[[accelerate.data_loader.IterableDatasetShard]]

####加速.data_loader.IterableDatasetShard[[accelerate.data_loader.IterableDatasetShard]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py#L274)包装 PyTorch `IterableDataset` 以仅为其中一个进程生成样本。该类的实例将
总是产生一些样本，其数量是实际批量大小的整数倍（取决于
`split_batches`，这是`batch_size` 或`batch_size x num_processes`）。取决于值
传递了批次采样器的`drop_last`属性，它将在第一个批次处停止迭代
太小或从头开始循环索引。

**参数：**

数据集 (`torch.utils.data.dataset.IterableDataset`) ：批量采样器拆分为多个分片。

batch_size (`int`，*可选*，默认为 1) ：每个分片的批次大小（如果`split_batches=False`）或批次的大小（如果`split_batches=True`）。

drop_last (`bool`，*可选*，默认为`False`) ：是否删除最后一个未完成的批次或使用从头开始的样本来完成最后的批次。

num_processes (`int`，*可选*，默认为 1) ：同时运行的进程数。

process_index (`int`, *可选*, 默认为 0) : 当前进程的索引。split_batches（`bool`，*可选*，默认为`False`）：是否应通过拆分批次以在每个进程上提供一部分来创建分片，或者通过在每个进程上生成不同的完整批次来创建分片。  在可迭代数据集产生`[0, 1, 2, 3, 4, 5, 6, 7]`的两个进程上，这将导致： - 如果此参数设置为`False`，进程 0 上的分片将产生`[0, 1, 2, 3]`，进程 1 上的分片将产生`[4, 5, 6, 7]`。 - 如果此参数设置为 `True`，则进程 0 上的分片将产生 `[0, 1, 4, 5]`，进程 1 上的采样器将产生 `[2, 3, 6, 7]`。

## DataLoaderShard[[accelerate.data_loader.DataLoaderShard]]

####加速.data_loader.DataLoaderShard[[accelerate.data_loader.DataLoaderShard]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py#L510)

`DataLoaderAdapter` 的子类，将处理设备放置和当前分布式设置。

**可用属性：**

- **total_batch_size** (`int`) -- 所有进程中数据加载器的总批次大小。
  `split_batches=True`时等于原始batch size；否则为原始批量大小 * 总计
  进程数

- **total_dataset_length** (`int`) -- 所有进程的内部数据集的总长度。

**参数：**

dataset (`torch.utils.data.dataset.Dataset`) ：用于构建此数据加载器的数据集。device (`torch.device`, *可选*) ：如果通过，则放置所有批次的设备。

rng_types（`str`或[RNGType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.RNGType)列表）：在每次迭代开始时同步的随机数生成器列表。应该是以下一项或多项： - `"torch"`：基本 Torch 随机数生成器 - `"cuda"`：CUDA 随机数生成器（仅限 GPU） - `"xla"`：XLA 随机数生成器（仅限 TPU） - `"generator"`：可选 `torch.Generator`

synchronized_generator (`torch.Generator`, *可选*) ：一个随机数生成器，用于在进程之间保持同步。

Skip_batches (`int`，*可选*，默认为 0) ：开始时要跳过的批次数。

use_stateful_dataloader (`bool`，*可选*，默认为`False`)：是否让此类从`torchdata`适应`StatefulDataLoader`，而不是常规的`DataLoader`。

- ****kwargs** （附加关键字参数，*可选*）：传递给常规 `DataLoader` 初始化的所有其他关键字参数。

## DataLoaderDispatcher[[accelerate.data_loader.DataLoaderDispatcher]]

####加速.data_loader.DataLoaderDispatcher[[accelerate.data_loader.DataLoaderDispatcher]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/data_loader.py#L722)`DataLoaderAdapter` 的子类，仅在进程 0 上迭代和预处理，然后在每个进程上分派
他们属于这批货。

**可用属性：**

- **total_batch_size** (`int`) -- 所有进程中数据加载器的总批次大小。
  `split_batches=True`时等于原始batch size；否则为原始批量大小 * 总计
  进程数

- **total_dataset_length** (`int`) -- 所有进程的内部数据集的总长度。

**参数：**

split_batches (`bool`，*可选*，默认为`False`)：生成的`DataLoader`是否应该跨设备分割原始数据加载器的批次或生成完整批次（在这种情况下，它将生成从第`process_index`开始的批次，并在每次迭代时前进`num_processes`批次）。另一种观察方法是，如果此选项设置为 `True`，则观察到的批量大小将与初始 `dataloader` 相同，否则初始 `dataloader` 的批量大小乘以 `num_processes`。将此选项设置为`True`要求`dataloader`的批量大小是`batch_size`的整数倍。

Skip_batches (`int`，*可选*，默认为 0) ：迭代开始时要跳过的批次数。use_stateful_dataloader (`bool`，*可选*，默认为`False`)：是否让此类从`torchdata`改编`StatefulDataLoader`，而不是常规的`DataLoader`。

## AcceleratedOptimizer[[accelerate.optimizer.AcceleratedOptimizer]]

####加速.optimizer.AcceleratedOptimizer[[accelerate.optimizer.AcceleratedOptimizer]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/optimizer.py#L38)

火炬优化器周围的内部包装。

如果执行梯度时需要同步梯度，则有条件地执行`step`和`zero_grad`
积累。

evalaccelerate.optimizer.AcceleratedOptimizer.eval https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/optimizer.py#L138[]

将优化器设置为“eval”模式。对于像 `schedule_free` 这样的优化器很有用

**参数：**

优化器 (`torch.optim.optimizer.Optimizer`) ：要包装的优化器。

device_placement（`bool`，*可选*，默认为`True`）：优化器是否应该处理设备放置。如果是，它将把`optimizer`的状态字典放在正确的设备上。

缩放器（`torch.amp.GradScaler`或`torch.cuda.amp.GradScaler`，*可选*）：如果使用混合精度进行训练，则在步骤函数中使用的缩放器。
#### 训练[[accelerate.optimizer.AcceleratedOptimizer.train]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/optimizer.py#L124)将优化器设置为“训练”模式。对于像 `schedule_free` 这样的优化器很有用

## AcceleratedScheduler[[accelerate.scheduler.AcceleratedScheduler]]

####加速.scheduler.AcceleratedScheduler[[accelerate.scheduler.AcceleratedScheduler]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/scheduler.py#L25)

学习率调度程序的包装器，仅当优化器有训练步骤时才会执行。有用的
避免当梯度溢出并且没有训练步骤时（在混合中）使调度程序步骤太快
精准训练）

当执行梯度累积时，调度器长度不应相应改变，Accelerate将始终
步骤调度程序来解释它。

**参数：**

调度程序 (`torch.optim.lr_scheduler._LRScheduler`) ：要包装的调度程序。

优化器（一个或一系列`torch.optim.Optimizer`）：使用的优化器。

step_with_optimizer (`bool`，*可选*，默认为`True`)：调度程序是否应该在每个优化器步骤中步进。split_batches (`bool`，*可选*，默认为`False`)：数据加载器是否在不同进程之间拆分一批（因此无论进程数量如何，批大小都是相同的）或在每个进程上创建批（因此批大小是原始批大小乘以进程数）。

### Megatron-LM 实用程序
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/megatron_lm.md