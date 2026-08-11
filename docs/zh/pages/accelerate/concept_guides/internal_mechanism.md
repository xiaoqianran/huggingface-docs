<!-- huggingface-docs: machine-translated zh-CN from English source -->

#加速内部机制

在内部，Accelerate 的工作原理是首先分析启动脚本的环境，以确定是哪个脚本
使用一种分布式设置，有多少个不同的进程以及当前脚本位于哪一个进程中。全部
该信息存储在`~AcceleratorState`中。

此类在您第一次实例化 [~Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 以及执行任何操作时初始化
您的分布式设置需要的特定初始化。然后它的状态被所有实例唯一共享
[AcceleratorState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.state.AcceleratorState)。 （同样的事情也可以用[PartialState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.PartialState)来完成，它继承了一个更准系统的版本）

然后，当调用[prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare)时，库：

- 将您的模型包装在适合分布式设置的容器中，
- 将优化器包装在 [AcceleratedOptimizer](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.optimizer.AcceleratedOptimizer) 中，
- 将您的调度程序包装在 [AcceleratedScheduler](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.scheduler.AcceleratedScheduler) 中
- 在 [DataLoaderShard](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.data_loader.DataLoaderShard) 或 [DataLoaderDispatcher](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.data_loader.DataLoaderDispatcher) 中创建数据加载器的新版本虽然模型、优化器和调度器只是放入简单的包装器中，但数据加载器是重新创建的。这主要是
因为 PyTorch 不允许用户在创建数据加载器后更改它的 `batch_sampler` 并且
库通过更改 `batch_sampler` 来生成其他所有数据，从而处理进程之间的数据分片
`num_processes` 批次（如果启用）。

[DataLoaderShard](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.data_loader.DataLoaderShard) 子类 `DataLoader` 添加以下功能：

- 它在每次新迭代时同步所有进程的适当随机数生成器，以确保任何
  随机化（如洗牌）在进程中以完全相同的方式完成。
- 在生成批次之前，它会将批次放在适当的设备上（除非您选择退出）
  `device_placement=True`）。
  
[DataLoaderDispatcher](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.data_loader.DataLoaderDispatcher) 子类与 [DataLoaderShard](/docs/accelerate/v1.14.0/en/package_reference/torch_wrappers#accelerate.data_loader.DataLoaderShard) 的不同之处在于，当迭代 `DataLoader` 时，数据全部从进程 0 开始，*然后*分割并发送到每个进程，而不是在数据集级别发生。

随机数生成器同步将默认同步：- PyTorch >= 1.6 的给定采样器的 `generator` 属性（如 PyTorch `RandomSampler`）
- PyTorch中的主要随机数生成器<=1.5.1

You can choose which random number generator(s) to synchronize with the ⟦T9⟧ argument of the main
⟦T28⟧. In PyTorch >= 1.6，建议依赖本地`generator`来避免
在所有进程的主随机数生成器中设置相同的种子。

    主火炬（或 CUDA 或 XLA）随机数生成器的同步将影响任何其他潜在的随机数
    数据集中可能存在的工件（例如随机数据增强），从某种意义上说，所有进程都会获得
    来自火炬随机模块的相同随机数（因此如果是，将应用相同的随机数据增强
    由火炬控制）。

    自定义采样器、批量采样器或可迭代数据集的随机化部分应使用本地完成
    `torch.Generator` 对象（在 PyTorch >= 1.6 中），请参阅传统的 `RandomSampler` 作为示例。

如果您安装了[⟦T13⟧](https://github.com/pytorch/data/tree/main)，并且已将`use_stateful_dataloader=True`传递给[DataLoaderConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.DataLoaderConfiguration)，则这些类将直接继承于`StatefulDataLoader`，并维护一个`state_dict`。

有关内部结构的更多详细信息，请参阅[Internals page](../package_reference/torch_wrappers)。

### FSDP 与 DeepSpeed
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/fsdp_and_deepspeed.md