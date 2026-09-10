<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Kwargs 处理程序

以下对象可以传递到主[Accelerator](/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator)来自定义某些PyTorch对象的方式
创建与分布式训练或混合精度相关的。

## AutocastKwargs[[accelerate.AutocastKwargs]]

#### 加速.AutocastKwargs[[加速.AutocastKwargs]]

```python
accelerate.AutocastKwargs(enabled: bool = True, cache_enabled: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L115)

在 [Accelerator](/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义 `torch.autocast` 的行为方式。请参阅
[context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) 的文档以了解更多信息
每个论点的信息。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import AutocastKwargs

kwargs = AutocastKwargs(cache_enabled=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## DistributedDataParallelKwargs[[accelerate.DistributedDataParallelKwargs]]

####加速.DistributedDataParallelKwargs[[accelerate.DistributedDataParallelKwargs]]

```python
accelerate.DistributedDataParallelKwargs(dim: int = 0, broadcast_buffers: bool = True, bucket_cap_mb: int = 25, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False, comm_hook: DDPCommunicationHookType = <DDPCommunicationHookType.NO: 'no'>, comm_wrapper: typing.Literal[<DDPCommunicationHookType.NO: 'no'>, <DDPCommunicationHookType.FP16: 'fp16'>, <DDPCommunicationHookType.BF16: 'bf16'>] = <DDPCommunicationHookType.NO: 'no'>, comm_state_option: dict = <factory>)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L157)

在您的 [Accelerator](/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义模型如何包装在
`torch.nn.parallel.DistributedDataParallel`。请参考这个文档
[wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) 了解更多
每个论点的信息。

`gradient_as_bucket_view` 仅在 PyTorch 1.7.0 及更高版本中可用。

`static_graph` 仅在 PyTorch 1.11.0 及更高版本中可用。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

####加速.utils.FP8RecipeKwargs[[accelerate.utils.FP8RecipeKwargs]]

```python
accelerate.utils.FP8RecipeKwargs(opt_level: typing.Literal['O1', 'O2'] = None, use_autocast_during_eval: typing.Optional[bool] = None, margin: typing.Optional[int] = None, interval: typing.Optional[int] = None, fp8_format: typing.Literal['HYBRID', 'E4M3', 'E5M2'] = None, amax_history_len: typing.Optional[int] = None, amax_compute_algo: typing.Literal['max', 'most_recent'] = None, override_linear_precision: tuple = None, use_mxfp8_block_scaling: typing.Optional[bool] = None, backend: typing.Literal['MSAMP', 'TE'] = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L457)

已弃用。请使用正确的 FP8 配方 kwargs 类之一，例如 `TERecipeKwargs` 或 `MSAMPRecipeKwargs`
相反。## ProfileKwargs[[accelerate.ProfileKwargs]]

####加速.ProfileKwargs[[加速.ProfileKwargs]]

```python
accelerate.ProfileKwargs(activities: typing.Optional[list[typing.Literal['cpu', 'xpu', 'mtia', 'cuda', 'hpu']]] = None, schedule_option: typing.Optional[dict[str, int]] = None, on_trace_ready: typing.Optional[typing.Callable] = None, record_shapes: bool = False, profile_memory: bool = False, with_stack: bool = False, with_flops: bool = False, with_modules: bool = False, output_trace_dir: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L486)

**参数：**

活动（`List[str]`，*可选*，默认为`None`）：要在分析中使用的活动组列表。必须是 `"cpu"`、`"xpu"`、`"mtia"`、“hpu”或 `"cuda"` 之一。

Schedule_option（`Dict[str, int]`，*可选*，默认为`None`）：用于探查器的计划选项。可用的键有 `wait`、`warmup`、`active`、`repeat` 和 `skip_first`。分析器将跳过前 `skip_first` 步骤，然后等待 `wait` 步骤，然后为接下来的 `warmup` 步骤进行预热，然后为接下来的 `active` 步骤进行活动记录，然后从 `wait` 步骤开始重复循环。可选的周期数由`repeat`参数指定，零值意味着周期将继续，直到分析完成。

on_trace_ready (`Callable`，*可选*，默认为 `None`) ：在分析过程中，当 Schedule 返回 `ProfilerAction.RECORD_AND_SAVE` 时，在每一步调用的可调用函数。

record_shapes（`bool`，*可选*，默认为`False`）：保存有关操作员输入形状的信息。

profile_memory (`bool`，*可选*，默认为`False`)：跟踪张量内存分配/释放with_stack (`bool`，*可选*，默认为`False`)：记录操作的源信息（文件和行号）。

with_flops (`bool`, *可选*, 默认为`False`) : 使用公式估计特定算子的FLOPS

with_modules (`bool`, *可选*, 默认为`False`) ：记录op的callstack对应的模块层次结构（包括函数名称）。

output_trace_dir（`str`，*可选*，默认为`None`）：以 Chrome JSON 格式导出收集的跟踪。 Chrome 使用 'chrome://tracing' 查看 json 文件。默认为 None，这意味着分析不存储 json 文件。

在 [Accelerator](/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义分析器的初始化。请参阅
[context manager](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) 的文档
有关每个论点的更多信息。

`torch.profiler` 仅在 PyTorch 1.8.1 及更高版本中可用。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import ProfileKwargs

kwargs = ProfileKwargs(activities=["cpu", "cuda"])
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

#### 构建[[accelerate.ProfileKwargs.build]]

```python
build()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L576)

**返回：** `torch.profiler.profile`

探查器对象。

使用当前配置构建探查器对象。

## GradScalerKwargs[[accelerate.GradScalerKwargs]]

####加速.GradScalerKwargs[[加速.GradScalerKwargs]]

```python
accelerate.GradScalerKwargs(init_scale: float = 65536.0, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000, enabled: bool = True)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L243)在您的 [Accelerator](/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator) 中使用此对象来自定义混合精度的行为，特别是如何
创建使用的`torch.amp.GradScaler`或`torch.cuda.amp.GradScaler`。请参考这个文档
[scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) 了解有关每个参数的更多信息。

`torch.cuda.amp.GradScaler` 仅在 PyTorch 1.5.0 及更高版本中可用，`torch.amp.GradScaler` 是
仅在 PyTorch 2.4.0 及更高版本中可用。

示例：

```python
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

kwargs = GradScalerKwargs(backoff_factor=0.25)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## InitProcessGroupKwargs[[accelerate.InitProcessGroupKwargs]]

####加速.InitProcessGroupKwargs[[加速.InitProcessGroupKwargs]]

```python
accelerate.InitProcessGroupKwargs(backend: typing.Optional[str] = 'nccl', init_method: typing.Optional[str] = None, timeout: typing.Optional[datetime.timedelta] = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L275)

在您的[Accelerator](/docs/accelerate/v1.15.0/en/package_reference/accelerator#accelerate.Accelerator)中使用此对象来自定义分布式进程的初始化。请参考
到这个文档
[method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) 了解更多
每个论点的信息。

注意：如果`timeout`设置为`None`，则默认值将基于`backend`的设置方式。

```python
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## KwargsHandler[[accelerate.utils.KwargsHandler]]

####加速.utils.KwargsHandler[[accelerate.utils.KwargsHandler]]

```python
accelerate.utils.KwargsHandler()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L70)

为数据类实现 `to_kwargs()` 方法的内部 mixin。

#### to_kwargs[[accelerate.utils.KwargsHandler.to_kwargs]]

```python
to_kwargs()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.15.0/src/accelerate/utils/dataclasses.py#L78)

返回一个字典，其中包含与此类的默认值不同的属性。### 有状态类
https://huggingface.co/docs/accelerate/v1.15.0/package_reference/state.md