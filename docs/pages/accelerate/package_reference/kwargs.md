# Kwargs handlers

The following objects can be passed to the main [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize how some PyTorch objects
related to distributed training or mixed precision are created.

## AutocastKwargs[[accelerate.AutocastKwargs]]

#### accelerate.AutocastKwargs[[accelerate.AutocastKwargs]]

```python
accelerate.AutocastKwargs(enabled: bool = True, cache_enabled: bool | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L104)

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize how `torch.autocast` behaves. Please refer to the
documentation of this [context manager](https://pytorch.org/docs/stable/amp.html#torch.autocast) for more
information on each argument.

Example:

```python
from accelerate import Accelerator
from accelerate.utils import AutocastKwargs

kwargs = AutocastKwargs(cache_enabled=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## DistributedDataParallelKwargs[[accelerate.DistributedDataParallelKwargs]]

#### accelerate.DistributedDataParallelKwargs[[accelerate.DistributedDataParallelKwargs]]

```python
accelerate.DistributedDataParallelKwargs(dim: int = 0, broadcast_buffers: bool = True, bucket_cap_mb: int = 25, find_unused_parameters: bool = False, check_reduction: bool = False, gradient_as_bucket_view: bool = False, static_graph: bool = False, comm_hook: DDPCommunicationHookType = <DDPCommunicationHookType.NO: 'no'>, comm_wrapper: typing.Literal[<DDPCommunicationHookType.NO: 'no'>, <DDPCommunicationHookType.FP16: 'fp16'>, <DDPCommunicationHookType.BF16: 'bf16'>] = <DDPCommunicationHookType.NO: 'no'>, comm_state_option: dict = <factory>)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L146)

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize how your model is wrapped in a
`torch.nn.parallel.DistributedDataParallel`. Please refer to the documentation of this
[wrapper](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) for more
information on each argument.

Example:

```python
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## ProfileKwargs[[accelerate.ProfileKwargs]]

#### accelerate.ProfileKwargs[[accelerate.ProfileKwargs]]

```python
accelerate.ProfileKwargs(activities: list[typing.Literal['cpu', 'xpu', 'mtia', 'cuda', 'hpu']] | None = None, schedule_option: dict[str, int] | None = None, on_trace_ready: typing.Callable | None = None, record_shapes: bool = False, profile_memory: bool = False, with_stack: bool = False, with_flops: bool = False, with_modules: bool = False, output_trace_dir: str | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L416)

**Parameters:**

activities (`List[str]`, *optional*, default to `None`) : The list of activity groups to use in profiling. Must be one of `"cpu"`, `"xpu"`, `"mtia"`, "hpu" or `"cuda"`.

schedule_option (`Dict[str, int]`, *optional*, default to `None`) : The schedule option to use for the profiler. Available keys are `wait`, `warmup`, `active`, `repeat` and `skip_first`. The profiler will skip the first `skip_first` steps, then wait for `wait` steps, then do the warmup for the next `warmup` steps, then do the active recording for the next `active` steps and then repeat the cycle starting with `wait` steps. The optional number of cycles is specified with the `repeat` parameter, the zero value means that the cycles will continue until the profiling is finished.

on_trace_ready (`Callable`, *optional*, default to `None`) : Callable that is called at each step when schedule returns `ProfilerAction.RECORD_AND_SAVE` during the profiling.

record_shapes (`bool`, *optional*, default to `False`) : Save information about operator’s input shapes.

profile_memory (`bool`, *optional*, default to `False`) : Track tensor memory allocation/deallocation

with_stack (`bool`, *optional*, default to `False`) : Record source information (file and line number) for the ops.

with_flops (`bool`, *optional*, default to `False`) : Use formula to estimate the FLOPS of specific operators

with_modules (`bool`, *optional*, default to `False`) : Record module hierarchy (including function names) corresponding to the callstack of the op.

output_trace_dir (`str`, *optional*, default to `None`) : Exports the collected trace in Chrome JSON format. Chrome use 'chrome://tracing' view json file. Defaults to None, which means profiling does not store json files.

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize the initialization of the profiler. Please refer to the
documentation of this [context manager](https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for
more information on each argument.

Example:

```python
from accelerate import Accelerator
from accelerate.utils import ProfileKwargs

kwargs = ProfileKwargs(activities=["cpu", "cuda"])
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

#### build[[accelerate.ProfileKwargs.build]]

```python
build()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L498)

**Returns:** `torch.profiler.profile`

The profiler object.

Build a profiler object with the current configuration.

## GradScalerKwargs[[accelerate.GradScalerKwargs]]

#### accelerate.GradScalerKwargs[[accelerate.GradScalerKwargs]]

```python
accelerate.GradScalerKwargs(init_scale: float = 65536.0, growth_factor: float = 2.0, backoff_factor: float = 0.5, growth_interval: int = 2000, enabled: bool = True)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L224)

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize the behavior of mixed precision, specifically how the
`torch.amp.GradScaler` or `torch.cuda.amp.GradScaler` used is created. Please refer to the documentation of this
[scaler](https://pytorch.org/docs/stable/amp.html?highlight=gradscaler) for more information on each argument.

Example:

```python
from accelerate import Accelerator
from accelerate.utils import GradScalerKwargs

kwargs = GradScalerKwargs(backoff_factor=0.25)
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## InitProcessGroupKwargs[[accelerate.InitProcessGroupKwargs]]

#### accelerate.InitProcessGroupKwargs[[accelerate.InitProcessGroupKwargs]]

```python
accelerate.InitProcessGroupKwargs(backend: str | None = 'nccl', init_method: str | None = None, timeout: datetime.timedelta | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L249)

Use this object in your [Accelerator](/docs/accelerate/v1.16.0/en/package_reference/accelerator#accelerate.Accelerator) to customize the initialization of the distributed processes. Please refer
to the documentation of this
[method](https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group) for more
information on each argument.

Note: If `timeout` is set to `None`, the default will be based upon how `backend` is set.

```python
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=800))
accelerator = Accelerator(kwargs_handlers=[kwargs])
```

## KwargsHandler[[accelerate.utils.KwargsHandler]]

#### accelerate.utils.KwargsHandler[[accelerate.utils.KwargsHandler]]

```python
accelerate.utils.KwargsHandler()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L59)

Internal mixin that implements a `to_kwargs()` method for a dataclass.

#### to_kwargs[[accelerate.utils.KwargsHandler.to_kwargs]]

```python
to_kwargs()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/utils/dataclasses.py#L67)

Returns a dictionary containing the attributes with values different from the default of this class.

### Utility functions and classes
https://huggingface.co/docs/accelerate/v1.16.0/package_reference/utilities.md
