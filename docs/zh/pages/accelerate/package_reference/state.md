<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 有状态的类

以下是 [singleton class](https://en.wikipedia.org/wiki/Singleton_pattern) 的变体，从某种意义上说，所有
实例共享相同的状态，该状态在第一次实例化时初始化。

这些类是不可变的，并存储有关某些配置或 
州。

## PartialState[[accelerate.PartialState]]

####加速.PartialState[[加速.PartialState]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L123)

Singleton 类，包含有关当前训练环境和功能的信息，以帮助完成流程
控制。设计用于仅需要过程控制和设备执行状态的情况。 *不需要*需要
从`Accelerator`初始化。

**可用属性：**- **设备** (`torch.device`) -- 要使用的设备。
- **distributed_type** ([DistributedType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.DistributedType)) -- 当前分布式环境的类型
  在使用中。
- **local_process_index** (`int`) -- 当前服务器上当前进程的索引。
- **mixed_ precision** (`str`) -- 当前脚本是否将使用混合精度，如果是，类型
  正在执行的混合精度。 （从“否”、“fp16”、“bf16”或“fp8”中选择）。
- **num_processes** (`int`) -- 当前并行启动的进程数。
- **process_index** (`int`) -- 当前进程的索引。
- **is_last_process** (`bool`) -- 当前进程是否是最后一个进程。
- **is_main_process** (`bool`) -- 当前进程是否为主进程。
- **is_local_main_process** (`bool`) -- 当前进程是否为本节点的主进程。
- **debug** (`bool`) -- 当前脚本是否在调试模式下运行。

示例：
```python
from accelerate.utils import InitProcessGroupKwargs

# To include `InitProcessGroupKwargs`, init then call `.to_kwargs()`
kwargs = InitProcessGroupKwargs(...).to_kwargs()
state = PartialState(**kwargs)
```

destroy_process_groupaccelerate.PartialState.destroy_process_grouphttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L845[{"name": "group", "val": " = None"}]销毁进程组。如果未指定，则销毁默认进程组。

**参数：**

cpu (`bool`, *可选*) : 是否强制脚本在CPU上执行。如果设置为 `True` 将忽略任何可用的加速器并强制在 CPU 上执行。

kwargs（附加关键字参数，*可选*）：传递给相关 `init_process_group` 函数的附加关键字参数。有效的`kwargs`可以在[utils.InitProcessGroupKwargs](/docs/accelerate/v1.14.0/en/package_reference/kwargs#accelerate.InitProcessGroupKwargs)中找到。详细用法请参见示例部分。
#### local_main_process_first[[accelerate.PartialState.local_main_process_first]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L534)

让本地主进程进入 with 块。

主进程退出后，其他进程将进入with块。

示例：

```python
>>> from accelerate.state import PartialState

>>> state = PartialState()
>>> with state.local_main_process_first():
...     # This will be printed first by local process 0 then in a seemingly
...     # random order by the other processes.
...     print(f"This will be printed by process {state.local_process_index}")
```
#### main_process_first[[accelerate.PartialState.main_process_first]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L513)

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
#### on_last_process[[accelerate.PartialState.on_last_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L616)

装饰器仅在最后一个进程上运行装饰函数。

示例：
```python
# Assume we have 4 processes.
from accelerate.state import PartialState

state = PartialState()

@state.on_last_process
def print_something():
    print(f"Printed on process {state.process_index}")

print_something()
"Printed on process 3"
```

**参数：**function (`Callable`) ：要装饰的函数。
#### on_local_main_process[[accelerate.PartialState.on_local_main_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L585)

装饰器仅在本地主进程上运行装饰函数。

示例：
```python
# Assume we have 2 servers with 4 processes each.
from accelerate.state import PartialState

state = PartialState()

@state.on_local_main_process
def print_something():
    print("This will be printed by process 0 only on each server.")

print_something()
# On server 1:
"This will be printed by process 0 only"
# On server 2:
"This will be printed by process 0 only"
```

**参数：**

function (`Callable`) ：要装饰的函数。
#### on_local_process[[accelerate.PartialState.on_local_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L677)

装饰器仅在当前节点上具有给定索引的进程上运行装饰函数。

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

local_process_index (`int`, *可选*) ：运行该函数的本地进程的索引。
#### on_main_process[[accelerate.PartialState.on_main_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L555)

装饰器只在主进程上运行装饰函数。

示例：

```python
>>> from accelerate.state import PartialState

>>> state = PartialState()

>>> @state.on_main_process
... def print_something():
...     print("This will be printed by process 0 only.")

>>> print_something()
"This will be printed by process 0 only"
```

**参数：**

function (`Callable`) ：要装饰的函数。
#### on_process[[accelerate.PartialState.on_process]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L644)

装饰器仅在具有给定索引的进程上运行装饰函数。

示例：
```python
# Assume we have 4 processes.
from accelerate.state import PartialState

state = PartialState()

@state.on_process(process_index=2)
def print_something():
    print(f"Printed on process {state.process_index}")

print_something()
"Printed on process 2"
```

**参数：**

function (`Callable`, `optional`) ：要装饰的函数。process_index (`int`, `optional`) ：运行函数的进程的索引。
#### set_device[[accelerate.PartialState.set_device]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L819)

将 `self.device` 中的设备设置为当前分布式环境。
#### split_ Between_processes[[accelerate.PartialState.split_ Between_processes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L425)

在 `self.num_processes` 之间快速拆分 `input`，然后可以在该流程中使用。做的时候有用
分布式推理，例如使用不同的提示。

请注意，使用 `dict` 时，所有键都需要具有相同数量的元素。

示例：

```python
# Assume there are two processes
from accelerate import PartialState

state = PartialState()
with state.split_between_processes(["A", "B", "C"]) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C"]

with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C", "C"]
```

**参数：**

输入（`list`、`tuple`、`torch.Tensor`、`dict` of `list`/`tuple`/`torch.Tensor`或`datasets.Dataset`）：要在进程之间拆分的输入。

apply_padding (`bool`, `optional`, 默认为`False`) : 是否通过重复输入的最后一个元素来应用填充，以便所有进程具有相同数量的元素。当尝试在输出上执行诸如 `gather()` 之类的操作或传入比进程少的输入时很有用。如果是这样，请记住随后删除填充的元素。
#### wait_for_everyone[[accelerate.PartialState.wait_for_everyone]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L377)将停止当前进程的执行，直到所有其他进程都到达该点（所以这
当脚本仅在一个进程中运行时什么也没有）。在保存模型之前执行此操作很有用。

示例：

```python
>>> # Assuming two GPU processes
>>> import time
>>> from accelerate.state import PartialState

>>> state = PartialState()
>>> if state.is_main_process:
...     time.sleep(2)
>>> else:
...     print("I'm waiting for the main process to finish its sleep...")
>>> state.wait_for_everyone()
>>> # Should print on every process at the same time
>>> print("Everyone is here")
```

## AcceleratorState[[accelerate.state.AcceleratorState]]

####加速.状态.AcceleratorState[[加速.状态.AcceleratorState]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L868)

Singleton 类，包含有关当前训练环境的信息。

**可用属性：**

- **设备** (`torch.device`) -- 要使用的设备。
- **distributed_type** ([DistributedType](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.DistributedType)) -- 当前分布式环境的类型
  在使用中。
- **parallelism_config** (`ParallelismConfig`) -- 并行度配置
  目前的训练环境。这用于配置分布式训练环境。
- **initialized** (`bool`) -- `AcceleratorState`是否已从`Accelerator`初始化。
- **local_process_index** (`int`) -- 当前服务器上当前进程的索引。
- **mixed_ precision** (`str`) -- 当前脚本是否使用混合精度，如果是，类型
  正在执行的混合精度。 （从“否”、“fp16”、“bf16”或“fp8”中选择）。- **num_processes** (`int`) -- 当前并行启动的进程数。
- **process_index** (`int`) -- 当前进程的索引。
- **is_last_process** (`bool`) -- 当前进程是否是最后一个进程。
- **is_main_process** (`bool`) -- 当前进程是否为主进程。
- **is_local_main_process** (`bool`) -- 当前进程是否为本节点的主进程。
- **debug** (`bool`) -- 当前脚本是否在调试模式下运行。

destroy_process_groupaccelerate.state.AcceleratorState.destroy_process_grouphttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1081[{"name": "group", "val": " = None"}]

销毁进程组。如果未指定，则销毁默认进程组。

如果 `self.fork_launched` 是 `True` 并且 `group` 是 `None`，则什么也不会发生。
#### get_deepspeed_plugin[[accelerate.state.AcceleratorState.get_deepspeed_plugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1198)

返回具有给定plugin_key的DeepSpeedPlugin。
#### local_main_process_first[[accelerate.state.AcceleratorState.local_main_process_first]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1174)

让本地主进程进入 with 块。主进程退出后，其他进程将进入with块。
#### main_process_first[[accelerate.state.AcceleratorState.main_process_first]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1164)

让主进程首先进入 with 块。

主进程退出后，其他进程将进入with块。
#### select_deepspeed_plugin[[accelerate.state.AcceleratorState.select_deepspeed_plugin]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1205)

使用给定的 `name` 激活 DeepSpeedPlugin，并将禁用所有其他插件。
#### split_ Between_processes[[accelerate.state.AcceleratorState.split_ Between_processes]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1122)

在 `self.num_processes` 之间快速拆分 `input`，然后可以在该流程中使用。做的时候有用
分布式推理，例如使用不同的提示。

请注意，使用 `dict` 时，所有键都需要具有相同数量的元素。

示例：

```python
# Assume there are two processes
from accelerate.state import AcceleratorState

state = AcceleratorState()
with state.split_between_processes(["A", "B", "C"]) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C"]

with state.split_between_processes(["A", "B", "C"], apply_padding=True) as inputs:
    print(inputs)
# Process 0
["A", "B"]
# Process 1
["C", "C"]
```

**参数：**

输入（`list`、`tuple`、`torch.Tensor` 或 `list`/`tuple`/`torch.Tensor` 的 `dict`）：要在进程之间拆分的输入。apply_padding (`bool`, `optional`, 默认为`False`) : 是否通过重复输入的最后一个元素来应用填充，以便所有进程具有相同数量的元素。当尝试在输出上执行诸如 `gather()` 之类的操作或传入比进程少的输入时很有用。如果是这样，请记住随后删除填充的元素。

## GradientState[[accelerate.state.GradientState]]

####加速.状态.GradientState[[加速.状态.GradientState]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/state.py#L1231)

Singleton 类，具有与梯度累积的梯度同步相关的信息

**可用属性：**- **end_of_dataloader** (`bool`) -- 是否已经到达当前数据加载器的末尾
- **剩余** (`int`) -- 通过填充数据加载器添加的额外样本数
- **sync_gradients** (`bool`) -- 渐变是否应在所有设备上同步
- **active_dataloader** (`Optional[DataLoader]`) -- 当前正在迭代的数据加载器
- **dataloader_references** (`List[Optional[DataLoader]]`) -- 对数据加载器的引用列表
  被迭代
- **num_steps** (`int`) -- 累计的步数
- **adjust_scheduler** (`bool`) -- 是否应该调整调度程序以考虑梯度
  积累
- **sync_with_dataloader** (`bool`) -- 渐变是否应在数据加载器末尾同步
  迭代和总步数重置
- **is_xla_gradients_synced** (`bool`) -- XLA 渐变是否已同步。已初始化
  为假。一旦在优化器步骤之前减少了梯度，该标志就会设置为 true。随后，
  每执行一步后，该标志都会重置为 false。 FSDP 将始终同步梯度，因此
  is_xla_gradients_synced 始终为 true。### 加速器
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/accelerator.md