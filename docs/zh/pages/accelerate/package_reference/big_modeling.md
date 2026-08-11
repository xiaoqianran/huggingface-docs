<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用大型模型

## 调度和卸载

### init_empty_weights[[accelerate.init_empty_weights]]

####加速.init_empty_weights[[accelerate.init_empty_weights]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L61)

一个上下文管理器，在该管理器下模型使用元设备上的所有参数进行初始化，从而创建一个
空模型。当刚刚初始化模型会耗尽可用 RAM 时很有用。

示例：

```python
import torch.nn as nn
from accelerate import init_empty_weights

# Initialize a model with 100 billions parameters in no time and without using any RAM.
with init_empty_weights():
    tst = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
```

在此上下文管理器下创建的任何模型都没有权重。因此你不能做类似的事情
`model.to(some_device)` 有了它。要在空模型中加载权重，请参阅[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)。
确保覆盖[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)的默认device_map参数，否则无法调度
叫。

**参数：**

include_buffers (`bool`, *可选*) ：初始化时是否将所有缓冲区也放入元设备上。

### cpu_offload[[accelerate.cpu_offload]]

####加速.cpu_offload[[加速.cpu_offload]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L179)激活模型的完整 CPU 卸载。结果，模型的所有参数都将被卸载，并且只有一个参数
将保留模型的状态字典副本。在前向传递过程中，将从中提取参数
状态字典并放在需要时传递的执行设备上，然后再次卸载。

**参数：**

model (`torch.nn.Module`) ：要卸载的模型。

execution_device (`torch.device`, *可选*) ：将执行模型前向传递的设备（应该是 GPU）。将默认为模型第一个参数设备。

offload_buffers (`bool`，*可选*，默认为`False`)：是否使用模型参数卸载缓冲区。

state_dict (`Dict[str, torch.Tensor]`, *可选*) ：将保存在 CPU 上的模型的状态字典。

preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。### cpu_offload_with_hook[[accelerate.cpu_offload_with_hook]]

####加速.cpu_offload_with_hook[[accelerate.cpu_offload_with_hook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L225)

将模型卸载到 CPU 上，并在执行时将其放回执行设备。与的区别
[cpu_offload()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.cpu_offload) 是模型在转发后保留在执行设备上，仅在以下情况下再次卸载：
调用返回的 `hook` 的 `offload` 方法。对于循环运行模型的管道很有用。

示例：

```py
model_1, hook_1 = cpu_offload_with_hook(model_1, device)
model_2, hook_2 = cpu_offload_with_hook(model_2, device, prev_module_hook=hook_1)
model_3, hook_3 = cpu_offload_with_hook(model_3, device, prev_module_hook=hook_2)

hid_1 = model_1(input)
for i in range(50):
    # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
    hid_2 = model_2(hid_1)
# model2 is offloaded to the CPU just before this forward.
hid_3 = model_3(hid_3)

# For model3, you need to manually call the hook offload method.
hook_3.offload()
```

**参数：**

model (`torch.nn.Module`) ：要卸载的模型。

execution_device(`str`, `int` 或 `torch.device`, *可选*) ：应执行模型的设备。如果可用，则默认为 MPS 设备，如果有加速器设备，则默认为设备 0，最后为 CPU。

prev_module_hook (`UserCpuOffloadHook`, *可选*) ：此函数为您正在运行的管道中的先前模型发送回的钩子。如果通过，它的 offload 方法将在该钩子所附加的模型的转发之前被调用。

### disk_offload[[accelerate.disk_offload]]

####加速.disk_offload[[加速.disk_offload]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L269)激活模型的全磁盘卸载。因此，模型的所有参数都将被卸载为
给定文件夹中的内存映射数组。在前向传递期间，将从该文件夹访问参数并
放入需要时传递的执行设备，然后再次卸载。

**参数：**

model (`torch.nn.Module`) ：要卸载的模型。

offload_dir （`str` 或 `os.PathLike`）：要在其中卸载模型权重的文件夹（或已卸载模型权重的文件夹）。

execution_device (`torch.device`, *可选*) ：将执行模型前向传递的设备（应该是 GPU）。将默认为模型的第一个参数设备。

offload_buffers (`bool`，*可选*，默认为`False`)：是否使用模型参数卸载缓冲区。preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。

###dispatch_model[[accelerate.dispatch_model]]

####加速.dispatch_model[[accelerate.dispatch_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L315)

根据给定的设备映射调度模型。模型的各层可能分布在 GPU 上，并卸载在
CPU 甚至磁盘。

**参数：**

model (`torch.nn.Module`) ：要调度的模型。

device_map (`Dict[str, Union[str, int, torch.device]]`) ：将模型`state_dict`中的模块名称映射到它们应该访问的设备的字典。请注意，即使 `"disk"` 不是 `torch.device` 的正确值，它也会被接受。

main_device (`str`, `int` 或 `torch.device`, *可选*) : 主执行设备。将默认为`device_map`中与`"cpu"`或`"disk"`不同的第一个设备。

state_dict (`Dict[str, torch.Tensor]`, *可选*) ：将保留在 CPU 上的模型部分的状态字典。offload_dir （`str` 或 `os.PathLike`）：要在其中卸载模型权重的文件夹（或已卸载模型权重的文件夹）。

offload_index (`Dict`, *可选*) ：从权重名称到其信息的字典（`dtype`/ `shape` 或安全张量文件名）。默认为`save_folder`中保存的索引。

offload_buffers (`bool`，*可选*，默认为`False`)：是否使用模型参数卸载缓冲区。

Skip_keys（`str` 或 `List[str]`，*可选*）：在设备之间移动输入或输出时要忽略的键列表。

preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。

force_hooks（`bool`，*可选*，默认为`False`）：是否强制将设备挂钩附加到模型，即使所有层都分派到单个设备。### load_checkpoint_and_dispatch[[accelerate.load_checkpoint_and_dispatch]]

####加速.load_checkpoint_and_dispatch[[accelerate.load_checkpoint_and_dispatch]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L522)

在模型内加载（可能是分片的）检查点，可能将权重按原样发送到给定设备
加载并添加各种挂钩，使该模型正常运行（即使跨设备拆分）。

示例：

```python
>>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
>>> from huggingface_hub import hf_hub_download
>>> from transformers import AutoConfig, AutoModelForCausalLM

>>> # Download the Weights
>>> checkpoint = "EleutherAI/gpt-j-6B"
>>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

>>> # Create a model and initialize it with empty weights
>>> config = AutoConfig.from_pretrained(checkpoint)
>>> with init_empty_weights():
...     model = AutoModelForCausalLM.from_config(config)

>>> # Load the checkpoint and dispatch it to the right devices
>>> model = load_checkpoint_and_dispatch(
...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
... )
```

**参数：**

model (`torch.nn.Module`) ：我们要在其中加载检查点的模型。

检查点（`str` 或 `os.PathLike`）：要加载的文件夹检查点。它可以是： - 包含整个模型状态字典的文件的路径 - 包含分片检查点索引的 `.json` 文件的路径 - 包含唯一 `.index.json` 文件和检查点分片的文件夹的路径。device_map (`Dict[str, Union[int, str, torch.device]]`, *可选*) ：指定每个子模块应该去哪里的映射。它不需要细化到每个参数/缓冲区名称，一旦给定的模块名称在里面，它的每个子模块都会被发送到同一个设备。  要让 Accelerate 自动计算最优化的 `device_map`，请设置 `device_map="auto"`。有关每个选项的更多信息，请参阅[here](../concept_guides/big_model_inference#designing-a-device-map)。默认为 None，这意味着 [dispatch_model()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.dispatch_model) 不会被调用。

max_memory (`Dict`, *可选*) ：最大内存的字典设备标识符。如果未设置，将默认为每个 GPU 可用的最大内存和可用 CPU RAM。

no_split_module_classes (`List[str]`, *可选*) ：永远不应跨设备拆分的层类名称列表（例如具有剩余连接的任何层）。

offload_folder（`str`或`os.PathLike`，*可选*）：如果`device_map`包含任何值`"disk"`，我们将在其中卸载权重的文件夹。

offload_buffers (`bool`, *可选*, 默认为`False`) : 在CPU或硬盘上卸载的层中，是否卸载缓冲区以及参数。

dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。offload_state_dict (`bool`, *可选*) ：如果`True`，将暂时卸载硬盘上的 CPU 状态字典，以避免在 CPU 状态字典 + 最大分片的权重不合适时耗尽 CPU RAM。如果选取的设备映射包含 `"disk"` 值，则默认为 `True`。

Skip_keys（`str`或`List[str]`，*可选*）：在设备之间移动输入或输出时要忽略的键列表。

preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。

force_hooks（`bool`，*可选*，默认为`False`）：是否强制将设备挂钩附加到模型，即使所有层都分派到单个设备。

strict (`bool`，*可选*，默认为`False`)：是否严格强制检查点state_dict中的键与模型state_dict的键匹配。full_state_dict（`bool`，*可选*，默认为`True`）：如果设置为`True`，则将收集加载的state_dict中的所有张量。加载的state_dict中不会有ShardedTensor和DTensor。

Broadcast_from_rank0（`False`，*可选*，默认为`False`）：当选项为`True`时，必须初始化分布式`ProcessGroup`。 rank0应该接收一个完整的state_dict，并将state_dict中的张量一一广播到其他rank。其他级别将根据模型中的本地分片接收张量和分片（如果适用）。

### load_checkpoint_in_model[[accelerate.load_checkpoint_in_model]]

####加速.load_checkpoint_in_model[[accelerate.load_checkpoint_in_model]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1805)

在模型内加载（可能是分片的）检查点，可能将权重按原样发送到给定设备
已加载。

一旦跨设备加载，您仍然需要在模型上调用[dispatch_model()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.dispatch_model)以使其能够运行。至
将检查点加载和调度分组到一个调用中，使用[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)。

**参数：**

model (`torch.nn.Module`) ：我们要在其中加载检查点的模型。检查点（`str` 或 `os.PathLike`）：要加载的文件夹检查点。它可以是： - 包含整个模型状态字典的文件的路径 - 包含分片检查点索引的 `.json` 文件的路径 - 包含唯一 `.index.json` 文件和检查点分片的文件夹的路径。 - 包含唯一 pytorch_model.bin 或 model.safetensors 文件的文件夹的路径。

device_map (`Dict[str, Union[int, str, torch.device]]`, *可选*) ：指定每个子模块应该去哪里的映射。它不需要细化到每个参数/缓冲区名称，一旦给定的模块名称在里面，它的每个子模块都会被发送到同一个设备。

offload_folder（`str`或`os.PathLike`，*可选*）：如果`device_map`包含任何值`"disk"`，我们将在其中卸载权重的文件夹。

dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。

offload_state_dict（`bool`，*可选*，默认为`False`）：如果`True`，将暂时将CPU状态字典卸载到硬盘上，以避免在CPU状态字典+最大分片的权重不适合时耗尽CPU RAM。offload_buffers (`bool`，*可选*，默认为`False`)：是否将缓冲区包含在卸载到磁盘的权重中。

keep_in_fp32_modules(`List[str]`, *可选*) ：我们保留在 `torch.float32` dtype 中的模块列表。

offload_8bit_bnb (`bool`, *可选*) : 是否启用 cpu/磁盘上的 8 位模块卸载。

strict (`bool`，*可选*，默认为`False`)：是否严格强制检查点state_dict中的键与模型state_dict的键匹配。

full_state_dict（`bool`，*可选*，默认为`True`）：如果设置为`True`，则将收集加载的state_dict中的所有张量。加载的state_dict中不会有ShardedTensor和DTensor。

Broadcast_from_rank0（`False`，*可选*，默认为`False`）：当选项为`True`时，必须初始化分布式`ProcessGroup`。 rank0应该接收一个完整的state_dict，并将state_dict中的张量一一广播到其他rank。其他级别将根据模型中的本地分片接收张量和分片（如果适用）。

### infer_auto_device_map[[accelerate.infer_auto_device_map]]

####加速.infer_auto_device_map[[accelerate.infer_auto_device_map]][Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L1295)

计算给定模型的设备映射，优先考虑 GPU，然后卸载到 CPU，最后卸载到磁盘，
这样：
- 我们不会超出任何 GPU 的可用内存。
- 如果需要卸载到 CPU，则 GPU 0 上始终留有空间来放回 CPU 上卸载的层
  具有最大的尺寸。
- 如果需要卸载到 CPU，我们不会超出 CPU 上可用的 RAM。
- 如果需要卸载到磁盘，CPU 上总是留有空间来放回磁盘上卸载的层
  其尺寸最大。

所有计算都是通过分析模型参数的大小和数据类型来完成的。因此，该模型可以在
元设备（就像在`init_empty_weights`上下文管理器中初始化一样）。

**参数：**

model (`torch.nn.Module`) ：要分析的模型。

max_memory (`Dict`, *可选*) ：最大内存的字典设备标识符。如果未设置，将默认为最大可用内存。示例：`max_memory={0: "1GB"}`。

no_split_module_classes (`List[str]`, *可选*) ：永远不应跨设备拆分的层类名称列表（例如具有剩余连接的任何层）。dtype（`str`或`torch.dtype`，*可选*）：如果提供，权重将在加载时转换为该类型。

Special_dtypes (`Dict[str, Union[str, torch.device]]`, *可选*) ：如果提供，则考虑某些特定权重的特殊 dtypes（将覆盖用作所有权重默认值的 dtype）。

verbose (`bool`，*可选*，默认为`False`)：是否在函数构建 device_map 时提供调试语句。

clean_result (`bool`，*可选*，默认为`True`) ：通过将同一设备上的所有子模块分组在一起来清理生成的 device_map。

offload_buffers (`bool`，*可选*，默认为`False`) ：在CPU或硬盘驱动器上卸载的层中，是否卸载缓冲区以及参数。

Fallback_allocation (`bool`，*可选*，默认为`False`)：当常规分配失败时，尝试使用 BFS 分配适合大小限制的模块。

## 钩子

### ModelHook[[accelerate.hooks.ModelHook]]

####加速.hooks.ModelHook[[accelerate.hooks.ModelHook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L58)一个钩子，包含在模型的前向方法之前和之后执行的回调。区别
PyTorch 现有的钩子是它们沿着 kwargs 传递。

类属性：
- **no_grad** (`bool`, *可选*, 默认为`False`) -- 是否执行实际的前向传递
  `torch.no_grad()` 上下文管理器。

detach_hookaccelerate.hooks.ModelHook.detach_hookhttps://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L106[{"name": "module", "val": ""}]- **module** (`torch.nn.Module`) -- 从此钩子分离的模块.0

当钩子与模块分离时执行。

**参数：**

module (`torch.nn.Module`) ：从该钩子上分离的模块。
#### init_hook[[accelerate.hooks.ModelHook.init_hook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L70)

当钩子附加到模块时执行。

**参数：**

module (`torch.nn.Module`) ：连接到此钩子的模块。
#### post_forward[[accelerate.hooks.ModelHook.post_forward]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L93)

在模型的forward方法之后执行。

**参数：**

module (`torch.nn.Module`) ：在此事件之前执行前向传播的模块。

输出（`Any`）：模块的输出。

**退货：**

``Any``处理后的`output`。
#### pre_forward[[accelerate.hooks.ModelHook.pre_forward]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L79)

在模型的forward方法之前执行。

**参数：**

module (`torch.nn.Module`) : 在此事件之后将执行前向传播的模块。

args (`Tuple[Any]`) ：传递给模块的位置参数。

kwargs (`Dict[Str, Any]`) ：传递给模块的关键字参数。

**退货：**

``Tuple[Tuple[Any], Dict[Str, Any]]``

包含经过处理的 `args` 和 `kwargs` 的元组。

### AlignDevicesHook[[accelerate.hooks.AlignDevicesHook]]

####加速.hooks.AlignDevicesHook[[accelerate.hooks.AlignDevicesHook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L242)

通用的`ModelHook`，确保输入和模型权重位于同一设备上，以进行前向传递
相关模块，可能在前向传递后卸载权重。

**参数：**

execution_device (`torch.device`, *可选*) ：在前向传递之前应放置输入和模型权重的设备。

offload (`bool`，*可选*，默认为`False`)：前向传递后是否应卸载权重。

io_same_device（`bool`，*可选*，默认为`False`）：输出是否应放置在与输入相同的设备上。weights_map (`Mapping[str, torch.Tensor]`, *可选*) ：卸载模型权重时，（可能是惰性的）从参数名称到张量值的映射。

offload_buffers (`bool`，*可选*，默认为`False`)：卸载时是否包含关联模块的缓冲区。

place_submodules (`bool`, *可选*, 默认为`False`) : 是否在`init_hook`事件期间将子模块放置在`execution_device`上。

### SequentialHook[[accelerate.hooks.SequentialHook]]

####加速.hooks.SequentialHook[[accelerate.hooks.SequentialHook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L116)

一个钩子可以包含多个钩子并在每个事件中迭代它们。

### LayerwiseCastingHook[[accelerate.hooks.LayerwiseCastingHook]]

####加速.hooks.LayerwiseCastingHook[[accelerate.hooks.LayerwiseCastingHook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L784)

一个钩子，将模块的权重转换为高精度数据类型以进行计算，并转换为低精度数据类型
用于存储。此过程可能会导致输出质量下降，但会显着减少内存
足迹。

## 添加钩子

### add_hook_to_module[[accelerate.hooks.add_hook_to_module]]

####加速.hooks.add_hook_to_module[[accelerate.hooks.add_hook_to_module]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L147)向给定模块添加一个钩子。这将重写模块的 `forward` 方法以包含钩子，以删除
这种行为并恢复原来的`forward`方法，使用`remove_hook_from_module`。

如果模块已经包含一个钩子，这将用默认传递的新钩子替换它。链接两个钩子
一起传递`append=True`，因此它将当前和新的钩子链接到`SequentialHook`类的实例中。

**参数：**

module (`torch.nn.Module`) ：要附加钩子的模块。

挂钩 (`ModelHook`) ：要连接的挂钩。

附加（`bool`，*可选*，默认为`False`）：钩子是否应与现有钩子链接（如果模块已包含钩子）。

**退货：**

``torch.nn.Module``

相同的模块，附加了钩子（模块被修改到位，所以结果可以
被丢弃）。

### Attach_execution_device_hook[[accelerate.hooks.attach_execution_device_hook]]

####加速.hooks.attach_execution_device_hook[[accelerate.hooks.attach_execution_device_hook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L443)

递归地将 `AlignDevicesHook` 附加到给定模型的所有子模块，以确保它们具有正确的
执行装置

**参数：**

module (`torch.nn.Module`) ：我们要附加钩子的模块。execution_device（`int`、`str`或`torch.device`）：在前向传递之前应放置输入和模型权重的设备。

Skip_keys（`str`或`List[str]`，*可选*）：在设备之间移动输入或输出时要忽略的键列表。

preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。

tie_params_map (可选[Dict[int, Dict[torch.device, torch.Tensor]]], *可选*, 默认为 `None`) : 指向已调度绑定权重的设备字典的数据指针映射。对于给定的执行设备，此参数可用于为所有其他执行设备重用共享权重的第一个可用指针，而不是复制内存。

### Attach_align_device_hook[[accelerate.hooks.attach_align_device_hook]]####加速.hooks.attach_align_device_hook[[accelerate.hooks.attach_align_device_hook]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L491)

递归地将 `AlignDevicesHook` 附加到给定模型的具有直接参数和/或
缓冲区。

**参数：**

module (`torch.nn.Module`) ：我们要附加钩子的模块。

execution_device (`torch.device`, *可选*) ：在前向传递之前应放置输入和模型权重的设备。

offload (`bool`，*可选*，默认为`False`)：前向传播后是否应卸载权重。

weights_map (`Mapping[str, torch.Tensor]`, *可选*) ：卸载模型权重时，（可能是惰性的）从参数名称到张量值的映射。

offload_buffers (`bool`，*可选*，默认为`False`)：卸载时是否包含关联模块的缓冲区。

module_name (`str`，*可选*，默认为`""`)：模块的名称。

Skip_keys（`str`或`List[str]`，*可选*）：在设备之间移动输入或输出时要忽略的键列表。preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。

tie_params_map (可选[Dict[int, Dict[torch.device, torch.Tensor]]], *可选*, 默认为 `None`) : 指向已调度绑定权重的设备字典的数据指针映射。对于给定的执行设备，此参数可用于为所有其他执行设备重用共享权重的第一个可用指针，而不是复制内存。

### Attach_align_device_hook_on_blocks[[accelerate.hooks.attach_align_device_hook_on_blocks]]

####加速.hooks.attach_align_device_hook_on_blocks[[accelerate.hooks.attach_align_device_hook_on_blocks]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L586)

根据需要将`AlignDevicesHook`附加到给定模型的所有块。

**参数：**

module (`torch.nn.Module`) ：我们要附加钩子的模块。execution_device（`torch.device`或`Dict[str, torch.device]`，*可选*）：在前向传递之前应放置输入和模型权重的设备。它可以是整个模块的一个设备，也可以是模块名称到设备的字典映射。

offload (`bool`，*可选*，默认为`False`)：前向传播后是否应卸载权重。它可以是整个模块的一个布尔值，也可以是将模块名称映射到布尔值的字典。

weights_map (`Mapping[str, torch.Tensor]`, *可选*) ：卸载模型权重时，（可能是惰性的）从参数名称到张量值的映射。

offload_buffers (`bool`，*可选*，默认为`False`)：卸载时是否包含关联模块的缓冲区。

module_name (`str`，*可选*，默认为`""`)：模块的名称。

Skip_keys（`str` 或 `List[str]`，*可选*）：在设备之间移动输入或输出时要忽略的键列表。preload_module_classes (`List[str]`, *可选*) ：类的列表，其实例应在转发开始时加载其所有权重（甚至在子模块中）。这应该仅用于具有已注册但在转发期间不直接调用的子模块的类，例如，如果注册了`dense`线性层，但在转发时，在某些操作中使用`dense.weight`和`dense.bias`而不是直接调用`dense`。

tie_params_map (可选[Dict[int, Dict[torch.device, torch.Tensor]]], *可选*, 默认为 `None`) : 指向已调度绑定权重的设备字典的数据指针映射。对于给定的执行设备，此参数可用于为所有其他执行设备重用共享权重的第一个可用指针，而不是复制内存。

### Attach_layerwise_casting_hooks[[accelerate.big_modeling.attach_layerwise_casting_hooks]]

####加速.big_modeling.attach_layerwise_casting_hooks[[accelerate.big_modeling.attach_layerwise_casting_hooks]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/big_modeling.py#L663)将分层铸造应用于给定模块。这里期望的模块是 PyTorch `nn.Module`。这有助于
当不想完全量化模型时减少内存需求。模型参数可以保存为，
`torch.float8_e4m3fn` 并在前向传递期间向上转换为更高精度，如 `torch.bfloat16` 并向下转换
返回`torch.float8_e4m3fn`以实现内存节省。

示例：

```python
>>> from accelerate.hooks import attach_layerwise_casting_hooks
>>> from transformers import AutoModelForCausalLM
>>> import torch

>>> # Model
>>> checkpoint = "EleutherAI/gpt-j-6B"
>>> model = AutoModelForCausalLM.from_pretrained(checkpoint)

>>> # Attach hooks and perform inference
>>> attach_layerwise_casting_hooks(model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
>>> with torch.no_grad():
...     model(...)
```

用户还可以传递他们想要避免沮丧的模块。

```py
>>> attach_layerwise_casting_hooks(
...     model, storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16, skip_modules_pattern=["norm"]
... )
```

**参数：**

module (`torch.nn.Module`) ：该模块的叶子模块将被转换为高精度 dtype 以进行计算，并转换为低精度 dtype 以进行存储。

storage_dtype (`torch.dtype`) ：将模块转换为前向传递之前/之后进行存储的 dtype。

compute_dtype (`torch.dtype`) ：在前向传递计算期间将模块转换为的 dtype。

Skip_modules_pattern（`tuple[str, ...]`，默认为`None`）：与分层铸造过程中要跳过的模块名称相匹配的模式列表。如果设置为 `None` 且 `skip_modules_classes` 为 `None`，则分层铸造将直接应用于模块而不是其内部子模块。Skip_modules_classes（`tuple[type[torch.nn.Module], ...]`，默认为`None`）：在分层铸造过程中要跳过的模块类列表。

non_blocking (`bool`，默认为`False`)：如果`True`，则权重铸造操作是非阻塞的。

## 移除钩子

### remove_hook_from_module[[accelerate.hooks.remove_hook_from_module]]

####加速.hooks.remove_hook_from_module[[accelerate.hooks.remove_hook_from_module]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L205)

删除通过 `add_hook_to_module` 连接到模块的任何钩子。

**参数：**

module (`torch.nn.Module`) ：要附加钩子的模块。

recurse (`bool`, **可选**) : 是否递归移除钩子

**退货：**

``torch.nn.Module``

相同的模块，钩子分离（模块就地修改，所以结果可以
被丢弃）。

### remove_hook_from_submodules[[accelerate.hooks.remove_hook_from_submodules]]

####加速.hooks.remove_hook_from_submodules[[accelerate.hooks.remove_hook_from_submodules]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/hooks.py#L574)

递归地删除附加在给定模型的子模块上的所有挂钩。

**参数：**

module (`torch.nn.Module`) ：要删除其上所有钩子的模块。

## 实用程序

### has_offloaded_params[[accelerate.utils.has_offloaded_params]]####加速.utils.has_offloaded_params[[accelerate.utils.has_offloaded_params]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L2152)

通过检查给定模块是否具有附加的 AlignDevicesHook 来检查模块是否已卸载参数
启用卸载

**参数：**

module (`torch.nn.Module`) ：检查卸载挂钩的模块。

**退货：**

`bool`

如果模块有卸载钩子并且启用了卸载，则为`True`，否则为`False`。

###align_module_device[[accelerate.utils.align_module_device]]

####加速.utils.align_module_device[[accelerate.utils.align_module_device]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/modeling.py#L2168)

将模块的参数移动到指定执行设备的上下文管理器。

**参数：**

module (`torch.nn.Module`) ：具有要对齐的参数的模块。

execution_device (`torch.device`, *可选*) ：如果提供，则覆盖上下文中模块的执行设备。否则，使用钩子执行器或通过

### 实验追踪器
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/tracking.md