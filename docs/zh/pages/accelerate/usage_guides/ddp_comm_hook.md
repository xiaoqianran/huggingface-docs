<!-- huggingface-docs: machine-translated zh-CN from English source -->

# DDP 通信挂钩

分布式数据并行（DDP）通信钩子提供了一个通用接口，通过覆盖 `DistributedDataParallel` 中的普通 allreduce 来控制如何在工作线程之间通信梯度。提供了一些内置的通信挂钩，用户可以轻松应用这些挂钩中的任何一个来优化通信。

- **FP16 压缩挂钩**：通过将梯度转换为半精度浮点格式 (`torch.float16`) 来压缩梯度，从而减少通信开销。
- **BF16 Compression Hook**：与 FP16 类似，但使用 Brain 浮点格式 (`torch.bfloat16`)，在某些硬件上可以更高效。
- **PowerSGD Hook**：一种先进的梯度压缩算法，可提供高压缩率并可以加速带宽限制的分布式训练。

在本教程中，您将了解如何快速设置 DDP 通信挂钩并使用 Accelerate 中提供的实用程序执行训练，这就像添加一行新代码一样简单！这演示了如何使用 DDP 通信挂钩来优化 Accelerate 库的分布式训练中的梯度通信。

## FP16 压缩钩

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from accelerate.test_utils.testing import get_backend

device_type, _, _ = get_backend()
device_id = getattr(torch, device_type, torch.cuda).current_device()

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[device_id])
model.register_comm_hook(state=None, hook=default_hooks.fp16_compress_hook)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.FP16)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### BF16 压缩钩BF16 Compression Hook API 是实验性的，需要 NCCL 版本高于 2.9.6。

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from accelerate.test_utils.testing import get_backend

device_type, _, _ = get_backend()
device_id = getattr(torch, device_type, torch.cuda).current_device()

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[device_id])
model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.BF16)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### PowerSGD 挂钩

PowerSGD 通常需要与模型梯度大小相同的额外内存来实现误差反馈，这可以补偿有偏差的压缩通信并提高准确性。

```python
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
from accelerate.test_utils.testing import get_backend

device_type, _, _ = get_backend()
device_id = getattr(torch, device_type, torch.cuda).current_device()

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

model = MyModel()
model = DDP(model, device_ids=[device_id])
state = powerSGD_hook.PowerSGDState(process_group=None)
model.register_comm_hook(state=state, hook=powerSGD_hook.powerSGD_hook)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(comm_hook=DDPCommunicationHookType.POWER_SGD)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

## DDP 通信挂钩实用程序

还有两个附加实用程序用于支持通信挂钩的可选功能。

### 通讯包装器

`comm_wrapper` 是一个用附加功能包装通信钩子的选项。例如，它可用于将 FP16 压缩与其他通信策略相结合。目前支持的包装器有 `no`、`fp16` 和 `bf16`。

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(
    comm_hook=DDPCommunicationHookType.POWER_SGD,
    comm_wrapper=DDPCommunicationHookType.FP16
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

### 通讯状态选项

`comm_state_option` 允许您传递某些通信挂钩所需的附加状态信息。这对于像`PowerSGD`这样的有状态钩子特别有用，它需要在训练步骤中维护超参数和内部状态。下面的示例展示了 `comm_state_option` 与 `PowerSGD` 挂钩的使用。

```python
from accelerate import Accelerator, DDPCommunicationHookType, DistributedDataParallelKwargs
import torch

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.layer(x)

# DDP Communication Hook setup
ddp_kwargs = DistributedDataParallelKwargs(
    comm_hook=DDPCommunicationHookType.POWER_SGD,
    comm_state_option={"matrix_approximation_rank": 2}
)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

model = MyModel()
optimizer = torch.optim.Adam(model.parameters())
data_loader = DataLoader(dataset, batch_size=16)

model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

# Training loop
for data, targets in data_loader:
    outputs = model(data)
    loss = criterion(outputs, targets)
    accelerator.backward(loss)
    optimizer.step()
    optimizer.zero_grad()
```

有关更高级的用法和附加挂钩，请参阅[PyTorch DDP Communication Hooks documentation](https://pytorch.org/docs/stable/ddp_comm_hooks.html)。### 分析器
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/profiler.md