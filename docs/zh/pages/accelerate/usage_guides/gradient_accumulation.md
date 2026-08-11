<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 Accelerate 进行梯度累积

梯度累积是一种可以在比之前更大的批量大小上进行训练的技术 
您的机器通常能够装入内存。这是通过累积梯度来完成的
多个批次，并且仅在执行了一定数量的批次后才步进优化器。

虽然技术上标准的梯度累积代码在分布式设置中可以正常工作，但它并不是最有效的
这样做的方法，您可能会遇到相当大的速度减慢！

在本教程中，您将了解如何快速设置梯度累积并使用 Accelerate 中提供的实用程序执行它，
这总共只需添加一行新代码！

此示例将使用一个非常简单的 PyTorch 训练循环，每两批执行一次梯度累积：

```python
device = "cuda"
model.to(device)

gradient_accumulation_steps = 2

for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (index + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## 将其转换为加速

首先，前面显示的代码将被转换为利用 Accelerate，而不需要特殊的梯度累积助手：

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for index, batch in enumerate(training_dataloader):
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      loss = loss / gradient_accumulation_steps
+     accelerator.backward(loss)
      if (index+1) % gradient_accumulation_steps == 0:
          optimizer.step()
          scheduler.step()
          optimizer.zero_grad()
```

  在当前状态下，由于称为梯度同步的过程，该代码不会有效地执行梯度累积。在 [Concepts tutorial](../concept_guides/gradient_synchronization) 中了解更多相关信息！## 让 Accelerate 处理梯度累积

现在剩下的就是让 Accelerate 为我们处理梯度累积。为此，您应该将 `gradient_accumulation_steps` 参数传递给 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)，指定数字 
每次调用 `step()` 之前要执行的步骤以及如何在调用 [backward()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.backward) 期间自动调整损耗：

```diff
  from accelerate import Accelerator
- accelerator = Accelerator()
+ accelerator = Accelerator(gradient_accumulation_steps=2)
```

或者，您可以将 `gradient_accumulation_plugin` 参数传递给 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 对象的 `__init__`，这将允许您进一步自定义梯度累积行为。 
在 [GradientAccumulationPlugin](../package_reference/accelerator#accelerate.utils.GradientAccumulationPlugin) 文档中了解更多相关信息。

从这里，您可以在训练循环内部使用 [accumulate()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.accumulate) 上下文管理器来自动为您执行梯度累积！
您只需将其包裹在我们代码的整个训练部分即可： 

```diff
- for index, batch in enumerate(training_dataloader):
+ for batch in training_dataloader:
+     with accelerator.accumulate(model):
          inputs, targets = batch
          outputs = model(inputs)
```

您可以删除对步骤号和损失调整的所有特殊检查：

```diff
- loss = loss / gradient_accumulation_steps
  accelerator.backward(loss)
- if (index+1) % gradient_accumulation_steps == 0:
  optimizer.step()
  scheduler.step()
  optimizer.zero_grad()
```

正如您所看到的，[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)能够跟踪您所在的批次号，并且它会自动知道是否单步执行准备好的优化器以及如何调整损失。通常，通过梯度累积，您需要调整步骤数以反映您所处理的总批次的变化 
上的训练。默认情况下，加速会自动为您执行此操作。在幕后，我们实例化一个 `GradientAccumulationPlugin` 配置来执行此操作。

[state.GradientState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.state.GradientState) 与正在迭代的活动数据加载器同步。因此，它天真地假设当我们到达数据加载器的末尾时，所有内容都将同步并执行一个步骤。要禁用此功能，请将 `GradientAccumulationPlugin` 中的 `sync_with_dataloader` 设置为 `False`：

```{python}
from accelerate import Accelerator
from accelerate.utils import GradientAccumulationPlugin

plugin = GradientAccumulationPlugin(sync_with_dataloader=False)
accelerator = Accelerator(..., gradient_accumulation_plugin=plugin)
```

## 完成的代码

下面是使用 Accelerate 执行梯度累积的完成实现

```python
from accelerate import Accelerator
accelerator = Accelerator(gradient_accumulation_steps=2)
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)
for batch in training_dataloader:
    with accelerator.accumulate(model):
        inputs, targets = batch
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

重要的是，**只能在上下文管理器`with accelerator.accumulate(model)`内完成一次前进/后退**。

要了解更多关于它的魔力，请阅读 [Gradient Synchronization concept guide](../concept_guides/gradient_synchronization)

## 独立的示例

这是一个独立的示例，您可以运行它来查看 Accelerate 的梯度累积效果：

```python
import torch
import copy
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import TensorDataset, DataLoader

# seed
set_seed(0)

# define toy inputs and labels
x = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.])
y = torch.tensor([2., 4., 6., 8., 10., 12., 14., 16.])
gradient_accumulation_steps = 4
per_device_batch_size = len(x) // gradient_accumulation_steps

# define dataset and dataloader
dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=per_device_batch_size)

# define model, optimizer and loss function
class SimpleLinearModel(torch.nn.Module):
    def __init__(self):
        super(SimpleLinearModel, self).__init__()
        self.weight = torch.nn.Parameter(torch.zeros((1, 1)))

    def forward(self, inputs):
        return inputs @ self.weight

model = SimpleLinearModel()
model_clone = copy.deepcopy(model)
criterion = torch.nn.MSELoss()
model_optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, model_optimizer, dataloader = accelerator.prepare(model, model_optimizer, dataloader)
model_clone_optimizer = torch.optim.SGD(model_clone.parameters(), lr=0.02)
print(f"initial model weight is {model.weight.mean().item():.5f}")
print(f"initial model weight is {model_clone.weight.mean().item():.5f}")
for i, (inputs, labels) in enumerate(dataloader):
    with accelerator.accumulate(model):
        inputs = inputs.view(-1, 1)
        print(i, inputs.flatten())
        labels = labels.view(-1, 1)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        accelerator.backward(loss)
        model_optimizer.step()
        model_optimizer.zero_grad()
loss = criterion(x.view(-1, 1) @ model_clone.weight, y.view(-1, 1))
model_clone_optimizer.zero_grad()
loss.backward()
model_clone_optimizer.step()
print(f"w/ accumulation, the final model weight is {model.weight.mean().item():.5f}")
print(f"w/o accumulation, the final model weight is {model_clone.weight.mean().item():.5f}")
```
```
initial model weight is 0.00000
initial model weight is 0.00000
0 tensor([1., 2.])
1 tensor([3., 4.])
2 tensor([5., 6.])
3 tensor([7., 8.])
w/ accumulation, the final model weight is 2.04000
w/o accumulation, the final model weight is 2.04000
```

## 可变大小训练样本上的梯度累积

正如这篇[blog-post](https://huggingface.co/blog/gradient_accumulation)中所指出的，它指出了对可变大小的训练样本进行梯度累积时出现的一个常见错误：> [...] 对于诸如因果 LM 训练之类的 token 级任务的梯度累积，正确的损失应该通过 **梯度累积步骤中所有批次的总损失** 除以 **这些批次中所有非填充 token 的总数** 来计算。这与每批次损失值的平均值不同。 

换句话说，必须对代币层面的损失进行一些调整。

### 骨架代码

```python
from accelerate import Accelerator
import math
import contextlib

gradient_accumulation_steps = 2
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)
model, optimizer, training_dataloader, scheduler = accelerator.prepare(
    model, optimizer, training_dataloader, scheduler
)

training_iterator = iter(training_dataloader)
num_samples_in_epoch = len(training_dataloader)
remainder = num_samples_in_epoch % gradient_accumulation_steps
remainder = remainder if remainder != 0 else gradient_accumulation_steps
total_updates = math.ceil(num_samples_in_epoch / gradient_accumulation_steps)
        

total_batched_samples = 0
for update_step in range(total_updates):
        # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
        # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
        batch_samples = []
        num_batches_in_step = gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
        for _ in range(num_batches_in_step):
            batch_samples += [next(training_iterator)]
            
        # get local num items in batch 
        num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
        num_items_in_batch = accelerator.gather(num_items_in_batch).sum().item()
            
        for i, batch in enumerate(batch_samples):
            # if we perform gradient accumulation in a multi-devices set-up, we want to avoid unnecessary communications when accumulating
            # cf: https://muellerzr.github.io/blog/gradient_accumulation.html
            if (i < len(batch_samples) - 1 and accelerator.num_processes > 1):
                ctx = model.no_sync
            else:
                ctx = contextlib.nullcontext
            
            total_batched_samples += 1

            with ctx():
                inputs, targets = batch
                outputs = model(inputs)
                loss = loss_function(outputs, targets) # the loss function should sum over samples rather than averaging
                
                # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps
                loss = (loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch
                
                accelerator.backward(loss)

        # Sync gradients and perform optimization steps once every gradient_accumulation_steps
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

### 自包含因果 LM 示例

```py
import torch
import copy
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import  get_logger
from torch.utils.data import Dataset, DataLoader
import math
import contexlib

# seed
set_seed(0)
logger = get_logger(__name__)

class MyDataset(Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.len = num_samples

    def __getitem__(self, index):
        input_ids = torch.arange(1, index+2, dtype=torch.float32)
        labels = torch.remainder(input_ids, 2)
        return {"input_ids": input_ids, "labels": labels}

    def __len__(self):
        return self.len
    
def collate_fn(features):
    input_ids = torch.nn.utils.rnn.pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=-100)
    labels = torch.nn.utils.rnn.pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)
    return {"input_ids": input_ids[..., None], "labels": labels[..., None]}

# define toy inputs and labels
gradient_accumulation_steps = 2
per_device_batch_size = 4

# define accelerator
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps)

# define dataset and dataloader
# for this toy example, we'll compute gradient descent over one single global batch
dataset = MyDataset(per_device_batch_size*gradient_accumulation_steps*accelerator.num_processes)
dataloader = DataLoader(dataset, batch_size=per_device_batch_size, collate_fn=collate_fn)

# define model, model_optimizer and loss function
model = torch.nn.Linear(1, 2, bias=False)
model_clone = copy.deepcopy(model)
criterion = torch.nn.CrossEntropyLoss(reduction="sum") # must sum over samples rather than averaging
model_optimizer = torch.optim.SGD(model.parameters(), lr=0.08)

logger.warning(f"initial model weight is {model.weight.detach().cpu().squeeze()}")
logger.warning(f"initial model clone weight is {model_clone.weight.detach().cpu().squeeze()}")

# prepare artifacts - accelerator handles device placement and dataloader splitting
model, model_optimizer = accelerator.prepare(model, model_optimizer)
dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)
training_iterator = iter(dataloader)

num_samples_in_epoch = len(dataloader)
remainder = num_samples_in_epoch % gradient_accumulation_steps
remainder = remainder if remainder != 0 else gradient_accumulation_steps
total_gradient_updates = math.ceil(num_samples_in_epoch / gradient_accumulation_steps)

total_batched_samples = 0
for update_step in range(total_gradient_updates):
        # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
        # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
        batch_samples = []
        num_batches_in_step = gradient_accumulation_steps if update_step != (total_gradient_updates - 1) else remainder
        for _ in range(num_batches_in_step):
            batch_samples += [next(training_iterator)]
            
        # get local num items in batch 
        local_num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
        logger.warning(f"Step {update_step} - Device {accelerator.process_index} - num items in the local batch {local_num_items_in_batch}", main_process_only=False)

        # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
        num_items_in_batch = accelerator.gather(local_num_items_in_batch).sum().item()
        logger.warning(f"Total num items {num_items_in_batch}")

        for i, batch in enumerate(batch_samples):
            inputs, labels = batch["input_ids"], batch["labels"]
            total_batched_samples += 1
            # if we perform gradient accumulation in a multi-devices set-up, we want to avoid unnecessary communications when accumulating
            # cf: https://muellerzr.github.io/blog/gradient_accumulation.html
            if (i < len(batch_samples) - 1 and accelerator.num_processes > 1):
                ctx = model.no_sync
            else:
                ctx = contextlib.nullcontext
            with ctx():

                outputs = model(inputs)
                loss = criterion(outputs.view(-1, 2), labels.view(-1).to(torch.int64))
                
                # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps 
                loss = (loss * gradient_accumulation_steps * accelerator.num_processes) / num_items_in_batch
                accelerator.backward(loss)
        model_optimizer.step()
        model_optimizer.zero_grad()
                

logger.warning(f"Device {accelerator.process_index} - w/ accumulation, the final model weight is {accelerator.unwrap_model(model).weight.detach().cpu().squeeze()}", main_process_only=False)

# We know do the same operation but on a single device and without gradient accumulation

if accelerator.is_main_process:
    # prepare one single entire batch
    dataloader = DataLoader(dataset, batch_size=len(dataset), collate_fn=collate_fn)
    full_batch_without_accum = next(iter(dataloader))
    total_inputs, total_labels = full_batch_without_accum["input_ids"], full_batch_without_accum["labels"]
    model_clone_optimizer = torch.optim.SGD(model_clone.parameters(), lr=0.08)
    
    # train the cloned model
    loss = torch.nn.CrossEntropyLoss(reduction="mean")(model_clone(total_inputs).view(-1, 2), total_labels.view(-1).to(torch.int64))
    model_clone_optimizer.zero_grad()
    loss.backward()
    model_clone_optimizer.step()
    
    # We should have the same final weights.
    logger.warning(f"w/o accumulation, the final model weight is {model_clone.weight.detach().cpu().squeeze()}")

```

单个设备上的结果 - 梯度累积步骤设置为 1，batch_size 设置为 8：
```
initial model weight is tensor([-0.0075,  0.5364])
initial model clone weight is tensor([-0.0075,  0.5364])
Step 0 - Device 0 - num items in the local batch 36
Total num items 36
Device 0 - w/ accumulation, the final model weight is tensor([0.0953, 0.4337])
w/o accumulation, the final model weight is tensor([0.0953, 0.4337])
```

两台设备设置的结果 - 梯度累积步骤设置为 2，batch_size 设置为 4。
```
initial model weight is tensor([-0.0075,  0.5364])
initial model clone weight is tensor([-0.0075,  0.5364])
Step 0 - Device 0 - num items in the local batch 52
Step 0 - Device 1 - num items in the local batch 84
Total num items 136
Device 1 - w/ accumulation, the final model weight is tensor([0.2117, 0.3172])
Device 0 - w/ accumulation, the final model weight is tensor([0.2117, 0.3172])
w/o accumulation, the final model weight is tensor([0.2117, 0.3172])
```

### 更进一步：

请在路径 [⟦T22⟧](https://github.com/huggingface/accelerate/blob/main/examples/by_feature/gradient_accumulation_for_autoregressive_models.py) 的示例文件夹中找到关于真实世界训练运行的完整示例脚本。

在全局批量大小恒定为 32 的多个训练配置上运行它会得到下图：请注意，直到训练步骤 20，训练损失都是完全相同的。此训练步骤之后的小偏差发生在第一个 epoch 的最后，因为到 [default](https://huggingface.co/docs/accelerate/en/package_reference/torch_wrappers#accelerate.data_loader.prepare_data_loader.even_batches)，当总批量大小没有完全划分数据集时，数据加载器会在数据集开头复制样本。

### 实验追踪器
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/tracking.md