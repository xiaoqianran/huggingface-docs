<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 检查点

使用 Accelerate 训练 PyTorch 模型时，您可能经常希望保存并继续训练状态。这样做需要
保存和加载模型、优化器、RNG 生成器和 GradScaler。 Accelerate 内部有两个方便的功能可以快速实现这一目标：
- 使用[save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state)将上述所有内容保存到文件夹位置
- 使用[load_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.load_state)加载早期`save_state`存储的所有内容

要进一步自定义通过 [save_state()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_state) 保存状态的位置和方式，可以使用 [ProjectConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.ProjectConfiguration) 类。例如 
如果启用`automatic_checkpoint_naming`，每个保存的检查点将位于`Accelerator.project_dir/checkpoints/checkpoint_{checkpoint_number}`。

应该注意的是，期望这些状态来自相同的训练脚本，而不应该来自两个单独的脚本。

- 通过使用[register_for_checkpointing()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.register_for_checkpointing)，您可以注册自定义对象以从前面的两个函数自动存储或加载，
只要该对象具有 `state_dict` **和** `load_state_dict` 功能。这可能包括诸如学习率调度程序之类的对象。 

下面是一个在训练期间使用检查点保存和重新加载状态的简短示例：

```python
from accelerate import Accelerator
import torch

accelerator = Accelerator(project_dir="my/save/path")

my_scheduler = torch.optim.lr_scheduler.StepLR(my_optimizer, step_size=1, gamma=0.99)
my_model, my_optimizer, my_training_dataloader = accelerator.prepare(my_model, my_optimizer, my_training_dataloader)

# Register the LR scheduler
accelerator.register_for_checkpointing(my_scheduler)

# Save the starting state
accelerator.save_state()

device = accelerator.device
my_model.to(device)

# Perform training
for epoch in range(num_epochs):
    for batch in my_training_dataloader:
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        accelerator.backward(loss)
        my_optimizer.step()
    my_scheduler.step()

# Restore the previous state
accelerator.load_state("my/save/path/checkpointing/checkpoint_0")
```

## 恢复DataLoader的状态从检查点恢复后，还可能需要从活动`DataLoader`中的特定点恢复，如果 
国家在一个时代的中期被拯救了。您可以使用 [skip_first_batches()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.skip_first_batches) 来执行此操作。 

```python
from accelerate import Accelerator

accelerator = Accelerator(project_dir="my/save/path")

train_dataloader = accelerator.prepare(train_dataloader)
accelerator.load_state("my_state")

# Assume the checkpoint was saved 100 steps into the epoch
skipped_dataloader = accelerator.skip_first_batches(train_dataloader, 100)

# After the first iteration, go back to `train_dataloader`

# First epoch
for batch in skipped_dataloader:
    # Do something
    pass

# Second epoch
for batch in train_dataloader:
    # Do something
    pass
```

### 模型量化
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/quantization.md