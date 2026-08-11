<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 实验追踪器

有大量可用的实验跟踪 API，但是让它们全部在多处理环境中工作通常很复杂。
Accelerate 提供了一个通用的跟踪 API，可用于通过 `Accelerator.log()` 在脚本过程中记录有用的项目

## 集成追踪器

目前`Accelerate`支持八个开箱即用的跟踪器：

- 张量板
- 旺德贝 
- Trackio
- 彗星ML
- 目标
- MLFlow
- 清除ML
- DVCLive

要使用其中任何一个，请将所选类型传递给 `Accelerate` 中的 `log_with` 参数：
```python
from accelerate import Accelerator
from accelerate.utils import LoggerType

accelerator = Accelerator(log_with="all")  # For all available trackers in the environment
accelerator = Accelerator(log_with="wandb")
accelerator = Accelerator(log_with=["wandb", LoggerType.TENSORBOARD])
```

在实验开始时，应使用 `Accelerator.init_trackers()` 设置项目，并可能添加要记录的任何实验超参数：
```python
hps = {"num_iterations": 5, "learning_rate": 1e-2}
accelerator.init_trackers("my_project", config=hps)
```

当您准备好记录任何数据时，应使用`Accelerator.log()`。
还可以传入 `step` 将数据与训练循环中的特定步骤相关联。
```python
accelerator.log({"train_loss": 1.12, "valid_loss": 0.8}, step=1)
```

完成训练后，请确保运行[Accelerator.end_training()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.end_training)，以便所有跟踪器都可以运行其完成功能（如果有）。
```python
accelerator.end_training()
```

完整的示例如下：
```python
from accelerate import Accelerator

accelerator = Accelerator(log_with="all")
config = {
    "num_iterations": 5,
    "learning_rate": 1e-2,
    "loss_function": str(my_loss_function),
}

accelerator.init_trackers("example_project", config=config)

my_model, my_optimizer, my_training_dataloader = accelerator.prepare(my_model, my_optimizer, my_training_dataloader)
device = accelerator.device
my_model.to(device)

for iteration in range(config["num_iterations"]):
    for step, batch in enumerate(my_training_dataloader):
        my_optimizer.zero_grad()
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        outputs = my_model(inputs)
        loss = my_loss_function(outputs, targets)
        accelerator.backward(loss)
        my_optimizer.step()
        accelerator.log({"training_loss": loss}, step=step)
accelerator.end_training()
```如果跟踪器需要一个目录来保存数据，例如`TensorBoard`，则将目录路径传递给`project_dir`。 `project_dir`参数很有用 
当[ProjectConfiguration](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.ProjectConfiguration)数据类中有其他配置需要组合时。例如，如果记录器仅应在主进程上执行，您可以将 TensorBoard 数据保存到 `project_dir`，其他所有内容都可以记录在 [⟦T25⟧: 

⟦T5⟧

## Implementing Custom Trackers

To implement a new tracker to be used in ⟦T26⟧, a new one can be made through implementing the ⟦T27⟧ class.
Every tracker must implement three functions and have three properties:
  - ⟦T28⟧: 
    - Should store a ⟦T29⟧ and initialize the tracker API of the integrated library. 
    - If a tracker stores their data locally (such as TensorBoard), a ⟦T30⟧ parameter can be added.
  - ⟦T31⟧: 
    - Should take in a ⟦T32⟧ dictionary and store them as a one-time experiment configuration
  - ⟦T33⟧: 
    - Should take in a ⟦T34⟧ dictionary and a ⟦T35⟧, and should log them to the run

  - ⟦T36⟧ (⟦T37⟧):
    - A unique string name for the tracker, such as ⟦T38⟧ for the wandb tracker. 
    - This will be used for interacting with this tracker specifically
  - ⟦T39⟧ (⟦T40⟧):
    - Whether a ⟦T41⟧ is needed for this particular tracker and if it uses one.
  - ⟦T42⟧: 
    - This should be implemented as a ⟦T43⟧ function 
    - Should return the internal tracking mechanism the library uses, such as the ⟦T44⟧ object for ⟦T45⟧.

Each method should also utilize the [state.PartialState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.PartialState) 类的 `logging_dir` 参数中。

下面是一个与权重和偏差集成的简短示例，仅包含相关信息并仅记录 
主要流程：
```python
from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb

class MyCustomTracker(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: str):
        self.run_name = run_name
        run = wandb.init(self.run_name)

    @property
    def tracker(self):
        return self.run.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        wandb.config(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)
```

当您准备好构建 `Accelerator` 对象时，请将跟踪器的 **实例** 传递给 `Accelerator.log_with` 以自动获取它
与 API 一起使用：

```python
tracker = MyCustomTracker("some_run_name")
accelerator = Accelerator(log_with=tracker)
```

这些也可以与现有的跟踪器混合使用，包括 `"all"`：

```python
tracker = MyCustomTracker("some_run_name")
accelerator = Accelerator(log_with=[tracker, "all"])
```

## 访问内部跟踪器 

如果可能需要直接与跟踪器进行一些自定义交互，您可以使用 
[Accelerator.get_tracker()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.get_tracker)方法。只需传入与跟踪器的`.name`属性对应的字符串即可 
它将在主进程上返回该跟踪器。

此示例展示了使用 wandb 执行此操作：

```python
wandb_tracker = accelerator.get_tracker("wandb")
```从那里你可以像平常一样与 `wandb` 的 `run` 对象进行交互：

```python
wandb_tracker.log_artifact(some_artifact_to_log)
```

  Accelerate 中内置的跟踪器将自动在正确的进程上执行， 
  因此，如果跟踪器仅打算在主进程上运行，它将这样做 
  自动。

如果你想真正完全去除加速的包裹，你可以 
达到相同的结果：

```python
wandb_tracker = accelerator.get_tracker("wandb", unwrap=True)
if accelerator.is_main_process:
    wandb_tracker.log_artifact(some_artifact_to_log)
```

## 当包装器无法工作时

如果库的 API 不遵循严格的 `.log` 和整体字典（例如 Neptune.AI），则可以在 `if accelerator.is_main_process` 语句下手动完成日志记录：
```diff
  from accelerate import Accelerator
+ import neptune

  accelerator = Accelerator()
+ run = neptune.init_run(...)

  my_model, my_optimizer, my_training_dataloader = accelerate.prepare(my_model, my_optimizer, my_training_dataloader)
  device = accelerator.device
  my_model.to(device)

  for iteration in config["num_iterations"]:
      for batch in my_training_dataloader:
          my_optimizer.zero_grad()
          inputs, targets = batch
          inputs = inputs.to(device)
          targets = targets.to(device)
          outputs = my_model(inputs)
          loss = my_loss_function(outputs, targets)
          total_loss += loss
          accelerator.backward(loss)
          my_optimizer.step()
+         if accelerator.is_main_process:
+             run["logs/training/batch/loss"].log(loss)
```

### Intel CPU 训练
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/intel_cpu.md