<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通过 DeepSpeed 使用多个模型

    本指南假设您已阅读并理解 [DeepSpeed usage guide](./deepspeed)。

使用 Accelerate 和 DeepSpeed 运行多个模型可用于：

* 知识蒸馏
* RLHF 等训练后技术（更多示例请参阅[TRL](https://github.com/huggingface/trl)库）
* 一次训练多个模型

目前，Accelerate 有一个**非常实验性的 API** 来帮助您使用多个模型。

本教程将重点关注两个常见用例：

1. 知识蒸馏，训练较小的学生模型来模仿较大、表现更好的教师。  如果学生模型适合单个 GPU，我们可以使用 ZeRO-2 进行训练，使用 ZeRO-3 对教师进行分片以进行推理。这比两种模型使用 ZeRO-3 都要快得多。
2. 一次训练多个“不相交”模型。

## 知识蒸馏

知识蒸馏是使用多个模型但只训练其中一个模型的一个很好的例子。通常，您会为两个型号使用一个 [utils.DeepSpeedPlugin](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin)。然而，在这种情况下，有两个单独的配置。 Accelerate 允许您创建和使用多个插件**当且仅当**它们位于 `dict` 中，以便您可以在需要时引用并启用正确的插件。

```python
from accelerate.utils import DeepSpeedPlugin

zero2_plugin = DeepSpeedPlugin(hf_ds_config="zero2_config.json")
zero3_plugin = DeepSpeedPlugin(hf_ds_config="zero3_config.json")

deepspeed_plugins = {"student": zero2_plugin, "teacher": zero3_plugin}
```

`zero2_config.json` 应配置为完整训练（因此，如果您不使用自己的训练，请指定 `scheduler` 和 `optimizer`），而 `zero3_config.json` 应仅配置为推理模型，如下例所示。

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "zero_optimization": {
        "stage": 3,
        "overlap_comm": true,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
    },
    "train_micro_batch_size_per_gpu": 1
}
```

下面显示了 `zero2_config.json` 配置示例。

```json
{
    "bf16": {
        "enabled": "auto"
    },
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": "auto",
            "weight_decay": "auto",
            "torch_adam": true,
            "adam_w_mode": true
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": "auto",
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto"
        }
    },
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        },
    },
    "gradient_accumulation_steps": 1,
    "gradient_clipping": "auto",
    "train_batch_size": "auto",
    "train_micro_batch_size_per_gpu": "auto",
}
```

    如果未指定 `train_micro_batch_size_per_gpu`，即使未训练此特定模型，DeepSpeed 也会引发错误。

从这里创建一个 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) 并传入两种配置。

```python
from accelerate import Accelerator

accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
```

现在让我们看看如何使用它们。

### 学生模型

默认情况下，Accelerate 将 `dict` 中的第一项设置为默认或启用的插件（`"student"` 插件）。通过使用 [utils.deepspeed.get_active_deepspeed_plugin()](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.utils.get_active_deepspeed_plugin) 函数来验证这一点，以查看启用了哪个插件。

```python
active_plugin = get_active_deepspeed_plugin(accelerator.state)
assert active_plugin is deepspeed_plugins["student"]
```

`AcceleratorState` 还保留保存在`state.deepspeed_plugin` 中的活动 DeepSpeed 插件。
```python
assert active_plugin is accelerator.deepspeed_plugin
```

由于 `student` 是当前活动的插件，让我们继续准备模型、优化器和调度器。

```python
student_model, optimizer, scheduler = ...
student_model, optimizer, scheduler, train_dataloader = accelerator.prepare(student_model, optimizer, scheduler, train_dataloader)
```现在是时候处理教师模型了。

### 教师模型

首先，您需要在[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)中指定应使用`zero3_config.json`配置。

```python
accelerator.state.select_deepspeed_plugin("teacher")
```

这将禁用 `"student"` 插件并启用 `"teacher"` 插件。的
Transformers 内部的 DeepSpeed 有状态配置已更新，并且它更改了使用时调用的插件配置
`deepspeed.initialize()`。这允许您使用 Transformers 提供的自动 `deepspeed.zero.Init` 上下文管理器集成。

```python
teacher_model = AutoModel.from_pretrained(...)
teacher_model = accelerator.prepare(teacher_model)
```

否则，您应该使用 `deepspeed.zero.Init` 手动初始化模型。
```python
with deepspeed.zero.Init(accelerator.deepspeed_plugin.config):
    model = MyModel(...)
```

### 训练

从这里开始，您的训练循环可以是任何您喜欢的，只要 `teacher_model` 从未被训练即可。

```python
teacher_model.eval()
student_model.train()
for batch in train_dataloader:
    with torch.no_grad():
        output_teacher = teacher_model(**batch)
    output_student = student_model(**batch)
    # Combine the losses or modify it in some way
    loss = output_teacher.loss + output_student.loss
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
```

## 训练多个不相交的模型

训练多个模型是一个更复杂的场景。
在当前状态下，我们假设每个模型在训练期间与另一个模型**完全脱节**。

这个场景仍然需要制作两个[utils.DeepSpeedPlugin](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin)。但是，您还需要第二个 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)，因为不同的 `deepspeed` 引擎在不同时间被调用。单个[Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator)一次只能携带一个实例。由于 [state.AcceleratorState](/docs/accelerate/v1.14.0/en/package_reference/state#accelerate.state.AcceleratorState) 是一个有状态对象，因此它已经知道 [utils.DeepSpeedPlugin](/docs/accelerate/v1.14.0/en/package_reference/deepspeed#accelerate.DeepSpeedPlugin) 都可用。您可以实例化第二个 [Accelerator](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator) ，无需额外参数。

```python
first_accelerator = Accelerator(deepspeed_plugins=deepspeed_plugins)
second_accelerator = Accelerator()
```

您可以调用`first_accelerator.state.select_deepspeed_plugin()`来启用或禁用
一个特定的插件，然后调用`prepare`。

```python
# can be `accelerator_0`, `accelerator_1`, or by calling `AcceleratorState().select_deepspeed_plugin(...)`
first_accelerator.state.select_deepspeed_plugin("first_model")
first_model = AutoModel.from_pretrained(...)
# For this example, `get_training_items` is a nonexistent function that gets the setup we need for training
first_optimizer, first_scheduler, train_dl, eval_dl = get_training_items(model1)
first_model, first_optimizer, first_scheduler, train_dl, eval_dl = accelerator.prepare(
    first_model, first_optimizer, first_scheduler, train_dl, eval_dl
)

second_accelerator.state.select_deepspeed_plugin("second_model")
second_model = AutoModel.from_pretrained(...)
# For this example, `get_training_items` is a nonexistent function that gets the setup we need for training
second_optimizer, second_scheduler, _, _ = get_training_items(model2)
second_model, second_optimizer, second_scheduler = accelerator.prepare(
    second_model, second_optimizer, second_scheduler
)
```

现在您可以训练：

```python
for batch in dl:
    outputs1 = first_model(**batch)
    first_accelerator.backward(outputs1.loss)
    first_optimizer.step()
    first_scheduler.step()
    first_optimizer.zero_grad()
    
    outputs2 = model2(**batch)
    second_accelerator.backward(outputs2.loss)
    second_optimizer.step()
    second_scheduler.step()
    second_optimizer.zero_grad()
```

## 资源

要查看更多示例，请查看当前[加速]中的[related tests](https://github.com/huggingface/accelerate/blob/main/src/accelerate/test_utils/scripts/external_deps/test_ds_multiple_model.py)。

### 完全分片数据并行
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/fsdp.md