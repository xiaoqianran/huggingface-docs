<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 训练定制

TRL 的设计考虑到了模块化，以便用户能够根据自己的需求高效地定制训练循环。以下是有关如何应用和测试不同技术的示例。

> [!注意]
> 尽管这些示例使用 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer)，但这些自定义方法适用于 TRL 中的大多数（如果不是全部）训练器。

## 使用不同的优化器和调度器

默认情况下，[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 创建一个`torch.optim.AdamW` 优化器。您可以创建并定义不同的优化器并将其传递给[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer)，如下所示：

```python
from datasets import load_dataset
from torch import optim
from transformers import AutoModelForCausalLM
from trl import DPOTrainer

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
optimizer = optim.SGD(model.parameters(), lr=1e-6)

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
    optimizers=(optimizer, None),
)
trainer.train()
```

### 添加学习率调度器

您还可以通过传递优化器和调度器来添加学习率调度器：

```python
from torch import optim

optimizer = optim.AdamW(model.parameters(), lr=1e-6)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

trainer = DPOTrainer(..., optimizers=(optimizer, lr_scheduler))
```

## 通过8位参考模型

由于使用 `from_pretrained` 从 `transformers` 加载模型时，`trl` 支持所有关键字参数，因此您还可以利用 `transformers` 中的 `load_in_8bit` 进行更高效的内存微调。

在 `transformers` [Load in 8bit or 4bit](https://huggingface.co/docs/transformers/en/peft) 中了解有关 8 位模型加载的更多信息。

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(load_in_8bit=True)
ref_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", quantization_config=quantization_config)

trainer = DPOTrainer(..., ref_model=ref_model)
```

## 添加自定义回调

您可以通过添加用于日志记录、监控或提前停止的回调来自定义训练循环。回调允许您在训练期间的特定点执行自定义代码。

```python
from transformers import TrainerCallback

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"Step {state.global_step}: {logs}")

trainer = DPOTrainer(..., callbacks=[CustomLoggingCallback()])
```

## 添加自定义评估指标您可以定义自定义评估指标以在训练期间进行跟踪。这对于监控特定任务的模型性能非常有用。

```python
def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # Add your metric computation here
    return {"custom_metric": 0.0}

training_args = DPOConfig(..., eval_strategy="steps", eval_steps=100)

trainer = DPOTrainer(..., eval_dataset=eval_dataset, compute_metrics=compute_metrics)
```

## 使用混合精度训练

混合精度训练可以显着加快训练速度并减少内存使用。您可以通过在训练配置中设置`bf16=True`或`fp16=True`来启用它。

```python
# Use bfloat16 precision (recommended for modern GPUs)
training_args = DPOConfig(..., bf16=True)
```

注意：对于 Ampere GPU（A100、RTX 30xx）或更新版本使用 `bf16=True`，对于较旧的 GPU 使用 `fp16=True`。

## 使用梯度累积

当使用有限的 GPU 内存进行训练时，梯度累积允许您在更新权重之前通过多个步骤累积梯度来模拟更大的批量大小。

```python
# Simulate a batch size of 32 with per_device_train_batch_size=4 and gradient_accumulation_steps=8
training_args = DPOConfig(
    ...,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
)
```

### 固态硬盘
https://huggingface.co/docs/trl/v1.9.2/ssd_trainer.md