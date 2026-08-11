<!-- huggingface-docs: machine-translated zh-CN from English source -->

# MergeModelCallback[[trl.experimental.merge_model_callback.MergeModelCallback]]

- **merge_config**（`experimental.merge_model_callback.MergeConfig`，*可选*）--
  用于合并过程的配置。如果没有提供，则默认
  使用`MergeConfig`。
- **merge_at_every_checkpoint**（`bool`，*可选*，默认为`False`）--
  是否在每个检查点合并模型。
- **push_to_hub**（`bool`，*可选*，默认为`False`）--
  合并后是否将合并后的模型推送到Hub。

[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback) 将策略模型（正在训练的模型）与另一个基于模型的模型合并
在合并配置上。

示例：

```python
>>> from trl.experimental.merge_model_callback import MergeConfig, MergeModelCallback

>>> config = MergeConfig()
>>> merge_callback = MergeModelCallback(config)
>>> trainer = DPOTrainer(..., callbacks=[merge_callback])
```

### 分发培训
https://huggingface.co/docs/trl/v1.9.2/distributing_training.md