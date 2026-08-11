<!-- huggingface-docs: machine-translated zh-CN from English source -->

# GSPO 令牌

在论文[Group Sequence Policy Optimization](https://huggingface.co/papers/2507.18071)中，作者提出了 GSPO 的代币级目标变体，称为 GSPO-token。要使用 GSPO 令牌，您可以使用 `trl.experimental.gspo_token` 中的 `GRPOTrainer` 类。

## 用法

```python
from trl.experimental.gspo_token import GRPOTrainer
from trl import GRPOConfig

training_args = GRPOConfig(
    importance_sampling_level="sequence_token",
    ...
)
```

> [!警告]
> 要利用 GSPO 令牌，用户需要为序列 \\( i \\) 中的每个令牌 \\( t \\) 提供每个令牌的优势 \\( \hat{A_{i,t}} \\) （即，使 \\( \hat{A_{i,t}} \\) 随 \\( t \\) 变化 — 这里的情况并非如此， \\( \hat{A_{i,t}}=\hat{A_{i}} \\))。否则，GSPO-Token 梯度仅相当于原始 GSPO 实现。

## GRPOTrainer[[trl.GRPOTrainer]]- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于 `True`，加载 *args.output_dir* 中由前一个实例保存的最后一个检查点
  `Trainer`。如果存在，训练将从此处加载的模型/优化器/调度器状态恢复。
- **试用**（`optuna.Trial`或`dict[str, Any]`，*可选*）--
  用于超参数搜索的试运行或超参数字典。
- **ignore_keys_for_eval** (`list[str]`，*可选*) --
  模型输出中的键列表（如果它是字典），在以下情况下应忽略这些键：
  收集训练期间评估的预测。`~trainer_utils.TrainOutput`包含全局步数、训练损失和指标的对象。

主要培训切入点。

将保存模型，以便您可以使用`from_pretrained()`重新加载它。

只会从主进程中保存。- **commit_message** (`str`，*可选*，默认为`"End of training"`) --
  推送时要提交的消息。
- **阻塞**（`bool`，*可选*，默认为`True`）--
  函数是否仅在 `git push` 完成时返回。
- **令牌**（`str`，*可选*，默认为`None`）--
  具有写入权限的令牌，可以覆盖 Trainer 的原始参数。
- **修订**（`str`，*可选*）--
  要提交的 git 修订版本。默认为“主”分支的头部。
- **kwargs**（`dict[str, Any]`，*可选*）--
  传递到 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

### KTO 训练师
https://huggingface.co/docs/trl/v1.9.2/kto_trainer.md