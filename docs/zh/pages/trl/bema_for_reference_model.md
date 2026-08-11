<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 参考模型的 BEMA

此功能实现了 BEMA 算法，以在 DPO 训练期间更新参考模型。

## 用法

```python
from trl.experimental.bema_for_ref_model import BEMACallback, DPOTrainer
from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/zen", "standard_preference", split="train")

bema_callback = BEMACallback(update_ref_model=True)

trainer = DPOTrainer(
    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    train_dataset=dataset,
    callbacks=[bema_callback],
)
trainer.train()
```

## DPOTrainer[[trl.DPOTrainer]]

- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
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

## BEMACallback[[trl.BEMACallback]]- **update_freq**（`int`，*可选*，默认为`400`）--
  每 X 步更新一次 BEMA 权重。在论文中将其表示为 \\( \phi \\) 。
- **ema_power**（`float`，*可选*，默认为`0.5`）--
  EMA 衰减因子的功率。论文中记为 \\( \kappa \\) 。要禁用 EMA，请将其设置为 `0.0`。
- **bias_power**（`float`，*可选*，默认为`0.2`）--
  BEMA 比例因子的幂。论文中记为 \\( \eta \\) 。一个很大的值（例如
  `8.0`) 使 \\( \alpha_t \\) 衰减到 `0`，近似禁用偏差校正； `0.0`
  相反，每个步骤都将 \\( \alpha_t \\) 固定在 `1` 处（最大、未衰减校正）。
- **滞后**（`int`，*可选*，默认为`10`）--
  权重衰减时间表中的初始偏移，通过充当虚拟控制器来控制早期平滑度
  更新的起始年龄。论文中记为 \\( \rho \\) 。
- **update_after**（`int`，*可选*，默认为`0`）--
  开始更新 BEMA 权重之前的老化时间。论文中记为 \\( \tau \\) 。
- **乘数**（`float`，*可选*，默认为`1.0`）--
  EMA 衰减因子的初始值。论文中记为\\(\gamma \\)。- **min_ema_multiplier**（`float`，*可选*，默认为`0.0`）--
  EMA 衰减因子的最小值。
- **设备**（`str`，*可选*，默认为`"cpu"`）--
  用于 BEMA 缓冲区的设备，例如`"cpu"` 或 `"cuda"`。请注意，在大多数情况下，该设备应该
  与用于训练的设备不同，以避免 OOM。
- **update_ref_model**（`bool`，*可选*，默认为`False`）--
  是否使用 BEMA 权重更新参考模型。这将创建一个滞后的、平滑的版本
  主模型作为参考模型。
- **ref_model_update_freq** （`int`，*可选*，默认为`400`）--
  每隔这么多步就用 BEMA 权重更新参考模型。
- **ref_model_update_after** (`int`，*可选*，默认为`0`) --
  开始更新参考模型之前要等待的步骤数。

实现 [BEMA](https://huggingface.co/papers/2508.00180) 的 [TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)
（偏差校正指数移动平均线）由 [Adam Block](https://huggingface.co/abblock) 和 [Cyril
Zhang](https://huggingface.co/cyrilzhang) 计算。代码来自 https://github.com/abblock/bema，并获得 MIT 许可。

BEMA 计算模型权重，其比例如下：

$$
\theta_t' = \alpha_t \cdot (\theta_t - \theta_0) + \text{EMA}_t
$$其中 \\( \theta_t \\) 是当前模型权重， \\( \theta_0 \\) 是模型权重的快照
第一个 `update_after` 步骤，\\( \text{EMA}_t \\) 是模型权重的指数移动平均值，并且
\\( \alpha_t \\) 是一个缩放因子，随着步数 \\( t \\) 衰减，如下所示

$$
\alpha_t = (\rho + \gamma \cdot t)^{-\eta}。
$$

EMA 计算如下：

$$
\text{EMA}_t = (1 - \beta_t) \cdot \text{EMA}_{t-1} + \beta_t \cdot \theta_t
$$

其中 \\( \beta_t \\) 是一个衰减因子，随着步数 \\( t \\) 衰减为

$$
\beta_t = (\rho + \gamma \cdot t)^{-\kappa}。
$$

示例：

```python
>>> from trl import BEMACallback

>>> trainer = Trainer(..., callbacks=[BEMACallback()])
```

### vLLM 整合
https://huggingface.co/docs/trl/v1.9.2/vllm_integration.md