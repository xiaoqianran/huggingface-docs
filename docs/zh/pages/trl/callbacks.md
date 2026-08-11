<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 回调

## RichProgressCallback[[trl.RichProgressCallback]]

使用 Rich 显示训练或评估进度的`TrainerCallback`。

## LogCompletionsCallback[[trl.LogCompletionsCallback]]

- **培训师** (`Trainer`) --
  回调将附加到的训练器。培训师的评估数据集必须包含`"prompt"`
  包含生成完成的提示的列。
- ** Generation_config ** （[GenerationConfig](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/text_generation#transformers.GenerationConfig)，*可选*）--
  用于生成完成的生成配置。
- **num_prompts** (`int`, *可选*) --
  为其生成补全的提示数。如果未提供，则默认为中的示例数
  评估数据集。
- **频率**（`int`，*可选*）--
  记录完成情况的频率。如果未提供，则默认为训练师的`eval_steps`。

一个 [TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback) 将完成情况记录到权重和偏差和/或 Comet。

用途：
```python
>>> trainer = DPOTrainer(...)
>>> completions_callback = LogCompletionsCallback(trainer=trainer)
>>> trainer.add_callback(completions_callback)
```

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
- **update_after** (`int`，*可选*，默认为`0`) --
  开始更新 BEMA 权重之前的老化时间。论文中记为 \\( \tau \\) 。
- **乘数**（`float`，*可选*，默认为`1.0`）--
  EMA 衰减因子的初始值。论文中记为\\(\gamma \\)。- **min_ema_multiplier**（`float`，*可选*，默认为`0.0`）--
  EMA 衰减因子的最小值。
- **设备**（`str`，*可选*，默认为`"cpu"`）--
  用于 BEMA 缓冲区的设备，例如`"cpu"` 或 `"cuda"`。请注意，在大多数情况下，该设备应该
  与用于训练的设备不同，以避免 OOM。

实现 [BEMA](https://huggingface.co/papers/2508.00180) 的 [TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)
（偏差校正指数移动平均线）由 [Adam Block](https://huggingface.co/abblock) 和 [Cyril
Zhang](https://huggingface.co/cyrilzhang) 计算。代码来自 https://github.com/abblock/bema，并获得 MIT 许可。

BEMA 计算模型权重，其比例如下：

$$
\theta_t' = \alpha_t \cdot (\theta_t - \theta_0) + \text{EMA}_t
$$

其中 \\( \theta_t \\) 是当前模型权重， \\( \theta_0 \\) 是模型权重的快照
第一个 `update_after` 步骤，\\( \text{EMA}_t \\) 是模型权重的指数移动平均值，并且
\\( \alpha_t \\) 是一个缩放因子，随着步数 \\( t \\) 衰减，如下所示

$$
\alpha_t = (\rho + \gamma \cdot t)^{-\eta}。
$$

EMA 计算如下：

$$
\text{EMA}_t = (1 - \beta_t) \cdot \text{EMA}_{t-1} + \beta_t \cdot \theta_t
$$

其中 \\( \beta_t \\) 是一个衰减因子，随着步数 \\( t \\) 衰减为$$
\beta_t = (\rho + \gamma \cdot t)^{-\kappa}。
$$

示例：

```python
>>> from trl import BEMACallback

>>> trainer = Trainer(..., callbacks=[BEMACallback()])
```

## WeaveCallback[[trl.WeaveCallback]]

- **培训师** (`Trainer`) --
  回调将附加到的训练器。培训师的评估数据集必须包含`"prompt"`
  包含生成完成的提示的列。
- **项目名称**（`str`，*可选*）--
  将记录数据的 Weave 项目的名称。如果未提供，将尝试使用现有的 weave 客户端
  或回退到活动 wandb 运行的项目名称。如果这些都不可用，则会引发错误。
- **记分员**（`dict[str, Callable]`，*可选*）--
  将记分器名称映射到记分器功能的字典。如果`None`，则在跟踪模式下运行（预测
  仅）。如果提供，则以评估模式运行（预测+分数+摘要）。记分员功能应
  有签名：`scorer(prompt: str, completion: str) -> float | int`
- ** Generation_config ** （[GenerationConfig](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/text_generation#transformers.GenerationConfig)，*可选*）--
  用于生成完成的生成配置。
- **num_prompts**（`int`或`None`，*可选*）--
  为其生成补全的提示数。如果未提供，则默认为示例中的示例数
  评估数据集。
- **数据集名称**（`str`，*可选*，默认为`"eval_dataset"`）--Weave 中数据集元数据的名称。
- **型号名称**（`str`，*可选*）--
  Weave 中模型元数据的名称。如果未提供，则尝试从模型配置中提取。

将跟踪和评估记录到 W&B Weave 的 [TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)。回调使用
https://weave-docs.wandb.ai/guides/evaluation/evaluation_logger/ 记录每次评估的跟踪和评估
步骤。

根据`scorers`参数支持两种模式：
- **跟踪模式**（当记分器=无时）：记录数据探索和分析的预测
- **评估模式**（当提供评分器时）：记录带有评分和摘要指标的预测

两种模式都使用Weave 的EvaluationLogger 进行结构化、一致的数据记录。

回调在评估阶段（`on_evaluate`）而不是训练步骤记录数据，使其更
高效且语义正确。它通过记录警告来优雅地处理丢失的编织安装
跳过特定于组织的功能。它还会在初始化新客户端之前检查现有的 weave 客户端。

用途：
```python
# Tracing mode (just log predictions)
trainer = DPOTrainer(...)
weave_callback = WeaveTraceCallback(trainer=trainer)  # project_name optional
trainer.add_callback(weave_callback)

# Or specify a project name
weave_callback = WeaveTraceCallback(trainer=trainer, project_name="my-llm-training")
trainer.add_callback(weave_callback)

# Evaluation mode (log predictions + scores + summary)
def accuracy_scorer(prompt: str, completion: str) -> float:
    # Your scoring logic here (metadata available via eval_attributes)
    return score

weave_callback = WeaveTraceCallback(
    trainer=trainer,
    project_name="my-llm-training",  # optional and needed only if weave client is not initialized
    scorers={"accuracy": accuracy_scorer},
)
trainer.add_callback(weave_callback)
```

训练开始时初始化 Weave。

### 迷你LLM培训师
https://huggingface.co/docs/trl/v1.9.2/minillm_trainer.md