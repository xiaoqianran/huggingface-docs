<!-- huggingface-docs: machine-translated zh-CN from English source -->

# A2PO

[⟦T87⟧](https://huggingface.co/models?other=a2po,trl)

TRL 支持 A\*-PO（最优优势回归），如 Kianté Brantley、Mingyu Chen、Zhaolin Gau、Jason D. Lee、Wen Sun、Wenhao Zhan 和 Xu Zhou Zhu 的论文 [Accelerating RL for LLM Reasoning with Optimal Advantage Regression](https://huggingface.co/papers/2505.20686) 中所述。

论文摘要如下：

> 强化学习 (RL) 已成为微调大型语言模型 (LLM) 以提高复杂推理能力的强大工具。然而，最先进的策略优化方法通常会遭受高计算开销和内存消耗，这主要是由于每个提示需要多代以及对批评者网络或当前策略的优势估计的依赖。在本文中，我们提出了 A\*-PO，一种新颖的两阶段策略优化框架，它直接逼近最优优势函数，并能够有效训练 LLM 的推理任务。在第一阶段，我们利用参考策略的离线采样来估计最优价值函数 V\*，从而消除了昂贵的在线价值估计的需要。在第二阶段，我们使用简单的最小二乘回归损失来执行策略更新，每个提示仅生成一次。理论上，我们建立性能保证并证明 KL 正则化 RL 目标可以在不需要复杂的探索策略的情况下进行优化。根据经验，与 PPO、GRPO 和 REBEL 相比，A\*-PO 在各种数学推理基准测试中实现了具有竞争力的性能，同时将训练时间缩短了 2 倍，峰值内存使用量降低了 30% 以上。

## 用法

A\*-PO 假设一个**二元的、可验证的奖励** (`r ∈ {0, 1}`) 并分两个阶段运行：

1. **离线价值估计。** 在训练之前，从每个提示的参考策略中采样`num_value_samples`完成情况，并使用`reward_funcs`进行评分。根据提示估计并缓存最佳值 `V*(x) = β₁·log(mean_i exp(r(x, yᵢ)/β₁))`。
2. **在策略回归。** 在训练期间，根据当前策略的每个提示生成一个完成。损失是隐性奖励`β₂·log(π(y|x)/π_ref(y|x))`和最优优势`r(x, y) − V*(x)`之间的平方误差。

```python
from trl.experimental.a2po import A2POConfig, A2POTrainer

# A*-PO assumes a binary, verifiable reward in {0, 1}.
def reward_correct(completions, ground_truth, **kwargs):
    return [float(completion.strip() == truth) for completion, truth in zip(completions, ground_truth)]

training_args = A2POConfig(
    output_dir="Qwen2.5-0.5B-A2PO",
    num_value_samples=8,  # Stage 1: samples per prompt from the reference policy to estimate V*
    beta1=0.5,  # Stage 1: KL temperature for the V* estimate
    beta2=1e-3,  # Stage 2: KL temperature for the regression target
)
trainer = A2POTrainer(
    model="Qwen/Qwen2.5-0.5B",
    reward_funcs=reward_correct,
    args=training_args,
    train_dataset=...,
)
trainer.train()
```

由于 `V*` 完全根据参考策略样本估计，因此 A\*-PO 不能超过参考策略的 Pass@K。官方实现可以在[ZhaolinGao/A-PO](https://github.com/ZhaolinGao/A-PO)找到。

## A2POTrainer[[trl.experimental.a2po.A2POTrainer]]- **型号**（`PreTrainedModel`或`str`）--
  要训练的模型，或传递给的模型标识符（字符串）
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained)。
- **reward_funcs** (`Callable` 或 `list[Callable]`) --
  奖励功能。每个都采用 `prompts` 和 `completions` （加上数据集列作为关键字参数）并且
  返回浮动奖励列表。当提供多个时，它们的加权和（参见
  `A2POConfig.reward_weights`）是标量奖励`r`，A*-PO 假设它是二进制的（在`{0, 1}`中）。
- **参数**（`A2POConfig`，*可选*）--
  该训练器的配置。如果`None`，则使用默认配置。
- **train_dataset**（`Dataset`，*可选*）--
  训练数据集。必须包含 `"prompt"` 列。
- **eval_dataset**（`Dataset`，*可选*）--
  评估数据集。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，*可选*）--
  处理类用于处理数据。如果`None`，则从模型名称加载
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)。
- **回调**（`list[~transformers.TrainerCallback]`，*可选*）--
  用于自定义训练循环的回调列表。
- **优化器**（`tuple[~torch.optim.Optimizer, ~torch.optim.lr_scheduler.LambdaLR]`，*可选*，默认为`(None, None)`）--
  包含优化器和学习率调度器的元组。

A*-PO（最优优势回归）方法的训练器，在[Accelerating RL for LLM Reasoning with
Optimal Advantage Regression](https://huggingface.co/papers/2505.20686)中引入。

A*-PO 分两个阶段运行：1. **离线值估计。** 在训练之前，从参考中采样`num_value_samples`完成值
   每个训练提示的策略并得分`reward_funcs`。最优值估计为
   `V*(x) = beta1 * log(mean_i exp(r(x, y_i) / beta1))` 并根据提示进行缓存。
2. **在策略回归。** 在训练期间，根据当前策略的每个提示生成一个完成。
   损失是隐含奖励`beta2 * log(pi(y|x) / pi_ref(y|x))`与最优奖励之间的平方误差
   优势估计`r(x, y) - V*(x)`。

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

## A2POConfig[[trl.experimental.a2po.A2POConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int |无 = 无"}, {"name": "remove_unused_columns", "val": ": bool = False"}, {"name": "label_names", "val": ": list[str] |无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": 字符串 |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "max_prompt_length", "val": ": int |无 = 512"}, {"name": "max_completion_length", "val": ": int |无 = 256"}, {"name": "温度", "val": ": float = 1.0"}, {"name": "top_p", "val": ": float = 1.0"}, {"name": "top_k", "val": ": int |无 = 无"}, {"name": "num_value_samples", "val": ": int = 8"},{"name": "beta1", "val": ": float = 0.5"}, {"name": "filter_all_in Correct", "val": ": bool = True"}, {"name": "beta2", "val": ": float = 0.001"}, {"name": "reward_weights", "val": ": list[float] | None = None"}]}>
控制模型和参考模型的参数

- **model_init_kwargs**（`dict[str, Any]`，*可选*）--
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 的关键字参数，在 `model` 时使用
  `A2POTrainer` 的参数以字符串形式提供。
- **trust_remote_code**（`bool`，*可选*，默认为`False`）--
  是否允许加载从 Hub 发送自定义 Python 代码的模型和标记器。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 和 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)。

控制数据预处理的参数

- **remove_unused_columns** (`bool`，*可选*，默认为`False`) --
  是否仅保留数据集中`"prompt"`列。如果您使用自定义奖励函数
  需要`"prompts"`和`"completions"`以外的任何列，您应该将其保留为`False`。

控制生成的参数- **max_prompt_length**（`int`或`None`，*可选*，默认为`512`）--
  提示的最大长度。如果提示符长于此长度，则会向左截断。
- **max_completion_length** （`int` 或 `None`，*可选*，默认为 `256`）--
  生成的补全的最大长度。
- **温度**（`float`，*可选*，默认为`1.0`）--
  采样温度，用于第一阶段和第二阶段生成。
- **top_p** (`float`，*可选*，默认为`1.0`) --
  控制要考虑的顶级令牌的累积概率的浮点数。必须在 (0, 1] 中。设置为
  `1.0` 考虑所有代币。
- **top_k** （`int` 或 `None`，*可选*）--
  要保留的最高概率词汇标记的数量。如果`None`，则禁用top-k过滤。

控制第 1 阶段的参数（离线最优值估计）- **num_value_samples**（`int`，*可选*，默认为`8`）--
  每个提示从参考政策中抽取的样本数，用于估计 `V*`。
- **beta1**（`float`，*可选*，默认为`0.5`）--
  吉隆坡温度用于估算第一阶段中的`V*`。
- **filter_all_in Correct** (`bool`，*可选*，默认为`True`) --
  是否删除所有参考样本都不正确的提示。

控制第 2 阶段的参数（策略回归）

- **beta2**（`float`，*可选*，默认为`1e-3`）--
  第 2 阶段回归目标中使用的 KL 温度。
- **奖励权重**（`list[float]`，*可选*）--
  每个奖励函数的权重。必须与奖励函数的数量相匹配。如果`None`，所有奖励都是
  与重量 `1.0` 同等加权。

`A2POTrainer`的配置类。

该类仅包含特定于 A2PO 训练的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

### 带有重播缓冲区的 GRPO
https://huggingface.co/docs/trl/v1.9.2/grpo_with_replay_buffer.md