<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 迷你LLM培训师

[⟦T89⟧](https://huggingface.co/models?other=minillm,trl)

## 概述

TRL 支持 MiniLLM Trainer 使用反向 KLD 将大型语言模型提炼为较小的语言模型，以提高精度、质量和性能，如 [Yuxian Gu](https://huggingface.co/t1101675)、[Li Dong](https://huggingface.co/unilm)、[Furu Wei](https://huggingface.co/thegenerality) 和 Minlie Huang 的论文 [Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543) 中所述。
论文摘要如下：

> 知识蒸馏 (KD) 是一种很有前途的技术，可减少大型语言模型 (LLM) 的高计算需求。然而，之前的KD方法主要应用于白盒分类模型或训练小模型来模仿ChatGPT等黑盒模型API。如何有效地从白盒生成法学硕士中提取知识仍处于探索之中，随着法学硕士的繁荣，这一点变得越来越重要。在这项工作中，我们提出了 MiniLLM，它可以从生成的较大语言模型中提取较小的语言模型。我们首先将标准 KD 方法中的前向 Kullback-Leibler 散度（KLD）目标替换为更适合生成语言模型上的 KD 的反向 KLD，以防止学生模型高估教师分布的低概率区域。然后，我们得出一个有效的优化化方法来学习这一目标。在指令跟踪设置中的大量实验表明，MiniLLM 模型可以生成更精确的响应，具有更高的整体质量、更低的曝光偏差、更好的校准和更高的长文本生成性能。我们的方法还可以针对具有 120M 至 13B 参数的不同模型系列进行扩展。我们将在 https://aka.ms/MiniLLM 发布我们的代码和模型检查点。

这个训练后的方法是[Yuxian Gu](https://huggingface.co/t1101675)贡献的。

它是[Think Machine Lab's On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/)的通用版本，可以选择添加分布级单步蒸馏信号（如`beta=1`时的GKD）和长上下文反向KLD信号。

$$
\开始{对齐}
L_{\text{MiniLLM}}&=\alpha_1\mathbb{E}_{x\sim \pi_{\theta}}\sum_{t'=t}^{|x|}\frac{\gamma^{t'-t}}{\sum_{t'}\gamma^{t'-t}}\left[\log \frac{\pi_{\theta}(x_{t'+1}|x_{1..t'})}{\pi_{\text{老师}}(x_{t'+1}|x_{1..t'})}\right] \\
&+ \alpha_2\mathbb{E}_{x\sim \pi_{\theta}} \text{KL}\left[\pi_\theta(\cdot|x_{1..t})||\pi_{\text{老师}}(\cdot | x_{1..t})\right]。
\结束{对齐}
$$

当 \\( \alpha_1=1 \\), \\( \alpha_2=0 \\), \\( \gamma=0 \\) 时，对应

```python
from trl.experimental.minillm import MiniLLMConfig

training_args = MiniLLMConfig(
    rkl_advantage=True,
    single_step_decomposition=False,
    gamma=0.0
)
```\\( L_{\text{MiniLLM}} \\) 成为 [Tinker](https://github.com/thinking-machines-lab/tinker-cookbook/blob/5d08be6d130596b7bedd02197861c41fa81ea436/tinker_cookbook/distillation/train_on_policy.py#L88) 中实现的同策略 KD：

$$
L_{\text{tinker}}=\mathbb{E}_{x\sim \pi_{\theta}}\left[\log \frac{\pi_{\theta}(x_{t'+1}|x_{1..t'})}{\pi_{\text{teacher}}(x_{t'+1}|x_{1..t'})}\right]。
$$

当 \\( \alpha_1=0 \\), \\( \alpha_2=1 \\) 时，对应

```python
from trl.experimental.minillm import MiniLLMConfig

training_args = MiniLLMConfig(
    rkl_advantage=False,
    single_step_decomposition=True
)
```

\\( L_{\text{MiniLLM}} \\) 成为 GKD 损失的反向 KLD 版本，如 [GKD Trainer](gkd_trainer) 中所示：

$$
L_{\text{GKD-RKL}}=\mathbb{E}_{x\sim \pi_{\theta}} \text{KL}\left[\pi_\theta(\cdot|x_{1..t})||\pi_{\text{老师}}(\cdot | x_{1..t})\right]。
$$

## MiniLLMTrainer[[trl.experimental.minillm.MiniLLMTrainer]]

- **型号** (`str | PreTrainedModel`) --
  待训练的模型。可以是：- 一个字符串，是在 Huggingface.co 上的模型存储库中托管的预训练模型的 *模型 id*，或者
    包含使用保存的模型权重的*目录*的路径
    [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
    将 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 与关键字参数一起使用
    `args.model_init_kwargs`。
  - 一个[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)对象。仅支持因果语言模型。
- **教师模型** (`PreTrainedModel | nn.Module | str`) --
  用于知识蒸馏的教师模型。实例化与`model`类似。
- **reward_funcs** (`RewardFunc | list[RewardFunc]`, *可选*) --
  用于计算奖励的奖励函数。为了计算奖励，我们将所有奖励称为
  具有提示和完成功能并总结奖励。可以是：- 单一奖励函数，例如：
    - 字符串：huggingface.co 上模型存储库内托管的预训练模型的 *模型 ID*，或
    包含使用保存的模型权重的*目录*的路径
    [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
    使用 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification.from_pretrained) 与 `num_labels=1` 以及
    `args.model_init_kwargs` 中的关键字参数。
    - [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)对象：仅支持序列分类模型。
    - 自定义奖励功能：该功能提供提示和生成的完成，
      加上数据集中的任何其他列。它应该返回奖励列表。定制奖励
      当奖励不适用于这些样本时，函数也可以返回`None`。这很有用
      用于多任务训练，其中不同的奖励函数适用于不同类型的样本。当一个
      奖励函数返回样本的`None`，该奖励函数被排除在奖励之外
      该样本的计算。欲了解更多详情，请参阅[Using a custom reward
      function](#using-a-custom-reward-function)。训练者的状态也会传递给奖励函数。训练器的状态是一个实例
      [TrainerState](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerState) 可以通过访问 `trainer_state` 参数来访问
      奖励函数的签名。
  - 奖励函数列表，其中每个项目可以独立地是上述任何类型。混合不同
  列表中的类型（例如，字符串模型 ID 和自定义奖励函数）是允许的。
- **参数**（[experimental.minillm.MiniLLMConfig](/docs/trl/v1.9.2/en/minillm_trainer#trl.experimental.minillm.MiniLLMConfig)，*可选*）--
  该训练器的配置。如果`None`，则使用默认配置。
- **train_dataset** (`Dataset` 或 `IterableDataset`) --
  用于训练的数据集。它必须包含列 `"prompt"`。数据集中的任何其他列都是
  被忽略。样本的格式可以是：- [Standard](dataset_formats#standard)：每个样本都包含纯文本。
  - [Conversational](dataset_formats#conversational)：每个样本都包含结构化消息（例如，角色
    和内容）。
- **eval_dataset** (`Dataset`、`IterableDataset` 或 `dict[str, Dataset | IterableDataset]`) --
  用于评估的数据集。必须满足与`train_dataset`相同的要求。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理类用于处理数据。填充边必须设置为“左”。如果`None`，则
  处理类从带有[from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)的模型名称加载。一个
  必须设置填充令牌`tokenizer.pad_token`。如果处理类没有设置填充标记，
  `tokenizer.eos_token` 将用作默认值。
- **奖励处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)或`list[PreTrainedTokenizerBase]`，*可选*）--
  与`reward_funcs`中指定的奖励函数对应的处理类。可以是：- 单一处理类：当`reward_funcs`仅包含一个奖励函数时使用。
  - 处理类列表：必须与`reward_funcs`中奖励函数的顺序和长度相匹配。
  如果设置为`None`，或者如果对应于[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)的列表元素是
  `None`，模型的分词器自动加载使用
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)。对于`reward_funcs`中自定义奖励的元素
  函数（不是[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)），`reward_processing_classes`中的相应条目
  被忽略。
- **回调**（[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)列表，*可选*）--
  用于自定义训练循环的回调列表。将它们添加到详细的默认回调列表中
  在[here](https://huggingface.co/docs/transformers/main_classes/callback)。如果您想删除使用的默认回调之一，请使用 [remove_callback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer.remove_callback)
  方法。
- **优化器**（`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`，*可选*，默认为`(None, None)`）--
  包含要使用的优化器和调度器的元组。将默认为您的 `AdamW` 实例
  模型和由`get_linear_schedule_with_warmup`给出的调度器，由`args`控制。
- **peft_config**（`PeftConfig`，*可选*）--
  PEFT 配置用于包裹模型。如果`None`，则模型未包装。
- **rollout_func** (`RolloutFunc`，*可选*) --
  用于生成补全的函数。它必须将提示、参数和处理类作为参数
  并返回一个带有 `"prompt_ids"`、`"completion_ids"` 和 `"logprobs"` 字段的字典。任何其他字段
  被转发到奖励函数。此功能是实验性的，可能会随时更改或删除
  时间恕不另行通知。

语言模型知识蒸馏 (MiniLLM) 方法的培训师。该算法最初被提出
在论文[Knowledge Distillation of Large Language Models](https://huggingface.co/papers/2306.08543)中。

示例：

```python
>>> from datasets import load_dataset
>>> from trl.experimental.minillm import MiniLLMTrainer

>>> dataset = load_dataset("trl-lib/tldr", split="train")

>>> trainer = MiniLLMTrainer(
...     model="Qwen/Qwen3-0.6B",
...     teacher_model="Qwen/Qwen3-1.7B",
...     train_dataset=dataset,
... )
>>> trainer.train()
```- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于`True`，加载*args.output_dir* 中由前一个实例保存的最后一个检查点
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
  该函数是否仅在 `git push` 完成时返回。
- **令牌**（`str`，*可选*，默认为`None`）--
  具有写入权限的令牌，可以覆盖 Trainer 的原始参数。
- **修订**（`str`，*可选*）--
  要提交的 git 修订版本。默认为“主”分支的头部。
- **kwargs**（`dict[str, Any]`，*可选*）--
  传递给 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## MiniLLMConfig[[trl.experimental.minillm.MiniLLMConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int |无 = 无"}, {"name": "remove_unused_columns", "val": ": bool |无 = False"}, {"name": "label_names", "val": ": list[str] |无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": 字符串 |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "cast_lm_head_to_fp32", "val": ": bool = False"}, {"name": "num_ Generations", "val": ": int |无 = 8"}, {"name": "num_ Generations_eval", "val": ": int |无 = 否ne"}, {"name": "max_completion_length", "val": ": int |无 = 256"}，{"name"："ds3_gather_for_ Generation"，"val"："：bool = True"}，{"name"："shuffle_dataset"，"val"："：bool |无 = True"}, {"name": "pad_to_multiple_of", "val": ": int |无 = 无"}, {"name": " Generation_batch_size", "val": ": int |无 = 无"}, {"name": "steps_per_ Generation", "val": ": int |无 = 无"}, {"name": "温度", "val": ": float = 1.0"}, {"name": "top_p", "val": ": float = 1.0"}, {"name": "top_k", "val": ": int = 0"}, {"name": "min_p", "val": ": float |无 = 无"}, {"name": " Generation_kwargs", "val": ": dict |无 = 无"}, {"name": "chat_template_kwargs", "val": ": dict | None = None"}, {"name": "repetition_penalty", "val": ": float = 1.0"}, {"name": "cache_implementation", "val": ": str |无 = 无"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": "vllm_mode", "val": ": str = 'colocate'"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_enable_sleep_mode", “val”：“：bool = False”}，{“name”：“vllm_structed_outputs_regex”，“val”：“：str |无 = 无"}, {"name": "vllm_server_base_url", "val": ": str |无 = 无"}, {"name": "vllm_server_host", "val":": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8000"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_gpu_memory_utilization", "val": ": float = 0.3"}, {"name": "vllm_max_model_length", "val": ": int | None = None"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "beta", "val": ": float = 0.0"}, {"name": "num_iterations", "val": ": int = 1"}, {"name": "epsilon", "val": ": float = 0.2"}, {"name": "delta", "val": ": float | None = None"}, {"name": "epsilon_high", "val": ": None = None"}, {"name": "sapo_temp_neg", "val": ": float = 1.05"}, {"name": "sapo_Temperature_pos", "val": ": float = 1.0"}, {"name": "vespo_k_pos", "val": ": float = 2.0"}, {"name": "vespo_lambda_pos", "val": ": float = 3.0"}, {"name": "vespo_k_neg", "val": ": float = 3.0"}, {"name": "vespo_lambda_neg", "val": ": float = 2.0"}, {"name": "importance_sampling_level", "val": ": str = 'token'"}, {"name": "reward_weights", "val": ": list[float] | None = None"}, {"name": "multi_objective_aggregation", "val": ": str = 'sum_then_normalize'"}, {"name": "scale_rewards", "val": ": str = 'group'"}, {"name": "loss_type", "val": ": str = 'dapo'"}, {"name": "mask_truncated_completions", "val": ": bool = False"}, {"name": "sync_ref_model", "val": ": bool = False"}, {"name": "ref_model_mixup_alpha", "val": ": float = 0.6"}, {"name": "ref_model_sync_steps", "val": ": int = 512"}, {"name": "top_entropy_quantile", "val": ": float = 1.0"}, {"name": "entropy_coef", "val": ": float = 0.0"}, {"name": "use_adaptive_entropy", "val": ": bool = False"}, {"name": "entropy_coef_min", "val": ": float = 0.0"}, {"name": "entropy_coef_max", "val": ": float = 1.0"}, {"name": "entropy_coef_delta", "val": ": float = 0.005"}, {"name": "entropy_target", "val": ": float = 0.2"}, {"name": "max_tool_calling_iterations", "val": ": int |无 = 无"}, {"name": "vllm_importance_sampling_ Correction", "val": ": bool = True"}, {"name": "vllm_importance_sampling_mode", "val": ": str = 'sequence_mask'"}, {"name": "vllm_importance_sampling_clip_max", "val": ": float |无 = 3.0"}, {"name": "vllm_importance_sampling_clip_min", "val": ": float |无 = 无"}, {"name": "off_policy_mask_threshold", "val": ": float |无 = 无"}, {"name": "use_bias_ Correction_kl", "val": ": bool = False"}, {"name": "log_completions", "val": ": bool = False"}, {"name": "log_multimodal", "val": ": bool = True"}, {"name": "num_completions_to_print", "val": ": int |无 = 无"}, {"name": "log_unique_prompts", "val": ": bool = False"}, {"name": "log_completions_hub_repo", "val": ": str |无 = 无"}, {"name": "use_transformers_continuous_batching", "val": ": bool = False"}, {"name": "transformers_continuous_batching_config", "val": ": dict |无 = 无"}, {"name": "use_transformers_paged", "val": ": bool = False"}, {"name": "vllm_importance_sampling_cap", "val": ": float |无 = 无"}, {"name": "teacher_model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "rkl_advantage", "val": ": bool = True"}, {"name": "single_step_decomposition", "val": ": bool = True"}, {"name": "kd_Temperature", "val": ": float = 1.0"}, {"name": "gamma", "val": ": float = 0.0"}, {"name": "length_normalization", "val": ": bool = True"}]}>
- **teacher_model_init_kwargs**（`dict[str, Any]`，*可选*）--
  实例化教师模型时传递给 `AutoModelForCausalLM.from_pretrained` 的关键字参数
  来自字符串。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型中禁用 dropout。- **rkl_advantage**（`bool`，*可选*，默认为`True`）--
  是否将反向KL优势添加到奖励优势中。
- **单步分解**（`bool`，*可选*，默认为`True`）--
  是否使用单步分解进行 KL 散度计算。
- **kd_温度**（`float`，*可选*，默认为`1.0`）--
  知识蒸馏的温度。较高的温度会产生较软的概率分布
  类。
- **伽玛**（`float`，*可选*，默认为`0.0`）--
  强化学习中未来奖励的折扣因子。
- **长度标准化**（`bool`，*可选*，默认为`True`）--
  是否对奖励应用长度标准化。

`MiniLLMTrainer`的配置类。

此类仅包含特定于 MiniLLM 训练的参数。获取完整的培训列表
参数请参考[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)和[GRPOConfig](/docs/trl/v1.9.2/en/grpo_trainer#trl.GRPOConfig)文档。

### Liger 内核集成
https://huggingface.co/docs/trl/v1.9.2/liger_kernel_integration.md