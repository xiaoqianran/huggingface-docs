<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PPO 培训师

[⟦T223⟧](https://huggingface.co/models?other=ppo,trl)

TRL 支持使用 [Proximal Policy Optimization (PPO)](https://huggingface.co/papers/1707.06347) 培训法学硕士。

参考文献：

- [Fine-Tuning Language Models from Human Preferences](https://github.com/openai/lm-human-preferences)
- [Learning to Summarize from Human Feedback](https://github.com/openai/summarize-from-feedback)
- [The N Implementation Details of RLHF with PPO](https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo)
- [The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031)

## 开始吧

要仅运行 PPO 脚本以确保训练器可以运行，您可以运行以下命令来使用虚拟奖励模型训练 PPO 模型。

```bash
python examples/scripts/ppo/ppo.py \
    --dataset_name trl-internal-testing/descriptiveness-sentiment-trl-style \
    --dataset_train_split descriptiveness \
    --learning_rate 3e-6 \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --output_dir models/minimal/ppo \
    --per_device_train_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --total_episodes 10000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --reward_model_path EleutherAI/pythia-1b-deduped \
    --missing_eos_penalty 1.0
```

## 记录的指标的解释

记录的指标如下。这是一个例子[tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35)

- `eps`：跟踪每秒的剧集数。
- `objective/kl`：当前政策和参考政策之间的平均 Kullback-Leibler (KL) 差异。
- `objective/entropy`：采样推出时的平均令牌级熵代理，在 PPO 优化之前计算为 `(-logprobs).sum(1).mean()`。
- `objective/non_score_reward`：来自非分数相关来源的平均奖励，基本上是`beta * kl.sum(1)`，其中`beta`是KL惩罚系数，`kl`是每个代币的KL散度。
- `objective/rlhf_reward`：RLHF 平均奖励，即`score - non_score_reward`。
- `objective/scores`：奖励模型/环境返回的平均分数。
- `policy/approxkl_avg`：连续 PPO 政策之间的平均近似 KL 散度。请注意，这与 `objective/kl` 不同。
- `policy/clipfrac_avg`：被剪裁的策略更新的平均比例，指示限制策略更新以防止发生较大变化的频率。- `loss/policy_avg`：平均保单损失，表明保单的执行情况。
- `loss/value_avg`：平均值损失，表示预测值与实际奖励之间的差异。
- `val/clipfrac_avg`：被剪裁的价值函数更新的平均比例，类似于policy/clipfrac_avg，但针对的是价值函数。
- `policy/entropy_avg`：PPO 优化期间当前策略的平均分类熵，根据每个小批量的`logits` 计算，并在 PPO 时期/小批量之间取平均值。
- `val/ratio`：当前政策概率与旧政策概率的平均比率，提供政策变化程度的衡量标准。
- `val/ratio_var`：`val/ratio`的方差，表明政策变化的可变性。
- `val/num_eos_tokens`：生成的结束序列（EOS）令牌的数量，可以指示完整响应的数量。
- `lr`: lr：优化器当前使用的学习率。
- `episode`：episode：训练过程中当前的episode计数。

`objective/entropy` 和 `policy/entropy_avg` 是故意不同的信号：- `objective/entropy` 是根据用于构建当前更新的 PPO 优势的采样行为策略部署来测量的。
- `policy/entropy_avg`是在PPO优化过程中对演化策略（梯度步骤之后）进行测量的，因此它可能与`objective/entropy`不同。

## 食谱

- 调试TIP：`objective/rlhf_reward`：这是RLHF训练的最终目标。如果培训按预期进行，该指标应该会继续上升。
- 调试提示：`val/ratio`：这个数字应该在 1.0 左右浮动，并且它会因 PPO 的代理损失而被 `--cliprange 0.2` 剪裁。因此，如果这个`ratio`太高（如2.0或1000.0）或太小（如0.1），则意味着连续策略之间的更新太剧烈。您应该尝试了解为什么会发生这种情况并尝试修复它。
- 内存提示：如果内存不足，可以尝试减少`--per_device_train_batch_size`或增加`--gradient_accumulation_steps`来减少内存占用。
- 内存提示：如果您有多个 GPU，您还可以使用 DeepSpeed stage 3 运行训练以减少内存占用`accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml`。- 使用提示：我们建议通过`--missing_eos_penalty`使用“EOS技巧”，它会从不以EOS代币结尾的完成分数中减去静态标量惩罚。这可以帮助模型学习生成更连贯的补全。

## 我的模型到底在做什么？

为了帮助您了解模型正在做什么，我们会定期记录模型的一些示例完成情况。这是一个完成的例子。在示例[tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/dd2o3g35)中，如下所示，让您可以看到模型在训练的不同阶段的响应。默认情况下，我们在训练期间生成`--num_sample_generations 10`，但您可以自定义生成数。

![ppov2_completions](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/ppov2_completions.gif)

在日志中，采样的代看起来像

```txt
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┓
┃ query                           ┃ model response                  ┃ score    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━┩
│  SUBREDDIT: r/AskReddit         │  I'm in love with a friend, and │ 3.921875 │
│                                 │ I don't know how to get rid of  │          │
│ TITLE: How do you get someone   │ those feelings. I'm             │          │
│ out of your head?               │ desperate.<|endoftext|>[PAD][P… │          │
│                                 │                                 │          │
│ POST: Hi,                       │                                 │          │
│ I'm 22, and I have been with my │                                 │          │
│ girlfriend for 5 years now. We  │                                 │          │
│ recently moved together. We've  │                                 │          │
│ always loved each other         │                                 │          │
│ intensely.                      │                                 │          │
│                                 │                                 │          │
│ Problem, I recently started to  │                                 │          │
│ have feelings for an other      │                                 │          │
│ person (a friend). This person  │                                 │          │
│ has had a boyfriend for now 3   │                                 │          │
│ years, and has absolutely no    │                                 │          │
│ ideas. Those feelings were so   │                                 │          │
│ strong, it was hard to hide     │                                 │          │
│ them. After 2 months of me      │                                 │          │
│ being distant and really sad,   │                                 │          │
│ my girlfriend forced me to say  │                                 │          │
│ what was bothering me. I'm not  │                                 │          │
│ a good liar, and now she knows. │                                 │          │
│                                 │                                 │          │
│ We decided to give us a week    │                                 │          │
│ alone, I went to my parents.    │                                 │          │
│                                 │                                 │          │
│ Now, I'm completely lost. I     │                                 │          │
│ keep on thinking about this     │                                 │          │
│ person, and I hate that. I      │                                 │          │
│ would like for those feelings   │                                 │          │
│ to go away, to leave me alone.  │                                 │          │
│ But I can't.                    │                                 │          │
│                                 │                                 │          │
│ What do I do? It's been 3       │                                 │          │
│ months now, and I'm just        │                                 │          │
│ desperate.                      │                                 │          │
│                                 │                                 │          │
│ TL;DR:                          │                                 │          │
├─────────────────────────────────┼─────────────────────────────────┼──────────┤
│  SUBREDDIT: r/pettyrevenge      │  My mom woke me up with a loud  │ 6.84375  │
│                                 │ TV. I blasted Gangnam Style on  │          │
│ TITLE: So, my mom woke me up    │ repeat, with the bass cranked   │          │
│ with a loud TV.                 │ up as high as it could          │          │
│                                 │ go.<|endoftext|>[PAD][PAD][PAD… │          │
│ POST: She was in her living     │                                 │          │
│ room, watching TV. This was at  │                                 │          │
│ about 8:30 in the morning, and  │                                 │          │
│ she was exercising. She turned  │                                 │          │
│ the TV up extra loud to hear it │                                 │          │
│ over her excercycle, and woke   │                                 │          │
│ me up. I went in there asking   │                                 │          │
│ for her to turn it down. She    │                                 │          │
│ said she didn't have to; I      │                                 │          │
│ explained that I always used    │                                 │          │
│ headphones so she didn't have   │                                 │          │
│ to deal with my noise and that  │                                 │          │
│ she should give me a little     │                                 │          │
│ more respect, given that I paid │                                 │          │
│ rent at the time.               │                                 │          │
│                                 │                                 │          │
│ She disagreed. I went back to   │                                 │          │
│ my room, rather pissed off at   │                                 │          │
│ the lack of equality. I had no  │                                 │          │
│ lock on my door; but I had a    │                                 │          │
│ dresser right next to it, so I  │                                 │          │
│ pulled one of the drawers out   │                                 │          │
│ enough so that it caused the    │                                 │          │
│ door to not be openable. Then,  │                                 │          │
│ I turned my speakers up really  │                                 │          │
│ loud and blasted Gangnam Style  │                                 │          │
│ on repeat, with the bass        │                                 │          │
│ cranked up as high as it could  │                                 │          │
│ go.                             │                                 │          │
│                                 │                                 │          │
│ If you hate Gangnam Style for   │                                 │          │
│ being overplayed, you will see  │                                 │          │
│ why I chose that particular     │                                 │          │
│ song. I personally don't mind   │                                 │          │
│ it. But here's the thing about  │                                 │          │
│ my bass; it vibrates the walls, │                                 │          │
│ making one hell of a lot of     │                                 │          │
│ noise. Needless to say, my mom  │                                 │          │
│ was not pleased and shut off    │                                 │          │
│ the internet. But it was oh so  │                                 │          │
│ worth it.                       │                                 │          │
│                                 │                                 │          │
│ TL;DR:                          │                                 │          │
└─────────────────────────────────┴─────────────────────────────────┴──────────┘
```

## 实施细节

此 PPO 实现基于[The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031)。

## 基准实验

为了验证 PPO 实现的效果，我们在 1B 模型上进行了实验。这是我们用来运行实验的命令。我们直接从[The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031)获取SFT / RM模型。

```shell
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/ppo/ppo_tldr.py \
    --dataset_name trl-lib/tldr \
    --dataset_test_split validation \
    --output_dir models/minimal/ppo_tldr \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --missing_eos_penalty 1.0 \
    --stop_token eos \
    --eval_strategy steps \
    --eval_steps 100
```

检查点和实验跟踪可在以下位置获得：

- [🤗 Model checkpoint](https://huggingface.co/trl-lib/ppo_tldr)
- [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/runs/dd2o3g35)PPO 检查点获得 64.7% 的首选率，而 SFT 检查点的首选率为 33.0%（以 GPT-4o mini 作为判断者进行评估）。这是一个好兆头，表明 PPO 培训正在按预期进行。

指标：

![PPO v2](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/ppov2.png)

```bash
# pip install openrlbenchmark==0.2.1a5
# see https://github.com/openrlbenchmark/openrlbenchmark#get-started for documentation
# to use it, change `?we=huggingface&wpn=trl` to your own project and `?tag=pr-1540` to your own tag
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=huggingface&wpn=trl&xaxis=train/episode&ceik=output_dir&cen=sft_model_path&metrics=train/objective/rlhf_reward&metrics=train/objective/scores&metrics=train/objective/kl&metrics=train/objective/non_score_reward&metrics=train/objective/entropy&metrics=train/policy/approxkl_avg&metrics=train/policy/clipfrac_avg&metrics=train/loss/policy_avg&metrics=train/loss/value_avg&metrics=train/val/clipfrac_avg&metrics=train/policy/entropy_avg&metrics=train/val/ratio&metrics=train/val/ratio_var&metrics=train/val/num_eos_tokens&metrics=train/lr&metrics=train/eps' \
        "cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr?tag=pr-1540" \
    --env-ids models/minimal/ppo_tldr \
    --pc.ncols 4 \
    --pc.ncols-legend 1 \
    --pc.xlabel "Episode" \
    --output-filename benchmark/trl/pr-1540/ppo \
    --scan-history
```

## PPOTrainer[[trl.experimental.ppo.PPOTrainer]]

- **参数** ([experimental.ppo.PPOConfig](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOConfig)) --
  训练论证。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)、[BaseImageProcessor](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/image_processor#transformers.BaseImageProcessor)、[FeatureExtractionMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin) 或 [ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)）--
  处理数据的类。
- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  待训练的模型。这就是政策模型。
- **参考模型**（[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)，*可选*）--
  用于计算 KL 散度的参考模型。如果`None`，则创建策略模型的副本。
- **奖励模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  用于计算奖励的奖励模型。
- **train_dataset** (`Dataset`) --
  用于训练的数据集。
- **值模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  价值模型用于预测状态的价值。
- **data_collator**（[DataCollatorWithPadding](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/data_collator#transformers.DataCollatorWithPadding)，*可选*）--
  数据整理器用于批处理和填充数据集中的样本。如果`None`，则创建默认数据整理器
  使用`processing_class`。
- **eval_dataset** （`Dataset` 或 `Dataset` 的 `dict`，*可选*）--
  用于评估的数据集。
- **优化器**（`torch.optim.Optimizer`和`torch.optim.lr_scheduler.LambdaLR`的`tuple`，*可选*，默认为`(None, None)`）--包含用于训练的优化器和学习率调度器的元组。如果`None`，则
  优化器和学习率调度器是使用以下命令创建的
  [create_optimizer_and_scheduler](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer.create_optimizer_and_scheduler)方法。
- **回调**（[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)的`list`，*可选*）--
  训练期间使用的回调。
- **peft_config**（`PeftConfig`，*可选*）--
  PEFT 配置使用 PEFT 进行训练。如果`None`，则不使用PEFT。如果提供，则政策`model`
  将用指定的 PEFT 适配器包裹。
近端策略优化 (PPO) 培训师。

有关PPO的详细信息，请参阅论文：[Proximal Policy Optimization
Algorithms](https://huggingface.co/papers/1707.06347)。- **commit_message** (`str`，*可选*，默认为`"End of training"`) --
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

## PPOConfig[[trl.experimental.ppo.PPOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}，{"name"："local_rank"，"val"："：int = -1"}，{"name"："dataset_num_proc"，"val"："：int |无 = 无"}, {"name": "num_mini_batches", "val": ": int = 1"}, {"name": "total_episodes", "val": ": int |无 = 无"}, {"name": "local_rollout_forward_batch_size", "val": ": int = 64"}, {"name": "num_sample_ Generations", "val": ": int = 10"}, {"name": "response_length", "val": ": int = 53"}, {"name": "stop_token", "val": ": Typing.Optional[typing.Literal['eos']] = None"}, {"name": "stop_token_id", "val": ": int |无 = 无"}, {"name": "温度", "val": ": float = 0.7"}, {"name": "missing_eos_penalty", "val": ": float |无 = 无"}, {"name": "sft_model_path", "val": ": str = 'EleutherAI/pythia-160m'"}, {"name": "world_size", "val": ": int |无 = 无"}, {"name": "num_total_batches", "val": ": int |无 = 无"}, {"name": "micro_batch_size", "val": ": int |无 = 无"}, {"name": "local_batch_size", "val": ": int | None = None"}, {"name": "batch_size", "val": ": int |无 = 无"}, {"name": "local_mini_batch_size", "val": ": int |无 = 无"}, {"name": "mini_batch_size", "val": ": int |无 = 无"}, {"name": "reward_model_path", "val": ": str = 'EleutherAI/pythia-160m'"}, {"name": "model_adapter_name", "val": ": str |无 = 无"}, {"name": "ref_adapter_name", "val": ": str | None = None"}, {"name": "num_ppo_epochs", "val": ": int = 4"}, {"name": "white_rewards", "val": ": bool = False"}, {"name": "kl_coef", "val": ": float = 0.05"}, {"name": "kl_estimator", "val": ": Typing.Literal['k1', 'k3'] = 'k1'"}, {"name": "cliprange", "val": ": float = 0.2"}, {"name": "vf_coef", "val": ": float = 0.1"}, {"name": "cliprange_value", "val": ": float = 0.2"}, {"name": "gamma", "val": ": float = 1.0"}, {"name": "lam", "val": ": float = 0.95"}, {"name": "ds3_gather_for_ Generation", "val": ": bool = True"}]}>
- **dataset_num_proc** (`int`，*可选*) --
  用于处理数据集的进程数。
- **num_mini_batches**（`int`，*可选*，默认为`1`）--
  将批次拆分成的小批次数。
- **总集数**（`int`，*可选*）--
  数据集中的剧集总数。
- **local_rollout_forward_batch_size** (`int`，*可选*，默认为`64`) --
  在推出阶段，每个排名都没有毕业生向前传递。
- **num_sample_ Generations** （`int`，*可选*，默认为`10`）--
  整个训练过程中调试样本生成的数量（即`generate_completions`调用）。
- **response_length**（`int`，*可选*，默认为`53`）--
  响应的长度。
- **stop_token**（`str`，*可选*）--
  指定用于文本生成的停止标记。该参数与
  `stop_token_id`。

  - `None`：不应用停止标记，除非指定`stop_token_id`。
  - `'eos'`：使用分词器的`eos_token`。- **stop_token_id** (`int`，*可选*) --
  指定用于文本生成的停止标记的 ID。如果`None`，不应用停止令牌ID，
  除非指定`stop_token`。该参数与`stop_token`互斥。
- **温度**（`float`，*可选*，默认为`0.7`）--
  取样温度。
- **missing_eos_penalty** (`float`，*可选*) --
  当模型无法生成 EOS 代币时，会对分数进行处罚。这有助于鼓励
  生成短于最大长度 (`max_new_tokens`) 的补全。惩罚必须是积极的
  值。
- **sft_model_path**（`str`，*可选*，默认为`"EleutherAI/pythia-160m"`）--
  SFT 模型的路径。
- **世界大小**（`int`，*可选*）--
  用于训练的进程 (GPU) 数量。
- **总批次数**（`int`，*可选*）--
  训练的总批次数。
- **micro_batch_size**（`int`，*可选*）--
  跨设备的微批量大小（HF 的`per_device_train_batch_size` * `world_size`）。
- **local_batch_size**（`int`，*可选*）--
  每个 GPU 的批量大小（HF 的 `per_device_train_batch_size` * `gradient_accumulation_steps`）。
- **batch_size** (`int`, *可选*) --
  跨设备的批量大小（HF 的 `per_device_train_batch_size` * `world_size` *
  `gradient_accumulation_steps`）。
- **local_mini_batch_size** (`int`，*可选*) --每个 GPU 的最小批量大小。
- **mini_batch_size**（`int`，*可选*）--
  跨 GPU 的最小批量大小。
- **push_to_hub**（`bool`，*可选*，默认为`False`）--
  训练后是否将模型推送到Hub。
- **reward_model_path**（`str`，*可选*，默认为`"EleutherAI/pythia-160m"`）--
  奖励模型的路径。
- **型号适配器名称**（`str`，*可选*）--
  当 LoRA 与多个适配器一起使用时，训练目标 PEFT 适配器的名称。
- **ref_adapter_name**（`str`，*可选*）--
  当 LoRA 与多个适配器一起使用时，参考 PEFT 适配器的名称。
- **num_ppo_epochs** (`int`，*可选*，默认为`4`) --
  训练的纪元数。
- **whiten_rewards**（`bool`，*可选*，默认为`False`）--
  是否对奖励进行美化。
- **kl_coef** (`float`，*可选*，默认为`0.05`) --
  KL系数。
- **kl_estimator**（`Literal["k1", "k3"]`，*可选*，默认为`"k1"`）--
  使用 [Approximating KL
  Divergence](http://joschu.net/blog/kl-approx.html) 中的哪个 KL 散度估计器。默认为“k1”，一个简单、无偏见的
  估计器。可以设置为“k3”，一个具有较低方差的无偏估计量，“似乎是一个严格的
  更好的估计器”。无法设置为“k2”，因为它用于记录目的。- **剪辑范围**（`float`，*可选*，默认为`0.2`）--
  剪辑范围。
- **vf_coef**（`float`，*可选*，默认为`0.1`）--
  值函数系数。
- **cliprange_value**（`float`，*可选*，默认为`0.2`）--
  值函数的剪辑范围。
- **伽玛**（`float`，*可选*，默认为`1.0`）--
  折扣系数。
- **lam** (`float`，*可选*，默认为`0.95`) --
  GAE 的 Lambda 值。
- **ds3_gather_for_ Generation**（`bool`，*可选*，默认为`True`）--
  此设置适用于 DeepSpeed ZeRO-3。如果启用，则会收集策略模型权重以进行生成，
  提高生成速度。但是，禁用此选项允许训练超出 VRAM 的模型
  单个 GPU 的容量，尽管代价是生成速度较慢。

[experimental.ppo.PPOTrainer](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOTrainer) 的配置类。

此类仅包含特定于 PPO 训练的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以将这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在上指定的参数
命令行。> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`3e-6`，而不是`5e-5`。

## PreTrainedModelWrapper[[trl.experimental.ppo.PreTrainedModelWrapper]]

- **预训练模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要包裹的模型。
- **父类** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要包装的模型的父类。
- **supported_args** (`list`) --
  包装类支持的参数列表。

作为标准 PyTorch `torch.nn.Module` 实现的 [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 的包装器。

该类提供了一个兼容层，保留了原始的关键属性和方法
[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)，同时公开与 PyTorch 模块一致的统一接口。它使
将预训练的 Transformer 模型无缝集成到自定义训练、评估或推理工作流程中。添加并加载奖励建模适配器。仅当型号为 `PeftModel` 并且您
使用 `reward_modeling_adapter_id` 参数初始化模型，指向奖励的 id
建模适配器。最新还需要包含分数头才能产生奖励。

计算给定输入的奖励分数。该方法首先启用适配器，然后计算
奖励分数。之后，模型禁用奖励建模适配器并启用默认的 ppo 适配器
再次。

- **预训练模型名称或路径** (`str` 或 [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  预训练模型的路径或其名称。
- ***model_args**（`list`，*可选*）--
  附加位置参数传递给底层模型的 `from_pretrained` 方法。
- **kwargs**（`dict`，*可选*）--
  传递给底层模型的 `from_pretrained` 方法的其他关键字参数。我们也
  预处理 kwargs 以提取特定于该参数的参数
  [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 类和特定于 trl 模型的参数。夸格人
  还支持来自 `peft` 库的 `prepare_model_for_kbit_training` 参数。从 `transformers` 的预训练模型实例化一个新模型。使用以下方法加载预训练模型
`from_pretrained` [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 类的方法。特定于的论点
[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 类沿着此方法传递并从 `kwargs` 中过滤掉
论点。

后初始化方法。在模型实例化并从检查点加载后调用此方法。
它可用于执行其他操作，例如加载 state_dict。

- ***args** (`list`, *可选*) --
  位置参数传递给底层模型的 `push_to_hub` 方法。
- **kwargs**（`dict`，*可选*）--
  关键字参数传递给底层模型的 `push_to_hub` 方法。

将预训练模型推送到集线器。这个方法是一个包装器
[push_to_hub](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub)。请参考文档
[push_to_hub](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.push_to_hub) 了解更多信息。

- ***args** (`list`, *可选*) --
  位置参数传递给底层模型的 `save_pretrained` 方法。
- **kwargs**（`dict`，*可选*）--
  关键字参数传递给底层模型的 `save_pretrained` 方法。将预训练模型保存到目录中。这个方法是一个包装器
[save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)。请参考文档
[save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) 了解更多信息。

返回预训练模型的state_dict。

## AutoModelForCausalLMWithValueHead[[trl.experimental.ppo.AutoModelForCausalLMWithValueHead]]

除了语言模型头之外还具有值头的自回归模型。这个类继承自
[experimental.ppo.PreTrainedModelWrapper](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PreTrainedModelWrapper) 并包装一个 [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 类。包装类
支持`from_pretrained`、`push_to_hub`、`generate`等经典功能。调用包装的方法
模型，只需操作该类的 `pretrained_model` 属性即可。类属性：
- **transformers_parent_class** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 包装模型的父类。
  这个
  对于此类，应设置为 `transformers.AutoModelForCausalLM`。
- **supported_args** (`tuple`) -- 用于标识支持的参数的字符串元组
  由`ValueHead`类。目前，支持的参数有：
  - **summary_dropout_prob** (`float`, `optional`, 默认为 `None`) -- 的 dropout 概率
    `ValueHead`类。
  - **v_head_initializer_range** (`float`, `optional`, 默认为 `0.2`) -- 初始化范围
    `ValueHead` 如果选择了特定的初始化策略。
  - **v_head_init_strategy** (`str`, `optional`, 默认为`None`) -- 初始化策略
    `ValueHead`。目前，支持的策略有：
    - **`None`** -- 使用随机分布初始化 `ValueHead` 的权重。这是
      默认策略。
    - **“正态”** -- 使用正态分布初始化 `ValueHead` 的权重。- **预训练模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要包装的模型。应该是GPT2这样的因果语言模型。或映射到的任何模型
  `AutoModelForCausalLM`类。
- **kwargs** (`dict`, `optional`) --
  其他关键字参数，传递给 `ValueHead` 类。

初始化模型。

- **input_ids** (*torch.LongTensor* 形状 *(batch_size,equence_length)*) --
  词汇表中输入序列标记的索引。
- **past_key_values** (*tuple(tuple(torch.FloatTensor))*, *可选*) --
  包含由模型计算的预先计算的隐藏状态（注意块中的键和值）
  （参见 *past_key_values* 输入）以加速顺序解码。
- **attention_mask**（*torch.FloatTensor*，形状*（batch_size，sequence_length）*，*可选*）--
  掩码以避免对填充标记索引进行关注。 `[0, 1]`中选择的掩码值：
  - 1 表示**未屏蔽**的令牌，
  - 0 表示被**屏蔽**的令牌。
- **return_past_key_values** (bool) -- 指示是否应返回计算的隐藏状态的标志。
- **kwargs** (*dict*, *可选*) --
  传递给包装模型的附加关键字参数。对包装模型应用前向传递并返回值头的 logits。

- ***args** (`list`, *可选*) --
  传递给包装模型的 `generate` 方法的位置参数。
- **kwargs**（`dict`，*可选*）--
  传递给包装模型的 `generate` 方法的关键字参数。

围绕包装模型的 `generate` 方法的简单包装器。请参阅
[⟦T209⟧](https://huggingface.co/docs/transformers/internal/generation_utils) 包装模型的方法
有关支持的参数的更多信息。

- ****kwargs** (`dict`, `optional`) --
  其他关键字参数，传递给 `ValueHead` 类。这些参数可以包含
  `v_head_init_strategy` 参数以及 `v_head_initializer_range` 参数。

初始化值头的权重。默认的初始化策略是随机的。用户可以通过
通过在调用时传递`v_head_init_strategy`参数来实现不同的初始化策略
`.from_pretrained`。支持的策略有：
- `normal`：用正态分布初始化权重。

## AutoModelForSeq2SeqLMWithValueHead[[trl.experimental.ppo.AutoModelForSeq2SeqLMWithValueHead]]- **预训练模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要包装的模型。应该是GPT2这样的因果语言模型。或映射到的任何模型
  [AutoModelForSeq2SeqLM](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSeq2SeqLM)类。
- **夸格斯** --
  传递给 `ValueHead` 类的其他关键字参数。

除了语言模型头之外，还有一个值头的 seq2seq 模型。这个类继承自
[experimental.ppo.PreTrainedModelWrapper](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PreTrainedModelWrapper) 并包装一个 [PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 类。包装类
支持`from_pretrained`和`push_to_hub`等经典功能，还提供了一些额外的功能
`generate` 等功能。

我们在包装模型上调用`generate`。

我们初始化值头的权重。

### TPO 培训师
https://huggingface.co/docs/trl/v1.9.2/tpo_trainer.md