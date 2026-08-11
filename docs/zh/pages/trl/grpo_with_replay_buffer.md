<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 带有重播缓冲区的 GRPO

该实验训练器使用 GRPO 训练模型，但将标准差为 0 的组（以及相应的完成情况）替换为具有高奖励和标准差的组，这些组已用于训练先前批次中的模型。

## 用法

```python
import torch
from trl.experimental.grpo_with_replay_buffer import GRPOWithReplayBufferConfig, GRPOWithReplayBufferTrainer
from datasets import load_dataset

dataset = load_dataset("trl-internal-testing/zen", "standard_prompt_only", split="train")

# Guarantee that some rewards have 0 std
def custom_reward_func(completions, **kwargs):
    if torch.rand(1).item() < 0.25:
        return [0] * len(completions)  # simulate some None rewards
    else:
        return torch.rand(len(completions)).tolist()

training_args = GRPOWithReplayBufferConfig(
    output_dir="./tmp",
    learning_rate=1e-4,
    per_device_train_batch_size=4,
    num_generations=4,
    max_completion_length=8,
    replay_buffer_size=8,
    report_to="none",
)

trainer = GRPOWithReplayBufferTrainer(
    model="trl-internal-testing/tiny-Qwen2ForCausalLM-2.5",
    reward_funcs=[custom_reward_func],
    args=training_args,
    train_dataset=dataset,
)

previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()
```

## GRPOWithReplayBufferTrainer[[trl.experimental.grpo_with_replay_buffer.GRPOWithReplayBufferTrainer]]

- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于`True`，加载*args.output_dir* 中由前一个实例保存的最后一个检查点
  `Trainer`。如果存在，训练将从此处加载的模型/优化器/调度器状态恢复。
- **试用**（`optuna.Trial`或`dict[str, Any]`，*可选*）--
  用于超参数搜索的试运行或超参数字典。
- **ignore_keys_for_eval** (`list[str]`，*可选*) --
  模型输出中的键列表（如果它是字典），在以下情况下应忽略这些键：
  收集训练期间评估的预测。`~trainer_utils.TrainOutput`包含全局步数、训练损失和指标的对象。

主要培训切入点。将保存模型，以便您可以使用`from_pretrained()`重新加载它。

只会从主进程中保存。

- **commit_message** (`str`，*可选*，默认为`"End of training"`) --
  推送时要提交的消息。
- **阻塞**（`bool`，*可选*，默认为`True`）--
  函数是否仅在 `git push` 完成时返回。
- **令牌**（`str`，*可选*，默认为`None`）--
  具有写入权限的令牌，可以覆盖 Trainer 的原始参数。
- **修订**（`str`，*可选*）--
  要提交的 git 修订版本。默认为“主”分支的头部。
- **kwargs**（`dict[str, Any]`，*可选*）--
  传递给 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## GRPOWithReplayBufferConfig[[trl.experimental.grpo_with_replay_buffer.GRPOWithReplayBufferConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int |无 = 无"}, {"name": "remove_unused_columns", "val": ": bool |无 = False"}, {"name": "label_names", "val": ": list[str] |无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": 字符串 |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "disable_dropout", "val": ": bool = False"}, {"name": "cast_lm_head_to_fp32", "val": ": bool = False"}, {"name": "num_ Generations", "val": ": int |无 = 8"}, {"name": "num_ Generations_eval", "val": ": int |无 = N一"}, {"name": "max_completion_length", "val": ": int |无 = 256"}，{"name"："ds3_gather_for_ Generation"，"val"："：bool = True"}，{"name"："shuffle_dataset"，"val"："：bool |无 = True"}, {"name": "pad_to_multiple_of", "val": ": int |无 = 无"}, {"name": " Generation_batch_size", "val": ": int |无 = 无"}, {"name": "steps_per_ Generation", "val": ": int |无 = 无"}, {"name": "温度", "val": ": float = 1.0"}, {"name": "top_p", "val": ": float = 1.0"}, {"name": "top_k", "val": ": int = 0"}, {"name": "min_p", "val": ": float |无 = 无"}, {"name": " Generation_kwargs", "val": ": dict |无 = 无"}, {"name": "chat_template_kwargs", "val": ": dict | None = None"}, {"name": "repetition_penalty", "val": ": float = 1.0"}, {"name": "cache_implementation", "val": ": str |无 = 无"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": "vllm_mode", "val": ": str = 'colocate'"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_enable_sleep_mode", “val”：“：bool = False”}，{“name”：“vllm_structed_outputs_regex”，“val”：“：str |无 = 无"}, {"name": "vllm_server_base_url", "val": ": str |无 = 无"}, {"name": "vllm_server_host", "val":": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8000"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_gpu_memory_utilization", "val": ": float = 0.3"}, {"name": "vllm_max_model_length", "val": ": int | None = None"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "beta", "val": ": float = 0.0"}, {"name": "num_iterations", "val": ": int = 1"}, {"name": "epsilon", "val": ": float = 0.2"}, {"name": "delta", "val": ": float | None = None"}, {"name": "epsilon_high", "val": ": None = None"}, {"name": "sapo_temp_neg", "val": ": float = 1.05"}, {"name": "sapo_Temperature_pos", "val": ": float = 1.0"}, {"name": "vespo_k_pos", "val": ": float = 2.0"}, {"name": "vespo_lambda_pos", "val": ": float = 3.0"}, {"name": "vespo_k_neg", "val": ": float = 3.0"}, {"name": "vespo_lambda_neg", "val": ": float = 2.0"}, {"name": "importance_sampling_level", "val": ": str = 'token'"}, {"name": "reward_weights", "val": ": list[float] | None = None"}, {"name": "multi_objective_aggregation", "val": ": str = 'sum_then_normalize'"}, {"name": "scale_rewards", "val": ": str = 'group'"}, {"name": "loss_type", "val": ": str = 'dapo'"}, {"name": "mask_truncated_completions", "val": ": bool = False"}, {"name": "sync_ref_model", "val": ": bool = False"}, {"name": "ref_model_mixup_alpha", "val": ": float = 0.6"}, {"name": "ref_model_sync_steps", "val": ": int = 512"}, {"name": "top_entropy_quantile", "val": ": float = 1.0"}, {"name": "entropy_coef", "val": ": float = 0.0"}, {"name": "use_adaptive_entropy", "val": ": bool = False"}, {"name": "entropy_coef_min", "val": ": float = 0.0"}, {"name": "entropy_coef_max", "val": ": float = 1.0"}, {"name": "entropy_coef_delta", "val": ": float = 0.005"}, {"name": "entropy_target", "val": ": float = 0.2"}, {"name": "max_tool_calling_iterations", "val": ": int |无 = 无"}, {"name": "vllm_importance_sampling_ Correction", "val": ": bool = True"}, {"name": "vllm_importance_sampling_mode", "val": ": str = 'sequence_mask'"}, {"name": "vllm_importance_sampling_clip_max", "val": ": float |无 = 3.0"}, {"name": "vllm_importance_sampling_clip_min", "val": ": float |无 = 无"}, {"name": "off_policy_mask_threshold", "val": ": float |无 = 无"}, {"name": "use_bias_ Correction_kl", "val": ": bool = False"}, {"name": "log_complitions", "val": ": bool = False"}, {"name": "log_multimodal", "val": ": bool = True"}, {"name": "num_completions_to_print", "val": ": int |无 = 无"}, {"name": "log_unique_prompts", "val": ": bool = False"}, {"name": "log_completions_hub_repo", "val": ": str |无 = 无"}, {"name": "use_transformers_continuous_batching", "val": ": bool = False"}, {"name": "transformers_continuous_batching_config", "val": ": dict |无 = 无"}, {"name": "use_transformers_paged", "val": ": bool = False"}, {"name": "vllm_importance_sampling_cap", "val": ": float |无 = 无"}, {"name": "replay_buffer_size", "val": ": int = 64"}]}>

新参数：
replay_buffer_size（`int`，*可选*，默认为`64`）：
一个缓存，用于存储每组具有最高优势分数和方差的部署。如果一个新的
组的方差为 0，它被替换为从重放缓冲区采样的组。

## ReplayBuffer[[trl.experimental.grpo_with_replay_buffer.ReplayBuffer]]

一个简单的重播缓冲区，用于存储和采样以前看到的卷展。

### PEFT 集成
https://huggingface.co/docs/trl/v1.9.2/peft_integration.md