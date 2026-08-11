<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 蒸馏训练器

## 概述

Distillation Trainer 实现按策略知识蒸馏，如 Rishabh Agarwal、Nino Vieillard、Yongchao Zhou、Piotr Stanczyk、Sabela Ramos、Matthieu Geist 和 Olivier Bachem 在[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649) 中所述。

> 知识蒸馏 (KD) 广泛用于压缩教师模型，通过训练较小的学生模型来减少其推理成本和内存占用。然而，当前用于自回归序列模型的 KD 方法存在训练期间看到的输出序列与学生在推理期间生成的输出序列之间分布不匹配的问题。为了解决这个问题，我们引入了广义知识蒸馏（GKD）。 GKD 不是仅仅依赖于一组固定的输出序列，而是通过利用教师对此类序列的反馈来训练学生自行生成的输出序列。与监督 KD 方法不同，GKD 还提供了在学生和教师之间采用替代损失函数的灵活性，这在学生缺乏模仿教师分布的表达能力时非常有用。`DistillationTrainer` 专为有效地将各种尺寸的教师模型提炼为较小的学生而设计。它通过三个关键优化扩展了 `GKDTrainer` 的思想：

1. **生成缓冲区** – 将训练微批量大小与生成批量大小解耦，让 vLLM 在跨梯度累积步骤的单个调用中批量处理许多提示。仅此一项就可以将训练速度提高 40 倍。
2. **教师服务器支持** – 将教师移至外部 vLLM 服务器，因此不需要与学生安装在相同的 GPU 上。
3. **二进制编码的 logprob 有效负载** – 将日志概率打包到 Base64 编码的 NumPy 数组中，而不是嵌套的 JSON 列表，将传输有效负载缩小约 5 倍。

> [!注意]
> Distillation Trainer 目前是 `trl.experimental` 命名空间的一部分。当功能迭代时，API 可能会发生更改，恕不另行通知。

## 快速开始

```python
from datasets import load_dataset
from trl.experimental.distillation import DistillationConfig, DistillationTrainer

# 1. Load dataset and format as prompt-only chat messages
dataset = load_dataset("openai/gsm8k", "main", split="train")
dataset = dataset.map(
    lambda x: {"messages": [{"role": "user", "content": x["question"]}]},
    remove_columns=dataset.column_names,
)

# 2. Configure distillation
config = DistillationConfig(
    output_dir="results/distill-qwen-gsm8k",
    num_train_epochs=1,
    bf16=True,
    save_strategy="no",
    # Distillation
    lmbda=1.0,                      # fully on-policy (student generates)
    beta=1.0,                       # reverse KL
    # Teacher
    teacher_model_init_kwargs={"dtype": "bfloat16"},
)

# 3. Train
trainer = DistillationTrainer(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    teacher_model="Qwen/Qwen2.5-7B-Instruct",
    args=config,
    train_dataset=dataset,
)
trainer.train()
trainer.save_model()
```

## 使用技巧

[experimental.distillation.DistillationTrainer](/docs/trl/v1.9.2/en/distillation_trainer#trl.experimental.distillation.DistillationTrainer)需要通过[experimental.distillation.DistillationConfig](/docs/trl/v1.9.2/en/distillation_trainer#trl.experimental.distillation.DistillationConfig)设置两个关键参数：* `lmbda`：控制学生数据比例，即符合政策的学生生成输出的比例。当`lmbda=0.0`时，训练完全脱离策略（仅数据集完成）。当`lmbda=1.0`时，训练完全符合策略（学生生成所有完成结果）。对于介于两者之间的值，每个梯度累积切片根据`lmbda`随机分配为on-policy或off-policy。
* `beta`：控制广义 Jensen-Shannon 散度中的插值。当`beta=0.0`时，损失近似于正向KL散度，而`beta=1.0`近似于反向KL散度。之间的值进行插值。

### 在政策与离政策

设置`lmbda=1.0`（完全在策略）通常优于离策略蒸馏，因为学生从自己的错误中学习，而不是模仿它可能永远不会产生的轨迹。生成缓冲区确保按策略训练保持高效：跨梯度累积步骤的提示被批量处理到单个 vLLM 调用中。

### 使用外部教师服务器

对于不适合训练 GPU 的教师（例如 100B+ 参数），请将教师托管在单独的 vLLM 服务器上，并设置 `use_teacher_server=True` 和 `teacher_model_server_url`：

```python
config = DistillationConfig(
    output_dir="distilled-model",
    use_teacher_server=True,
    teacher_model_server_url="http://teacher-host:8000",
    loss_top_k=1,       # required with teacher server when beta > 0
    beta=1.0,
    lmbda=1.0,
)

trainer = DistillationTrainer(
    model="Qwen/Qwen3-4B",
    args=config,
    train_dataset=dataset,
)
trainer.train()
```使用教师服务器时：
- 当`beta=0.0`（向前KL）时，`loss_top_k`必须是`> 0`
- 当`beta > 0`时，`loss_top_k`必须恰好是`1`（反向KL或JSD）
- 不支持`reverse_kl_top_1_mode="argmax"`
- 不支持 Liger 内核

### 预期的数据集类型

数据集应格式化为 [conversational](dataset_formats#conversational) [language modeling](dataset_formats#language-modeling) 数据集：

```python
{"messages": [{"role": "user", "content": "What color is the sky?"},
              {"role": "assistant", "content": "It is blue."}]}
```

当使用完全同策略蒸馏（`lmbda=1.0`）时，可以省略辅助轮，因为学生将生成自己的完成：

```python
{"messages": [{"role": "user", "content": "What color is the sky?"}]}
```

## 示例脚本

使用 [⟦T27⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/distillation.py) 从命令行启动蒸馏训练。该脚本通过标准 `ModelConfig` 标志支持完整训练、混合开/关政策和 LoRA。

```bash
# Full training (off-policy only, lmbda=0):
python examples/scripts/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lmbda 0.0 \
    --output_dir distilled-model \
    --num_train_epochs 1
```

```bash
# Mixed on/off-policy (lmbda=0.5):
python examples/scripts/distillation.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --teacher_model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name trl-lib/chatbot_arena_completions \
    --learning_rate 2e-5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lmbda 0.5 \
    --beta 0.5 \
    --output_dir distilled-model \
    --num_train_epochs 1
```

## DistillationTrainer[[trl.experimental.distillation.DistillationTrainer]]

从教师模型到学生模型的知识蒸馏训练器。

支持：
- 广义 JSD 损失（正向 KL、反向 KL 或通过 `beta` 插值的 JSD）
- 政策上的蒸馏：学生完成任务，老师对其进行评分
- 本地教师模型
- 通过 vLLM 或 model.generate() 生成学生政策
- Liger 内核，用于内存高效的融合 JSD 损失- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
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
  传递给 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## DistillationConfig[[trl.experimental.distillation.DistillationConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "温度", "val": ": float = 1.0"}, {"name": "beta", "val": ": float = 1.0"}, {"name": "max_completion_length", "val": ": int = 512"}, {"name": "max_prompt_length", "val": ": int |无 = 无"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "teacher_model_name_or_path", "val": ": str |无 = 无"}, {"name": "teacher_model_revision", "val": ": str |无 = 无"}, {"name": "teacher_model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "num_ Generations", "val": ": int = 1"}, {"name": " Generation_batch_size", "val": ": int | None = None"}, {"name": "top_p", "val": ": float = 0.95"}, {"name": "top_k", "val": ": int = 0"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": "vllm_mode", "val": ": str = 'colocate'"}, {“名称”：“vllm_server_base_url”，“val”：“：str |无 = 无"}, {"name": "vllm_server_host", "val": ": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8001"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_gpu_memory_utilization", "val": ": float = 0.3"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "vllm_max_model_length", "val": “：整数|无 = 无"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_structed_outputs_regex", "val": ": str |无 = 无"}, {"name": "vllm_sync_Frequency", "val": ": int = 1"}, {"name": "vllm_enable_sleep_mode", "val": ": bool = False"}, {"name": "log_completions", "val": ": bool = False"}, {"name": "log_completions_steps", "val": ": int = 100"}, {"name": "num_completions_to_print", "val": ": int | None = None"}]}>
控制模型的参数

- **model_init_kwargs**（`dict[str, Any]`，*可选*）--
  `AutoModelForCausalLM.from_pretrained` 的关键字参数，当 `model` 参数
  trainer 以字符串形式提供。
- **trust_remote_code**（`bool`，*可选*，默认为`False`）--
  是否允许加载从 Hub 发送自定义 Python 代码的模型和标记器。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 和
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)，对于学生和老师。
- **max_length**（`int`或`None`，*可选*，默认为`1024`）--
  标记化和截断的最大总序列长度（提示+完成）。

控制蒸馏的参数- **温度**（`float`，*可选*，默认为`1.0`）--
  生成过程中采样和计算蒸馏损失的温度。产生更高的价值
  更软的概率分布。
- **beta**（`float`，*可选*，默认为`1.0`）--
  广义 Jensen-Shannon 散度损失的插值系数。当`0.0`时，损失为
  前向 KL 散度。当`1.0`时，损失是反向KL散度。当`0.5`时，为标准
  JSD。
- **max_completion_length**（`int`，*可选*，默认为`512`）--
  在策略生成期间每次完成生成的最大令牌数。
- **max_prompt_length**（`int`或`None`，*可选*）--
  提示的最大标记数。如果`None`，则自动计算为`max_length - max_completion_length`。
  提示根据分词器的 `truncation_side` 设置进行截断。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在训练期间禁用学生模型中的 dropout。

控制教师模型的参数- **教师模型名称或路径**（`str`或`None`，*可选*）--
  教师模型的模型名称或路径。当教师在本地加载时使用。
- **教师模型修订版**（`str` 或 `None`，*可选*）--
  教师模型的模型修订（例如，分支名称、标签或提交哈希）。
- **teacher_model_init_kwargs** （`dict[str, Any]` 或 `None`，*可选*）--
  实例化教师模型时传递给 `AutoModelForCausalLM.from_pretrained` 的关键字参数
  来自字符串。
控制同策略生成的参数

- **num_ Generations** （`int`，*可选*，默认为`1`）--
  在策略生成期间每个提示生成的完成数。
- ** Generation_batch_size ** （`int`或`None`，*可选*）--
  每个优化器步骤每个工作人员的唯一提示数。如果`None`，计算自
  `(per_device_train_batch_size * gradient_accumulation_steps) // num_generations`。
- **top_p** (`float`，*可选*，默认为`0.95`) --
  用于同策略生成的 Top-p（核心）采样参数。
- **top_k** (`int`，*可选*，默认为`0`) --
  用于同策略生成的 Top-k 采样参数。 `0` 禁用 top-k 过滤。

控制学生生成 vLLM 的参数- **use_vllm**（`bool`，*可选*，默认为`False`）--
  是否使用 vLLM 从学生模型生成策略内完成。
- **vllm_mode**（`str`，*可选*，默认为`"colocate"`）--
  学生 vLLM 整合模式。 `"server"` 或 `"colocate"`。
- **vllm_server_base_url** （`str` 或 `None`，*可选*）--
  学生 vLLM 服务器的基本 URL。如果提供，`vllm_server_host` 和 `vllm_server_port` 将被忽略。
- **vllm_server_host**（`str`，*可选*，默认为`"0.0.0.0"`）--
  学生 vLLM 服务器的主机。
- **vllm_server_port**（`int`，*可选*，默认为`8001`）--
  学生 vLLM 服务器的端口。
- **vllm_server_timeout**（`float`，*可选*，默认为`240.0`）--
  连接到学生 vLLM 服务器超时。
- **vllm_group_port**（`int`，*可选*，默认为`51216`）--
  vLLM 权重更新组（NCCL 通信器）的端口。
- **vllm_gpu_memory_utilization**（`float`，*可选*，默认为`0.3`）--
  并置学生 vLLM 引擎的 GPU 内存利用率。
- **vllm_tensor_parallel_size**（`int`，*可选*，默认为`1`）--
  位于同一位置的学生 vLLM 引擎的张量并行大小。- **vllm_max_model_length**（`int`或`None`，*可选*）--
  并置 vLLM 引擎的最大模型序列长度。
- **vllm_model_impl**（`str`，*可选*，默认为`"vllm"`）--
  vLLM 的模型实现后端。使用`"vllm"`或`"transformers"`。
- **vllm_structed_outputs_regex** （`str` 或 `None`，*可选*）--
  vLLM 结构化输出的正则表达式模式。
- **vllm_sync_Frequency**（`int`，*可选*，默认为`1`）--
  将学生模型权重同步到 vLLM 引擎的频率（在训练步骤中）。
- **vllm_enable_sleep_mode**（`bool`，*可选*，默认为`False`）--
  启用 vLLM 睡眠模式以在优化器步骤期间减轻学生的体重。

控制日志记录的参数- **log_completions**（`bool`，*可选*，默认为`False`）--
  是否每 `log_completions_steps` 步骤记录（提示、完成）对的样本。如果 `rich` 是
  安装后，它会打印样本。如果启用 `wandb` 和/或 `trackio` 日志记录，则会将其记录到 `wandb`
  和/或`trackio`。
- **log_completions_steps**（`int`，*可选*，默认为`100`）--
  记录完成之间的步骤数。仅当 `log_completions` 为 `True` 时使用。
- **num_completions_to_print** （`int` 或 `None`，*可选*）--
  要打印的完成数。如果`None`，则记录所有完成情况。

`DistillationTrainer` 的配置类。

使用特定于知识蒸馏的参数扩展[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)。这个配置是
独立于[SFTConfig](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTConfig)——所有必需的字段都在这里声明。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以把这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在
命令行。

### DeepSpeed 集成
https://huggingface.co/docs/trl/v1.9.2/deepspeed_integration.md