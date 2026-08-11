<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 广义知识蒸馏训练器

[⟦T91⟧](https://huggingface.co/models?other=gkd,trl)

## 概述

广义知识蒸馏（GKD）由 Rishabh Agarwal、Nino Vieillard、Yongchao Zhou、Piotr Stanczyk、Sabela Ramos、Matthieu Geist 和 Olivier Bachem 在[On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes](https://huggingface.co/papers/2306.13649) 中提出。

论文摘要如下：

> 知识蒸馏 (KD) 广泛用于压缩教师模型，通过训练较小的学生模型来减少其推理成本和内存占用。然而，当前用于自回归序列模型的 KD 方法存在训练期间看到的输出序列与学生在推理期间生成的输出序列之间分布不匹配的问题。为了解决这个问题，我们引入了广义知识蒸馏（GKD）。 GKD 不是仅仅依赖于一组固定的输出序列，而是通过利用教师对此类序列的反馈来训练学生自行生成的输出序列。与监督 KD 方法不同，GKD 还提供了在学生和教师之间采用替代损失函数的灵活性，这在学生缺乏模仿教师分布的表达能力时非常有用。此外，GKD 有助于无缝集成使用 RL 微调 (RLHF) 进行蒸馏。我们展示了 GKD 在摘要、翻译和算术推理任务上提炼自回归语言模型以及用于指令调优的任务无关提炼的功效。

GKD 的关键方面是：

1. 它通过在其自生成的输出序列上训练学生模型来解决自回归序列模型中的训练推理分布不匹配问题。
2. GKD 允许通过广义詹森-香农散度 (JSD) 灵活地选择学生和教师模型之间的不同散度度量，这在学生缺乏完全模仿教师的能力时非常有用。

这个后训练方法是[Kashif Rasul](https://huggingface.co/kashif)和[Lewis Tunstall](https://huggingface.co/lewtun)贡献的。

## 使用技巧

[experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer) 是 [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) 类的包装，它接受教师模型参数。需要通过[experimental.gkd.GKDConfig](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDConfig)设置三个参数，即：* `lmbda`：控制学生数据比例，即符合政策的学生生成输出的比例。当 `lmbda=0.0` 时，损失减少到有监督的 JSD，其中学生使用教师的 token 级概率进行训练。当`lmbda=1.0`时，损失减少到同策略JSD，其中学生生成输出序列以及教师对这些序列的特定于令牌的反馈。对于 [0, 1] 之间的值，根据每个批次的 `lmbda` 值，两者之间是随机的。
* `seq_kd`：控制是否执行Sequence-Level KD（可以看作是教师生成输出上的监督FT）。当`seq_kd=True`和`lmbda=0.0`时，损失减少到有监督的JSD，其中教师生成输出序列，学生从教师那里接收关于这些序列的特定于标记的反馈。
* `beta`：控制广义 Jensen-Shannon 散度中的插值。  当`beta=0.0`时，损失近似于正向KL散度，而对于`beta=1.0`，损失近似于反向KL散度。对于 [0, 1] 之间的值，它会在两者之间进行插值。

作者发现，同策略数据（高`lmbda`）表现更好，并且最佳`beta`根据任务和评估方法而变化。> [!警告]
> 训练[Gemma models](https://huggingface.co/models?other=gemma2)时确保`attn_implementation="kernels-community/flash-attn2"`。否则，由于该架构采用的[soft capping technique](https://huggingface.co/blog/gemma2#soft-capping-and-attention-implementations)，您将在 logits 中遇到 NaN。

基本API如下：

```python
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gkd import GKDConfig, GKDTrainer

NUM_DUMMY_SAMPLES = 100

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# The model to optimise
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
# The teacher model to calculate the KL divergence against
teacher_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

train_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "Hi, how are you?"},
                {"role": "assistant", "content": "I'm great thanks"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)
eval_dataset = Dataset.from_dict(
    {
        "messages": [
            [
                {"role": "user", "content": "What colour is the sky?"},
                {"role": "assistant", "content": "The sky is blue"},
            ]
        ]
        * NUM_DUMMY_SAMPLES
    }
)

training_args = GKDConfig(output_dir="gkd-model", per_device_train_batch_size=1)
trainer = GKDTrainer(
    model=model,
    teacher_model=teacher_model,
    args=training_args,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)
trainer.train()
```

### 预期的数据集类型

数据集应格式化为“消息”列表，其中每条消息都是具有以下键的字典列表：

* `role`：`system`、`assistant` 或 `user`
* `content`: 消息内容

## GKDTrainer[[trl.experimental.gkd.GKDTrainer]]

- **型号**（[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)或`torch.nn.Module`或`str`，*可选*）--
  要训练的模型，或要从预训练模型实例化的模型的字符串标识符。
- **教师模型**（[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)或`torch.nn.Module`或`str`，*可选*）--
  用于知识蒸馏的教师模型，或要从实例化的模型的字符串标识符
  预训练模型。
- **参数**（[experimental.gkd.GKDConfig](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDConfig)，*可选*）--
  训练论证。
- **data_collator**（`DataCollator`，*可选*）--
  数据整理器用于对数据集中的样本进行批处理。它默认为
  `experimental.utils.DataCollatorForChatML` 使用 `processing_class`。
- **train_dataset**（`Dataset`，*可选*）--
  用于训练的数据集。
- **eval_dataset** （`Dataset` 或 `Dataset` 的 `dict`，*可选*）--
  用于评估的数据集。- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，[BaseImageProcessor](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/image_processor#transformers.BaseImageProcessor)，[FeatureExtractionMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin)或[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理数据的类。
- **compute_metrics**（`Callable`，*可选*）--
  计算评估指标的函数。必须接受[EvalPrediction](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.EvalPrediction)并返回
  要浮动的字典字符串。
- **回调**（[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)的`list`，*可选*）--
  训练期间使用的回调。
- **优化器**（`torch.optim.Optimizer`和`torch.optim.lr_scheduler.LambdaLR`的`tuple`，*可选*，默认为`(None, None)`）--
  包含用于训练的优化器和学习率调度器的元组。
- **preprocess_logits_for_metrics** (`Callable`，*可选*) --
  在计算指标之前预处理 logits 的函数。必须包含 `logits` 和 `labels` 以及
  返回用于指标计算的 logits。
- **peft_config**（`PeftConfig`，*可选*）--
  PEFT 配置使用 PEFT 进行训练。如果`None`，则不使用PEFT。如果提供，`model` 将是
  用指定的 PEFT 适配器包裹。
- **formatting_func** (`Callable`，*可选*) --
  格式化数据集的函数。必须接受一个示例并返回一个示例。
语言模型的广义知识蒸馏 (GKD) 训练器。

有关 GKD 的详细信息，请参阅论文：[On-Policy Distillation of Language Models: Learning from Self-Generated
Mistakes](https://huggingface.co/papers/2306.13649)。- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
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

## GKDConfig[[trl.experimental.gkd.GKDConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "chat_template_path", "val": ": str | None = None"}, {"name": "dataset_text_field", "val": ": str = 'text'"}, {"name": "dataset_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "dataset_text_field", "val": ": str = 'text'"}无 = 无"}, {"name": "dataset_num_proc", "val": ":整数 |无 = 无"}, {"name": "eos_token", "val": ": str |无 = 无"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "truncation_mode", "val": ": str = 'keep_start'"}, {"name": "shuffle_dataset", "val": ": bool = False"}, {"name": "packing", "val": ": bool = False"}, {"name": "packing_strategy", "val": ": str = 'bfd'"}, {"name": "padding_free", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": int |无 = 无"}, {"name": "eval_packing", "val": ": bool |无 = 无"}, {"name": "completion_only_loss", "val": ": bool |无 = 无"}, {"name": "assistant_only_loss", "val": ": bool = False"}, {"name": "loss_type", "val": ": str |无 = 无"}, {"name": "activation_offloading", "val": ": bool = False"}, {"name": "pad_token", "val": ": str |无 = 无"}, {"name": "温度", "val": ": float = 0.9"}, {"name": "lmbda", "val": ": float = 0.5"}, {"name": "beta", "val": ": float = 0.5"}, {"name": "max_new_tokens", "val": ": int = 128"}, {"name": "teacher_model_name_or_path", "val": ": str |无 = 无"}, {"name": "teacher_model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "disable_dropout", "val": ": bool = True"},{"name": "seq_kd", "val": ": bool = False"}]}>
- **温度**（`float`，*可选*，默认为`0.9`）--
  取样温度。温度越高，完成的随机性越大。
- **lmbda**（`float`，*可选*，默认为`0.5`）--
  控制学生数据比例（即在政策中的比例）的 Lambda 参数
  学生生成的输出）。
- **beta**（`float`，*可选*，默认为`0.5`）--
  广义 Jensen-Shannon 散度损失的 `0.0` 和 `1.0` 之间的插值系数。当
  beta 为`0.0`，损失为 KL 散度。当 beta 为 `1.0` 时，损失为逆 KL 散度。
- **max_new_tokens** (`int`，*可选*，默认为`128`) --
  每次完成生成的最大令牌数。
- **教师模型名称或路径**（`str`，*可选*）--
  教师模型的模型名称或路径。如果`None`，教师模型将与正在的模型相同
  训练有素。
- **teacher_model_init_kwargs**（`dict[str, Any]`，*可选*）--
  实例化教师模型时传递给 `AutoModelForCausalLM.from_pretrained` 的关键字参数
  来自字符串。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型中禁用 dropout。- **seq_kd**（`bool`，*可选*，默认为`False`）--
  Seq_kd参数，控制是否执行Sequence-Level KD（可以看作是有监督的FT
  教师生成的输出）。

[experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer)的配置类。

该类仅包含特定于 GKD 训练的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)和[SFTConfig](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTConfig)文档。

### GMPO
https://huggingface.co/docs/trl/v1.9.2/gmpo.md