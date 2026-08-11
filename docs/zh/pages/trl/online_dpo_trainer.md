<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在线 DPO 培训师

[⟦T203⟧](https://huggingface.co/models?other=online-dpo,trl)

## 概述

在线DPO由ShangminGuo、BiaoZhang、TianlinLiu、TianqiLiu、MishaKhalman、FelipeLlinares、AlexandreRame、ThomasMesnard、YaoZhao、BilalPiot、JohanFerret和MathieuBlondel在[Direct Language Model Alignment from Online AI Feedback](https://huggingface.co/papers/2402.04792)中提出。

论文摘要如下：

> 偏好直接调整 (DAP) 方法（例如 DPO）最近已成为人类反馈强化学习 (RLHF) 的有效替代方法，不需要单独的奖励模型。然而，DAP 方法中使用的偏好数据集通常是在训练之前收集的，并且从未更新，因此反馈纯粹是离线的。此外，这些数据集中的响应通常是从与正在对齐的语言模型不同的语言模型中采样的，并且由于模型在训练过程中不断发展，因此对齐阶段不可避免地会偏离策略。在这项研究中，我们认为在线反馈是关键并改进了 DAP 方法。我们的方法，在线人工智能反馈（OAIF），使用 LLM 作为注释器：在每次训练迭代中，我们从当前模型中采样两个响应，并提示 LLM 注释器选择首选哪一个，从而提供在线反馈。尽管它很简单，我们通过多项任务中的人工评估证明 OAIF 优于离线 DAP 和 RLHF 方法。我们进一步表明，通过 LLM 注释器的指令提示，OAIF 中利用的反馈很容易控制。

这种后训练方法是由[Michael Noukhovitch](https://huggingface.co/mnoukhov)、[Shengyi Costa Huang](https://huggingface.co/vwxyzjn)、[Quentin Gallouédec](https://huggingface.co/qgallouedec)、[Edward Beeching](https://huggingface.co/edbeeching)贡献的。

## 快速开始

此示例演示如何使用在线 DPO 方法训练模型。我们使用[Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)作为基础模型和[trl-lib/Qwen2-0.5B-Reward](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward)奖励模型。我们使用[UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback)的提示。您可以在此处查看数据集中的提示：

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

以下是训练模型的脚本：

```python
# train_online_dpo.py
from datasets import load_dataset
from trl.experimental.online_dpo import OnlineDPOConfig, OnlineDPOTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = OnlineDPOConfig(output_dir="Qwen2-0.5B-OnlineDPO")
trainer = OnlineDPOTrainer(
    model=model, reward_funcs=reward_model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_online_dpo.py
```

分布在 8 个 GPU 上，训练大约需要 1 小时。您可以通过查看奖励图来验证训练进度。拒绝和选择的完成的奖励都呈增加趋势，表明模型正在改进并随着时间的推移产生更好的响应。

![](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/online-dpo-qwen2.png)

要查看[trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-OnlineDPO)的性能，您可以使用[Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models)。

$ 变形金刚聊天 trl-lib/Qwen2-0.5B-OnlineDPO
<quentin_gallouedec>：
最好的编程语言是什么？<trl-lib/Qwen2-0.5B-OnlineDPO>：
最好的编程语言取决于您的具体需求和优先级。有些人喜欢命令式编程语言（如 Haskell 或 Lisp），而另一些人则喜欢函数式编程语言（如 Scala 或 Python）。选择编程语言时，考虑您的工作方式、编程环境和项目要求非常重要。

## 预期的数据集类型

在线 DPO 仅需要 [prompt-only dataset](dataset_formats#prompt-only)（与离线 DPO 不同，离线 DPO 需要 [preference dataset](dataset_formats#preference)）。 [experimental.online_dpo.OnlineDPOTrainer](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOTrainer)支持[conversational](dataset_formats#conversational)和[standard](dataset_formats#standard)数据集格式。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。

## 使用技巧

### 鼓励 EOS 代币生成

当使用奖励模型时，我们可能希望模型在给定长度内生成完成。在训练期间，模型将生成最多达到 [experimental.online_dpo.OnlineDPOConfig](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOConfig) 的 `max_new_tokens` 参数中指定的最大长度的补全。如果你想惩罚模型在达到最大长度之前没有生成 EOS 代币，你可以使用 [experimental.online_dpo.OnlineDPOConfig](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOConfig) 的 `missing_eos_penalty` 参数：

```python
training_args = OnlineDPOConfig(..., max_new_tokens=128, missing_eos_penalty=1.0)
```

### 记录完成情况为了更好地了解模型在训练期间的行为，您可以使用 [LogCompletionsCallback](/docs/trl/v1.9.2/en/callbacks#trl.LogCompletionsCallback) 定期记录样本完成情况。

```python
trainer = OnlineDPOTrainer(..., eval_dataset=eval_dataset)
completions_callback = LogCompletionsCallback(trainer, num_prompts=8)
trainer.add_callback(completions_callback)
```

此回调将模型生成的完成直接记录到权重和偏差。

![Logged Completions](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/wandb_completions.png)

## 示例脚本

我们提供了一个示例脚本来使用在线 DPO 方法训练模型。该脚本可在 [⟦T8⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo_online.py) 中找到

要使用 [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback) 上的 [Qwen2.5 0.5B model](https://huggingface.co/trl-lib/Qwen/Qwen2.5-0.5B-Instruct) 测试在线 DPO 脚本，请运行以下命令：

```bash
python examples/scripts/dpo_online.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model_path trl-lib/Qwen2-0.5B-Reward \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 5.0e-7 \
    --output_dir Qwen2.5-0.5B-Online-DPO \
    --warmup_steps 0.1 \
    --push_to_hub
```

## 记录的指标

在培训和评估时，我们记录以下指标。这是一个例子[tracked run at Weights and Biases](https://wandb.ai/huggingface/trl/runs/w4apmsi9)* `objective/kl`：当前模型与参考模型之间的平均 Kullback-Leibler (KL) 散度。
* `objective/entropy`：模型的平均熵，表示模型选择的动作的随机性。
* `objective/non_score_reward`：来自非分数相关来源的平均奖励，基本上是`beta * kl.sum(1)`，其中`beta`是KL惩罚系数，`kl`是每个代币的KL散度。
* `objective/rlhf_reward`：平均RLHF奖励，即`scores - non_score_reward`。 `rlhf_reward`是在线DPO培训的最终目标。如果培训按预期进行，该指标应该会继续上升。
* `objective/scores`：奖励模型返回的平均分数。
* `objective/scores_margin`：选择和拒绝的完成之间的平均分数差（根据外部奖励模型）。
* `rewards/chosen`：所选完成的平均奖励（根据在线 DPO 的隐式奖励模型）。
* `rewards/rejected`：被拒绝的完成的平均奖励（根据在线DPO的隐式奖励模型）。
* `rewards/accuracies`：在线DPO隐式奖励模型的准确性。
* `rewards/margins`：选择和拒绝的完成之间的平均奖励幅度（根据在线 DPO 的隐式奖励模型）。
* `logps/chosen`：所选完成的平均对数概率。* `logps/rejected`：被拒绝完成的平均对数概率。
* `val/contain_eos_token`：包含 EOS 代币的完成部分。
* `beta`：控制代表与参考模型偏差的损失项权重的参数。通常是固定的，但可以通过将列表传递给 [experimental.online_dpo.OnlineDPOConfig](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOConfig) 来使其动态化。

## 基准实验

为了验证在线 DPO 实施的效果，我们在 8 个 H100 的单个节点上使用 Pythia 1B、2.8B 和 6.9B 模型进行了实验。这是我们用来运行实验的命令。我们直接从[The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization](https://huggingface.co/papers/2403.17031)获取SFT/RM型号。

```shell
# 1B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/multi_gpu.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-1b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-1b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-1b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_steps 0.1 \
    --missing_eos_penalty 1.0 \
    --save_steps 0.1 \
    --push_to_hub

# 2.8B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-2.8b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-2.8b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-2.8b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_steps 0.1 \
    --missing_eos_penalty 1.0 \
    --save_steps 0.1 \
    --push_to_hub

# 6.9B Online DPO experiment
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/dpo_online.py \
    --model_name_or_path trl-lib/pythia-6.9b-deduped-tldr-sft  \
    --reward_model_path trl-lib/pythia-6.9b-deduped-tldr-rm \
    --dataset_name trl-lib/tldr \
    --learning_rate 5.0e-7 \
    --output_dir pythia-6.9b-deduped-tldr-online-dpo \
    --beta 0.1 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --max_new_tokens 53 \
    --warmup_steps 0.1 \
    --missing_eos_penalty 1.0 \
    --save_steps 0.1 \
    --push_to_hub
```

检查点和实验跟踪可在以下位置获得：

* [🤗 Model checkpoints](https://huggingface.co/collections/trl-lib/online-dpo-66acd3fa38a331a9cd457b07)
* [🐝 Tracked experiment](https://wandb.ai/huggingface/trl/reports/Online-DPO-experiments-for-TL-DR-summarisation--Vmlldzo5MTczMDU0)

随着我们扩大模型大小，在线 DPO 检查点的胜率越来越高。这是一个好兆头，表明在线 DPO 实施正在按预期进行。

## OnlineDPOTrainer[[trl.experimental.online_dpo.OnlineDPOTrainer]]

- **型号** (`str | nn.Module | PreTrainedModel`) --
  待训练的模型。可以是：- 一个字符串，是在 Huggingface.co 上的模型存储库中托管的预训练模型的 *模型 id*，或者
    包含使用保存的模型权重的*目录*的路径
    [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
    将 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 与关键字参数一起使用
    `args.model_init_kwargs`。
  - 一个[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)对象。仅支持因果语言模型。
- **参考模型**（[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)或`torch.nn.Module`或`None`）--
  用于训练的参考模型。如果未指定，则将从以下位置创建参考模型
  模型。
- **reward_funcs** (`RewardFunc | list[RewardFunc]`) --
  用于计算奖励的奖励函数。为了计算奖励，我们将所有奖励称为
  具有提示和完成功能并总结奖励。可以是：- 单个奖励函数：可以是字符串（模型路径）、[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel) 或
    自定义可调用函数。
  - 奖励函数列表：必须全部是兼容类型。
- **参数** ([experimental.online_dpo.OnlineDPOConfig](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOConfig)) --
  用于训练的在线 DPO 配置参数。
- **数据整理器** (`DataCollator`) --
  用于训练的数据整理器。如果没有指定，则默认数据整理器
  将使用 (`experimental.utils.DPODataCollatorWithPadding`) 将序列填充到
  给定配对序列的数据集，批次中序列的最大长度。
- **train_dataset** (`Dataset` 或 `IterableDataset`) --
  用于训练的数据集。
- **eval_dataset** (`Dataset`、`IterableDataset` 或 `dict[str, Dataset | IterableDataset]`) --
  用于评估的数据集。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)或[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理类用于处理数据。如果提供，将用于自动处理输入
  对于模型，它将与模型一起保存，以便更容易重新运行中断的训练或
  重用微调后的模型。
- **奖励处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)或`list[PreTrainedTokenizerBase]`，*可选*）--
  与`reward_funcs`中指定的奖励函数对应的处理类。可以是：- 单一处理类：当`reward_funcs`仅包含一个奖励函数时使用。
  - 处理类列表：必须与`reward_funcs`中奖励函数的顺序和长度相匹配。

  如果设置为`None`，每个基于模型的奖励函数的标记器会自动加载
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoTokenizer.from_pretrained)。
- **peft_config**（`PeftConfig`，*可选*）--
  PEFT 配置用于包裹模型。如果`None`，则模型未包装。
- **计算指标**（`Callable[[EvalPrediction], dict]`，*可选*）--
  用于计算指标的函数。必须采用 `EvalPrediction` 并返回字典字符串
  度量值。
- **回调** (`list[transformers.TrainerCallback]`) --
  用于训练的回调。
- **优化器** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`) --
  用于训练的优化器和调度器。
- **preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`) --
  在计算指标之前用于预处理 logits 的函数。

初始化 OnlineDPOTrainer。- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
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
  函数是否仅在 `git push` 完成时返回。
- **令牌**（`str`，*可选*，默认为`None`）--
  具有写入权限的令牌，可以覆盖 Trainer 的原始参数。
- **修订**（`str`，*可选*）--
  要提交的 git 修订版本。默认为“主”分支的头部。
- **kwargs**（`dict[str, Any]`，*可选*）--
  传递到 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## OnlineDPOConfig[[trl.experimental.online_dpo.OnlineDPOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int |无 = 无"}, {"name": "remove_unused_columns", "val": ": bool = False"}, {"name": "label_names", "val": ": list[str] |无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": 字符串 |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "reward_model_path", "val": ": str |无 = 无"}, {"name": "max_new_tokens", "val": ": int = 64"}, {"name": "max_length", "val": ": int = 512"}, {"name": "温度", "val": ": float = 0.9"}, {"name": "top_p", "val": ": float = 1.0"}, {"name": "top_k", "val": ": int = 0"}, {"name": "min_p", "val": ": float | None = None"}, {"name": "repetition_penalty", "val": ": float = 1.0"}, {"name": " Generation_kwargs", "val": ":字典 |无 = 无"}, {"name": "cache_implementation", "val": ": str |无 = 无"}, {"name": "missing_eos_penalty", "val": ": float | None = None"}, {"name": "beta", "val": ": list = "}, {"name": "loss_type", "val": ": str = 'sigmoid'"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_structed_outputs_regex", "val": ": str |无 = 无"}, {"name": "vllm_gpu_memory_utilization", "val": ": float |无 = 0.55"}, {"name": "vllm_mode", "val": ": str = 'colocate'"}, {"name": "vllm_server_base_url", "val": ": str |无 = 无"}, {"name": "vllm_server_host", "val": ": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8000"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "vllm_enable_sleep_mode", "val": ": bool = False"}, {"name": "ds3_gather_for_ Generation", "val": ": bool = True"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "reward_weights", "val": ": list[float] | None = None"}]}>
- **奖励_模型_路径**（`str`，*可选*）--
  奖励模型的路径。
- **max_new_tokens** (`int`，*可选*，默认为`64`) --
  每次完成生成的最大令牌数。
- **max_length**（`int`，*可选*，默认为`512`）--
  用于计算对数概率的序列的最大总长度（提示+完成）。如果
  序列超过此限制，最左边的标记将被截断以保留尽可能多的完成内容
  可能的。
- **温度**（`float`，*可选*，默认为`0.9`）--
  取样温度。温度越高，完成的随机性越大。
- **missing_eos_penalty** (`float`，*可选*) --
  当模型无法生成 EOS 代币时，会对分数进行处罚。这有助于鼓励
  生成短于最大长度 (`max_new_tokens`) 的补全。惩罚必须是积极的
  值。
- **beta**（`float` 或 `list[float]`，*可选*，默认为 `0.1`）--
  控制与参考模型的偏差的参数。较高的β意味着较小的偏差参考模型。对于 IPO 损失 (`loss_type="ipo"`)，β 是正则化参数，用 τ 表示
  [paper](https://huggingface.co/papers/2310.12036)。如果提供了浮点数列表，则 β 为
  为每个新纪元选择，最后一个 β 用于其余纪元。
- **loss_type** (`str`，*可选*，默认为`"sigmoid"`) --
  使用的损失类型。可能的值为：

  - `"sigmoid"`：原始[DPO](https://huggingface.co/papers/2305.18290)论文中的 sigmoid 损失。
  - `"ipo"`：[IPO](https://huggingface.co/papers/2310.12036)论文中的IPO损失。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型和参考模型中禁用 dropout。

控制生成的参数- **top_p**（`float`，*可选*，默认为`1.0`）--
  控制要考虑的顶级令牌的累积概率的浮点数。必须在 (0, 1] 中。设置为
  `1.0` 考虑所有代币。
- **top_k** (`int`，*可选*，默认为`0`) --
  要保留用于 top-k 过滤的最高概率词汇标记的数量。如果`0`，top-k-filtering是
  禁用并考虑所有令牌。
- **min_p** (`float`, *可选*) --
  最小令牌概率，将按最可能令牌的概率进行缩放。它必须是一个
  值介于 `0.0` 和 `1.0` 之间。典型值在`0.01-0.2`范围内。
- **重复惩罚**（`float`，*可选*，默认为`1.0`）--
  根据新令牌是否出现在提示中以及到目前为止生成的文本中对新令牌进行惩罚的浮动。
  值 > `1.0` 鼓励模型使用新的代币，而值 < ⟦T113⟧ encourage the model to repeat
  tokens.
- **cache_implementation** (⟦T114⟧, *optional*) --
  Implementation of the cache method for faster generation when ⟦T115⟧ is set to ⟦T116⟧.
- **generation_kwargs** (⟦T117⟧, *optional*) --
  Additional keyword arguments to pass to ⟦T246⟧ (if using transformers) or
  ⟦T118⟧ (if using vLLM) when sampling completions. This can be used to further customize the
  generation behavior, such as setting ⟦T119⟧, ⟦T120⟧, etc. If it contains keys that conflict
  with the other generation parameters (like ⟦T121⟧, ⟦T122⟧, etc.), they will override them.

Parameters that control generation acceleration powered by vLLM

- **use_vllm** (⟦T123⟧, *optional*, defaults to ⟦T124⟧) --
  Whether to use vLLM for generating completions. If set to ⟦T125⟧, the trainer will use vLLM for generation
  instead of the default model.generate(). Requires ⟦T126⟧ to be installed.
- **vllm_model_impl** (⟦T127⟧, *optional*, defaults to ⟦T128⟧) --
  Model implementation to use for vLLM. Must be one of ⟦T129⟧ or ⟦T130⟧. ⟦T131⟧: Use
  the ⟦T132⟧ backend for model implementation. ⟦T133⟧: Use the ⟦T134⟧ library for model
  implementation.
- **vllm_mode** (⟦T135⟧, *optional*, defaults to ⟦T136⟧) --
  Mode to use for vLLM integration when ⟦T137⟧ is set to ⟦T138⟧. Must be one of ⟦T139⟧ or
  ⟦T140⟧.

  - ⟦T141⟧: The trainer will send generation requests to a separate vLLM server. Make sure a TRL vLLM
    server is running (start with ⟦T142⟧).
  - ⟦T143⟧: vLLM will run in the same process and share the training GPUs. This avoids the need for a
    separate server but may cause resource contention with training.
- **vllm_structured_outputs_regex** (⟦T144⟧, *optional*) --
  Regex for vLLM structured outputs. If ⟦T145⟧ (default), structured outputs is disabled.

Parameters that control the vLLM server (only used when ⟦T146⟧ is ⟦T147⟧)

- **vllm_server_base_url** (⟦T148⟧, *optional*) --
  Base URL for the vLLM server (e.g., ⟦T149⟧). If provided, ⟦T150⟧ and
  ⟦T151⟧ are ignored.
- **vllm_server_host** (⟦T152⟧, *optional*, defaults to ⟦T153⟧) --
  Host of the vLLM server to connect to. Ignored if ⟦T154⟧ is provided.
- **vllm_server_port** (⟦T155⟧, *optional*, defaults to ⟦T156⟧) --
  Port of the vLLM server to connect to. Ignored if ⟦T157⟧ is provided.
- **vllm_server_timeout** (⟦T158⟧, *optional*, defaults to ⟦T159⟧) --
  Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up after the
  timeout, a ⟦T160⟧ is raised.
- **vllm_group_port** (⟦T161⟧, *optional*, defaults to ⟦T162⟧) --
  Port number for the weight update group. This is used to communicate with the vLLM server. Unless the port
  is occupied, there is no need to change it.

Parameters that control colocated vLLM execution (only used when ⟦T163⟧ is ⟦T164⟧)

- **vllm_gpu_memory_utilization** (⟦T165⟧, *optional*, defaults to ⟦T166⟧) --
  Control the GPU memory utilization for vLLM. This setting only applies when ⟦T167⟧ is set to
  ⟦T168⟧. If you are using ⟦T169⟧, this parameter must be passed separately when
  launching the vLLM server via the ⟦T170⟧ flag.
- **vllm_tensor_parallel_size** (⟦T171⟧, *optional*, defaults to ⟦T172⟧) --
  Control the tensor parallel size for vLLM. This setting only applies when ⟦T173⟧ is set to
  ⟦T174⟧. If you are using ⟦T175⟧, this parameter must be passed separately when
  launching the vLLM server via the ⟦T176⟧ flag.
- **vllm_enable_sleep_mode** (⟦T177⟧, *optional*, defaults to ⟦T178⟧) --
  Enable vLLM sleep mode to offload weights/cache during the optimizer step. Keeps GPU memory usage low, but
  waking the engine adds host–device transfer latency.

Other parameters

- **ds3_gather_for_generation** (⟦T179⟧, *optional*, defaults to ⟦T180⟧) --
  This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for generation,
  improving generation speed. However, disabling this option allows training models that exceed the VRAM
  capacity of a single GPU, albeit at the cost of slower generation. Disabling this option is not compatible
  with vLLM generation.
- **model_init_kwargs** (⟦T181⟧, *optional*) --
  Keyword arguments to pass to ⟦T182⟧ when instantiating the model from a
  string.
- **trust_remote_code** (⟦T183⟧, *optional*, defaults to ⟦T184⟧) --
  Whether to allow loading models that ship custom Python code from the Hub. Forwarded to
  ⟦T247⟧. Also applied to reward-model and reward-tokenizer
  loads.
- **reward_weights** (⟦T185⟧, *optional*) --
  Weights for combining multiple reward functions. Must match the number of reward functions. If ⟦T186⟧, all
  reward functions are equally weighted.

Configuration class for the ⟦T248⟧.

This class includes only the parameters that are specific to Online DPO training. For a full list of training
arguments, please refer to the ⟦T249⟧ documentation. Note that default values in this
class may differ from those in ⟦T250⟧.

Using ⟦T251⟧ we can turn this class into
⟦T252⟧ arguments that can be specified on the
command line.

> [!NOTE]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`5e-7`，而不是`5e-5`。> - `remove_unused_columns`：默认为`False`，而不是`True`。

### MergeModelCallback[[trl.experimental.merge_model_callback.MergeModelCallback]]
https://huggingface.co/docs/trl/v1.9.2/merge_model_callback.md