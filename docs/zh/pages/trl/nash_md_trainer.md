<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 纳什-MD 训练师

[⟦T64⟧](https://huggingface.co/models?other=nash-md,trl)

## 概述

Nash-MD 是由 Rémi Munos、[Michal Valko](https://huggingface.co/misovalko)、Daniele Calandriello、Mohammad Gheshlaghi Azar、Mark Rowland、DanielGuo、Yunhao Tang、Matthieu Geist、Thomas Mésnard 和 Andrea Michi 在论文 [Nash Learning from Human Feedback](https://huggingface.co/papers/2312.00886) 中提出的。

论文摘要如下：

> 基于人类反馈的强化学习 (RLHF) 已成为使大型语言模型 (LLM) 与人类偏好保持一致的主要范例。通常，RLHF 涉及从人类反馈中学习奖励模型的第一步，通常表示为预先训练的法学硕士生成的文本生成对之间的偏好。随后，法学硕士的政策通过强化学习算法进行优化以最大化奖励模型来进行微调。然而，当前奖励模型的固有局限性是它们无法完全代表人类偏好的丰富性及其对抽样分布的依赖性。在这项研究中，我们引入了一种使用成对人类反馈对法学硕士进行微调的替代流程。我们的方法需要初始学习偏好模型，该模型以给定提示的两个输入为条件，然后追求政策始终产生优于任何竞争政策所产生的响应，从而定义了该偏好模型的纳什均衡。我们将这种方法称为纳什从人类反馈中学习（NLHF）。在表格策略表示的背景下，我们提出了一种基于镜像下降原理的新颖算法解决方案 Nash-MD。该算法产生一系列策略，最后一次迭代收敛于正则化纳什均衡。此外，我们探索策略的参数表示，并为深度学习架构引入梯度下降算法。为了证明我们方法的有效性，我们提出了涉及文本摘要任务的法学硕士微调的实验结果。我们相信 NLHF 为偏好学习和政策优化提供了一条引人注目的途径，并有潜力推进法学硕士与人类偏好相一致的领域。

这种后训练方法是由[Kashif Rasul](https://huggingface.co/kashif)和[Daniil Tiapkin](https://huggingface.co/dtiapkin)、[Pierre Ménard](https://huggingface.co/menardprr)、Daniele Calandriello和[Quentin Gallouédec](https://huggingface.co/qgallouedec)贡献的。

## 快速开始此示例演示如何使用 Nash-MD 方法训练模型。我们使用[Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)作为基础模型和[trl-lib/Qwen2-0.5B-Reward](https://huggingface.co/trl-lib/Qwen2-0.5B-Reward)奖励模型。我们使用[UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback)的提示。您可以在此处查看数据集中的提示：

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-prompt/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

以下是训练模型的脚本：

```python
# train_nash_md.py
from datasets import load_dataset
from trl.experimental.nash_md import NashMDConfig, NashMDTrainer
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
reward_model = AutoModelForSequenceClassification.from_pretrained("trl-lib/Qwen2-0.5B-Reward", num_labels=1)
train_dataset = load_dataset("trl-lib/ultrafeedback-prompt", split="train")

training_args = NashMDConfig(output_dir="Qwen2-0.5B-NashMD")
trainer = NashMDTrainer(
    model=model, reward_funcs=reward_model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset
)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_nash_md.py
```

分布在 8 个 GPU 上，训练大约需要 3 小时。

要查看[trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-NashMD)的性能，您可以使用[Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models)。

$ 变形金刚聊天 trl-lib/Qwen2-0.5B-NashMD
<quentin_gallouedec>：
最好的编程语言是什么？

<trl-lib/Qwen2-0.5B-NashMD>：
最好的编程语言取决于个人喜好、项目的复杂性以及任务的具体要求。经常推荐的一些编程语言包括Python、Java和JavaScript，还有许多其他语言可以根据个人需求进行选择。

## 预期的数据集类型

Nash-MD 需要[prompt-only dataset](dataset_formats#prompt-only)。 [experimental.nash_md.NashMDTrainer](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDTrainer)支持[conversational](dataset_formats#conversational)和[standard](dataset_formats#standard)数据集格式。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。

## 使用技巧

### 鼓励 EOS 代币生成我们可能希望模型在给定长度内生成补全。在训练期间，模型将生成最多达到 [experimental.nash_md.NashMDConfig](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDConfig) 的 `max_new_tokens` 参数中指定的最大长度的补全。如果你想惩罚模型在达到最大长度之前没有生成 EOS 代币，你可以使用 [experimental.nash_md.NashMDConfig](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDConfig) 的 `missing_eos_penalty` 参数：

```python
training_args = NashMDConfig(..., max_new_tokens=128, missing_eos_penalty=1.0)
```

> [!警告]
> 确保 SFT 模型和奖励模型使用相同的聊天模板和相同的分词器。否则，您可能会发现训练期间模型完成度的评分不正确。

### 记录完成情况

为了更好地了解模型在训练期间的行为，您可以使用 [LogCompletionsCallback](/docs/trl/v1.9.2/en/callbacks#trl.LogCompletionsCallback) 定期记录样本完成情况。

```python
trainer = NashMDTrainer(..., eval_dataset=eval_dataset)
completions_callback = LogCompletionsCallback(trainer, num_prompts=8)
trainer.add_callback(completions_callback)
```

此回调将模型生成的完成直接记录到权重和偏差。

![Logged Completions](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/wandb_completions.png)

## 示例脚本

我们提供了一个示例脚本来使用 Nash-MD 方法训练模型。该脚本可在 [⟦T7⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/nash_md.py) 中找到

要在 [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback) 上使用 [Qwen2.5 0.5B model](https://huggingface.co/trl-lib/Qwen/Qwen2.5-0.5B-Instruct) 测试 Nash-MD 脚本，请运行以下命令：

```bash
python examples/scripts/nash_md.py \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --reward_model_path trl-lib/Qwen2-0.5B-Reward \
    --dataset_name trl-lib/ultrafeedback-prompt \
    --learning_rate 5.0e-7 \
    --output_dir Qwen2.5-0.5B-NashMD \
    --warmup_steps 0.1 \
    --push_to_hub
```

## 记录的指标

在培训和评估时，我们记录以下指标：* `loss/kl`：模型与参考数据之间的平均 KL 散度。
* `objective/entropy`：模型和参考数据的平均熵。
* `loss/score`：平均强化分数损失。
* `rewards/chosen`：模型完成的平均分数（根据奖励模型）。
* `rewards/rejected`：混合完成的平均分数（根据奖励模型）。
* `rewards/probabilities`：所选模型完成与混合完成的平均概率（根据奖励模型）。
* `rewards/accuracies`：Nash-MD 隐式奖励模型的准确性。
* `rewards/margins`：所选完成和混合完成之间的平均奖励幅度（根据奖励模型）。
* `logps/chosen`：所选完成的平均对数概率。
* `logps/rejected`：参考完成的平均对数概率。
* `val/model_contain_eos_token`：模型输出包含eos代币的次数。
* `val/ref_contain_eos_token`：混合物输出中包含 eos 代币的次数。
* `beta`：控制代表与参考模型偏差的损失项权重的参数。通常是固定的，但可以通过将列表传递给 [experimental.nash_md.NashMDConfig](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDConfig) 来使其动态化。* `mixture_coef`：模型和参考模型的 Logit 混合系数。通常是固定的，但可以通过将列表传递给 [experimental.nash_md.NashMDConfig](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDConfig) 来使其动态化。

## NashMDTrainer[[trl.experimental.nash_md.NashMDTrainer]]

- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要训练的模型，最好是`AutoModelForCausalLM`。
- **参考模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  拥抱脸部变形金刚模型，带有休闲语言造型头。用于隐式奖励计算
  和损失。如果没有提供参考模型，培训师将创建一个具有相同参考模型的参考模型
  架构作为要优化的模型。
- **reward_funcs** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  用于对完成情况进行评分的奖励模型，最好是
  [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification)。
- **参数** ([experimental.nash_md.NashMDConfig](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDConfig)) --
  用于训练的 NashMD 配置参数。
- **数据整理器** (`DataCollator`) --
  用于训练的数据整理器。如果没有指定，则默认数据整理器
  将使用 (`experimental.utils.DPODataCollatorWithPadding`) 将序列填充到
  给定配对序列的数据集，批次中序列的最大长度。
- **train_dataset** (`Dataset`) --
  用于训练的数据集。
- **eval_dataset** (`Dataset`) --
  用于评估的数据集。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，[BaseImageProcessor](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/image_processor#transformers.BaseImageProcessor)，[FeatureExtractionMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin)或[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--处理类用于处理数据。如果提供，将用于自动处理输入
  对于模型，它将与模型一起保存，以便更容易重新运行中断的训练或
  重用微调后的模型。
- **peft_config**（`PeftConfig`，*可选*）--
  用于训练的 peft 配置。
- **compute_metrics**（`Callable[[EvalPrediction], dict]`，*可选*）--
  用于计算指标的函数。必须采用 `EvalPrediction` 并返回字典字符串
  度量值。
- **回调** (`list[transformers.TrainerCallback]`) --
  用于训练的回调。
- **优化器** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`) --
  用于训练的优化器和调度器。
- **preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`) --
  在计算指标之前用于预处理 logits 的函数。

Nash-MD 方法的训练器。

它作为[experimental.online_dpo.OnlineDPOTrainer](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOTrainer)的子类实现。- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
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

## NashMDConfig[[trl.experimental.nash_md.NashMDConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int |无 = 无"}, {"name": "remove_unused_columns", "val": ": bool = False"}, {"name": "label_names", "val": ": list[str] |无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": 字符串 |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "reward_model_path", "val": ": str |无 = 无"}, {"name": "max_new_tokens", "val": ": int = 64"}, {"name": "max_length", "val": ": int = 512"}, {"name": "温度", "val": ": float = 0.9"}, {"name": "top_p", "val": ": float = 1.0"}, {"name": "top_k", "val": ": int = 0"}, {"name": "min_p", "val": ": float | None = None"}, {"name": "repetition_penalty", "val": ": float = 1.0"}, {"name": " Generation_kwargs", "val": ":字典 |无 = 无"}, {"name": "cache_implementation", "val": ": str |无 = 无"}, {"name": "missing_eos_penalty", "val": ": float | None = None"}, {"name": "beta", "val": ": list = "}, {"name": "loss_type", "val": ": str = 'sigmoid'"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "use_vllm", "val": ": bool = False"}, {"name": "vllm_model_impl", "val": ": str = 'vllm'"}, {"name": "vllm_structed_outputs_regex", "val": ": str |无 = 无"}, {"name": "vllm_gpu_memory_utilization", "val": ": float |无 = 0.55"}, {"name": "vllm_mode", "val": ": str = 'colocate'"}, {"name": "vllm_server_base_url", "val": ": str |无 = 无"}, {"name": "vllm_server_host", "val": ": str = '0.0.0.0'"}, {"name": "vllm_server_port", "val": ": int = 8000"}, {"name": "vllm_server_timeout", "val": ": float = 240.0"}, {"name": "vllm_group_port", "val": ": int = 51216"}, {"name": "vllm_tensor_parallel_size", "val": ": int = 1"}, {"name": "vllm_enable_sleep_mode", "val": ": bool = False"}, {"name": "ds3_gather_for_ Generation", "val": ": bool = True"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "reward_weights", "val": ": list[float] | None = None"}, {"name": "mixture_coef", "val": ": list = "}]}>
- **mixture_coef** （`float` 或 `list[float]`，*可选*，默认为 `0.5`）--
  模型和参考模型的 Logit 混合系数。如果提供了浮点数列表，则
  为每个新时期选择混合系数，最后一个系数用于其余时期
  纪元。

[experimental.nash_md.NashMDTrainer](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDTrainer) 的配置类别。

[experimental.online_dpo.OnlineDPOConfig](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOConfig) 的子类我们可以使用它的所有参数并添加以下内容：

### ORPO 训练师
https://huggingface.co/docs/trl/v1.9.2/orpo_trainer.md