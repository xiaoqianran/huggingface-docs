<!-- huggingface-docs: machine-translated zh-CN from English source -->

# ORPO 训练师

[⟦T91⟧](https://huggingface.co/models?other=orpo,trl)[⟦T92⟧](https://github.com/huggingface/smol-course/tree/main/2_preference_alignment)

## 概述

比值比偏好优化 (ORPO) 在 [ORPO: Monolithic Preference Optimization without Reference Model](https://huggingface.co/papers/2403.07691) 中由 [Jiwoo Hong](https://huggingface.co/JW17)、[Noah Lee](https://huggingface.co/nlee-208) 和 [James Thorne](https://huggingface.co/j6mes) 引入。

论文摘要如下：

> 虽然最近的语言模型偏好对齐算法已经证明了有希望的结果，但监督微调 (SFT) 仍然是实现成功收敛的必要条件。在本文中，我们研究了 SFT 在偏好对齐背景下的关键作用，强调对不受欢迎的生成方式进行较小的惩罚对于偏好对齐的 SFT 来说就足够了。在此基础上，我们引入了一种简单且创新的无参考模型的整体优势比偏好优化算法 ORPO，消除了额外的偏好调整阶段的必要性。我们从经验和理论上证明，优势比是在 SFT 期间对比从 125M 到 7B 的不同大小的受欢迎和不受欢迎风格的明智选择。具体来说，仅在 UltraFeedback 上使用 ORPO 微调 Phi-2 (2.7B)、Llama-2 (7B) 和 Mistral (7B) 就超越了最先进的语言模型的性能，超过 7B 和 13B para米：在 AlpacaEval_{2.0} 上实现高达 12.20%（图 1），在 IFEval（指令级松散，表 6）上实现高达 66.19%，在 MT-Bench 上实现高达 7.32（图 12）。我们发布了 Mistral-ORPO-alpha (7B) 和 Mistral-ORPO-beta (7B) 的代码和模型检查点。

它研究了 SFT 在偏好调整背景下的关键作用。使用偏好数据，该方法假设对不受欢迎的一代的轻微惩罚以及通过附加到 NLL 损失的简单对数优势比项对所选响应的强适应信号足以实现偏好对齐的 SFT。

因此，ORPO 是一种无参考模型的偏好优化算法，无需额外的偏好对齐阶段，从而节省了计算和内存。

官方代码可以在[xfactlab/orpo](https://github.com/xfactlab/orpo)找到。

这种后训练方法是由[Kashif Rasul](https://huggingface.co/kashif)、[Lewis Tunstall](https://huggingface.co/lewtun)和[Alvaro Bartolome](https://huggingface.co/alvarobartt)贡献的。

## 快速开始

此示例演示如何使用 ORPO 方法训练模型。我们使用[Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)作为基础模型。我们使用来自[UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback)的偏好数据。您可以在此处查看数据集中的数据：

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

以下是训练模型的脚本：

```python
# train_orpo.py
from datasets import load_dataset
from trl.experimental.orpo import ORPOConfig, ORPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = ORPOConfig(output_dir="Qwen2-0.5B-ORPO")
trainer = ORPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_orpo.py
```训练分布在 8 个 GPU 上，大约需要 30 分钟。您可以通过查看奖励图来验证训练进度。奖励幅度的增加趋势表明模型正在改进并随着时间的推移产生更好的响应。

![orpo qwen2 reward margin](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/orpo-qwen2-reward-margin.png)

要查看[trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-ORPO)的性能，您可以使用[Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models)。

$ 变形金刚聊天 trl-lib/Qwen2-0.5B-ORPO
<quentin_gallouedec>：
最好的编程语言是什么？

<trl-lib/Qwen2-0.5B-ORPO>：
确定最好的编程语言具有挑战性，因为没有一种语言是完美的，任务的复杂性和项目的类型是重要因素。一些流行的语言包括 Java、Python、JavaScript 和
C++。如果您对特定项目有特定需求或要求，那么选择最适合这些需求的语言非常重要。

为项目选择编程语言时需要考虑以下一些其他因素：• 语言熟练程度：好的编程语言更容易理解和使用，并且允许开发人员更有效地协作进行项目。
 • 易于使用：有一些工具和库可以使编程变得更容易，因此开发人员应该选择一种可以帮助他们更轻松地开始的语言。
 • 代码可读性：清晰简洁的代码库应该易于阅读和理解，尤其是在处理大型项目时。
 • 工具和框架支持：有许多可用于Python、Java 和JavaScript 的库，以及IDE 和静态代码分析工具等工具。
 • 辅助功能：某些语言和工具具有使残疾开发人员更容易使用的功能，例如对屏幕阅读器的支持。
 • 版本控制：随着项目的增长和复杂性的增加，版本控制工具有助于跟踪更改。

## 预期的数据集类型

ORPO 需要[preference dataset](dataset_formats#preference)。 [experimental.orpo.ORPOTrainer](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOTrainer)同时支持[conversational](dataset_formats#conversational)和[standard](dataset_formats#standard)数据集格式。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。虽然[experimental.orpo.ORPOTrainer](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOTrainer)支持显式和隐式提示，但我们建议使用显式提示。如果提供隐式提示数据集，训练器将自动从`"chosen"`和`"rejected"`列中提取提示。有关更多信息，请参阅[preference style](dataset_formats#preference)部分。

## 示例脚本

我们提供了一个示例脚本来使用 ORPO 方法训练模型。该脚本可在 [⟦T5⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py) 中找到

要使用 [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) 上的 [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 测试 ORPO 脚本，请运行以下命令：

```bash
accelerate launch examples/scripts/orpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --output_dir Qwen2-0.5B-ORPO
```

## 使用技巧

### 对于混合专家模型：启用辅助损失

如果负载在专家之间平均分配，MOE 的效率最高。  
为了确保我们在偏好调整期间以类似的方式训练 MOE，将负载均衡器的辅助损失添加到最终损失中是有益的。

通过在模型配置中设置`output_router_logits=True`（例如[MixtralConfig](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/mixtral#transformers.MixtralConfig)）来启用此选项。  
要缩放辅助损失对总损失的贡献程度，请在模型配置中使用超参数`router_aux_loss_coef=...`（默认值：`0.001`）。

## 记录的指标

在培训和评估时，我们记录以下指标：- `rewards/chosen`：所选响应的策略模型的平均对数概率按 beta 缩放
- `rewards/rejected`：被拒绝响应的策略模型的平均对数概率按 beta 缩放
- `rewards/accuracies`：所选奖励大于相应拒绝奖励的频率的平均值
- `rewards/margins`：所选奖励与相应拒绝奖励之间的平均差
- `log_odds_chosen`：所选响应与拒绝响应的平均对数优势比
- `log_odds_ratio`：`log(sigmoid(log_odds_chosen))`的平均值
- `nll_loss`：所选响应损失的 SFT 部分的平均负对数似然损失

## ORPOTrainer[[trl.experimental.orpo.ORPOTrainer]]- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要训练的模型，最好是[AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification)。
- **参数** ([experimental.orpo.ORPOConfig](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOConfig)) --
  用于训练的 ORPO 配置参数。
- **数据整理器** (`DataCollator`) --
  用于训练的数据整理器。如果没有指定，则默认数据整理器
  将使用 (`experimental.utils.DPODataCollatorWithPadding`) 将序列填充到
  给定配对序列的数据集，批次中序列的最大长度。
- **train_dataset** (`Dataset`) --
  用于训练的数据集。
- **eval_dataset** (`Dataset`) --
  用于评估的数据集。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，[BaseImageProcessor](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/image_processor#transformers.BaseImageProcessor)，[FeatureExtractionMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin)或[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理类用于处理数据。如果提供，将用于自动处理输入
  对于模型，它将与模型一起保存，以便更容易重新运行中断的训练或
  重用微调后的模型。
- **model_init** (`Callable[[], transformers.PreTrainedModel]`) --
  用于训练的模型初始值设定项。如果未指定 None，则默认模型初始值设定项将为
  使用过。
- **回调** (`list[transformers.TrainerCallback]`) --
  用于训练的回调。
- **优化器** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`) --
  用于训练的优化器和调度器。
- **preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`) --在计算指标之前用于预处理 logits 的函数。
- **peft_config**（`PeftConfig`，*可选*）--
  用于训练的 PEFT 配置。如果您传递 PEFT 配置，模型将被包装在
  PEFT 模型。
- **compute_metrics**（`Callable[[EvalPrediction], dict]`，*可选*）--
  用于计算指标的函数。必须采用 `EvalPrediction` 并返回字典字符串
  度量值。

初始化 ORPOTrainer。

- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于 `True`，加载 *args.output_dir* 中由前一个实例保存的最后一个检查点
  `Trainer`。如果存在，训练将从此处加载的模型/优化器/调度器状态恢复。
- **试用**（`optuna.Trial`或`dict[str, Any]`，*可选*）--
  用于超参数搜索的试运行或超参数字典。
- **ignore_keys_for_eval** (`list[str]`，*可选*) --
  模型输出中的键列表（如果它是字典），在以下情况下应忽略这些键：
  收集训练期间评估的预测。`~trainer_utils.TrainOutput`包含全局步数、训练损失和指标的对象。主要培训切入点。

将保存模型，以便您可以使用`from_pretrained()`重新加载它。

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
  传递到 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## ORPOConfig[[trl.experimental.orpo.ORPOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "max_completion_length", "val": ": int |无 = 无"}, {"name": "beta", "val": ": float = 0.1"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "padding_value", "val": ": int |无 = 无"}, {"name": "generate_during_eval", "val": ": bool = False"}, {"name": "is_encoder_decoder", "val": ": bool |无 = 无"}, {"name": "model_init_kwargs", "val": ": dict[str, 打字。任意] | STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "dataset_num_proc", "val": ": int |无 = 无"}]}>
- **max_length**（`int`或`None`，*可选*，默认为`1024`）--
  批次中序列的最大长度（提示+完成）。如果您愿意，则需要此参数
  使用默认数据整理器。
- **max_completion_length** (`int`, *可选*) --
  完成的最大长度。如果您想使用默认数据整理器，则需要此参数
  你的模型是一个编码器-解码器。
- **beta**（`float`，*可选*，默认为`0.1`）--
  控制 ORPO 损失中相对损失重量比率的参数。在
  [paper](https://huggingface.co/papers/2403.07691)，用λ表示。在
  [code](https://github.com/xfactlab/orpo)，记为`alpha`。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型中禁用 dropout。
- **padding_value**（`int`，*可选*）--
  要使用的填充值。如果`None`，则使用分词器的填充值。
- **generate_during_eval** (`bool`，*可选*，默认为`False`) --
  如果`True`，则在评估期间生成模型的完成情况并将其记录到 W&B 或 Comet。- **is_encoder_decoder** (`bool`, *可选*) --
  当使用 `model_init` 参数（可调用）而不是 `model` 参数来实例化模型时，
  您需要指定可调用返回的模型是否是编码器-解码器模型。
- **model_init_kwargs**（`dict[str, Any]`，*可选*）--
  从实例化模型时传递给 `AutoModelForCausalLM.from_pretrained` 的关键字参数
  字符串。
- **trust_remote_code**（`bool`，*可选*，默认为`False`）--
  是否允许加载从 Hub 发送自定义 Python 代码的模型。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained)。
- **dataset_num_proc** (`int`，*可选*) --
  用于处理数据集的进程数。

[experimental.orpo.ORPOTrainer](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOTrainer) 的配置类。

此类仅包含特定于 ORPO 训练的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以将这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在
命令行。> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`1e-6`，而不是`5e-5`。

### XPO 培训师
https://huggingface.co/docs/trl/v1.9.2/xpo_trainer.md