<!-- huggingface-docs: machine-translated zh-CN from English source -->

#首席采购官培训师

[⟦T136⟧](https://huggingface.co/models?other=cpo,trl)

## 概述

对比偏好优化 (CPO)，如 [Haoran Xu](https://huggingface.co/haoranxu)、[Amr Sharaf](https://huggingface.co/amrsharaf)、[Yunmo Chen](https://huggingface.co/yunmochen)、Weiting Tan、Lingfeng Shen、Benjamin Van Durme、[Kenton Murray](https://huggingface.co/Kenton) 和 [Young Jin Kim](https://huggingface.co/ykim362) 在论文 [Contrastive Preference Optimization: Pushing the Boundaries of LLM Performance in Machine Translation](https://huggingface.co/papers/2401.08417) 中介绍的。在较高层面上，CPO 训练模型以避免在机器翻译 (MT) 任务中生成足够但不完美的翻译。然而，CPO 是 DPO 损失的一般近似值，可以应用于其他领域，例如聊天。

CPO 旨在减轻 SFT 的两个基本缺点。首先，SFT 最小化预测输出和黄金标准参考之间差异的方法本质上限制了训练数据质量水平的模型性能。其次，SFT 缺乏防止模型拒绝翻译错误的机制。 CPO 目标源自 DPO 目标。

## 快速开始

此示例演示如何使用 CPO 方法训练模型。我们使用[Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)作为基础模型。我们使用来自[UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback)的偏好数据。您可以在此处查看数据集中的数据：

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

以下是训练模型的脚本：

```python
# train_cpo.py
from datasets import load_dataset
from trl.experimental.cpo import CPOConfig, CPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

training_args = CPOConfig(output_dir="Qwen2-0.5B-CPO")
trainer = CPOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_cpo.py
```

## 预期的数据集类型CPO 需要[preference dataset](dataset_formats#preference)。 [experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer)支持[conversational](dataset_formats#conversational)和[standard](dataset_formats#standard)数据集格式。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。

## 示例脚本

我们提供了一个示例脚本来使用 CPO 方法训练模型。该脚本可在 [⟦T3⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/cpo.py) 中找到

要使用 [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/ultrafeedback_binarized) 上的 [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 测试 CPO 脚本，请运行以下命令：

```bash
accelerate launch examples/scripts/cpo.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/ultrafeedback_binarized \
    --num_train_epochs 1 \
    --output_dir Qwen2-0.5B-CPO
```

## 记录的指标

在培训和评估时，我们记录以下指标：

* `rewards/chosen`：所选响应的策略模型的平均对数概率按 beta 缩放
* `rewards/rejected`：被拒绝响应的策略模型的平均对数概率按 beta 缩放
* `rewards/accuracies`：所选奖励大于相应拒绝奖励的频率的平均值
* `rewards/margins`：选择的奖励和相应拒绝的奖励之间的平均差
* `nll_loss`：所选响应的策略模型的平均负对数似然损失

## CPO 变体

### 简单偏好优化 (SimPO)[Yu Meng](https://huggingface.co/yumeng5)、[Mengzhou Xia](https://huggingface.co/mengzhouxia)和[Danqi Chen](https://huggingface.co/cdq10131)提出的[Simple Preference Optimization](https://huggingface.co/papers/2405.14734) (SimPO) 提出了一种比 DPO 更简单、更有效的偏好优化算法，无需使用参考模型。 SimPO 的关键设计是 (1) 使用长度归一化对数似然作为隐式奖励，以及 (2) 在 Bradley-Terry 排名目标中纳入目标奖励裕度。官方代码可以在[princeton-nlp/SimPO](https://github.com/princeton-nlp/SimPO)找到。

论文摘要如下：

> 直接偏好优化 (DPO) 是一种广泛使用的离线偏好优化算法，可根据人类反馈 (RLHF) 重新参数化强化学习中的奖励函数，以增强简单性和训练稳定性。在这项工作中，我们提出了 SimPO，这是一种更简单但更有效的方法。 SimPO 的有效性归功于一个关键设计：使用序列的平均对数概率作为隐式奖励。这种奖励公式可以更好地与模型生成保持一致，并且无需参考模型，从而提高计算和内存效率。此外，我们在 Bradley-Terry 目标中引入了目标奖励幅度，以鼓励获胜和失败响应之间有更大的差距，进一步增强算法他们的表现。我们将 SimPO 与 DPO 及其最新变体在各种最先进的训练设置中进行比较，包括基础模型和指令调整模型，例如 Mistral 和 Llama3。我们评估了广泛的指令跟踪基准测试，包括 AlpacaEval 2、MT-Bench 和最近具有挑战性的 Arena-Hard 基准测试。我们的结果表明，SimPO 始终显着优于现有方法，而无需大幅增加响应长度。具体来说，SimPO 在 AlpacaEval 2 上的表现比 DPO 高出 6.4 分，在 Arena-Hard 上高出 7.5 分。我们基于 Llama3-8B-Instruct 构建的性能最佳模型，在 AlpacaEval 2 上实现了 44.7 的长度控制胜率，超越了排行榜上的 Claude 3 Opus，在 Arena-Hard 上实现了 33.8 的胜率，使其成为最强的 8B 开源模型。

SimPO 损失集成在[experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer) 中，因为它是一种替代损失，增加了奖励裕度，允许长度标准化，并且不使用 BC 正则化。要使用此损失，只需打开[experimental.cpo.CPOConfig](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOConfig)中的`loss_type="simpo"`和`cpo_alpha=0.0`并将`simpo_gamma`设置为推荐值即可。

### CPO-SimPO我们还提供 CPO 和 SimPO 的组合使用，从而实现更稳定的训练并提高性能。了解更多详情[CPO-SimPO GitHub](https://github.com/fe1ixxu/CPO_SIMPO)。要使用此方法，只需在 [experimental.cpo.CPOConfig](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOConfig) 中设置 `loss_type="simpo"` 和非零 `cpo_alpha` 即可启用 SimPO。

### 阿尔法PO

Aman Gupta、Shao Tang、Qingquan Song、Sirou Zhu、[Jiwoo Hong](https://huggingface.co/JW17)、Ankan Saha、Viral Gupta、Noah Lee、Eunki Kim、Jason Zhu、Natesh Pillai 和 S. Sathiya Keerthi 的[AlphaPO -- Reward shape matters for LLM alignment](https://huggingface.co/papers/2501.03884) (AlphaPO) 方法也在[experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer) 中实现。 AlphaPO 是一种替代方法，它在 SimPO 损失的情况下对奖励函数形状进行变换。论文摘要如下：> 人类反馈强化学习 (RLHF) 及其变体在有效协调大型语言模型 (LLM) 以遵循指令并反映人类价值观方面取得了巨大进步。最近，出现了直接对齐算法 (DAA)，其中通过将奖励直接描述为所学习策略的函数来跳过 RLHF 的奖励建模阶段。 DAA 的一些流行示例包括直接偏好优化 (DPO) 和简单偏好优化 (SimPO)。这些方法经常遭受似然位移的影响，这种现象导致首选响应的概率经常被不期望地降低。在本文中，我们认为，对于 DAA 来说，奖励（功能）形状很重要。我们引入了 AlphaPO，这是一种新的 DAA 方法，它利用 α 参数来帮助改变奖励函数的形状，使其超出标准对数奖励。 AlphaPO 有助于维持对可能性位移和过度优化的细粒度控制。与性能最佳的 DAA 之一 SimPO 相比，AlphaPO 使 Mistral-7B 和 Llama3-8B 的指令版本的对准性能相对提高约 7% 至 10%，同时实现 15% 的 to 与相同型号的 DPO 相比，相对改进 50%。所提供的分析和结果强调了奖励形状的重要性，以及如何系统地改变它以影响训练动态并提高对齐性能。

要按照论文中所述使用此损失，我们可以设置`loss_type="alphapo"`，它会自动将`loss_type="simpo"`和`cpo_alpha=0.0`以及`alpha`和`simpo_gamma`设置为[experimental.cpo.CPOConfig](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOConfig)中的推荐值。或者，您可以手动将`loss_type="simpo"`、`cpo_alpha=0.0`以及`alpha`和`simpo_gamma`设置为推荐值。此方法的其他变体也是可能的，例如将`loss_type="ipo"`和`alpha`设置为任何非零值。

## 损失函数

CPO 算法支持多种损失函数。损失函数可以使用[experimental.cpo.CPOConfig](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOConfig)中的`loss_type`参数进行设置。支持以下损失函数：| `loss_type=` |描述 |
| --- | --- |
| `"sigmoid"`（默认）|给定偏好数据，我们可以根据 Bradley-Terry 模型拟合二元分类器，事实上，[DPO](https://huggingface.co/papers/2305.18290) 作者提出了通过 `logsigmoid` 归一化似然的 sigmoid 损失来拟合逻辑回归。 |
| `"hinge"` | [RSO](https://huggingface.co/papers/2309.06657) 作者建议在 [SLiC](https://huggingface.co/papers/2305.10425) 论文中的归一化似然上使用铰链损失。在这种情况下，`beta`是边距的倒数。 |
| `"ipo"` | [IPO](https://huggingface.co/papers/2310.12036) 作者提供了对 DPO 算法更深入的理论理解，并识别了过度拟合的问题并提出了替代损失。在这种情况下，`beta`是所选完成对与拒绝完成对的对数似然比之间差距的倒数，因此`beta`越小，该差距就越大。根据论文，损失是对完成的对数似然进行平均的（与仅求和的 DPO 不同）。 || `"simpo"` | [SimPO](https://huggingface.co/papers/2405.14734)方法也在[experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer)中实现。 SimPO 是一种替代损失，它增加了奖励裕度，允许长度标准化，并且不使用 BC 正则化。要使用此损失，只需将[experimental.cpo.CPOConfig](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOConfig)和`simpo_gamma`中的`loss_type="simpo"`和`cpo_alpha=0.0`设置为推荐值即可。 |
| `"alphapo"` | [AlphaPO](https://huggingface.co/papers/2501.03884)方法也在[experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer)中实现。这是自动设置`loss_type="simpo"`和`cpo_alpha=0.0`的语法糖。当 `alpha` 参数非零时，AlphaPO 在 SimPO 损失的背景下对奖励函数形状进行变换。 |

### 对于混合专家模型：启用辅助损失

如果负载在专家之间平均分配，MOE 的效率最高。  
为了确保我们在偏好调整期间以类似的方式训练 MOE，将负载均衡器的辅助损失添加到最终损失中是有益的。

通过在模型配置中设置`output_router_logits=True`（例如[MixtralConfig](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/mixtral#transformers.MixtralConfig)）来启用此选项。  
要缩放辅助损失对总损失的贡献程度，请在模型配置中使用超参数`router_aux_loss_coef=...`（默认值：`0.001`）。

## CPOTrainer[[trl.experimental.cpo.CPOTrainer]]- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要训练的模型，最好是[AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification)。
- **参数** ([experimental.cpo.CPOConfig](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOConfig)) --
  用于训练的 CPO 配置参数。
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

初始化 CPOTrainer。

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
  传递给 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## CPOConfig[[trl.experimental.cpo.CPOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "max_completion_length", "val": ": int |无 = 无"}, {"name": "beta", "val": ": float = 0.1"}, {"name": "label_smoothing", "val": ": float = 0.0"}, {"name": "loss_type", "val": ": str = 'sigmoid'"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "cpo_alpha", "val": ": float = 1.0"}, {"name": "simpo_gamma", "val": ": float = 0.5"}, {"name": "alpha", "val": ": float = 0.0"}, {"name": "generate_during_eval", "val": ": bool = False"}, {"name": "is_encoder_decoder", "val": ": bool |无 = 无"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "dataset_num_proc", "val": ": int |无 = 无"}]}>
- **max_length**（`int`或`None`，*可选*，默认为`1024`）--
  批次中序列的最大长度（提示+完成）。如果您愿意，则需要此参数
  使用默认数据整理器。
- **max_completion_length** (`int`, *可选*) --
  完成的最大长度。如果您想使用默认数据整理器，则需要此参数
  你的模型是一个编码器-解码器。
- **beta**（`float`，*可选*，默认为`0.1`）--
  控制与参考模型的偏差的参数。较高的β意味着较小的偏差
  参考模型。对于 IPO 损失 (`loss_type="ipo"`)，β 是正则化参数，用 τ 表示
  [paper](https://huggingface.co/papers/2310.12036)。
- **label_smoothing**（`float`，*可选*，默认为`0.0`）--
  标签平滑因子。如果您想使用默认数据整理器，则需要此参数。- **loss_type** (`str`，*可选*，默认为`"sigmoid"`) --
  使用的损失类型。可能的值为：

  - `"sigmoid"`：原始[DPO](https://huggingface.co/papers/2305.18290)论文中的 sigmoid 损失。
  - `"hinge"`：将损失取决于归一化似然
    [SLiC](https://huggingface.co/papers/2305.10425)纸。
  - `"ipo"`：[IPO](https://huggingface.co/papers/2310.12036)论文的IPO损失。
  - `"simpo"`：[SimPO](https://huggingface.co/papers/2405.14734) 论文中的 SimPO 损失。
  - `"alphapo"`：[AlphaPO](https://huggingface.co/papers/2501.03884) 论文中的 AlphaPO 损失。这个
    自动设置`loss_type="simpo"`和`cpo_alpha=0.0`。

- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型中禁用 dropout。
- **cpo_alpha**（`float`，*可选*，默认为`1.0`）--
  CPO 训练中 BC 正则化器的权重。
- **simpo_gamma**（`float`，*可选*，默认为`0.5`）--
  SimPO 损失的目标奖励保证金，仅在`loss_type="simpo"` 时使用。
- **alpha**（`float`，*可选*，默认为`0.0`）--
  Alpha 参数控制所有损失类型的奖励函数形状。当 alpha=0（默认）时，使用
  标准对数概率奖励。当`alpha != 0`时，应用AlphaPO变换：`r = (1 - p^(-alpha))
  / alpha` 来自 [AlphaPO paper](https://huggingface.co/papers/2501.03884)。该参数适用于所有
  损失类型。
- **generate_during_eval** (`bool`，*可选*，默认为`False`) --如果`True`，则在评估期间生成模型的完成情况并将其记录到 W&B 或 Comet。
- **is_encoder_decoder** (`bool`, *可选*) --
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

[experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer) 的配置类。

此类仅包含特定于 CPO 培训的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以把这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在
命令行。> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`1e-6`，而不是`5e-5`。