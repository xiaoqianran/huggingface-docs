<!-- huggingface-docs: machine-translated zh-CN from English source -->

# BCO 培训师

[⟦T103⟧](https://huggingface.co/models?other=bco,trl)

TRL 支持二元分类器优化 (BCO)。
[BCO](https://huggingface.co/papers/2404.04656) 作者训练了一个二元分类器，其 logit 作为奖励，以便分类器将 {提示，选择的完成} 对映射到 1，将 {提示，拒绝的完成} 对映射到 0。
有关完整示例，请查看[⟦T4⟧](https://github.com/huggingface/trl/blob/main/examples/scripts/bco.py)。

## 预期的数据集类型

[experimental.bco.BCOTrainer](/docs/trl/v1.9.2/en/bco_trainer#trl.experimental.bco.BCOTrainer) 需要[unpaired preference dataset](dataset_formats#unpaired-preference)。
[experimental.bco.BCOTrainer](/docs/trl/v1.9.2/en/bco_trainer#trl.experimental.bco.BCOTrainer)支持[conversational](dataset_formats#conversational)和[standard](dataset_formats#standard)数据集格式。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。

## 预期的模型格式

BCO 培训师期望模型为 `AutoModelForCausalLM`，而 PPO 期望价值函数为 `AutoModelForCausalLMWithValueHead`。

## 使用`BCOTrainer`

有关详细示例，请查看 `examples/scripts/bco.py` 脚本。在高层次上，我们需要使用我们希望训练的`model`和参考`ref_model`来初始化`BCOTrainer`，我们将使用它来计算首选和拒绝响应的隐式奖励。

`beta`指隐式奖励的超参数，数据集包含上面列出的3个条目。请注意，`model`和`ref_model`需要具有相同的架构（即仅解码器或编码器-解码器）。

```python
from trl.experimental.bco import BCOConfig, BCOTrainer

training_args = BCOConfig(
    beta=0.1,
)

bco_trainer = BCOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
)
```

之后就可以调用：

```python
bco_trainer.train()
```## 底层分布匹配 (UDM)

在实际场景中，赞成和反对数据集可能具有不同的潜在提示分布。
考虑为用户反馈而部署的 LLM：如果模型在编写任务方面表现出色，但在编码方面表现不佳，那么赞成的数据集将主要由与写作相关的提示主导，而反对的数据集将主要包含与编码相关的提示。  
如果您所需和不需要的数据集中的提示差异很大，则启用 UDM 会很有用。  

选择嵌入模型和分词器：

```python
embedding_model = AutoModel.from_pretrained(your_model_id)
embedding_tokenizer = AutoTokenizer.from_pretrained(your_model_id)

# customize this function depending on your embedding model
def embed_prompt(input_ids, attention_mask, model):
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    return outputs.last_hidden_state.mean(dim=1)

embedding_model = Accelerator().prepare_model(self.embedding_model)
embedding_func = partial(embed_prompt, model=embedding_model)
```

设置`prompt_sample_size`定义选择多少个提示来训练UDM分类器并使用提供的嵌入函数开始训练：

```python
training_args = BCOConfig(
    beta=0.1,
    prompt_sample_size=512,
)

bco_trainer = BCOTrainer(
    model,
    model_ref,
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    embedding_func=embedding_func,
    embedding_tokenizer=self.embedding_tokenizer,
)

bco_trainer.train()
```

### 对于混合专家模型：启用辅助损失

如果负载在专家之间平均分配，MOE 的效率最高。  
为了确保我们在偏好调整期间以类似的方式训练 MOE，将负载均衡器的辅助损失添加到最终损失中是有益的。通过在模型配置（例如 MixtralConfig）中设置 `output_router_logits=True` 来启用此选项。  
要衡量辅助损失对总损失的贡献程度，请使用超参数`router_aux_loss_coef=...`（默认值：0.001）。

## BCOTrainer[[trl.experimental.bco.BCOTrainer]]

- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要训练的模型，最好是[AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForSequenceClassification)。
- **参考模型** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  拥抱脸部变形金刚模型，带有休闲语言造型头。用于隐式奖励计算
  和损失。如果没有提供参考模型，培训师将创建一个具有相同参考模型的参考模型
  架构作为要优化的模型。
- **参数** ([experimental.bco.BCOConfig](/docs/trl/v1.9.2/en/bco_trainer#trl.experimental.bco.BCOConfig)) --
  用于训练的参数。
- **train_dataset** (`Dataset`) --
  用于训练的数据集。
- **评估数据集** (`Dataset`) --
  用于评估的数据集。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，[BaseImageProcessor](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/image_processor#transformers.BaseImageProcessor)，[FeatureExtractionMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/feature_extractor#transformers.FeatureExtractionMixin)或[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理类用于处理数据。如果提供，将用于自动处理输入
  对于模型，它将与模型一起保存，以便更容易重新运行中断的训练或
  重用微调后的模型。
- **data_collat​​or**（`DataCollator`，*可选*）--用于训练的数据整理器。如果没有指定，则默认数据整理器
  将使用 (`experimental.utils.DPODataCollatorWithPadding`) 将序列填充到
  给定配对序列的数据集，批次中序列的最大长度。
- **model_init** (`Callable[[], transformers.PreTrainedModel]`) --
  用于训练的模型初始值设定项。如果未指定 None，则默认模型初始值设定项将为
  使用过。
- **回调** (`list[transformers.TrainerCallback]`) --
  用于训练的回调。
- **优化器** (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`) --
  用于训练的优化器和调度器。
- **preprocess_logits_for_metrics** (`Callable[[torch.Tensor, torch.Tensor], torch.Tensor]`) --
  在计算指标之前用于预处理 logits 的函数。
- **peft_config**（`PeftConfig`，*可选*）--
  用于训练的 PEFT 配置。如果您传递 PEFT 配置，模型将被包装在
  PEFT 模型。
- **计算指标**（`Callable[[EvalPrediction], dict]`，*可选*）--
  用于计算指标的函数。必须采用 `EvalPrediction` 并返回字典字符串
  度量值。
- **model_adapter_name**（`str`，默认为`None`）--
  当 LoRA 与多个适配器一起使用时，训练目标 PEFT 适配器的名称。
- **ref_adapter_name** (`str`，默认为`None`) --当 LoRA 与多个适配器一起使用时，参考 PEFT 适配器的名称。
- **embedding_func** (`Callable`，*可选*) --
  计算提示嵌入的函数，用于训练底层分布匹配（UDM）分类器
  当所需和不需要的数据集具有不同的提示分布时。需要 scikit-learn
  和 joblib 库。
- **embedding_tokenizer**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)，*可选*）--
  Tokenizer 用于为 `embedding_func` 准备提示。

从[BCO](https://huggingface.co/papers/2404.04656)论文初始化BCOTrainer。- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
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

## BCOConfig[[trl.experimental.bco.BCOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str |无 = 无"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "max_completion_length", "val": ": int |无 = 无"}, {"name": "beta", "val": ": float = 0.1"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "generate_during_eval", "val": ": bool = False"}, {"name": "is_encoder_decoder", "val": ": bool |无 = 无"}, {"name": "precompute_ref_log_probs", "val": ": bool = False"}, {"name": "model_init_kwargs", "val":": dict[str, Typing.Any] | str | None = None"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "dataset_num_proc", "val": ": int | None = None"}, {"name": "prompt_sample_size", "val": ": int = 1024"}, {"name": "min_密度_比率", "val": ": 浮点数 = 0.5"}, {"名称": "最大密度比率", "val": ": 浮点数 = 10.0"}]}>
- **max_length**（`int`或`None`，*可选*，默认为`1024`）--
  批次中序列的最大长度（提示+完成）。如果您愿意，则需要此参数
  使用默认数据整理器。
- **max_completion_length**（`int`，*可选*）--
  完成的最大长度。如果您想使用默认数据整理器，则需要此参数
  你的模型是一个编码器-解码器。
- **beta**（`float`，*可选*，默认为`0.1`）--
  控制与参考模型的偏差的参数。较高的β意味着较小的偏差
  参考模型。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型和参考模型中禁用 dropout。
- **generate_during_eval** (`bool`，*可选*，默认为`False`) --如果`True`，则生成模型和参考模型的完成情况并将其记录到 W&B 或 Comet
  评估期间。
- **is_encoder_decoder** (`bool`, *可选*) --
  当使用 `model_init` 参数（可调用）而不是 `model` 参数来实例化模型时，
  您需要指定可调用返回的模型是否是编码器-解码器模型。
- **precompute_ref_log_probs** (`bool`，*可选*，默认为`False`) --
  是否预先计算训练和评估数据集的参考模型对数概率。这是
  在没有参考模型的情况下进行训练时很有用，可以减少所需的总 GPU 内存。
- **model_init_kwargs**（`dict[str, Any]`，*可选*）--
  实例化模型时传递给 `AutoModelForCausalLM.from_pretrained` 的关键字参数
  来自字符串的参考模型。
- **trust_remote_code**（`bool`，*可选*，默认为`False`）--
  是否允许加载从 Hub 发送自定义 Python 代码的模型。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 对于模型和参考模型。
- **dataset_num_proc** (`int`，*可选*) --
  用于处理数据集的进程数。
- **prompt_sample_size**（`int`，*可选*，默认为`1024`）--馈送到密度比分类器的提示数量。
- **最小密度比**（`float`，*可选*，默认为`0.5`）--
  密度比的最小值。估计的密度比被限制在该值。
- **最大密度比**（`float`，*可选*，默认为`10.0`）--
  密度比的最大值。估计的密度比被限制在该值。

[experimental.bco.BCOTrainer](/docs/trl/v1.9.2/en/bco_trainer#trl.experimental.bco.BCOTrainer) 的配置类。

此类仅包含特定于 BCO 培训的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以把这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在
命令行。

> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`5e-7`，而不是`5e-5`。

### 培训定制
https://huggingface.co/docs/trl/v1.9.2/customization.md