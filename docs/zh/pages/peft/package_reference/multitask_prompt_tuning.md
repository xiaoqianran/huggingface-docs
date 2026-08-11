<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 多任务提示调优

[Multitask prompt tuning](https://huggingface.co/papers/2303.02861) 将每个任务的软提示分解为单个学习的可转移提示，而不是每个任务的单独提示。单个学习的提示可以通过乘法低等级更新来适应每个任务。

论文摘要是：

*提示调优，即通过学习提示向量来调整基本预训练模型以适应每个任务，已成为一种有效地将大型语言模型适应多个下游任务的有前景的方法。然而，现有方法通常从头开始学习软提示向量，并且尚不清楚如何在多任务学习环境中利用提示向量利用丰富的跨任务知识。我们提出了多任务提示调整（MPT），它首先通过从多个特定于任务的源提示中提取知识来学习单个可转移的提示。然后，我们学习对此共享提示的乘法低等级更新，以有效地使其适应每个下游目标任务。对 23 个 NLP 数据集的大量实验表明，我们提出的方法优于最先进的方法，包括在某些情况下的完整微调基线，尽管仅调整了 0.035%尽可能多的特定于任务的参数*。

    

多任务提示调整可实现参数高效的迁移学习。

MPT由两个阶段组成：

1.源训练——对于每个任务，其软提示被分解为特定于任务的向量。将特定于任务的向量相乘以形成另一个矩阵W，并且在W和共享提示矩阵P之间使用Hadamard乘积来生成特定于任务的提示矩阵。特定于任务的提示被提炼成在所有任务之间共享的单个提示矩阵。该提示是通过多任务训练进行训练的。
2.目标适应-为了适应目标任务的单个提示，目标提示被初始化并表示为共享提示矩阵和特定于任务的低秩提示矩阵的Hadamard乘积。

    

迅速分解。

## 基准概述

这种方法还没有基准。欢迎贡献一个实验
配置，但确保首先创建一个问题
[here](https://github.com/huggingface/peft/issues)。

# API

## MultitaskPromptTuningConfig[[peft.MultitaskPromptTuningConfig]]

#### peft.MultitaskPromptTuningConfig[[peft.MultitaskPromptTuningConfig]]

```python
peft.MultitaskPromptTuningConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, prompt_tuning_init: typing.Union[peft.tuners.multitask_prompt_tuning.config.MultitaskPromptTuningInit, str] = <MultitaskPromptTuningInit.RANDOM: 'RANDOM'>, prompt_tuning_init_text: typing.Optional[str] = None, tokenizer_name_or_path: typing.Optional[str] = None, tokenizer_kwargs: typing.Optional[dict] = None, prompt_tuning_init_state_dict_path: typing.Optional[str] = None, prompt_tuning_init_task: typing.Optional[int] = 0, num_ranks: typing.Optional[int] = 1, num_tasks: typing.Optional[int] = 1)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/multitask_prompt_tuning/config.py#L37)## MultitaskPromptEmbedding[[peft.tuners.MultitaskPromptEmbedding]]

#### peft.tuners.MultitaskPromptEmbedding[[peft.tuners.MultitaskPromptEmbedding]]

```python
peft.tuners.MultitaskPromptEmbedding(config: MultitaskPromptTuningConfig, word_embeddings)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/multitask_prompt_tuning/model.py#L28)

### MonteCLoRA（蒙特卡罗低阶适应）
https://huggingface.co/docs/peft/v0.20.0/package_reference/lora_variant_monteclora.md