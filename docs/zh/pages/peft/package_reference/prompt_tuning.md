<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 及时调整

    

仅训练和存储一组明显较小的特定于任务的提示参数（图像源）。

[Prompt tuning](https://hf.co/papers/2104.08691) 在输入中添加特定于任务的虚拟提示，该输入由嵌入空间中的可训练向量组成。虚拟令牌参数的更新独立于冻结的预训练模型参数。

论文摘要是：

*在这项工作中，我们探索“提示调整”，这是一种简单而有效的机制，用于学习“软提示”来调节冻结的语言模型以执行特定的下游任务。与 GPT-3 使用的离散文本提示不同，软提示是通过反向传播学习的，并且可以调整以合并来自任意数量的标记示例的信号。我们的端到端学习方法大大优于 GPT-3 的“少样本”学习。更值得注意的是，通过使用 T5 对模型大小进行消融，我们表明即时调整在规模上变得更具竞争力：当模型超过数十亿个参数时，我们的方法“缩小了差距”并匹配了模型调整的强大性能（其中所有模型权重都得到调整）。这一发现尤其重要，因为大型模型的共享和服务成本高昂，而且能够为多个下游任务重用一个冻结模型可以减轻这一负担。我们的方法可以看作是 Li 和 Liang (2021) 最近提出的“前缀调整”的简化，并且我们提供了与此方法和其他类似方法的比较。最后，我们表明，与完整模型调整*相比，使用软提示调节冻结模型可以增强域转移的鲁棒性。

与[prefix tuning](../package_reference/prefix_tuning)相比，只有
第一层的输入接收虚拟令牌。

## 用法

有两个决定需要做出：将多少虚拟代币添加到
模型的输入 (`num_virtual_tokens`) - 这将定义有多少
将会有可训练的参数 - 以及这些令牌是如何初始化的。

创建一个 [PromptTuningConfig](/docs/peft/v0.20.0/en/package_reference/prompt_tuning#peft.PromptTuningConfig) ，其中包含任务类型、用于训练模型的初始提示调整文本、要添加和学习的虚拟标记数量以及标记生成器。

```py
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
peft_config = PromptTuningConfig(
    task_type="CAUSAL_LM",
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path="bigscience/bloomz-560m",
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 8,192 || all params: 559,222,784 || trainable%: 0.0014648902430985358"
```

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PROMPT_TUNING"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## PromptTuningConfig[[peft.PromptTuningConfig]]

#### peft.PromptTuningConfig[[peft.PromptTuningConfig]]

```python
peft.PromptTuningConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, prompt_tuning_init: typing.Union[peft.tuners.prompt_tuning.config.PromptTuningInit, str] = <PromptTuningInit.RANDOM: 'RANDOM'>, prompt_tuning_init_text: typing.Optional[str] = None, tokenizer_name_or_path: typing.Optional[str] = None, tokenizer_kwargs: typing.Optional[dict] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/prompt_tuning/config.py#L30)

**参数：**Prompt_tuning_init (Union[`PromptTuningInit`, `str`]) ：提示嵌入的初始化。 `TEXT` 将使用您的文本进行初始化。 `SAMPLE_VOCAB` 将使用模型词汇表中随机采样的标记进行初始化。 `RANDOM` 将使用随机采样的连续软令牌进行初始化（警告：采样的软令牌可能会落在嵌入流形之外）

Prompt_tuning_init_text (`str`, *可选*) ：初始化提示嵌入的文本。仅当 `prompt_tuning_init` 为 `TEXT` 时使用。

tokenizer_name_or_path (`str`, *可选*) ：分词器的名称或路径。仅当 `prompt_tuning_init` 为 `TEXT` 时使用。

tokenizer_kwargs (`dict`, *可选*) ：传递给`AutoTokenizer.from_pretrained`的关键字参数。仅当 `prompt_tuning_init` 为 `TEXT` 时使用。

这是存储[PromptEmbedding](/docs/peft/v0.20.0/en/package_reference/prompt_tuning#peft.PromptEmbedding)配置的配置类。

## PromptEmbedding[[peft.PromptEmbedding]]

#### peft.PromptEmbedding[[peft.PromptEmbedding]]

```python
peft.PromptEmbedding(config, word_embeddings)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/prompt_tuning/model.py#L24)

**参数：**

config ([PromptTuningConfig](/docs/peft/v0.20.0/en/package_reference/prompt_tuning#peft.PromptTuningConfig)) ：提示嵌入的配置。

word_embeddings (`torch.nn.Module`) ：基础 Transformer 模型的词嵌入。

将虚拟令牌编码为提示嵌入的模型。

**属性**：
- **embedding** (`torch.nn.Embedding`) -- 提示嵌入的嵌入层。

示例：

```py
>>> from peft import PromptEmbedding, PromptTuningConfig

>>> config = PromptTuningConfig(
...     peft_type="PROMPT_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     prompt_tuning_init="TEXT",
...     prompt_tuning_init_text="Predict if sentiment of this review is positive, negative or neutral",
...     tokenizer_name_or_path="t5-base",
... )

>>> # t5_model.shared is the word embeddings of the base model
>>> prompt_embedding = PromptEmbedding(config, t5_model.shared)
```输入形状：（`batch_size`，`total_virtual_tokens`）

输出形状：（`batch_size`、`total_virtual_tokens`、`token_dim`）

### BEFT：低数据机制中语言模型的偏差有效微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/beft.md