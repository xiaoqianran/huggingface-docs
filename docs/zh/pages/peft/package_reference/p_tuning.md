<!-- huggingface-docs: machine-translated zh-CN from English source -->

# P 调整

    

提示标记可以插入输入序列中的任何位置，并且它们由提示编码器（图像源）进行优化。

[P-tuning](https://hf.co/papers/2103.10385) 专为自然语言理解（NLU）任务和所有语言模型而设计。

论文摘要是：

*虽然采用传统微调的 GPT 未能在自然语言理解 (NLU) 方面取得优异的结果，但我们表明，在 NLU 任务上，GPT 使用新颖的 P-tuning 方法（采用可训练的连续提示嵌入），可以优于或媲美类似大小的 BERT。在知识探测（LAMA）基准上，最好的 GPT 在测试期间没有提供任何额外文本的情况下恢复了 64\% (P@1) 的世界知识，这将之前的最佳成绩大幅提高了 20+ 个百分点。在 SuperGlue 基准测试中，GPT 在监督学习中的性能与类似大小的 BERT 相当，有时甚至更好。重要的是，我们发现 P-tuning 还提高了 BERT 在小样本和监督设置中的性能，同时大大减少了对即时工程的需求。因此，P 调优在少样本 SuperGlue 基准测试中优于最先进的方法。*。该方法将可训练的提示嵌入添加到由提示编码器优化的输入中，以找到更好的提示，从而无需手动设计提示。提示标记可以添加到输入序列中的任何位置，p-tuning 还引入了锚标记以提高性能。提示编码器（双向长短期记忆网络或 LSTM）用于优化提示参数。与前缀调整不同：

- 提示标记可以插入输入序列中的任何位置，并且不仅限于开头
- 提示标记仅添加到输入中，而不是添加到模型的每一层
- 引入 *anchor* 标记可以提高性能，因为它们指示输入序列中组件的特征

该论文的结果表明，P-tuning 比手动制作提示更有效，并且它使得类似 GPT 的模型能够在 NLU 任务上与类似 BERT 的模型竞争。

## 用法

创建一个[PromptEncoderConfig](/docs/peft/v0.20.0/en/package_reference/p_tuning#peft.PromptEncoderConfig)，其中包含任务类型、要添加和学习的虚拟标记的数量以及用于学习提示参数的编码器的隐藏大小。

```py
from peft import PromptEncoderConfig, get_peft_model

peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=128)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 300,288 || all params: 559,514,880 || trainable%: 0.05366935013417338"
```

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=P_TUNING"
	frameborder="0"
	width="850"
	height="1000"
>

# API## PromptEncoderConfig[[peft.PromptEncoderConfig]]

#### peft.PromptEncoderConfig[[peft.PromptEncoderConfig]]

```python
peft.PromptEncoderConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, encoder_reparameterization_type: typing.Union[str, peft.tuners.p_tuning.config.PromptEncoderReparameterizationType] = <PromptEncoderReparameterizationType.MLP: 'MLP'>, encoder_hidden_size: int = None, encoder_num_layers: int = 2, encoder_dropout: float = 0.0)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/p_tuning/config.py#L29)

**参数：**

coder_reparameterization_type (Union[`PromptEncoderReparameterizationType`, `str`]) ：要使用的重新参数化类型。

encoder_hidden_​​size (`int`) ：提示编码器的隐藏大小。

encoder_num_layers (`int`) ：提示编码器的层数。

encoder_dropout (`float`) ：提示编码器的丢失概率。

这是存储[PromptEncoder](/docs/peft/v0.20.0/en/package_reference/p_tuning#peft.PromptEncoder)配置的配置类。

## PromptEncoder[[peft.PromptEncoder]]

#### peft.PromptEncoder[[peft.PromptEncoder]]

```python
peft.PromptEncoder(config)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/p_tuning/model.py#L24)

**参数：**

config ([PromptEncoderConfig](/docs/peft/v0.20.0/en/package_reference/p_tuning#peft.PromptEncoderConfig)) ：提示编码器的配置。

用于生成用于 p-tuning 的虚拟令牌嵌入的提示编码器网络。

示例：

```py
>>> from peft import PromptEncoder, PromptEncoderConfig

>>> config = PromptEncoderConfig(
...     peft_type="P_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     encoder_reparameterization_type="MLP",
...     encoder_hidden_size=768,
... )

>>> prompt_encoder = PromptEncoder(config)
```**属性**：
- **embedding** (`torch.nn.Embedding`) -- 提示编码器的嵌入层。
- **mlp_head** (`torch.nn.Sequential`) -- 提示编码器的 MLP 头，如果`inference_mode=False`。
- **lstm_head** (`torch.nn.LSTM`) -- 提示编码器的 LSTM 头，如果 `inference_mode=False` 且
`encoder_reparameterization_type="LSTM"`。
- **token_dim** (`int`) -- 基础变压器模型的隐藏嵌入维度。
- **input_size** (`int`) -- 提示编码器的输入大小。
- **output_size** (`int`) -- 提示编码器的输出大小。
- **hidden_​​size** (`int`) -- 提示编码器的隐藏大小。
- **total_virtual_tokens** (`int`): 虚拟代币总数
提示编码器。
- **encoder_type** (Union[`PromptEncoderReparameterizationType`, `str`]): 提示符的编码器类型
  编码器。

输入形状：(`batch_size`, `total_virtual_tokens`)

输出形状：(`batch_size`、`total_virtual_tokens`、`token_dim`)

### VB-LoRA：使用向量库进行极端参数高效微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/vblora.md