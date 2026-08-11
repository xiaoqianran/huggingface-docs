<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 前缀调整

    

优化每个任务的前缀参数（图像源）。

[Prefix tuning](https://hf.co/papers/2101.00190) 将一系列特定于任务的向量添加到输入序列中，这些向量可以在保持预训练模型冻结的同时进行学习。前缀参数被插入到所有模型层中。

论文摘要是：

*微调是利用大型预训练语言模型执行下游任务的事实上的方法。但是，它会修改所有语言模型参数，因此需要为每个任务存储完整副本。在本文中，我们提出了前缀调优，这是自然语言生成任务微调的轻量级替代方案，它保持语言模型参数冻结，但优化了一个小的连续特定于任务的向量（称为前缀）。前缀调整从提示中汲取灵感，允许后续令牌关注此前缀，就好像它是“虚拟令牌”一样。我们将前缀调整应用于 GPT-2 以生成表到文本，并将前缀调整应用于 BART 以进行摘要。我们发现，通过仅学习 0.1% 的参数，前缀调优在完整数据设置中获得了可比较的性能，在低数据设置中优于微调，并推断出更好的结果r 包含训练期间未见过的主题的示例*。

**注意** 对于编码器-解码器模型 (seq2seq)，前缀仅应用于解码器，这与论文规范不符（参见图 2）。前缀调整仍然可以在这些模型架构上进行微调，但性能可能会低于标准；考虑对编码器-解码器模型使用其他 PEFT 方法。

前缀调整与[prompt tuning](../package_reference/prompt_tuning)非常相似。主要区别在于，前缀参数被插入到模型的**所有**层中，而提示调整仅将提示参数添加到模型输入嵌入中。前缀参数还通过单独的前馈网络（FFN）进行优化，而不是直接在软提示上进行训练，因为这会导致不稳定并损害性能。更新软提示后，FFN 将被丢弃。

结果，作者发现，尽管参数少了 1000 倍，但前缀调优的性能与完全微调模型相当，而且在低数据设置中表现甚至更好。

## 基本用法

创建一个[PrefixTuningConfig](/docs/peft/v0.20.0/en/package_reference/prefix_tuning#peft.PrefixTuningConfig)，其中包含要添加和学习的任务类型和虚拟令牌数量。

```py
from peft import PrefixTuningConfig, get_peft_model

peft_config = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
"trainable params: 983,040 || all params: 560,197,632 || trainable%: 0.1754809274167014"
```

## 可能的初始化默认情况下，前缀调整使用随机初始化的虚拟令牌。还有初始化向量的选项
接近无操作（初始化为零，它仍然会稍微改变概率质量）。
这意味着 KV 缓存注入的前缀从一开始就产生较小的影响，并减少了训练中的方差
性能。

PEFT 还提供实用程序来从现有的 KV 缓存前缀（例如，从
提示/语料库的第一个 `p` 标记）。仅当`prefix_projection=False`（默认）时才支持，因为
在这种情况下，学习到的参数就是 KV 前缀本身。

```py
from transformers import AutoModelForCausalLM, AutoTokenizer

from peft import PrefixTuningConfig, get_peft_model, initialize_kv_prefix_from_text

base = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")

peft_cfg = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, prefix_projection=False)
model = get_peft_model(base, peft_cfg)

initialize_kv_prefix_from_text(
    model,
    tok,
    text="...a long context with at least num_virtual_tokens tokens...",
    use_chat_template=False,
)m peft import PrefixTuningConfig, get_peft_model, initialize_kv_prefix_from_text

base = AutoModelForCausalLM.from_pretrained("gpt2")
tok = AutoTokenizer.from_pretrained("gpt2")

peft_cfg = PrefixTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, prefix_projection=False)
model = get_peft_model(base, peft_cfg)

initialize_kv_prefix_from_text(
    model,
    tok,
    text="...a long context with at least num_virtual_tokens tokens...",
    use_chat_template=False,
)

```

确保文本足够长以产生至少 `num_virtual_tokens` 令牌，否则初始化将失败。

作为指导方针：

* 使用`initialize_kv_prefix_from_text`以中性起始序列开始，它可以是一个非常短的字符串，例如
  “问题：”
* 如果这没有帮助，请使用与任务相关的较长序列（即设计的提示），为您提供更多虚拟
  适合的代币，但也有更多的模型指导
* 如果无法使用初始化文本或者您想快速检查前缀调整是否可行，
  使用不带投影的零初始化## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PREFIX_TUNING"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## PrefixTuningConfig[[peft.PrefixTuningConfig]]

#### peft.PrefixTuningConfig[[peft.PrefixTuningConfig]]

```python
peft.PrefixTuningConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, init_weights: typing.Optional[typing.Literal['zero']] = None, encoder_hidden_size: int = None, prefix_projection: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/prefix_tuning/config.py#L23)

**参数：**

init_weights (`Optional[str]`) ：如果未设置，则权重随机初始化，如果设置为“零”，则初始化权重，以便激活将是无操作（零）。

encoder_hidden_​​size (`int`) ：提示编码器的隐藏大小。

prefix_projection (`bool`) ：是否投影前缀嵌入。

这是存储[PrefixEncoder](/docs/peft/v0.20.0/en/package_reference/prefix_tuning#peft.PrefixEncoder)配置的配置类。

## PrefixEncoder[[peft.PrefixEncoder]]

#### peft.PrefixEncoder[[peft.PrefixEncoder]]

```python
peft.PrefixEncoder(config)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/prefix_tuning/model.py#L20)

**参数：**

config ([PrefixTuningConfig](/docs/peft/v0.20.0/en/package_reference/prefix_tuning#peft.PrefixTuningConfig)) ：前缀编码器的配置。

对前缀进行编码的 `torch.nn` 模型。

示例：

```py
>>> from peft import PrefixEncoder, PrefixTuningConfig

>>> config = PrefixTuningConfig(
...     peft_type="PREFIX_TUNING",
...     task_type="SEQ_2_SEQ_LM",
...     num_virtual_tokens=20,
...     token_dim=768,
...     num_transformer_submodules=1,
...     num_attention_heads=12,
...     num_layers=12,
...     encoder_hidden_size=768,
... )
>>> prefix_encoder = PrefixEncoder(config)
```

**属性**：
- **embedding** (`torch.nn.Embedding`) -- 前缀编码器的嵌入层。
- **transform** (`torch.nn.Sequential`) -- 用于变换前缀嵌入的两层 MLP 如果
  `prefix_projection` 是`True`。
- **prefix_projection** (`bool`) -- 是否投影前缀嵌入。

输入形状：(`batch_size`，`num_virtual_tokens`)

输出形状：(`batch_size`、`num_virtual_tokens`、`2*layers*hidden`)

#### load_prompt_embeddings[[peft.PrefixEncoder.load_prompt_embeddings]]

```python
load_prompt_embeddings(prompt_embeddings: Tensor)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/prefix_tuning/model.py#L89)加载 PEFT (`prompt_embeddings`) 保存的扁平化提示嵌入。

对于前缀调整，仅当`prefix_projection=False`时才支持，因为在这种情况下学习到的
参数是 KV 前缀本身（`embedding.weight` 具有形状`[num_virtual_tokens，
num_layers*2*token_dim]`)。

如果`prefix_projection=True`，参数是（虚拟令牌嵌入+ MLP）并且没有通用方法
反转投影以从展平的 KV 前缀中恢复这些参数。

### OSF（正交子空间微调）
https://huggingface.co/docs/peft/v0.20.0/package_reference/osf.md