<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 情境感知提示调整：利用对抗性方法推进情境学习

    

CPT 仅优化特定的标记嵌入，同时保持模型的其余部分冻结（图像源）。

[Context-Aware Prompt Tuning (CPT)](https://huggingface.co/papers/2410.17222) 旨在通过仅细化上下文嵌入来增强小样本分类。
这种方法结合了上下文学习（ICL）、[Prompt Tuning](../package_reference/prompt_tuning)（PT）和对抗性优化的思想，重点是使模型适应参数高效且有效。
在 CPT 中，仅优化特定的上下文标记嵌入，而模型的其余部分保持冻结。
为了防止过度拟合并保持稳定性，CPT 使用受控扰动将上下文嵌入允许的更改限制在定义的范围内。
此外，为了解决新近度偏差现象（即上下文末尾附近的示例往往优先于较早的示例），CPT 应用了衰减损失因子。

论文摘要是：> 大型语言模型 (LLM) 可以使用基于优化的方法或上下文学习 (ICL) 执行少量学习。基于优化的方法经常会出现过度拟合的情况，因为它们需要用有限的数据更新大量参数。相比之下，ICL 避免了过度拟合，但与基于优化的方法相比通常表现不佳，并且对演示示例的选择、顺序和格式高度敏感。为了克服这些挑战，我们引入了上下文感知提示调整（CPT），这是一种受 ICL、提示调整（PT）和对抗性攻击启发的方法。 CPT 建立在 ICL 策略的基础上，即在输入之前连接示例，并通过合并类似 PT 的学习来扩展该策略，通过迭代优化来细化上下文嵌入，从训练示例中提取更深入的见解。考虑到上下文中示例的独特结构，我们的方法仔细修改了特定的上下文标记。除了通过类似 PT 的优化来更新上下文之外，CPT 还从对抗性攻击中汲取灵感，根据上下文中存在的标签调整输入，同时保留用户提供的固有价值。数据。为了确保优化过程中的鲁棒性和稳定性，我们采用了投影梯度下降算法，限制令牌嵌入保持接近其原始值并保障上下文的质量。我们的方法使用各种 LLM 模型在多个分类任务中展示了卓越的准确性，优于现有基线，并有效解决了少样本学习中的过度拟合挑战。

请查看 [Example](https://github.com/huggingface/peft/blob/main/examples/cpt_finetuning/README.md)，获取有关如何使用 CPT 训练模型的分步指南。

## 基准概述

这种方法还没有基准。欢迎贡献一个实验
配置，但确保首先创建一个问题
[here](https://github.com/huggingface/peft/issues)。

# API

## CPTConfig[[peft.CPTConfig]]

#### peft.CPTConfig[[peft.CPTConfig]]

```python
peft.CPTConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None, cpt_token_ids: typing.Optional[list[int]] = None, cpt_mask: typing.Optional[list[int]] = None, cpt_tokens_type_mask: typing.Optional[list[int]] = None, opt_weighted_loss_type: typing.Optional[typing.Literal['none', 'decay']] = 'none', opt_loss_decay_factor: typing.Optional[float] = 1.0, opt_projection_epsilon: typing.Optional[float] = 0.1, opt_projection_format_epsilon: typing.Optional[float] = 0.1, tokenizer_name_or_path: typing.Optional[str] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cpt/config.py#L23)

CPT 配置类扩展了 PeftConfig，用于上下文感知提示调整 (CPT)。

该类引入了CPT所需的附加参数，例如：
- 令牌类型掩码
- 提示调整初始化
- 损失加权
- 投影设置

有关更多详细信息，请参阅论文：https://huggingface.co/papers/2410.17222

## CPTEmbedding[[peft.CPTEmbedding]]

#### peft.CPTEmbedding[[peft.CPTEmbedding]]

```python
peft.CPTEmbedding(config, word_embeddings)
```[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cpt/model.py#L23)

CPTEmbedding 是一个自定义嵌入层，专为 PEFT 中的上下文感知提示调整 (CPT) 而设计。它初始化
嵌入，应用特定于提示的投影，并使用标签掩码计算损失。

####计算损失[[peft.CPTMedding.calculate_loss]]

```python
calculate_loss(base_model_output, labels, cpt_type_mask, config)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cpt/model.py#L143)

**参数：**

base_model_output (ModelOutput) ：包含 logits 的基本模型的输出。

labels (torch.Tensor) ：输入标记的真实标签。

cpt_type_mask (torch.Tensor) ：用于过滤有效损失项的令牌类型掩码。

config（命名空间）：包含与丢失相关的超参数的配置对象。

**返回：** `ModelOutput`

具有计算损失的基本模型输出。

计算具有可选指数衰减的 CPT 模型的损失。

#### 转发[[peft.CPTEmbedding.forward]]

```python
forward(indices)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cpt/model.py#L65)

**参数：**

indexs (torch.Tensor) ：要嵌入的令牌的索引。

**返回：** `torch.Tensor`

提示嵌入和增量嵌入的总和。

计算提示嵌入并应用增量调整。

#### get_projection[[peft.CPTMedding.get_projection]]

```python
get_projection()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cpt/model.py#L125)

将基于 epsilon 的投影应用于 delta 嵌入以控制其范数。#### set_updated_tokens[[peft.CPTEmbedding.set_updated_tokens]]

```python
set_updated_tokens()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/cpt/model.py#L86)

设置向后钩子以根据 CPT 令牌类型掩码有选择地更新令牌梯度。

### 及时调整
https://huggingface.co/docs/peft/v0.20.0/package_reference/prompt_tuning.md