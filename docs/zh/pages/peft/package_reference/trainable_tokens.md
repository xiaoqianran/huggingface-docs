<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 可训练代币

可训练令牌方法提供了一种针对特定令牌嵌入进行微调的方法，而无需求助于
训练完整的嵌入矩阵或在嵌入矩阵上使用适配器。它基于最初的实施
[here](https://github.com/huggingface/peft/pull/1541)。

该方法仅针对特定令牌并选择性地训练您指定的令牌索引。因此
与存储完整的微调嵌入矩阵相比，所需的 RAM 会更低，磁盘内存也显着降低。

使用[this script](https://github.com/huggingface/peft/blob/main/scripts/train_memory.py)获得的一些初步基准
建议对于 `gemma-2-2b` （具有相当大的嵌入矩阵），您可以使用可训练令牌节省约 4 GiB VRAM
过度微调嵌入矩阵。虽然 LoRA 将使用相当数量的 VRAM，但它也可能瞄准
您不想更改的令牌。请注意，这些只是指示，不同的嵌入矩阵大小可能会出现偏差
这些数字有点。请注意，此方法不会为您添加标记，您必须自己将标记添加到标记生成器并调整标记的大小
相应地模型的嵌入矩阵。此方法只会重新训练您指定的标记的嵌入。
此方法还可以与 LoRA 层结合使用！参见[the LoRA documentation](lora#efficiently-train-tokens-alongside-lora)。

> [!提示]
> 使用 [save_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.save_pretrained) 保存模型或使用检索状态字典
> [get_peft_model_state_dict()](/docs/peft/v0.20.0/en/package_reference/functional#peft.get_peft_model_state_dict) 添加新标记时可能会保存完整的嵌入矩阵而不是仅保存差异
> 作为预防措施，因为嵌入矩阵已调整大小。为了节省空间，您可以通过设置禁用此行为
> 调用 `save_pretrained` 时`save_embedding_layers=False`。只要您不修改
> 也可以通过其他方式嵌入矩阵，因为可训练令牌不会跟踪此类更改。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=TRAINABLE_TOKENS"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## TrainableTokensConfig[[peft.TrainableTokensConfig]]

#### peft.TrainableTokensConfig[[peft.TrainableTokensConfig]]

```python
peft.TrainableTokensConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, token_indices: list[int] = <factory>, target_modules: Optional[Union[list[str], str]] = None, init_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/trainable_tokens/config.py#L25)

**参数：**token_indices (`list[int]`) ：整数列表，表示您想要训练的标记的索引。要使用分词器查找标记的索引，您可以对字符串进行分词并查看返回的`input_ids`。索引的数量越接近代币的总量，该方法的效率就越低。

target_modules (`Optional[Union[list[str], str]]`) ：要替换为我们的 `TrainableTokensLayer` 的模块名称列表或模块名称的正则表达式。如果未定义，如果模型具有 `get_input_embeddings` 方法（变压器模型通常这样做），它将尝试获取模型的输入嵌入层，如果失败，则默认为“embed_tokens”。其他示例目标有 `embedding`、`encoder.embeddings` 或 `decoder.embeddings`。

init_weights (`bool`) ：默认情况下，新令牌权重被初始化为与相应令牌嵌入相同。这使得 TrainableTokens 在未训练时成为无操作。如果设置为`False`，权重将为随机值。除非您确切知道自己在做什么，否则请勿更改此设置。

`TrainableTokens` 方法的配置。允许训练新的标记（并重新训练现有的标记），而无需训练完整的嵌入矩阵。由
将一些选择的标记（由它们的索引标识）标记为可训练并保持其余部分不变，此方法可以
用于添加新令牌或更改现有令牌的嵌入，同时节省内存。两者的存储也是如此
与完全训练嵌入矩阵相比，工作内存使用量减少了。

请注意，可能尚未完全支持使用 FSDP/DeepSpeed 进行训练。

## TrainableTokensModel[[peft.TrainableTokensModel]]

#### peft.TrainableTokensModel[[peft.TrainableTokensModel]]

```python
peft.TrainableTokensModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/trainable_tokens/model.py#L26)

### 密苏里州
https://huggingface.co/docs/peft/v0.20.0/package_reference/miss.md