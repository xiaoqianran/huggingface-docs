<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 骆驼适配器

    

LLaMA-Adapter：利用零初始注意力对语言模型进行高效微调

[Llama-Adapter](https://hf.co/papers/2303.16199)是一种专门为将Llama转变为指令跟随模型而设计的PEFT方法。 Llama 模型被冻结，并且仅学习一组以输入指令标记为前缀的适应提示。由于插入模型中的随机初始化模块可能会导致模型丢失一些现有知识，因此 Llama-Adapter 使用零门控的零初始化注意力逐步向模型添加指导提示。

论文摘要是：*我们提出了 LLaMA-Adapter，这是一种轻量级的适应方法，可以有效地将 LLaMA 微调为指令跟踪模型。使用 52K 自指令演示，LLaMA-Adapter 仅在冻结的 LLaMA 7B 模型上引入了 1.2M 可学习参数，并且在 8 个 A100 GPU 上进行微调的成本不到一小时。具体来说，我们采用一组可学习的适应提示，并将它们添加到更高转换器层的输入文本标记中。然后，提出了一种零门控的零初始注意机制，自适应地将新的教学线索注入LLaMA，同时有效地保留其预先训练的知识。通过高效的训练，LLaMA-Adapter 可以生成高质量的响应，可与具有完全微调的 7B 参数的 Alpaca 相媲美。此外，我们的方法可以简单地扩展到多模态输入，例如图像，用于图像条件 LLaMA，这在 ScienceQA 上实现了卓越的推理能力。我们在 https://github.com/ZrrSkywalker/LLaMA-Adapter* 发布了我们的代码。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=ADAPTION_PROMPT"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## AdaptionPromptConfig[[peft.AdaptionPromptConfig]]

#### peft.AdaptionPromptConfig[[peft.AdaptionPromptConfig]]

```python
peft.AdaptionPromptConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, target_modules: str = None, adapter_len: int = None, adapter_layers: int = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adaption_prompt/config.py#L25)存储[AdaptionPromptModel](/docs/peft/v0.20.0/en/package_reference/llama_adapter#peft.AdaptionPromptModel)的配置。

## AdaptionPromptModel[[peft.AdaptionPromptModel]]

#### peft.AdaptionPromptModel[[peft.AdaptionPromptModel]]

```python
peft.AdaptionPromptModel(model, configs: dict, adapter_name: str)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adaption_prompt/model.py#L25)

实现适应提示，如 https://huggingface.co/papers/2303.16199 中所述。

前 L 个注意力模块被替换为包含原始注意力模块的 AdaptedAttention 模块，但插入
带门的可训练提示（零初始化）。

关于多适配器模式的注意事项：
- 我们通过保存由适配器索引的 AdaptedAttention 模块的字典来存储不同适配器的状态
  名字。
- 每次我们切换适配器时，我们都会从模型中删除当前活动适配器的模块，并将它们存储起来
  在字典中，并将它们替换为新适配器的模块。
- 为了避免重复和可能不一致的状态，当前活动的适配器始终从
  字典。
- 禁用适配器也会导致模块从模型中删除。

#### add_adapter[[peft.AdaptionPromptModel.add_adapter]]

```python
add_adapter(adapter_name: str, config: AdaptionPromptConfig)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adaption_prompt/model.py#L60)

添加具有给定名称和配置的适配器。

####disable_adapter_layers[[peft.AdaptionPromptModel.disable_adapter_layers]]

```python
disable_adapter_layers()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adaption_prompt/model.py#L115)通过交换 AdaptedAttention 模块来禁用适配器层。

####enable_adapter_layers[[peft.AdaptionPromptModel.enable_adapter_layers]]

```python
enable_adapter_layers()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adaption_prompt/model.py#L110)

通过交换缓存的 AdaptedAttention 模块来启用适配器层。

#### set_adapter[[peft.AdaptionPromptModel.set_adapter]]

```python
set_adapter(adapter_name: str, inference_mode: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/adaption_prompt/model.py#L95)

将模型设置为使用具有给定名称的适配器。

### 洛拉
https://huggingface.co/docs/peft/v0.20.0/package_reference/lora.md