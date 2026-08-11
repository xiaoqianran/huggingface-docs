<!-- huggingface-docs: machine-translated zh-CN from English source -->

# FourierFT：离散傅里叶变换微调

[FourierFT](https://huggingface.co/papers/2405.03003) 是一种参数高效的微调技术，利用离散傅立叶变换来压缩模型的可调权重。该方法在 GLUE 基准测试和常见 ViT 分类任务中使用更少的参数，性能优于 LoRA。

FourierFT目前有以下限制：

- 仅支持`nn.Linear`层。
- 不支持量化层。

如果这些约束不适用于您的用例，请考虑其他方法。

论文摘要是：> 低阶适应（LoRA）最近在微调基础模型方面引起了人们的广泛兴趣。它通过合并低秩矩阵 A 和 B 来表示权重变化，有效减少了可训练参数的数量，即 Delta W=BA。尽管 LoRA 取得了进步，但在处理广泛的定制改编或更大的基础模型时，它仍面临存储挑战。在这项工作中，我们的目标是通过享受傅里叶变换强大的表现力来进一步压缩可训练参数。具体来说，我们引入 FourierFT，它将 Delta W 视为空间域中的矩阵，并且仅学习其谱系数的一小部分。利用经过训练的频谱系数，我们实施离散傅里叶逆变换来恢复 Delta W。根据经验，我们的 FourierFT 方法在各种任务（包括自然语言理解、自然语言生成、指令调优和图像分类）上显示出与 LoRA 相比可比或更好的性能，并且参数更少。例如，在 LLaMA2-7B 模型上执行指令调优时，FourierFT 仅以 0.064M 可训练参数超越 LoRA，而 LoRA 为 33.5M。

## 基准概述<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=FOURIERFT"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## FourierFTConfig[[peft.FourierFTConfig]]

#### peft.FourierFTConfig[[peft.FourierFTConfig]]

```python
peft.FourierFTConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, n_frequency: int = 1000, scaling: float = 150.0, random_loc_seed: Optional[int] = 777, fan_in_fan_out: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, bias: str = 'none', modules_to_save: Optional[list[str]] = None, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, n_frequency_pattern: Optional[dict] = <factory>, init_weights: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/fourierft/config.py#L25)

**参数：**

n_Frequency (`int`)：离散傅里叶变换的可学习频率数。 'n_Frequency' 是一个大于 0 且小于或等于 d^2 的整数（假设权重 W 的尺寸为 d × d）。此外，它是更新每个 delta W 权重所需的可训练参数的数量。 'n_Frequency' 将影响 PEFT 的性能和效率。具体来说，它对训练速度影响不大，但它的值越高（通常）会导致更大的 GPU 内存成本和更好的准确性。在相同`target_modules`的情况下，LoRA的参数数量是FourierFT的(2*d*r/n_Frequency)倍。以下有关“n_Frequency”的设置示例可供用户参考。对于 RoBERTa-large 模型的 NLU 任务，采用 'n_Frequency': 1000 几乎可以达到与 LoRA 中的 'r': 8 类似的结果。此时LoRA的参数数量约为FourierFT的16倍。对于 Vit-large 模型的图像分类任务，采用 'n_Frequency': 3000 几乎可以达到与 'r': 16 类似的结果在LoRA中，LoRA的参数数量约为FourierFT的11倍。

缩放 (`float`) : delta W 矩阵的缩放值。这是用于缩放的重要超参数，类似于 LoRA 方法中的“lora_alpha”参数。 “缩放”可以在超参数搜索过程中确定。但如果用户想跳过此过程，可以参考以下场景的设置。对于所有 NLU (GLUE) 任务中的 RoBERTa-base 和 RoBERTa-large 模型，此参数可以设置为 100.0 或 150.0。对于所有 LLaMA 系列模型的所有指令调整，此参数均可设置为 300.0。对于所有图像分类任务中的 ViT-base 和 ViT-large 模型，此参数可以设置为 300.0。

random_loc_seed (`int`) ：频率随机位置的种子，即频谱条目矩阵。

target_modules (`Union[list[str],str]`) ：要替换为 FourierFT 的模块名称列表或模块名称的正则表达式。例如，['q', 'v'] 或 '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'。仅支持线性层。except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

fan_in_fan_out (`bool`) ：如果要替换的层存储像 (fan_in, fan_out) 这样的权重，则将此设置为 True。

bias (`str`)：FourierFT 的偏置类型。可以是“无”、“全部”或“fourier_only”。

module_to_save (`list[str]`) ：除了 FourierFT 层之外的模块列表，要设置为可训练并保存在最终检查点中。例如，在序列分类或令牌分类任务中，最后一层`classifier/score`是随机初始化的，因此需要可训练和保存。

Layers_to_transform (`Union[list[int],int]`) ：要变换的图层索引，如果指定了此参数，PEFT 将仅变换此列表中指定的图层索引。如果传递单个整数，PEFT 将仅转换该索引处的图层。Layers_pattern (`Optional[Union[List[str], str]]`) ：图层模式名称，仅当 `layers_to_transform` 与 None 不同且图层模式不在公共图层模式中时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

n_Frequency_pattern (`dict`) ：从图层名称或正则表达式到 n_Frequency 的映射，与默认指定的不同。例如，`{model.decoder.layers.0.encoder_attn.k_proj: 1000`}。

init_weights (`bool`) ：傅里叶权重的初始化。如果谱图初始化为标准正态分布，则将其设置为 False（默认值）。如果频谱初始化为零，则将其设置为 True。

这是存储[FourierFTModel](/docs/peft/v0.20.0/en/package_reference/fourierft#peft.FourierFTModel)配置的配置类。

## FourierFTModel[[peft.FourierFTModel]]

#### peft.FourierFTModel[[peft.FourierFTModel]]

```python
peft.FourierFTModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/fourierft/model.py#L31)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([FourierFTConfig](/docs/peft/v0.20.0/en/package_reference/fourierft#peft.FourierFTConfig)) : FourierFT 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

傅立叶傅立叶变换模型。从预训练的 Transformer 模型创建 FourierFT 模型。

该方法在 https://huggingface.co/papers/2405.03003 中有详细描述。

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([FourierFTConfig](/docs/peft/v0.20.0/en/package_reference/fourierft#peft.FourierFTConfig))：傅里叶模型的配置。

### Lily：跨层低阶互联适应
https://huggingface.co/docs/peft/v0.20.0/package_reference/lily.md