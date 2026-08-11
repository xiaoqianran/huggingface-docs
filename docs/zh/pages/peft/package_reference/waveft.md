<!-- huggingface-docs: machine-translated zh-CN from English source -->

# WaveFT：小波微调

[WaveFT](https://huggingface.co/papers/2505.12532) 是一种新颖的参数高效微调（PEFT）方法，它在残差矩阵的**小波域**中引入稀疏更新。与受离散低秩选择限制的 LoRA 不同，WaveFT 通过直接学习变换空间中的一组稀疏系数，可以对可训练参数的数量进行细粒度控制。然后，这些系数通过离散小波逆变换 (IDWT) 映射回权重域，从而产生高秩更新，而不会产生推理开销。

WaveFT目前有以下限制：

- 仅支持`nn.Linear`层。

论文摘要是：>有效地适应大型基础模型至关重要，尤其是在计算和内存预算紧张的情况下。 LoRA 等参数高效微调 (PEFT) 方法在少参数状态下提供有限的粒度和有效性。我们提出了小波微调（WaveFT），这是一种新颖的 PEFT 方法，可以学习残差矩阵小波域中的高度稀疏更新。 WaveFT 允许精确控制可训练参数，提供细粒度的容量调整，并且具有极低的参数数量（可能远少于 LoRA 的最小值），非常适合极端参数高​​效的场景。使用 Stable Diffusion XL 作为基线对个性化文本到图像生成进行评估，WaveFT 显着优于 LoRA 和其他 PEFT 方法，尤其是在低参数数量下；实现卓越的主题保真度、快速对齐和图像多样性。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=WAVEFT"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## WaveFTConfig[[peft.WaveFTConfig]]

#### peft.WaveFTConfig[[peft.WaveFTConfig]]

```python
peft.WaveFTConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, n_frequency: int = 2592, scaling: float = 25.0, wavelet_family: str = 'db1', use_idwt: bool = True, random_loc_seed: int = 777, fan_in_fan_out: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, bias: str = 'none', modules_to_save: Optional[list[str]] = None, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, n_frequency_pattern: Optional[dict] = <factory>, proportional_parameters: bool = False, init_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/waveft/config.py#L27)

**参数：**n_Frequency (`int`)：离散小波变换 (DWT) 的可学习小波系数的数量。 'n_Frequency' 是一个大于 0 且小于或等于原始权重矩阵 (d_out * d_in) 中元素总数的整数。该参数直接控制每个适应层的可训练参数的数量。较高的“n_频率”通常会带来更好的性能，但也会增加 GPU 内存使用量，对训练速度影响较小。

缩放 (`float`) ：应用于重建的 delta W 矩阵的缩放因子。这是一个至关重要的超参数，类似于 LoRA 中的`lora_alpha`。它可以在超参数搜索期间进行调整。我们的 SDXL 个性化默认值为 25。

wavelet_family (`str`) ：用于 DWT 和逆 DWT (IDWT) 的小波族（例如 'db1'、'sym2'、'coif1'）。默认为“db1”（哈尔小波）。不同的小波族具有不同的滤波器长度，这会极大地影响训练时间use_idwt (`bool`) ：设置为 False 以实现高效适配。是否使用逆离散小波变换 (IDWT) 根据学习的小波系数重建增量权重。如果`True`（默认），则应用 IDWT。如果`False`，则直接使用学习到的系数形成稀疏增量权重矩阵，该矩阵速度更快，但对于 SDXL 个性化任务表现较差。

random_loc_seed (`int`) ：用于确定全小波系数矩阵中`n_frequency`可学习小波系数的随机位置的种子。

target_modules (`Union[list[str],str]`) ：模块名称列表或标识要与 WaveFT 适配的模块的正则表达式。例如，`['q_proj', 'v_proj']`或`'.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$'`。目前仅支持线性图层 (`torch.nn.Linear`)。

except_modules (`Optional[Union[List[str], str]]`) ：要从 WaveFT 适配中排除的模块名称列表或正则表达式。

fan_in_fan_out (`bool`) ：如果要替换的层的权重以`(fan_in, fan_out)`格式存储，则设置为`True`。默认为`False`。偏差 (`str`) ：WaveFT 的偏差类型。可以是“无”、“全部”或“waveft_only”。 （“fourier_only”可能是一个拼写错误，如果它暗示仅对调整后的参数有偏差，则已更正为“waveft_only”）如果“waveft_only”，则偏差仅添加到 WaveFT 组件。如果为“全部”，则偏差将添加到基础组件和 WaveFT 组件中。如果为“none”，则不会添加新的偏差。

module_to_save (`list[str]`) ：除了 WaveFT 层之外，还应标记为可训练并保存在最终检查点中的模块列表。对于随机初始化且需要训练的序列分类器或令牌分类任务等层很有用。

Layers_to_transform (`Union[list[int],int]`) ：要变换的特定层索引。如果提供，PEFT 将仅调整这些索引的层。如果给出单个整数，则仅转换该层。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层名称的模式，如果指定了 `layers_to_transform` 并且图层模式不是标准的（例如，不是“layers”或“h”），则使用该模式。这应该针对模型中的 `nn.ModuleList` 属性。

n_Frequency_pattern (`dict`) ：将层名称（或正则表达式）映射到特定`n_frequency`值的字典，覆盖全局`n_frequency`。示例：`{"model.decoder.layers.0.encoder_attn.k_proj": 1000}`。init_weights (`bool`) ：可学习小波系数（谱）的初始化策略。如果`True`（默认），系数将初始化为零。如果`False`，则系数根据按小因子缩放的标准正态分布进行初始化。

比例参数（`bool`）：如果`True`，则`n_frequency`按比例分配给每层的`input_dim * output_dim`。默认为 `False`。注意：包含此选项是为了实验的彻底性，以便研究人员能够重现论文结果，而不是为了实际实用性，因为尚未确定有益的方案。

这是存储[WaveFTModel](/docs/peft/v0.20.0/en/package_reference/waveft#peft.WaveFTModel)配置的配置类。它用于定义
基于小波的微调 (WaveFT) 的参数，这是一种利用小波变换稀疏性的方法
用于预训练模型的参数高效微调。

## WaveFTModel[[peft.WaveFTModel]]

#### peft.WaveFTModel[[peft.WaveFTModel]]

```python
peft.WaveFTModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/waveft/model.py#L30)

### VerA：基于向量的随机矩阵适应
https://huggingface.co/docs/peft/v0.20.0/package_reference/vera.md