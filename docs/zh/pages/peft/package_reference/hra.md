<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通过 Householder Reflection Adaptation (HRA) 缩小低阶适应和正交适应之间的差距

    

通过家庭反射适应弥合低阶适应和正交适应之间的差距

[HRA](https://huggingface.co/papers/2405.17484)提供了连接LoRA和OFT的新视角，这意味着它可以利用两种策略的优势，通过利用[Householder reflections](https://en.wikipedia.org/wiki/Householder_transformation)减少参数和计算成本，同时惩罚预训练知识的损失。它始终以更少的可训练参数实现更好的性能，并且在不同模型（包括大型语言模型（LLM）和条件图像生成器）上优于最先进的适配器。HRA 构建了一系列`r` 可训练的 Householder 反射 (HR)。由于Householder反射矩阵是正交矩阵，且正交矩阵的乘积也是正交矩阵，因此HRA满足正交微调（OFT）的理论保证。同时，HRA也可以被视为低阶微调适配器。 `r`越高，可训练的参数越多，模型容量越大，性能越好。此外，由于链式结构，HR平面的正交性影响HRA的容量和规律性。为了实现模型容量和正则性之间的权衡，将 HR 平面的正交正则化器添加到损失函数中。权重 \\(\lambda\\) 可以控制正则化器的强度。

论文摘要是：> 尽管遵循不同的技术路线，低秩和正交适应技术都可以基于一小部分可训练参数有效地适应特定任务或领域的大规模预训练模型。在本研究中，我们弥合了这两种技术之间的差距，提出了一种基于 Householder 反思的简单但有效的适应方法。给定预先训练的模型，我们的方法通过将每个冻结权重矩阵与由可学习的 Householder 反射 (HR) 链构建的正交矩阵相乘来微调其层。这种基于 HR 的正交微调相当于自适应低秩适应。此外，我们还表明，与 HR 相对应的反射平面的正交性会影响模型的容量和规律性。该分析促使我们对 HR 的正交性进行正则化，从而导致所提出的 Householder 反射适应 (HRA) 方法的不同实现。与最先进的方法相比，HRA 在适应大型语言模型和条件图像生成器时以更少的可学习参数实现了卓越的性能。该代码可在 [peft](https://github.com/huggingface/peft/tree/main/src/peft/tuners/hra) 和 [HRA](https://github.com/DaShenZi721/HRA) 获取。

# API## HRAConfig[[peft.HRAConfig]]

#### pft.HRAConfig[[peft.HRAConfig]]

```python
peft.HRAConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, apply_GS: bool = False, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, bias: str = 'none', modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hra/config.py#L25)

**参数：**

r (`int`) ：不同层的 HRA 排名。最好将“r”设置为偶数；否则，默认的初始化方法将不起作用。

apply_GS (`bool`) ：是否应用 Gram-Schmidt 正交化。

target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果将其指定为“全线性”，则选择所有线性模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

init_weights (`bool`) ：是否执行HRA权重的初始化。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

偏差 (`str`) ：HRA 的偏差类型。可以是 `'none'`、`'all'` 或 `'hra_only'`。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

这是存储[HRAModel](/docs/peft/v0.20.0/en/package_reference/hra#peft.HRAModel)配置的配置类。

## HRAModel[[peft.HRAModel]]

#### peft.HRAModel[[peft.HRAModel]]

```python
peft.HRAModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/hra/model.py#L24)

**参数：**model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。

config ([HRAConfig](/docs/peft/v0.20.0/en/package_reference/hra#peft.HRAConfig)) ：HRA 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

HRA 模型。

从预训练模型创建 Householder 反射适应 (HRA) 模型。该方法描述于
https://huggingface.co/papers/2405.17484

示例：
```py
>>> from diffusers import StableDiffusionPipeline
>>> from peft import HRAModel, HRAConfig

>>> config_te = HRAConfig(
...     r=8,
...     target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
...     init_weights=True,
... )
>>> config_unet = HRAConfig(
...     r=8,
...     target_modules=[
...         "proj_in",
...         "proj_out",
...         "to_k",
...         "to_q",
...         "to_v",
...         "to_out.0",
...         "ff.net.0.proj",
...         "ff.net.2",
...     ],
...     init_weights=True,
... )

>>> model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
>>> model.text_encoder = HRAModel(model.text_encoder, config_te, "default")
>>> model.unet = HRAModel(model.unet, config_unet, "default")
```

**属性**：
- **model** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([HRAConfig](/docs/peft/v0.20.0/en/package_reference/hra#peft.HRAConfig))：HRA 模型的配置。

### 阿达洛拉
https://huggingface.co/docs/peft/v0.20.0/package_reference/adalora.md