<!-- huggingface-docs: machine-translated zh-CN from English source -->

# PEANuT：使用权重感知神经调整器进行参数高效适应

[PEANuT](https://arxiv.org/abs/2410.01870)是一种参数高效的微调技术，引入了
权重感知神经调整器从冻结的预训练权重本身生成适配器更新。而不是
学习 LoRA 中的纯线性低秩更新，PEANuT 以基本权重为条件调节适配器转换，
这使得更新规则更具表现力，同时保持可训练参数的数量较少。

PEANuT 使用输入投影 `A`、输出投影 `B` 以及可选的中间残差编码器/解码器
与非线性激活配对。这使得可以对比与权重无关的更复杂的更新模式进行建模
线性适配器，同时仍保持在 PEFT 设置内。

PEANuT 目前有以下权衡：

优点：
- 比线性低秩更新更高的理论表达力。
- 在类似预算下的一系列任务上，性能优于 LoRA。
- 在非常低参数的情况下效果很好，例如围绕 `0.2M` 可训练参数。缺点：
- 比 LoRA 更高的内存使用量，因为`ΔW`是在应用之前显式构造的。
- 训练和推理速度比 LoRA 慢，更深的中间层进一步增加了开销。
- 非线性可能需要更仔细的超参数调整，尤其是学习率和相关的优化设置。

如果这些权衡不适合您的用例，请考虑其他 PEFT 方法，例如 LoRA。

论文摘要是：

> 微调大型预训练基础模型通常会产生出色的下游性能，但更新所有参数时成本高昂。 LoRA 等参数高效微调 (PEFT) 方法通过引入轻量级更新模块来缓解这一问题，但它们通常依赖于与权重无关的线性近似，从而限制了其表达能力。在这项工作中，我们提出了 PEANuT，一种新颖的 PEFT 框架，它引入了权重感知神经调整器，紧凑的神经模块，可以根据冻结的预训练权重生成任务自适应更新。 PEANuT 提供了一种灵活而有效的方法来捕获复杂的更新模式，而无需进行完整的模型调整。我们从理论上证明 PEANuT 达到了同等或更大的效果与参数相当或更少的现有线性 PEFT 方法相比，具有更高的表达能力。跨越四个基准和二十多个数据集的广泛实验表明，PEANuT 在 NLP 和视觉任务中始终优于强大的基线，同时保持较低的计算开销。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=PEANUT"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## PeanutConfig[[peft.PeanutConfig]]

#### peft.PeanutConfig[[peft.PeanutConfig]]

```python
peft.PeanutConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 32, depth: int = 0, act_fn: str = 'relu', scaling: float = 1.0, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, init_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/peanut/config.py#L26)

**参数：**

r (`int`) ：PEANuT 等级。这是适配器使用的隐藏尺寸。与 LoRA 等级类似，较大的`r`会增加适配器容量和可训练参数。

深度 (`int`) ：PEANuT 中每个编码器/解码器侧的隐藏适配器层数。除了这些隐藏层之外，输入投影`A`和输出投影`B`始终存在。因此，`depth`必须是非负整数。  - `depth=0`：`A`、`B`。 - `depth=1`：`A`，一个编码器，一个解码器，`B`。 - `depth=2`：`A`，两个编码器，两个解码器，`B`。 - `depth=3`：`A`，三个编码器，三个解码器，`B`等。act_fn (`str`) ：PEANuT 网络中应用的非线性激活。这对应于普通 PyTorch 实现中的`non_linear`。默认为`"relu"`。支持`transformers.activations.ACT2FN`中可用的任何激活密钥，并且可以在不同的任务上表现更好。

缩放 (`float`) ：在将 PEANuT 输出添加到冻结基础层输出之前，应用于 PEANuT 输出的标量乘法器。最终的适配器贡献是`scaling * (x @ delta_w)`。

target_modules (`Union[List[str], str]`, *可选*) ：要应用 PEANuT 的模块的名称。可以是模块名称字符串列表（例如`['q_proj', 'v_proj']`）或正则表达式模式。

module_to_save (`List[str]`, *可选*) ：除 PEANuT 层之外的模块列表，要设置为可训练并保存在最终检查点中。

except_modules (`Union[List[str], str]`, *可选*) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。Layers_to_transform (`Union[list[int], int]`, *可选*) ：要变换的图层索引。如果指定了此参数，PEFT 将仅转换此列表中指定的图层索引。如果传递单个整数，PEFT 将仅转换该索引处的图层。

Layers_pattern (`Optional[Union[List[str], str]]`, *可选*) ：图层模式名称，仅在 `layers_to_transform` 不是 None 并且图层模式不在公共图层模式中时使用。

init_weights (`bool`) : 是否使用默认初始化方案初始化PEANuT适配器权重： - 如果`True`：除`B`之外的所有权重均使用Kaiminguniform初始化，`B`初始化为零。 - 如果`False`：所有权重（包括`B`）均使用 Kaiming 制服初始化。  将 `B` 初始化为零会使适配器以精确的无操作方式启动。

这是存储[PeanutModel](/docs/peft/v0.20.0/en/package_reference/peanut#peft.PeanutModel)配置的配置类。

注意事项：
PEANuT 使用重量感知途径，其中增量重量以基本重量为条件。 `A` 适配器
应用于基本权重的输出维度，因此 `A` 具有形状 `(out_dim -> r)` 而不是通常的形状
类似 LoRA 的方法使用`(in_dim -> r)`。

## PeanutModel[[peft.PeanutModel]]

#### peft.PeanutModel[[peft.PeanutModel]]

```python
peft.PeanutModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/peanut/model.py#L28)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([PeanutConfig](/docs/peft/v0.20.0/en/package_reference/peanut#peft.PeanutConfig)) ：PEANuT 模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

**返回：** `torch.nn.Module`

PEANuT PEFT 模型。

从预训练的 Transformer 模型创建 PEANuT 模型。

该方法在 https://arxiv.org/abs/2410.01870 中有详细描述。

**属性**：
- **model** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) -- 要适配的模型。
- **peft_config** ([PeanutConfig](/docs/peft/v0.20.0/en/package_reference/peanut#peft.PeanutConfig))：PEANuT 模型的配置。

### RandLora：大型模型的全秩参数高效微调
https://huggingface.co/docs/peft/v0.20.0/package_reference/randlora.md