<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 路德

[RoAd](https://huggingface.co/papers/2409.00119) 是一种参数高效的微调技术，通过学习应用于隐藏维度对的一小组 2×2 旋转矩阵（和可选的缩放因子）来适应大型语言模型。与可训练参数低于 0.1% 的其他 PEFT 方法相比，RoAd 实现了具有竞争力或优越的性能。与 LoRA 的批量低秩更新不同，RoAd 的稀疏轮换重新构造为简单的逐元素操作，在处理同一批次的异构请求（即同时服务多个适配器）时产生显着更高的服务吞吐量。此外，RoAd 无缝集成到分布式交换干预框架中，将其稀疏的 2D 旋转解释为隐藏表示的学习子空间内的特定任务干预。这些正交子空间可以组合起来合并多个特定于任务的行为，例如多语言能力或指令遵循，无需额外的微调，从而在法学硕士中实现模块化、可解释的适应。与 LoRA 或类似方法相比，使用 RoAd 进行微调通常需要更高的学习率，约为 1e-3。目前 RoAd 仅支持线性层，可用于使用位和字节（4 位或 8 位）量化的模型。

有关在同一批次中使用不同 RoAd 适配器运行推理的信息，请参阅 [Inference with different LoRA adapters in the same batch](lora#inference-with-different-lora-adapters-in-the-same-batch)。

## 基准概述

<iframe
	src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=ROAD"
	frameborder="0"
	width="850"
	height="1000"
>

# API

## RoadConfig[[peft.RoadConfig]]

#### peft.RoadConfig[[peft.RoadConfig]]

```python
peft.RoadConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, variant: Union[str, RoadVariant] = 'road_1', group_size: int = 64, init_weights: bool = True, target_modules: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/road/config.py#L28)

**参数：**

变体 (Union[`RoadVariant`, `str`]) ：要使用的道路模型的变体。它可以是 road_1、road_2 或 road_4 之一。请参阅论文了解更多详细信息。 - road_1：对所有元素对使用相同的比例和角度。此变体的参数数量最少，它存储的数量等于 RoAd 所应用的每层参数的输出隐藏大小。 - road_2：每个元素使用相同的比例和角度。与 road_1 相比，该变体的参数数量是 road_1 的 2 倍。 - road_4：每个元素使用两种不同的比例和角度。与 road_1 相比，该变体的参数数量是 road_1 的 4 倍。group_size (`int`) ：组大小定义如何将元素分组到 2D 向量中以进行旋转。在每个组内，元素 0 与元素 group_size/2 配对，然后元素 1 与元素 group_size/2+1 配对，依此类推。这对模型性能没有影响，因为元素是无序的，但是当用于例如计算时，它会对推理速度产生一些影响。 VLLM。为了获得最佳速度，建议组大小至少为 32 或 64（默认值）。请注意，模型隐藏大小（或与张量并行性一起使用时每个分区的隐藏大小）必须可被 group_size 整除，因此对于非常小的模型，您可能需要减少此参数。

init_weights (`bool`) ：是否执行RoAd权重的初始化。target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果指定为“全线性”，则选择所有线性/Conv1D 模块（如果模型是 PreTrainedModel，则排除输出层）。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

module_to_save (`List[str]`) ：除 Road 层之外的模块列表，要设置为可训练并保存在最终检查点中。

这是存储[RoadModel](/docs/peft/v0.20.0/en/package_reference/road#peft.RoadModel)配置的配置类。 RoAd 适配器建议于
https://huggingface.co/papers/2409.00119。

## RoadModel[[peft.RoadModel]]

#### peft.RoadModel[[peft.RoadModel]]

```python
peft.RoadModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/road/model.py#L39)

### 希拉
https://huggingface.co/docs/peft/v0.20.0/package_reference/hira.md