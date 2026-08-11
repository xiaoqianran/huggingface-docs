<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 管道并行性

Accelerate 支持使用 PyTorch [torch.distributed.pipelining](https://pytorch.org/docs/stable/distributed.pipelining.html) API 进行大规模训练的管道并行性。

## 准备_pippy[[accelerate.prepare_pippy]]

####加速.prepare_pippy[[accelerate.prepare_pippy]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/inference.py#L126)

包装 `model` 以进行管道并行推理。

**参数：**

model (`torch.nn.Module`) ：我们想要拆分用于管道并行推理的模型

split_points（`str`或`List[str]`，默认为“auto”）：如何生成分割点并在每个GPU上对模型进行分块。 “auto”将找到给定任何模型的最佳平衡分割。应该是模型中要按其他方式分割的图层名称列表。

no_split_module_classes (`List[str]`) ：我们不想拆分的层的类名称列表。

example_args（模型输入元组）：对*单个流程*使用基于订单的输入的模型的预期输入。如果可能的话建议使用此方法。

example_kwargs（模型输入的字典）：使用基于字典的输入进行*单个进程*的模型的预期输入。这是一个“高度”限制的结构，要求“所有”推理调用中都存在相同的密钥。除非先决条件对于所有情况都成立，否则不建议使用。num_chunks（`int`，默认为可用 GPU 的数量）：管道将具有的不同阶段的数量。默认情况下，它会为每个 GPU 分配一个块，但这可以进行调整和使用。一般来说，应该有 num_chunks >= num_gpus。

Gather_output（`bool`，默认为`False`）：如果`True`，最后一个GPU（保存真实输出）的输出将发送到所有GPU。

### 数据加载器、优化器和调度器
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/torch_wrappers.md