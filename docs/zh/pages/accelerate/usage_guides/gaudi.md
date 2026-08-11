<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 英特尔高迪

用户可以利用英特尔 Gaudi AI 加速器显着加快且经济高效的模型训练和推理。
Intel Gaudi AI加速器家族目前包括三代产品：[Intel Gaudi 1](https://habana.ai/products/gaudi/)、[Intel Gaudi 2](https://habana.ai/products/gaudi2/)、[Intel Gaudi 3](https://habana.ai/products/gaudi3/)。每台服务器配备 8 个设备，称为 Habana 处理单元 (HPU)，在 Gaudi 3 上提供 128GB 内存，在 Gaudi 2 上提供 96GB 内存，在第一代 Gaudi 上提供 32GB 内存。有关底层硬件架构的更多详细信息，请查看[Gaudi Architecture Overview](https://docs.habana.ai/en/latest/Gaudi_Overview/Gaudi_Architecture.html)。

## 它是如何开箱即用的

如果检测到 Intel Gaudi 设备，则默认启用它。
要禁用它，请将`--cpu`标志传递给`accelerate launch`命令或在回答`accelerate config`问卷时回答相应的问题。

您可以直接运行以下脚本在Intel Gaudi上进行测试：

```bash
accelerate launch /examples/cv_example.py --data_dir images
```

## 限制

以下功能不属于 Accelerate 库，需要 [Optimum for Intel Gaudi](https://huggingface.co/docs/optimum/main/en/habana/index)：- `fast_ddp` 通过在梯度上应用全归约来实现 DDP，而不是 Torch DDP 包装器。
- `minimize_memory`，用于 fp8 训练，可以在前向和后向传递之间将 fp8 权重保留在内存中，从而以额外的 fp8 转换为代价减少内存占用。
- `context_parallel_size`，用于上下文/序列并行（CP/SP），并沿序列维度对网络输入和激活进行分区，以减少内存占用并提高吞吐量。

### 大模型推理
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/big_modeling.md