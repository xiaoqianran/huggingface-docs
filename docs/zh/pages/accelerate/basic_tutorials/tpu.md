<!-- huggingface-docs: machine-translated zh-CN from English source -->

#TPU训练

[TPU (Tensor Processing Unit)](https://cloud.google.com/tpu/docs/intro-to-tpu)是一种专门为高效训练模型而设计的硬件。 Accelerate 支持 TPU 训练，但有一些事情你应该注意，即图编译。本教程简要讨论了编译，有关更多详细信息，请查看 [Training on TPUs with Accelerate](../concept_guides/training_tpu) 指南。

## 编译

TPU 创建训练步骤中所有操作的图表，例如前向传递、后向传递和优化器步骤。这就是为什么第一个训练步骤总是需要一段时间，因为构建和编译该图需要时间。但是，一旦编译完成，它就会被缓存，并且所有后续步骤都会快得多。

关键是避免再次编译代码，否则训练速度会非常慢。这意味着您的所有操作必须完全相同：

* 批次中的所有张量必须具有相同的长度（例如，NLP 任务没有动态填充）
* 您的代码必须是静态的（例如，没有带有 for 循环的层，这些循环根据输入（例如 LSTM）而具有不同的长度）

## 重量绑定常见的语言模型设计是将嵌入层和 softmax 层的权重联系起来。但是，将模型移动到 TPU（您自己或将其传递给 [prepare()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.prepare) 方法）会破坏权重绑定，您需要重新绑定权重。

要在 TPU 脚本中添加特殊行为（如权重绑定），请首先将 `distributed_type` 设置为 `DistributedType.TPU`。然后你可以使用[tie_weights](https://huggingface.co/docs/transformers/v5.11.0/en/main_classes/model#transformers.PreTrainedModel.tie_weights)方法来绑定权重。

```py
if accelerator.distributed_type == DistributedType.TPU:
    model.tie_weights()
```