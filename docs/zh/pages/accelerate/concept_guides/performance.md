<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 比较分布式设置的性能

如果您不知道要寻找什么，那么评估和比较不同设置的性能可能会非常棘手。
例如，您无法使用 Accelerate 在 TPU、多 GPU 和单 GPU 上运行具有相同批处理大小的相同脚本 
并期望您的结果一致。 

但为什么？

本教程将介绍以下三个原因： 

1. **设置正确的种子**
2. **观察到的批次大小**
3. **学习率**

## 设置种子 

虽然这个问题还没有出现那么多，但请确保使用[utils.set_seed()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.set_seed)在所有分布式情况下完全设置种子，以便训练可以重现：

```python
from accelerate.utils import set_seed

set_seed(42)
```

为什么这很重要？在底层，这将设置 **5** 不同的种子设置：

```python
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # or torch.xpu.manual_seed_all, etc
    # ^^ safe to call this function even if cuda is not available
    if is_torch_xla_available():
        xm.set_rng_state(seed)
```

随机状态、numpy 的状态、torch、torch 的设备状态以及 TPU 可用时 torch_xla 的 cuda 状态。

## 观察到的批量大小 

使用 Accelerate 进行训练时，传递给数据加载器的批量大小是**每个 GPU 的批量大小**。这意味着什么 
两个 GPU 上的批处理大小 64 实际上是 128 的批处理大小。因此，在单个 GPU 上进行测试时需要考虑到这一点，
TPU 也是如此。下表可用作尝试不同批量大小的快速参考：

在此示例中，有两个用于“多 GPU”的 GPU 和一个具有 8 个工作线程的 TPU pod

|单 GPU 批量大小 |多 GPU 等效批量大小 | TPU 等效批量大小 |
|------------------------|------------------------------------------------|----------------------------------------|
| 256 | 256 128 | 128 32 | 32
| 128 | 128 64 | 64 16 | 16
| 64 | 64 32 | 32 8 |
| 32 | 32 16 | 16 4 |

## 学习率 

正如多个来源[[1](https://aws.amazon.com/blogs/machine-learning/scalable-multi-node-deep-learning-training-using-gpus-in-the-aws-cloud/)][[2](https://docs.nvidia.com/clara/clara-train-sdk/pt/model.html#classification-models-multi-gpu-training)]中所述，学习率应根据存在的设备数量“线性”缩放。下面的 
代码片段显示了使用 Accelerate 执行此操作：

由于用户可以定义自己的学习率调度程序，因此我们将其留给用户来决定是否希望扩展其学习率调度程序 
学习率与否。
 

```python
learning_rate = 1e-3
accelerator = Accelerator()
learning_rate *= accelerator.num_processes

optimizer = AdamW(params=model.parameters(), lr=learning_rate)
```您还会发现`accelerate`将根据正在训练的进程数量逐步调整学习率。这是因为 
前面提到的观察到的批量大小。因此，在 2 个 GPU 的情况下，学习率的步进频率将是单个 GPU 的两倍
考虑到批处理大小是原来的两倍（如果没有对单个 GPU 实例上的批处理大小进行更改）。

## 梯度累积和混合精度

当使用梯度累加和混合精度时，由于梯度平均的工作原理（累加）和精度损失（混合精度）， 
预计性能会有所下降。在比较不同计算之间的批量损失时，可以清楚地看到这一点 
设置。然而，训练结束时的总体损失、指标和总体表现应该大致相同。

### TPU 培训
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/training_tpu.md