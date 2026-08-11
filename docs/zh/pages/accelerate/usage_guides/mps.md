<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在 Mac 上加速 PyTorch 训练

随着 PyTorch v1.12 版本的发布，开发人员和研究人员可以利用 Apple 芯片 GPU 显着加快模型训练速度。 
这解锁了在 Mac 上本地执行机器学习工作流程（例如原型设计和微调）的能力。
Apple 的 Metal Performance Shaders (MPS) 作为 PyTorch 的后端实现了这一点，并且可以通过新的 `"mps"` 设备使用。 
这将在 MPS Graph 框架和 MPS 提供的调整内核上映射计算图和原语。
更多信息请参考官方文档[Introducing Accelerated PyTorch Training on Mac](https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/)
和[MPS BACKEND](https://pytorch.org/docs/stable/notes/mps.html)。

### 使用 Apple 芯片进行训练和推理的好处

1. 使用户能够在本地训练更大的网络或批量大小
2. 由于统一的内存架构，减少了数据检索延迟，并为 GPU 提供了对完整内存存储的直接访问。 
因此，提高端到端性能。
3. 降低与基于云的开发相关的成本或额外本地 GPU 的需求。

**先决条件**：要安装支持 mps 的 torch， 
请关注这篇不错的中等文章[GPU-Acceleration Comes to PyTorch on M1 Macs](https://medium.com/towards-data-science/gpu-acceleration-comes-to-pytorch-on-m1-macs-195c399efcc1)。## 它是如何开箱即用的
它在配备 MPS 的 Apple Silicon GPU 的 MacO 机器上默认启用。
要禁用它，请将`--cpu`标志传递给`accelerate launch`命令或在回答`accelerate config`问卷时回答相应的问题。

您可以直接运行以下脚本在启用 MPS 的 Apple Silicon 机器上进行测试：
```bash
accelerate launch /examples/cv_example.py --data_dir images
```

## 需要注意的一些注意事项

1. 分布式设置`gloo`和`nccl`不适用于`mps`设备。 
这意味着目前只能使用`mps`设备类型的单个GPU。

最后，请记住，`Accelerate`仅集成了MPS后端，因此如果您
对 MPS 后端使用有任何问题或疑问，请通过 [PyTorch GitHub](https://github.com/pytorch/pytorch/issues) 提出问题。

### DDP 通信挂钩
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/ddp_comm_hook.md