<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 从这里开始！

请使用下面的交互式工具来帮助您开始了解特定的知识 
Accelerate 的功能以及如何使用它！它将为您提供代码差异、解释
了解正在发生的事情，并为您提供一些有用的链接来探索更多内容
文档！

大多数代码示例从以下 python 代码开始，然后以某种方式集成 Accelerate：

```python
for batch in dataloader:
    optimizer.zero_grad()
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

	<iframe 
        src="https://hf-accelerate-accelerate-examples.hf.space?__theme=light"
        width="850"
        height="1600"
    >

    <iframe 
        src="https://hf-accelerate-accelerate-examples.hf.space?__theme=dark"
        width="850"
        height="1600"
    >

### 模型内存估计器
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/model_size_estimator.md