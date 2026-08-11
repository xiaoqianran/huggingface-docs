<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 将本地 SGD 与 Accelerate 结合使用

局部 SGD 是一种分布式训练技术，其中梯度并非每一步都同步。因此，每个进程都会更新自己的模型权重版本，并且在给定数量的步骤之后，通过对所有进程进行平均来同步这些权重。这提高了通信效率，并且可以显着提高训练速度，特别是当计算机缺乏 NVLink 等更快的互连时。
与梯度累积（提高通信效率需要增加有效批量大小）不同，本地 SGD 不需要更改批量大小或学习率/时间表。然而，如果有必要，局部 SGD 也可以与梯度累积相结合。

在本教程中，您将了解如何快速设置 Local SGD Accelerate。与标准 Accelerate 设置相比，这仅需要两行额外的代码。

此示例将使用一个非常简单的 PyTorch 训练循环，每两批执行一次梯度累积：

```python
device = "cuda"
model.to(device)

gradient_accumulation_steps = 2

for index, batch in enumerate(training_dataloader):
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    outputs = model(inputs)
    loss = loss_function(outputs, targets)
    loss = loss / gradient_accumulation_steps
    loss.backward()
    if (index + 1) % gradient_accumulation_steps == 0:
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

## 将其转换为加速

首先，前面显示的代码将转换为使用 Accelerate，既不使用 LocalSGD，也不使用梯度累积助手：

```diff
+ from accelerate import Accelerator
+ accelerator = Accelerator()

+ model, optimizer, training_dataloader, scheduler = accelerator.prepare(
+     model, optimizer, training_dataloader, scheduler
+ )

  for index, batch in enumerate(training_dataloader):
      inputs, targets = batch
-     inputs = inputs.to(device)
-     targets = targets.to(device)
      outputs = model(inputs)
      loss = loss_function(outputs, targets)
      loss = loss / gradient_accumulation_steps
+     accelerator.backward(loss)
      if (index+1) % gradient_accumulation_steps == 0:
          optimizer.step()
          scheduler.step()
```## 让 Accelerate 处理模型同步 

现在剩下的就是让 Accelerate 为我们处理模型参数同步**和**梯度累积。为简单起见，我们假设需要每 8 个步骤同步一次。这是
通过在每个优化器步骤之后添加一个 `with LocalSGD` 语句和一个调用 `local_sgd.step()` 来实现：

```diff
+local_sgd_steps=8

+with LocalSGD(accelerator=accelerator, model=model, local_sgd_steps=8, enabled=True) as local_sgd:
    for batch in training_dataloader:
        with accelerator.accumulate(model):
            inputs, targets = batch
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
+           local_sgd.step()
```

在幕后，本地 SGD 代码**禁用**自动梯度同步（但累积仍然按预期工作！）。相反，它每隔 `local_sgd_steps` 步骤（以及在训练循环结束时）对模型参数进行平均。

## 限制

当前的实现仅适用于基本的多 GPU（或多 CPU）训练，没有 [DeepSpeed.](https://github.com/deepspeedai/DeepSpeed) 等。

## 参考文献

    尽管我们不知道这种简单方法的真正起源，但本地 SGD 的想法相当古老并且已经存在
    至少回到：

    张，J.，德萨，C.，米利亚格卡斯，I.，＆Ré，C.（2016）。 [Parallel SGD: When does averaging help?. arXiv preprint
    arXiv:1606.07365.](https://huggingface.co/papers/1606.07365)

    我们将本地 SGD 一词归功于以下论文（但可能有我们不知道的早期参考文献）。

    斯蒂奇，塞巴斯蒂安·厄本。 ["Local SGD Converges Fast and Communicates Little." ICLR 2019-International Conference on
    Learning Representations. No. CONF. 2019.](https://huggingface.co/papers/1805.09767)

### 编译
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/compilation.md