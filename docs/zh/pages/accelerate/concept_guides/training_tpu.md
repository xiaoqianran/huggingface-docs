<!-- huggingface-docs: machine-translated zh-CN from English source -->

# TPU 训练

即使使用 Accelerate，TPU 上的训练也可能与多 GPU 上的训练略有不同。本指南旨在向您展示 
哪里应该小心、为什么要小心，以及一般的最佳实践。

## 在笔记本中进行训练

TPU 训练时的主要关注点来自[notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher)。正如[notebook tutorial](../usage_guides/notebook)中提到的，您需要 
将训练代码重组为可以传递给 [notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher) 函数的函数，并注意不要在 GPU 上声明任何张量。

虽然在 TPU 上，最后一部分并不那么重要，但需要理解的一个关键部分是，当您从笔记本启动代码时，您是通过一个称为 **forking** 的过程来实现的。 
从命令行启动时，您执行**生成**，其中 python 进程当前未运行，并且您*生成*一个新进程。因为您的 Jupyter 笔记本已经 
使用 python 进程，您需要从中 *fork* 一个新进程来启动您的代码。这一点在声明模型时变得很重要。在分叉的 TPU 进程中，建议您实例化您的模型 *一次* 并将其传递到您的 
训练功能。这与在 GPU 上进行训练不同，在 GPU 上创建 `n` 模型，使其梯度在某些时刻同步并反向传播。相反，一个 
模型实例在所有节点之间共享，并且来回传递。这一点尤其重要，尤其是在低资源 TPU（例如 Kaggle 内核中提供的 TPU）上进行训练时
在谷歌合作实验室上。 

下面是在 CPU 或 GPU 上训练时传递给 [notebook_launcher()](/docs/accelerate/v1.14.0/en/package_reference/launchers#accelerate.notebook_launcher) 的训练函数示例：

    此代码片段基于 `simple_nlp_example` 笔记本中的代码片段，发现 [here](https://github.com/huggingface/notebooks/blob/main/examples/accelerate_examples/simple_nlp_example.ipynb) 有轻微的变化 
    为了简单起见进行修改

```python
def training_function():
    # Initialize accelerator
    accelerator = Accelerator()
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
    train_dataloader, eval_dataloader = create_dataloaders(
        train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
    )

    # Instantiate optimizer
    optimizer = AdamW(params=model.parameters(), lr=hyperparameters["learning_rate"])

    # Prepare everything
    # There is no specific order to remember, we just need to unpack the objects in the same order we gave them to the
    # prepare method.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    num_epochs = hyperparameters["num_epochs"]
    # Now we train the model
    for epoch in range(num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            optimizer.zero_grad()
```

```python
from accelerate import notebook_launcher

notebook_launcher(training_function)
```

    如果为 TPU 配置了加速，则`notebook_launcher` 将默认为 8 个进程

如果您使用此示例并在训练循环*内部*声明模型，那么在资源匮乏的系统上您可能会看到错误 
像：

```
ProcessExitedException: process 0 terminated with signal SIGSEGV
```这个错误*极其*神秘，但基本的解释是你的系统内存用完了。您可以通过将训练函数重新配置为来完全避免这种情况 
接受单个 `model` 参数，并在外部单元格中声明它：

```python
# In another Jupyter cell
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
```

```diff
+ def training_function(model):
      # Initialize accelerator
      accelerator = Accelerator()
-     model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
      train_dataloader, eval_dataloader = create_dataloaders(
          train_batch_size=hyperparameters["train_batch_size"], eval_batch_size=hyperparameters["eval_batch_size"]
      )
  ...
```

最后调用训练函数：

```diff
  from accelerate import notebook_launcher
- notebook_launcher(training_function)
+ notebook_launcher(training_function, (model,))
```

    仅当在 Google Colaboratory 或 Kaggle 等低资源服务器上从 Jupyter Notebook 启动 TPU 实例时，才需要使用上述解决方法。如果 
    不需要使用脚本或在更强大的服务器上启动来预先声明模型。

## 混合精度和全局变量 

正如[mixed precision tutorial](../usage_guides/mixed_precision)中提到的，Accelerate支持fp16和bf16，两者都可以在TPU上使用。
话虽如此，理想情况下应该使用`bf16`，因为它的使用效率非常高。

在 TPU 上使用 `bf16` 和 Accelerate 时，有两个“层”：基础层和操作层。 

在基础级别，将 `mixed_precision="bf16"` 传递到 `Accelerator` 时启用此功能，例如：
```python
accelerator = Accelerator(mixed_precision="bf16")
```
默认情况下，这会将 TPU 上的 `torch.float` 和 `torch.double` 转换为 `bfloat16`。 
设置的具体配置是将环境变量`XLA_USE_BF16`设置为`1`。您可以执行进一步的配置，即设置 `XLA_DOWNCAST_BF16` 环境变量。如果设置为`1`，则 
`torch.float` 是`bfloat16`，`torch.double` 是`float32`。

这是在传递 `downcast_bf16=True` 时在 `Accelerator` 对象中执行的：
```python
accelerator = Accelerator(mixed_precision="bf16", downcast_bf16=True)
```

当您尝试计算指标、日志值等原始 bf16 张量无法使用的情况时，在任何地方使用向下转型而不是 bf16 是有好处的。 

## TPU 上的训练时间

当您启动脚本时，您可能会注意到训练一开始似乎异常缓慢。这是因为 TPU
首先运行几批数据以查看要分配多少内存，然后再最终使用此配置 
内存分配非常有效。 

如果您注意到由于使用的批量大小较大，计算模型指标的评估代码需要更长的时间， 
如果速度太慢，建议保持批量大小与训练数据相同。否则内存会重新分配给这个 
前几次迭代后的新批量大小。 

    仅仅因为分配了内存并不意味着它将被使用，也不意味着返回训练数据加载器时批处理大小会增加。### 将大模型加载到内存中
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/big_model_inference.md