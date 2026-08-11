<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型内存估计器

在探索在您的机器上使用的潜在模型时，一个非常困难的方面是了解模型的大小将“适合”当前设备的内存（例如将模型加载到 CUDA 或 XPU 上）。

为了帮助缓解这一问题，Accelerate 通过 `accelerate estimate-memory` 提供了 CLI 界面。本教程将 
帮助引导您完成使用它、期待什么，最后链接到 Hub 上托管的交互式演示，该演示将 
甚至让您将这些结果直接发布在模型存储库上！

目前我们支持搜索`timm`和`transformers`可以使用的型号。

    该API会将模型加载到`meta`设备上的内存中，因此我们实际上并没有下载 
    并将模型的全部权重加载到内存中，我们也不需要这样做。结果是 
    完美测量 80 亿个参数模型（或更多），无需担心 
    如果你的CPU可以处理的话！

## 录音室演示

下面是一些与上述内容相关的渐变演示。第一个是官方的 Hugging Face 内存估计空间，直接利用 Accelerate：

	<iframe 
        src="https://hf-accelerate-model-memory-usage.hf.space?__theme=light"
        width="850"
        height="1600"
    >

    <iframe 
        src="https://hf-accelerate-model-memory-usage.hf.space?__theme=dark"
        width="850"
        height="1600"
    >社区成员采纳了这个想法并进一步扩展了它，允许您直接过滤模型并查看是否可以在给定 GPU 限制和 LoRA 配置的情况下运行特定的 LLM。要使用它，请参阅[here](https://huggingface.co/spaces/Vokturz/can-it-run-llm)了解更多详细信息。

## 命令

使用`accelerate estimate-memory`时，需要传入你想要使用的模型名称，可能是框架
使用的模型（如果无法自动找到）以及您希望模型加载的数据类型。

例如，以下是我们如何计算 `bert-base-cased` 的内存占用量：

```bash
accelerate estimate-memory bert-base-cased
```

这将下载`bert-based-cased`的`config.json`，将模型加载到`meta`设备上，并报告有多少空间
它将使用：

加载`bert-base-cased`的内存使用情况：

|数据类型 |最大层|总尺寸|使用 Adam 进行训练 |
|--------|-------------|------------|----------|
|浮动32 | 84.95 MB | 418.18 MB | 1.61 GB | 1.61 GB
|浮动16 | 42.47 MB​​ | 206.59 MB | 826.36 MB |
| int8 | 21.24 MB | 103.29 MB | 413.18 MB |
| int4 | 10.62 MB | 51.65 MB | 206.59 MB |默认情况下，它将返回所有支持的数据类型（`int4`到`float32`），但如果您对特定的数据类型感兴趣，可以过滤这些数据类型。

### 特定库

如果无法自动确定源库（就像`bert-base-cased`的情况一样），可以使用库名称
被传入。 

```bash
accelerate estimate-memory HuggingFaceM4/idefics-80b-instruct --library_name transformers
```

加载`HuggingFaceM4/idefics-80b-instruct`的内存使用情况：

|数据类型 |最大层|总尺寸|使用 Adam 进行训练 |
|--------|-------------|------------|----------|
|浮动32 | 3.02 GB | 3.02 GB 297.12 GB | 297.12 GB 1.16TB |
|浮动16 | 1.51 GB | 1.51 GB 148.56 GB | 148.56 GB 594.24GB|
| int8 | 772.52 MB | 74.28 GB | 297.12 GB | 297.12 GB
| int4 | 386.26 MB | 37.14 GB | 37.14 GB 148.56 GB | 148.56 GB

```bash
accelerate estimate-memory timm/resnet50.a1_in1k --library_name timm
```

加载`timm/resnet50.a1_in1k`的内存使用情况：

|数据类型 |最大层|总尺寸|使用 Adam 进行训练 |
|--------|-------------|------------|----------|
|浮动32 | 9.0 MB | 97.7 MB | 390.78 MB |
|浮动16 | 4.5 MB | 48.85 MB | 195.39 MB |
| int8 | 2.25 MB | 24.42 MB | 97.7 MB |
| int4 | 1.12 MB | 12.21 MB | 48.85 MB |

### 特定数据类型如前所述，虽然我们默认返回 `int4` 到 `float32`，但可以使用 `float32`、`float16`、`int8` 和 `int4` 中的任何 dtype。

为此，请在指定 `--dtypes` 后传入它们：

```bash
accelerate estimate-memory bert-base-cased --dtypes float32 float16
```

加载`bert-base-cased`的内存使用情况：

|数据类型 |最大层|总尺寸|使用 Adam 进行训练 |
|--------|-------------|------------|----------|
|浮动32 | 84.95 MB | 413.18 MB | 1.61 GB | 1.61 GB
|浮动16 | 42.47 MB​​ | 206.59 MB | 826.36 MB |

## 使用此计算器的注意事项

该计算器将告诉您纯粹加载模型需要多少内存，*而不是*执行推理。

此计算的精确度在实际值的百分之几内，因此可以很好地了解它将占用多少内存。例如，当以全精度加载到 CUDA 上时，加载 `bert-base-cased` 实际上需要 `413.68 MB`，并且计算器会估计 `413.18 MB`。

执行推理时，您预计会额外增加 20%，如[EleutherAI](https://blog.eleuther.ai/transformer-math/) 所示。我们将进行研究，寻找对这些值的更准确的估计，并将更新 
这个计算器一旦完成。

### 亚马逊 SageMaker
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/sagemaker.md