<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 参数高效的微调方法

PEFT 方法训练尽可能少的参数，同时追求与完全微调相当的性能。可训练参数越少，表达能力越差，因此无法保证相同的性能。作为交换，您使用更少的内存，通常更少的计算，并获得诸如专家适配器之间的快速热交换和更少忘记先验知识之类的功能。

为训练大型模型提供一般建议很困难，但对于生成模型，尤其是语言模型，您可以按照以下步骤操作：

1. 使用提示（例如提示中的少数示例）来查看模型是否已经能够胜任该任务。如果该模型解决了您的问题，那就太好了！您现在可以使用[Prompt-based methods](#prompt-based-methods)来学习提示并节省宝贵的代币。
2. 如果基于提示的方法不够，您可以使用[layer tuning](#layer-tuning)和[adapter methods](#adapter-methods)。这些方法通常比基于提示的方法更具表现力，并且更接近完全微调。
3. 确保衡量已学知识的保留情况，因为每个微调步骤都可能会忘记过去的知识。

[PEFT method comparison suite](https://huggingface.co/spaces/peft-internal-testing/PEFT-method-comparison) 旨在粗略概述（大多数）在选定的基准和模型上实现的方法。> [!注意]
> 并非所有 PEFT 方法都是一样的，并且功能之间存在差异：
> * 量化：并非所有方法都支持量化基础模型
> * 功能：并非所有方法都支持所有功能（例如，多个适配器、混合适配器推理、合并/取消合并）
> * 层类型：一般支持线性层，但并非所有适配器方法都支持嵌入（对于扩展词汇量很重要）或卷积层（对于某些图像模型很重要）
> * 运行时：PEFT 方法通常会增加运行时开销，但其中一些开销可以减轻（例如，通过 [merging the adapter weights](../developer_guides/checkpoint#merge-the-weights)）

## 基于提示的方法

提示通过包含描述任务甚至演示任务示例的文本提示，为特定下游任务启动冻结的预训练模型。通过提示，您可以避免为每个下游任务完全训练单独的模型，而是使用相同的冻结预训练模型。这要容易得多，因为您可以使用相同的模型来执行多个不同的任务，并且训练和存储较小的提示参数集比训练所有模型的参数要高效得多。提示方式有两类：

- 硬提示是带有离散输入标记的手动文本提示；缺点是需要付出很大的努力才能创建一个好的提示
- 软提示是可学习的张量，与可优化为数据集的输入嵌入相连接；缺点是它们不是人类可读的，因为您没有将这些“虚拟标记”与真实单词的嵌入相匹配

PEFT 库支持多种类型的提示方法（p 调优、前缀调优、提示调优……），浏览目录以获取软提示方法的完整列表。
如果您有兴趣将这些方法应用于其他任务和用例，请查看我们的[notebook collection](https://huggingface.co/spaces/PEFT/soft-prompting)！

> [!提示]
> 熟悉训练因果语言模型的一般过程将会非常有帮助，并且可以让您专注于软提示方法。如果您是新手，我们建议您首先查看 Transformers 文档中的 [Causal language modeling](https://huggingface.co/docs/transformers/tasks/language_modeling) 指南​​。当您准备好后，回来看看将 PEFT 纳入您的训练是多么容易！

## 图层调整层调整对专门针对一种类型的层或层的一个方面的方法进行分类，例如[LayerNorm Tuning](../package_reference/layernorm_tuning)仅针对[⟦T0⟧](https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html)层，[TrainableTokens](../package_reference/trainable_tokens)仅针对嵌入矩阵中的特定标记。这与基于提示的方法形成鲜明对比，基于提示的方法与模型输入或适配器方法一起使用，后者扩展了现有权重，并且通常更独立于层类型，针对线性或卷积层。

## 适配器方法

适配器方法可以被视为向现有模型添加相对较小的可训练矩阵以进行微调的方法。目标是引入少量可训练参数来引导大模型执行需要微调以节省资源（例如内存或计算）的任务。实现适配器的一种流行方法是插入较小的可训练矩阵，这些矩阵是适应权重布局的低秩分解，以节省内存。有几种不同的方法可以将权重矩阵表示为低秩分解，但[Low-Rank Adaptation (LoRA)](../package_reference/lora)是最常见的方法。 PEFT 库支持该公式的其他几种变体 - 有些是 LoRA 的直接变体并在 LoRA 下记录，有些差异足以算作自己的方法，例如 [Low-Rank Hadamard Product (LoHa)](../package_reference/loha)、[Low-Rank Kronecker Product (LoKr)](../package_reference/lokr) 和 [Adaptive Low-Rank Adaptation (AdaLoRA)](../package_reference/adalora)。如果您有兴趣将这些方法应用于其他任务和用例，例如语义分割、标记分类，请看看我们的[notebook collection](https://huggingface.co/collections/PEFT/notebooks-6573b28b33e5a4bf5b157fc1)！

> [!提示]
> LoRA 是最流行的 PEFT 方法之一，如果您刚刚开始使用 PEFT，这是一个很好的起点。它最初是为大型语言模型开发的，但由于其效率和有效性，它是扩散模型的一种非常流行的训练方法。低阶适配器只是一种可能的适配器配方，PEFT 还实现了许多其他类型的适配器。例如，正交微调方法（[OFT](../package_reference/oft)，[BOFT](../package_reference/boft)，...）使用适配器权重的正交分解来实现小尺寸。像[MiSS](../package_reference/miss)这样的方法对矩阵进行分片并共享这些分片以节省内存。 [IA3](../package_reference/ia3) 引入了重新调整键、值和前馈激活的学习向量。

### PEFT 集成
https://huggingface.co/docs/peft/v0.20.0/guides/peft_integrations.md