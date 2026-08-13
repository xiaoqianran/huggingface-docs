<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据集卡

## 什么是数据集卡？

每个数据集都可以由存储库中的 `README.md` 文件记录。该文件称为**数据集卡**，Hugging Face Hub 将在数据集的主页上呈现其内容。为了告知用户如何负责任地使用数据，最好包含有关数据集中任何潜在偏差的信息。一般来说，数据集卡可以帮助用户理解数据集的内容，并提供如何使用数据集的上下文。

您还可以将数据集元数据添加到卡中。元数据描述有关数据集的重要信息，例如其许可证、语言和大小。它还包含帮助用户发现 Hub 上的数据集的标签和 [data files configuration](./datasets-manual-configuration) 选项。标签在 `README.md` 文件顶部的 YAML 元数据部分中定义。

## 数据集卡元数据

数据集存储库会将其 README.md 呈现为数据集卡。要控制 Hub 如何显示卡片，您应该在 README 文件中创建一个 YAML 部分来定义一些元数据。首先在顶部添加三个 ---，然后包含所有相关元数据，并用另一组 --- 关闭该部分，如下例所示：

```yaml
language: 
- "List of ISO 639-1 code for your language"
- lang1
- lang2
pretty_name: "Pretty Name of the Dataset"
tags:
- tag1
- tag2
license: "any valid license identifier"
task_categories:
- task1
- task2
```您添加到数据集卡的元数据可在 Hub 上启用某些交互。例如：

* 允许用户在 https://huggingface.co/datasets 上过滤和发现数据集。
* 如果您使用[this table](./repositories-licenses)右栏中列出的关键字选择许可证，则该许可证将显示在数据集页面上。

在 Hub 上的数据集存储库中创建 README.md 文件时，使用元数据 UI 填充主要元数据：

    
    

要查看元数据字段，请参阅详细的[Dataset Card specifications](https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1)。

### 数据集卡创建指南

有关创建数据集卡的分步指南，请查看 [Create a dataset card](https://huggingface.co/docs/datasets/dataset_card) 指南。

阅读现有的数据集卡（例如 [ELI5 dataset card](https://huggingface.co/datasets/eli5/blob/main/README.md)）是熟悉常见约定的好方法。

### 链接论文

如果数据集卡包含指向 Paper 页面的链接（HF 或 Arxiv 摘要/PDF），Hub 将提取 arXiv ID 并将其包含在格式为 `arxiv:<PAPER ID>` 的数据集标签中。单击该标签将使您：

* 访问论文页面
* 过滤 Hub 上引用同一论文的其他模型。

了解有关纸质页面的更多信息[here](./paper-pages)。

### 强制设置数据集模式Hub 将根据数据集包含的文件（音频、视频、地理空间等）自动检测数据集的模式。如果要强制使用特定模态，可以向数据集卡元数据添加标签：`3d`、`audio`、`geospatial`、`image`、`tabular`、`text`、`timeseries`、`video`。

例如，要强制模态为`audio`，请将以下内容添加到数据集卡元数据中：

```yaml
tags:
- audio
```

### 将库关联到数据集

数据集页面会自动显示能够本地加载数据集的库和工具，但如果您想显示其他特定库，您可以在数据集卡元数据中添加标签：`argilla`、`dask`、`datasets`、`distilabel`、`fiftyone`、`mlcroissant`、`pandas`、 `webdataset`。请参阅 [list of supported libraries](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/dataset-libraries.ts) 了解更多信息，或建议添加新库。

例如，要将 `argilla` 库关联到数据集卡，请将以下内容添加到数据集卡元数据中：

```yaml
tags:
- argilla
```

### 空间作为 API 端点
https://huggingface.co/docs/hub/spaces-api-endpoints.md