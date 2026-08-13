<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 上传数据集

[Hub](https://huggingface.co/datasets) 拥有大量社区策划和研究数据集。我们鼓励您将数据集分享到中心，以帮助发展 ML 社区并加快每个人的进步。欢迎所有贡献；添加数据集只需拖放即可！

如果您还没有，请从 [creating a Hugging Face Hub account](https://huggingface.co/join) 开始。

## 使用 Hub UI 上传

该中心基于网络的界面允许没有任何开发经验的用户上传数据集。

### 创建存储库

存储库托管所有数据集文件，包括修订历史记录，从而可以存储多个数据集版本。

1. 单击您的个人资料并选择 **新数据集** 以创建 [new dataset repository](https://huggingface.co/new-dataset)。
2. 为您的数据集选择一个名称，然后选择它是公共数据集还是私有数据集。公共数据集对任何人都可见，而私有数据集只能由您或您组织的成员查看。

    

### 上传数据集1. 创建存储库后，导航到 **文件和版本** 选项卡以添加文件。选择 **添加文件** 上传您的数据集文件。我们支持许多文本、音频、图像和其他数据扩展，例如`.csv`、`.mp3`和`.jpg`（请参阅[File formats](#file-formats)的完整列表）。

    

2. 拖放数据集文件。

    

3. 上传数据集文件后，它们将存储在您的数据集存储库中。

    

### 创建数据集卡

添加数据集卡对于帮助用户找到您的数据集并了解如何负责任地使用它非常有价值。

1. 点击**创建数据集卡**，创建[Dataset card](./datasets-cards)。此按钮会在您的存储库中创建一个 `README.md` 文件。

    

2. 在顶部，您将看到 **元数据 UI**，其中有多个字段可供选择，例如许可证、语言和任务类别。这些是帮助用户在 Hub 上发现您的数据集（如果适用）的最重要标签。当您为字段选择一个选项时，它将自动添加到数据集卡的顶部。

    您还可以查看 [Dataset Card specifications](https://github.com/huggingface/hub-docs/blob/main/datasetcard.md?plain=1)，它具有一整套允许的标签，包括像 `annotations_creators` 这样的可选标签，以帮助您选择对您的数据集有用的标签。3. 在数据集卡中编写您的数据集文档，向社区介绍您的数据集并帮助用户了解其中的内容：用例和限制是什么、数据来自哪里、重要的道德考虑因素是什么以及任何其他相关细节。

    您可以单击编辑器顶部的**导入数据集卡模板**链接来自动创建数据集卡模板。有关良好数据集卡应是什么样子的详细示例，请查看 [CNN DailyMail Dataset card](https://huggingface.co/datasets/cnn_dailymail)。

## 使用`huggingface_hub`客户端库

`huggingface_hub`库中设置的丰富功能允许您管理存储库，包括创建存储库和将数据集上传到Hub。请访问[the client library's documentation](/docs/huggingface_hub/index)了解更多信息。

## 使用其他库

一些库（如 [🤗 Datasets](/docs/datasets/index)、[Pandas](https://pandas.pydata.org/)、[Polars](https://pola.rs)、[Dask](https://www.dask.org/)、[DuckDB](https://duckdb.org/) 或 [Daft](https://daft.ai/)）可以将文件上传到 Hub。
更多信息请参见[Libraries supported by the Datasets Hub](./datasets-libraries)列表。

## 使用 Git

由于数据集存储库是 Git 存储库，因此您可以使用 Git 将数据文件推送到 Hub。按照 [Getting Started with Repositories](repositories-getting-started) 上的指南了解如何使用 `git` CLI 提交和推送数据集。

## 摄取数据集如果您的数据库、云存储或 API 中有数据，您可以将它们作为即用型数据集提取到 Hugging Face 中。

在 [documentation on ingesting datasets](./datasets-ingesting) 中查找更多信息。

## 文件格式

Hub本身支持多种文件格式：

- 镶木地板 (.parquet)
- CSV（.csv、.tsv）
- JSON 行、JSON（.jsonl、.json）
- Arrow 流媒体和 IPC 格式 (.arrow)
- 文本（.txt）
- 图片（.png、.jpg 等）
- 音频（.wav、.mp3 等）
- 视频（.mp4、.mov、.avi 等）
- PDF (.pdf)
- [WebDataset](./datasets-webdataset) (.tar)
- [Lance](./datasets-lance) (.lance)

它支持使用 ZIP (.zip)、GZIP (.gz)、ZSTD (.zst)、BZ2 (.bz2)、LZ4 (.lz4) 和 LZMA (.xz) 压缩的文件。

图像和音频文件还可以有附加的元数据文件。请参阅有关图像和音频数据集的[Data files Configuration](./datasets-data-files-configuration#image-and-audio-datasets)，以及有关 CSV、TSV 和图像的[example datasets](https://huggingface.co/datasets-examples) 集合。

您可能希望将文件转换为这些格式，以受益于 Hub 的所有功能。
Hub 可能无法识别其他格式和结构。

### 我应该使用哪种文件格式？对于大多数类型的数据集，**Parquet** 是推荐的格式，因为它具有高效的压缩、丰富的类型，并且多种工具通过优化的读取和批处理操作支持这种格式。或者，CSV 或 JSON Lines/JSON 可用于表格数据（对于嵌套数据，首选 JSON Lines）。虽然与 Parquet 相比易于解析，但不建议用于大于几 GB 的数据。对于图像和音频数据集，上传原始文件对于大多数用例来说是最实用的，因为可以轻松访问单个文件。对于大规模图像和音频数据集流，应优先选择[WebDataset](https://github.com/webdataset/webdataset)而不是原始图像和音频文件，以避免访问单个文件的开销。尽管对于涉及分析、数据过滤或元数据解析的更一般用例，Parquet 是大规模图像和音频数据集的推荐选项。

### 数据工作室

[Data Studio](./data-studio) 对于在下载数据之前了解数据的实际外观非常有用。
默认情况下，它对所有公共数据集启用。它还可用于 [PRO user](https://huggingface.co/pricing) 或 [Team or Enterprise organization](https://huggingface.co/enterprise) 拥有的私有数据集。上传数据集后，请确保数据集查看器正确显示您的数据，或[Configure the Dataset Viewer](./datasets-viewer-configure)。

## 大规模数据集

Hugging Face Hub 支持大规模数据集，通常以 Parquet 格式上传（例如通过 `push_to_hub()` 使用 [🤗 Datasets](/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.push_to_hub)）或 [WebDataset](https://github.com/webdataset/webdataset) 格式。

您可以使用`huggingface_hub`库高速上传大规模数据集。

请参阅[how to upload a folder by chunks](/docs/huggingface_hub/guides/upload#upload-a-folder-by-chunks)、[tips and tricks for large uploads](/docs/huggingface_hub/guides/upload#tips-and-tricks-for-large-uploads) 和[repository storage limits and recommendations](./storage-limits)。

### 使用 SDK 构建
https://huggingface.co/docs/hub/agents-sdk.md