<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据文件配置

对于如何构建数据集存储库没有任何限制。

但是，如果您希望数据集查看器显示某些数据文件，或者在训练/验证/测试拆分中分离数据集，则需要相应地构建数据集。
通常，它就像根据数据文件的拆分名称命名一样简单，例如`train.csv` 和 `test.csv`。

## 什么是分割和子集？

机器学习数据集通常有分割，也可能有子集。数据集通常由在训练和评估模型的不同阶段使用的_split_（例如`train`和`test`）组成。 _子集_（也称为_配置_）是包含在较大数据集中的子数据集。子集在多语言语音数据集中尤其常见，其中每种语言可能有不同的子集。如果您有兴趣了解有关分割和子集的更多信息，请查看 [Splits and subsets](/docs/datasets-server/configs_and_splits) 指南！

![split-configs-server](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/split-configs-server.gif)

## 自动分割检测

根据文件和目录名称自动检测拆分。例如，这是一个具有 `train`、`test` 和 `validation` 分割的数据集：

```
my_dataset_repository/
├── README.md
├── train.csv
├── test.csv
└── validation.csv
```要通过根据拆分名称命名数据文件或目录来构建数据集，请参阅 [File names and splits](./datasets-file-names-and-splits) 文档和 [companion collection of example datasets](https://huggingface.co/collections/datasets-examples/file-names-and-splits-655e28af4471bd95709eb135)。

## 手动分割和子集配置

您可以使用 YAML 选择要在数据集查看器中显示的数据文件。
如果您想手动指定哪个文件进入哪个分割，它会很有用。

您还可以为数据集定义多个子集，并传递数据集构建参数（例如用于 CSV 文件的分隔符）。

下面是一个配置示例，定义了一个名为“基准”的子集，其中包含 `test` 分割。

```yaml
configs:
- config_name: benchmark
  data_files:
  - split: test
    path: benchmark.csv
```

有关更多信息，请参阅 [Manual configuration](./datasets-manual-configuration) 的文档。另请注意[example datasets](https://huggingface.co/collections/datasets-examples/manual-configuration-655e293cea26da0acab95b87)。

## 支持的文件格式

请参阅 [File formats](./datasets-adding#file-formats) 文档页面，查找支持的格式列表以及数据集的建议。如果您的数据集使用 CSV 或 TSV 文件，您可以在 [example datasets](https://huggingface.co/collections/datasets-examples/format-csv-and-tsv-655f681cb9673a4249cccb3d) 中找到更多信息。

### 数据集查看器大小限制错误 (`TooBigContentError`)

如果您看到`Error code: TooBigContentError`，则数据集查看器无法在其限制内读取预览。常见消息包括`Parquet error: Scan size limit exceeded`和`The size of the content of the first rows exceeds the maximum supported size`。

你可以做什么：- 对于 Parquet 文件，使用较小的行组并包含页面索引 (`write_page_index=True`)，以便查看器只能读取所需的内容。
- 避免第一行中出现非常大的值（非常长的字符串、大的 JSON blob、base64 有效负载）。尽可能将大型有效负载移动到单独的文件中。
- 将非常大的文件分割成较小的碎片或分割，然后重新上传。
- 如果问题仍然存在，请查看 [Configure the Dataset Viewer](./datasets-viewer-configure) 并在数据集页面上打开讨论并提供完整的错误文本。

## 图像、音频和视频数据集

对于图像/音频/视频分类数据集，您还可以使用目录来命名图像/音频/视频类。
如果您的图像/音频/视频文件具有元数据（例如标题、边界框、转录等），则您可以在它们旁边放置元数据文件。

我们提供了两个指南供您查看：

- [How to create an image dataset](./datasets-image) ([example datasets](https://huggingface.co/collections/datasets-examples/image-dataset-6568e7cf28639db76eb92d65))
- [How to create an audio dataset](./datasets-audio) ([example datasets](https://huggingface.co/collections/datasets-examples/audio-dataset-66aca0b73e8f69e3d069e607))
- [How to create a video dataset](./datasets-video)

### 🟧 空间标签工作室
https://huggingface.co/docs/hub/spaces-sdks-docker-label-studio.md