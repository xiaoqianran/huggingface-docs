<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 图像数据集

本指南将向您展示如何使用图像文件配置数据集存储库。您可以在此 [Image datasets examples collection](https://huggingface.co/collections/datasets-examples/image-dataset-6568e7cf28639db76eb92d65) 中找到随附的存储库示例。

具有受支持结构和 [file formats](./datasets-adding#file-formats) 的数据集会自动在 Hub 的页面上显示数据集查看器。

只要您将有关图像的其他信息（例如用于对象检测的标题或边界框）包含在元数据文件 (`metadata.csv`/`metadata.jsonl`/`metadata.parquet`) 中，就会自动加载。

或者，图像可以位于 Parquet 文件中或遵循 [WebDataset](https://github.com/webdataset/webdataset) 格式的 TAR 存档中。

## 仅图像

如果您的数据集仅包含一列图像，您可以简单地将图像文件存储在根目录中：

```
my_dataset_repository/
├── 1.jpg
├── 2.jpg
├── 3.jpg
└── 4.jpg
```

或在子目录中：

```
my_dataset_repository/
└── images
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    └── 4.jpg
```

同时支持多个[formats](./datasets-adding#file-formats)，包括PNG、JPEG、TIFF和WebP。

```
my_dataset_repository/
└── images
    ├── 1.jpg
    ├── 2.png
    ├── 3.tiff
    └── 4.webp
```

如果您有多个分割，您可以将图像放入相应命名的目录中： 

```
my_dataset_repository/
├── train
│   ├── 1.jpg
│   └── 2.jpg
└── test
    ├── 3.jpg
    └── 4.jpg
```

请参阅 [File names and splits](./datasets-file-names-and-splits) 了解更多信息以及通过拆分组织数据的其他方法。

## 附加列如果您想要包含有关数据集的其他信息，例如文本标题或边界框，请将其作为 `metadata.csv` 文件添加到存储库中。这使您可以快速为不同的计算机视觉任务（例如[text captioning](https://huggingface.co/tasks/image-to-text)或[object detection](https://huggingface.co/tasks/object-detection)）创建数据集。

```
my_dataset_repository/
└── train
    ├── 1.jpg
    ├── 2.jpg
    ├── 3.jpg
    ├── 4.jpg
    └── metadata.csv
```

您的 `metadata.csv` 文件必须有一个 `file_name` 列，用于链接图像文件及其元数据：

```csv
file_name,text
1.jpg,a drawing of a green pokemon with red eyes
2.jpg,a green and yellow toy with a red nose
3.jpg,a red and white ball with an angry look on its face
4.jpg,a cartoon ball with a smile on its face
```

您还可以使用 [JSONL](https://jsonlines.org/) 文件 `metadata.jsonl`：

```jsonl
{"file_name": "1.jpg","text": "a drawing of a green pokemon with red eyes"}
{"file_name": "2.jpg","text": "a green and yellow toy with a red nose"}
{"file_name": "3.jpg","text": "a red and white ball with an angry look on its face"}
{"file_name": "4.jpg","text": "a cartoon ball with a smile on its face"}
```

对于更大的数据集或者如果您对高级数据检索功能感兴趣，您可以使用 [Parquet](https://parquet.apache.org/) 文件 `metadata.parquet`。

## 相对路径

元数据文件必须位于与其链接到的图像相同的目录中，或者位于任何父目录中，如下例所示： 

```
my_dataset_repository/
└── train
    ├── images
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   ├── 3.jpg
    │   └── 4.jpg
    └── metadata.csv
```

在这种情况下，`file_name`列必须是图像的完整相对路径，而不仅仅是文件名：

```csv
file_name,text
images/1.jpg,a drawing of a green pokemon with red eyes
images/2.jpg,a green and yellow toy with a red nose
images/3.jpg,a red and white ball with an angry look on its face
images/4.jpg,a cartoon ball with a smile on it's face
```

元数据文件不能放在图像所在目录的子目录中。

更一般地说，任何名为 `file_name` 或 `*_file_name` 的列都应包含图像的完整相对路径。

## 图像分类

对于图像分类数据集，您还可以使用简单的设置：使用目录来命名图像类。将图像文件存储在如下目录结构中：

```
my_dataset_repository/
├── green
│   ├── 1.jpg
│   └── 2.jpg
└── red
    ├── 3.jpg
    └── 4.jpg
```使用此结构创建的数据集包含两列：`image`和`label`（值为`green`和`red`）。

您还可以提供多个拆分。为此，您的数据集目录应具有以下结构（有关更多信息，请参阅[File names and splits](./datasets-file-names-and-splits)）：

```
my_dataset_repository/
├── test
│   ├── green
│   │   └── 2.jpg
│   └── red
│       └── 4.jpg
└── train
    ├── green
    │   └── 1.jpg
    └── red
        └── 3.jpg
```

您可以在 [YAML configuration](./datasets-manual-configuration) 中禁用自动添加 `label` 列。如果您的目录名称没有特殊含义，请在 README 标头中设置 `drop_labels: true`：

```yaml
configs:
  - config_name: default  # Name of the dataset subset, if applicable.
    drop_labels: true
```

## 大规模数据集

### Web数据集格式

[WebDataset](./datasets-webdataset) 格式非常适合大规模图像数据集（例如，参见 [timm/imagenet-12k-wds](https://huggingface.co/datasets/timm/imagenet-12k-wds)）。
它由包含图像及其元数据的 TAR 存档组成，并针对流式传输进行了优化。如果您有大量图像并需要流数据加载器进行大规模训练，那么它会很有用。

```
my_dataset_repository/
├── train-0000.tar
├── train-0001.tar
├── ...
└── train-1023.tar
```

要制作 WebDataset TAR 存档，请创建一个包含要存档的图像和元数据文件的目录，并使用例如创建 TAR 存档`tar` 命令。
每个存档的大小通常约为 1GB。
确保每个图像和元数据对共享相同的文件前缀，例如：

```
train-0000/
├── 000.jpg
├── 000.json
├── 001.jpg
├── 001.json
├── ...
├── 999.jpg
└── 999.json
```请注意，为了方便用户并启用 [Dataset Viewer](./data-studio)，Hub 中托管的每个数据集都会自动转换为最大 5GB 的 Parquet 格式。
请参阅 [Parquet format](./data-studio#access-the-parquet-files) 文档了解更多相关信息。

### 镶木地板格式

您可以将所有内容嵌入到 [Parquet](https://parquet.apache.org/) 文件中，而不是将图像和元数据作为单独的文件上传。
如果您有大量图像、想要嵌入多个图像列或者想要在同一文件中存储有关图像的附加信息，这会很有用。
Parquet 对于存储原始字节等数据也很有用，而 JSON/CSV 不支持这种数据。

```
my_dataset_repository/
└── train.parquet
```

可以使用 `pandas` 或 `datasets` 库创建包含图像数据的 Parquet 文件。要使用`pandas`中的图像数据创建Parquet文件，您可以使用[pandas-image-methods](https://github.com/lhoestq/pandas-image-methods)和`df.to_parquet()`。在`datasets`中，您可以将列类型设置为`Image()`并使用`ds.to_parquet(...)`方法或`ds.push_to_hub(...)`。您可以在 `datasets` [here](/docs/datasets/image_load) 中找到有关加载图像数据集的指南。或者，您可以手动设置使用其他工具创建的 Parquet 的图像类型。首先，确保图像列的类型为 _struct_，其中包含用于图像数据的二进制字段 `"bytes"` 和用于图像文件名或路径的字符串字段 `"path"`。然后，您应该直接在 README 标题中的 YAML 中指定列的功能类型，例如：

```yaml
dataset_info:
  features:
  - name: image
    dtype: image
  - name: caption
    dtype: string
```

请注意，建议对小图像（每个图像 <1MB）和小行组（每个行组 100 行，`datasets` 用于图像）使用 Parquet。对于较大的图像，建议使用 WebDataset 格式，或共享原始图像文件（可以选择元数据文件，并遵循 [repositories recommendations and limits](https://huggingface.co/docs/hub/en/storage-limits) 的存储和文件数量）。

### 收藏
https://huggingface.co/docs/hub/collections.md