<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 视频数据集

本指南将向您展示如何使用视频文件配置数据集存储库。

具有受支持结构和 [file formats](./datasets-adding#file-formats) 的数据集会自动在 Hub 上的页面上显示数据集查看器。

只要您将有关视频的其他信息（例如用于对象检测的字幕或边界框）包含在元数据文件 (`metadata.csv`/`metadata.jsonl`/`metadata.parquet`) 中，系统就会自动加载这些信息。

或者，视频可以采用 Parquet 文件或遵循 [WebDataset](https://github.com/webdataset/webdataset) 格式的 TAR 存档。

## 仅视频

如果您的数据集仅包含一列视频，您可以简单地将视频文件存储在根目录中：

```
my_dataset_repository/
├── 1.mp4
├── 2.mp4
├── 3.mp4
└── 4.mp4
```

或在子目录中：

```
my_dataset_repository/
└── videos
    ├── 1.mp4
    ├── 2.mp4
    ├── 3.mp4
    └── 4.mp4
```

同时支持多个[formats](./datasets-adding#file-formats)，包括MP4、MOV和AVI。

```
my_dataset_repository/
└── videos
    ├── 1.mp4
    ├── 2.mov
    └── 3.avi
```

如果您有多个分割，您可以将视频放入相应命名的目录中： 

```
my_dataset_repository/
├── train
│   ├── 1.mp4
│   └── 2.mp4
└── test
    ├── 3.mp4
    └── 4.mp4
```

请参阅 [File names and splits](./datasets-file-names-and-splits) 了解更多信息以及通过拆分组织数据的其他方法。

## 附加列

如果您想要包含有关数据集的其他信息（例如文本标题或边界框），请将其作为 `metadata.csv` 文件添加到存储库中。这使您可以快速为不同的计算机视觉任务（例如[video generation](https://huggingface.co/tasks/text-to-video)或[object detection](https://huggingface.co/tasks/object-detection)）创建数据集。```
my_dataset_repository/
└── train
    ├── 1.mp4
    ├── 2.mp4
    ├── 3.mp4
    ├── 4.mp4
    └── metadata.csv
```

您的 `metadata.csv` 文件必须有一个 `file_name` 列，用于链接视频文件及其元数据：

```csv
file_name,text
1.mp4,an animation of a green pokemon with red eyes
2.mp4,a short video of a green and yellow toy with a red nose
3.mp4,a red and white ball shows an angry look on its face
4.mp4,a cartoon ball is smiling
```

您还可以使用 [JSONL](https://jsonlines.org/) 文件 `metadata.jsonl`：

```jsonl
{"file_name": "1.mp4","text": "an animation of a green pokemon with red eyes"}
{"file_name": "2.mp4","text": "a short video of a green and yellow toy with a red nose"}
{"file_name": "3.mp4","text": "a red and white ball shows an angry look on its face"}
{"file_name": "4.mp4","text": "a cartoon ball is smiling"}
```

对于更大的数据集或者如果您对高级数据检索功能感兴趣，您可以使用 [Parquet](https://parquet.apache.org/) 文件 `metadata.parquet`。

## 相对路径

元数据文件必须位于与其链接到的视频相同的目录中，或者位于任何父目录中，如下例所示： 

```
my_dataset_repository/
└── train
    ├── videos
    │   ├── 1.mp4
    │   ├── 2.mp4
    │   ├── 3.mp4
    │   └── 4.mp4
    └── metadata.csv
```

在这种情况下，`file_name`列必须是视频的完整相对路径，而不仅仅是文件名：

```csv
file_name,text
videos/1.mp4,an animation of a green pokemon with red eyes
videos/2.mp4,a short video of a green and yellow toy with a red nose
videos/3.mp4,a red and white ball shows an angry look on its face
videos/4.mp4,a cartoon ball is smiling
```

元数据文件不能放入视频目录的子目录中。

更一般地说，任何名为 `file_name` 或 `*_file_name` 的列都应包含视频的完整相对路径。

## 视频分类

对于视频分类数据集，您还可以使用简单的设置：使用目录来命名视频类。将视频文件存储在如下目录结构中：

```
my_dataset_repository/
├── green
│   ├── 1.mp4
│   └── 2.mp4
└── red
    ├── 3.mp4
    └── 4.mp4
```

使用此结构创建的数据集包含两列：`video`和`label`（值为`green`和`red`）。

您还可以提供多个拆分。为此，您的数据集目录应具有以下结构（有关更多信息，请参阅[File names and splits](./datasets-file-names-and-splits)）：

```
my_dataset_repository/
├── test
│   ├── green
│   │   └── 2.mp4
│   └── red
│       └── 4.mp4
└── train
    ├── green
    │   └── 1.mp4
    └── red
        └── 3.mp4
```您可以在 [YAML configuration](./datasets-manual-configuration) 中禁用自动添加 `label` 列。如果您的目录名称没有特殊含义，请在 README 标头中设置 `drop_labels: true`：

```yaml
configs:
  - config_name: default  # Name of the dataset subset, if applicable.
    drop_labels: true
```

## 大规模数据集

### Web数据集格式

[WebDataset](./datasets-webdataset) 格式非常适合大规模视频数据集。
它由包含视频及其元数据的 TAR 档案组成，并针对流媒体进行了优化。如果您有大量视频并需要获取流数据加载器以进行大规模训练，那么它会很有用。

```
my_dataset_repository/
├── train-0000.tar
├── train-0001.tar
├── ...
└── train-1023.tar
```

要制作 WebDataset TAR 存档，请创建一个包含要存档的视频和元数据文件的目录，并使用例如创建 TAR 存档`tar` 命令。
每个存档的大小通常约为 1GB。
确保每个视频和元数据对共享相同的文件前缀，例如：

```
train-0000/
├── 000.mp4
├── 000.json
├── 001.mp4
├── 001.json
├── ...
├── 999.mp4
└── 999.json
```

请注意，为了方便用户并启用 [Dataset Viewer](./data-studio)，Hub 中托管的每个数据集都会自动转换为最大 5GB 的 Parquet 格式。由于视频可能非常大，因此视频的 URL 存储在转换后的 Parquet 数据中，而不包含视频字节本身。在 [Parquet format](./data-studio#access-the-parquet-files) 文档中阅读更多相关信息。

### 在拥抱脸部时使用 MLX
https://huggingface.co/docs/hub/mlx.md