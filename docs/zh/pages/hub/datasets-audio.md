<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 音频数据集

本指南将向您展示如何使用音频文件配置数据集存储库。您可以在此 [Audio datasets examples collection](https://huggingface.co/collections/datasets-examples/audio-dataset-66aca0b73e8f69e3d069e607) 中找到随附的存储库示例。

具有受支持结构和 [file formats](./datasets-adding#file-formats) 的数据集会自动在 Hub 的页面上显示数据集查看器。

---

只要您将有关音频文件的其他信息（例如转录）包含在元数据文件 (`metadata.csv`/`metadata.jsonl`/`metadata.parquet`) 中，就会自动加载。

或者，音频文件可以采用 Parquet 文件或遵循 [WebDataset](https://github.com/webdataset/webdataset) 格式的 TAR 存档。

## 仅音频文件

如果您的数据集仅包含一列音频，您可以简单地将音频文件存储在根目录中：

```plaintext
my_dataset_repository/
├── 1.wav
├── 2.wav
├── 3.wav
└── 4.wav
```

或在子目录中：

```plaintext
my_dataset_repository/
└── audio
    ├── 1.wav
    ├── 2.wav
    ├── 3.wav
    └── 4.wav
```

同时支持多个[formats](./datasets-adding#file-formats)，包括AIFF、FLAC、MP3、OGG和WAV。

```plaintext
my_dataset_repository/
└── audio
    ├── 1.aiff
    ├── 2.ogg
    ├── 3.mp3
    └── 4.flac
```

如果您有多个分割，您可以将音频文件放入相应命名的目录中：

```plaintext
my_dataset_repository/
├── train
│   ├── 1.wav
│   └── 2.wav
└── test
    ├── 3.wav
    └── 4.wav
```

请参阅 [File names and splits](./datasets-file-names-and-splits) 了解更多信息以及通过拆分组织数据的其他方法。

## 附加列如果您想包含有关数据集的其他信息（例如转录），请将其作为 `metadata.csv` 文件添加到存储库中。这使您可以快速为不同的音频任务（例如[text-to-speech](https://huggingface.co/tasks/text-to-speech)或[automatic speech recognition](https://huggingface.co/tasks/automatic-speech-recognition)）创建数据集。

```plaintext
my_dataset_repository/
├── 1.wav
├── 2.wav
├── 3.wav
├── 4.wav
└── metadata.csv
```

您的 `metadata.csv` 文件必须有一个 `file_name` 列，用于链接图像文件及其元数据：

```csv
file_name,animal
1.wav,cat
2.wav,cat
3.wav,dog
4.wav,dog
```

您还可以使用 [JSONL](https://jsonlines.org/) 文件 `metadata.jsonl`：

```jsonl
{"file_name": "1.wav","text": "cat"}
{"file_name": "2.wav","text": "cat"}
{"file_name": "3.wav","text": "dog"}
{"file_name": "4.wav","text": "dog"}
```

对于更大的数据集或者如果您对高级数据检索功能感兴趣，您可以使用 [Parquet](https://parquet.apache.org/) 文件 `metadata.parquet`。

## 相对路径

元数据文件必须位于与其链接的音频文件相同的目录中，或者位于任何父目录中，如下例所示：

```plaintext
my_dataset_repository/
└── test
    ├── audio
    │   ├── 1.wav
    │   ├── 2.wav
    │   ├── 3.wav
    │   └── 4.wav
    └── metadata.csv
```

在这种情况下，`file_name`列必须是音频文件的完整相对路径，而不仅仅是文件名：

```csv
file_name,animal
audio/1.wav,cat
audio/2.wav,cat
audio/3.wav,dog
audio/4.wav,dog
```

元数据文件不能与音频文件放在同一目录的子目录中。

更一般地说，任何名为 `file_name` 或 `*_file_name` 的列都应包含音频文件的完整相对路径。

在本例中，`test`目录用于设置训练分割的名称。请参阅[File names and splits](./datasets-file-names-and-splits)了解更多信息。

## 音频分类对于音频分类数据集，您还可以使用简单的设置：使用目录来命名音频类。将音频文件存储在如下目录结构中：

```plaintext
my_dataset_repository/
├── cat
│   ├── 1.wav
│   └── 2.wav
└── dog
    ├── 3.wav
    └── 4.wav
```

使用此结构创建的数据集包含两列：`audio`和`label`（值为`cat`和`dog`）。

您还可以提供多个拆分。为此，您的数据集目录应具有以下结构（有关更多信息，请参阅[File names and splits](./datasets-file-names-and-splits)）：

```plaintext
my_dataset_repository/
├── test
│   ├── cat
│   │   └── 2.wav
│   └── dog
│       └── 4.wav
└── train
    ├── cat
    │   └── 1.wav
    └── dog
        └── 3.wav
```

您可以在 [YAML configuration](./datasets-manual-configuration) 中禁用自动添加 `label` 列。如果您的目录名称没有特殊含义，请在 README 标头中设置 `drop_labels: true`：

```yaml
configs:
  - config_name: default  # Name of the dataset subset, if applicable.
    drop_labels: true
```

## 大规模数据集

### Web数据集格式

[WebDataset](./datasets-webdataset) 格式非常适合大规模音频数据集（例如，参见 [AlienKevin/sbs_cantonese](https://huggingface.co/datasets/AlienKevin/sbs_cantonese)）。
它由包含音频文件及其元数据的 TAR 存档组成，并针对流媒体进行了优化。如果您有大量音频文件并需要获取流数据加载器以进行大规模训练，那么它会很有用。

```plaintext
my_dataset_repository/
├── train-0000.tar
├── train-0001.tar
├── ...
└── train-1023.tar
```要制作 WebDataset TAR 存档，请创建一个包含要存档的音频文件和元数据文件的目录，并使用例如创建 TAR 存档`tar` 命令。
每个存档的大小通常约为 1GB。
确保每个音频文件和元数据对共享相同的文件前缀，例如：

```plaintext
train-0000/
├── 000.flac
├── 000.json
├── 001.flac
├── 001.json
├── ...
├── 999.flac
└── 999.json
```

请注意，为了方便用户并启用 [Dataset Viewer](./data-studio)，Hub 中托管的每个数据集都会自动转换为最大 5GB 的 Parquet 格式。
请在 [Parquet format](./data-studio#access-the-parquet-files) 文档中阅读更多相关信息。

### 镶木地板格式

您可以将所有内容嵌入到 [Parquet](https://parquet.apache.org/) 文件中，而不是将音频文件和元数据作为单独的文件上传。
如果您有大量音频文件、想要嵌入多个音频列或者想要在同一文件中存储有关音频的其他信息，这会非常有用。
Parquet 对于存储原始字节等数据也很有用，而 JSON/CSV 不支持这种数据。

```plaintext
my_dataset_repository/
└── train.parquet
```可以使用 `pandas` 或 `datasets` 库创建包含音频数据的 Parquet 文件。要使用 `pandas` 中的音频数据创建 Parquet 文件，您可以使用 [pandas-audio-methods](https://github.com/lhoestq/pandas-audio-methods) 和 `df.to_parquet()`。在`datasets`中，您可以将列类型设置为`Audio()`并使用`ds.to_parquet(...)`方法或`ds.push_to_hub(...)`。您可以在 `datasets` [here](/docs/datasets/audio_load) 中找到有关加载音频数据集的指南。

或者，您可以手动设置使用其他工具创建的 Parquet 的音频类型。首先，确保您的音频列的类型为 _struct_，其中包含用于音频数据的二进制字段 `"bytes"` 和用于音频文件名或路径的字符串字段 `"path"`。然后，您应该直接在 README 标题中的 YAML 中指定列的功能类型，例如：

```yaml
dataset_info:
  features:
  - name: audio
    dtype: audio
  - name: caption
    dtype: string
```

请注意，对于小型音频文件（每个音频文件 <1MB）和小型行组（每个行组 100 行，`datasets` 用于音频），建议使用 Parquet。对于较大的音频文件，建议使用 WebDataset 格式，或共享原始音频文件（可选地与元数据文件）。

### 存储库
https://huggingface.co/docs/hub/repositories.md