<!-- huggingface-docs: machine-translated zh-CN from English source -->

#PyArrow

[Arrow](https://github.com/apache/arrow) 是一种柱状格式和用于快速数据交换和内存分析的工具箱。
由于PyArrow支持[fsspec](https://filesystem-spec.readthedocs.io)读写远程数据，因此您可以使用Hugging Face路径（[⟦T16⟧](/docs/huggingface_hub/guides/hf_file_system#integrations)）在Hub上读写数据。
它对于 [Parquet](https://parquet.apache.org/) 数据特别有用，因为 Parquet 是 Hugging Face 上最常见的文件格式。
事实上，Parquet 由于其结构、类型、元数据和压缩而特别高效。

## 加载表

您可以从本地文件或远程存储（例如拥抱脸部数据集）加载数据。 PyArrow 支持多种格式，包括 CSV、JSON 以及更重要的是 Parquet：

```python
>>> import pyarrow.parquet as pq
>>> table = pq.read_table("path/to/data.parquet")
```

要从 Hugging Face 加载文件，路径需要以 `hf://` 开头。例如，[stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb)数据集存储库的路径为`hf://datasets/stanfordnlp/imdb`。 Hugging Face 上的数据集包含多个 Parquet 文件。 Parquet 文件格式旨在提高数据帧的读写效率，并使跨数据分析语言共享数据变得容易。以下是如何将文件 `plain_text/train-00000-of-00001.parquet` 作为 pyarrow Table 加载（它需要 `pyarrow>=21.0`）：

```python
>>> import pyarrow.parquet as pq
>>> table = pq.read_table("hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
>>> table
pyarrow.Table
text: string
label: int64
----
text: [["I rented I AM CURIOUS-YELLOW from my video store because of all the controversy that surrounded it (... 1542 chars omitted)", ...],...,[..., "The story centers around Barry McKenzie who must go to England if he wishes to claim his inheritan (... 221 chars omitted)"]]
label: [[0,0,0,0,0,...,0,0,0,0,0],...,[1,1,1,1,1,...,1,1,1,1,1]]
```

如果您不想加载完整的 Parquet 数据，您可以获取 Parquet 元数据或按行组加载行组：

```python
>>> import pyarrow.parquet as pq
>>> pf = pq.ParquetFile("hf://datasets/stanfordnlp/imdb/plain_text/train-00000-of-00001.parquet")
>>> pf.metadata
<pyarrow._parquet.FileMetaData object at 0x1171b4090>
  created_by: parquet-cpp-arrow version 12.0.0
  num_columns: 2
  num_rows: 25000
  num_row_groups: 25
  format_version: 2.6
  serialized_size: 62036
>>> for i in pf.num_row_groups:
...     table = pf.read_row_group(i)
...     ...
```有关 Hugging Face 路径及其实现方式的更多信息，请参阅[the client library's documentation on the HfFileSystem](/docs/huggingface_hub/guides/hf_file_system)。

## 保存表格

您可以使用 `pyarrow.parquet.write_table` 将 pyarrow Table 保存到本地文件或直接保存到 Hugging Face。

要保存 Hugging Face 上的表格，您首先需要[Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)，例如使用：

```
hf auth login
```

然后你可以[create a dataset repository](/docs/huggingface_hub/quick-start#create-a-repository)，例如使用：

```python
from huggingface_hub import HfApi

HfApi().create_repo(repo_id="username/my_dataset", repo_type="dataset")
```

最后，您可以在 PyArrow 中使用 [Hugging Face paths](/docs/huggingface_hub/guides/hf_file_system#integrations)：

```python
import pyarrow.parquet as pq

pq.write_table(table, "hf://datasets/username/my_dataset/imdb.parquet", use_content_defined_chunking=True)

# or write in separate files if the dataset has train/validation/test splits
pq.write_table(table_train, "hf://datasets/username/my_dataset/train.parquet", use_content_defined_chunking=True)
pq.write_table(table_valid, "hf://datasets/username/my_dataset/validation.parquet", use_content_defined_chunking=True)
pq.write_table(table_test , "hf://datasets/username/my_dataset/test.parquet", use_content_defined_chunking=True)
```

请注意，Hugging Face 上的 Parquet 文件经过优化，可提高存储效率、加速下载和上传，并实现高效的数据集流式传输和编辑：

* [Parquet Content Defined Chunking](https://huggingface.co/blog/parquet-cdc) 针对 Hugging Face 的存储后端[Xet](https://huggingface.co/docs/hub/en/xet/index) 优化了 Parquet。由于基于块的重复数据删除，它可以加速上传和下载，并允许高效的文件编辑
* 页面索引可在流式传输时加速过滤器并实现高效的随机访问，例如在[Dataset Viewer](https://huggingface.co/docs/dataset-viewer)

PyArrow 需要额外的参数来编写优化的 Parquet 文件：

```python
import pyarrow.parquet as pq

pq.write_table(
    table,
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)
```

* `use_content_defined_chunking=True` 为 [deduplication](https://huggingface.co/blog/parquet-cdc) 和 [editing](./datasets-editing) 启用 Parquet 内容定义分块（需要 `pyarrow>=21.0`）
* `write_page_index=True` 在 Parquet 元数据中包含页面索引，对于 [streaming and random access](./datasets-streaming)> [!提示]
> 内容定义分块 (CDC) 使 Parquet 编写器以相同方式对重复数据进行分块和压缩的方式对数据页进行分块。
> 如果没有 CDC，页面会被任意分块，因此由于压缩而无法检测到重复数据。
> 感谢 CDC，Hugging Face 的 Parquet 上传和下载速度更快，因为重复数据仅上传或下载一次。

查找有关 Xet [here](https://huggingface.co/join/xet) 的更多信息。

## 利用 Xet 重复数据删除技术进行 Parquet

优化的 Parquet 文件是使用内容定义分块写入的，可实现重复数据删除。
这可以加速上传，因为 Hugging Face 上已经存在的数据块不需要再次上传，从而节省了大量的 I/O。

例如，此代码上传`table`的内容，然后对于`edited_table`，上传速度更快，因为它只上传更改的块：

```python
import pyarrow.parquet as pq

pq.write_table(
    table,
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)

edited_table = ...  # e.g. with added/modified/removed rows or columns

pq.write_table(
    edited_table,
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)
```

块约为 64kB，Parquet 每列保存数据列，因此实际上，编辑优化的 Parquet 文件时会发生以下情况：

* 添加一个新列 -> 仅上传新列的块
* 添加/编辑/删除一行 -> 每列上传一个块除此之外，包含元数据的 Parquet 页脚块也会被上传。

## 使用图像

您可以加载包含元数据文件的文件夹，其中包含图像名称或路径的字段，结构如下：

```
Example 1:            Example 2:
folder/               folder/
├── metadata.parquet  ├── metadata.parquet
├── img000.png        └── images
├── img001.png            ├── img000.png
...                       ...
└── imgNNN.png            └── imgNNN.png
```

您可以像这样迭代图像路径：

```python
from pathlib import Path
import pyarrow as pq

folder_path = Path("path/to/folder")
table = pq.read_table(folder_path + "metadata.parquet")
for file_name in table["file_name"].to_pylist():
    image_path = folder_path / file_name
    ...
```

由于数据集位于 [supported structure](https://huggingface.co/docs/hub/en/datasets-image#additional-columns)（带有 `file_name` 字段的 `metadata.parquet` 文件）中，因此您可以将此数据集保存到 Hugging Face，数据集查看器会同时显示元数据和图像。

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=folder_path,
    repo_id="username/my_image_dataset",
    repo_type="dataset",
)
```

### 在 Parquet 中嵌入图像

PyArrow 具有二进制类型，允许在 Arrow 表中包含图像字节。因此，它可以将数据集保存为一个包含图像（字节和路径）和样本元数据的 Parquet 文件：

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Embed the image bytes in Arrow
image_array = pa.array([
    {
        "bytes": (folder_path / file_name).read_bytes(),
        "path": file_name,
    }
    for file_name in table["file_name"].to_pylist()
])
table.append_column("image", image_array)

# (Optional) Set the HF Image type for the Dataset Viewer and the `datasets` library
features = {"image": {"_type": "Image"}}  # or using datasets.Features(...).to_dict()
schema_metadata = {"huggingface": {"dataset_info": {"features": features}}}
table = table.replace_schema_metadata(schema_metadata)

# Save to Parquet
# (Optional) with use_content_defined_chunking for faster uploads and downloads
pq.write_table(table, "data.parquet", use_content_defined_chunking=True)
```

在箭头模式元数据中设置图像类型允许其他库和拥抱面部数据集查看器知道“图像”包含图像而不仅仅是二进制数据。

## 使用音频

您可以加载包含元数据文件的文件夹，其中包含音频名称或路径的字段，结构如下：

```
Example 1:            Example 2:
folder/               folder/
├── metadata.parquet  ├── metadata.parquet
├── rec000.wav        └── audios
├── rec001.wav            ├── rec000.wav
...                       ...
└── recNNN.wav            └── recNNN.wav
```

您可以像这样迭代音频路径：

```python
from pathlib import Path
import pyarrow as pq

folder_path = Path("path/to/folder")
table = pq.read_table(folder_path + "metadata.parquet")
for file_name in table["file_name"].to_pylist():
    audio_path = folder_path / file_name
    ...
```由于数据集位于 [supported structure](https://huggingface.co/docs/hub/en/datasets-audio#additional-columns)（带有 `file_name` 字段的 `metadata.parquet` 文件）中，因此您可以将其保存到 Hugging Face，并且 Hub 数据集查看器会同时显示元数据和音频。 

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path=folder_path,
    repo_id="username/my_audio_dataset",
    repo_type="dataset",
)
```

### 在 Parquet 中嵌入音频

PyArrow 具有二进制类型，允许在 Arrow 表中包含音频字节。因此，它可以将数据集保存为一个包含音频（字节和路径）和样本元数据的 Parquet 文件：

```python
import pyarrow as pa
import pyarrow.parquet as pq

# Embed the audio bytes in Arrow
audio_array = pa.array([
    {
        "bytes": (folder_path / file_name).read_bytes(),
        "path": file_name,
    }
    for file_name in table["file_name"].to_pylist()
])
table.append_column("audio", audio_array)

# (Optional) Set the HF Audio type for the Dataset Viewer and the `datasets` library
features = {"audio": {"_type": "Audio"}}  # or using datasets.Features(...).to_dict()
schema_metadata = {"huggingface": {"dataset_info": {"features": features}}}
table = table.replace_schema_metadata(schema_metadata)

# Save to Parquet
# (Optional) with use_content_defined_chunking for faster uploads and downloads
pq.write_table(table, "data.parquet", use_content_defined_chunking=True)
```

在箭头架构元数据中设置音频类型使其他库和拥抱面部数据集查看器能够识别“音频”包含音频数据，而不仅仅是二进制数据。

### 自定义 Python 空间
https://huggingface.co/docs/hub/spaces-sdks-python.md