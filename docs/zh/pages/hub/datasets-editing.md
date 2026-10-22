<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 编辑数据集

[Hub](https://huggingface.co/datasets) 支持社区和研究数据集的协作管理。我们鼓励您探索 Hub 上提供的数据集并为其改进做出贡献，以帮助发展 ML 社区并加速每个人的进步。欢迎所有贡献！

如果您还没有，请从 [creating a Hugging Face Hub account](https://huggingface.co/join) 开始。

## 使用 Hub UI 进行编辑

> [!警告]
> 此功能目前仅适用于 CSV、TSV 和 Parquet 数据集。

Hub 的 Web 界面允许没有任何技术专业知识的用户编辑数据集。

打开数据集页面并导航到 **Data Studio** 选项卡以开始编辑。

单击 **切换编辑模式** 以启用数据集编辑。

根据需要编辑任意数量的单元格，最后单击“**提交**”以提交更改并留下提交消息。

## 使用`huggingface_hub`客户端库

`huggingface_hub` 库可以管理 Hub 存储库，包括编辑数据集。

例如，以下是如何使用 [Hugging Face FileSystem API](https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system) 编辑 CSV 文件：

```python
from huggingface_hub import hffs

path = f"datasets/{repo_id}/data.csv"

with hffs.open(path, "r") as f:
    content = f.read()
edited_content = content.replace("foo", "bar")
with hffs.open(path, "w") as f:
    f.write(edited_content)
```

您还可以在磁盘上本地应用编辑并提交更改：

```python
from huggingface_hub import hf_hub_download, upload_file

local_path = hf_hub_download(repo_id=repo_id, path_in_repo= "data.csv", repo_type="dataset")

with open(path, "r") as f:
    content = f.read()
edited_content = content.replace("foo", "bar")
with open(path, "w") as f:
    f.write(edited_content)

upload_file(repo_id=repo_id, path_in_repo=local_path, repo_type="dataset")
```

> [!提示]
>
> 要在本地拥有整个数据集存储库并同时编辑多个文件，请使用 `snapshot_download` 和 `upload_folder` 而不是 `hf_hub_download` 和 `upload_file`请访问[the client library's documentation](/docs/huggingface_hub/index)了解更多信息。

## 集成库

如果 Hub 上的数据集与 [supported library](./datasets-libraries) 兼容，则加载、编辑和推送数据集只需几行代码。

Here is how to edit a CSV file with Pandas:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(f"hf://datasets/{repo_id}/data.csv")

# Edit
df = df.apply(...)

# Commit the changes
df.to_csv(f"hf://datasets/{repo_id}/data.csv")
```

像 Polars 和 DuckDB 这样的库也实现了 `hf://` 协议来读取、编辑和写入 Hugging Face 上的文件。其他库可用于编辑由许多文件组成的数据集，例如 Spark、Dask 或 🤗 数据集。 See the full list of supported libraries [here](./datasets-libraries)

有关访问网站上的数据集的信息，您可以单击数据集页面上的“使用此数据集”按钮以了解如何操作。
For example, [⟦T12⟧](https://huggingface.co/datasets/knkarthick/samsum?library=datasets) shows how to do so with 🤗 Datasets below.

## Only upload the new data

Hugging Face's storage is powered by [Xet](https://huggingface.co/docs/hub/en/xet), which uses chunk deduplication to make uploads more efficient.
与传统云存储不同，Xet 不需要重新上传整个数据集来提交更改。
相反，它会自动检测数据集的哪些部分已更改，并指示客户端库仅上传更新的部分。
To do that, Xet uses a smart algorithm to find chunks of 64kB that already exist on Hugging Face.Let's see our previous example with Pandas:

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(f"hf://datasets/{repo_id}/data.csv")

# Edit part of the dataset
df = df.apply(...)

# Commit the changes
df.to_csv(f"hf://datasets/{repo_id}/data.csv")
```

This code first loads a dataset and then edits it.
编辑完成后，`to_csv()` 在内存中具体化文件，对其进行分块，询问 Xet 哪些块已经在 Hugging Face 上以及哪些块已更改，然后仅上传新数据。

## Optimized Parquet editing

The amount of data to upload depends on the edits and the file structure.

The Parquet format is columnar and compressed at the page level (pages are around ~1MB).
We optimized Parquet for Xet with [Parquet Content Defined Chunking](https://huggingface.co/blog/parquet-cdc), which ensures unchanged data generally result in unchanged pages.

例如，此代码上传`df`的内容，然后对于`edited_df`，上传速度更快，因为它只上传更改的块：

```python
import pandas as pd

df.to_parquet(
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)

edited_df = ...  # e.g. with added/modified/removed rows or columns

edited_df.to_parquet(
    "hf://datasets/username/my_dataset/imdb.parquet",
    # Optimize for Xet
    use_content_defined_chunking=True,
    write_page_index=True,
)
```

Chunks are ~64kB and Parquet saves data column per column, so in practice this is what happens when editing an Optimized Parquet file:

* add a new column -> only the chunks of the new column are uploaded
* add/edit/delete a row -> one chunk per column is uploaded

And in addition to this, the chunks of the Parquet footer containing metadata are also uploaded.

Check out if your library supports Optimized Parquet in the [supported libraries](./datasets-libraries) page.

## 流媒体对于大数据集，建议使用具有用于端到端流管道的数据集流功能的库。
在这种情况下，随着旧数据到达并且新数据上传到集线器，数据集处理逐渐运行。

在 [supported libraries](./datasets-libraries) 页面查看您的图书馆是否支持流式传输。

### 特工踪迹
https://huggingface.co/docs/hub/agent-traces.md