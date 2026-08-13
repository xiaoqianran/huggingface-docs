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

以下是使用 Pandas 编辑 CSV 文件的方法：

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(f"hf://datasets/{repo_id}/data.csv")

# Edit
df = df.apply(...)

# Commit the changes
df.to_csv(f"hf://datasets/{repo_id}/data.csv")
```

Polars 和 DuckDB 等库也实现了 `hf://` 协议来读取、编辑和写入 Hugging Face 上的文件。其他库可用于编辑由许多文件组成的数据集，例如 Spark、Dask 或 🤗 数据集。查看支持的库的完整列表[here](./datasets-libraries)

有关访问网站上的数据集的信息，您可以单击数据集页面上的“使用此数据集”按钮以了解如何操作。
例如，[⟦T12⟧](https://huggingface.co/datasets/knkarthick/samsum?library=datasets)展示了如何使用下面的🤗数据集来做到这一点。

## 只上传新数据

Hugging Face 的存储由[Xet](https://huggingface.co/docs/hub/en/xet) 提供支持，它使用块重复数据删除来提高上传效率。
与传统云存储不同，Xet 不需要重新上传整个数据集来提交更改。
相反，它会自动检测数据集的哪些部分已更改，并指示客户端库仅上传更新的部分。
为此，Xet 使用智能算法来查找 Hugging Face 上已存在的 64kB 块。让我们看看之前的 Pandas 示例：

```python
import pandas as pd

# Load the dataset
df = pd.read_csv(f"hf://datasets/{repo_id}/data.csv")

# Edit part of the dataset
df = df.apply(...)

# Commit the changes
df.to_csv(f"hf://datasets/{repo_id}/data.csv")
```

此代码首先加载数据集，然后对其进行编辑。
编辑完成后，`to_csv()` 在内存中具体化文件，对其进行分块，询问 Xet 哪些块已经在 Hugging Face 上以及哪些块已更改，然后仅上传新数据。

## 优化 Parquet 编辑

要上传的数据量取决于编辑和文件结构。

Parquet 格式是柱状的，并在页面级别进行压缩（页面约为 1MB）。
我们使用 [Parquet Content Defined Chunking](https://huggingface.co/blog/parquet-cdc) 为 Xet 优化了 Parquet，这确保未更改的数据通常会导致未更改的页面。

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

块约为 64kB，Parquet 每列保存数据列，因此实际上，编辑优化的 Parquet 文件时会发生以下情况：

* 添加一个新列 -> 仅上传新列的块
* 添加/编辑/删除一行 -> 每列上传一个块

除此之外，包含元数据的 Parquet 页脚块也会被上传。

在 [supported libraries](./datasets-libraries) 页面查看您的图书馆是否支持 Optimized Parquet。

## 流媒体对于大数据集，建议使用具有用于端到端流管道的数据集流功能的库。
在这种情况下，随着旧数据到达并且新数据上传到集线器，数据集处理逐渐运行。

在 [supported libraries](./datasets-libraries) 页面查看您的图书馆是否支持流式传输。

### 存储桶
https://huggingface.co/docs/hub/storage-buckets.md