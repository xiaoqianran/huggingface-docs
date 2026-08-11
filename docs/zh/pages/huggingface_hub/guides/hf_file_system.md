<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通过文件系统 API 与 Hub 交互

除了 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 之外，`huggingface_hub` 库还提供 [HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)，这是 Hugging Face Hub 的 Pythonic [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) 文件接口。 [HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 构建在 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 之上，并提供典型的文件系统样式操作，如 `cp`、`mv`、`ls`、`du`、`glob`、`get_file` 和 `put_file`。

> [!警告]
> [HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 提供 fsspec 兼容性，这对于需要它的库很有用（例如，阅读
> 直接使用 `pandas` 拥抱人脸数据集和存储桶）。然而，由于这种兼容性，它引入了额外的开销
> 层。为了获得更好的性能和可靠性，建议尽可能使用[HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi)方法。

## 用法

```python
>>> from huggingface_hub import hffs

>>> # List all files in a dataset directory
>>> hffs.ls("datasets/my-username/my-dataset-repo/data", detail=False)
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # List all files in a bucket directory
>>> hffs.ls("buckets/my-username/my-bucket/experiment-data", detail=False)
['bucket/my-username/my-bucket/data/train-0000.parquet', 'bucket/my-username/my-bucket/data/train-0001.parquet', ...]

>>> # List all ".csv" files in a dataset repository
>>> hffs.glob("datasets/my-username/my-dataset-repo/**/*.csv")
['datasets/my-username/my-dataset-repo/data/train.csv', 'datasets/my-username/my-dataset-repo/data/test.csv']

>>> # Read a remote file
>>> with hffs.open("datasets/my-username/my-dataset-repo/data/train.csv", "r") as f:
...     train_data = f.readlines()

>>> # Read the content of a remote file as a string
>>> train_data = hffs.read_text("datasets/my-username/my-dataset-repo/data/train.csv", revision="dev")

>>> # Write a remote file
>>> with hffs.open("datasets/my-username/my-dataset-repo/data/validation.csv", "w") as f:
...     f.write("text,label")
...     f.write("Fantastic movie!,good")
```

可以传递可选的 `revision` 参数来从特定提交（例如分支、标记名称或提交哈希）运行操作。请注意，`revision` 与 Buckets 不兼容。 

与Python内置的`open`不同，`fsspec`的`open`默认为二进制模式`"rb"`。这意味着您必须显式地将模式设置为`"r"`（用于在文本模式下读取）和`"w"`（用于在文本模式下写入）。尚不支持附加到文件（模式 `"a"` 和 `"ab"`）。

## 集成[HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 可以与集成 `fsspec` 的任何库一起使用，前提是 URL 遵循以下方案：

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

对于数据集，`repo_type_prefix` 是`datasets/`，对于空间，`spaces/`，并且模型不需要在 URL 中添加前缀。

除了存储库之外，[HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)还支持Hugging Face Buckets，这是一种类似S3的对象存储（有关更多详细信息，请参阅[this guide](./buckets)）：

```
hf://buckets/<bucket_id>/<path/in/bucket>
```

下面列出了一些有趣的集成，其中 [HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 简化了与 Hub 的交互：

* 从 Hub 存储库读取/写入 [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) DataFrame：

  ```python
  >>> import pandas as pd

  >>> # Read a remote CSV file into a dataframe
  >>> df = pd.read_csv("hf://datasets/my-username/my-dataset-repo/train.csv")
  >>> df = pd.read_csv("hf://buckets/my-username/my-bucket/train.csv")

  >>> # Write a dataframe to a remote CSV file
  >>> df.to_csv("hf://datasets/my-username/my-dataset-repo/test.csv")
  >>> df.to_csv("hf://buckets/my-username/my-bucket/test.csv")
  ```

相同的工作流程也可用于 [Dask](https://docs.dask.org/en/stable/how-to/connect-to-remote-data.html) 和 [Polars](https://pola-rs.github.io/polars/py-polars/html/reference/io.html) DataFrame。

* 使用 [DuckDB](https://duckdb.org/docs/guides/python/filesystems) 查询（远程）Hub 文件：

  ```python
  >>> from huggingface_hub import HfFileSystem
  >>> import duckdb

  >>> fs = HfFileSystem()
  >>> duckdb.register_filesystem(fs)
  >>> # Query a remote file and get the result back as a dataframe
  >>> fs_query_file = "hf://datasets/my-username/my-dataset-repo/data_dir/data.parquet"
  >>> df = duckdb.query(f"SELECT * FROM '{fs_query_file}' LIMIT 10").df()
  ```

* 使用 Hub 作为数组存储，并使用 [Zarr](https://zarr.readthedocs.io/en/stable/tutorial.html#io-with-fsspec)：

  ```python
  >>> import numpy as np
  >>> import zarr

  >>> embeddings = np.random.randn(50000, 1000).astype("float32")

  >>> # Write an array to a repo
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w") as root:
  ...    foo = root.create_group("embeddings")
  ...    foobar = foo.zeros('experiment_0', shape=(50000, 1000), chunks=(10000, 1000), dtype='f4')
  ...    foobar[:] = embeddings

  >>> # Read an array from a repo
  >>> with zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r") as root:
  ...    first_row = root["embeddings/experiment_0"][0]
  ```

## 身份验证

在许多情况下，您必须使用 Hugging Face 帐户登录才能与 Hub 交互。请参阅文档的[Authentication](../quick-start#authentication)部分，了解有关集线器上的身份验证方法的更多信息。

也可以通过将 `token` 作为参数传递给 [HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 以编程方式登录：

```python
>>> from huggingface_hub import HfFileSystem
>>> hffs = HfFileSystem(token=token)
```

如果您以这种方式登录，请注意在分享源代码时不要意外泄漏令牌！### 序列化
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/serialization.md