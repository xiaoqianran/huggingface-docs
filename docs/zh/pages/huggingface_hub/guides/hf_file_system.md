<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通过文件系统 API 与 Hub 交互

除了 [HfApi](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_api#huggingface_hub.HfApi) 之外，`huggingface_hub` 库还提供 [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)，这是 Hugging Face Hub 的 Pythonic [fsspec-compatible](https://filesystem-spec.readthedocs.io/en/latest/) 文件接口。 [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 构建在 [HfApi](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_api#huggingface_hub.HfApi) 之上，并提供典型的文件系统样式操作，如 `cp`、`mv`、`ls`、`du`、`glob`、`get_file` 和 `put_file`。

> [!警告]
> [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 提供 fsspec 兼容性，这对于需要它的库很有用（例如，阅读
> 直接使用 `pandas` 拥抱人脸数据集和存储桶）。然而，由于这种兼容性，它引入了额外的开销
> 层。为了获得更好的性能和可靠性，建议尽可能使用[HfApi](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_api#huggingface_hub.HfApi)方法。

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

与Python内置的`open`不同，`fsspec`的`open`默认为二进制模式`"rb"`。这意味着您必须明确将模式设置为`"r"`（用于在文本模式下读取）和`"w"`（用于在文本模式下写入）。尚不支持追加到文件（模式 `"a"` 和 `"ab"`）。

## 列出大型存储库递归列出文件的操作，例如`find`和`glob`，需要获取整个文件
存储库的内容。中心逐页返回这些条目，并且只有在返回后才能请求页面
已收到上一个：列出包含 800k 文件的存储库需要约 800 个连续请求，并且
因此大约需要30分钟。

为了避免这种情况，[HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)并行地按目录列出大型存储库：

```python
>>> from huggingface_hub import hffs

>>> # Takes ~30s on a repository holding 800k files
>>> files = hffs.glob("datasets/allenai/objaverse/**/*.glb", maxdepth=3)
```

并发请求数由[HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)的`max_workers`参数控制。默认为 16
并且可以设置为 1 以禁用并行列表。
请注意，传递 `maxdepth` 也会显着加快列表速度，因为 Hub 仅返回条目
实际要求的。

## 保持列表新鲜

目录只会列出一次，并且其内容会被缓存，因此其他人刚刚更改的目录不会被缓存
自动拾取。 [HfFileSystem.invalidate_cache()](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem.invalidate_cache) 删除缓存列表，无论是对于一个路径（及其
父母）或在任何地方调用时不带参数：

```python
>>> hffs.ls("buckets/my-username/my-bucket/data", detail=False)  # listing is cached
>>> # ... another process uploads a file to the bucket ...
>>> hffs.invalidate_cache("buckets/my-username/my-bucket/data")
>>> hffs.ls("buckets/my-username/my-bucket/data", detail=False)  # listed again
['buckets/my-username/my-bucket/data/train.csv', 'buckets/my-username/my-bucket/data/new-file.csv']
```

对于存储桶，`huggingface_hub` 可以为您做到这一点：集线器在存储桶发生文件更改时流式传输
[HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 一旦发生更改，就会删除受更改影响的列表。启用它
`live_follow=True`：

```python
>>> from huggingface_hub import HfFileSystem
>>> hffs = HfFileSystem(live_follow=True)
>>> hffs.ls("buckets/my-username/my-bucket/data", detail=False)  # starts following that bucket
>>> # ... another process uploads a file to the bucket ...
>>> hffs.ls("buckets/my-username/my-bucket/data", detail=False)  # refreshed as changes arrive
['buckets/my-username/my-bucket/data/train.csv', 'buckets/my-username/my-bucket/data/new-file.csv']
```它还可以通过 fsspec、`fsspec.filesystem("hf", live_follow=True)` 或通过
`storage_options` 接受一个库的参数（pandas，arrow，...）。

关于它有几点需要了解：

- **仅限存储桶。** 模型、数据集和空间存储库是 git 存储库，Hub 不会传输它们的数据
  更改：将 `refresh=True` 传递给 [HfFileSystem.ls()](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem.ls) （或使用 [HfApi.list_repo_tree()](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree)）以查看新提交。
- **仅刷新列表及其携带的元数据。** 读取文件总是请求其内容
  Hub，因此实时关注仅影响`ls`，`info`，`exists`，`glob`，...报告。
- 列出存储桶后立即开始关注，并且每个关注的一个后台连接保持打开状态
  存储桶，由所有[HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)实例共享。这就是为什么该选项是选择加入的原因。

## 集成

[HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 可以与集成 `fsspec` 的任何库一起使用，前提是 URL 遵循以下方案：

```
hf://[<repo_type_prefix>]<repo_id>[@<revision>]/<path/in/repo>
```

对于数据集，`repo_type_prefix` 是 `datasets/`，对于空间，`spaces/`，并且模型不需要在 URL 中添加前缀。

除了存储库之外，[HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)还支持Hugging Face Buckets，这是一种类似S3的对象存储（有关更多详细信息，请参阅[this guide](./buckets)）：

```
hf://buckets/<bucket_id>/<path/in/bucket>
```

下面列出了一些有趣的集成，其中 [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 简化了与 Hub 的交互：* 从 Hub 存储库读取/写入 [Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html#reading-writing-remote-files) DataFrame：

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
  >>> root = zarr.open_group("hf://my-username/my-model-repo/array-store", mode="w")
  >>> foo = root.create_group("embeddings")
  >>> foobar = foo.zeros(name="experiment_0", shape=(50000, 1000), chunks=(10000, 1000), dtype="f4")
  >>> foobar[:] = embeddings

  >>> # Read an array from a repo
  >>> root = zarr.open_group("hf://my-username/my-model-repo/array-store", mode="r")
  >>> first_row = root["embeddings/experiment_0"][0]
  ```

## 身份验证

在许多情况下，您必须使用 Hugging Face 帐户登录才能与 Hub 交互。请参阅文档的[Authentication](../quick-start#authentication)部分，了解有关集线器上的身份验证方法的更多信息。

也可以通过将 `token` 作为参数传递给 [HfFileSystem](/docs/huggingface_hub/v2.4.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 以编程方式登录：

```python
>>> from huggingface_hub import HfFileSystem
>>> hffs = HfFileSystem(token=token)
```

如果您以这种方式登录，请注意在分享源代码时不要意外泄漏令牌！

### 运行和管理作业
https://huggingface.co/docs/huggingface_hub/v2.4.0/guides/jobs.md