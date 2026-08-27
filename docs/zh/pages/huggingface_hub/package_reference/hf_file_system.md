<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 文件系统API

`HfFileSystem`类为基于[⟦T21⟧](https://filesystem-spec.readthedocs.io/en/latest/)的Hugging Face Hub提供了Pythonic文件接口。

## HfFileSystem[[huggingface_hub.HfFileSystem]]

`HfFileSystem`基于[fsspec](https://filesystem-spec.readthedocs.io/en/latest/)，因此它与其提供的大多数API兼容。有关更多详细信息，请查看 [our guide](../guides/hf_file_system) 和 fsspec 的 [API Reference](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem)。

#### Huggingface_hub.HfFileSystem[[huggingface_hub.HfFileSystem]]

```python
huggingface_hub.HfFileSystem(*args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L140)

**参数：**

端点（`str`，*可选*）：集线器的端点。默认为 .

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

block_size (`int`, *可选*) ：读写文件的块大小。

Expand_info (`bool`, *可选*) : 是否扩展文件信息。

- **storage_options**（`dict`，*可选*）：文件系统的附加选项。参见[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.__init__)。

访问远程 Hugging Face Hub 存储库，就像访问本地文件系统一样。> [!警告]
> [HfFileSystem](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem) 提供 fsspec 兼容性，这对于需要它的库很有用（例如，阅读
> 直接使用`pandas`拥抱人脸数据集）。然而，由于这种兼容性，它引入了额外的开销
> 层。为了获得更好的性能和可靠性，建议尽可能使用`HfApi`方法。

文件系统支持 `hf://` 协议的路径，该协议遵循以下 URL 方案：

* 模型、数据集和空间存储库：

```
hf://<repo-id>[@<revision>]/<path/in/repo>
hf://datasets/<repo-id>[@<revision>]/<path/in/repo>
hf://spaces/<repo-id>[@<revision>]/<path/in/repo>
```

* 桶（通用存储）：

```
hf://buckets/<bucket-id>/<path/in/bucket>
```

注意：当直接使用[HfFileSystem](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)时，在路径中传递`hf://`协议前缀是可选的。

用途：

```python
>>> from huggingface_hub import hffs

>>> # List files
>>> hffs.glob("my-username/my-model/*.bin")
['my-username/my-model/pytorch_model.bin']
>>> hffs.ls("datasets/my-username/my-dataset", detail=False)
['datasets/my-username/my-dataset/.gitattributes', 'datasets/my-username/my-dataset/README.md', 'datasets/my-username/my-dataset/data.json']

>>> # Read/write files
>>> with hffs.open("my-username/my-model/pytorch_model.bin") as f:
...     data = f.read()
>>> with hffs.open("my-username/my-model/pytorch_model.bin", "wb") as f:
...     f.write(data)
```

指定用于身份验证的令牌：
```python
>>> from huggingface_hub import HfFileSystem
>>> hffs = HfFileSystem(token=token)
```

#### cp_file[[huggingface_hub.HfFileSystem.cp_file]]

```python
cp_file(path1: str, path2: str, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L796)

**参数：**

path1 (`str`) ：从中复制的源路径。

path2 (`str`) ：要复制到的目标路径。

revision (`str`, *可选*) ：要从中复制的 git 修订版。

在存储库内或存储库之间复制文件。

> [!警告]
> 注意：如果可能，请使用 `HfApi.upload_file()` 以获得更好的性能。

#### 存在[[huggingface_hub.HfFileSystem.exists]]

```python
exists(path, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L991)

**参数：**

path (`str`) ：要检查的路径。

**返回：** `bool`

如果文件存在则为 True，否则为 False。检查文件是否存在。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.exists)。

> [!警告]
> 注意：如果可能，请使用 `HfApi.file_exists()` 以获得更好的性能。

#### 查找[[huggingface_hub.HfFileSystem.find]]

```python
find(path: str, maxdepth: int | None = None, withdirs: bool = False, detail: bool = False, refresh: bool = False, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L735)

**参数：**

path (`str`) : 列出文件的根路径。

maxdepth (`int`, *可选*) : 下降到子目录的最大深度。

withdirs (`bool`, *可选*) ：在输出中包含目录路径。默认为 False。

详细信息（`bool`，*可选*）：如果为True，则返回将路径映射到文件信息的字典。默认为 False。

刷新（`bool`，*可选*）：如果为 True，则绕过缓存并获取最新数据。默认为 False。

revision (`str`, *可选*) ：要列出的 git 修订版本。

**返回：** `Union[list[str], dict[str, dict[str, Any]]]`

文件信息的路径或字典列表。

列出路径下的所有文件。

更多详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.find)。

#### get_file[[huggingface_hub.HfFileSystem.get_file]]

```python
get_file(rpath, lpath, callback = <fsspec.callbacks.NoOpCallback object at 0x7fbf3136cbe0>, outfile = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L1078)

**参数：**

rpath (`str`) ：下载的远程路径。

lpath (`str`) ：下载到的本地路径。

回调（`Callback`，*可选*）：可选回调来跟踪下载进度。默认不回调。outfile (`IO`, *可选*) ：可选的要写入的类文件对象。如果提供，`lpath` 将被忽略。

将单个远程文件复制到本地。

> [!警告]
> 注意：如果可能，请使用 `HfApi.hf_hub_download()` 或 `HfApi.download_bucket_files` 以获得更好的性能。

#### glob[[huggingface_hub.HfFileSystem.glob]]

```python
glob(path: str, maxdepth: int | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L717)

**参数：**

path (`str`) ：要匹配的路径模式。

maxdepth (`int`, *可选*) : 下降到目录的最大深度。默认情况下，没有限制。

**返回：** `list[str]`

与模式匹配的路径列表。

通过全局匹配查找文件。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.glob)。

####信息[[huggingface_hub.HfFileSystem.info]]

```python
info(path: str, refresh: bool = False, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L881)

**参数：**

path (`str`) : 获取信息的路径。

刷新（`bool`，*可选*）：如果为 True，则绕过缓存并获取最新数据。默认为 False。

revision (`str`, *可选*) ：从中获取信息的 git 版本。

**返回：** `dict[str, Any]`

包含文件信息（类型、大小、提交信息等）的字典。

获取有关文件或目录的信息。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.info)。

> [!警告]
> 注意：如果可能，请使用 `HfApi.get_paths_info()` 或 `HfApi.repo_info()` 以获得更好的性能
> （或 `HfApi.get_bucket_paths_info()` 或 `HfApi.bucket_info()` 对于存储桶）#### invalidate_cache[[huggingface_hub.HfFileSystem.invalidate_cache]]

```python
invalidate_cache(path: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L355)

**参数：**

path (`str`, *可选*) : 从缓存中清除的路径。如果未提供，请清除整个缓存。

清除给定路径的缓存。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.invalidate_cache)。

#### isdir[[huggingface_hub.HfFileSystem.isdir]]

```python
isdir(path)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L1016)

**参数：**

path (`str`) ：要检查的路径。

**返回：** `bool`

如果路径是目录则为 True，否则为 False。

检查路径是否是目录。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.isdir)。

#### isfile[[huggingface_hub.HfFileSystem.isfile]]

```python
isfile(path)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L1034)

**参数：**

path (`str`) ：要检查的路径。

**返回：** `bool`

如果路径是文件则为 True，否则为 False。

检查路径是否是文件。

更多详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.isfile)。

#### ls[[huggingface_hub.HfFileSystem.ls]]

```python
ls(path: str, detail: bool = True, refresh: bool = False, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L472)

**参数：**

路径 (`str`) ：目录的路径。

详细信息（`bool`，*可选*）：如果为 True，则返回包含文件信息的字典列表。如果为 False，则返回文件路径列表。默认为 True。

刷新（`bool`，*可选*）：如果为True，则绕过缓存并获取最新数据。默认为 False。

revision (`str`, *可选*) ：要列出的 git 修订版本。**返回：** `list[Union[str, dict[str, Any]]]`

文件路径列表（如果detail=False）或文件信息列表
字典（如果详细信息= True）。

列出目录的内容。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.ls)。

> [!警告]
> 注意：如果可能，请使用 `HfApi.list_repo_tree()` 以获得更好的性能。

#### 修改[[huggingface_hub.HfFileSystem.modified]]

```python
modified(path: str, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L856)

**参数：**

路径 (`str`) ：文件的路径。

**返回：** `datetime`

文件的最后修改时间。

获取文件的最后修改时间。

更多详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.modified)。

#### 解析路径[[huggingface_hub.HfFileSystem.resolve_path]]

```python
resolve_path(path: str, revision: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L276)

**参数：**

path (`str`) ：解析路径。

revision (`str`, *可选*) ：要解析的存储库的修订版本。默认为路径中指定的修订版。

**返回：** `HfFileSystemResolvedPath`

解析的路径信息包含 `repo_type`、`repo_id`、`revision` 和 `path_in_repo`。

**加薪：** ``ValueError`` or ``NotImplementedError``

- ``ValueError`` -- 
  如果路径包含冲突的修订信息。
- ``NotImplementedError`` -- 
  如果尝试列出存储库。

将 Hugging Face 文件系统路径解析为其组件。

#### rm[[huggingface_hub.HfFileSystem.rm]]

```python
rm(path: str, recursive: bool = False, maxdepth: int | None = None, revision: str | None = None, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L422)

**参数：**

path (`str`) ：要删除的路径。recursive (`bool`, *可选*) ：如果为 True，则删除目录及其所有内容。默认为 False。

maxdepth (`int`, *可选*) : 递归删除时要访问的子目录的最大数量。

revision (`str`, *可选*) : 要从中删除的 git 修订版。

从存储库中删除文件。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.rm)。

> [!警告]
> 注意：如果可能，请使用 `HfApi.delete_file()` 以获得更好的性能。

#### url[[huggingface_hub.HfFileSystem.url]]

```python
url(path: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L1052)

**参数：**

path (`str`) : 获取 URL 的路径。

**退货：** `str`

用于访问 Hub 上的文件或目录的 HTTP URL。

获取给定路径的 HTTP URL。

#### 步行[[huggingface_hub.HfFileSystem.walk]]

```python
walk(path: str, *args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/hf_file_system.py#L701)

**参数：**

path (`str`) : 列出文件的根路径。

**退货：** `Iterator[tuple[str, list[str], list[str]]]`

（路径、目录名列表、文件名列表）元组的迭代器。

返回给定路径下的所有文件。

详情请参阅[fsspec documentation](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.walk)。

### 桶
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/buckets.md