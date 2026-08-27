<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 缓存系统参考

缓存系统在v0.8.0中更新为中央缓存系统共享
跨依赖于 Hub 的库。阅读[cache-system guide](../guides/manage-cache)
有关 HF 缓存的详细介绍。

## 助手

### try_to_load_from_cache[[huggingface_hub.try_to_load_from_cache]]

#### Huggingface_hub.try_to_load_from_cache[[huggingface_hub.try_to_load_from_cache]]

```python
huggingface_hub.try_to_load_from_cache(repo_id: str, filename: str, cache_dir: str | pathlib.Path | None = None, revision: str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/file_download.py#L1482)

**参数：**

cache_dir (`str` 或 `os.PathLike`) ：缓存文件所在的文件夹。

repo_id (`str`) ：huggingface.co 上的存储库的 ID。

filename (`str`) ：要在`repo_id`内查找的文件名。

修订版（`str`，*可选*）：要使用的特定模型版本。如果未提供且未提供 `commit_hash`，则默认为 `"main"`。

repo_type (`str`, *可选*) ：存储库的类型。默认为`"model"`。

**返回：** `Optional[str]` 或 `_CACHED_NO_EXIST`

如果文件未缓存，将返回`None`。否则：
- 缓存文件的确切路径（如果在缓存中找到）
- 如果文件在给定的提交哈希中不存在，则一个特殊值`_CACHED_NO_EXIST`，并且这一事实是
  缓存。

探索缓存以返回给定修订版的最新缓存文件（如果找到）。如果文件未缓存，此函数不会引发任何异常。

示例：

```python
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST

filepath = try_to_load_from_cache()
if isinstance(filepath, str):
    # file exists and is cached
    ...
elif filepath is _CACHED_NO_EXIST:
    # non-existence of file is cached
    ...
else:
    # file is not cached
    ...
```

### cached_assets_path[[huggingface_hub.cached_assets_path]]

#### Huggingface_hub.cached_assets_path[[huggingface_hub.cached_assets_path]]

```python
huggingface_hub.cached_assets_path(library_name: str, namespace: str = 'default', subfolder: str = 'default', assets_dir: str | pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_assets.py#L19)

**参数：**

library_name (`str`) ：将管理缓存文件夹的库的名称。示例：`"dataset"`。

命名空间（`str`，*可选*，默认为“default”）：数据所属的命名空间。示例：`"SQuAD"`。

子文件夹（`str`，*可选*，默认为“默认”）：将存储数据的子文件夹。示例：`extracted`。

asset_dir (`str`, `Path`, *可选*) ：缓存资源的文件夹路径。这不能与缓存 Hub 文件的文件夹相同。如果未提供，则默认为 `HF_HOME / "assets"`。也可以通过`HF_ASSETS_CACHE`环境变量进行设置。

**退货：**

缓存文件夹的路径 (`Path`)。

返回缓存任意文件的文件夹路径。

`huggingface_hub` 提供了一个规范的文件夹路径来存储资源。这是
建议将缓存集成到下游库中的方法，因为它将受益于
内置工具可以正确扫描和删除缓存。从 Hub 缓存的文件和资产之间存在区别。文件来自
Hub 以 git 感知方式缓存，并完全由 `huggingface_hub` 管理。参见
[related documentation](https://huggingface.co/docs/huggingface_hub/how-to-cache)。
下游库缓存的所有其他文件都被视为“资产”
（从外部源下载的文件，从 .tar 存档中提取，经过预处理
用于培训，...）。

一旦生成文件夹路径，就保证它存在并且是一个目录。
该路径基于 3 个深度级别：库名称、命名空间和
子文件夹。这 3 个级别提供了灵活性，同时允许 `huggingface_hub`
扫描/删除部分资源缓存时需要文件夹。在图书馆内，
预计所有名称空间共享相同的子文件夹名称子集，但这
不是强制性规则。下游库可以完全控制哪个文件
在其缓存中采用的结构。命名空间和子文件夹是可选的（将
默认为 `"default/"` 子文件夹），但库名称是强制性的，因为我们想要每个
下游库来管理自己的缓存。

预期树：
```text
    assets/
    └── datasets/
    │   ├── SQuAD/
    │   │   ├── downloaded/
    │   │   ├── extracted/
    │   │   └── processed/
    │   ├── Helsinki-NLP--tatoeba_mt/
    │       ├── downloaded/
    │       ├── extracted/
    │       └── processed/
    └── transformers/
        ├── default/
        │   ├── something/
        ├── bert-base-cased/
        │   ├── default/
        │   └── training/
    hub/
    └── models--julien-c--EsperBERTo-small/
        ├── blobs/
        │   ├── (...)
        │   ├── (...)
        ├── refs/
        │   └── (...)
        └── [ 128]  snapshots/
            ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
            │   ├── (...)
            └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
                └── (...)
```

示例：
```py
>>> from huggingface_hub import cached_assets_path

>>> cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
PosixPath('/home/wauplin/.cache/huggingface/extra/datasets/SQuAD/download')

>>> cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="extracted")
PosixPath('/home/wauplin/.cache/huggingface/extra/datasets/SQuAD/extracted')

>>> cached_assets_path(library_name="datasets", namespace="Helsinki-NLP/tatoeba_mt")
PosixPath('/home/wauplin/.cache/huggingface/extra/datasets/Helsinki-NLP--tatoeba_mt/default')

>>> cached_assets_path(library_name="datasets", assets_dir="/tmp/tmp123456")
PosixPath('/tmp/tmp123456/datasets/default/default')
```

### scan_cache_dir[[huggingface_hub.scan_cache_dir]]

#### Huggingface_hub.scan_cache_dir[[huggingface_hub.scan_cache_dir]]

```python
huggingface_hub.scan_cache_dir(cache_dir: str | pathlib.Path | None = None)
```[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L597)

**参数：**

cache_dir (`str` 或 `Path`, `optional`) ：要缓存的缓存目录。默认为默认 HF 缓存目录。

扫描整个 HF 缓存系统并返回 [~HFCacheInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo) 结构。

使用 `scan_cache_dir` 以编程方式扫描您的缓存系统。缓存
将由仓库扫描仓库。如果存储库损坏，则会出现 [~CorruptedCacheException](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CorruptedCacheException)
将在内部抛出，但在 [~HFCacheInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo) 中捕获并返回
结构。只有有效的回购协议才能获得正确的报告。

```py
>>> from huggingface_hub import scan_cache_dir

>>> hf_cache_info = scan_cache_dir()
HFCacheInfo(
    size_on_disk=3398085269,
    repos=frozenset({
        CachedRepoInfo(
            repo_id='t5-small',
            repo_type='model',
            repo_path=PosixPath(...),
            size_on_disk=970726914,
            nb_files=11,
            revisions=frozenset({
                CachedRevisionInfo(
                    commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                    size_on_disk=970726339,
                    snapshot_path=PosixPath(...),
                    files=frozenset({
                        CachedFileInfo(
                            file_name='config.json',
                            size_on_disk=1197
                            file_path=PosixPath(...),
                            blob_path=PosixPath(...),
                        ),
                        CachedFileInfo(...),
                        ...
                    }),
                ),
                CachedRevisionInfo(...),
                ...
            }),
        ),
        CachedRepoInfo(...),
        ...
    }),
    warnings=[
        CorruptedCacheException("Snapshots dir doesn't exist in cached repo: ..."),
        CorruptedCacheException(...),
        ...
    ],
)
```

您还可以使用以下命令直接从 `hf` 命令行打印详细报告：
```text
> hf cache ls
ID                          SIZE     LAST_ACCESSED LAST_MODIFIED REFS
--------------------------- -------- ------------- ------------- -----------
dataset/nyu-mll/glue          157.4M 2 days ago    2 days ago    main script
model/LiquidAI/LFM2-VL-1.6B     3.2G 4 days ago    4 days ago    main
model/microsoft/UserLM-8b      32.1G 4 days ago    4 days ago    main

Done in 0.0s. Scanned 6 repo(s) for a total of 3.4G.
Got 1 warning(s) while scanning. Use -vvv to print details.
```

> [!警告]
> 加薪：
>
> `CacheNotFound`
> 如果缓存目录不存在。
>
> [⟦T60⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果缓存目录是文件，而不是目录。

返回：一个[~HFCacheInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo)对象。

## 数据结构

所有结构都是由[scan_cache_dir()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.scan_cache_dir)构建和返回的，并且是不可变的。

### HFCacheInfo[[huggingface_hub.HFCacheInfo]]

#### Huggingface_hub.HFCacheInfo[[huggingface_hub.HFCacheInfo]]

```python
huggingface_hub.HFCacheInfo(size_on_disk: int, repos: frozenset, incomplete_files: frozenset, warnings: list)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L349)

**参数：**

size_on_disk (`int`) ：缓存系统中所有有效存储库大小的总和。

存储库 (`frozenset[CachedRepoInfo]`) ：一组[~CachedRepoInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedRepoInfo)，描述扫描时在缓存系统上找到的所有有效缓存存储库。incomplete_files (`frozenset[CachedIncompleteFileInfo]`) ：一组 [~CachedIncompleteFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedIncompleteFileInfo) 描述因下载中断而留下的孤立 `*.incomplete` 文件。

warnings (`list[CorruptedCacheException]`) ：扫描缓存时发生的[~CorruptedCacheException](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CorruptedCacheException)列表。捕获这些异常以便扫描可以继续。扫描中会跳过损坏的存储库。

冻结的数据结构保存有关整个缓存系统的信息。

该数据结构由[scan_cache_dir()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.scan_cache_dir)返回并且是不可变的。

> [!警告]
> 这里 `size_on_disk` 等于所有存储库大小（仅限 blob）的总和。然而如果
> 一些缓存的存储库已损坏，它们的大小不被考虑在内。

#### delete_revisions[[huggingface_hub.HFCacheInfo.delete_revisions]]

```python
delete_revisions(*revisions: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L393)

准备策略以删除本地缓存的一个或多个修订。

输入修订可以是任何修订哈希。如果在
本地缓存，会引发警告但不会引发错误。修订可以来自
不同的缓存存储库，因为哈希值在存储库中是唯一的，

示例：
```py
>>> from huggingface_hub import scan_cache_dir
>>> cache_info = scan_cache_dir()
>>> delete_strategy = cache_info.delete_revisions(
...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
... )
>>> print(f"Will free {delete_strategy.expected_freed_size_str}.")
Will free 7.9K.
>>> delete_strategy.execute()
Cache deletion done. Saved 7.9K.
```

```py
>>> from huggingface_hub import scan_cache_dir
>>> scan_cache_dir().delete_revisions(
...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa",
...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
...     "6c0e6080953db56375760c0471a8c5f2929baf11",
... ).execute()
Cache deletion done. Saved 8.6G.
```

> [!警告]
> `delete_revisions` 返回一个 [DeleteCacheStrategy](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.DeleteCacheStrategy) 对象，需要
> 被处决。 [DeleteCacheStrategy](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.DeleteCacheStrategy) 并不是要修改，而是
> 允许在实际执行删除之前进行试运行。#### export_as_table[[huggingface_hub.HFCacheInfo.export_as_table]]

```python
export_as_table(verbosity: int = 0)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L502)

**参数：**

详细程度（`int`，*可选*）：详细程度。默认为 0。

**返回：** `str`

表作为字符串。

从 [HFCacheInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo) 对象生成一个表。

通过 `verbosity=0` 获取每个存储库包含一行和列的表
“repo_id”、“repo_type”、“磁盘大小”、“nb_files”、“last_accessed”、“last_modified”、“refs”、“local_path”。

通过 `verbosity=1` 获取一个表，每个存储库和修订包含一行（因此单个存储库可以出现多行），其中包含列
“repo_id”、“repo_type”、“修订版”、“磁盘大小”、“nb_files”、“last_modified”、“refs”、“local_path”。

示例：
```py
>>> from huggingface_hub.utils import scan_cache_dir

>>> hf_cache_info = scan_cache_dir()
HFCacheInfo(...)

>>> print(hf_cache_info.export_as_table())
REPO ID                                             REPO TYPE SIZE ON DISK NB FILES LAST_ACCESSED LAST_MODIFIED REFS LOCAL PATH
--------------------------------------------------- --------- ------------ -------- ------------- ------------- ---- --------------------------------------------------------------------------------------------------
roberta-base                                        model             2.7M        5 1 day ago     1 week ago    main ~/.cache/huggingface/hub/models--roberta-base
suno/bark                                           model             8.8K        1 1 week ago    1 week ago    main ~/.cache/huggingface/hub/models--suno--bark
t5-base                                             model           893.8M        4 4 days ago    7 months ago  main ~/.cache/huggingface/hub/models--t5-base
t5-large                                            model             3.0G        4 5 weeks ago   5 months ago  main ~/.cache/huggingface/hub/models--t5-large

>>> print(hf_cache_info.export_as_table(verbosity=1))
REPO ID                                             REPO TYPE REVISION                                 SIZE ON DISK NB FILES LAST_MODIFIED REFS LOCAL PATH
--------------------------------------------------- --------- ---------------------------------------- ------------ -------- ------------- ---- -----------------------------------------------------------------------------------------------------------------------------------------------------
roberta-base                                        model     e2da8e2f811d1448a5b465c236feacd80ffbac7b         2.7M        5 1 week ago    main ~/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b
suno/bark                                           model     70a8a7d34168586dc5d028fa9666aceade177992         8.8K        1 1 week ago    main ~/.cache/huggingface/hub/models--suno--bark/snapshots/70a8a7d34168586dc5d028fa9666aceade177992
t5-base                                             model     a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1       893.8M        4 7 months ago  main ~/.cache/huggingface/hub/models--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1
t5-large                                            model     150ebc2c4b72291e770f58e6057481c8d2ed331a         3.0G        4 5 months ago  main ~/.cache/huggingface/hub/models--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a
```

### CachedRepoInfo[[huggingface_hub.CachedRepoInfo]]

#### Huggingface_hub.CachedRepoInfo[[huggingface_hub.CachedRepoInfo]]

```python
huggingface_hub.CachedRepoInfo(repo_id: str, repo_type: typing.Literal['model', 'dataset', 'space'], repo_path: Path, size_on_disk: int, nb_files: int, revisions: frozenset, last_accessed: float, last_modified: float)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L175)

**参数：**

repo_id (`str`) ：集线器上存储库的存储库 ID。示例：`"google/fleurs"`。

repo_type (`Literal["dataset", "model", "space"]`) ：缓存存储库的类型。

repo_path (`Path`) ：缓存存储库的本地路径。

size_on_disk (`int`) ：缓存存储库中 blob 文件大小的总和。

nb_files (`int`) ：缓存存储库中的 blob 文件总数。revisions (`frozenset[CachedRevisionInfo]`) ：一组[~CachedRevisionInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedRevisionInfo)，描述存储库中缓存的所有修订。

last_accessed (`float`) ：上次访问存储库的 blob 文件的时间戳。

last_modified (`float`) ：上次修改/创建存储库的 blob 文件的时间戳。

冻结的数据结构保存有关缓存存储库的信息。

> [!警告]
> `size_on_disk` 不一定是所有修订大小的总和，因为
> 重复的文件。此外，只考虑斑点，而不考虑（可以忽略不计的）
> 文件夹和符号链接的大小。

> [!警告]
> `last_accessed` 和 `last_modified` 可靠性可能取决于您所使用的操作系统。
> 参见[python documentation](https://docs.python.org/3/library/os.html#os.stat_result)
> 了解更多详情。

#### size_on_disk_str[[huggingface_hub.CachedRepoInfo.size_on_disk_str]]

```python
size_on_disk_str()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L237)

（属性）作为人类可读字符串的 blob 文件大小总和。

示例：“42.2K”。

#### 参考文献[[huggingface_hub.CachedRepoInfo.refs]]

```python
refs()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L251)

（属性）`refs`和修订数据结构之间的映射。

### CachedRevisionInfo[[huggingface_hub.CachedRevisionInfo]]

#### Huggingface_hub.CachedRevisionInfo[[huggingface_hub.CachedRevisionInfo]]

```python
huggingface_hub.CachedRevisionInfo(commit_hash: str, snapshot_path: Path, size_on_disk: int, files: frozenset, refs: frozenset, last_modified: float)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L104)

**参数：**commit_hash (`str`) ：修订版的哈希值（唯一）。示例：`"9338f7b671827df886678df2bdd7cc7b4f36dffd"`。

snapshot_path (`Path`) ：`snapshots`文件夹中修订目录的路径。它包含与 Hub 上的存储库完全相同的树结构。

files : (`frozenset[CachedFileInfo]`)：一组[~CachedFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedFileInfo)，描述快照中包含的所有文件。

refs (`frozenset[str]`) ：指向此修订版的`refs`集。如果修订版没有`refs`，则视为分离。示例：`{"main", "2.4.0"}` 或 `{"refs/pr/1"}`。

size_on_disk (`int`) ：修订版符号链接的 blob 文件大小的总和。

last_modified (`float`) ：上次创建/修改修订版的时间戳。

冻结数据结构保存有关修订的信息。

修订版本对应于 `snapshots` 文件夹中的一个文件夹，并填充有
与 Hub 上的存储库完全相同的树结构，但仅包含符号链接。一个
修订版可以被 1 个或多个 `refs` 引用，也可以“分离”（无引用）。

> [!警告]
> `last_accessed` 无法在单个版本上正确确定为 blob 文件
> 在各个修订版之间共享。> [!警告]
> `size_on_disk` 不一定是所有文件大小的总和，因为可能
> 重复的文件。此外，只考虑斑点，而不考虑（可以忽略不计的）
> 文件夹和符号链接的大小。

#### size_on_disk_str[[huggingface_hub.CachedRevisionInfo.size_on_disk_str]]

```python
size_on_disk_str()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L157)

（属性）作为人类可读字符串的 blob 文件大小总和。

示例：“42.2K”。

#### nb_files[[huggingface_hub.CachedRevisionInfo.nb_files]]

```python
nb_files()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L166)

（属性）修订版中的文件总数。

### CachedFileInfo[[huggingface_hub.CachedFileInfo]]

#### Huggingface_hub.CachedFileInfo[[huggingface_hub.CachedFileInfo]]

```python
huggingface_hub.CachedFileInfo(file_name: str, file_path: Path, blob_path: Path, size_on_disk: int, blob_last_accessed: float, blob_last_modified: float)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L40)

**参数：**

file_name (`str`) ：文件的名称。示例：`config.json`。

file_path (`Path`) ：`snapshots`目录中的文件路径。文件路径是一个符号链接，引用 `blobs` 文件夹中的 blob。

blob_path (`Path`) ：blob 文件的路径。这相当于`file_path.resolve()`。

size_on_disk (`int`) ：blob 文件的大小（以字节为单位）。

blob_last_accessed (`float`) ：上次访问 blob 文件的时间戳（来自任何修订版）。blob_last_modified (`float`) ：上次修改/创建 blob 文件的时间戳。

冻结的数据结构保存有关单个缓存文件的信息。

> [!警告]
> `blob_last_accessed` 和 `blob_last_modified` 可靠性可能取决于您使用的操作系统
> 正在使用。参见[python documentation](https://docs.python.org/3/library/os.html#os.stat_result)
> 了解更多详情。

#### size_on_disk_str[[huggingface_hub.CachedFileInfo.size_on_disk_str]]

```python
size_on_disk_str()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L93)

（属性）作为人类可读字符串的 blob 文件的大小。

示例：“42.2K”。

### CachedIncompleteFileInfo[[huggingface_hub.CachedIncompleteFileInfo]]

#### Huggingface_hub.CachedIncompleteFileInfo[[huggingface_hub.CachedIncompleteFileInfo]]

```python
huggingface_hub.CachedIncompleteFileInfo(file_path: Path, size_on_disk: int)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L330)

**参数：**

file_path (`Path`) ：`blobs`文件夹中`.incomplete`文件的路径。

size_on_disk (`int`) ：部分下载的文件的大小（以字节为单位）。

冻结的数据结构保存有关单个不完整下载的信息。

下载中断会留下 `<cache>/<repo>/blobs/<etag>.incomplete` 文件。
这些不是任何已提交修订的一部分，因此它们由
[scan_cache_dir()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.scan_cache_dir)。

### DeleteCacheStrategy[[huggingface_hub.DeleteCacheStrategy]]

#### Huggingface_hub.DeleteCacheStrategy[[huggingface_hub.DeleteCacheStrategy]]

```python
huggingface_hub.DeleteCacheStrategy(expected_freed_size: int, blobs: frozenset, refs: frozenset, repos: frozenset, snapshots: frozenset)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L260)

**参数：**Expected_freed_size (`float`) ：执行策略后预期释放的大小。

blob (`frozenset[Path]`) ：要删除的 blob 文件路径集。

refs (`frozenset[Path]`) ：要删除的参考文件路径集。

repos (`frozenset[Path]`) ：要删除的整个仓库路径集。

snapshots (`frozenset[Path]`) ：要删除的快照集（符号链接目录）。

冻结数据结构保存删除缓存修订的策略。

该对象并不意味着以编程方式实例化，而是由
[delete_revisions()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo.delete_revisions)。有关使用示例，请参阅文档。

#### Expected_freed_size_str[[huggingface_hub.DeleteCacheStrategy.expected_freed_size_str]]

```python
expected_freed_size_str()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/utils/_cache_manager.py#L285)

（属性）将作为人类可读字符串释放的预期大小。

示例：“42.2K”。

## 例外情况

### CorruptedCacheException[[huggingface_hub.CorruptedCacheException]]

#### Huggingface_hub.CorruptedCacheException[[huggingface_hub.CorruptedCacheException]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/errors.py#L22)

Huggingface 缓存系统中任何意外结构的例外情况。

### 环境变量
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/environment_variables.md