<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 下载文件

## 下载单个文件

### hf_hub_download[[huggingface_hub.hf_hub_download]]

#### Huggingface_hub.hf_hub_download[[huggingface_hub.hf_hub_download]]

```python
huggingface_hub.hf_hub_download(repo_id: str, filename: str, subfolder: str | None = None, repo_type: str | None = None, revision: str | None = None, library_name: str | None = None, library_version: str | None = None, cache_dir: str | pathlib.Path | None = None, local_dir: str | pathlib.Path | None = None, user_agent: dict | str | None = None, force_download: bool = False, etag_timeout: float = 10, token: bool | str | None = None, local_files_only: bool = False, headers: dict[str, str] | None = None, endpoint: str | None = None, tqdm_class: type[tqdm.asyncio.tqdm_asyncio] | None = None, dry_run: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/file_download.py#L836)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

filename (`str`) ：存储库中文件的名称。

子文件夹（`str`，*可选*）：与模型存储库内的文件夹相对应的可选值。

repo_type (`str`, *可选*) ：如果从数据集、空间或内核存储库下载，则设置为 `"dataset"`、`"space"` 或 `"kernel"`；如果从模型下载，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：可选的 Git 修订 ID，可以是分支名称、标签或提交哈希。

library_name (`str`, *可选*) ：对象对应的库的名称。

library_version (`str`, *可选*) ：库的版本。

cache_dir (`str`, `Path`, *可选*) : 存储缓存文件的文件夹路径。

local_dir (`str` 或 `Path`, *可选*) : 如果提供，下载的文件将放置在此目录下。user_agent (`dict`, `str`, *可选*) ：字典或字符串形式的用户代理信息。

force_download（`bool`，*可选*，默认为`False`）：即使文件已存在于本地缓存中，是否也应该下载该文件。

etag_timeout (`float`, *可选*, 默认为`10`) : 获取ETag时，等待服务器发送数据多少秒后放弃，传递给`httpx.request`。

令牌（`str`、`bool`、*可选*）：用于下载的令牌。 - 如果是`True`，则从 HuggingFace 配置文件夹中读取令牌。 - 如果是字符串，则将其用作身份验证令牌。

local_files_only (`bool`，*可选*，默认为`False`)：如果`True`，避免下载文件，并返回本地缓存文件的路径（如果存在）。

headers (`dict`, *可选*) ：随请求一起发送的附加标头。

端点（`str`，*可选*）：将请求发送到的集线器端点。默认为 `HF_ENDPOINT` 的值。tqdm_class (`tqdm`, *可选*) ：如果提供，则覆盖进度条的默认行为。传递的参数必须继承自 `tqdm.auto.tqdm` 或至少模仿其行为。默认为自定义 HF 进度条，可以通过设置 `HF_HUB_DISABLE_PROGRESS_BARS` 环境变量来禁用。

dry_run（`bool`，*可选*，默认为`False`）：如果`True`，则执行试运行而不实际下载文件。返回一个 [DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 对象，其中包含有关将下载的内容的信息。

**返回：** `str` 或 [DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo)

- 如果`dry_run=False`：文件的本地路径，或者如果网络关闭，则文件的最新版本缓存在磁盘上。
- 如果`dry_run=True`：包含下载信息的[DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo)对象。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 `~utils.RemoteEntryNotFoundError` 或 [LocalEntryNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.LocalEntryNotFoundError) 或 ``EnvironmentError`` or ``OSError`` or ``ValueError``- [RepositoryNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到要下载的存储库。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果找不到要下载的修订版本。
- `~utils.RemoteEntryNotFoundError` -- 
  如果找不到要下载的文件。
- [LocalEntryNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.LocalEntryNotFoundError) -- 
  如果网络被禁用或不可用并且在缓存中找不到文件。
- [⟦T62⟧](https://docs.python.org/3/library/exceptions.html#EnvironmentError) -- 
  如果`token=True`但是找不到token。
- [⟦T64⟧](https://docs.python.org/3/library/exceptions.html#OSError) -- 
  如果无法确定 ETag。
- [⟦T65⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果某些参数值无效。

如果本地缓存中尚不存在给定文件，则下载该文件。

新的缓存文件布局如下所示：
- 缓存目录包含每个 repo_id 一个子文件夹（按存储库类型命名）
- 在每个 repo 文件夹内：
  - refs 是最新已知修订版的列表 => commit_hash 对
  - blobs 包含实际的文件 blob（由其 git-sha 或 sha256 标识，具体取决于
    无论它们是否是 LFS 文件）
  - 快照每次提交包含一个子文件夹，每个“提交”包含文件的子集
    已在该特定提交中解决。每个文件名都是指向 blob 的符号链接
    在那个特定的提交。

```
[  96]  .
└── [ 160]  models--julien-c--EsperBERTo-small
    ├── [ 160]  blobs
    │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
    │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
    ├── [  96]  refs
    │   └── [  40]  main
    └── [ 128]  snapshots
        ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
        │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
        │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
            ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
            └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
```如果提供了`local_dir`，则存储库中的文件结构将被复制到此位置。使用此功能时
选项，`cache_dir`将不会被使用，并且将在`local_dir`的根目录下创建一个`.cache/huggingface/`文件夹
存储与下载的文件相关的一些元数据。虽然这种机制不如主要机制那么强大
缓存系统，它针对定期拉取存储库的最新版本进行了优化。

### hf_hub_url[[huggingface_hub.hf_hub_url]]

#### Huggingface_hub.hf_hub_url[[huggingface_hub.hf_hub_url]]

```python
huggingface_hub.hf_hub_url(repo_id: str, filename: str, subfolder: str | None = None, repo_type: str | None = None, revision: str | None = None, endpoint: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/file_download.py#L201)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）名称和存储库名称，由 `/` 分隔。

filename (`str`) ：存储库中文件的名称。

子文件夹（`str`，*可选*）：与存储库内的文件夹相对应的可选值。

repo_type (`str`, *可选*) ：如果从数据集、空间或内核存储库下载，则设置为 `"dataset"`、`"space"` 或 `"kernel"`；如果从模型下载，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：可选的 Git 修订 ID，可以是分支名称、标签或提交哈希。

端点（`str`，*可选*）：将请求发送到的集线器端点。默认为 `HF_ENDPOINT` 的值。根据给定信息构造文件的 URL。

解析后的地址可以是 Huggingface.co 托管的 url，也可以是指向的链接
Cloudfront（内容分发网络或 CDN），适用于大型文件
超过几MB。

示例：

```python
>>> from huggingface_hub import hf_hub_url

>>> hf_hub_url(
...     repo_id="julien-c/EsperBERTo-small", filename="pytorch_model.bin"
... )
'https://huggingface.co/julien-c/EsperBERTo-small/resolve/main/pytorch_model.bin'
```

> [!提示]
> 注意事项：
>
> Cloudfront 在全球范围内进行复制，因此下载速度更快
> 最终用户（这也降低了我们的带宽成本）。
>
> Cloudfront 默认情况下会主动缓存文件（默认 TTL 为 24
> 小时），但这不是问题，因为我们实施了
> Huggingface.co 上基于 git 的版本控制系统，这意味着我们存储
> 以内容可寻址方式存储在 S3/Cloudfront 上的文件（即文件
> 名称是它的哈希值）。使用内容可寻址文件名意味着缓存不能
> 永远陈旧。
>
> 就该库的客户端缓存而言，我们基于我们的缓存
> 在对象的实体标签（`ETag`）上，这是一个标识符
> 资源的特定版本 [1]_.对象的 ETag 是：它的 git-sha1
> 如果存储在 git 中，或者如果存储在 git-lfs 中，则为 sha256。

参考文献：

- [1] https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/ETag## 下载存储库的快照[[huggingface_hub.snapshot_download]]

#### Huggingface_hub.snapshot_download[[huggingface_hub.snapshot_download]]

```python
huggingface_hub.snapshot_download(repo_id: str, repo_type: str | None = None, revision: str | None = None, cache_dir: str | pathlib.Path | None = None, local_dir: str | pathlib.Path | None = None, library_name: str | None = None, library_version: str | None = None, user_agent: dict | str | None = None, etag_timeout: float = 10, force_download: bool = False, token: bool | str | None = None, local_files_only: bool = False, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, max_workers: int = 8, tqdm_class: type[tqdm.asyncio.tqdm_asyncio] | None = None, headers: dict[str, str] | None = None, endpoint: str | None = None, dry_run: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_snapshot_download.py#L113)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

repo_type (`str`, *可选*) ：如果从数据集、空间或内核存储库下载，则设置为 `"dataset"`、`"space"` 或 `"kernel"`；如果从模型下载，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：可选的 Git 修订 ID，可以是分支名称、标签或提交哈希。

cache_dir (`str`, `Path`, *可选*) : 存储缓存文件的文件夹路径。

local_dir (`str` 或 `Path`, *可选*) : 如果提供，下载的文件将放置在此目录下。

library_name (`str`, *可选*) ：对象对应的库的名称。

library_version (`str`, *可选*) ：库的版本。

user_agent (`str`, `dict`, *可选*) ：字典或字符串形式的用户代理信息。etag_timeout (`float`, *可选*, 默认为`10`) : 获取ETag时，等待服务器发送数据多少秒后放弃，传递给`httpx.request`。

force_download (`bool`，*可选*，默认为`False`)：即使文件已存在于本地缓存中，是否也应该下载该文件。

令牌（`str`、`bool`、*可选*）：用于下载的令牌。 - 如果是`True`，则从 HuggingFace 配置文件夹中读取令牌。 - 如果是字符串，则将其用作身份验证令牌。

headers (`dict`, *可选*) ：请求中包含的附加标头。这些标头优先于其他标头。

端点（`str`，*可选*）：将请求发送到的集线器端点。默认为 `HF_ENDPOINT` 的值。

local_files_only（`bool`，*可选*，默认为`False`）：如果是`True`，则不下载任何文件，即使它们不在`cache_dir`或`local_dir`中。

allowed_pa​​tterns（`list[str]`或`str`，*可选*）：如果提供，则仅下载至少匹配一种模式的文件。

ignore_patterns（`list[str]` 或 `str`，*可选*）：如果提供，则不会下载与任何模式匹配的文件。max_workers (`int`, *可选*) ：下载文件的并发线程数（1个线程 = 1个文件下载）。默认为 8。

tqdm_class (`tqdm`, *可选*) ：如果提供，则覆盖进度条的默认行为。传递的参数必须继承自 `tqdm.auto.tqdm` 或至少模仿其行为。请注意，`tqdm_class` 不会传递给每个单独的下载。默认为自定义 HF 进度条，可以通过设置 `HF_HUB_DISABLE_PROGRESS_BARS` 环境变量来禁用。

dry_run（`bool`，*可选*，默认为`False`）：如果`True`，则执行试运行而不实际下载文件。返回 [DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 对象列表，其中包含有关将下载的内容的信息。

**返回：** `str` 或 [DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 列表

- 如果`dry_run=False`：本地快照路径。
- 如果`dry_run=True`：包含下载信息的[DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo)对象列表。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 [IncompleteSnapshotError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.IncompleteSnapshotError) 或 ``EnvironmentError`` or ``OSError`` or ``ValueError``- [RepositoryNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到要下载的存储库。这可能是因为它不存在
  或者因为它设置为 `private` 并且您无权访问。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果找不到要下载的修订版本。
- [IncompleteSnapshotError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.IncompleteSnapshotError) -- 
  如果无法访问集线器（离线、连接问题或`local_files_only=True`）并且
  缓存的快照缺少一些请求的文件。
- [⟦T141⟧](https://docs.python.org/3/library/exceptions.html#EnvironmentError) -- 
  如果`token=True`并且找不到令牌。
- [⟦T143⟧](https://docs.python.org/3/library/exceptions.html#OSError) -- 如果
  无法确定 ETag。
- [⟦T144⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果某些参数值无效。

下载存储库文件。

下载指定版本的存储库文件的完整快照。当您想要来自的所有文件时，这很有用
一个回购协议，因为你不知道你_先验_需要哪些。所有文件都嵌套在一个文件夹中以保留其
相对于该文件夹的路径和文件名。您还可以使用 `allow_patterns` 过滤要下载的文件
和`ignore_patterns`。如果提供了`local_dir`，则存储库中的文件结构将被复制到此位置。使用此功能时
选项，`cache_dir`将不会被使用，并且会在`local_dir`的根目录下创建一个`.cache/huggingface/`文件夹
存储与下载的文件相关的一些元数据。虽然这种机制不如主要机制那么强大
缓存系统，它针对定期拉取存储库的最新版本进行了优化。

另一种方法是克隆存储库，但这需要正确安装 git 和 git-lfs
配置。使用 git 克隆存储库时也不可能过滤要下载的文件。

## 读取缓存的repo树[[huggingface_hub.get_cached_repo_tree]]

#### Huggingface_hub.get_cached_repo_tree[[huggingface_hub.get_cached_repo_tree]]

```python
huggingface_hub.get_cached_repo_tree(repo_id: str, repo_type: str | None = None, revision: str | None = None, cache_dir: str | pathlib.Path | None = None, local_dir: str | pathlib.Path | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_snapshot_download.py#L584)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

repo_type (`str`, *可选*) ：如果从数据集、空间或内核存储库中列出，则设置为 `"dataset"`、`"space"` 或 `"kernel"`；如果从模型中列出，则设置为 `None` 或 `"model"`。默认为`None`。revision (`str`, *可选*) ：可选的 Git 修订 ID，可以是分支名称、标签或提交哈希。默认为默认分支。使用本地缓存（`refs/`）将分支/标签名称解析为提交哈希。

cache_dir (`str`, `Path`, *可选*) ：存储缓存文件的文件夹路径。默认为 `HF_HUB_CACHE` 的值。

local_dir（`str`或`Path`，*可选*）：如果提供，则读取由`local_dir`下载（来自`local_dir/.cache/huggingface/`）而不是主缓存缓存的树列表。分支/标签修订仍然使用主缓存（`cache_dir`）解析为提交哈希。

**退货：** `list[RepoFile]`

为此版本缓存的 [RepoFile](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.RepoFile) 对象列表。

**加薪：** [CachedRepoTreeNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.CachedRepoTreeNotFoundError)

- [CachedRepoTreeNotFoundError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.CachedRepoTreeNotFoundError) -- 
  如果没有缓存所请求修订版的树列表（例如，在此修订版中从未下载过存储库）。

返回给定修订版的存储库的缓存树列表，无需任何网络调用。

树列表是​​提交时存储库的一组文件（及其下载元数据）。人口稠密
在磁盘上作为[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)的副作用（请参阅`trees/<commit_hash>.json`缓存文件）并且是
用于在后续下载时跳过网络调用。该函数直接公开该缓存。如果您需要 Hub 上存储库的当前树列表，请改用 [list_repo_tree()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree)。

示例：
```py
>>> from huggingface_hub import get_cached_repo_tree
>>> files = get_cached_repo_tree("openai-community/gpt2")
>>> [f.path for f in files]
['.gitattributes', 'config.json', 'model.safetensors', ...]
```

## 解决修订

将分支/标签名称解析为提交哈希一次，然后传递结果以将每次下载固定到同一个提交。请参阅 [HfApi.resolve_revision()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.resolve_revision) 和 [cache-system guide](../guides/manage-cache#pin-a-revision-advanced)。

### ResolvedRevision[[huggingface_hub.ResolvedRevision]]

#### Huggingface_hub.ResolvedRevision[[huggingface_hub.ResolvedRevision]]

```python
huggingface_hub.ResolvedRevision(resolved: str, initial: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/_revision.py#L4)

**参数：**

初始（`str`或`None`）：用户最初请求的修订。如果`None`，则字符串值默认为`"main"`。

resolved (`str`) ：`initial`解析为的提交哈希。

已解析为提交哈希的 git 修订版。

`ResolvedRevision` 是 `str` 子类，因此它可以传递给任何采用 `revision` 的 `huggingface_hub` 方法
论点。它的字符串值是用户最初请求的修订（例如`"main"`，`"refs/pr/4"`），
它保持 URL 和错误消息可读，而 `.resolved` 保存它指向的提交哈希。

实例由[HfApi.resolve_revision()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.resolve_revision)构建，它还缓存`revision` -> `commit hash`映射
在本地缓存（`refs/`文件夹）中。

示例：
```python
>>> from huggingface_hub import resolve_revision
>>> revision = resolve_revision("openai-community/gpt2")
>>> revision
ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')
>>> revision == "main"  # it's a string
True
>>> revision.resolved
'607a30d783dfa663caf39e06633721c8d4cfcd7e'
```

## 获取文件的元数据

### get_hf_file_metadata[[huggingface_hub.get_hf_file_metadata]]#### Huggingface_hub.get_hf_file_metadata[[huggingface_hub.get_hf_file_metadata]]

```python
huggingface_hub.get_hf_file_metadata(url: str, token: bool | str | None = None, timeout: float | None = 10, library_name: str | None = None, library_version: str | None = None, user_agent: dict | str | None = None, headers: dict[str, str] | None = None, endpoint: str | None = None, retry_on_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/file_download.py#L1576)

**参数：**

url (`str`) ：文件url，例如由[hf_hub_url()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_url)返回。

令牌（`str` 或 `bool`，*可选*）：用于下载的令牌。 - 如果是`True`，则从 HuggingFace 配置文件夹中读取令牌。 - 如果`False`或`None`，则不提供令牌。 - 如果是字符串，则将其用作身份验证令牌。

timeout (`float`，*可选*，默认为10) ：放弃之前等待服务器发送元数据的秒数。

library_name (`str`, *可选*) ：对象对应的库的名称。

library_version (`str`, *可选*) ：库的版本。

user_agent (`dict`, `str`, *可选*) ：字典或字符串形式的用户代理信息。

headers (`dict`, *可选*) ：随请求一起发送的附加标头。

端点（`str`，*可选*）：集线器的端点。默认为 .

retry_on_errors (`bool`，*可选*，默认为`False`)：是否重试错误（429、5xx、超时、网络错误）。如果为 False，则不重试快速回退到本地缓存。

**退货：**包含元数据的[HfFileMetadata](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.HfFileMetadata)对象，例如位置、etag、大小和
commit_hash。

获取 Hub 上给定 url 版本控制的文件的元数据。

### HfFileMetadata[[huggingface_hub.HfFileMetadata]]

#### Huggingface_hub.HfFileMetadata[[huggingface_hub.HfFileMetadata]]

```python
huggingface_hub.HfFileMetadata(commit_hash: str | None, etag: str | None, location: str, size: int | None, xet_file_data: huggingface_hub.utils._xet.XetFileData | None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/file_download.py#L147)

**参数：**

commit_hash (`str`, *可选*) : 与文件相关的commit_hash。

etag (`str`, *可选*) ：服务器上文件的 Etag。

location (`str`) ：下载文件的位置。可以是 Hub url，也可以不是 (CDN)。

size (`size`) ：文件的大小。如果是 LFS 文件，则包含实际 LFS 文件的大小，而不是指针。

xet_file_data (`XetFileData`, *可选*) : 文件的 Xet 信息。仅当使用 Xet 存储存储文件时才设置此项。

包含有关集线器上的文件版本信息的数据结构。

由 [get_hf_file_metadata()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.get_hf_file_metadata) 根据 URL 返回。

## 缓存

上面显示的方法旨在与缓存系统一起使用，以防止
重新下载文件。缓存系统在v0.8.0更新，成为中央
缓存系统在依赖于集线器的库之间共享。

请阅读[cache-system guide](../guides/manage-cache)，了解 HF 缓存的详细介绍。### 推理端点
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/inference_endpoints.md