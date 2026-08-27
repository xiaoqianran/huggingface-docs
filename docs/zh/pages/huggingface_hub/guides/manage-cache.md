<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 了解缓存

`huggingface_hub`利用本地磁盘作为两个缓存，避免再次重新下载项目。第一个缓存是基于文件的缓存，它缓存从集线器下载的各个文件，并确保在存储库更新时不会再次下载相同的文件。第二个缓存是块缓存，其中每个块代表文件中的一个字节范围，并确保跨文件共享的块仅下载一次。

> [!提示]
> 本指南涵盖了`huggingface_hub`提供的特定于Python的缓存管理工具。有关 Hugging Face Hub 缓存系统如何工作的与语言无关的概述，请参阅 [Hub documentation on local caching](https://huggingface.co/docs/hub/local-cache)。

## 基于文件的缓存

Hugging Face Hub 缓存系统旨在成为跨图书馆共享的中央缓存
这取决于集线器。 v0.8.0 已更新，以防止重新下载相同的文件
修订之间。

缓存系统设计如下：

```
<CACHE_DIR>
├─ <MODELS>
├─ <DATASETS>
├─ <SPACES>
```

默认 `<CACHE_DIR>` 为 `~/.cache/huggingface/hub`。但是，可以使用所有方法上的 `cache_dir` 参数，或者通过指定 `HF_HOME` 或 `HF_HUB_CACHE` 环境变量来自定义它。模型、数据集和空间有一个共同的根。这些存储库中的每一个都包含
存储库类型、名称空间（组织或用户名）（如果存在）以及
存储库名称：

```
<CACHE_DIR>
├─ models--julien-c--EsperBERTo-small
├─ models--lysandrejik--arxiv-nlp
├─ models--bert-base-cased
├─ datasets--glue
├─ datasets--huggingface--DataMeasurementsFiles
├─ spaces--dalle-mini--dalle-mini
```

现在将从集线器下载所有文件在这些文件夹中。缓存可确保
如果文件已存在且未更新，则不会下载两次；但如果更新的话
如果您要求最新的文件，那么它将下载最新的文件（同时保留
之前的文件完好无损，以防您再次需要）。

为了实现这一点，所有文件夹都包含相同的骨架：

```
<CACHE_DIR>
├─ datasets--glue
│  ├─ refs
│  ├─ blobs
│  ├─ snapshots
│  ├─ trees
...
```

每个文件夹均设计为包含以下内容：

### 参考文献

`refs` 文件夹包含指示给定参考的最新版本的文件。例如，
如果我们之前从存储库的 `main` 分支获取了文件，则 `refs`
文件夹将包含一个名为 `main` 的文件，该文件本身包含当前头的提交标识符。

如果 `main` 的最新提交以 `aaaaaa` 作为标识符，那么它将包含 `aaaaaa`。如果同一个分支被更新为一个新的提交，并且以 `bbbbbb` 作为标识符，那么
从该参考重新下载文件将更新 `refs/main` 文件以包含 `bbbbbb`。

### 斑点

`blobs`文件夹包含我们下载的实际文件。每个文件的名称就是它们的哈希值。

### 快照

`snapshots` 文件夹包含指向上述 blob 的符号链接。它本身由几个文件夹组成：
每个已知修订版一个！

在上面的解释中，我们最初从 `aaaaaa` 版本中获取文件，然后再从
`bbbbbb` 修订版。在这种情况下，`snapshots`文件夹中现在有两个文件夹：`aaaaaa`
和`bbbbbb`。

在每个文件夹中，都有包含我们下载的文件名称的实时符号链接。例如，
如果我们下载了修订版 `aaaaaa` 的 `README.md` 文件，我们将拥有以下路径：

```
<CACHE_DIR>/<REPO_NAME>/snapshots/aaaaaa/README.md
```

该 `README.md` 文件实际上是链接到具有文件哈希值的 blob 的符号链接。

通过以这种方式创建骨架，我们打开了文件共享的机制：如果在
修订版`bbbbbb`，它将具有相同的哈希值，并且不需要重新下载文件。

### 树`trees` 文件夹缓存存储库在给定提交时包含的文件列表。提交是不可变的，因此它的文件列表永远不会改变。这意味着该列表可以永久缓存，无需再次针对集线器进行检查。

每个缓存列表均以提交哈希命名并存储为 JSON 文件，例如 `trees/aaaaaa.json`。对于该提交时存储库中的每个文件，它都会记录下载该文件所需的内容：其路径、大小和哈希值。这与集线器返回的信息相同，但通常每个文件需要一次网络调用才能获取它。

该缓存是由[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)写入的。第一次下载提交时，文件列表将被提取一次并保存在此处。下次下载相同的提交时，将从磁盘读取该列表，而不是再次获取。因此，当所有内容都已缓存时重新运行下载会花费一次网络调用：将分支或标记名称解析为提交哈希所需的网络调用。[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)和[hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)都会读取此缓存以避免网络调用。当您下载带有提交哈希作为修订版的文件时（这正是 [snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 在内部对每个文件执行的操作），将从缓存文件列表中读取下载元数据，并跳过每个文件的网络调用。这意味着单个文件的 [hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 也受益于早期 [snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 为同一提交保存的文件列表。

因为缓存文件列表准确地描述了提交应该包含的内容，所以[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)还可以判断本地快照是否完整。如果无法访问集线器（您处于离线状态、连接失败或通过了 `local_files_only=True`）并且本地快照中缺少一些预期文件，[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 会引发 [IncompleteSnapshotError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.IncompleteSnapshotError) 而不是返回部分文件夹。在此之前，会以静默方式返回不完整的快照，这可能会让您在不知情的情况下处理丢失的文件。 `allow_patterns` 或 `ignore_patterns` 排除的文件不计为丢失。该异常通过其 `snapshot_path` 属性公开不完整快照的路径，因此您仍然可以在需要时找到部分缓存的文件。

### .no_exist（高级）除了 `blobs`、`refs` 和 `snapshots` 文件夹之外，您还可能会找到 `.no_exist` 文件夹
在你的缓存中。此文件夹跟踪您曾经尝试下载但不存在的文件
在集线器上。其结构与 `snapshots` 文件夹相同，每个已知版本有 1 个子文件夹：

```
<CACHE_DIR>/<REPO_NAME>/.no_exist/aaaaaa/config_that_does_not_exist.json
```

与 `snapshots` 文件夹不同，文件是简单的空文件（没有符号链接）。在这个例子中，
集线器上不存在修订版 `"aaaaaa"` 的文件 `"config_that_does_not_exist.json"`。
由于它只存储空文件，因此该文件夹在磁盘使用方面可以忽略不计。

现在您可能想知道，为什么这些信息如此相关？
在某些情况下，框架会尝试加载模型的可选文件。拯救不存在
可选文件的数量使得加载模型的速度更快，因为它为每个可能的可选文件节省了 1 个 HTTP 调用。
例如，在 `transformers` 中，每个标记生成器都可以支持其他文件。
第一次在计算机上加载分词器时，它将缓存存在哪些可选文件（以及
事实并非如此）以使下一次初始化的加载时间更快。要测试文件是否在本地缓存（不发出任何 HTTP 请求），您可以使用 [try_to_load_from_cache()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.try_to_load_from_cache)
帮手。它将返回文件路径（如果存在并缓存）、对象`_CACHED_NO_EXIST`（如果不存在）
被缓存）或`None`（如果我们不知道）。

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

### 实践中

实际上，您的缓存应类似于以下树：

```text
    [  96]  .
    └── [ 160]  models--julien-c--EsperBERTo-small
        ├── [ 160]  blobs
        │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
        ├── [  96]  refs
        │   └── [  40]  main
        ├── [ 128]  snapshots
        │   ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
        │   │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
        │   │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        │   └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
        │       ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
        │       └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        └── [  96]  trees
            ├── [ 521]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f.json
            └── [ 521]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48.json
```

### CacheDIR.TAG

`huggingface_hub` 自动创建一个
缓存目录中的[⟦T74⟧](https://bford.info/cachedir/)文件。这个
标签遵循*缓存目录标记标准*并告诉备份工具（例如Borg，
Restic、rsync）该目录包含可重新下载的缓存数据，并且可以安全地
从备份中排除。

### 限制为了拥有高效的缓存系统，`huggingface-hub` 使用符号链接。然而，
并非所有机器都支持符号链接。这是一个已知的限制，尤其是
窗户。在这种情况下，`huggingface_hub`不使用`blobs/`目录，而是
而是直接将文件存储在`snapshots/`目录中。此解决方法允许
用户从 Hub 下载和缓存文件的方式完全相同。检查工具
还支持删除缓存（见下文）。然而，缓存系统较少
高效，因为如果多个版本的单个文件可能会被下载多次
下载相同的存储库。

如果您想从 Windows 计算机上基于符号链接的缓存系统中受益，您可以
要么需要 [activate Developer Mode](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development)
或以管理员身份运行Python。

如果您想主动使用无符号链接缓存模式（例如，在不处理符号链接的共享文件系统上）
好吧），您可以将 [⟦T79⟧](../package_reference/environment_variables#hfhubdisablesymlinks) 环境变量设置为 `1`。文件将被复制到`snapshots/`
直接而不是符号链接到`blobs/`。当不支持符号链接时，会向用户显示警告消息以提醒
他们使用的是缓存系统的降级版本。可以禁用此警告
通过将 `HF_HUB_DISABLE_SYMLINKS_WARNING` 环境变量设置为 true。

## 固定修订版（高级）

> [!提示]
> 如果您将 Hub 集成到 ML 库中，单个 [snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 调用仍然是推荐的方法：它会解析一次修订版本，并行下载所有内容并缓存文件列表。下面的内容仅适用于单独下载和加载许多组件（配置、权重、分词器、处理器、适配器...）并且不能使用单个调用的复杂库。

当库逐个下载多个文件时，每次调用都必须再次将 `revision="main"` 解析为提交哈希。这会花费每个文件一次 HTTP 调用，更糟糕的是，如果存储库在其间更新，则相隔几秒进行的两次调用可能会导致两次不同的提交。

[HfApi.resolve_revision()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.resolve_revision) 解析一次修订并返回 [ResolvedRevision](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.ResolvedRevision)：

```py
>>> from huggingface_hub import resolve_revision
>>> revision = resolve_revision("openai-community/gpt2")
>>> revision
ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')
```[ResolvedRevision](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.ResolvedRevision) 是 `str` 子类，因此它可以传递给任何采用 `revision` 参数的 `huggingface_hub` 方法。它的字符串值是用户最初请求的（这里是`"main"`，因此是可读的错误消息），而`.resolved`保存提交哈希：

```py
>>> revision == "main"
True
>>> revision.resolved
'607a30d783dfa663caf39e06633721c8d4cfcd7e'
```

下载帮助程序（[hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)、[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)、[get_cached_repo_tree()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.get_cached_repo_tree)）检测[ResolvedRevision](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.ResolvedRevision)并直接使用提交哈希。每个文件都保证来自同一个提交，并且一旦文件被缓存，就根本不需要 HTTP 调用：

```py
>>> from huggingface_hub import hf_hub_download
>>> config = hf_hub_download("openai-community/gpt2", "config.json", revision=revision)
>>> weights = hf_hub_download("openai-community/gpt2", "model.safetensors", revision=revision)
```

`revision` -> `commit hash` 映射也会写入缓存的 `refs/` 文件夹（请参阅 [Refs](#refs)）。这意味着，如果稍后无法到达集线器（离线模式、连接错误、超时、集线器停机），[HfApi.resolve_revision()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.resolve_revision) 会透明地回退到缓存的值。如果也没有缓存任何内容，则会引发 [RevisionResolutionError](/docs/huggingface_hub/v1.29.0/en/package_reference/utilities#huggingface_hub.errors.RevisionResolutionError)。

## 基于块的缓存 (Xet)

为了提供更高效的文件传输，`hf_xet`在现有的`huggingface_hub`缓存中添加了`xet`目录，创建额外的缓存层以启用基于块的重复数据删除。该缓存保存块（大小约为 64KB 的文件的不可变字节范围）和分片（将文件映射到块的数据结构）。有关 Xet 存储系统的更多信息，请参阅此[section](https://huggingface.co/docs/hub/xet/index)。`xet`目录默认位于`~/.cache/huggingface/xet`，包含两个缓存，用于上传和下载。它具有以下结构：

```bash
<CACHE_DIR>
├─ xet
│  ├─ environment_identifier
│  │  ├─ chunk_cache
│  │  ├─ shard_cache
│  │  ├─ staging
```

`environment_identifier` 目录是一个编码字符串（它可能在您的计算机上显示为 `https___cas_serv-tGqkUaZf_CBPHQ6h`）。这在开发过程中使用，允许本地和生产版本的缓存同时存在。从位于不同 [storage regions](https://huggingface.co/docs/hub/storage-regions) 的存储库下载时也会使用它。你可能会在`xet`目录中看到多个这样的条目，每个条目对应不同的环境，但它们的内部结构是相同的。 

内部目录有以下用途：
* `chunk-cache` 包含用于加速下载的缓存数据块。
* `shard-cache` 包含在上传路径上使用的缓存分片。 
* `staging`是一个旨在支持断点续传的工作空间。

这些记录如下。请注意，`xet` 缓存系统与 `hf_xet` 的其余部分一样，与 `huggingface_hub` 完全集成。  如果您使用现有 API 与缓存资产交互，则无需更新您的工作流程。 `xet` 缓存作为优化层构建在现有 `hf_xet` 基于块的重复数据删除和 `huggingface_hub` 缓存系统之上。 

### `chunk_cache`

该缓存用于下载路径。缓存目录结构基于来自支持每个支持 Xet 的存储库的内容寻址存储 (CAS) 的 Base-64 编码哈希。 CAS 哈希充当查找数据存储位置偏移量的关键。注意：从 `hf_xet` 1.2.0 开始，默认情况下禁用 chunk_cache。要启用它，请在启动 Python 进程之前将 `HF_XET_CHUNK_CACHE_SIZE_BYTES` 环境变量设置为适当的大小。

在最顶层，base 64 编码的 CAS 哈希的前两个字母用于在 `chunk_cache` 中创建一个子目录（共享这前两个字母的键在此分组）。  内部级别由子目录组成，完整的键作为目录名。底层是缓存项，它们是包含缓存块的块范围。

```bash
<CACHE_DIR>
├─ xet
│  ├─ chunk_cache
│  │  ├─ A1
│  │  │  ├─ A1GerURLUcISVivdseeoY1PnYifYkOaCCJ7V5Q9fjgxkZWZhdWx0
│  │  │  │  ├─ AAAAAAEAAAA5DQAAAAAAAIhRLjDI3SS5jYs4ysNKZiJy9XFI8CN7Ww0UyEA9KPD9
│  │  │  │  ├─ AQAAAAIAAABzngAAAAAAAPNqPjd5Zby5aBvabF7Z1itCx0ryMwoCnuQcDwq79jlB

```请求文件时，`hf_xet` 做的第一件事是与 Xet 存储的内容寻址存储 (CAS) 通信以获取重建信息。重建信息包含有关下载完整文件所需的 CAS 密钥的信息。 

在执行 CAS 密钥请求之前，会查阅`chunk_cache`。如果缓存中的密钥与 CAS 密钥匹配，则没有理由发出对该内容的请求。 `hf_xet` 使用存储在目录中的块。

由于 `chunk_cache` 纯粹是一种优化，而不是保证，因此 `hf_xet` 采用了计算高效的驱逐策略。当`chunk_cache`已满时（见下文`Limits and Limitations`），`hf_xet`在选择逐出候选者时实施随机逐出策略。这显着减少了管理强大的缓存系统（例如 LRU）的开销，同时仍然提供缓存块的大部分好处。 

### `shard_cache`

将内容上传到集线器时会使用此缓存。该目录是扁平的，仅包含分片文件，每个分片文件使用一个 ID 作为分片名称。 

```sh
<CACHE_DIR>
├─ xet
│  ├─ shard_cache
│  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
│  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  ├─ ceeeb7ea4cf6c0a8d395a2cf9c08871211fbbd17b9b5dc1005811845307e6b8f.mdb
│  │  ├─ e8535155b1b11ebd894c908e91a1e14e3461dddd1392695ddc90ae54a548d8b2.mdb
```

`shard_cache` 包含以下碎片：- 本地生成并成功上传至CAS
- 作为全局重复数据删除算法的一部分从 CAS 下载

分片提供文件和块之间的映射。在上传过程中，每个文件都会被分块并保存该块的哈希值。然后查询缓存中的每个分片。如果分片包含正在上传的本地文件中存在的块哈希，则可以丢弃该块，因为它已经存储在 CAS 中。 

所有分片自下载之日起都有 3-4 周的有效期。过期的分片在上传时不会加载，并在过期后一周被删除。 

### `staging`

如果上传在新内容提交到存储库之前终止，您将需要恢复文件传输。但是，有可能在中断之前某些块已成功上传。 

这样您就不必从头开始，`staging`目录在上传期间充当工作空间，存储成功上传块的元数据。 `staging`目录具有以下形状：

```
<CACHE_DIR>
├─ xet
│  ├─ staging
│  │  ├─ shard-session
│  │  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  │  ├─ xorb-metadata
│  │  │  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
```当文件被处理并且块成功上传时，它们的元数据作为分片存储在`xorb-metadata`中。恢复上传会话后，将再次处理每个文件并查阅该目录中的分片。任何已成功上传的内容都会被跳过，并上传任何新内容（并保存其元数据）。 

同时，`shard-session`存储已处理文件的文件和块信息。成功完成上传后，这些分片中的内容将移动到更持久的`shard-cache`。

### 限制和限制

`chunk_cache` 的大小限制为 10GB，而 `shard_cache` 的软限制为 4GB。  根据设计，这两个缓存都没有高级 API，尽管它们的大小可以通过 `HF_XET_CHUNK_CACHE_SIZE_BYTES` 和 `HF_XET_SHARD_CACHE_SIZE_LIMIT` 环境变量进行配置。 

这些缓存主要用于促进文件的重建（下载）或上传。要与资产本身交互，建议您使用[⟦T134⟧ cache system APIs](https://huggingface.co/docs/huggingface_hub/guides/manage-cache)。

如果您需要回收任一缓存所使用的空间或需要调试任何潜在的缓存相关问题，只需运行 `rm -rf ~/<cache_dir>/xet` 即可完全删除 `xet` 缓存，其中 `<cache_dir>` 是 Hugging Face 缓存的位置，通常为 `~/.cache/huggingface`完整的`xet`缓存目录树示例：

```sh
<CACHE_DIR>
├─ xet
│  ├─ chunk_cache
│  │  ├─ L1
│  │  │  ├─ L1GerURLUcISVivdseeoY1PnYifYkOaCCJ7V5Q9fjgxkZWZhdWx0
│  │  │  │  ├─ AAAAAAEAAAA5DQAAAAAAAIhRLjDI3SS5jYs4ysNKZiJy9XFI8CN7Ww0UyEA9KPD9
│  │  │  │  ├─ AQAAAAIAAABzngAAAAAAAPNqPjd5Zby5aBvabF7Z1itCx0ryMwoCnuQcDwq79jlB
│  ├─ shard_cache
│  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
│  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  ├─ ceeeb7ea4cf6c0a8d395a2cf9c08871211fbbd17b9b5dc1005811845307e6b8f.mdb
│  │  ├─ e8535155b1b11ebd894c908e91a1e14e3461dddd1392695ddc90ae54a548d8b2.mdb
│  ├─ staging
│  │  ├─ shard-session
│  │  │  ├─ 906ee184dc1cd0615164a89ed64e8147b3fdccd1163d80d794c66814b3b09992.mdb
│  │  │  ├─ xorb-metadata
│  │  │  │  ├─ 1fe4ffd5cf0c3375f1ef9aec5016cf773ccc5ca294293d3f92d92771dacfc15d.mdb
```

要了解有关 Xet Storage 的更多信息，请参阅此[section](https://huggingface.co/docs/hub/xet/index)。

## 缓存资源

除了缓存来自 Hub 的文件外，下游库通常还需要缓存
其他与 HF 相关但未由 `huggingface_hub` 直接处理的文件（例如：文件
从 GitHub 下载，预处理数据，日志，...）。为了缓存这些文件，
称为`assets`，可以使用[cached_assets_path()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.cached_assets_path)。这个小助手生成路径
根据请求的库的名称以统一的方式存储在 HF 缓存中，并且
可选的名称空间和子文件夹名称。目标是让每一个下游
图书馆以自己的方式管理其资产（例如，结构上没有规则），只要它
保留在正确的资产文件夹中。然后，这些库可以利用以下工具
`huggingface_hub` 管理缓存，特别是扫描和删除部分内容
来自 CLI 命令的资产。

```py
from huggingface_hub import cached_assets_path

assets_path = cached_assets_path(library_name="datasets", namespace="SQuAD", subfolder="download")
something_path = assets_path / "something.json" # Do anything you like in your assets folder !
```

> [!提示]
> [cached_assets_path()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.cached_assets_path)是推荐的资产存储方式，但不是强制的。如果
> 您的库已经使用了自己的缓存，请随意使用它！

### 实践中的资产

实际上，您的资源缓存应类似于以下树：

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

## 管理基于文件的缓存

### 检查你的缓存目前，缓存的文件永远不会从本地目录中删除：当您下载时
分支的新修订版，以前的文件将被保留，以备您再次需要它们。
因此，检查缓存目录以了解哪个存储库会很有用
并且修订占用了大部分磁盘空间。 `huggingface_hub`提供您可以的帮助
从 `hf` CLI 或 Python 使用。

**从终端检查缓存**

运行 `hf cache ls` 来探索本地存储的内容。默认情况下该命令聚合
按存储库的信息：

```text
➜ hf cache ls
ID                                   SIZE   LAST_ACCESSED LAST_MODIFIED REFS
------------------------------------ ------- ------------- ------------- -------------------
dataset/glue                         116.3K 4 days ago     4 days ago     2.4.0 main 1.17.0
dataset/google/fleurs                 64.9M 1 week ago     1 week ago     main refs/pr/1
model/Jean-Baptiste/camembert-ner    441.0M 2 weeks ago    16 hours ago   main
model/bert-base-cased                  1.9G 1 week ago     2 years ago
model/t5-base                          10.1K 3 months ago   3 months ago   main
model/t5-small                        970.7M 3 days ago     3 days ago     main refs/pr/1

Found 6 repo(s) for a total of 12 revision(s) and 3.4G on disk.
```

添加 `--revisions` 列出每个缓存的快照和链过滤器以关注哪些内容
很重要。过滤器了解人类友好的大小和持续时间，因此诸如
`size>1GB` 或 `accessed>30d` 开箱即用：

```text
➜ hf cache ls --revisions --filter "size>1GB" --filter "accessed>30d"
ID                                   REVISION            SIZE   LAST_MODIFIED REFS
------------------------------------ ------------------ ------- ------------- -------------------
model/bert-base-cased                6d1d7a1a2a6cf4c2    1.9G  2 years ago
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main

Found 2 repo(s) for a total of 2 revision(s) and 3.0G on disk.
```

需要机器友好的输出吗？使用`--format json`获取结构化对象或
`--format csv` 用于电子表格。或者 `--quiet` 仅打印标识符（一个
每行），这样您就可以将它们通过管道传输到其他工具中。使用 `--sort` 按 `accessed`、`modified`、`name` 或 `size` 对条目进行排序（附加 `:asc` 或 `:desc` 来控制顺序），并使用 `--limit` 将结果限制为前 N 个条目。将这些选项与
`--cache-dir` 当您需要检查存储在 `HF_HOME` 之外的缓存时。**使用常用的shell工具进行过滤**

表格输出意味着您可以继续使用您已经知道的工具。例如，
下面的代码片段找到与 `t5-small` 相关的每个缓存修订版：

```text
➜ eval "hf cache ls --revisions" | grep "t5-small"
model/t5-small                       1c610f6b3f5e7d8a    1.1G  3 months ago  main
model/t5-small                       8f3ad1c90fed7a62    820.1M 2 weeks ago   refs/pr/1
```

**从 Python 检查缓存**

对于更高级的用法，请使用 [scan_cache_dir()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.scan_cache_dir)，这是由
CLI 工具。

您可以使用它来获取围绕 4 个数据类构建的详细报告：

- [HFCacheInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo)：[scan_cache_dir()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.scan_cache_dir)返回的完整报告
- [CachedRepoInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedRepoInfo)：有关缓存存储库的信息
- [CachedRevisionInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedRevisionInfo)：有关存储库内缓存修订版（例如“快照”）的信息
- [CachedFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.CachedFileInfo)：快照中缓存文件的信息

这是一个简单的使用示例。详细信息请参阅参考资料。

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
            last_accessed=1662971707.3567169,
            last_modified=1662971107.3567169,
            revisions=frozenset({
                CachedRevisionInfo(
                    commit_hash='d78aea13fa7ecd06c29e3e46195d6341255065d5',
                    size_on_disk=970726339,
                    snapshot_path=PosixPath(...),
                    # No `last_accessed` as blobs are shared among revisions
                    last_modified=1662971107.3567169,
                    files=frozenset({
                        CachedFileInfo(
                            file_name='config.json',
                            size_on_disk=1197
                            file_path=PosixPath(...),
                            blob_path=PosixPath(...),
                            blob_last_accessed=1662971707.3567169,
                            blob_last_modified=1662971107.3567169,
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

### 验证你的缓存

`huggingface_hub` 可以验证您的缓存文件是否与集线器上的校验和匹配。使用 `hf cache verify` CLI 验证特定存储库的特定修订版的文件一致性：

```bash
>>> hf cache verify meta-llama/Llama-3.2-1B-Instruct
✅ Verified 13 file(s) for 'meta-llama/Llama-3.2-1B-Instruct' (model) in ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
  All checksums match.
```

验证特定的缓存版本：

```bash
>>> hf cache verify meta-llama/Llama-3.1-8B-Instruct --revision 0e9e39f249a16976918f6564b8830bc894c89659
```

> [!提示]
> 查看 [⟦T165⟧ CLI reference](../package_reference/cli#hf-cache-verify) 了解有关用法的更多详细信息和完整的选项列表。

### 清理你的缓存扫描缓存很有趣，但接下来您真正想做的通常是
删除一些部分以释放驱动器上的一些空间。这可以使用
`hf cache rm` 和 `hf cache prune` CLI 命令。还可以通过编程方式使用
来自 [HFCacheInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo) 对象的 [delete_revisions()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo.delete_revisions) 辅助函数返回
扫描缓存。

**删除策略**

要删除某些缓存，您需要传递要删除的修订版本列表。该工具将
根据此列表定义释放空间的策略。它返回一个
[DeleteCacheStrategy](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.DeleteCacheStrategy) 描述哪些文件和文件夹将被删除的对象。
[DeleteCacheStrategy](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.DeleteCacheStrategy) 可以告诉您预计释放多少空间。
一旦您同意删除，您必须执行才能使删除生效。在
为了避免差异，您无法手动编辑策略对象。

删除修订的策略如下：

- 包含修订符号链接的`snapshot`文件夹被删除。
- 仅以要删除的修订为目标的 blob 文件也会被删除。
- 如果修订版链接到 1 个或多个 `refs`，则引用将被删除。
- 如果删除存储库中的所有修订，则整个缓存的存储库将被删除。> [!提示]
> 修订哈希值在所有存储库中都是唯一的。 `hf cache rm` 因此接受
> 存储库标识符（例如 `model/bert-base-uncased`）或裸版本哈希值；当
> 传递哈希值，您不需要单独指定存储库。

> [!警告]
> 如果在缓存中未找到修订版本，它将被静默忽略。此外，如果一个文件
> 或尝试删除时找不到文件夹，将记录警告但不会
> 抛出错误。继续删除该文件中包含的其他路径
> [DeleteCacheStrategy](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.DeleteCacheStrategy) 对象。

**从终端清除缓存**

使用 `hf cache rm` 从缓存中永久删除存储库或修订版本。通行证
一个或多个存储库标识符（例如`model/bert-base-uncased`）或修订哈希值：

```text
➜ hf cache rm model/bert-base-cased
About to delete 1 repo(s) totalling 1.9G.
  - model/bert-base-cased (entire repo)
Proceed with deletion? [y/N]: y
Deleted 1 repo(s) and 1 revision(s); freed 1.9G.
```

您还可以将 `hf cache rm` 与 `hf cache ls --quiet` 结合使用来批量删除过滤器识别的条目：

```bash
>>> hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

在同一个调用中混合存储库和修订版本。添加`--dry-run`预览影响，
或 `--yes` 在编写脚本时跳过确认提示：

```text
➜ hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```

在默认缓存位置之外工作时，将该命令与
`--cache-dir PATH`。要批量清理缓存垃圾，请运行`hf cache prune`。它会自动删除两者
分支或标签不再引用的修订以及任何剩余的 `.incomplete`
来自中断下载的文件：

```text
➜ hf cache prune
About to delete 3 unreferenced revision(s) and 2 incomplete download(s) (2.4G total).
  - model/t5-small:
      1c610f6b [refs/pr/1] 820.1M
      d4ec9b72 [(detached)] 640.5M
  - dataset/google/fleurs:
      2b91c8dd [(detached)] 937.6M
Proceed? [y/N]: y
Deleted 3 unreferenced revision(s) and 2 incomplete download(s); freed 2.4G.
```

`.incomplete` 文件是下载中断时留下的部分 blob。他们是
基于修订的扫描不会跟踪，因此 `hf cache ls` 仅用提示标记它们
(`Found X incomplete download(s) ...`) 而`hf cache prune`实际上是命令
删除它们。 `hf cache rm` 从不接触它们，除非它删除整个存储库。

这两个命令都支持 `--dry-run`、`--yes` 和 `--cache-dir`，因此您可以预览、自动化、
并根据需要定位备用缓存目录。

**从 Python 中清理缓存**

为了更加灵活，您还可以使用[delete_revisions()](/docs/huggingface_hub/v1.29.0/en/package_reference/cache#huggingface_hub.HFCacheInfo.delete_revisions)方法
以编程方式。这是一个简单的例子。详细信息请参阅参考资料。

```py
>>> from huggingface_hub import scan_cache_dir

>>> delete_strategy = scan_cache_dir().delete_revisions(
...     "81fd1d6e7847c99f5862c9fb81387956d99ec7aa"
...     "e2983b237dccf3ab4937c97fa717319a9ca1a96d",
...     "6c0e6080953db56375760c0471a8c5f2929baf11",
... )
>>> print("Will free " + delete_strategy.expected_freed_size_str)
Will free 8.6G

>>> delete_strategy.execute()
Cache deletion done. Saved 8.6G.
```

### 创建并分享模型卡
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/model-cards.md