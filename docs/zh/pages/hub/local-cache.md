<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 集线器本地缓存

本文档描述了 HF Hub 本地缓存的磁盘布局。它旨在作为以任何语言重新实现缓存系统的参考。

以下是使用此缓存布局的库和应用程序的部分列表。请打开 PR 来添加您自己的 PR。

### 图书馆

|图书馆 |语言 |笔记|
|--------|----------|--------|
| [⟦T12⟧](https://github.com/huggingface/huggingface_hub) |蟒蛇 |以及任何依赖于它的库（例如`transformers`，`diffusers`，`datasets`，`mlx`，`vllm`...）|
| [⟦T18⟧](https://github.com/huggingface/hf-hub) |铁锈| |
| [⟦T19⟧](https://github.com/huggingface/swift-huggingface) |斯威夫特 | |
| [⟦T20⟧](https://github.com/huggingface/huggingface.js) | JavaScript |仅限 Node.js |
| [⟦T21⟧](https://github.com/haifengl/smile) |爪哇 |请参阅[⟦T22⟧](https://haifengl.github.io/api/java/smile/util/HuggingFaceHub.html)文档 |

### 应用程序

|应用 |语言 |笔记|
|------------|---------|--------|
| [⟦T23⟧](https://github.com/ggml-org/llama.cpp) | C++ |自 [#20775](https://github.com/ggml-org/llama.cpp/pull/20775) |
| [⟦T24⟧](https://github.com/ggml-org/LlamaBarn) |斯威夫特 | macOS 应用程序，基于 `llama.cpp` 构建 |
| [⟦T26⟧](https://github.com/bodaay/HuggingFaceModelDownloader) |去 | |
| [⟦T27⟧](https://github.com/jundot/omlx) |蟒蛇 | macOS MLX 推理服务器，自 [v0.4.0](https://github.com/jundot/omlx/commit/341aebbeaba1ca3452f83dedf03a0d102cef9fd4) |

## 缓存位置

默认缓存目录是：

```
~/.cache/huggingface/hub
```

这可以用环境变量覆盖：
- `HF_HUB_CACHE` - 缓存目录的直接路径（优先）
- `HF_HOME` - Hugging Face 主目录的路径；如果设置，缓存位于`$HF_HOME/hub`

## 概述

```
<CACHE_DIR>/
├── .locks/                                  # Lock files for concurrent download safety
├── models--<org>--<repo>/                   # Cached model repositories
├── datasets--<org>--<repo>/                 # Cached dataset repositories
└── spaces--<org>--<repo>/                   # Cached space repositories
```每个下载的存储库都有一个单一的平面文件夹。在每个存储库文件夹内，文件一次存储在内容寻址的 `blobs/` 目录中，并通过 `snapshots/` 符号链接访问。命名引用（分支、标签）在 `refs/` 中跟踪。

## 架构

```
                     ┌──────────────────────────────────────────┐
                     │           Repository folder              │
                     │  models--julien-c--EsperBERTo-small      │
                     └──────────────┬───────────────────────────┘
                                    │
              ┌─────────────┬───────┴───────┬──────────────┐
              │             │               │              │
              v             v               v              v
          ┌───────┐    ┌────────┐    ┌────────────┐   ┌──────────┐
          │ refs/ │    │ blobs/ │    │ snapshots/ │   │.no_exist/│
          └───┬───┘    └────┬───┘    └──────┬─────┘   └────┬─────┘
              │             │               │              │
              │             │               │              │
   "main" contains     Files stored     One folder     Empty marker
   commit hash         by content       per commit     files for
   e.g. "aaaaaa"       hash (SHA-1      hash, e.g.     files known
                       or SHA-256)      aaaaaa/        not to exist
   Resolves a                           bbbbbb/
   branch/tag to                            │
   a snapshot ──────────────────────►  Contains symlinks
                                       to ../../blobs/{hash}
```

## 存储库文件夹命名

存储库作为平面目录存储在缓存根目录中。文件夹名称对存储库类型和存储库 ID 进行编码：

```
{type}s--{repo_id_with_slashes_replaced_by_--}
```

规则：
- 仓库类型是**复数**：`models`、`datasets`、`spaces`
- 存储库 ID 中的正斜杠 (`/`) 替换为 `--`
- 所有部分之间的分隔符是`--`
- 外壳被保留

示例：

|集线器存储库 ID |回购类型 |缓存文件夹名称 |
|--------------------------------------|------------------------|---------------------------------------------------------|
| `julien-c/EsperBERTo-small` |型号| `models--julien-c--EsperBERTo-small` |
| `huggingface/DataMeasurementsFiles` |数据集| `datasets--huggingface--DataMeasurementsFiles` |
| `dalle-mini/dalle-mini` |空间| `spaces--dalle-mini--dalle-mini` |

> [!注意]
> 存储桶不由该缓存处理，因为它们不受 git 支持。请改用专用的 `hf buckets sync` 命令。

## 在存储库文件夹内

每个缓存存储库都有相同的内部结构：

```
<repo_folder>/
├── blobs/
├── refs/
├── snapshots/
└── .no_exist/    # may not always be present
```

### `blobs/`：内容寻址文件存储`blobs/`目录存储实际的文件内容。每个文件都以其在 Hub 上的文件 etag 命名：

- **Git 跟踪的文件**：由其 **SHA-1** 哈希命名（40 个十六进制字符）
- **Git LFS 文件**：由其 **SHA-256** 哈希命名（64 个十六进制字符）

这是一个平面目录——没有子目录。不同版本的相同文件仅存储一次。

```
blobs/
├── 403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd   # SHA-256 (LFS)
├── 7cb18dc9bafbfcf74629a4b760af1b160957a83e                           # SHA-1 (git)
└── d7edf6bd2a681fb0175f7735299831ee1b22b812                           # SHA-1 (git)
```

### `refs/`：分支和标签引用

`refs/` 目录将人类可读的引用（分支名称、标签、PR 号）映射到提交哈希值。

每个引用都是一个纯文本文件，包含一行：完整提交哈希（40 个十六进制字符）。

```
refs/
├── main            # contains e.g. "bbc77c8132af1cc5cf678da3f1ddf2de43606d48"
├── 2.4.0           # a tag
└── refs/
    └── pr/
        └── 1       # pull request reference
```

当使用分支或标记名称下载文件时，将使用最新的提交哈希创建或更新相应的引用文件。

### `snapshots/`：修订视图

`snapshots/` 目录包含每个缓存修订版（提交哈希）的一个子目录。每个修订目录都镜像 Hub 上存储库的文件结构，但文件是指向 `../../blobs/{hash}` 的**符号链接**。

```
snapshots/
├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
│   ├── README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
│   └── pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
└── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
    ├── README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
    └── pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
```主要特性：
- 符号链接使用**相对路径**：`../../blobs/{hash}`
- 如果文件在两个修订版之间未更改，则两个符号链接都指向**相同的 blob** 且没有数据重复
- Hub 上子目录中的文件在快照中表示为子目录（保留完整的相对路径）

快照之间的切换类似于在本地 git 存储库中使用 `git checkout`。

### `.no_exist/`：不存在的缓存

`.no_exist/` 目录跟踪已请求但集线器上不存在的文件。这避免了对可选文件的重复 HTTP 请求。

结构镜像`snapshots/`：每个提交哈希一个子目录，包含以丢失文件命名的**空文件**（不是符号链接）。

```
.no_exist/
└── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
    └── config_that_does_not_exist.json    # empty file
```

磁盘使用量可以忽略不计，因为这些只是空标记文件。

## 锁定文件

锁定文件可防止并发进程同时下载相同的 blob。它们存储在缓存根目录的 `.locks/` 目录中（不在 repo 文件夹内）：

```
<CACHE_DIR>/.locks/<repo_folder_name>/<blob_hash>.lock
```

示例：
```
<CACHE_DIR>/.locks/models--julien-c--EsperBERTo-small/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd.lock
```

## 完整示例

```
~/.cache/huggingface/hub/
├── .locks/
│   └── models--julien-c--EsperBERTo-small/
│       └── 403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd.lock
│
└── models--julien-c--EsperBERTo-small/
    ├── blobs/
    │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
    │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
    │
    ├── refs/
    │   └── main    # contains "bbc77c8132af1cc5cf678da3f1ddf2de43606d48"
    │
    ├── snapshots/
    │   ├── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
    │   │   ├── README.md         -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
    │   │   └── pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │   │
    │   └── bbc77c8132af1cc5cf678da3f1ddf2de43606d48/
    │       ├── README.md         -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
    │       └── pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │
    └── .no_exist/
        └── 2439f60ef33a0d46d85da5001d52aeda5b00ce9f/
            └── optional_config.json    # empty file
```

请注意 `pytorch_model.bin` 如何在两个版本中指向 **相同的 blob**。 321 MB 文件仅在磁盘上存储一次。

## 文件解析逻辑

要在磁盘上找到缓存文件：1. **解决提交哈希的修订**
   - 如果修订版本已经是 40 个字符的十六进制字符串，则直接使用它
   - 否则，读取`refs/{revision}`处的文件以获取提交哈希

2. **查看快照**
   - 寻找`snapshots/{commit_hash}/{relative_path}`
   - 如果它存在（作为符号链接或文件），则该文件被缓存。按照符号链接获取内容

3. **检查不存在**
   - 寻找`.no_exist/{commit_hash}/{relative_path}`
   - 如果存在，则该文件已知在此版本的集线器上不存在

4. **缓存未命中**
   - 如果两个路径都不存在，则文件尚未被缓存

## Windows 行为

缓存依赖于**符号链接**。在符号链接不可用的 Windows 系统上，缓存以 **降级模式** 运行：实际文件副本直接放置在 `snapshots/` 而不是符号链接中。该模式下不使用`blobs/`目录。

这意味着相同的文件内容可能会在各个版本之间重复，从而增加磁盘使用量。要在 Windows 上启用符号链接支持，请激活 [Developer Mode](https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development) 或以管理员身份运行。

### 使用 Keras 拥抱脸部
https://huggingface.co/docs/hub/keras.md