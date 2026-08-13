<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 Xet 存储

##Python

要访问 Xet 感知版本的 `huggingface_hub`，只需安装最新版本：

```bash
pip install -U huggingface_hub
```

从 `huggingface_hub` 0.32.0 开始，这也将安装 `hf_xet`。 `hf_xet` 包将 `huggingface_hub` 与 [⟦T16⟧](https://github.com/huggingface/xet-core)（Xet 后端的 Rust 客户端）集成。

如果您使用 `transformers` 或 `datasets` 库，则它已经在使用 `huggingface_hub`。只要`huggingface_hub`的版本>=0.32.0，就不需要采取进一步的操作。

其中 `huggingface_hub` >= 0.30.0 和 < 0.32.0 are installed, ⟦T22⟧ must be installed explicitly:

⟦T1⟧

And that's it! You now get the benefits of Xet deduplication for both uploads and downloads. Team members using a version of ⟦T23⟧ < 0.30.0 will still be able to upload and download repositories through the ⟦T163⟧.

To see more detailed usage docs, refer to the ⟦T24⟧ docs for:

- ⟦T164⟧
- ⟦T165⟧
- ⟦T166⟧

## Git

Git users can access the benefits of Xet by downloading and installing the Git Xet extension. Once installed, simply use the ⟦T167⟧ - no additional changes necessary.

### Prerequisites

Install ⟦T168⟧ and ⟦T169⟧. 

### Install on macOS or Linux (amd64 or aarch64)

Install using an installation script with the following command in your terminal (requires ⟦T26⟧ and ⟦T27⟧):
⟦T2⟧

Or, install using ⟦T170⟧:
⟦T3⟧

To verify the installation, run:
⟦T4⟧

### Windows (amd64)

Using ⟦T28⟧:
⟦T5⟧

Using an installer: 
 - Download ⟦T29⟧ (⟦T171⟧) and unzip. 
 - Run the ⟦T30⟧ installer file and follow the prompts.

Manual installation:
 - Download ⟦T31⟧ (⟦T172⟧) and unzip. 
 - Place the extracted ⟦T32⟧ under a ⟦T33⟧ directory.
 - Run ⟦T34⟧ in a terminal.

To verify the installation, run:
  ⟦T6⟧

### Using Git Xet

Once installed on your platform, using Git Xet is as simple as following the Hub's standard Git workflows.

Make sure all ⟦T173⟧, follow the ⟦T174⟧, then commit your changes, and ⟦T35⟧ to the Hub:

  ⟦T7⟧
Under the hood, the ⟦T175⟧ is invoked to upload large files directly to Xet storage, increasing upload speeds through the power of ⟦T176⟧.

### Uninstall on macOS or Linux

Using Homebrew:
⟦T8⟧
If you used the installation script (for MacOS or Linux), run the following in your terminal:
⟦T9⟧
### Uninstall on Windows

If you used ⟦T36⟧:
⟦T10⟧

If you used the installer:
-  Navigate to Settings -> 应用程序 -> 已安装的应用程序
- 找到“Git-Xet”。
- 选择上下文菜单中的“卸载”选项。

如果您手动安装：
- 在终端中运行`git xet uninstall`。 
- 从最初放置的位置删除 `git-xet.exe` 文件。

## 建议

Xet 与 Hub 的所有工作流程无缝集成。但是，您可以考虑采取一些步骤来从 Xet 存储中获得最大收益。

使用Python上传或下载时：- **确保安装`hf_xet`**：虽然 Xet 仍然向后兼容针对 Git LFS 优化的旧版客户端，但 `hf_xet` 与 `huggingface_hub` 集成可提供最佳的基于块的性能和对大文件的更快迭代。
- **默认启用自适应并发**：`hf_xet`根据实时网络状况自动调整并行传输流的数量 - 无需配置。默认设置将使大多数网络路径饱和，无需任何调整。
- **高级调整**：对于细粒度控制，`HF_XET_FIXED_DOWNLOAD_CONCURRENCY` 和 `HF_XET_FIXED_UPLOAD_CONCURRENCY` 让您可以绕过自适应控制器将并发固定为固定值。请参阅 `hf_xet` 的 [environment variables](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#xet) 了解完整的选项列表。

在 Git 或 Python 中上传或下载时：- **利用频繁的增量提交**：Xet 的块级重复数据删除意味着您可以安全地对模型或数据集进行增量更新。仅上传更改的块，因此频繁提交既快速又高效。
- **在 .gitattributes 中具体化**：定义 Xet 或 LFS 模式时，使用精确的文件扩展名（例如，`*.safetensors`、`*.bin`）以避免不必要地通过大文件存储路由较小的文件。
- **优先考虑社区访问**：Xet 大幅提高了大文件传输的效率和规模。不要构建存储库以减少其总大小（或单个文件的大小），而是为协作者和社区用户组织存储库，以便他们可以轻松导航和检索所需的内容。

## 环境变量

`hf_xet` 和 Git Xet 均由 `xet-core` 提供支持，可以通过环境变量进行配置。下表列出了细粒度控制的各个变量。大多数用户不需要更改任何这些 - 默认值已调整为自动饱和大多数网络路径。> [!注意]
> `HF_XET_HIGH_PERFORMANCE=1` 是一个方便的标志，可以一次调整多个设置（并发范围、缓冲区大小和并行文件限制）。它适用于具有高带宽**和至少 64 GB RAM** 用于缓冲的机器。在内存较少的机器上，它可能会降低性能。

### 一般

大多数用户首先接触到的高级标志。

|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_HIGH_PERFORMANCE`（别名`HF_XET_HP`）|关闭 |通过一次性提高并发性、缓冲区大小和并行文件限制来最大化网络和 CPU 使用率的便利标志。请参阅上面的注释 - 最好在具有高带宽和至少 64 GB RAM 的计算机上使用。 |
| `HF_XET_CACHE` | `$HF_HOME/xet` | Xet 在本地缓存数据的目录（下载的块和重复数据删除分片）。优先于`HF_HOME`。 |

### 自适应并发

默认情况下，`xet-core`使用自适应并发——根据实时网络状况动态调整并行度。这些是在大多数情况下不太可能需要的高级设置。以下变量控制自适应控制器的行为：|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_CLIENT_ENABLE_ADAPTIVE_CONCURRENCY` | `true` |启用或禁用自适应并发控制。禁用时，并发度保持初始值。 |
| `HF_XET_CLIENT_AC_INITIAL_UPLOAD_CONCURRENCY` | `1` |并发上传流的起始数量。 HP模式：`16`。 |
| `HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY` | `1` |并发下载流的起始数量。 HP模式：`16`。 |
| `HF_XET_CLIENT_AC_MIN_UPLOAD_CONCURRENCY` | `1` |上传并发下限。 HP模式：`4`。 |
| `HF_XET_CLIENT_AC_MIN_DOWNLOAD_CONCURRENCY` | `1` |下载并发下限。 HP模式：`4`。 |
| `HF_XET_CLIENT_AC_MAX_UPLOAD_CONCURRENCY` | `64` |上传并发上限。 HP模式：`124`。 |
| `HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY` | `64` |下载并发上限。 HP模式：`124`。 |
| `HF_XET_CLIENT_AC_TARGET_RTT` | `60s` |目标往返时间。只要完整传输的预测往返时间低于此值，并发性就会增加。 |
| `HF_XET_CLIENT_AC_MAX_HEALTHY_RTT` | `90s` |可接受的最大往返时间。自适应控制器将花费超过此时间的传输视为失败。 |
| `HF_XET_CLIENT_AC_HEALTHY_SUCCESS_RATIO_THRESHOLD` | `0.8` |成功率高于该成功率控制器会增加并发性。 |
| `HF_XET_CLIENT_AC_UNHEALTHY_SUCCESS_RATIO_THRESHOLD` | `0.5` |成功率低于该值控制器会降低并发性。 |
| `HF_XET_CLIENT_AC_LOGGING_INTERVAL_MS` | `10000` |记录并发状态的时间间隔（以毫秒为单位）。 |> [!提示]
> 要将并发固定为固定值（绕过自适应控制器），请使用方便的别名 `HF_XET_FIXED_UPLOAD_CONCURRENCY` 和 `HF_XET_FIXED_DOWNLOAD_CONCURRENCY`。它们将初始、最小和最大并发设置为相同的值。

### 网络并重试

|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_CLIENT_RETRY_MAX_ATTEMPTS` | `5` |失败请求的最大重试次数。 |
| `HF_XET_CLIENT_RETRY_BASE_DELAY` | `3000ms` |重试之间的基本延迟（具有指数退避）。 |
| `HF_XET_CLIENT_RETRY_MAX_DURATION` | `360s` |重试请求所花费的最大总时间。 |
| `HF_XET_CLIENT_CONNECT_TIMEOUT` | `60s` | TCP 连接超时。 |
| `HF_XET_CLIENT_READ_TIMEOUT` | `120s` |读取 HTTP 响应超时。 |
| `HF_XET_CLIENT_IDLE_CONNECTION_TIMEOUT` | `60s` |空闲连接关闭之前超时。 |
| `HF_XET_CLIENT_MAX_IDLE_CONNECTIONS` | `16` |池中的最大空闲连接数。 |

### 数据传输|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_DATA_MAX_CONCURRENT_FILE_INGESTION` | `8` |上传期间同时处理的最大文件数。 HP模式：`100`。 |
| `HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS` | `8` |同时下载的最大文件数。 |
| `HF_XET_DATA_INGESTION_BLOCK_SIZE` | `8mb` |文件摄取期间读取的块的大小。 |
| `HF_XET_DATA_PROGRESS_UPDATE_INTERVAL` | `200ms` |进度条更新的频率。 |
| `HF_XET_DATA_PROGRESS_UPDATE_SPEED_SAMPLING_WINDOW` | `10s` |用于在进度报告中聚合传输速度测量值的时间窗口。 |

### 下载缓冲区

这些控制下载期间的内存使用。 `HF_XET_HIGH_PERFORMANCE=1` 显着提高了这些。|环境变量 |默认|惠普模式|描述 |
|---|---|---|---|
| `HF_XET_RECONSTRUCTION_MIN_RECONSTRUCTION_FETCH_SIZE` | `256mb` | `1gb` |重建请求的最小获取大小。 |
| `HF_XET_RECONSTRUCTION_MAX_RECONSTRUCTION_FETCH_SIZE` | `8gb` | `16gb` |重建请求的最大获取大小。 |
| `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_SIZE` | `2gb` | `16gb` |总下载缓冲区大小。 |
| `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_PERFILE_SIZE` | `512mb` | `2gb` |每个文件下载缓冲区大小。 |
| `HF_XET_RECONSTRUCTION_DOWNLOAD_BUFFER_LIMIT` | `8gb` | `64gb` |总下载缓冲区内存的硬限制。 |
| `HF_XET_RECONSTRUCTION_TARGET_BLOCK_COMPLETION_TIME` | `15m` | — |完成预取块的目标时间。用于确定下载期间提前预取多少数据。 |
| `HF_XET_RECONSTRUCTION_MIN_PREFETCH_BUFFER` | `1gb` | — |下载期间保持预取的最小数据量，无论预计完成时间如何。 |

### 分片缓存

**分片缓存** 在磁盘上的 Xet 缓存目录下保存重复数据删除索引（描述已上传块的“分片”）。更大的缓存可以让客户端对更多以前见过的数据进行重复数据删除，从而减少重新上传的字节数。|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_SHARD_CACHE_SIZE_LIMIT` | `16gb` |磁盘分片缓存的软上限。缓存在运行的**开始**时被修剪到此大小，但可能会在单个长时间运行的会话中增长超过它。作为粗略指南，大小 *X* 的缓存可对大约 1000 × *X* 数据（16 GB → ~16 TB）进行重复数据删除。提高它可以在非常大的存储库上更好地进行重复数据删除；如果不断增长的缓存有填满磁盘的风险，请降低它。接受人类可读的大小（例如`32gb`）或原始字节计数。 |
| `HF_XET_SHARD_CHUNK_INDEX_TABLE_MAX_SIZE` | `64mb` |内存中块索引的最大大小。一旦达到，就不再加载更多块以进行重复数据删除。 |
| `HF_XET_SHARD_CACHE_SUBDIR` | `shard-cache` | Xet 缓存目录中分片缓存的子目录。 |

### 块缓存

**块缓存**将下载的字节范围（块）存储在磁盘上，因此不会从存储中重新获取重叠的数据。当重复下载相关模型或数据集或生成相关模型或数据集的新修订版时，它最有用。|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_CHUNK_CACHE_SIZE_BYTES` | `0`（已禁用）|本地块缓存的大小。 `hf_xet` Python 包附带缓存**默认情况下禁用**；设置字节数（例如 `10000000000` 表示 10 GB）以启用它，或设置 `0` 禁用它。该变量在 Git Xet v0.1 中无效。 |

### 日志记录

|环境变量 |默认|描述 |
|---|---|---|
| `HF_XET_LOG_DEST` | （无）|日志目的地。接受文件路径或目录路径（以`/`结尾）。当设置为目录时，将使用带时间戳的名称创建日志文件。当设置为空字符串时，日志将转到控制台。取消设置后，日志将转到 Hugging Face Xet 缓存目录中的 `logs/` 子目录。 |
| `HF_XET_LOG_FORMAT` | （无）|日志格式。对于 JSON 格式的日志，设置为`json`；否则纯文本。默认情况下，文件日志记录使用 JSON，控制台日志记录使用文本。 |
| `HF_XET_LOG_PREFIX` | `xet` |登录到目录时日志文件名的前缀。 |
| `HF_XET_LOG_DIR_DISABLE_CLEANUP` | `false` |禁用日志目录中旧日志文件的自动清理。 |
| `HF_XET_LOG_DIR_MAX_SIZE` | `250mb` |日志目录中日志文件的最大总大小。旧文件会被修剪以保持在此限制以下。 || `HF_XET_LOG_DIR_MIN_DELETION_AGE` | `1d` |清理期间可以删除日志文件的最短期限。 |
| `HF_XET_LOG_DIR_MAX_RETENTION_AGE` | `14d` |日志文件的最长期限。清理过程中始终会删除早于此时间的文件。 |

## 目前的限制

虽然 Xet 为基于 Git 的存储带来了细粒度的重复数据删除和增强的性能，但一些功能和平台兼容性仍在开发中。因此，在使用支持 Xet 的存储库时，请记住以下限制：

- **仅限 64 位系统**：`hf_xet` 和 Git Xet 目前都需要 64 位架构；不支持 32 位系统。

### Xet：我们的存储后端
https://huggingface.co/docs/hub/xet/index.md