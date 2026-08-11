<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 环境变量

`huggingface_hub`可以使用环境变量进行配置。

如果您不熟悉环境变量，这里有关于它们的通用文章
[on macOS and Linux](https://linuxize.com/post/how-to-set-and-list-environment-variables-in-linux/)
以及[Windows](https://phoenixnap.com/kb/windows-set-environment-variable)。

此页面将引导您了解特定于 `huggingface_hub` 的所有环境变量
以及它们的意义。

> [!提示]
> 所有环境变量均在`huggingface_hub` 导入时读取。任何修改
> 之后制作的内容将不予考虑。确保设置环境变量
> 导入`huggingface_hub` 之前。

## 通用

### HF_INFERENCE_ENDPOINT

配置推理 api 基本 url。如果您的组织您可能需要设置此变量
指向 API 网关而不是直接指向推理 api。

默认为`"https://api-inference.huggingface.co"`。

### HF_HOME

配置`huggingface_hub`本地存储数据的位置。特别是，您的代币
并且缓存将存储在该文件夹中。

默认为 `"~/.cache/huggingface"`，除非设置了 [XDG_CACHE_HOME](#xdgcachehome)。

### HF_HUB_CACHE

配置来自 Hub 的存储库在本地缓存的位置（模型、数据集和
空格）。

默认为 `"$HF_HOME/hub"`（例如默认为 `"~/.cache/huggingface/hub"`）。

### HF_XET_CACHE

配置 Xet 块（Xet 后端管理的文件的字节范围）在本地缓存的位置。默认为 `"$HF_HOME/xet"`（例如默认为 `"~/.cache/huggingface/xet"`）。

### HF_ASSETS_CACHE

配置下游库创建[assets](../guides/manage-cache#caching-assets)的位置
将被缓存在本地。这些资产可以是预处理数据、从 GitHub 下载的文件、
日志、...

默认为 `"$HF_HOME/assets"`（例如默认为 `"~/.cache/huggingface/assets"`）。

### HF_TOKEN

配置用户访问令牌以向集线器进行身份验证。如果设置，该值将
覆盖存储在机器上的令牌（如果未设置，则在`$HF_TOKEN_PATH`或`"$HF_HOME/token"`中）。

有关身份验证的更多详细信息，请查看[this section](../quick-start#authentication)。

### HF_TOKEN_PATH

配置 `huggingface_hub` 应存储用户访问令牌的位置。默认为`"$HF_HOME/token"`（例如默认为`~/.cache/huggingface/token`）。

### HF_HUB_VERBOSITY

设置 `huggingface_hub` 记录器的详细级别。必须是其中之一
`{"debug", "info", "warning", "error", "critical"}`。

默认为`"warning"`。

欲了解更多详情，请参阅[logging reference](../package_reference/utilities#huggingface_hub.utils.logging.get_verbosity)。

### HF_HUB_ETAG_TIMEOUT整数值，用于定义在下载文件之前从存储库获取最新元数据时等待服务器响应的秒数。如果请求超时，`huggingface_hub`将默认使用本地缓存的文件。设置较低的值可以加快连接速度较慢且已缓存文件的计算机的工作流程。较高的值可保证元数据调用在更多情况下成功。默认为 10 秒。

### HF_HUB_DOWNLOAD_TIMEOUT

用于定义下载文件时等待服务器响应的秒数的整数值。如果请求超时，则会引发 TimeoutError。设置较高的值对于连接速度较慢的计算机是有益的。较小的值会使进程在网络完全中断的情况下失败得更快。默认为 10 秒。

## Xet 

### 其他 Xet 环境变量
* [⟦T22⟧](../package_reference/environment_variables#hfhubdisablexet)
* [⟦T23⟧](../package_reference/environment_variables#hfxetcache)
* [⟦T24⟧](../package_reference/environment_variables#hfxethighperformance)
* [⟦T25⟧](../package_reference/environment_variables#hfxetreconstructwritesequentially)

### HF_XET_CHUNK_CACHE_SIZE_BYTES本地设置 Xet 块缓存的大小。默认情况下，块缓存是禁用的。如果您要生成现有模型或数据集的新修订，块缓存可能会很有用，因为它用于缓存从 S3 获取的术语/块。更大的缓存可以更好地利用跨存储库和文件的重复数据删除。要启用块缓存，请将环境变量设置为较大的数字 (10GB) 或更大。然而，在大多数情况下，当下载或上传新数据时，禁用块缓存会有更好的性能，这就是默认情况下禁用它的原因。

默认为`0`（0字节，表示块缓存被禁用）。

### HF_XET_SHARD_CACHE_SIZE_LIMIT

本地设置 Xet 分片缓存的大小。增加此值将提高上传效率，因为缓存分片文件中引用的块不会重新上传。请注意，默认软限制可能足以满足大多数工作负载。 

默认为 `16000000000` (16GB)。

### HF_XET_NUM_CONCURRENT_RANGE_GETS设置每个文件从 S3 下载的并发项数（xorb 内的字节范围，通常称为块）。如果有可用的网络带宽，增加此值将有助于提高下载文件的速度。 

默认为`16`。

## 布尔值

以下环境变量需要布尔值。将考虑变量
如果其值为 `{"1", "ON", "YES", "TRUE"}` 之一（不区分大小写），则为 `True`。任何其他值
（或未定义）将被视为`False`。

### HF_DEBUG

如果设置，`huggingface_hub` 记录器的日志级别将设置为 DEBUG。此外，HF 库发出的所有请求都将记录为等效的 cURL 命令，以便于调试和再现。

### HF_HUB_OFFLINE

如果设置，则不会对 Hugging Face Hub 进行 HTTP 调用。如果您尝试下载文件，则只会访问缓存的文件。如果未检测到缓存文件，则会引发错误。如果您的网络速度较慢并且您不关心文件的最新版本，这非常有用。

如果`HF_HUB_OFFLINE=1`设置为环境变量，并且调用[HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi)的任何方法，都会引发[OfflineModeIsEnabled](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.OfflineModeIsEnabled)异常。**注意：** 即使缓存了文件的最新版本，调用 `hf_hub_download` 仍然会触发 HTTP 请求来检查新版本是否可用。设置 `HF_HUB_OFFLINE=1` 将跳过此调用，从而加快加载时间。

如果你想检查离线模式是否启用，你可以使用[is_offline_mode()](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.is_offline_mode)助手。

### HF_HUB_DISABLE_IMPLICIT_TOKEN

并非每个发送到集线器的请求都必须进行身份验证。例如，请求
`"gpt2"`型号详情无需认证。但是，如果用户是
[logged in](../package_reference/login)，默认行为是始终发送令牌
为了在访问私有或门控存储库时减轻用户体验（永远不会出现 HTTP 401 Unauthorized）。为了保护隐私，您可以
通过设置 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 禁用此行为。在这种情况下，
令牌将仅针对“写访问”调用发送（例如：创建提交）。

**注意：** 禁用隐式发送令牌可能会产生奇怪的副作用。例如，
如果您想列出 Hub 上的所有模型，则您的私有模型将不会列出。你
需要在脚本中显式传递 `token=True` 参数。

### HF_HUB_DISABLE_PROGRESS_BARS对于耗时的任务，`huggingface_hub`默认显示进度条（使用tqdm）。
您可以通过设置`HF_HUB_DISABLE_PROGRESS_BARS=1`一次性禁用所有进度条。

### HF_HUB_DISABLE_SYMLINKS

如果设置，`huggingface_hub`将永远不会在缓存中创建符号链接。相反，文件将被复制或直接移动到快照目录中。这是一项高级用户功能，使缓存目录以降级模式运行，其中巨大的文件最终会在硬盘驱动器上重复。

一个示例用例是当共享网络驱动器（例如 NAS）在运行不同操作系统的机器上用作 `HF_HUB_CACHE` 时
系统。在 Linux 上创建的符号链接并不总是可以在 Windows 上遍历，从而导致错误。设置`HF_HUB_DISABLE_SYMLINKS=1`可以避免此问题，但代价是磁盘空间重复数据删除。

### HF_HUB_DISABLE_SYMLINKS_WARNING如果您使用的是Windows机器，建议启用开发者模式或运行
`huggingface_hub` 处于管理模式。否则，`huggingface_hub`将无法创建
缓存系统中的符号链接。您将能够执行除用户体验之外的任何脚本
将会降级，因为一些大文件可能最终会在您的硬盘驱动器上重复。警告
系统会触发消息来警告您此行为。设置`HF_HUB_DISABLE_SYMLINKS_WARNING=1`，
禁用此警告。

欲了解更多详情，请参阅[cache limitations](../guides/manage-cache#limitations)。

### HF_HUB_DISABLE_EXPERIMENTAL_WARNING

`huggingface_hub` 的一些功能是实验性的。这意味着您可以使用它们，但我们不保证它们会被使用
保留在未来。特别是，我们可能会更新此类功能的 API 或行为，而不会造成任何弃用
循环。使用实验性功能时会触发警告消息来警告您。如果您愿意使用实验性功能调试任何潜在问题，则可以设置 `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` 来禁用警告。

如果您正在使用实验性功能，请告诉我们！您的反馈可以帮助我们设计和改进它。

### HF_HUB_DISABLE_TELEMETRY默认情况下，HF 库（`transformers`、`datasets`、`gradio`、..）收集一些数据，以监控使用情况、调试问题并帮助确定功能的优先级。
每个库定义了自己的策略（即要监视哪些使用情况），但核心实现发生在`huggingface_hub`（请参阅`send_telemetry`）。

您可以将 `HF_HUB_DISABLE_TELEMETRY=1` 设置为环境变量以全局禁用遥测。

### HF_HUB_DISABLE_UPDATE_CHECK

默认情况下，`hf` CLI 在启动时检查 PyPI 是否有较新版本（最多每 24 小时一次），并在可用时向 stderr 打印一行黄色警告，建议使用 `hf update`。该检查在开发版本和预发布版本上已经是禁止操作。

它还检查（最多每 24 小时一次，纯本地）`hf-cli` 技能是否已安装并由运行版本生成，如果没有，则建议 `hf skills add` / `hf skills update`。该检查仅打印提示，它永远不会触及您的技能目录。

设置 `HF_HUB_DISABLE_UPDATE_CHECK=1` 跳过 PyPI 请求并完全静音两个提示。在离线 CI 环境中或当您喜欢更安静的 shell 输出时很有用。

### HF_HUB_DISABLE_XET设置为禁用使用 `hf-xet`，即使它在您的 Python 环境中可用。这是因为如果找到`hf-xet`，它将自动使用，这允许显式禁用其使用。如果您要禁用 Xet，请考虑 [filing an issue and including the diagnostics](https://github.com/huggingface/xet-core?tab=readme-ov-file#issues-diagnostics--debugging) 信息，以帮助我们了解为什么 Xet 不适合您。

### HF_HUB_ENABLE_HF_TRANSFER

> [!警告]
> 这是一个已弃用的环境变量。
> 现在 Hugging Face Hub 完全由 Xet 存储后端提供支持，所有文件传输都通过 `hf-xet` 二进制包进行。它使用基于块的重复数据删除策略提供高效的传输，并与`huggingface_hub`无缝集成。
> 这意味着`hf_transfer`不能再使用了。如果您对更高的性能感兴趣，请查看[⟦T66⟧ section](#hf_xet_high_performance)

### HF_XET_HIGH_PERFORMANCE

将 `hf-xet` 设置为以增加的设置运行，以最大限度地利用计算机上的网络和磁盘资源。启用高性能模式将尝试使本机的网络带宽饱和，并利用所有 CPU 核心进行并行上传/下载活动。

将此视为类似于旧版 `HF_HUB_ENABLE_HF_TRANSFER=1` 环境变量，但应用于 `hf-xet`。要了解有关 Xet 存储和 `hf_xet` 的优势的更多信息，请参阅此 [section](https://huggingface.co/docs/hub/xet/index)。

### HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY

让`hf-xet`顺序写入本地磁盘，而不是并行。 `hf-xet` 专为 SSD/NVMe 磁盘而设计（使用直接寻址的并行写入）。如果您使用的是 HDD（旋转硬盘），设置此项会将磁盘写入更改为顺序写入而不是并行写入。对于速度较慢的硬盘，这可以提高整体写入性能，因为磁盘不会旋转来寻求并行写入。

## 已弃用的环境变量

为了标准化 Hugging Face 生态系统中的所有环境变量，一些变量已被标记为已弃用。尽管它们仍然有效，但它们不再优先于其替代品。下表概述了已弃用的变量及其相应的替代变量：

|已弃用的变量 |更换|
| ------------------------ | | ------------------ |
| `HUGGINGFACE_HUB_CACHE` | `HF_HUB_CACHE` |
| `HUGGINGFACE_ASSETS_CACHE` | `HF_ASSETS_CACHE` |
| `HUGGING_FACE_HUB_TOKEN` | `HF_TOKEN` |

## 来自外部工具

某些环境变量并非特定于 `huggingface_hub`，但在设置时仍会考虑在内。### 请勿追踪

布尔值。相当于`HF_HUB_DISABLE_TELEMETRY`。当设置为 true 时，遥测在 Hugging Face Python 生态系统中全局禁用（`transformers`、`diffusers`、`gradio` 等）。请参阅 https://donottrack.sh/ 了解更多详细信息。

### 无颜色

布尔值。设置后，`hf` CLI 将不会打印任何 ANSI 颜色。
参见[no-color.org](https://no-color.org/)。

### XDG_CACHE_HOME

仅当`HF_HOME`未设置时使用！

这是配置[user-specific non-essential (cached) data should be written](https://wiki.archlinux.org/title/XDG_Base_Directory)的默认方式
在 Linux 机器上。

如果未设置`HF_HOME`，则默认主页为`"$XDG_CACHE_HOME/huggingface"`
`"~/.cache/huggingface"`。

### Webhook 服务器
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/webhooks_server.md