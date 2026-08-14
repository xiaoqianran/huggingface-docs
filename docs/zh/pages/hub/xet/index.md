<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Xet：我们的存储后端

Hugging Face Hub 上的存储库与软件开发平台上的存储库不同。它们包含以下文件：

- 大 - 模型或数据集文件在 GB 及以上范围内。我们有一些 TB 级的文件！
- 二进制 - 默认情况下不是人类可读的格式（例如 [Safetensors](https://huggingface.co/docs/safetensors/en/index) 或 [Parquet](https://huggingface.co/docs/dataset-viewer/en/parquet#what-is-parquet)）

虽然 Hub 在 Git 的支持下利用现代版本控制，但这些差异使得 [Model](https://huggingface.co/docs/hub/models) 和 [Dataset](https://huggingface.co/docs/hub/datasets) 存储库与仅包含源代码的存储库截然不同。

将这些文件直接存储在纯 Git 存储库中是不切实际的。 Git 存储库背后的典型存储系统不仅不适合此类文件，而且当您克隆存储库时，Git 会检索整个历史记录，包括所有文件修订。对于大量二进制文件来说，这可能太大了，迫使您下载可能永远不需要的千兆字节的历史数据。相反，在 Hub 上，这些大文件使用“指针文件”进行跟踪，并通过 `.gitattributes` 文件进行识别（这两个文件将在下面更详细地讨论），这些文件保留在 Git 存储库中，而实际数据存储在远程存储中（如 [Amazon S3](https://aws.amazon.com/s3/)）。因此，存储库保持较小，并且典型的 Git 工作流程仍然高效。

从历史上看，Hub 存储库一直依赖 [Git LFS](https://git-lfs.com/) 来实现此机制。虽然 Git LFS 仍然受支持（请参阅[Backwards Compatibility & Legacy](./legacy-git-lfs)），但 Hub 已采用 Xet，这是一种专为 AI/ML 开发而构建的现代自定义存储系统。与 Git LFS 相比，它支持块级重复数据删除、更小的上传和更快的下载。

## 开源 Xet 协议

如果您希望了解底层 Xet 协议或希望构建新的客户端库来访问 Xet 存储，请查看 [Xet Protocol Specification](https://huggingface.co/docs/xet/index)。

在这些页面中，您将开始使用 Xet Storage。

## 内容

- [Xet History & Overview](./overview)
- [Using Xet Storage](./using-xet-storage)
- [Security](./security)
- [Backwards Compatibility & Legacy](./legacy-git-lfs)
- [Deduplication](./deduplication)

### 使用 Xet 存储
https://huggingface.co/docs/hub/xet/using-xet-storage.md