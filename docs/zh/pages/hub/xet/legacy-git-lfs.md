<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 向后兼容 LFS

即使存储库已经由 Xet 支持，来自旧版/非 Xet 感知客户端的上传仍遵循标准 GitLFS 路径。文件上传到 LFS 后，后台进程会自动将文件迁移到使用 Xet 存储。 Xet 架构通过提供 Git LFS 桥为旧客户端从 Xet 支持的存储库下载文件提供向后兼容性。虽然支持 Xet 的客户端将从 CAS 接收文件重建信息以下载 Xet 支持的文件，但旧版客户端将从桥接器获取单个 URL，该桥接器负责重建请求文件并将 URL 返回到资源。这允许通过 URL 下载文件，以便您可以继续使用 Hub 的 Web 界面或 `curl`。通过自动迁移 LFS 文件上传并让旧客户端继续从 Xet 支持的存储库下载文件，维护人员和 Hub 的其他成员可以按照自己的节奏更新其管道。Xet 存储为现有 Hub 存储库提供无缝过渡。根本没有必要知道 Xet 后端是否参与其中。 Xet 支持的存储库继续使用 Git LFS 指针文件格式；添加 `Xet backed hash` 只是为了方便而添加到 Web 界面中。实际上，这意味着如果您对现有存储库和新创建的存储库执行`bare clone`，它们看起来不会有任何不同。每个大文件（或二进制文件）将继续拥有一个与 Git LFS 指针文件规范匹配的指针文件。

这种对称性允许非 Xet 感知的客户端（例如，旧版本的`huggingface_hub`）无需担心即可与 Xet 支持的存储库进行交互。事实上，在存储库中，支持 Git LFS 和 Xet 支持的文件的混合。 Xet 后端指示文件是否位于 Git LFS 或 Xet 存储中，从而允许下游服务从 S3 请求正确的 URL，无论哪个存储系统保存内容。

## 遗留存储：Git LFSHub 上的旧存储系统 Git LFS 使用许多与 Xet 支持的存储库相同的约定。 Hub 的 Git LFS 后端是[Amazon Simple Storage Service (S3)](https://aws.amazon.com/s3/)。当调用 Git LFS 时，它会使用 SHA256 哈希将文件内容存储在 S3 中，以命名文件以供将来访问。这种存储架构相对简单，允许 Hub 存储数百万个模型、数据集和空间存储库文件。

Git LFS 的主要限制是其以文件为中心的重复数据删除方法。对文件的任何更改，无论更改有多大或有多小，都意味着整个文件都会被版本控制 - 当上传整个文件（如果提交到存储库）或下载（如果将最新版本拉到您的计算机）时，会在文件传输中产生大量开销。

这会导致更糟糕的开发人员体验以及额外存储的激增。

### Xet：我们的存储后端
https://huggingface.co/docs/hub/xet/index.md