<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Xet 历史与概述

[In August 2024 Hugging Face acquired XetHub](https://huggingface.co/blog/xethub-joins-hf)、[seed-stage startup based in Seattle](https://www.geekwire.com/2023/ex-apple-engineers-raise-7-5m-for-new-seattle-data-storage-startup/)，用于替换 Hub 上的 Git LFS。

与 Git LFS 一样，Xet 支持的存储库利用 S3 作为远程存储，并在存储库根目录中使用 `.gitattributes` 文件帮助确定应远程存储哪些文件。

Git LFS 指针文件提供元数据来定位远程存储中的实际文件内容：

- **SHA256**：为实际大文件提供唯一标识符。该标识符是通过计算文件内容的 SHA-256 哈希值生成的。
- **指针大小**：Git 存储库中存储的指针文件的大小。
- **远程文件的大小**：指示实际大文件的大小（以字节为单位）。此元数据对于验证目的以及管理存储和传输操作都很有用。

Xet 指针在设计上包含所有这些信息。请参阅 [backwards compatibility with Git LFS](legacy-git-lfs#backward-compatibility-with-lfs) 部分，并添加 `Xet backed hash` 字段以引用 Xet 存储中的文件。与在文件级别进行重复数据删除的 Git LFS 不同，支持 Xet 的存储库在字节级别进行重复数据删除。当 Xet 存储支持的文件更新时，只有修改后的数据才会上传到远程存储，从而显着节省网络传输。对于许多工作流程，例如模型检查点的增量更新或将新数据附加/插入到数据集中，这可以提高您和您的协作者的迭代速度。要了解有关 Xet 存储中重复数据删除的更多信息，请参阅[Deduplication](deduplication)。

### 使用 Xet 存储
https://huggingface.co/docs/hub/xet/using-xet-storage.md