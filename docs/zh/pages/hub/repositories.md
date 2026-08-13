<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储库

模型、空间和数据集以 [Git repositories](https://git-scm.com/about) 的形式托管在 Hugging Face Hub 上，这意味着版本控制和协作是 Hub 的核心元素。简而言之，存储库（也称为 **repo**）是可以存储代码和资产以备份您的工作、与社区共享以及在团队中工作的地方。

> [!提示]
> 正在寻找非版本化的可变存储？查看 [Storage Buckets](./storage-buckets)，它提供类似 S3 的对象存储，无需 Git 历史记录。

与其他协作平台不同，我们的 Git 存储库针对机器学习和 AI 文件进行了优化 - 大型二进制文件，通常采用 Parquet 和 Safetensors 等特定文件格式，最高可达 [Terabyte-scale sizes](https://huggingface.co/blog/from-files-to-chunks)！
为了实现这一目标，我们构建了[Xet](./xet/index)，这是一个专为 AI/ML 开发而构建的现代定制存储系统，可实现块级重复数据删除、更小的上传和更快的下载。

在这些页面中，您将了解 Git 和 Xet 入门以及与 Hub 上的存储库交互的基础知识。一旦掌握了窍门，您就可以探索我们为有效存储库使用而编制的最佳实践和后续步骤。

＃＃ 内容- [Getting Started with Repositories](./repositories-getting-started)
- [Settings](./repositories-settings)
- [Storage Limits](./storage-limits)
- [Storage Backend (Xet)](./xet/index)
- [Local Cache](./local-cache)
- [Pull Requests & Discussions](./repositories-pull-requests-discussions)
  - [Pull Requests advanced usage](./repositories-pull-requests-discussions#pull-requests-advanced-usage)
- [Collections](./collections)
- [Notifications](./notifications)
- [Webhooks](./webhooks)
- [Next Steps](./repositories-next-steps)
- [Licenses](./repositories-licenses)

### 高级主题
https://huggingface.co/docs/hub/models-advanced.md