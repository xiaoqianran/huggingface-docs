<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据集下载统计

## 数据集的下载量如何计算？

计算数据集的下载数量并不是一项简单的任务，因为单个数据集存储库可能包含来自多个子集和拆分（例如训练/验证/测试）的多个文件，有时单个拆分中包含许多文件。为了解决这个问题并避免多次计算一个人的下载量，我们将给定存储库中用户在 5 分钟窗口内下载的所有文件（基于其 IP 地址）视为单个数据集下载。下载文件时（通过 GET 或 HEAD 请求），此计数会自动在我们的服务器上进行，无需收集任何用户信息或进行其他调用。

## 2024 年 9 月之前

该中心过去仅提供可通过 `datasets` 库加载的数据集的下载统计信息。为了确定下载次数，Hub 之前每次在 Python 中调用 `load_dataset` 时都会进行计数，不包括 GitHub 上的 Hugging Face 的 CI 工具。用户没有发送任何信息，也没有为此拨打任何其他电话。当我们提供文件下载时，计数是在服务器端完成的。这意味着：* 无论数据是直接存储在 Hub 存储库上还是存储库具有 [script](/docs/datasets/dataset_script) 从外部源加载数据，下载计数都是相同的。
* 如果用户使用 `wget` 或 Hub 的用户界面 (UI) 等工具手动下载数据，这些下载不包含在下载计数中。

## 如果我的数据集需要更精细的下载数据怎么办？

如果您需要更精细的下载数据，例如：
- 区分数据文件和元数据，
- 排除来自 CI/CD 管道的下载，
- 或删除重复用户（即计算唯一的下载者），

那么[Publisher Analytics](./publisher-analytics)，特别是[granular logs](./publisher-analytics#unique-downloaders-and-more-granular-logs)功能，可以为您的组织发布的所有模型和数据集提供匿名的请求级访问日志。

这些作为原始日志提供，因为大多数组织都希望应用自己的自定义规则。

### 在拥抱脸部使用 mlx-image
https://huggingface.co/docs/hub/mlx-image.md