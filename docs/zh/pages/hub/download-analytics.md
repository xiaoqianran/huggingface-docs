<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 下载分析

下载分析为组织提供了 Hugging Face 使用情况的带宽级别视图：正在下载多少数据以及由谁下载。

  <img
    class="block dark:hidden m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/download-analytics.png"
    alt="screenshot of the organization Downloads settings page, showing daily egress usage and the per-member breakdown"
  />
  <img
    class="hidden dark:block m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-download-analytics.png"
    alt="screenshot of the organization Downloads settings page, showing daily egress usage and the per-member breakdown"
  />

## 访问下载分析

组织管理员可以在组织的下载设置页面`https://huggingface.co/organizations/[organizationIdentifier]/settings/downloads`找到下载分析。个人用户可以在`https://huggingface.co/settings/downloads`查看自己的使用情况。

## 出口概览

该页面显示每日出站流量（下载带宽）图表，以及当月至今的总出站流量与计划包含的出站流量限额的比较。仅统计通过 Hugging Face CDN 的流量，随着数据的稳定，数字可能会随着时间的推移而增加。

## 每个成员的详细信息

对于组织的每个成员，细分显示：

- 下载的总数据量（以字节为单位）
- 最后一次下载的日期

这使管理员能够快速了解谁在推动整个组织的带宽使用，一目了然。

## 数据范围

下载分析报告每个成员的总带宽；它不记录下载了哪些特定存储库、修订版或文件。

如果您发布自己的模型或数据集并希望了解这些存储库的每个存储库下载计数，请参阅[Publisher Analytics](./publisher-analytics)。## 数据可用性和保留

CDN 使用计量保留 90 天。

### 处理大型数据集
https://huggingface.co/docs/hub/jobs-large-datasets.md