<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 出版商分析

> [!警告]
> 此功能是团队和企业计划的一部分。

## 发布商分析仪表板

通过详细的下载概述跟踪您的所有存储库活动，该概述显示您的组织发布的所有模型和数据集的总下载量。在“所有时间”和“上个月”视图之间切换，以深入了解不同时期的存储库。

### 每个存储库细分

使用每个存储库的深入分析表探索各个存储库的指标。利用内置的搜索功能快速找到特定的存储库。每行还具有一个时间序列图，说明下载量随时间变化的趋势。

## 将发布商分析导出为 CSV

下载包含所有存储库分析的综合 CSV 文件，包括模型和数据集下载活动。

您还可以通过以下 API 端点以编程方式访问此数据：

```bash
curl -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  "https://huggingface.co/organizations/YOUR_ORG_NAME/settings/publisher-analytics/download-breakdown" \
  --output breakdown.csv
```

### 响应结构

CSV 文件由每个模型和数据集的每日下载记录组成。

```csv
repoType,repoName,total,timestamp,downloads
model,huggingface/CodeBERTa-small-v1,4362460,2021-01-22T00:00:00.000Z,4
model,huggingface/CodeBERTa-small-v1,4362460,2021-01-23T00:00:00.000Z,7
model,huggingface/CodeBERTa-small-v1,4362460,2021-01-24T00:00:00.000Z,2
dataset,huggingface/documentation-images,2167284,2021-11-27T00:00:00.000Z,3
dataset,huggingface/documentation-images,2167284,2021-11-28T00:00:00.000Z,18
dataset,huggingface/documentation-images,2167284,2021-11-29T00:00:00.000Z,7
```

### 存储库对象结构

CSV 中的每条记录包含：- `repoType`：存储库的类型（例如“模型”、“数据集”）
- `repoName`：包含组织的完整存储库名称（例如“huggingface/documentation-images”）
- `total`：该存储库的累计下载次数
- `timestamp`：ISO 8601 格式日期 (UTC)
- `downloads`：当天的下载量

记录按时间顺序排序，并提供每个存储库下载活动的每日详细视图。

> [!注意]
> 用户**不会**对下载数据进行重复数据删除。如果您需要唯一下载计数，请参阅下一节。

## 独特的下载器和更精细的日志

> [!警告]
> 此功能是 Enterprise Plus 计划的附加功能。

作为一项高级功能，Hugging Face 可以导出组织发布的所有模型和数据集的匿名请求级访问日志。每行代表一个与下载相关的请求，为您提供模型和数据集下载数据的完整粒度。您的团队负责摄取这些日志并对其运行计算。导出有意包含原始 HTTP 状态代码和方法，以便您可以根据自己的分析需求对 `HEAD`、部分内容、重定向和其他请求模式进行分类。

|专栏 |描述 |
| -------------- | ------------------------------------------------------- |
| `timestamp` |请求时间戳|
| `status` | HTTP 状态代码（例如 `200`、`206`、`302`、`307`、`304`）|
| `method` | HTTP 方法（例如 `GET`、`HEAD`）|
| `repoName` |完整的存储库名称（例如 `nvidia/segformer-b0`）|
| `repoType` |存储库类型：`model`、`dataset` 或 `space` |
| `hashedUserId` |用户 ID 的不可逆哈希值（如果经过身份验证）|
| `hashedIp` | IP 地址的不可逆散列（如果未经身份验证）|
| `country` |国家 ISO 代码 |
| `region` |地区或城市名称 |
| `userAgent` | HTTP 用户代理标头 |由于它需要在我们这边设置自定义数据导出管道（自定义弹性索引等），因此这只能作为 Enterprise Plus 的附加组件提供。

### 您的第一个 Docker 空间：使用 T5 生成文本
https://huggingface.co/docs/hub/spaces-sdks-docker-first-demo.md