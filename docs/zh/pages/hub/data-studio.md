<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据工作室

每个数据集页面都包含一个表格，其中包含数据集的内容，按 100 行的页面排列。您可以使用表格底部的按钮在页面之间导航。

## 检查数据分布

在列的顶部，您可以看到代表数据分布的图表。这使您可以快速了解类的平衡程度、数值数据的范围和分布以及文本的长度以及列数据缺失的部分。

## 按值过滤

如果您单击数字列中的直方图条，数据集查看器将过滤数据并仅显示值落在所选范围内的行。
同样，如果您从分类列中选择一个类，它将仅显示所选类别中的行。

## 在数据集中搜索一个单词

您可以通过在表格顶部的搜索栏中输入单词来搜索数据集中的单词。搜索不区分大小写，并且将匹配包含该单词的任何行。即使值嵌套在字典或列表中，也会在 `string` 的列中搜索文本。

## 对数据集运行 SQL 查询您可以使用 SQL 控制台在浏览器中对数据集运行 SQL 查询。此功能还利用了我们的[auto-conversion to Parquet](data-studio#access-the-parquet-files)。

有关更多信息，请参阅我们的 [SQL Console](./datasets-viewer-sql-console) 指南。您可以使用 [⟦T3⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-sql) 或直接使用 [DuckDB](./datasets-duckdb) 从命令行运行相同的 DuckDB SQL 查询。

## 共享特定行

您可以通过单击特定行，然后复制浏览器地址栏中的 URL 来共享该行。例如 https://huggingface.co/datasets/nyu-mll/glue/viewer/mrpc/test?p=2&row=241 将在 MRPC 数据集、测试分割和第 241 行上打开数据集工作室。

## 代理痕迹

将原始 JSONL 跟踪从 Claude Code、Codex 或 Pi 等工具推送到数据集中，以便在 Data Studio 中浏览它们，或者将它们同步到 [Storage Bucket](./storage-buckets) 并在其中打开单独的 `.jsonl` 文件。两条路径都使用相同的跟踪查看器来逐步执行会话、轮次、工具调用和模型响应。请参阅 [Agent Traces](./agent-traces) 了解支持的代理和查看跟踪。

## 大规模数据集

数据集查看器支持大规模数据集，但根据数据格式，它可能只显示数据集的前 5GB：- 对于 Parquet 数据集：数据集查看器显示完整数据集，但仅在前 5GB 上启用排序、过滤和搜索。
- 对于其他格式（例如[WebDataset](https://github.com/webdataset/webdataset)或JSON Lines）> 5GB的数据集：数据集查看器仅显示前5GB，并且在前5GB上启用排序、过滤和搜索。

在这种情况下，一条信息性消息会让您知道查看器是部分的。这应该是一个足够大的样本，可以准确地表示完整的数据集，如果您需要更大的样本，请告诉我们。

## 访问镶木地板文件

为了支持数据集查看器，每个数据集的前 5GB 都会自动转换为 Parquet 格式（除非它已经是 Parquet 数据集）。在数据集查看器中（例如，参见 [GLUE](https://huggingface.co/datasets/nyu-mll/glue)），您可以单击 [_"Auto-converted to Parquet"_](https://huggingface.co/datasets/nyu-mll/glue/tree/refs%2Fconvert%2Fparquet/cola) 来访问 Parquet 文件。请参阅[dataset viewer docs](/docs/datasets-server/parquet_process)了解如何使用Polars、Pandas或DuckDB等库查询数据集parquet文件。> [!提示]
> Parquet 是一种针对查询和处理大型数据集而优化的列式存储格式。 Parquet 是大数据处理和分析的热门选择，广泛用于数据处理和机器学习。您可以在文档中详细了解与此格式相关的优点。

### 转换机器人

当您创建新数据集时，[⟦T5⟧ bot](https://huggingface.co/parquet-converter) 将数据集转换为 Parquet 后会通知您。它在存储库中打开的 [discussion](./repositories-pull-requests-discussions) 提供了有关 Parquet 格式的详细信息以及 Parquet 文件的链接。

### 编程访问

您还可以使用 [Hub API](./api#get-apidatasetsrepoidparquet) 以编程方式访问 Parquet 文件列表；例如，端点 [⟦T6⟧](https://huggingface.co/api/datasets/nyu-mll/glue/parquet) 列出了 `nyu-mll/glue` 数据集的 parquet 文件。 `hf` CLI 提供与 [⟦T9⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-parquet) 相同的列表，[⟦T10⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-sql) 允许您使用 SQL 查询文件。

我们还有关于[Dataset Viewer API](https://huggingface.co/docs/dataset-viewer)的具体文档，您可以直接调用。该 API 允许您访问所有 Hugging Face Hub 数据集的内容、元数据和基本统计​​数据，并为数据集查看器前端提供支持。

## 数据集预览对于最大的数据集，该页面显示前 100 行的预览，而不是全功能查看器。此限制仅适用于超过 5GB、本身不是 Parquet 格式或尚未自动转换为 Parquet 的数据集。

## 在网页中嵌入数据集查看器

您可以使用 iframe 将数据集查看器嵌入到您自己的网页中。要使用的 URL 为 `https://huggingface.co/datasets/<namespace>/<dataset-name>/embed/viewer`，其中 `<namespace>` 是数据集的所有者，`<dataset-name>` 是数据集的名称。您还可以传递其他参数，例如子集、拆分、过滤器、搜索或选定行。

有关更多信息，请参阅我们的 [How to embed the Dataset Viewer in a webpage](./datasets-viewer-embed) 指南。

## 配置数据集查看器

要为您的数据集提供正常工作的数据集查看器，请确保您的数据集采用受支持的格式和结构。
还有一个选项可以使用 YAML 配置数据集。

您可以通过在数据集的 `README.md` 文件顶部添加 YAML 配置块来指定要在数据集查看器中显示的文件。例如，选择哪个文件进入哪个分割：

```yaml
---
configs:
- config_name: default
  data_files:
  - split: train
    path: "data.csv"
  - split: test
    path: "holdout.csv"
---
```

您还可以为每个拆分选择多个文件或使用 glob 模式：

```yaml
---
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - "data/train_part1.csv"
    - "data/train_part2.csv"
  - split: test
    path: "data/*.csv"
---
```

对于 **私有** 数据集，为 [PRO users](https://huggingface.co/pricing) 和 [Team or Enterprise organizations](https://huggingface.co/enterprise) 启用数据集查看器。有关更多信息，请参阅我们的 [How to configure the Dataset Viewer](./datasets-viewer-configure) 指南。

### 恶意软件扫描
https://huggingface.co/docs/hub/security-malware.md