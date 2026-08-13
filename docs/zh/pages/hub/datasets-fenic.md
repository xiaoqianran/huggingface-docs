<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 芬尼克

[fenic](https://github.com/typedef-ai/fenic) 是一个受 PySpark 启发的 DataFrame 框架，设计用于构建生产 AI 和代理应用程序。 fenic 支持直接从 Hugging Face Hub 读取数据集。

## 开始使用

首先，pip install `fenic`：

```bash
pip install fenic
```

### 创建会话

使用默认配置实例化 fenic 会话（足以读取数据集和其他非语义操作）：

```python
import fenic as fc

session = fc.Session.get_or_create(fc.SessionConfig())
```

## 概述

fenic 是一个固执己见的数据处理框架，它结合了：
- **DataFrame API**：受 PySpark 启发的操作，用于熟悉的数据操作
- **语义操作**：内置 AI/LLM 操作，包括语义函数、嵌入和聚类
- **模型集成**：对 AI 提供商（Anthropic、OpenAI、Cohere、Google）的本机支持
- **查询优化**：通过逻辑计划转换自动优化

## 阅读 Hugging Face Hub

fenic 可以使用 `hf://` 协议直接从 Hugging Face Hub 读取数据集。此功能内置于 fenic 的 DataFrameReader 接口中。

### 支持的格式

fenic 支持从 Hugging Face 读取以下格式：
- **镶木地板文件** (`.parquet`)
- **CSV 文件** (`.csv`)### 读取数据集

要从 Hugging Face Hub 读取数据集：

```python
import fenic as fc

session = fc.Session.get_or_create(fc.SessionConfig())

# Read a CSV file from a public dataset
df = session.read.csv("hf://datasets/datasets-examples/doc-formats-csv-1/data.csv")

# Read Parquet files using glob patterns
df = session.read.parquet("hf://datasets/cais/mmlu/astronomy/*.parquet")

# Read from a specific dataset revision
df = session.read.parquet("hf://datasets/datasets-examples/doc-formats-csv-1@~parquet/**/*.parquet")
```

### 使用模式管理阅读

```python
# Read multiple CSV files with schema merging
df = session.read.csv("hf://datasets/username/dataset_name/*.csv", merge_schemas=True)

# Read multiple Parquet files with schema merging
df = session.read.parquet("hf://datasets/username/dataset_name/*.parquet", merge_schemas=True)
```

> **注意：** 在 fenic 中，模式是列名及其数据类型的集合。当您启用 `merge_schemas` 时，fenic 会尝试通过用空值填充缺失的列并在可能的情况下扩大类型来协调文件之间的差异。有些布局仍然无法合并 - 请参阅 fenic 文档了解 [CSV schema merging limitations](https://docs.fenic.ai/latest/reference/fenic/?h=parquet#fenic.DataFrameReader.csv) 和 [Parquet schema merging limitations](https://docs.fenic.ai/latest/reference/fenic/?h=parquet#fenic.DataFrameReader.parquet)。

### 身份验证

要读取私有数据集，您需要将 Hugging Face 令牌设置为环境变量：

```shell
export HF_TOKEN="REDACTED"
```

### 路径格式

fenic 中的 Hugging Face 路径格式遵循以下结构：
```
hf://{repo_type}/{repo_id}/{path_to_file}
```

您还可以指定数据集修订或版本：
```
hf://{repo_type}/{repo_id}@{revision}/{path_to_file}
```

特点：
- 支持全局模式（`*`、`**`）
- 使用 `@` 表示法的数据集修订/版本：
  - 具体提交：`@d50d8923b5934dc8e74b66e6e4b0e2cd85e9142e`
  - 分行：`@refs/convert/parquet`
  - 分行别名：`@~parquet`
- 私有数据集需要`HF_TOKEN`环境变量

### 混合数据源

fenic 允许您在单个读取操作中组合多个数据源，包括混合不同的协议：

```python
# Mix HF and local files in one read call
df = session.read.parquet([
    "hf://datasets/cais/mmlu/astronomy/*.parquet",
    "file:///local/data/*.parquet",
    "./relative/path/data.parquet"
])
```

这种灵活性使您能够在数据处理管道中无缝组合来自 Hugging Face Hub 的数据和本地文件。## 处理拥抱脸部数据

从 Hugging Face 加载后，您可以使用 fenic 的完整 DataFrame API：

### 基本数据帧操作

```python
import fenic as fc

session = fc.Session.get_or_create(fc.SessionConfig())

# Load IMDB dataset from Hugging Face
df = session.read.parquet("hf://datasets/imdb/plain_text/train-*.parquet")

# Filter and select
positive_reviews = df.filter(fc.col("label") == 1).select("text", "label")

# Group by and aggregate
label_counts = df.group_by("label").agg(
    fc.count("*").alias("count")
)
```

### 人工智能驱动的运营

要使用语义和嵌入操作，请在 SessionConfig 中配置语言和嵌入模型。配置完成后：

```python
import fenic as fc

# Requires OPENAI_API_KEY to be set for language and embedding calls
session = fc.Session.get_or_create(
    fc.SessionConfig(
        semantic=fc.SemanticConfig(
            language_models={
                "gpt-4o-mini": fc.OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=60,
                    tpm=60000,
                )
            },
            embedding_models={
                "text-embedding-3-small": fc.OpenAIEmbeddingModel(
                    model_name="text-embedding-3-small",
                    rpm=60,
                    tpm=60000,
                )
            },
        )
    )
)

# Load a text dataset from Hugging Face
df = session.read.parquet("hf://datasets/imdb/plain_text/train-00000-of-00001.parquet")

# Add embeddings to text columns
df_with_embeddings = df.select(
    "*",
    fc.semantic.embed(fc.col("text")).alias("embedding")
)

# Apply semantic functions for sentiment analysis
df_analyzed = df_with_embeddings.select(
    "*",
    fc.semantic.analyze_sentiment(
        fc.col("text"),
        model_alias="gpt-4o-mini",  # Optional: specify model
    ).alias("sentiment")
)
```

## 示例：分析 MMLU 数据集

```python
import fenic as fc

# Requires OPENAI_API_KEY to be set for semantic calls
session = fc.Session.get_or_create(
    fc.SessionConfig(
        semantic=fc.SemanticConfig(
            language_models={
                "gpt-4o-mini": fc.OpenAILanguageModel(
                    model_name="gpt-4o-mini",
                    rpm=60,
                    tpm=60000,
                )
            },
        )
    )
)

# Load MMLU astronomy subset from Hugging Face
df = session.read.parquet("hf://datasets/cais/mmlu/astronomy/*.parquet")

# Process the data
processed_df = (df
    # Filter for specific criteria
    .filter(fc.col("subject") == "astronomy")
    # Select relevant columns
    .select("question", "choices", "answer")
    # Add difficulty analysis using semantic.map
    .select(
        "*",
        fc.semantic.map(
            "Rate the difficulty of this question from 1-5: {{question}}",
            question=fc.col("question"),
            model_alias="gpt-4o-mini"  # Optional: specify model
        ).alias("difficulty")
    )
)

# Show results
processed_df.show()
```

## 资源

- [fenic GitHub Repository](https://github.com/typedef-ai/fenic)
- [fenic Documentation](https://docs.fenic.ai/latest/)

### 合并数据集并导出
https://huggingface.co/docs/hub/datasets-duckdb-combine-and-export.md