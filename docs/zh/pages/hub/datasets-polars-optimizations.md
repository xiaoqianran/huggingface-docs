<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 优化

我们简要讨论了惰性求值和急切求值之间的区别。在本页中，我们将展示如何使用惰性 API 来获得巨大的性能优势。

## 懒惰与渴望

Polars 支持两种操作模式：lazy 和 eager。在急切 API 中，查询会立即执行，而在惰性 API 中，查询仅在“需要”时才进行评估。将执行推迟到最后一刻可以带来显着的性能优势，这就是为什么在大多数非交互式情况下首选惰性 API 的原因。

## 示例

我们将使用上一页中的示例来展示使用惰性 API 的性能优势。下面的代码将计算来自`archive.org`的上传数量。

### 渴望

```python
import polars as pl
import datetime

df = pl.read_csv("hf://datasets/commoncrawl/statistics/tlds.csv", try_parse_dates=True)

df = df.select("suffix", "crawl", "date", "tld", "pages", "domains")
df = df.filter(
    (pl.col("date") >= datetime.date(2020, 1, 1)) |
    pl.col("crawl").str.contains("CC")
)
df = df.with_columns(
    (pl.col("pages") / pl.col("domains")).alias("pages_per_domain")
)
df = df.group_by("tld", "date").agg(
    pl.col("pages").sum(),
    pl.col("domains").sum(),
)
df = df.group_by("tld").agg(
    pl.col("date").unique().count().alias("number_of_scrapes"),
    pl.col("domains").mean().alias("avg_number_of_domains"),
    pl.col("pages").sort_by("date").pct_change().mean().alias("avg_page_growth_rate"),
).sort("avg_number_of_domains", descending=True).head(10)
```

### 懒惰

```python
import polars as pl
import datetime

lf = (
    pl.scan_csv("hf://datasets/commoncrawl/statistics/tlds.csv", try_parse_dates=True)
    .filter(
        (pl.col("date") >= datetime.date(2020, 1, 1)) |
        pl.col("crawl").str.contains("CC")
    ).with_columns(
        (pl.col("pages") / pl.col("domains")).alias("pages_per_domain")
    ).group_by("tld", "date").agg(
        pl.col("pages").sum(),
        pl.col("domains").sum(),
    ).group_by("tld").agg(
        pl.col("date").unique().count().alias("number_of_scrapes"),
        pl.col("domains").mean().alias("avg_number_of_domains"),
        pl.col("pages").sort_by("date").pct_change().mean().alias("avg_page_growth_rate"),
    ).sort("avg_number_of_domains", descending=True).head(10)
)
df = lf.collect()
```

### 时间

在具有家庭互联网连接的普通笔记本电脑上运行这两个查询会导致以下运行时间：

- 渴望：`1.96`秒
- 懒惰：`410`毫秒惰性查询比急切查询快约 5 倍。原因在于查询优化器：如果我们将数据集的 `collect` 延迟到最后，Polars 将能够推断出需要哪些列和行，并在读取数据时尽早应用过滤器。对于包含元数据（例如特定行组中的最小值、最大值）的 Parquet 等文件格式，差异甚至可能更大，因为 Polars 可以根据过滤器和元数据跳过整个行组，而无需通过网络发送数据。

### GGUF 在 LM Studio 中的使用
https://huggingface.co/docs/hub/lmstudio.md