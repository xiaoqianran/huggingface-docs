<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 转换你的数据集

在此页面上，我们将指导您完成进行数据分析时最常用的一些操作。这只是 Polars 中可能实现的功能的一小部分。欲了解更多信息，请访问[Documentation](https://docs.pola.rs/)。

在本例中，我们将使用 [Common Crawl statistics](https://huggingface.co/datasets/commoncrawl/statistics) 数据集。这些统计数据包括：页面数量、顶级域名的分布、抓取重叠等。有关更详细的信息和图表，请访问他们的[official statistics page](https://commoncrawl.github.io/cc-crawl-statistics/plots/tlds)。

## 阅读

```python
import polars as pl

df = pl.read_csv(
    "hf://datasets/commoncrawl/statistics/tlds.csv",
    try_parse_dates=True,
)
df.head(3)
```

```bash
┌─────┬────────┬───────────────────┬────────────┬───┬───────┬──────┬───────┬─────────┐
│     ┆ suffix ┆ crawl             ┆ date       ┆ … ┆ pages ┆ urls ┆ hosts ┆ domains │
│ --- ┆ ---    ┆ ---               ┆ ---        ┆   ┆ ---   ┆ ---  ┆ ---   ┆ ---     │
│ i64 ┆ str    ┆ str               ┆ date       ┆   ┆ i64   ┆ i64  ┆ f64   ┆ f64     │
╞═════╪════════╪═══════════════════╪════════════╪═══╪═══════╪══════╪═══════╪═════════╡
│ 0   ┆ a.se   ┆ CC-MAIN-2008-2009 ┆ 2009-01-12 ┆ … ┆ 18    ┆ 18   ┆ 1.0   ┆ 1.0     │
│ 1   ┆ a.se   ┆ CC-MAIN-2009-2010 ┆ 2010-09-25 ┆ … ┆ 3462  ┆ 3259 ┆ 166.0 ┆ 151.0   │
│ 2   ┆ a.se   ┆ CC-MAIN-2012      ┆ 2012-11-02 ┆ … ┆ 6957  ┆ 6794 ┆ 172.0 ┆ 150.0   │
└─────┴────────┴───────────────────┴────────────┴───┴───────┴──────┴───────┴─────────┘
```

## 选择列

数据集包含一些我们不需要的列。要删除它们，我们将使用 `select` 方法：

```python
df = df.select("suffix", "date", "tld", "pages", "domains")
df.head(3)
```

```bash
┌────────┬───────────────────┬────────────┬─────┬───────┬─────────┐
│ suffix ┆ crawl             ┆ date       ┆ tld ┆ pages ┆ domains │
│ ---    ┆ ---               ┆ ---        ┆ --- ┆ ---   ┆ ---     │
│ str    ┆ str               ┆ date       ┆ str ┆ i64   ┆ f64     │
╞════════╪═══════════════════╪════════════╪═════╪═══════╪═════════╡
│ a.se   ┆ CC-MAIN-2008-2009 ┆ 2009-01-12 ┆ se  ┆ 18    ┆ 1.0     │
│ a.se   ┆ CC-MAIN-2009-2010 ┆ 2010-09-25 ┆ se  ┆ 3462  ┆ 151.0   │
│ a.se   ┆ CC-MAIN-2012      ┆ 2012-11-02 ┆ se  ┆ 6957  ┆ 150.0   │
└────────┴───────────────────┴────────────┴─────┴───────┴─────────┘
```

## 过滤

我们可以使用`filter`方法过滤数据集。此方法接受复杂的表达式，但让我们从基于抓取日期的过滤开始简单：

```python
import datetime

df = df.filter(pl.col("date") >= datetime.date(2020, 1, 1))
```

您可以使用 `&` 或 `|` 运算符组合多个谓词：

```python
df = df.filter(
    (pl.col("date") >= datetime.date(2020, 1, 1)) |
    pl.col("crawl").str.contains("CC")
)
```

## 转变

为了向数据集添加新列，请使用`with_columns`。在下面的示例中，我们计算每个域的总页面数，并使用 `alias` 方法添加新列 `pages_per_domain`。 `with_columns` 中的整个语句称为表达式。在 [Polars user guide](https://docs.pola.rs/user-guide/expressions/) 中阅读有关表达式以及如何使用它们的更多信息

```python
df = df.with_columns(
    (pl.col("pages") / pl.col("domains")).alias("pages_per_domain")
)
df.sample(3)
```

```bash
┌────────┬─────────────────┬────────────┬─────┬───────┬─────────┬──────────────────┐
│ suffix ┆ crawl           ┆ date       ┆ tld ┆ pages ┆ domains ┆ pages_per_domain │
│ ---    ┆ ---             ┆ ---        ┆ --- ┆ ---   ┆ ---     ┆ ---              │
│ str    ┆ str             ┆ date       ┆ str ┆ i64   ┆ f64     ┆ f64              │
╞════════╪═════════════════╪════════════╪═════╪═══════╪═════════╪══════════════════╡
│ net.bt ┆ CC-MAIN-2014-41 ┆ 2014-10-06 ┆ bt  ┆ 4     ┆ 1.0     ┆ 4.0              │
│ org.mk ┆ CC-MAIN-2016-44 ┆ 2016-10-31 ┆ mk  ┆ 1445  ┆ 430.0   ┆ 3.360465         │
│ com.lc ┆ CC-MAIN-2016-44 ┆ 2016-10-31 ┆ lc  ┆ 1     ┆ 1.0     ┆ 1.0              │
└────────┴─────────────────┴────────────┴─────┴───────┴─────────┴──────────────────┘
```## 聚合与排序

为了将数据聚合在一起，您可以使用 `group_by`、`agg` 和 `sort` 方法。在聚合上下文中，您可以组合表达式来创建强大且易于阅读的语句。

首先，我们将每个抓取日期的所有数据聚合到顶级域`tld`：

```python
df = df.group_by("tld", "date").agg(
    pl.col("pages").sum(),
    pl.col("domains").sum(),
)
```

现在我们可以计算每个顶级域的几个统计数据：

- 独特抓取日期的数量
- 抓取期间的平均域名数量
- 页数平均增长率

```python
df = df.group_by("tld").agg(
    pl.col("date").unique().count().alias("number_of_scrapes"),
    pl.col("domains").mean().alias("avg_number_of_domains"),
    pl.col("pages").sort_by("date").pct_change().mean().alias("avg_page_growth_rate"),
)
df = df.sort("avg_number_of_domains", descending=True)
df.head(10)
```

```bash
┌─────┬───────────────────┬───────────────────────┬─────────────────────────────────┐
│ tld ┆ number_of_scrapes ┆ avg_number_of_domains ┆ avg_percent_change_in_number_o… │
│ --- ┆ ---               ┆ ---                   ┆ ---                             │
│ str ┆ u32               ┆ f64                   ┆ f64                             │
╞═════╪═══════════════════╪═══════════════════════╪═════════════════════════════════╡
│ com ┆ 101               ┆ 1.9571e7              ┆ 0.022182                        │
│ de  ┆ 101               ┆ 1.8633e6              ┆ 0.5232                          │
│ org ┆ 101               ┆ 1.5049e6              ┆ 0.019604                        │
│ net ┆ 101               ┆ 1.5020e6              ┆ 0.021002                        │
│ cn  ┆ 101               ┆ 1.1101e6              ┆ 0.281726                        │
│ ru  ┆ 101               ┆ 1.0561e6              ┆ 0.416303                        │
│ uk  ┆ 101               ┆ 827453.732673         ┆ 0.065299                        │
│ nl  ┆ 101               ┆ 710492.623762         ┆ 1.040096                        │
│ fr  ┆ 101               ┆ 615471.594059         ┆ 0.419181                        │
│ jp  ┆ 101               ┆ 615391.455446         ┆ 0.246162                        │
└─────┴───────────────────┴───────────────────────┴─────────────────────────────────┘
```

### 空间 Livebook
https://huggingface.co/docs/hub/spaces-sdks-docker-livebook.md