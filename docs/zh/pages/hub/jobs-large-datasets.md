<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 处理大数据集

每个作业都附带固定数量的本地磁盘，由其 [hardware flavor](./jobs-pricing#pricing) 设置（**临时存储**列，也由 `hf jobs hardware` 显示）。您不需要在该磁盘上安装整个数据集即可使用它：数据集和[Storage Buckets](./storage-buckets)可以直接从集线器读取 - 流式传输、查询或安装 - 因此单个作业可以处理远远大于其磁盘的数据。本页介绍了各个选项以及何时进行。

## 哪种方法？

- **适合磁盘** → 普通的 `load_dataset(...)` 按原样工作（或者只是选择更大的版本 - 最多 1 TB 的临时磁盘）。
- **迭代行、处理或训练** → [stream](#stream-the-dataset) 它。
- **过滤或列修剪扫描** → 使用 Polars 或 DuckDB 查询 [directly over ⟦T11⟧](#read-and-filter-over-hf)。
- **需要本地文件路径的工具** → [mount](#mount-a-dataset-model-or-bucket) 存储库并懒惰地读取它。
- **持久结果** → 将它们写入[Storage Bucket](#save-results)，以便它们在作业中幸存下来。

## 流式传输数据集

当您的代码使用这些示例时，流式传输会从 Hub 读取示例 — 无需下载，无需本地副本，并且可以扩展到多 TB 数据集。最近的版本使其成为[up to 100× more efficient](https://huggingface.co/blog/streaming-datasets)，在对许多工作人员进行培训时达到了与本地 SSD 相当的性能：

```python
from datasets import load_dataset

ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
for example in ds.take(1000):
    ...  # streams in as you iterate, nothing hits disk
```流式数据集是支持惰性 `.filter()`、`.map()`、`.shuffle(buffer_size=...)` 和 `.batch()` 的`IterableDataset`，并且可以直接传递到 PyTorch `DataLoader` 或 `Trainer` 以训练大于磁盘的数据。有关完整 API，请参阅 [Stream guide](/docs/datasets/stream)，有关单节点或分布式设置的端到端训练演练，请参阅 [Examples & Tutorials](./jobs-examples)。

流式处理也可以使用 Spark 在分布式设置中工作：

```python
import pyspark_huggingface

df = spark.read.format("huggingface").option("config", "sample-10BT").load("HuggingFaceFW/fineweb-edu")
```

生成的 Spark 数据帧是分布式的：每个工作线程都从自己的文件子集进行流式传输。有关读取、写入以及有效过滤行和列的示例，请参阅[Spark documentation](./datasets-spark)。

## 阅读并过滤`hf://`

许多数据库直接通过 `hf://` 路径读取 Hub 数据集 — **Polars**、**DuckDB** 和 **pandas** 都以本机方式扫描 Hub Parquet，将过滤器和列选择推送到扫描中，因此单个作业可以处理比内存或磁盘容纳的数据多得多的数据。此查询在默认 CPU 风格上大约四分钟内总结了约 28 GB 的 Parquet：

```python
import polars as pl

agg = (
    pl.scan_parquet("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet")
    .filter(pl.col("int_score") >= 4)
    .group_by("int_score")
    .agg(pl.len().alias("docs"), pl.col("token_count").sum().alias("tokens"))
    .sort("int_score")
    .collect()
)
print(agg)
```像这样的扫描是网络绑定的而不是内存绑定的，并且引擎在并行远程读取的积极程度方面有所不同，因此时间因库而异。两个实际后果：将 `--timeout` 设置为高于默认的 30 分钟以进行长时间扫描，并将工作分散到多个并行运行的作业中以加快速度。有关每个库的示例，请参阅 [Polars](./datasets-polars)、[DuckDB](./datasets-duckdb) 和 [pandas](./datasets-pandas)，有关通过 `hf://`（经过 [⟦T23⟧](/docs/huggingface_hub/guides/hf_file_system)）读取 **存储桶** 的信息，请参阅 [Python data tools](./storage-buckets-access#python-data-tools)。

## 挂载数据集、模型或存储桶

使用 `-v` / `--volume` 将存储库或存储桶作为本地路径挂载到作业中；当您的代码读取文件时，文件会通过网络延迟获取，因此任何读取本地文件的工具都可以正常工作：

```bash
hf jobs uv run --flavor cpu-upgrade \
  -v hf://datasets/HuggingFaceFW/fineweb-edu:/mnt/data \
  process.py
```

```python
# /// script
# dependencies = ["polars"]
# ///
# process.py — the mounted repo is just a directory of files
from pathlib import Path

import polars as pl

for path in Path("/mnt/data/sample/10BT").glob("*.parquet"):
    shard = pl.read_parquet(path)
    ...  # process one shard at a time, write results out
```

当整个文件（模型权重、音频或图像文件、档案）被消耗时，或者当工具仅接受文件路径时，挂载是自然的选择。对于大型多文件 Parquet 扫描，查询 [directly over ⟦T26⟧](#read-and-filter-over-hf) 通常比扫描装载快几倍。

数据集和模型以只读方式安装；存储桶是可读写的，这使得它们成为[save results](#save-results)的好地方。请参阅 [Configuration](./jobs-configuration#volumes) 了解完整的 `-v` 语法，并参阅 [bucket access patterns](./storage-buckets-access#volume-mounts-in-jobs-and-spaces) 了解详细信息。> [!提示]
> 通过挂载读取的文件缓存在作业的临时磁盘上，因此延迟读取（一次一个文件）可以保持较小的占用空间。使用 `hf jobs uv run` 运行本地脚本时，脚本目录会挂载在 `/data`，因此请将数据挂载到其他位置（例如 `/mnt/data`）。

## 保存结果

临时磁盘无法在作业中保留下来，因此请将您想要保留的任何内容写入[Storage Bucket](./storage-buckets)安装的读写器，或将其作为数据集推送到集线器。 DuckDB 可以通过 `hf://` 过滤源，并在一个核外查询中将匹配项直接写入已安装的存储桶，因此结果永远不必适合内存：

```bash
hf jobs uv run --flavor cpu-upgrade --timeout 1h \
  -v hf://buckets/username/my-output:/mnt/out \
  filter.py
```

```python
# /// script
# dependencies = ["duckdb"]
# ///
# filter.py — scan ~28 GB of Parquet, keep only the matching rows
import duckdb

duckdb.sql(
    """
    COPY (
        SELECT text, url, token_count
        FROM 'hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet'
        WHERE int_score >= 4 AND token_count >= 4000
    ) TO '/mnt/out/result.parquet' (FORMAT parquet)
    """
)
```

作业结束后，写入存储桶挂载路径下的文件仍会保留。要发布已处理的数据集，请使用[⟦T32⟧](/docs/datasets/upload_dataset)。

## 工作示例：查询 Common Crawl 而不下载它

Common Crawl 将其存档镜像到存储桶 [⟦T33⟧](https://huggingface.co/buckets/commoncrawl/commoncrawl) — 数百 TB。直接从 `hf://` 流式传输一个 WET（明文）分片，对其进行解析，并使用 DuckDB 进行查询；只有几 MB 传输，因为 gzip 是按顺序读取并提前停止的：

```python
# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub>=1.9", "fastwarc>=0.15", "duckdb>=1.0"]
# ///
import duckdb
from fastwarc.warc import ArchiveIterator, WarcRecordType
from huggingface_hub import hffs

WET = (
    "buckets/commoncrawl/commoncrawl/crawl-data/CC-MAIN-2026-17/"
    "segments/1775805908305.14/wet/"
    "CC-MAIN-20260410081153-20260410111153-00000.warc.wet.gz"
)

rows = []
with hffs.open(WET, "rb") as f:
    for rec in ArchiveIterator(f, record_types=WarcRecordType.conversion):
        lang = (rec.headers.get("WARC-Identified-Content-Language", "") or "und").split(",")[0]
        rows.append((rec.headers.get("WARC-Target-URI", ""), lang, len(rec.reader.read())))
        if len(rows) >= 5000:
            break

con = duckdb.connect()
con.execute("CREATE TABLE wet(url VARCHAR, lang VARCHAR, n_chars BIGINT)")
con.executemany("INSERT INTO wet VALUES (?,?,?)", rows)
con.sql("SELECT lang, count(*) AS docs FROM wet GROUP BY lang ORDER BY docs DESC LIMIT 10").show()
```

使用 `hf jobs uv run cc_wet.py` 运行它 — 它在默认 CPU 风格上大约一分钟即可完成并打印：

```
┌─────────┬───────┐
│  lang   │ docs  │
│ varchar │ int64 │
├─────────┼───────┤
│ eng     │  1974 │
│ zho     │   586 │
│ rus     │   434 │
│ jpn     │   244 │
│ …       │    …  │
└─────────┴───────┘
```

## 另请参阅- [Stream](/docs/datasets/stream) · [Streaming datasets: 100× more efficient](https://huggingface.co/blog/streaming-datasets)
- [Pricing & hardware](./jobs-pricing#pricing) — 每种口味的临时磁盘 · [Configuration](./jobs-configuration#volumes) — 卷
- [Storage Buckets](./storage-buckets)·[access patterns](./storage-buckets-access)·[integrations](./storage-buckets-integrations)

### 在拥抱脸部时使用 ML-Agents
https://huggingface.co/docs/hub/ml-agents.md