<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储桶集成

大多数工具已经可以通过以下两种方式之一使用存储桶：

- **基于 fsspec** 构建的 Python 库（pandas、Dask、PyArrow 等）通过 [⟦T15⟧ filesystem interface](/docs/huggingface_hub/guides/hf_file_system) 读取和写入 `hf://buckets/` 路径。
- **与 S3 通信的工具**（boto3、AWS CLI、rclone、DVC 等）针对 [S3-compatible API](./storage-buckets-s3) 工作。

本页面涵盖具有内置存储桶支持的库，以及常见数据库的示例。对于安装和卷安装，请参阅[Access Patterns](./storage-buckets-access)。

## 熊猫

```python
import pandas as pd

df = pd.read_parquet("hf://buckets/username/my-bucket/data.parquet")
df.to_parquet("hf://buckets/username/my-bucket/output.parquet")
```

## 达斯克

Dask 延迟读取和写入，每个文件一个分区，因此您可以直接从存储桶处理大于内存的数据集：

```python
import dask.dataframe as dd

df = dd.read_parquet("hf://buckets/username/my-bucket/data/")
df.to_parquet("hf://buckets/username/my-bucket/output/")
```

更多信息请参见[Dask on the Hub](./datasets-dask)。

## 愚蠢

Daft 原生支持 `hf://buckets/` 路径，默认启用 [Xet-accelerated reads](https://docs.daft.ai/en/stable/connectors/huggingface/)：

```python
import daft
from daft.io import IOConfig, HuggingFaceConfig
from huggingface_hub import get_token

io_config = IOConfig(hf=HuggingFaceConfig(token=get_token()))
df = daft.read_parquet("hf://buckets/username/my-bucket/data.parquet", io_config=io_config)
```

## PyArrow

```python
import pyarrow.parquet as pq

table = pq.read_table("hf://buckets/username/my-bucket/data.parquet")
```

## PySpark

安装[⟦T17⟧](https://github.com/huggingface/pyspark_huggingface)后：

```python
df = (
    spark.read.format("huggingface")
    .option("data_files", '["data.parquet"]')
    .load("buckets/username/my-bucket")
)
```

更多信息请参见[PySpark on the Hub](./datasets-spark)。

## 🤗 数据集

```python
from datasets import load_dataset

ds = load_dataset("buckets/username/my-bucket", data_files=["data.parquet"])
```

## 数据库

[datatrove](https://github.com/huggingface/datatrove) 读者和作者接受 `hf://buckets/` 路径。 `HuggingFaceBucketWriter` 使用 Xet 上传大量输出：

```python
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import HuggingFaceBucketWriter

reader = ParquetReader(data_folder="hf://buckets/username/my-bucket/raw/")
writer = HuggingFaceBucketWriter(bucket="username/my-bucket", prefix="filtered")
```

有关更多信息，请参阅[datatrove README](https://github.com/huggingface/datatrove#hf-storage-buckets)。

## 乐机器人

[LeRobot](https://huggingface.co/docs/lerobot/main/en/training_dataset_streaming) 可以从存储桶中传输训练数据集。该存储桶必须在其根部保存完整的数据集（`meta/`、Parquet 和视频文件）：

```bash
lerobot-train \
  --dataset.repo_id=username/my-bucket \
  --dataset.repo_type=bucket \
  --dataset.streaming=true \
  --policy.type=act
```

## 扩散器

`diffusers-cli run` 将其输出上传到`--push-to` 的存储桶。如果存储桶不存在，则创建该存储桶：```bash
diffusers-cli run --model black-forest-labs/FLUX.1-dev --dtype bf16 \
  --pipeline-kwargs '{"prompt": "an astronaut riding a horse"}' \
  --push-to hf://buckets/username/my-bucket/outputs
```

有关更多信息，请参阅[Diffusers CLI guide](https://huggingface.co/docs/diffusers/using-diffusers/cli)。

## 追踪

当[Trackio](https://huggingface.co/docs/trackio)登录到空间时，它将运行数据存储在根据空间ID创建的存储桶中。将 `bucket_id` 传递给 `trackio.init()` 以使用您选择的存储桶。

## 检查人工智能

[Inspect AI](https://inspect.aisi.org.uk/)可以将评估日志直接写入存储桶——将其日志目录指向`hf://buckets/`路径（需要`huggingface_hub>=1.6.0`）。首先创建存储桶并进行身份验证（Inspect 不会为您创建它）：

```bash
hf auth login
hf buckets create username/my-bucket --private

export INSPECT_LOG_DIR=hf://buckets/username/my-bucket/eval-logs
inspect eval popularity.py --model openai/gpt-4
inspect view
```

详情请参阅[Inspect's eval logs guide](https://inspect.aisi.org.uk/eval-logs.html#sec-hugging-face-storage-buckets)。

## 天空飞行员

[SkyPilot](https://docs.skypilot.co/) 跨 20 多个云、Kubernetes 和本地运行 AI 工作负载，并且可以使用 Hugging Face 存储作为后端 - 因此可以从每个云读取一个存储桶，而无需每个云复制。在 `file_mounts` 条目上设置 `store: hf` 以将存储桶挂载为可读写或将存储库挂载为只读：

```yaml
# qwen-sft.yaml — launch anywhere: sky launch qwen-sft.yaml --infra aws|gcp|...
resources:
  accelerators: H100:1

file_mounts:
  /base-model:
    source: hf://Qwen/Qwen2.5-3B           # model repo, read-only
    store: hf
    mode: MOUNT
  /data:
    source: hf://datasets/username/my-data@v1.0   # dataset repo, pinned to a tag, read-only
    store: hf
    mode: MOUNT
  /checkpoints:
    source: hf://buckets/username/qwen-sft   # bucket, read-write — checkpoints sync back
    store: hf
    mode: MOUNT

run: |
  python train.py --model /base-model --output_dir /checkpoints
```

验证一次 - `hf auth login` （或 `export HF_TOKEN=REDACTED 就是 SkyPilot 所需要的。它将您本地的 Hugging Face 令牌转发到每个云，因此存储桶和存储库安装会自动进行验证：

```bash
pip install "skypilot[huggingface]"
hf auth login                              # or: export HF_TOKEN=REDACTED
sky launch qwen-sft.yaml
```

如果您自己的 `run` 代码拉取门控存储库，请将 `--secret HF_TOKEN` 添加到启动命令中，以将令牌公开为环境变量。> [!提示]
> `MOUNT` 和 `MOUNT_CACHED` 与 `hf` 的行为相同，并使用 [hf-mount](https://github.com/huggingface/hf-mount) FUSE 后端，该后端需要 glibc ≥ 2.34 和 `/dev/fuse` 的基础映像。裸虚拟机云同时提供这两种功能。 SkyPilot 的默认 Kubernetes 映像附带较旧的 glibc，因此设置较新的 `image_id`（例如 `docker:mirror.gcr.io/ubuntu:22.04`）。当前环境要求请参见[SkyPilot storage docs](https://docs.skypilot.co/en/latest/reference/storage.html)。

请参阅 [SkyPilot + Hugging Face storage blog post](https://huggingface.co/blog/skypilot-hf-storage) 了解基准测试和完整演练。

## 文件系统操作

对于直接文件操作，`huggingface_hub`公开了预实例化的[filesystem object](/docs/huggingface_hub/guides/hf_file_system)、`hffs`：

```python
from huggingface_hub import hffs

with hffs.open("buckets/username/my-bucket/hello.txt", "w") as f:
    f.write("Hello world!")

hffs.cp("buckets/username/my-bucket/hello.txt", "buckets/username/my-bucket/hello2.txt")
hffs.rm("buckets/username/my-bucket/hello2.txt")
files = hffs.ls("buckets/username/my-bucket")
text_files = hffs.glob("buckets/username/my-bucket/*.txt")
```

## 鸭数据库

DuckDB 2.0 及更高版本通过 `httpfs` 扩展（并使用 `hf` 扩展进行写入）从 CLI 和每个客户端本地读取 `hf://buckets/` 路径：

```sql
SELECT * FROM 'hf://buckets/username/my-bucket/data/**/*.parquet' LIMIT 10;
COPY tbl TO 'hf://buckets/username/my-bucket/out.parquet';
```

有关私有存储桶和较旧的 DuckDB 版本，请参阅 [Query with DuckDB](./storage-buckets-access#query-with-duckdb)。

## 其他语言

[OpenDAL](https://opendal.apache.org/) 为 Rust、Java、Go、JavaScript 等提供了类似的文件系统接口。

## 即将推出

更多库即将提供本机 `hf://` URL 支持 - 包括 WebDataset。与此同时，所有这些现在都已通过 [S3-compatible API](./storage-buckets-s3) 发挥作用。

### marimo 空间
https://huggingface.co/docs/hub/spaces-sdks-docker-marimo.md