<!-- huggingface-docs: machine-translated zh-CN from English source -->

#泽芬

[Zephon](https://github.com/datologyai/zephon)是一个用于基础模型训练的数据加载器，由[DatologyAI](https://huggingface.co/DatologyAI)构建。
它在训练运行时对数据进行标记、打包和混合，并且当您更改 GPU 数量（弹性决定论）时，它会提供相同的批次。
Zephon 使用 `hf://` 路径直接从 Hugging Face Hub 读取数据集。

## 安装

`hf` 额外添加了对 Hub 数据集和 Parquet 文件的支持。下面的示例也需要 `transformers` 进行标记化：

```bash
pip install "zephon[hf]" transformers
```

## 从 Hub 加载数据集

将 `hf://` 路径传递到 `Dataset.from_path`。该路径命名数据集、配置和拆分，例如 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) 的 `sample-10BT` 配置的 `train` 拆分。
Zephon通过其[shard cache](https://datologyai.github.io/zephon/datasets/storage_backends_and_shard_cache.html)读取Hub数据集，因此必须启用缓存。将缓存放在快速本地磁盘上并设置大小限制：

```python
from zephon import Pipeline
from zephon.io import CacheOptions, Dataset, StoreOptions
from zephon.work import MixtureSpec, StaticMixtureWorkSource

dataset = Dataset.from_path("fineweb_edu", "hf://HuggingFaceFW/fineweb-edu/sample-10BT/train")
work_source = StaticMixtureWorkSource(
    datasets=[dataset],
    mixture=MixtureSpec({"fineweb_edu": 1.0}),
    seed=42,
)
cache = CacheOptions(enabled=True, root="/tmp/zephon-cache", limit_bytes=50 << 30)  # 50 GiB

pipeline = (
    Pipeline(work_source)
    .options(io_options=StoreOptions(cache=cache))
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=2049, algorithm="wrap")
    .batch(microbatch_size=8)
)

for batch in pipeline:
    tensors = batch.to_training(return_labels=True)
    print(tensors["input_ids"].shape, tensors["labels"].shape)  # (8, 2048) (8, 2048)
    break
```

`tokenize` 将 `text` 列转换为令牌 ID。 `pack_flat` 将标记化文档连接到 `max_length` 标记序列中。将 `max_length` 设置为上下文长度加一：使用 `return_labels=True`、`to_training` 将标签移动一位，因此 `input_ids` 和 `labels` 都有上下文长度。

`Pipeline` 是一个 Python 可迭代对象，因此您可以在任何训练循环中使用它。 Zephon 在读取之前会下载完整的分片，因此第一批可能需要一些时间。要读取私有或门控数据集，请先使用您的 Hugging Face 帐户登录，例如使用 `hf auth login`。 Zephon 还读取 `HF_TOKEN` 环境变量。

## 路径格式

Zephon 的 Hub 数据集的 `hf://` 路径具有以下格式：

```
hf://{org}/{name}[@[{revision}][~{source}]]/[{config}/]{split}
```

- `revision`：分支、标签或提交。默认为`main`。 Zephon 在加载数据集时固定提交的路径，因此所有等级都会读取相同的文件。
- `source`：`original`读取上传到存储库的文件，`parquet`读取Hub的[automatic Parquet conversion](https://huggingface.co/docs/dataset-viewer/parquet)。将来源放在`@`之后，例如`@~parquet`。默认情况下，Zephon 在支持格式时读取上传的文件，如果不支持则使用 Parquet 转换。
- `config`：数据集配置。当数据集具有多个配置时包含它。|路径|阅读 |
| ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `hf://HuggingFaceFW/fineweb-edu/sample-10BT/train` | `sample-10BT` 配置的 `train` 拆分，来自 `main` 分支 |
| `hf://HuggingFaceFW/fineweb-edu@v1.4.0/sample-10BT/train` |与 `v1.4.0` 分支相同的分裂 |
| `hf://yahma/alpaca-cleaned@~parquet/train` |自动 Parquet 转换中的 `train` 分割 |

> [!警告]
> Zephon 路径命名数据集和分割，而不是文件。它们不使用 pandas、Polars 和 DuckDB 使用的 `hf://datasets/...` 文件路径。使用`hf://HuggingFaceFW/fineweb-edu/sample-10BT/train`，而不是`hf://datasets/HuggingFaceFW/fineweb-edu/sample-10BT/train`。

## 混合数据集

要在多个 Hub 数据集上进行训练，请将它们添加到工作源中，并注明每个数据集的比例。此示例将 FineWeb-Edu 与英语维基百科混合在一起，目标是通过标记实现 70/30 的混合：

```python
from zephon import Pipeline
from zephon.io import CacheOptions, Dataset, StoreOptions
from zephon.work import MixtureSpec, StaticMixtureWorkSource, TokenEstimation

work_source = StaticMixtureWorkSource(
    datasets=[
        Dataset.from_path("fineweb_edu", "hf://HuggingFaceFW/fineweb-edu/sample-10BT/train"),
        Dataset.from_path("wikipedia", "hf://wikimedia/wikipedia/20231101.en/train"),
    ],
    mixture=MixtureSpec({"fineweb_edu": 0.7, "wikipedia": 0.3}),
    seed=42,
    token_estimation=TokenEstimation(primer="measure", calibration_samples=2048, calibration_shards_max=4),
)
cache = CacheOptions(enabled=True, root="/tmp/zephon-cache", limit_bytes=50 << 30)  # 50 GiB

pipeline = (
    Pipeline(work_source)
    .options(io_options=StoreOptions(cache=cache))
    .tokenize(tokenizer_id="gpt2", field="text")
    .pack_flat(max_length=2049, algorithm="wrap")
    .batch(microbatch_size=8)
)
```

要按样品混合，请省略 `token_estimation`。

## 了解更多

- [Zephon documentation](https://datologyai.github.io/zephon/)
- [Launch blog post](https://www.datologyai.com/blog/zephon)
- [Zephon paper](https://arxiv.org/abs/2610.03087)
- [DatologyAI on the Hub](https://huggingface.co/DatologyAI)

### 搜索
https://huggingface.co/docs/hub/search.md