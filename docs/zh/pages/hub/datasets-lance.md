<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 兰斯

[Lance](https://lance.org) 是一种面向 AI 的开放式多模式 Lakehouse 表格式。您可以使用 Hugging Face 路径 (`hf://`) 访问 Hub 上的 Lance 数据集。这使您可以扫描和搜索 Hugging Face Hub 上的大型数据集，而无需在本地复制整个数据集。

## 开始使用

首先，pip install `pylance` 和 `pyarrow`：

```bash
pip install pylance pyarrow
```

## 为什么是兰斯？

- 针对 ML/AI 工作负载进行了优化：Lance 是一种现代柱状格式，专为快速随机访问而设计，而不影响扫描性能，使其可用于搜索、分析、培训、特征工程和更多用例。
- 多模式资产在 Lance 中以字节或二进制对象（“[blobs as files](https://lance.org/guide/blob/)”）的形式与嵌入和传统标量数据一起存储——这使得通过 Hub 更容易管理、共享和分发大型数据集。
- 索引是一等公民（格式本身本身）：Lance 提供快速、磁盘上、可扩展的 [vector](https://lance.org/quickstart/vector-search) 和 FTS 索引，它们位于 Hub 上的数据集旁边，因此您不仅可以共享数据，还可以共享嵌入和索引，而无需用户重新计算它们。- 灵活的模式和 [data evolution](https://lance.org/guide/data_evolution) 让您可以增量添加新功能/列（审核标签、嵌入等）**无需**需要重写整个表。

## 将所有数据存储在一个地方

在 Lance 中，您的多模式数据资产（图像、音频、视频）与标量元数据和嵌入一起存储为原始字节。这使得您可以轻松地在一个地方扫描和过滤数据集，而无需将多个存储系统拼接在一起。

## 使用 `datasets` 从集线器进行流式传输

使用`load_dataset(..., streaming=True)`扫描并迭代数据，无需下载到本地。

```python
from datasets import load_dataset

# Return as a Hugging Face dataset
ds = load_dataset(
    "lance-format/laion-1m",
    split="train",
    streaming=True
)
# Take first three rows
for row in ds.take(3):
    print(row["caption"])
```

流式传输非常适合对元数据进行采样，以了解您拥有的内容。对于矢量搜索或处理大型二进制 blob，您可以使用 Lance `dataset` API，如下所述。

> [!警告]
> 流式传输对于简单标量元数据采样速度很快，但对于嵌入或大型多模式资产则不那么快。要处理大型数据集，建议扫描元数据，识别所需的子集，并在本地下载数据集的该部分，以避免面临 Hub 速率限制：
> `hf download lance-format/laion-1m --repo-type dataset --local-dir ./laion`

## 使用 `lance.dataset` 从集线器进行流式传输您还可以使用 `hf://` 路径说明符扫描存储在 Hugging Face Hub 上的 Lance 数据集。这会扫描远程数据集，而不需要您将其下载到本地。使用 Lance `dataset` API，可以非常简单地设置限制、过滤器和预测以仅获取您需要的数据。

```python
import lance

# Return as a Lance dataset
ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

scanner = ds.scanner(
    columns=["caption", "url", "similarity"],
    limit=5
)

rows = scanner.to_table().to_pylist()
for row in rows:
    print(row)
```

## 使用二进制资产

下面的示例展示了如何从 Lance 数据集中检索图像作为 `image` 列中的原始 JPEG 字节，并在下游使用。
使用 `ds.take` 获取字节并将其写入磁盘，以便您可以在其他地方使用它们。

```python
import lance
from pathlib import Path

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

dir_name = "laion_samples"
Path(dir_name).mkdir(exist_ok=True)

rows = ds.take([0, 1], columns=["image", "caption"]).to_pylist()
for idx, row in enumerate(rows):
    with open(f"{dir_name}/{idx}.jpg", "wb") as f:
        f.write(row["image"])
        print(f"Wrote image with caption: {row['caption']}")
```

## 将子集写入新的 Lance 数据集

使用大型数据集？运行过滤扫描以从 Hub 中选择行的子集并将其具体化到本地 Lance 数据集中非常简单。

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")
scanner = ds.scanner(
    columns=["image", "caption", "width", "height"],
    filter="width >= 200 AND height >= 100",
    limit=10,
)
subset = scanner.to_table()

lance.write_dataset(subset, "./laion_subset")
```

## 创建索引

如果您的数据集还没有与之关联的索引，您可以在本地下载后创建一个索引。

```python
# ds is a local Lance dataset
ds.create_index(
    "img_emb",
    index_type="IVF_PQ",
    num_partitions=256,
    num_sub_vectors=96,
    replace=True,
)
```

有关更详细的示例，请参阅关于向量索引创建的[Lance docs](https://lance.org/quickstart/vector-search/)。创建向量索引后，您可以通过嵌入对数据运行相似性搜索。

## 向量搜索因为索引是 Lance 的一等公民，所以您不仅可以存储数据，还可以存储嵌入和索引，并**直接在 Hub 上**查询它们。只需使用`describe_indices()`方法即可列出数据集的索引信息。如果数据集中不存在索引，您可以使用`lance.write_dataset()`编写数据集的本地版本，并使用[LanceDataset.create_index](https://lance-format.github.io/lance-python-doc/all-modules.html#lance.dataset.LanceDataset.create_index)创建满足您需要的索引。

下面的示例显示了一个数据集，我们已经在 `img_emb` 字段上拥有向量索引：

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

print(ds.list_indices())

# Returns
# [
#   IndexDescription(
#       name=img_emb_idx,
#       type_url=/lance.table.VectorIndexDetails,
#       num_rows_indexed=1209588,
#       fields=[15],
#       field_names=["img_emb"],
#       num_segments=1
#.   )
# ]
```

您可以直接在远程数据集上运行矢量搜索查询，而无需下载它（或者，如果您愿意，可以在本地下载数据集并创建新索引）。下面的示例展示了如何使用图像嵌入作为查询向量对向量索引运行最近邻搜索。

```python
import lance
import pyarrow as pa

ds = lance.dataset("hf://datasets/lance-format/laion-1m/data/train.lance")

emb_field = ds.schema.field("img_emb")
ref = ds.take([0], columns=["img_emb"]).to_pylist()[0]["img_emb"]
query = pa.array([ref], type=emb_field.type)

neighbors = ds.scanner(
    nearest={
        "column": emb_field.name,
        "q": query[0],
        "k": 6,
        "nprobes": 16,
        "refine_factor": 30,
    },
    columns=["caption", "url", "similarity"],
).to_table().to_pylist()
```

> [!注意]
> 设置较大的 `k` 或 `nprobes` 值，或一次发送大量查询可能会达到 Hub 速率限制。对于大量使用，请在本地下载数据集（或其子集）并将 Lance 指向本地路径以避免限制。

## 数据集演变Lance 最强大的功能之一是灵活、零成本的数据演化，这意味着您可以轻松添加派生列，**无需**重写原始表。对于包含大量大 blob 的非常大的表，I/O 节省可能非常显着。如果您正在试验 ML/AI 工程任务的数据，并且经常发现自己添加新功能、嵌入或派生元数据，则此功能非常相关。

下面的示例展示了如何添加派生的 `moderation_label` 列，根据现有分数列将图像标记为 `NSFW`。进行此更改时，回填新列**仅**写入新列数据，而不会触及原始图像 blob 或其他列中的数据。您还可以选择仅添加新列架构而不回填任何数据。

```python
import lance
import pyarrow as pa

# Assumes you ran the export to Lance example above to store a local subset of the data
local_ds = lance.dataset("./laion_subset")

# schema only (data to be added later)
local_ds.add_columns(pa.field("moderation_label", pa.string()))

# with data backfill
local_ds.add_columns(
    {
        "moderation_label": "case WHEN \"NSFW\" > 0.5 THEN 'review' ELSE 'ok' END"
    }
)
```

请参阅 [data evolution](https://lance.org/guide/data_evolution/) 上的 Lance 文档，了解如何更改和删除 Lance 数据集中的列。

## 处理视频 blob

Lance 表还支持大型内联视频 blob。 `OpenVid-1M` 数据集（来自 [this paper](https://arxiv.org/abs/2407.02371)）包含高质量、富有表现力的视频及其字幕。视频数据存储在Hub上以下Lance数据集的`video_blob`列中。

```python
import lance

lance_ds = lance.dataset("hf://datasets/lance-format/Openvid-1M/data/train.lance")
blob_file = lance_ds.take_blobs("video_blob", ids=[0])[0]
video_bytes = blob_file.read()
```与其他数据格式不同，大型多模式二进制对象（blob）是 Lance 中的一等公民。 [blob API](https://lance.org/guide/blob/) 提供了一个高级 API 来存储和检索 Lance 数据集中的大型 blob。以下示例演示如何在不加载较重的视频 blob 的情况下高效浏览元数据，然后按需获取相关视频 blob。

```python
import lance

ds = lance.dataset("hf://datasets/lance-format/Openvid-1M/data/train.lance")

# 1. Browse metadata without loading video blobs.
metadata = ds.scanner(
    columns=["caption", "aesthetic_score"],
    filter="aesthetic_score >= 4.5",
    limit=2,
).to_table().to_pylist()

# 2. Fetch a single video blob by row index.
selected_index = 0
blob_file = ds.take_blobs("video_blob", ids=[selected_index])[0]
with open("video_0.mp4", "wb") as f:
    f.write(blob_file.read())
```

## 准备训练数据

训练是 Lance 的快速随机访问和扫描性能发挥作用的另一个领域。您可以使用 Lance 数据集作为训练数据的存储机制，将其打乱并加载到批次中，作为训练管道的一部分。

Lance 中的 blob API 与 `torchcodec` 兼容，因此您可以轻松地将视频 blob 解码为 `torch` 张量：

```python
from torchcodec.decoders import VideoDecoder
decoder = VideoDecoder(blob_file)
tensor = decoder[0]  # uint8 tensor of shape [C, H, W]
```

有关高效解码视频的更多功能，请参阅[torchcodec docs](https://docs.pytorch.org/torchcodec/stable/generated/torchcodec.decoders.VideoDecoder.html)。

此外，您还可以查看[Lance documentation](https://lance.org/examples/python/clip_training/)以获取更多将图像数据加载到`torchvision`中以训练您自己的图像模型的示例。

## 探索更多 Lance 数据集Lance 是一种开放格式，原生支持多模式 blob 以及传统表格数据。
通过 Hugging Face Hub 集成，您可以轻松处理图像、音频、视频、文本、嵌入和
标量元数据全部集中在一处。

在 [Hugging Face Hub](https://huggingface.co/datasets?format=format:lance) 上探索更多 Lance 数据集，
并与社区中的其他人分享您自己的 Lance 数据集！
您可以访问[lance.org](https://lance.org/integrations/huggingface/)获取更多代码片段和示例。

### 团队和企业计划
https://huggingface.co/docs/hub/enterprise.md