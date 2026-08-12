<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 流数据集

## 集成库

如果 Hub 上的数据集与允许从 Hugging Face 进行流式传输的[supported library](./datasets-libraries) 兼容，则只需几行即可完成数据集的流式传输。有关访问数据集的信息，您可以单击数据集页面上的“使用此数据集”按钮以了解如何执行此操作。例如，[⟦T8⟧](https://huggingface.co/datasets/knkarthick/samsum?library=datasets) 显示了如何使用下面的 `datasets` 执行此操作。

## 使用 Hugging Face 客户端库

您可以使用 [⟦T10⟧](/docs/huggingface_hub) 库来创建、删除和访问存储库中的文件。例如，要在 Python 中流式传输 `allenai/c4` 数据集，只需安装该库（我们建议使用最新版本）并运行以下代码。

```bash
pip install -U huggingface_hub
```
```python
from huggingface_hub import hffs

repo_id = "allenai/c4"
path_in_repo = "en/c4-train.00000-of-01024.json.gz"

# Stream the file
with hffs.open(f"datasets/{repo_id}/{path_in_repo}", "r", compression="gzip") as f:
    print(f.readline())  # read only the first line
    # {"text":"Beginners BBQ Class Taking Place in Missoula!...}
```

请参阅[⟦T12⟧ documentation](https://huggingface.co/docs/huggingface_hub/en/guides/hf_file_system)了解更多信息。

您还可以将其集成到您自己的库中！例如，您可以使用 Pandas 批量快速传输 CSV 数据集。
```py
import pandas as pd

repo_id = "YOUR_REPO_ID"
path_in_repo = "data.csv"

batch_size = 5

# Stream the file
with hffs.open(f"datasets/{repo_id}/{path_in_repo}") as f:
    for df in pd.read_csv(f, iterator=True, chunksize=batch_size):  # read 5 lines at a time
        print(len(df))  # 5
```

流式传输对于逐步读取 Hugging Face 上的大文件或仅读取一小部分特别有用。
例如，`tarfile`可以迭代TAR存档的文件，`zipfile`可以从ZIP存档读取文件，`pyarrow`可以访问Parquet文件的行组。

> [!提示]
> [OpenDAL](https://github.com/apache/opendal) 中有一个等效的 Rust 文件系统实现。

## 使用卷曲由于 Hub 上的所有文件都可以通过 HTTP 获取，因此您可以使用 `cURL` 流式传输文件：

```bash
>>> curl -L https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv | head -n 5
"act","prompt"
"An Ethereum Developer","Imagine you are an experienced Ethereum developer tasked with creating...
"SEO Prompt","Using WebPilot, create an outline for an article that will be 2,000 words on the ...
"Linux Terminal","I want you to act as a linux terminal. I will type commands and you will repl...
"English Translator and Improver","I want you to act as an English translator, spelling correct...
```

使用范围请求访问文件的特定部分：

```bash
>>> curl -r 40-88 -L https://huggingface.co/datasets/fka/awesome-chatgpt-prompts/resolve/main/prompts.csv
Imagine you are an experienced Ethereum developer
```

使用 [access token](https://huggingface.co/docs/hub/en/security-tokens) 从私有存储库进行流式传输：

```bash
>>> export HF_TOKEN=hf_xxx
>>> curl -H "Authorization: Bearer $HF_TOKEN" -L https://huggingface.co/...
```

## 流式镶木地板

Parquet 是 AI 数据集的绝佳格式。它提供良好的压缩、用于高效处理和投影的柱状结构以及用于快速过滤的多级元数据，并且适用于各种规模的数据集。

Parquet 文件分为行组，每个行组通常约为 100MB。这使得数据加载器和数据处理框架能够逐步流式传输数据，并在行组上进行迭代。

行组内部是单独的列，这些列被分为多个页面。页是大约 1MB 的压缩块，其中包含实际数据。

### 流行组

使用 PyArrow 在 Hugging Face 上流式传输 Parquet 文件中的行组：

```python
import pyarrow.parquet as pq

repo_id = "HuggingFaceFW/finewiki"
path_in_repo = "data/enwiki/000_00000.parquet"

# Stream the Parquet file row group per row group
with pq.ParquetFile(f"hf://datasets/{repo_id}/{path_in_repo}") as pf:
    for row_group_idx in range(pf.num_row_groups):
        row_group_table = pf.read_row_group(row_group_idx)
        df = row_group_table.to_pandas()
```

> [!提示]
> PyArrow 支持开箱即用的 `hf://` 路径并自动使用 `HfFileSystem`

在 [PyArrow documentation](./datasets-pyarrow) 中查找更多信息。

### 高效的随机访问行组进一步分为列，列又分为页。页面通常约为 1MB，是 Parquet 中最小的数据单元，因为这是应用压缩的地方。访问页面可以加载特定行，而无需加载完整的行组，如果 Parquet 文件具有页面索引，则可以实现这一点。然而，并非每个 Parquet 框架都支持页面级别的读取。例如，PyArrow 不会，但 Rust 中的 `parquet` 箱会：

```rust
use std::sync::Arc;
use object_store::path::Path;
use object_store_opendal::OpendalStore;
use opendal::services::Huggingface;
use opendal::Operator;
use parquet::arrow::async_reader::ParquetObjectReader;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use futures::TryStreamExt;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_id = "HuggingFaceFW/finewiki";
    let path_in_repo = Path::from("data/enwiki/000_00000.parquet");
    let offset = 0;
    let limit = 10;

    let builder = Huggingface::default().repo_type("dataset").repo_id(repo_id);
    let operator = Operator::new(builder)?.finish();
    let store = Arc::new(OpendalStore::new(operator));
    let reader = ParquetObjectReader::new(store, path_in_repo.clone());
    let batch_stream =
        ParquetRecordBatchStreamBuilder::new(reader).await?
            .with_offset(offset as usize)
            .with_limit(limit as usize)
            .build()?;
    let results = batch_stream.try_collect::<Vec<_>>().await?;
    println!("Read {} batches", results.len());
    Ok(())
}
```

> [!提示]
> 在 Rust 中，我们使用 OpenDAL 的 `Huggingface` 服务，相当于 python 中的 `HfFileSystem`

在 PyArrow 中传递 `write_page_index=True` 以包含实现高效随机访问的页面索引。
它特别将“offset_index_offset”和“offset_index_length”添加到 Parquet 列中，您可以在 [Parquet metadata viewer on Hugging Face](https://huggingface.co/blog/cfahlgren1/intro-to-parquet-format) 中看到。
页面索引还可以加快 [Hugging Face Dataset Viewer](https://huggingface.co/docs/dataset-viewer) 的速度，并允许其显示数据而不受行组大小限制。

### GitHub 操作
https://huggingface.co/docs/hub/repositories-github-actions.md