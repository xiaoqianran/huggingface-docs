<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 达斯克

[Dask](https://www.dask.org/?utm_source=hf-docs) 是一个并行分布式计算库，可扩展现有的 Python 和 PyData 生态系统。

特别是，我们可以使用 [Dask DataFrame](https://docs.dask.org/en/stable/dataframe.html?utm_source=hf-docs) 来扩展 pandas 工作流程。 Dask DataFrame 并行 pandas 来处理大型表格数据。它密切反映了 pandas API，使得从单个数据集测试到处理完整数据集的转换变得简单。 Dask 对于​​ Hugging Face 数据集的默认格式 Parquet 特别有效，因为它支持丰富的数据类型、高效的列式过滤和压缩。

Dask 的一个很好的实际用例是以分布式方式在数据集上运行数据处理或模型推理。例如，请参阅[Coiled's](https://www.coiled.io/?utm_source=hf-docs)关于[Scaling AI-Based Data Processing with Hugging Face + Dask](https://huggingface.co/blog/dask-scaling)的优秀博客文章。

## 读写

由于 Dask 使用 [fsspec](https://filesystem-spec.readthedocs.io) 来读写远程数据，因此您可以使用 Hugging Face 路径 ([⟦T13⟧](/docs/huggingface_hub/guides/hf_file_system#integrations)) 在 Hub 上读写数据。

首先你需要[Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)，例如使用：

```
hf auth login
```

然后你可以[Create a dataset repository](/docs/huggingface_hub/quick-start#create-a-repository)，例如使用：

```python
from huggingface_hub import HfApi

HfApi().create_repo(repo_id="username/my_dataset", repo_type="dataset")
```

最后，您可以在Dask中使用[Hugging Face paths](/docs/huggingface_hub/guides/hf_file_system#integrations)。
Dask DataFrame 支持在 Hugging Face 上分布式写入 Parquet，它使用提交来跟踪数据集更改：

```python
import dask.dataframe as dd

df.to_parquet("hf://datasets/username/my_dataset")

# or write in separate directories if the dataset has train/validation/test splits
df_train.to_parquet("hf://datasets/username/my_dataset/train")
df_valid.to_parquet("hf://datasets/username/my_dataset/validation")
df_test .to_parquet("hf://datasets/username/my_dataset/test")
```由于这会为每个文件创建一次提交，因此建议在上传后压缩历史记录：

```python
from huggingface_hub import HfApi

HfApi().super_squash_history(repo_id=repo_id, repo_type="dataset")
```

这将创建一个数据集存储库`username/my_dataset`，其中包含 Parquet 格式的 Dask 数据集。
您可以稍后重新加载：

```python
import dask.dataframe as dd

df = dd.read_parquet("hf://datasets/username/my_dataset")

# or read from separate directories if the dataset has train/validation/test splits
df_train = dd.read_parquet("hf://datasets/username/my_dataset/train")
df_valid = dd.read_parquet("hf://datasets/username/my_dataset/validation")
df_test  = dd.read_parquet("hf://datasets/username/my_dataset/test")
```

有关 Hugging Face 路径及其实现方式的更多信息，请参阅[the client library's documentation on the HfFileSystem](/docs/huggingface_hub/guides/hf_file_system)。

## 处理数据

要使用 Dask 并行处理数据集，您可以首先为 pandas DataFrame 或 Series 定义数据处理函数，然后使用 Dask `map_partitions` 函数将此函数并行应用于数据集的所有分区：

```python
def dummy_count_words(texts):
    return pd.Series([len(text.split(" ")) for text in texts])
```

或使用 pandas 字符串方法的类似函数（更快）：

```python
def dummy_count_words(texts):
    return texts.str.count(" ")
```

在 pandas 中，您可以在文本列上使用此函数：

```python
# pandas API
df["num_words"] = dummy_count_words(df.text)
```

在 Dask 中，您可以在每个分区上运行此函数：

```python
# Dask API: run the function on every partition
df["num_words"] = df.text.map_partitions(dummy_count_words, meta=int)
```

请注意，您还需要提供 `meta`，它是函数输出中 pandas Series 或 DataFrame 的类型。
这是必需的，因为 Dask DataFrame 使用惰性 API。由于 Dask 仅在调用 `.compute()` 后才会运行数据处理，因此需要
同时使用 `meta` 参数了解新列的类型。

## 谓词和投影下推从 Hugging Face 读取 Parquet 数据时，Dask 会自动利用 Parquet 文件中的元数据跳过整个文件或行组（如果不需要）。例如，如果您对 Parquet 格式的 Hugging Face 数据集应用过滤器（谓词），或者选择列的子集（投影），Dask 将读取 Parquet 文件的元数据以丢弃不需要的部分，而无需下载它们。

这要归功于支持查询优化的[reimplementation of the Dask DataFrame API](https://docs.coiled.io/blog/dask-dataframe-is-fast.html?utm_source=hf-docs)，这使得 Dask 更快、更健壮。

例如，FineWeb-Edu 的这个子集包含许多 Parquet 文件。如果您可以过滤数据集以保留最近的 CC 转储中的文本，Dask 将跳过大部分文件并仅下载与过滤器匹配的数据：

```python
import dask.dataframe as dd

df = dd.read_parquet("hf://datasets/HuggingFaceFW/fineweb-edu/sample/10BT/*.parquet")

# Dask will skip the files or row groups that don't
# match the query without downloading them.
df = df[df.dump >= "CC-MAIN-2023"]
```

Dask 还将只读取计算所需的列并跳过其余的列。
例如，如果您在代码中后期删除了一列，那么如果不需要的话，它不会费心在管道中尽早加载它。
当您想要操作列的子集或进行分析时，这非常有用：

```python
# Dask will download the 'dump' and 'token_count' needed
# for the filtering and computation and skip the other columns.
df.token_count.mean().compute()
```

## 客户端

`dask` 中的大多数功能都针对集群或本地 `Client` 进行了优化，以启动并行计算：

```python
import dask.dataframe as dd
from distributed import Client

if __name__ == "__main__":  # needed for creating new processes
    client = Client()
    df = dd.read_parquet(...)
    ...
```对于本地使用，`Client` 默认使用具有多处理功能的 Dask `LocalCluster`。您可以手动配置 `LocalCluster` 的多处理

```python
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=8, threads_per_worker=8)
client = Client(cluster)
```

请注意，如果您在本地使用默认线程调度程序而不使用`Client`，则 DataFrame 在执行某些操作后可能会变慢（更多详细信息[here](https://github.com/dask/dask-expr/issues/1181)）。

在 [Deploying Dask documentation](https://docs.dask.org/en/latest/deploying.html) 中查找有关设置本地或云集群的更多信息。

### 计费
https://huggingface.co/docs/hub/billing.md