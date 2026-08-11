<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 愚蠢

[Daft](https://daft.ai/)是一个高性能数据引擎，为任何模式和规模提供简单可靠的数据处理。 Daft 原生支持读取和写入 Hugging Face 数据集。

## 开始使用

首先，使用 `huggingface` 功能 pip install `daft`：

```bash
pip install 'daft[huggingface]'
```

## 阅读

Daft 能够使用 [⟦T7⟧](https://docs.daft.ai/en/stable/api/io/#daft.read_huggingface) 函数或通过 `hf://datasets/` 协议直接从 Hugging Face Hub 读取数据集。

### 读取整个数据集

使用[⟦T9⟧](https://docs.daft.ai/en/stable/api/io/#daft.read_huggingface)，您可以轻松加载数据集。

```python
import daft

df = daft.read_huggingface("username/dataset_name")
```

这会将整个数据集读入 DataFrame。

### 读取特定文件

您不仅可以读取整个数据集，还可以从数据集存储库中读取单个文件。使用接受路径（例如 [⟦T10⟧](https://docs.daft.ai/en/stable/api/io/#daft.read_parquet)、[⟦T11⟧](https://docs.daft.ai/en/stable/api/io/#daft.read_csv) 或 [⟦T12⟧](https://docs.daft.ai/en/stable/api/io/#daft.read_json)）的读取函数，通过 `hf://datasets/` 前缀指定 Hugging Face 数据集路径：

```python
import daft

# read a specific Parquet file
df = daft.read_parquet("hf://datasets/username/dataset_name/file_name.parquet")

# or a csv file
df = daft.read_csv("hf://datasets/username/dataset_name/file_name.csv")

# or a set of Parquet files using a glob pattern
df = daft.read_parquet("hf://datasets/username/dataset_name/**/*.parquet")
```

## 写

Daft 能够使用 [⟦T14⟧](https://docs.daft.ai/en/stable/api/dataframe/#daft.DataFrame.write_deltalake) 将 Parquet 文件写入 Hugging Face 数据集存储库。 Daft 支持 [Content-Defined Chunking](https://huggingface.co/blog/parquet-cdc) 和 [Xet](https://huggingface.co/blog/xet-on-the-hub)，以实现更快的重复数据删除写入。

基本用法：

```python
import daft

df: daft.DataFrame = ...

df.write_huggingface("username/dataset_name")
```

有关更多信息，请参阅 [⟦T15⟧](https://docs.daft.ai/en/stable/api/dataframe/#daft.DataFrame.write_huggingface) API 页面。

＃＃ 验证[⟦T17⟧](https://docs.daft.ai/en/stable/api/config/#daft.io.HuggingFaceConfig) 中的 `token` 参数可用于为需要身份验证的请求指定 Hugging Face 访问令牌（例如，读取私有数据集存储库或写入数据集存储库）。

Example of loading a dataset with a specified token:

```python
from daft.io import IOConfig, HuggingFaceConfig

io_config = IOConfig(hf=HuggingFaceConfig(token="your_token"))
df = daft.read_parquet("hf://datasets/username/dataset_name", io_config=io_config)
```

### 在拥抱脸部使用 BERTopic
https://huggingface.co/docs/hub/bertopic.md