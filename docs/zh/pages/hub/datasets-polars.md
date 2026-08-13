<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 北极星

[Polars](https://pola.rs/) 是一个位于 [OLAP](https://en.wikipedia.org/wiki/Online_analytical_processing) 查询引擎之上的内存中 DataFrame 库。它速度快、易于使用，而且[open source](https://github.com/pola-rs/polars/)。

从版本`1.2.0`开始，Polars 为 Hugging Face 文件系统提供_原生_支持。这意味着将应用 Polars 查询优化器的所有优势（例如谓词和投影下推），并且 Polars 将仅加载完成查询所需的数据。这显着加快了读取速度，尤其是对于大型数据集（参见[optimizations](./datasets-polars-optimizations)）

您可以使用 Hugging Face 路径 (`hf://`) 访问 Hub 上的数据：

## 开始使用

首先，您只需将 `pip install` Polars 添加到您的环境中即可：

```bash
pip install polars
```

安装完 Polars 后，您可以直接根据 Hugging Face URL 查询数据集。为此不需要其他依赖项。

```python
import polars as pl

pl.read_parquet("hf://datasets/roneneldan/TinyStories/data/train-00000-of-00004-2d5a1467fff1081b.parquet")
```

> [!提示]
> Polars 提供了两个 API：lazy API (`scan_parquet`) 和 eager API (`read_parquet`)。我们建议使用 eager API 来实现交互式工作负载，使用惰性 API 来提高性能，因为它可以实现更好的查询优化。有关该主题的更多信息，请查看[Polars user guide](https://docs.pola.rs/user-guide/concepts/lazy-api/#when-to-use-which)。

Polars 支持通配将多个文件一次下载到单个 DataFrame 中。

```python
pl.read_parquet("hf://datasets/roneneldan/TinyStories/data/train-*.parquet")
```

### 拥抱脸部网址拥抱脸部 URL 可以由 `username` 和 `dataset` 名称构造，如下所示：

- `hf://datasets/{username}/{dataset}/{path_to_file}`

该路径可能包括通配模式，例如 `**/*.parquet` 来查询与该模式匹配的所有文件。此外，对于任何不受支持的 [file formats](./datasets-polars-file-formats)，您可以使用 Hugging Face 提供的 `@~parquet branch` 自动转换的镶木地板文件：

- `hf://datasets/{my-username}/{my-dataset}@~parquet/{path_to_file}`

### 图书馆
https://huggingface.co/docs/hub/datasets-libraries.md