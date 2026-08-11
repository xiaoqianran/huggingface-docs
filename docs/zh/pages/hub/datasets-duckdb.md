<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 鸭数据库

[DuckDB](https://github.com/duckdb/duckdb) 是一个进程内 SQL [OLAP](https://en.wikipedia.org/wiki/Online_analytical_processing) 数据库管理系统。
您可以使用 Hugging Face 路径 (`hf://`) 访问 Hub 上的数据：

[DuckDB CLI](https://duckdb.org/docs/api/cli/overview.html)（命令行界面）是一个单一的、无依赖性的可执行文件。 
还有其他可用于运行 DuckDB 的 API，包括 Python、C++、Go、Java、Rust 等。有关更多详细信息，请访问他们的 [clients](https://duckdb.org/docs/api/overview.html) 页面。

> [!提示]
> 安装详情请访问[installation page](https://duckdb.org/docs/installation)。

> [!提示]
> 如果您使用 `hf` CLI，您可以直接使用 [⟦T9⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-sql) 运行 DuckDB 查询，例如`hf datasets sql "FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3"`。门控和私有数据集的身份验证是根据您登录的令牌自动配置的，并且 `--format json` 返回机器可读的输出。

从版本 `v0.10.3` 开始，DuckDB CLI 包含对通过具有 `hf://` 方案的 URL 访问 Hugging Face Hub 上的数据集的本机支持。以下是您可以利用这个强大工具的一些功能：

- 查询公共数据集以及您自己的门控和私有数据集
- 分析数据集并执行SQL操作
- 合并数据集并将其导出为不同的格式
- 对嵌入数据集进行向量相似度搜索
- 对数据集实施全文搜索有关 DuckDB 功能的完整列表，请访问 DuckDB [documentation](https://duckdb.org/docs/)。

要启动 CLI，请在安装文件夹中执行以下命令：

```bash
./duckdb
```

## 伪造拥抱脸 URL

要访问 Hugging Face 数据集，请使用以下 URL 格式：

```plaintext
hf://datasets/{my-username}/{my-dataset}/{path_to_file} 
```

- **我的用户名**，数据集的用户或组织，例如`ibm`
- **my-dataset**，数据集名称，例如：`duorc`
- **path_to_parquet_file**，支持 glob 模式的 parquet 文件路径，例如 `**/*.parquet`，用于查询所有 parquet 文件

> [!提示]
> 您可以使用 @~parquet 分支查询自动转换的 Parquet 文件，该分支对应于 `refs/convert/parquet` 修订版。有关更多详细信息，请参阅 https://huggingface.co/docs/datasets-server/en/parquet#conversion-to-parquet 上的文档。
>
> 要引用数据集的 `refs/convert/parquet` 修订版，请使用以下语法：
>
> ```plaintext
> hf://datasets/{my-username}/{my-dataset}@~parquet/{path_to_file} 
> ```
>
> 以下是遵循上述语法的示例 URL：
>
> ```plaintext
> hf://datasets/ibm/duorc@~parquet/ParaphraseRC/test/0000.parquet
> ```

让我们从一个快速演示开始，查询数据集的所有行：

```sql
FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3;
```

或者使用传统的 SQL 语法：

```sql
SELECT * FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3;
```
在以下部分中，我们将介绍您可以使用 DuckDB 在 Hugging Face 数据集上执行的更复杂的操作。> [!提示]
> **查询存储桶**：使用 DuckDB Python 客户端时，可以通过注册 Hugging Face 文件系统来查询[Storage Buckets](./storage-buckets)中存储的数据：
> ```python
> import duckdb
> from huggingface_hub import HfFileSystem
> duckdb.register_filesystem(HfFileSystem())
> duckdb.sql("SELECT * FROM 'hf://buckets/username/my-bucket/data.parquet' LIMIT 10")
> ```
DuckDB 预计在未来版本中提供本机 `hf://buckets/` 支持。

### 机器学习文档工具的概况
https://huggingface.co/docs/hub/model-card-landscape-analysis.md