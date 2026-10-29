<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 鸭数据库

[DuckDB](https://github.com/duckdb/duckdb) 是一个进程内 SQL [OLAP](https://en.wikipedia.org/wiki/Online_analytical_processing) 数据库管理系统。
您可以使用 Hugging Face 路径 (`hf://`) 访问 Hub 上的数据：

[DuckDB CLI](https://duckdb.org/docs/api/cli/overview.html)（命令行界面）是一个单一的、无依赖性的可执行文件。 
还有其他可用于运行 DuckDB 的 API，包括 Python、C++、Go、Java、Rust 等。有关更多详细信息，请访问他们的 [clients](https://duckdb.org/docs/api/overview.html) 页面。

> [!提示]
> 安装详情请访问[installation page](https://duckdb.org/docs/installation)。

> [!提示]
> 如果您使用 `hf` CLI，则可以直接使用 [⟦T15⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-sql) 运行 DuckDB 查询，例如`hf datasets sql "FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3"`。门控和私有数据集的身份验证是根据您登录的令牌自动配置的，并且 `--format json` 返回机器可读的输出。

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
- **path_to_parquet_file**，支持 glob 模式的 parquet 文件路径，例如`**/*.parquet`，查询所有镶木地板文件

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

## 查询数据集

让我们从一个快速演示开始，查询数据集的所有行：

```sql
FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3;
```

或者使用传统的 SQL 语法：

```sql
SELECT * FROM 'hf://datasets/ibm/duorc/ParaphraseRC/*.parquet' LIMIT 3;
```
在以下部分中，我们将介绍您可以使用 DuckDB 在 Hugging Face 数据集上执行的更复杂的操作。> [!提示]
> **查询存储桶**：DuckDB 2.0 及更高版本还从 CLI 和每个客户端本地读取 [Storage Buckets](./storage-buckets)：
> ```sql
> SELECT * FROM 'hf://buckets/username/my-bucket/data.parquet' LIMIT 10;
> ```
> 存储桶没有版本控制，因此 `@revision` 语法不适用于存储桶路径。在 DuckDB 1.x 上，从 Python 注册 Hugging Face 文件系统：`duckdb.register_filesystem(HfFileSystem())`。详情请参阅[Query with DuckDB](./storage-buckets-access#query-with-duckdb)。

## 保存数据集

使用 `COPY ... TO` 语法将数据集保存到 Hugging Face。

1. 安装 DuckDB >=2.0 的 [extension](https://huggingface.co/duckdb-extensions)：

```sql
SET allow_extension_repositories = 'allowed';
CREATE OR REPLACE EXTENSION REPOSITORY hf_repo
  WITH PREFIX 'https://hf.co/buckets/duckdb-extensions/builds/resolve';
INSTALL hf FROM hf_repo;
LOAD hf FROM hf_repo;
```

2. 使用 `hf` CLI 或直接在 DuckDB 中进行身份验证：

* 在 CLI 中：

  ```bash
  hf auth login
  ```

* 在 DuckDB 中：

  ```sql
  CREATE SECRET (TYPE HUGGINGFACE, TOKEN 'hf_...');
  ```

3. 创建数据集存储库（如果尚不存在）：

* 在 CLI 中：

  ```bash
  hf repos create username/my-dataset --repo-type dataset --private
  ```

* 在 DuckDB 中：

  ```sql
  FROM hf_create_repo('hf://datasets/username/my-dataset', private := true);
  ```

4. 将表格导出到 Hugging Face，例如以镶木地板格式：

```sql
COPY tbl TO 'hf://datasets/username/my-dataset/out.parquet';
```

> [!提示]
> 使用`PARTITION_BY`根据分区键将表拆分为多个文件：`COPY tbl TO 'hf://datasets/username/my-dataset/out-directory' (FORMAT parquet, PARTITION_BY (year, month));`。文件根据分区键值放置在正确的子目录中，例如`'hf://datasets/username/my-dataset/out-directory/year=2026/month=1/data_1.parquet'`。

由于数据集是 git 存储库，因此每个新文件都是数据集历史记录中的提交。

### 管理组织
https://huggingface.co/docs/hub/organizations-managing.md