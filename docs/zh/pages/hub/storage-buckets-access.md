<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 访问模式

除了 [CLI and Python SDK](./storage-buckets#managing-files) 之外，还有多种方法可以从现有工具和工作流程访问存储桶数据。

## 选择访问方法

|方法|最适合 |详情 |
|--------|----------|---------|
| **hf-安装** |作为本地文件系统挂载 — 任何工具都可以使用 | [See below](#mount-as-a-local-filesystem) |
| **卷安装** | HF 工作和空间（同样的想法，为您管理）| [See below](#volume-mounts-in-jobs-and-spaces) |
| **DuckDB** |从 CLI 或任何客户端对存储桶文件进行 SQL 查询 | [See below](#query-with-duckdb) |
| **hf:// 路径** (fsspec) | Python 数据工具（pandas 和其他 fsspec 感知库）| [See below](#python-data-tools) |
| **CLI 同步** |批量传输、备份| [Sync docs](./storage-buckets#syncing-directories) |
| **S3 API** |现有 S3 工具（AWS CLI、boto3、s5cmd）| [S3-Compatible API](./storage-buckets-s3) |

对于使用存储桶作为后端的工具（SkyPilot、Inspect 等），请参阅 [Integrations](./storage-buckets-integrations)。要使客户端与存储桶保持同步而不轮询它，请参阅[Tracking Changes](./storage-buckets#tracking-changes)。

## 挂载为本地文件系统

[hf-mount](https://github.com/huggingface/hf-mount) 允许您通过 NFS（推荐）或 FUSE 将存储桶（和存储库）安装为本地文件系统。文件是延迟获取的——只有代码读取的字节才会到达网络。

使用[Homebrew](https://brew.sh/)安装：

```bash
brew install hf-mount
```

安装桶：

```bash
hf-mount start bucket username/my-bucket /mnt/data
```安装后，任何读取或写入文件的工具都可以与您的存储桶配合使用 - pandas、DuckDB、vLLM、训练脚本、shell 命令等。

> [!提示]
> 存储桶以读写方式挂载；回购协议是只读的。请参阅 [hf-mount repository](https://github.com/huggingface/hf-mount) 获取完整文档，包括后端选项、缓存和写入模式。

## 在作业和空间中安装卷

[Jobs](./jobs)和[Spaces](./spaces)中的卷挂载与`hf-mount`相同，由平台为您管理 - 无需额外设置。默认情况下，存储桶以读写方式安装。

```bash
hf jobs run -v hf://buckets/username/my-bucket:/data python:3.12 python script.py
```

作业还可以将**本地目录**作为卷源（`-v ./training-data:/data`）：该目录会同步到您的私有`jobs-artifacts`存储桶并从那里安装，因此增量重新同步和输出拉回是免费的。

有关完整卷安装语法和 Python API，请参阅 [Jobs configuration docs](./jobs-configuration#volumes) 和 [Spaces volume mount guide](/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space)。

## 使用 DuckDB 查询

[DuckDB](https://duckdb.org/) 2.0 及更高版本通过 `httpfs` 扩展本机读取 `hf://buckets/` 路径。从 DuckDB CLI 或任何客户端，安装并加载 `httpfs`，然后查询存储桶：

```sql
INSTALL httpfs;
LOAD httpfs;

SELECT * FROM 'hf://buckets/username/my-bucket/data.parquet' LIMIT 10;
```全局模式的工作方式与数据集的工作方式相同，例如`'hf://buckets/username/my-bucket/data/**/*.parquet'`。存储桶没有版本控制，因此 `@revision` 语法不适用于存储桶路径。本机 `hf://buckets/` 访问目前是只读的；要将查询结果写入存储桶，请使用[S3-compatible API](./storage-buckets-s3#query-a-bucket-with-duckdb)或带有`HfFileSystem`的Python客户端。

**私人存储桶：** 首先创建一个 Hugging Face 秘密。它从 `hf auth login` 或 `HF_TOKEN` 环境变量中获取令牌：

```sql
CREATE SECRET hf (TYPE huggingface, PROVIDER credential_chain);
```

> [!注意]
> 在 DuckDB 1.x 上，`httpfs` 无法识别 `hf://buckets/` 路径。请改为从 Python 注册 [⟦T20⟧](/docs/huggingface_hub/guides/hf_file_system)，或使用 CLI 和其他客户端中的 [S3-compatible API](./storage-buckets-s3#query-a-bucket-with-duckdb)。

## Python 数据工具

[⟦T21⟧](/docs/huggingface_hub/guides/hf_file_system) 使用`hf://buckets/` 路径提供对存储桶的[fsspec](https://filesystem-spec.readthedocs.io) 兼容访问。任何支持 fsspec 的 Python 库都可以直接读写存储桶数据。

**熊猫：**

```python
import pandas as pd

df = pd.read_parquet("hf://buckets/username/my-bucket/data.parquet")
df.to_parquet("hf://buckets/username/my-bucket/output.parquet")
```

有关 `hf://` 路径和支持的操作的更多信息，请参阅 [⟦T24⟧ guide](/docs/huggingface_hub/guides/hf_file_system) 和 [Buckets Python guide](/docs/huggingface_hub/guides/buckets)。

### 特工踪迹
https://huggingface.co/docs/hub/agent-traces.md