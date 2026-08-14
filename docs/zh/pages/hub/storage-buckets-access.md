<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 访问模式

除了 [CLI and Python SDK](./storage-buckets#managing-files) 之外，还有多种方法可以从现有工具和工作流程访问存储桶数据。

## 选择访问方法

|方法|最适合 |详情 |
|--------|----------|---------|
| **hf-安装** |作为本地文件系统挂载 — 任何工具都可以使用 | [See below](#mount-as-a-local-filesystem) |
| **卷安装** | HF 工作和空间（同样的想法，为您管理）| [See below](#volume-mounts-in-jobs-and-spaces) |
| **hf:// 路径** (fsspec) | Python数据工具（pandas、DuckDB）| [See below](#python-data-tools) |
| **CLI 同步** |批量传输、备份| [Sync docs](./storage-buckets#syncing-directories) |
| **S3 API** |现有 S3 工具（AWS CLI、boto3、s5cmd）| [S3-Compatible API](./storage-buckets-s3) |

对于使用存储桶作为后端的工具（SkyPilot、Inspect 等），请参阅 [Integrations](./storage-buckets-integrations)。

## 挂载为本地文件系统

[hf-mount](https://github.com/huggingface/hf-mount) 允许您通过 NFS（推荐）或 FUSE 将存储桶（和存储库）安装为本地文件系统。文件是延迟获取的——只有代码读取的字节才会到达网络。

使用[Homebrew](https://brew.sh/)安装：

```bash
brew install hf-mount
```

安装桶：

```bash
hf-mount start bucket username/my-bucket /mnt/data
```

安装后，任何读取或写入文件的工具都可以与您的存储桶配合使用 - pandas、DuckDB、vLLM、训练脚本、shell 命令等。

> [!提示]
> 存储桶以读写方式挂载；回购协议是只读的。请参阅 [hf-mount repository](https://github.com/huggingface/hf-mount) 了解完整文档，包括后端选项、缓存和写入模式。## 在作业和空间中安装卷

[Jobs](./jobs)和[Spaces](./spaces)中的卷挂载与`hf-mount`相同，由平台为您管理 - 无需额外设置。默认情况下，存储桶以读写方式安装。

```bash
hf jobs run -v hf://buckets/username/my-bucket:/data python:3.12 python script.py
```

作业还可以将**本地目录**作为卷源（`-v ./training-data:/data`）：该目录会同步到您的私有`jobs-artifacts`存储桶并从那里安装，因此增量重新同步和输出拉回是免费的。

有关完整卷安装语法和 Python API，请参阅 [Jobs configuration docs](./jobs-configuration#volumes) 和 [Spaces volume mount guide](/docs/huggingface_hub/guides/manage-spaces#mount-volumes-in-your-space)。

## Python 数据工具

[⟦T8⟧](/docs/huggingface_hub/guides/hf_file_system) 使用`hf://buckets/` 路径提供对存储桶的[fsspec](https://filesystem-spec.readthedocs.io) 兼容访问。任何支持 fsspec 的 Python 库都可以直接读写存储桶数据。

**熊猫：**

```python
import pandas as pd

df = pd.read_parquet("hf://buckets/username/my-bucket/data.parquet")
df.to_parquet("hf://buckets/username/my-bucket/output.parquet")
```

**DuckDB**（Python 客户端）：

```python
import duckdb
from huggingface_hub import HfFileSystem

duckdb.register_filesystem(HfFileSystem())
duckdb.sql("SELECT * FROM 'hf://buckets/username/my-bucket/data.parquet' LIMIT 10")
```

有关 `hf://` 路径和支持的操作的更多信息，请参阅 [⟦T11⟧ guide](/docs/huggingface_hub/guides/hf_file_system) 和 [Buckets Python guide](/docs/huggingface_hub/guides/buckets)。

### 特工踪迹
https://huggingface.co/docs/hub/agent-traces.md