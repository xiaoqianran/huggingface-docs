<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 迁移数据到存储桶

本指南展示了如何将大型数据集从当前所在的任何位置移动到[Storage Bucket](./storage-buckets)：本地磁盘、另一个云或集线器。在下表中找到您的来源并跳至该部分。

尽可能使数据作为文件系统可读并运行 `hf sync`。它在客户端上进行重复数据删除，因此您上传的字节数比原始副本少。

## 概览

|您的数据现在在哪里 |推荐|替代方案|
|---|---|---|
|本地磁盘或集群文件系统 | [⟦T18⟧](#from-local-disk-or-a-cluster-filesystem) | — |
| AWS S3 或 S3 兼容（R2、MinIO）| [Mount with ⟦T19⟧, then ⟦T20⟧](#from-aws-s3-or-s3-compatible-storage) | [rclone through the S3 gateway](#alternative-rclone-through-the-s3-gateway) |
|谷歌云存储 | [Mount with ⟦T21⟧, then ⟦T22⟧](#from-google-cloud-storage-or-azure-blob-storage) | rclone，或 [Job](#run-the-migration-in-the-background-with-jobs) |
| Azure Blob 存储 | [Mount with ⟦T23⟧, then ⟦T24⟧](#from-google-cloud-storage-or-azure-blob-storage) | rclone，或 [Job](#run-the-migration-in-the-background-with-jobs) |
| Hub 存储库或另一个存储桶 | [⟦T25⟧ (server-side)](#from-a-hub-repository-or-another-bucket) | — |
|非常大的 S3 前缀 | [hf-s3ream (alpha)](#very-large-s3-prefixes-with-hf-s3ream) |安装 + `hf sync`、[split by subpath](#split-by-subpath-experimental) |

为什么`hf sync`是默认值：

- **您上传的数据较少。** `hf sync` 通过 [Xet](./xet/index) 客户端上传，该客户端首先对数据进行分块并删除重复数据，因此仅发送 Hub 未见过的数据块。原始副本（例如，通过 S3 网关的 rclone）传输每个字节。
- **至少同样快。** 在我们迄今为止支持的迁移中，已安装源上的 `hf sync` 在吞吐量方面已与 rclone 匹配或击败。## 开始之前

本指南中的所有命令均使用 `hf`、[Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)。它的帮助输出是代理可读的，因此编码代理可以为您组合并监视迁移（请参阅[CLI for AI Agents](./agents-cli)）。在大规模运行命令之前查看命令。

1. **安装 CLI。** Xet 客户端附带它。

   ```bash
   curl -LsSf https://hf.co/cli/install.sh | bash
   hf version
   ```

2. **使用具有 **写入** 访问权限 ([create one](https://huggingface.co/settings/tokens)) 的令牌登录**：

   ```bash
   hf auth login
   hf auth whoami
   ```

3. **创建目标存储桶。** `<namespace>` 是您的用户名或组织。如果没有 `--region`，存储桶将转到命名空间的默认 [storage region](./storage-regions)，这始终是美国团队和企业计划之外的计划。在这些计划中，通过`--region us`或`--region eu`进行选择。

   ```bash
   hf buckets create <namespace>/<bucket> --private
   ```

> [!提示]
> 从试运行开始。无论您使用哪条路径，请在一小部分（一个文件夹或`--include`模式）上排练，使用`--dry-run`和`hf buckets ls`检查结果，然后启动完整迁移。本指南中的每条路径都可以恢复，因此测试运行不需要任何费用。

## 从本地磁盘或集群文件系统

当您的数据位于本地磁盘、附加卷或群集暂存上时，请使用此选项：计算机可以将任何内容读取为文件。一旦安装了源，这也是下面每个云路径的最后一步。

```bash
hf sync ./data hf://buckets/<namespace>/<bucket>/data
````hf sync` 比较源和目标并仅传输差异，因此重新运行相同的命令将恢复中断的迁移：已上传的文件将被跳过。常见变体：

```bash
# Mirror exactly: also delete destination files that are gone from the source
hf sync ./data hf://buckets/<namespace>/<bucket>/data --delete

# Preview what would transfer, without uploading
hf sync ./data hf://buckets/<namespace>/<bucket>/data --dry-run

# Only some files
hf sync ./data hf://buckets/<namespace>/<bucket>/data --include "*.parquet" --exclude "tmp/*"
```

要在运行之前检查大型迁移，请将计划写入文件，检查它，然后应用它。这会扫描源和目标一次，而不是一次进行空运行，然后再次扫描一次进行实际运行。

```bash
hf sync ./data hf://buckets/<namespace>/<bucket>/data --plan plan.jsonl   # scan once, write the plan
hf sync --apply plan.jsonl                                                # run exactly that plan
```

从创建计划的目录运行`--apply`，因为计划会记录您键入的源路径。 `--apply` 不会检查已上传的内容，因此要恢复中断的运行，请重新运行普通的 `hf sync` 命令。

要进行验证，请检查试运行是否没有任何可传输的内容，并抽查内容：

```bash
hf sync ./data hf://buckets/<namespace>/<bucket>/data --dry-run
hf buckets ls hf://buckets/<namespace>/<bucket>/data
```

### 调整- **将缓存保留在快速磁盘上。** Xet 客户端在 `HF_HOME` 下缓存块和重复数据删除索引（分片缓存）。如果您的主目录不在其中，请将 `HF_HOME` 指向本地 NVMe。
- **打开高性能模式**，在资源充足的计算机上进行批量迁移。 `HF_XET_HIGH_PERFORMANCE=1` 提高了并发性、缓冲区大小和一次摄取的文件数量，因此客户端可以使用每个 CPU 核心并使网络饱和。它适用于具有高带宽和快速磁盘的机器；对于 1.7.0 之前的 `hf_xet` 版本，仅使用至少 64 GB 的 RAM（使用 `hf env` 检查您的 RAM）。参见[Environment Variables](./xet/using-xet-storage#environment-variables)。
- **限制分片缓存。** 它的大小限制是软的：缓存仅在运行开始时被修剪，因此在一次长期运行中它可以增长到远远超过限制。在大型迁移中，尤其是在高性能模式下，它可能会增长到数百 GB 并填满磁盘，从而中止运行。将 `HF_XET_SHARD_CACHE_SIZE_LIMIT` 设置为低于可用磁盘空间，但不低于您的需要：大小 *X* 的缓存可对大约 1000 × *X* 的数据进行重复数据删除。参见[Shard Cache](./xet/using-xet-storage#shard-cache)。

在开始同步之前设置这些。尺寸接受人类可读的值，例如 `64gb`：

```bash
export HF_HOME=/mnt/nvme/hf                 # chunk and shard caches on NVMe
export HF_XET_HIGH_PERFORMANCE=1            # use all cores and the full network
export HF_XET_SHARD_CACHE_SIZE_LIMIT=64gb   # keep the shard cache under free disk space

hf sync ./data hf://buckets/<namespace>/<bucket>/data
```

### 按子路径分割（实验）对于非常大的数据集，您可以将源拆分为子路径并同时运行多个 `hf sync` 进程。早期测试表明吞吐量大致呈线性扩展，但这需要更多验证。两条规则：

- 分割相关数据，以便每个工作人员的文件能够很好地相互删除重复数据。
- 为每个工作人员提供自己的`HF_HOME`，这样他们就不会争夺缓存。每个缓存都占用磁盘空间，因此在工作人员之间划分 `HF_XET_SHARD_CACHE_SIZE_LIMIT`。

```bash
HF_HOME=/mnt/nvme/hf-a hf sync ./data/groupA hf://buckets/<namespace>/<bucket>/groupA &
HF_HOME=/mnt/nvme/hf-b hf sync ./data/groupB hf://buckets/<namespace>/<bucket>/groupB &
wait
```

单个进程共享一个缓存并能最好地进行重复数据删除，因此仅当一个进程无法使网络饱和时才进行拆分。

### 故障排除

|症状|修复|
|---|---|
| `401` 或身份验证错误 |使用 **Write** 令牌运行 `hf auth login`，并使用 `hf auth whoami` 检查它。 |
|比预期慢|将`HF_HOME`放在NVMe上，设置`HF_XET_HIGH_PERFORMANCE=1`和[run close to your data](#where-to-run-a-large-migration)。 |
|运行因磁盘已满错误而中止 |分片缓存已占满磁盘。将 `HF_XET_SHARD_CACHE_SIZE_LIMIT` 设置为低于可用空间，或将 `HF_HOME` 移动到更大的磁盘。 |
|传输中断 |重新运行相同的 `hf sync` 命令。已上传的文件将被跳过。 |

## 来自 AWS S3 或 S3 兼容存储`hf sync` 无法读取 `s3://` URL，因此使用 [Mountpoint for Amazon S3](https://github.com/awslabs/mountpoint-s3) 将源存储桶挂载为文件系统，然后从挂载同步。您的数据仍然通过 Xet 客户端，因此您可以获得客户端重复数据删除。

```bash
mount-s3 my-source-bucket /mnt/src
hf sync /mnt/src hf://buckets/<namespace>/<bucket>
umount /mnt/src
```

对于 S3 兼容存储（例如 Cloudflare R2 或 MinIO），请将 `--endpoint-url <url>` 添加到 `mount-s3`。对于允许匿名读取的公共存储桶，例如[AWS Open Data registry](https://registry.opendata.aws/)中的存储桶，请添加`--no-sign-request`。验证、调整和故障排除与[for local data](#from-local-disk-or-a-cluster-filesystem)相同。

### 替代方案：通过 S3 网关进行 rclone

如果您已经运行 rclone 并且不想挂载，请使用此选项。权衡：rclone 通过 [S3 gateway](./storage-buckets-s3) 发送每个字节，没有客户端重复数据删除，因此它传输的数据超过 `hf sync`。您需要[S3 credentials](./storage-buckets-s3#generating-s3-credentials)作为网关。

> [!警告]
> 使用 rclone 1.70 或更高版本（检查`rclone version`，从[rclone downloads](https://rclone.org/downloads/)升级）。旧版本可能会耗尽列出非常大的存储桶的内存。老一辈的建议经常推荐[⟦T67⟧](https://rclone.org/docs/#fast-list)；使用最近的 rclone，您不需要它，并且在一个巨大的列表中，它将所有内容保存在内存中，并且可能导致相同的失败。

将网关添加为`~/.config/rclone/rclone.conf`中的遥控器，位于源遥控器旁边（`aws`如下）：

```ini
[hf]
type = s3
provider = Other
endpoint = https://<namespace>.s3.hf.co
access_key_id = HFAK...
secret_access_key = ...
region = us-east-1
force_path_style = true
list_version = 2
upload_cutoff = 256M
chunk_size = 256M
```网关仅支持ListObjectsV2，因此需要`list_version = 2`。对于 256M 部分，大文件的部分比 rclone 默认的 5M 少得多。 rclone 每次传输在内存中保存 4 个部分，因此下面的`--transfers 16` 需要大约 16 GB RAM；将其降低到较小的机器上。然后复制并检查：

```bash
rclone copy aws:my-source-bucket hf:<bucket> --transfers 16 --checkers 16 --progress
rclone check aws:my-source-bucket hf:<bucket>
```

`rclone copy` 仅传输丢失或更改的对象，因此重新运行恢复是安全的。在进行大量导入之前，请阅读[gateway limitations](./storage-buckets-s3#limitations-and-differences-from-aws-s3)，尤其是对象键命名规则。

## 来自 Google 云存储或 Azure Blob 存储

与 S3 相同的方法：使用云自己的驱动程序安装源，然后运行 `hf sync`。

**Google Cloud Storage**：使用[gcsfuse](https://cloud.google.com/storage/docs/gcsfuse)挂载。

```bash
gcsfuse my-gcs-bucket /mnt/src
hf sync /mnt/src hf://buckets/<namespace>/<bucket>
```

**Azure Blob 存储**：使用 [blobfuse2](https://learn.microsoft.com/azure/storage/blobs/blobfuse2-what-is) 挂载。配置文件命名您的存储帐户和容器。

```bash
blobfuse2 mount /mnt/src --config-file=./blobfuse2.yaml
hf sync /mnt/src hf://buckets/<namespace>/<bucket>
```

如果没有安装，您可以使用 rclone（它本机读取 GCS 和 Azure；重用 `hf` 远程 [above](#alternative-rclone-through-the-s3-gateway)，具有相同的无重复数据删除权衡）或 [run the migration as a Job](#run-the-migration-in-the-background-with-jobs)。

## 使用 Jobs 在后台运行迁移

[Jobs](./jobs) 在 Hugging Face 计算上运行脚本：您选择硬件，从 CLI 启动它，然后[pay by the minute](./jobs-pricing)。当您希望在无人值守的情况下运行长时间的多 TB 迁移，而不是在笔记本电脑或必须保持打开的会话上运行时，请使用它。**1.编写迁移脚本**（`migrate.py`）。此示例通过 Xet 客户端将 S3 前缀中的每个对象复制到存储桶中，以便对写入进行重复数据删除。它适合中等前缀的一次性副本；对于非常大的，请参阅[hf-s3ream](#very-large-s3-prefixes-with-hf-s3ream)。调整顶部的设置：

```python
# /// script
# dependencies = ["huggingface_hub", "boto3", "tqdm"]
# ///
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from huggingface_hub import HfFileSystem
from tqdm.contrib.concurrent import thread_map

# --- adapt these ---
SRC_BUCKET = "my-source-bucket"
SRC_PREFIX = "some/prefix/"
DST = "buckets/<namespace>/<bucket>"
WORKERS = 8   # files copied in parallel
# -------------------

s3 = boto3.client("s3")   # AWS credentials come from the Job's secrets
# For a public bucket (for example from the AWS Open Data registry), read anonymously
# instead, and drop the AWS secrets from the launch command:
# s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
fs = HfFileSystem()       # HF token comes from the Job's HF_TOKEN secret

# List the destination once: {path: size} of the files already copied
done = {path: info["size"] for path, info in fs.find(DST, detail=True).items()}

# List the source and keep the objects not copied yet
todo = []
pages = s3.get_paginator("list_objects_v2").paginate(Bucket=SRC_BUCKET, Prefix=SRC_PREFIX)
for page in pages:
    for obj in page.get("Contents", []):
        key = obj["Key"]
        if key.endswith("/"):
            continue  # folder marker: bucket keys can't end with "/"
        if done.get(f"{DST}/{key}") != obj["Size"]:
            todo.append(key)

def copy(key):
    body = s3.get_object(Bucket=SRC_BUCKET, Key=key)["Body"]
    with fs.open(f"{DST}/{key}", "wb") as out:
        for chunk in iter(lambda: body.read(64 << 20), b""):
            out.write(chunk)

# Copy WORKERS files at a time, with a progress bar
thread_map(copy, todo, max_workers=WORKERS)
```

- 每个对象都暂存在作业的磁盘上，并在关闭时上传，因此磁盘必须同时容纳 `WORKERS` 最大的对象。举起`WORKERS`可以加速许多小物体；对于非常大的，降低它。
- 存储桶中已有的相同大小的文件将被跳过，因此重新运行脚本将恢复中断的复制。它不会重新复制已更改但保持其大小的源文件：为了使存储桶与更改的源保持同步，请改为[mount the source and run ⟦T78⟧](#from-aws-s3-or-s3-compatible-storage)。

对于 GCS 或 Azure，将 `boto3` 替换为该云的 SDK（`google-cloud-storage`、`azure-storage-blob`）并调整列表和读取调用。目的地一半保持不变。

**2.将其作为后台作业**启动，将您的凭据作为秘密传递：

```bash
hf jobs uv run --detach --flavor cpu-performance --timeout 24h \
  --secrets HF_TOKEN \
  --secrets AWS_ACCESS_KEY_ID \
  --secrets AWS_SECRET_ACCESS_KEY \
  migrate.py
```每个 `--secrets NAME` 都会从您的环境传递该变量的值（对于 `HF_TOKEN`，如果未设置，则使用您的登录令牌）。默认情况下，作业会在 30 分钟后停止，因此请为整个传输设置足够长的`--timeout`；如果作业提前停止，请再次启动它，它会跳过已复制的文件。

`--detach` 立即返回并带有作业 ID。使用 `hf jobs logs -f <job_id>` 跟踪进度，使用 `hf jobs ls` 列出运行情况，并使用 `hf jobs hardware` 查看硬件选项。

## 来自 Hub 存储库或其他存储桶

如果数据已位于集线器上、模型或数据集存储库或其他存储桶中，请将其复制到服务器端。对于 [Xet](./xet/index) 跟踪的文件，只有内容哈希会移动，因此副本几乎是即时的，并且不会重新上传任何内容。

```bash
# From a dataset repository into a bucket
hf buckets cp hf://datasets/<org>/<dataset>/<path> hf://buckets/<namespace>/<bucket>/<path>

# From one bucket to another
hf buckets cp hf://buckets/<namespace>/<bucket-a>/<path> hf://buckets/<namespace>/<bucket-b>/<path>
```

- 源和目标必须位于同一存储区域。
- 您需要对源的读取权限和对目标的写入权限。
- 未进行 Xet 跟踪的小文件（配置、自述文件）会自动下载并重新上传。
- 从存储桶复制回存储库尚不可用；它在路线图上。

请参阅 [Copying files between repos and buckets](./storage-buckets#copying-files-between-repos-and-buckets) 了解 Python 等效项。

## 带有 hf-s3ream 的非常大的 S3 前缀> [!警告]
> hf-s3ream 是一个早期 (alpha) 社区工具，不是官方支持的 Hugging Face 产品。在将其用于大型迁移之前，先在小前缀上进行测试。

[hf-s3ream](https://huggingface.co/spaces/glutamatt/hf-s3ream)（[source on GitHub](https://github.com/glutamatt/hf-s3ream)，Apache-2.0）从 Web UI 运行大型 S3 到存储桶的副本：

1. 抱脸登录。
2. 输入 S3 源前缀、目标存储桶和您的 AWS 凭证。对于允许匿名读取的公共存储桶，请打开**公共存储桶**：不需要 AWS 凭证，并且作业会读取未签名的 S3。
3. **Analyze** 运行廉价的试运行，检查访问、检测源区域并建议配置。如果目标存储桶（私有）不存在，它将创建它。
4. **运行** 启动规划器作业。规划器构建基于范围的复印计划，启动复印机作业，并在实时图表上显示进度。如果您关闭选项卡，副本将继续运行，但页面不会重新连接到它。要停止复印，请单击 **停止复印**，或取消计划器作业 (`hf jobs cancel <planner-job-id>`)：计划器取消其复印机。

hf-s3ream 在您的帐户下的 [Jobs](./jobs) 上运行，因此工作将向您计费。作业无法访问仅允许从专用网络 (VPC) 访问的 S3 存储桶。## 在哪里运行大型迁移

对于多 TB 迁移，您通常会在云虚拟机而不是笔记本电脑上运行传输。如果您不想管理虚拟机，[run it as a Job](#run-the-migration-in-the-background-with-jobs)。

- **在与源相同的云和区域中运行。** 通过本地网络读取源是最快且最便宜的，而从源云的出口通常是最大的成本。然后通过互联网上传到 Hugging Face（美国区域位于 AWS `us-east-1`）。
- **一进程多线程。** 高性能模式下单个`hf sync`共享一个去重缓存。 [Split by subpath](#split-by-subpath-experimental) 仅当一个进程无法使您的网络饱和时。
- **保持长时间运行。** 通过 SSH，在 [⟦T92⟧](https://github.com/tmux/tmux/wiki)（或 `screen`、或 `nohup`）内开始迁移，以便它在丢失的会话中继续存在：`tmux new -s migrate`，使用 Ctrl-b 然后 d 分离，稍后使用 `tmux attach -t migrate` 重新连接。

调整起点，以在测试运行中进行验证：|资源 |指导| AWS 示例 | GCP 示例 |
|---|---|---|---|
|磁盘 |适用于 `HF_HOME` 的本地 NVMe，可用空间高于分片缓存限制。避免使用网络连接磁盘作为缓存。 |实例存储 NVMe（`c7i`/`m7i` `d` 变体，或 `i4i`）| `n2` 或 `c3` 虚拟机上的本地 SSD |
|中央处理器|许多 vCPU：客户端在 CPU 上对数据进行分块和哈希处理。 | `c7i.8xlarge` 或更大 | `c3-standard-22` 或更大 |
|内存|充足的高性能模式缓冲区； 1.7.0 之前至少 64 GB，具有 `hf_xet`。 | — | — |
|网络|高带宽互联网（25 Gbps 级）。 | — | — |

## 后续步骤

- **使用就地数据。** 使用 [hf-mount](./storage-buckets-access#mount-as-a-local-filesystem) 将存储桶安装为本地文件系统，或安装为 [volume in Jobs and Spaces](./storage-buckets-access#volume-mounts-in-jobs-and-spaces)。
- **从您的工具中读取。** pandas、Dask、PyArrow、Daft、DuckDB、🤗 数据集和 Spark 直接读取 `hf://buckets/...` 路径。参见[Bucket Integrations](./storage-buckets-integrations)。

> [!警告]
> 不要通过安装运行批量摄取。 hf-mount 阶段写入磁盘缓存（默认为`/tmp/hf-mount-cache`），并且通过它进行大量迁移可以填充该磁盘。使用上述路径之一将数据放入存储桶中。如果必须通过挂载写入大量数据，请将缓存指向具有足够可用空间的磁盘：`hf-mount start --cache-dir /mnt/big bucket <namespace>/<bucket> /mnt/data`（该标志位于`bucket`之前）。### 在拥抱面部使用_Adapters_
https://huggingface.co/docs/hub/adapters.md