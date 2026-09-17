<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储桶

存储桶是 Hugging Face Hub 上的一种存储库类型，提供类似 S3 的对象存储，由 [Xet](./xet/index) 存储后端提供支持。与基于 Git 的[repositories](./repositories)（模型、数据集、空间）不同，存储桶是**非版本化**和**可变**，专为需要简单、快速存储的用例而设计，例如训练检查点、日志、中间工件或任何不需要版本控制的大型文件集合。

您可以使用 Hub Web 界面、[⟦T20⟧ CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-buckets) 或 [Python API](https://huggingface.co/docs/huggingface_hub/guides/buckets) 与存储桶交互。

> [!提示]
> 存储桶可供所有用户和组织使用。有关定价详情，请参阅[hf.co/storage](https://huggingface.co/storage)。

> [!提示]
> 请参阅 [Access Patterns](./storage-buckets-access) 了解如何从工具获取存储桶数据（作为文件系统挂载、`hf://` 路径、作业/空间中的卷挂载），[S3-Compatible API](./storage-buckets-s3) 了解如何使用现有 S3 工具（AWS CLI、boto3、s5cmd），以及 [Bucket Integrations](./storage-buckets-integrations) 了解流行数据库（如 pandas、Dask 和火花。

## 存储桶与存储库

Hub 提供两种类型的存储：基于 Git 的**存储库**，用于版本化、协作工作；**存储桶**，用于快速、可变的对象存储。|特色|存储库（基于 Git）|存储桶|
| ------------------ | ------------------------------------------- | ----------------------------------- |
|版本控制 |完整的 Git 历史记录 |无（可变，就地覆盖）|
|类型 |模型、数据集、空间 |独立桶|
|主要用例 |发布成品 |工作存储/中间数据|
|运营| Hub API、Git 推/拉 |类似 S3 `sync`、`cp`、`rm` |
|重复数据删除 | Xet 块级 | Xet 块级 |
|请求请求 |是的 |没有 |
|模型/数据集卡|是的 |否（但呈现简单的自述文件）|

当您需要版本历史记录、协作功能（PR、讨论）和库集成时，请使用**存储库**。当您需要快速、可变地存储经常更改的数据时，请使用**存储桶** — 文件可以就地覆盖或删除。

## 创建一个桶

### 从 Hub UI

1. 导航至[huggingface.co/new-bucket](https://huggingface.co/new-bucket)：2. 指定存储桶的所有者：这可以是您或您所属的任何组织。

3. 输入存储桶名称。

4. 选择存储桶是公共的还是私有的。

5. （可选）预先选择 [CDN pre-warming](#pre-warming-and-cdn) 区域，以便从一开始就将数据缓存到更靠近计算的位置。

创建存储桶后，您应该看到存储桶页面：

### 从 CLI

```bash
# Create a bucket under your namespace
hf buckets create my-bucket

# Create a private bucket
hf buckets create my-bucket --private

# Create a bucket under an organization
hf buckets create my-org/shared-bucket
```

### 来自Python

```python
from huggingface_hub import create_bucket

# Create a bucket under your namespace
create_bucket("my-bucket")

# Create a private bucket
create_bucket("my-bucket", private=True)

# Create a bucket under an organization
create_bucket("my-org/shared-bucket")
```

### 改变可见性

您可以在创建后更改存储桶的可见性：

```bash
# Make a bucket private
hf buckets settings username/my-bucket --private

# Make it public again
hf buckets settings username/my-bucket --public
```

```python
from huggingface_hub import update_bucket_settings

update_bucket_settings("username/my-bucket", private=True)
update_bucket_settings("username/my-bucket", private=False)
```

有关完整的 Python API 参考（包括删除、移动和列出存储桶），请参阅 [⟦T25⟧ Buckets guide](https://huggingface.co/docs/huggingface_hub/guides/buckets)。

## 浏览集线器上的存储桶

每个存储桶在 Hub 上都有一个页面，您可以在其中浏览其内容、导航目录和查看文件详细信息。 Bucket 页面可在 `https://huggingface.co/buckets/<owner>/<bucket-name>` 获取。

### 自述文件渲染

如果存储桶中的目录包含 `README.md` 文件，Hub 会将其呈现在该目录页面上的文件列表下方。这适用于存储桶根和任何子目录 - 对于记录存储桶包含的内容、数据的组织方式或下游工具应如何使用它非常有用。

### 从 CLI 列出

您还可以从 CLI 列出存储桶内容：

```bash
# List files in a bucket (with human-readable sizes)
hf buckets list julien-c/my-training-bucket -h
                     Feb 17 14:46  art/
                     Feb 17 14:58  arxivqa/
                     Feb 17 15:02  arxivqa2/
                     Feb 17 15:04  arxivqa3/
                     Feb 17 14:47  captcha/
                     Feb 17 14:53  captcha2/
                     Feb 24 17:22  julien/

# Recursive listing
hf buckets list julien-c/my-training-bucket/art -h -R
    423.6 MB         Feb 17 14:29  art/train-00000-of-00011.parquet
    441.0 MB         Feb 17 14:29  art/train-00001-of-00011.parquet
    521.7 MB         Feb 17 14:29  art/train-00002-of-00011.parquet
    481.4 MB         Feb 17 14:29  art/train-00003-of-00011.parquet
    444.6 MB         Feb 17 14:29  art/train-00004-of-00011.parquet
    461.6 MB         Feb 17 14:29  art/train-00005-of-00011.parquet
    466.4 MB         Feb 17 14:29  art/train-00006-of-00011.parquet
    486.3 MB         Feb 17 14:29  art/train-00007-of-00011.parquet
    477.0 MB         Feb 17 14:29  art/train-00008-of-00011.parquet
    454.0 MB         Feb 17 14:29  art/train-00009-of-00011.parquet
    483.1 MB         Feb 17 14:29  art/train-00010-of-00011.parquet

# Tree view
hf buckets list julien-c/my-training-bucket --tree -h -R
                        ├── art/
423.6 MB  Feb 17 14:29  │   ├── train-00000-of-00011.parquet
441.0 MB  Feb 17 14:29  │   ├── train-00001-of-00011.parquet
521.7 MB  Feb 17 14:29  │   ├── train-00002-of-00011.parquet
481.4 MB  Feb 17 14:29  │   ├── train-00003-of-00011.parquet
444.6 MB  Feb 17 14:29  │   ├── train-00004-of-00011.parquet
461.6 MB  Feb 17 14:29  │   ├── train-00005-of-00011.parquet
466.4 MB  Feb 17 14:29  │   ├── train-00006-of-00011.parquet
486.3 MB  Feb 17 14:29  │   ├── train-00007-of-00011.parquet
477.0 MB  Feb 17 14:29  │   ├── train-00008-of-00011.parquet
454.0 MB  Feb 17 14:29  │   ├── train-00009-of-00011.parquet
483.1 MB  Feb 17 14:29  │   └── train-00010-of-00011.parquet
                        ├── arxivqa/
495.9 MB  Feb 17 14:32  │   ├── train-00000-of-00164.parquet
518.3 MB  Feb 17 14:32  │   ├── train-00001-of-00164.parquet
495.5 MB  Feb 17 14:32  │   ├── train-00002-of-00164.parquet
486.6 MB  Feb 17 14:32  │   ├── train-00003-of-00164.parquet
490.4 MB  Feb 17 14:32  │   ├── train-00004-of-00164.parquet
...
```

## 管理文件您可以直接从 Hub 上的存储桶页面上传和下载文件，或使用 CLI 和 Python API 进行编程访问。使用 `hf://buckets/` 路径（例如 `hf://buckets/username/my-bucket/path/to/file`）引用存储桶文件。 `hf buckets cp` 命令处理单个文件传输，而 `hf buckets sync` 更适合目录。所有命令都双向工作——本地到远程和远程到本地。

如果您的数据已经存在于模型、数据集或空间存储库（或另一个存储桶）中，您可以使用 `hf buckets cp` 将其复制到**服务器端** — 无需下载或重新上传。参见[Copying files between repos and buckets](#copying-files-between-repos-and-buckets)。

### 上传文件

为了快速上传，您可以将文件直接拖放到浏览器中的存储桶页面上。对于编程使用，`hf buckets cp` 将单个文件复制到存储桶中。源是本地路径，目标是`hf://buckets/`路径。您还可以从标准输入传输数据，这对于以编程方式生成的内容非常方便。

**命令行界面：**
```bash
# Upload a single file
hf buckets cp ./model.safetensors hf://buckets/username/my-bucket/models/model.safetensors

# Upload from stdin
cat config.json | hf buckets cp - hf://buckets/username/my-bucket/config.json
```

在 Python 中，使用 `batch_bucket_files` 在一次调用中上传一个或多个文件。每个条目都是`(local_path, remote_path)`的元组。

**Python：**
```python
from huggingface_hub import batch_bucket_files

batch_bucket_files(
    "username/my-bucket",
    add=[
        ("./model.safetensors", "models/model.safetensors"),
        ("./config.json", "models/config.json"),
    ],
)
```

更多上传选项（原始字节、组合上传+删除等），请参阅[⟦T37⟧ upload guide](https://huggingface.co/docs/huggingface_hub/guides/buckets#upload-files)。

### 下载文件您可以通过单击直接从 Hub 上的存储桶页面下载各个文件。对于编程访问，下载镜像了上传语法——交换 `hf buckets cp` 中的源和目标。您还可以使用 `-` 作为目标将文件流式传输到标准输出，这样您就可以将存储桶内容直接通过管道传输到其他工具中。

**命令行界面：**
```bash
# Download a single file
hf buckets cp hf://buckets/username/my-bucket/models/model.safetensors ./model.safetensors

# Download to stdout and pipe
hf buckets cp hf://buckets/username/my-bucket/config.json - | jq .
```

在 Python 中，将 `download_bucket_files` 与 `(remote_path, local_path)` 元组列表一起使用。

**Python：**
```python
from huggingface_hub import download_bucket_files

download_bucket_files(
    "username/my-bucket",
    files=[
        ("models/model.safetensors", "./local/model.safetensors"),
        ("config.json", "./local/config.json"),
    ],
)
```

要使用预取元数据加快下载速度，请参阅 [⟦T42⟧ download guide](https://huggingface.co/docs/huggingface_hub/guides/buckets#download-files)。

### 同步目录

`sync` 命令的工作方式类似于 `rsync` 或 `aws s3 sync` — 它比较源和目标，仅传输已更改的文件。这是保持本地目录和存储桶同步的最有效方法。默认情况下，`sync`仅添加和更新文件。传递 `--delete` 也可以删除目标中不再存在于源中的文件。使用 `--dry-run` 预览会发生什么，而无需实际传输任何内容。

**命令行界面：**
```bash
# Upload a local directory to a bucket
hf buckets sync ./data hf://buckets/username/my-bucket/data

# Download from a bucket to a local directory
hf buckets sync hf://buckets/username/my-bucket/data ./data

# Sync with deletion of extraneous files
hf buckets sync ./data hf://buckets/username/my-bucket/data --delete

# Preview what would be synced without executing
hf buckets sync ./data hf://buckets/username/my-bucket/data --dry-run

# Plan and apply: review the sync plan before executing
hf buckets sync ./data hf://buckets/username/my-bucket/data --plan sync-plan.jsonl
# ... review the plan file, then apply it
hf buckets sync --apply sync-plan.jsonl
```

> [!提示]
> `hf sync` 是 `hf buckets sync` 的方便别名。

**Python：**
```python
from huggingface_hub import sync_bucket

# Upload a local directory to a bucket
sync_bucket("./data", "hf://buckets/username/my-bucket/data")

# Download from a bucket to a local directory
sync_bucket("hf://buckets/username/my-bucket/data", "./data")
````sync` 命令支持过滤（`--include`、`--exclude`）、比较模式（`--ignore-times`、`--existing`）以及用于在执行操作之前检查操作的**计划和应用**工作流程。有关完整的选项集，请参阅[⟦T56⟧ sync guide](https://huggingface.co/docs/huggingface_hub/guides/buckets#sync-directories)。

### 删除文件

由于存储桶没有版本控制，因此删除是立即且永久的——无法恢复已删除的文件。在删除文件之前使用 `--dry-run` 进行仔细检查，尤其是在使用 `--recursive` 时。

**命令行界面：**
```bash
# Remove a single file
hf buckets rm username/my-bucket/old-model.bin

# Remove all files under a prefix
hf buckets rm username/my-bucket/logs/ --recursive

# Preview what would be deleted
hf buckets rm username/my-bucket/checkpoints/ --recursive --dry-run
```

**Python：**
```python
from huggingface_hub import batch_bucket_files

batch_bucket_files("username/my-bucket", delete=["old-model.bin", "logs/debug.log"])
```

有关更多删除选项（基于模式的过滤、递归删除等），请参阅[⟦T59⟧ delete guide](https://huggingface.co/docs/huggingface_hub/guides/buckets#delete-files)。

### 在存储库和存储桶之间复制文件

您可以将 [Xet](./xet/index) 跟踪的文件从任何存储库（模型、数据集、空间）或存储桶复制到目标存储桶，而无需重新上传数据。复制是在服务器端进行的：仅迁移 Xet 内容哈希，因此即使非常大的文件也会立即复制，这要归功于 [chunk-level deduplication](./xet/deduplication)。

> [!注意]
> 仅 Xet 跟踪的文件会在服务器之间复制。小的非 Xet 文件（例如配置文件和自述文件）会自动下载并重新上传。服务器端复制还要求源和目标位于同一存储区域。

**命令行界面：**
```bash
hf buckets cp \
  hf://datasets/HuggingFaceFW/fineweb/data \
  hf://buckets/username/fineweb-data
```

**Python：**
```python
from huggingface_hub import HfApi

api = HfApi()

api.copy_files(
    "hf://datasets/HuggingFaceFW/fineweb/data",
    "hf://buckets/username/fineweb-data",
)
```您需要对源存储库或存储桶的读取权限以及对目标存储桶的写入权限。

请注意，在不重新上传的情况下以其他方式将数据从存储桶传输到存储库（模型、数据集、空间）尚不可用，但已在路线图上。

## 跟踪更改

存储桶是可变的，因此保存存储桶视图的工具（挂载、文件系统层、同步守护进程、仪表板）需要知道文件何时发生更改。有两种机制可用：

- [Webhooks](./webhooks#buckets)：HTTP 回调到您控制的服务器，以实现自动化和集成。
- **实时关注**：您的客户端订阅的服务器发送的事件流。

### 直播关注

`GET https://huggingface.co/api/buckets/<owner>/<bucket-name>/events` 将存储桶的文件更改流式传输为 [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)。该请求必须携带`Accept: text/event-stream`（否则返回`400`），并且它需要与列出存储桶相同的读取访问权限 - 公共存储桶不需要令牌。有关完整参数和响应模式，请参阅[OpenAPI spec](https://huggingface.co/spaces/huggingface/openapi#tag/buckets/GET/api/buckets/{namespace}/{repo}/events)。

```bash
curl -N -H "Accept: text/event-stream" \
  -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/buckets/username/my-bucket/events"
```

该流发出四种事件类型：|活动 |数据|意义|
| ----------- | ---------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `ready` | `{"cursor": "..."}` |任何请求的重播均已完成，随后进行实时更改。                                         |
| `changes` | `{"cursor": "...", "changes": [...]}` |一批文件更改在一个短窗口内合并。                                       |
| `reset` | `{"reason": "cursor_too_old"}` |恢复点无法重播；流结束，您应该重新列出存储桶。       |
| `reconnect` | `{"cursor": "..."}` |服务器故意关闭流；重新连接该光标。                      |`changes` 中的每个条目都有一个`path` 和一个`op`（`add`、`update` 或`delete`）。 `add` 或 `update` 还携带已更改的字段 — `size`、`xetHash`、`uploadedAt`、`mtime`、`mtimeNanos` — 因此，当文件被删除时，`update` 可能与新的 `uploadedAt` 一样小。重新上传相同。没有改变的字段被省略；当上传清除它们时，`mtime`/`mtimeNanos`也可能是`null`，因此对待缺席和`null`一视同仁。仅当您具有存储桶内容的读取访问权限时，才会包含 `xetHash`。

```
event: ready
data: {"cursor":"..."}

event: changes
data: {"cursor":"...","changes":[{"path":"data/train.txt","op":"add","size":20,"uploadedAt":"2026-09-16T09:21:45.000Z"},{"path":"data/old.txt","op":"delete"}]}
```

**恢复中。** 每个`ready`和`changes`事件都带有不透明的`cursor`。使用 `?cursor=<cursor>` 重新连接以获取之后发生的更改，或者使用 `?since=<ISO timestamp>`（包括，例如 `2026-09-16T09:21:45Z`）从瞬间恢复。如果不使用这两个参数，您只会收到连接后发生的更改。

只能重播**最后 ~15 分钟**的更改。如果您的 `cursor` 或 `since` 早于该时间，您将收到 `reset` 而不是重播：列出存储桶一次以重建您的视图，然后从新流的 `ready` 事件的 `cursor` 再次关注。恢复旨在弥补短暂的间隙，例如连接断开或重新启动；离开时间较长的客户应该会重新列出。**重新连接。** 长期连接会被回收：大约每 20 分钟（以及在部署期间）服务器发送 `reconnect`，然后结束流。以相同的方式处理流的*任何*结尾 - 重新连接您收到的最后一个光标。如果 `reconnect` 到达时没有光标，请使用您最初请求的相同 `cursor` 或 `since` 重新连接。每 30 秒发送一次 `: ping` 注释以保持连接处于活动状态，因此完全静默的流可以被视为已死。

`503` 响应意味着实时跟随暂时不可用 - 在 `Retry-After` 标头中的延迟后重试。

## 预热和 CDN

默认情况下，存储桶位于 Hub 的全局存储中。对于存储位置直接影响吞吐量的工作负载，您可以**预热**存储桶数据，使其更接近您的计算。

预热将文件缓存在特定云提供商和区域附近的边缘位置，因此您的作业在本地读取数据，而不是跨区域拉取数据。这对于以下情况特别有用：- 需要快速访问大型数据集或检查点的训练集群
- 多区域设置，管道的不同部分在不同的云中运行
- 向全球许多消费者分发大型文物

有关可用区域以及启用预热的详细信息，请参阅[hf.co/storage](https://huggingface.co/storage)。

## 用例

### 训练检查点和日志

运行训练作业时（例如，通过[Jobs](./jobs)），将检查点和日志保存到存储桶中。与 Git 存储库不同，您可以覆盖最新的检查点，而无需累积版本历史记录，并且 `sync` 确保仅传输更改的数据。

```bash
# After each evaluation step, sync checkpoints to a bucket
hf sync ./checkpoints hf://buckets/my-org/training-run-42/checkpoints
```

由于存储桶是基于[Xet](./xet/index)构建的，因此模型大部分被冻结的连续检查点受益于块级重复数据删除。仅上传更改的块。

### 数据处理管道

存储桶充当数据处理工作流的暂存区域。处理原始数据，将中间输出写入存储桶，然后在管道完成时将最终工件提升到版本化的[Dataset](./datasets)存储库。这使您的版本化存储库保持干净，同时为您的管道提供快速的可变存储。请注意，在不重新上传的情况下将数据从存储桶传输到存储库尚不可用，但已在路线图上。

### 代理存储

人工智能代理需要临时存储来存储中间结果、工具输出、痕迹和工作内存。存储桶为这些数据提供了 Hub 原生位置：快速可变访问，无需 Git 开销，标准 Hugging Face 权限，并且可通过 Hub 生态系统中的 `hf://buckets/` 路径进行寻址。

### 滚动备份

存储桶非常适合维护滚动备份。使用基于 Git 的 [Dataset](./datasets) 存储库，删除过时的文件并不会释放存储空间 - Git 历史记录会保留每个过去的版本，因此您需要压缩提交或重写历史记录才能实际回收空间。使用存储桶，旧文件一旦删除就会真正消失，您只需为当前存储的内容付费。

```bash
# Sync today's backup, removing files that no longer exist locally
hf sync ./daily-backup hf://buckets/my-user/backups/latest --delete
```

### 将模型链接到存储桶

您可以通过将 `buckets` 字段添加到模型卡元数据来创建模型和存储桶之间的双向链接。然后，链接的模型将显示在存储桶页面上，并且存储桶将在模型页面上显示为标签。

```yaml
# In the model card YAML frontmatter
buckets:
- my-org/my-bucket
```

有关更多详细信息，请参阅模型卡文档中的[Specifying a bucket](./model-cards#specifying-a-bucket)。

## 定价存储桶根据存储的数据量进行计费，并采用简单的按 TB 定价。企业计划受益于基于重复数据删除的计费，其中跨文件共享的块直接减少了计费占用空间。

至于其他存储库，存储桶可以自由创建，并且有免费的存储空间。对于[free tier](https://huggingface.co/docs/hub/storage-limits)以上的用法，请参阅[hf.co/storage](https://huggingface.co/storage)。有关一般计费信息，请参阅 [Billing](./billing) 文档。

### 在空间中使用 OpenCV
https://huggingface.co/docs/hub/spaces-using-opencv.md