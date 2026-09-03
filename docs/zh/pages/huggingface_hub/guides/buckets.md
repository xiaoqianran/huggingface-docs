<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 桶

Buckets 在 Hugging Face 上提供类似 S3 的对象存储，由 Xet 存储后端提供支持。与存储库（基于 git 并跟踪文件历史记录）不同，存储桶是远程对象存储容器，专为具有内容可寻址重复数据删除功能的大型文件而设计。它们专为需要简单、快速、可变存储的用例而设计，例如存储训练检查点、日志、中间工件或任何不需要版本控制的大型文件集合。

您可以使用 Python API ([HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi)) 或 CLI (`hf buckets`) 与存储桶交互。在本指南中，我们将逐步介绍所有可用的操作。

> [!提示]
> 所有 CLI 命令均可在 `hf buckets <command>` 下使用。运行 `hf buckets --help` 了解更多信息。

## 创建和管理存储桶

### 创建一个桶

使用 [create_bucket()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_bucket) 创建一个存储桶。您需要提供存储桶名称。如果您未指定命名空间，则会在您的用户名下创建存储桶。

```py
>>> from huggingface_hub import create_bucket

# Create a bucket under your namespace
>>> url = create_bucket("my-bucket")
>>> url.bucket_id
'username/my-bucket'
>>> url.uri.to_uri()
'hf://buckets/username/my-bucket'

# Create a private bucket
>>> create_bucket("my-private-bucket", private=True)
BucketUrl(...)

# Don't error if bucket already exists
>>> create_bucket("my-bucket", exist_ok=True)
BucketUrl(...)

# Create a bucket in a specific region
>>> create_bucket("my-bucket", region="us")
BucketUrl(...)
```

或者通过 CLI：

```bash
>>> hf buckets create my-bucket
✓ Bucket created
  uri: hf://buckets/Wauplin/my-bucket
  url: https://huggingface.co/buckets/Wauplin/my-bucket

# Create a private bucket
>>> hf buckets create my-bucket --private

# Don't error if bucket already exists
>>> hf buckets create my-bucket --exist-ok

# Create a bucket in a specific region
>>> hf buckets create my-bucket --region us
```

您还可以指定完整的`namespace/bucket_name`格式来在组织下创建存储桶：

```py
>>> from huggingface_hub import create_bucket
>>> create_bucket("my-org/shared-bucket")
```

或者通过 CLI：

```bash
>>> hf buckets create my-org/shared-bucket
```

### 获取桶信息

使用 [bucket_info()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.bucket_info) 获取有关存储桶的元数据，包括其可见性、总大小、文件计数和创建日期。

```py
>>> from huggingface_hub import bucket_info
>>> bucket_info("username/my-bucket")
BucketInfo(
  id='username/my-bucket',
  private=False,
  created_at=datetime.datetime(2026, 2, 12, 17, 42, 12,
  tzinfo=datetime.timezone.utc),
  size=8411791508,
  total_files=128
)
```或者通过 CLI：

```bash
# JSON output
>>> hf buckets info username/my-bucket
{
  "id": "username/my-bucket",
  "private": false,
  ...
}
```

### 列出存储桶

使用 [list_buckets()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_buckets) 列出命名空间中的所有存储桶。默认情况下，它列出当前用户的命名空间中的存储桶。

```py
>>> from huggingface_hub import list_buckets

# List your own buckets
>>> for bucket in list_buckets():
...     print(bucket.id, bucket.size, bucket.total_files)

# List buckets in an organization
>>> for bucket in list_buckets(namespace="huggingface"):
...     print(bucket.id)
```

或者通过 CLI（`hf buckets ls` 是 `hf buckets list` 的简写）：

```bash
# Table format (default)
>>> hf buckets list
ID                   PRIVATE      SIZE TOTAL_FILES CREATED_AT
-------------------- ------- --------- ----------- ----------
username/my-bucket                  32           5 2026-02-16
username/checkpoints         117609095         700 2026-02-13
username/logs                321757477        2000 2026-02-13

# Human-readable sizes
>>> hf buckets list -h
ID                   PRIVATE     SIZE TOTAL_FILES CREATED_AT
-------------------- ------- -------- ----------- ----------
username/my-bucket               32 B           5 2026-02-16
username/checkpoints         117.6 MB         700 2026-02-13
username/logs                321.8 MB        2000 2026-02-13

# List buckets in a specific namespace
>>> hf buckets ls huggingface

# Filter buckets by name
>>> hf buckets list --search "checkpoint"
```

您还可以使用 `search` 按名称过滤存储桶：

```py
>>> for bucket in list_buckets(search="checkpoint"):
...     print(bucket.id)
```

您可以使用 `--quiet` 和 `--format json` 选项来获取不同的输出格式。如果您想将输出通过管道传输到另一个工具（例如 `grep` 或 `jq`），这会特别有趣。

```bash
# Quiet mode: prints one bucket ID per line
>>> hf buckets list --quiet
username/my-bucket
username/checkpoints
username/logs

# JSON format
>>> hf buckets list --format json
[
  {
    "id": "username/my-bucket",
    "private": false,
    "created_at": "2026-02-16T15:28:32+00:00",
    "size": 32,
    "total_files": 5
  },
  ...
]
```

### 更改存储桶可见性

使用 [update_bucket_settings()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_bucket_settings) 在私有和公共之间切换现有存储桶。

```py
>>> from huggingface_hub import update_bucket_settings

# Make a bucket private
>>> update_bucket_settings("username/my-bucket", private=True)

# Make it public again
>>> update_bucket_settings("username/my-bucket", private=False)
```

或者通过 CLI：

```bash
# Make a bucket private
>>> hf buckets settings username/my-bucket --private
✓ Bucket settings updated
  bucket_id: username/my-bucket
  private: True

# Make it public again
>>> hf buckets settings username/my-bucket --public
```

### 删除一个桶

使用[delete_bucket()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_bucket)删除桶。此操作是不可逆的。

```py
>>> from huggingface_hub import delete_bucket
>>> delete_bucket("username/my-bucket")

# Don't error if bucket doesn't exist
>>> delete_bucket("username/my-bucket", missing_ok=True)
```

或者通过 CLI：

```bash
# Prompts for confirmation
>>> hf buckets delete username/my-bucket

# Skip confirmation
>>> hf buckets delete username/my-bucket --yes

# Don't error if bucket doesn't exist
>>> hf buckets delete username/my-bucket --yes --missing-ok
```

### 删除文件

使用 [batch_bucket_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.batch_bucket_files) 和 `delete` 参数从存储桶中删除文件：

```py
>>> from huggingface_hub import batch_bucket_files

# Delete specific files
>>> batch_bucket_files("username/my-bucket", delete=["old-model.bin", "logs/debug.log"])
```

或者通过 CLI 使用 `hf buckets rm`（或 `hf buckets remove`）：

```bash
# Remove a single file
>>> hf buckets rm username/my-bucket/old-model.bin

# Remove all files under a prefix (requires --recursive)
>>> hf buckets rm username/my-bucket/logs/ --recursive

# Remove only files matching a pattern across the entire bucket
>>> hf buckets rm username/my-bucket --recursive --include "*.tmp"

# Exclude specific files from removal
>>> hf buckets rm username/my-bucket/data/ --recursive --exclude "*.safetensors"

# Preview what would be deleted
>>> hf buckets rm username/my-bucket/checkpoints/ --recursive --dry-run
```

### 移动一个桶

使用 [move_bucket()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.move_bucket) 移动或重命名存储桶。您可以在同一命名空间内重命名或转移到不同的命名空间（用户或组织）。

```py
>>> from huggingface_hub import move_bucket

# Rename a bucket
>>> move_bucket(from_id="username/old-name", to_id="username/new-name")

# Transfer to an organization
>>> move_bucket(from_id="username/my-bucket", to_id="my-org/my-bucket")
```

或者通过 CLI：

```bash
# Rename a bucket
>>> hf buckets move username/old-bucket username/new-bucket

# Transfer to an organization
>>> hf buckets move username/my-bucket my-org/my-bucket

# Using the hf://buckets/ format
>>> hf buckets move hf://buckets/username/old-bucket hf://buckets/username/new-bucket
```

> [!提示]
> 有关移动存储库和存储桶的更多信息，请参阅[Hub documentation](https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo)。

## 浏览存储桶内容

### 列出文件

使用 [list_bucket_tree()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_bucket_tree) 列出存储桶中的文件和目录。

```py
>>> from huggingface_hub import list_bucket_tree

# List all files
>>> for item in list_bucket_tree("username/my-bucket"):
...     print(item.path, item.size)
file.txt 5
big.bin 2048
sub/nested.txt 14
sub/deep/file.txt 4

# List top-level entries only
>>> for item in list_bucket_tree("username/my-bucket", recursive=False):
...     print(item.type, item.path)
file file.txt
file big.bin
directory sub

# Filter by prefix
>>> for item in list_bucket_tree("username/my-bucket", prefix="sub"):
...     print(item.path)
```或者通过 CLI，支持表格、人类可读和 ASCII 树格式。默认情况下，CLI 是非递归的。

```bash
# Default table format (non-recursive, directories shown as entries)
>>> hf buckets list username/my-bucket
        2048  2026-01-15 10:30:00  big.bin
           5  2026-01-15 10:30:00  file.txt
              2026-01-15 10:30:00  sub/

# Recursive listing (all files, including nested)
>>> hf buckets list username/my-bucket -R
        2048  2026-01-15 10:30:00  big.bin
           5  2026-01-15 10:30:00  file.txt
          14  2026-01-15 10:30:00  sub/nested.txt
           4  2026-01-15 10:30:00  sub/deep/file.txt

# Human-readable sizes and short dates
>>> hf buckets list username/my-bucket -R -h
      2.0 KB         Jan 15 10:30  big.bin
         5 B         Jan 15 10:30  file.txt
        14 B         Jan 15 10:30  sub/nested.txt
         4 B         Jan 15 10:30  sub/deep/file.txt

# ASCII tree format
>>> hf buckets list username/my-bucket --tree -h -R
2.0 KB  Jan 15 10:30  ├── big.bin
   5 B  Jan 15 10:30  ├── file.txt
                      └── sub/
                          ├── deep/
   4 B  Jan 15 10:30  │       └── file.txt
  14 B  Jan 15 10:30  └── nested.txt

# Tree structure only (no sizes/dates)
>>> hf buckets list username/my-bucket --tree --quiet -R
├── big.bin
├── file.txt
└── sub/
    ├── deep/
    │   └── file.txt
    └── nested.txt

# Quiet mode: one path per line
>>> hf buckets list username/my-bucket -q
big.bin
file.txt
sub/

# Filter by prefix
>>> hf buckets list username/my-bucket/sub -R
```

> [!提示]
> `hf buckets list` 命令接受短格式 (`username/my-bucket/sub`) 和完整句柄 (`hf://buckets/username/my-bucket/sub`) 作为参数。

## 上传文件

### 使用Python上传

使用[batch_bucket_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.batch_bucket_files)将文件上传到存储桶。您可以从本地文件路径或原始字节上传：

```py
>>> from huggingface_hub import batch_bucket_files

# Upload from local file paths
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[
...         ("./model.safetensors", "models/model.safetensors"),
...         ("./config.json", "models/config.json"),
...     ],
... )

# Upload from raw bytes
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[
...         (b'{"key": "value"}', "config.json"),
...     ],
... )
```

您还可以使用 `copy` 参数从另一个存储桶或存储库复制 xet 文件。这是服务器端操作 - 不会下载或重新上传数据：

```python
# Copy files by xet hash (source_repo_type, source_repo_id, xet_hash, destination)
>>> batch_bucket_files(
...     "username/my-bucket",
...     copy=[
...         ("bucket", "username/source-bucket", "<xethash_1>", "models/model.safetensors"),
...         ("model", "username/my-model", "<xethash_2>", "models/config.safetensors"),
...     ],
... )
```

可以使用 `list_repo_tree` 检索 Xet 哈希值。

您还可以在上传其他文件时删除文件。

```python
# Upload and delete in one batch
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[("./new-model.safetensors", "model.safetensors")],
...     delete=["old-model.bin"],
... )
```

> [!警告]
> 对 [batch_bucket_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.batch_bucket_files) 的呼叫是非事务性的。如果在此过程中发生错误，则某些文件可能已上传、复制或删除，而其他文件则尚未上传、复制或删除。

### 使用 CLI 上传单个文件

使用`hf buckets cp`上传单个文件：

> [!提示]
> `hf buckets cp` 是统一的 `hf cp` 命令的别名（也公开为 `hf repos cp`）。这三个都是相同的，因此下面的任何 `hf buckets cp` 示例也适用于 `hf cp`。

```bash
# Upload to bucket root (uses local filename as remote name)
>>> hf buckets cp ./config.json hf://buckets/username/my-bucket

# Upload to a subdirectory
>>> hf buckets cp ./data.csv hf://buckets/username/my-bucket/logs/

# Upload with a different remote filename
>>> hf buckets cp ./model.safetensors hf://buckets/username/my-bucket/v2/model.safetensors
```

您还可以使用 `-` 将命令的结果直接传输到新文件中：

```
# Upload from stdin
>>> echo "hello" | hf buckets cp - hf://buckets/username/my-bucket/hello.txt
>>> cat model.safetensors | hf buckets cp - hf://buckets/username/my-bucket/model.safetensors
```

### 使用 CLI 上传目录使用 `hf buckets sync` 将整个本地目录上传到存储桶：

```bash
# Upload a local directory to a bucket
>>> hf buckets sync ./data hf://buckets/username/my-bucket

# Upload to a specific prefix in the bucket
>>> hf buckets sync ./data hf://buckets/username/my-bucket/train
```

有关完整的同步选项集，请参阅下面的[Sync directories](#sync-directories)部分。

## 下载文件

### 使用Python下载

使用 [download_bucket_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.download_bucket_files) 从存储桶下载文件：

```py
>>> from huggingface_hub import download_bucket_files

# Download specific files by path
>>> download_bucket_files(
...     "username/my-bucket",
...     files=[
...         ("models/model.safetensors", "./local/model.safetensors"),
...         ("config.json", "./local/config.json"),
...     ],
... )
```

为了获得更好的性能，您可以传递从 [list_bucket_tree()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_bucket_tree) 获取的 [BucketFile](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.BucketFile) 对象而不是字符串路径。这会跳过元数据获取步骤：

```py
>>> from huggingface_hub import list_bucket_tree, download_bucket_files

# List and filter files, then download
>>> parquet_files = [
...     item for item in list_bucket_tree("username/my-bucket", recursive=True)
...     if item.type == "file" and item.path.endswith(".parquet")
... ]
>>> download_bucket_files(
...     "username/my-bucket",
...     files=[(f, f"./local/{f.path}") for f in parquet_files],
... )
```

### 使用 CLI 下载单个文件

使用`hf buckets cp`下载单个文件：

```bash
# Download to a specific file
>>> hf buckets cp hf://buckets/username/my-bucket/config.json ./config.json

# Download to a directory (uses original filename)
>>> hf buckets cp hf://buckets/username/my-bucket/config.json ./data/

# Download to current directory (omit destination)
>>> hf buckets cp hf://buckets/username/my-bucket/config.json
```

您还可以使用 `-` 将文件内容直接通过管道传输到 `stdout`：

```bash
# Download to stdout and pretty-print with jq
>>> hf buckets cp hf://buckets/username/my-bucket/config.json - | jq .
```

### 使用 CLI 下载目录

使用 `hf buckets sync` 将存储桶中的所有文件下载到本地目录：

```bash
# Download bucket contents to a local directory
>>> hf buckets sync hf://buckets/username/my-bucket ./data

# Download only a specific prefix
>>> hf buckets sync hf://buckets/username/my-bucket/models ./local-models
```

有关完整的同步选项集，请参阅下面的[Sync directories](#sync-directories)部分。

## 将文件复制到存储桶

使用 [copy_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.copy_files) 将 Hub 上已托管的文件复制到存储桶：

```py
>>> from huggingface_hub import copy_files

# Bucket to bucket (same or different bucket)
>>> copy_files(
...     "hf://buckets/username/source-bucket/checkpoints/model.safetensors",
...     "hf://buckets/username/destination-bucket/archive/model.safetensors",
... )

# Repo to bucket
>>> copy_files(
...     "hf://datasets/username/my-dataset/processed/",
...     "hf://buckets/username/my-bucket/datasets/processed/",
... )
```

通过统一的 `hf cp` 命令（也公开为 `hf buckets cp` 和 `hf repos cp` — 所有三个都是相同的），可以从 CLI 获得相同的功能：

```bash
# Bucket to bucket
>>> hf cp hf://buckets/username/source-bucket/logs/ hf://buckets/username/destination-bucket/logs/

# Repo to bucket
>>> hf cp hf://username/my-model/config.json hf://buckets/username/my-bucket/models/config.json
```

复制文件夹时，源上的尾随 `/` 使用 rsync 样式语义 - 仅复制文件夹的*内容*，而不嵌套文件夹本身：

```bash
# Without trailing slash: "logs" dir is nested => destination/logs/...
>>> hf cp hf://buckets/username/source-bucket/logs hf://buckets/username/destination-bucket/

# With trailing slash: only contents of "logs" are copied => destination/...
>>> hf cp hf://buckets/username/source-bucket/logs/ hf://buckets/username/destination-bucket/
```

注意事项：- 尚不支持存储桶到存储库的复制。
- 服务器端副本仅在同一[storage region](https://huggingface.co/docs/hub/storage-regions)内工作。
- 使用 Xet 跟踪的文件（在存储桶或存储库中）通过哈希值复制到服务器端 — 不会下载或重新上传数据。
- 下载存储库源上未使用 Xet 跟踪的小文本文件，并将其重新上传到目标存储桶。
- [copy_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.copy_files) 也可用于存储库到存储库的副本。更多详情请参阅[repository guide](./repository#copy-files)。

## 同步目录

`hf buckets sync` 命令（及其等效的 API [sync_bucket()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_bucket)）是在本地目录和存储桶之间传输文件的最强大方法。它比较源和目标，并仅传输已更改的文件。

### 基本同步

```bash
# Upload: local directory -> bucket
>>> hf buckets sync ./data hf://buckets/username/my-bucket

# Download: bucket -> local directory
>>> hf buckets sync hf://buckets/username/my-bucket ./data
```

> [!提示]
> `hf sync` 是 `hf buckets sync` 的方便别名。两个命令是相同的。
> ```bash
> >>> hf sync ./data hf://buckets/username/my-bucket
> ```

或者通过Python：

```py
>>> from huggingface_hub import sync_bucket

# Upload: local directory -> bucket
>>> sync_bucket("./data", "hf://buckets/username/my-bucket")

# Download: bucket -> local directory
>>> sync_bucket("hf://buckets/username/my-bucket", "./data")
```

### 删除无关文件

默认情况下，同步仅添加或更新文件。使用 `--delete` （或 Python 中的 `delete=True`）也可以删除目标中源中不存在的文件：

```bash
# Upload and remove remote files not present locally
>>> hf buckets sync ./data hf://buckets/username/my-bucket --delete

# Download and remove local files not present in bucket
>>> hf buckets sync hf://buckets/username/my-bucket ./data --delete
```

或者通过Python：

```py
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", delete=True)
>>> sync_bucket("hf://buckets/username/my-bucket", "./data", delete=True)
```

### 过滤

您可以使用包含/排除模式控制同步哪些文件：

```bash
# Only sync .txt files
>>> hf buckets sync ./data hf://buckets/username/my-bucket --include "*.txt"

# Exclude log files
>>> hf buckets sync ./data hf://buckets/username/my-bucket --exclude "*.log"

# Combine include and exclude
>>> hf buckets sync ./data hf://buckets/username/my-bucket --include "*.safetensors" --exclude "*.tmp"
```

或者通过Python：

```py
# Only sync .txt files
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", include=["*.txt"])

# Exclude log files
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", exclude=["*.log"])

# Combine include and exclude
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", include=["*.safetensors"], exclude=["*.tmp"])
```

对于更复杂的过滤，请使用带有`--filter-from`（或Python中的`filter_from`）的过滤器文件：

```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket --filter-from filters.txt
```过滤器文件使用 `+`（包含）和 `-`（排除）前缀。以 `#` 开头的行是注释。规则按顺序评估（第一个匹配的规则获胜）：

```text
# filters.txt
- *.log
- *.tmp
+ *.safetensors
+ *.json
```

### 比较模式

默认情况下，sync 使用大小和修改时间来比较文件。您可以自定义此行为：

```bash
# Only compare sizes (ignore modification times)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-times

# Only compare modification times (ignore sizes)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-sizes

# Only update files that already exist on the receiver (skip new files)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --existing

# Only create new files (skip files that already exist on the receiver)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-existing
```

或者通过Python：

```py
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", ignore_times=True)
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", ignore_sizes=True)
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", existing=True)
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", ignore_existing=True)
```

### 计划并应用

对于关键操作，您可以在执行之前查看同步计划：

```bash
# Step 1: Generate a plan file (nothing is transferred)
>>> hf buckets sync ./data hf://buckets/username/my-bucket --plan sync-plan.jsonl
Sync plan: ./data -> hf://buckets/username/my-bucket
  Uploads: 3
  Downloads: 0
  Deletes: 0
  Skips: 1
Plan saved to: sync-plan.jsonl

# Step 2: Review the plan file (JSONL format)
>>> cat sync-plan.jsonl

# Step 3: Apply the plan
>>> hf buckets sync --apply sync-plan.jsonl
```

或者通过Python：

```py
# Step 1: Generate a plan file (nothing is transferred)
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", plan="sync-plan.jsonl")

# Step 2: Review the plan file (JSONL format), then apply
>>> sync_bucket(apply="sync-plan.jsonl")
```

> [!提示]
> 计划文件是一个 JSONL 文件，其中包含一个标题行，后跟每个操作一行。每个操作都包含操作（`upload`、`download`、`delete` 或 `skip`）、文件路径以及操作原因。您可以在应用之前手动编辑此文件，但请注意语法。

### 试运行

使用 `--dry-run` （或 Python 中的 `dry_run=True`）来获取同步计划而不执行任何操作：

```bash
# Preview what would be synced
>>> hf buckets sync ./data hf://buckets/username/my-bucket --dry-run | jq '.action'
```

或者通过Python：

```py
>>> plan = sync_bucket("./data", "hf://buckets/username/my-bucket", dry_run=True)
>>> plan.summary()
{'uploads': 3, 'downloads': 0, 'deletes': 0, 'skips': 1, 'total_size': 4096}
```

> [!提示]
> `--dry-run` 输出与 `--plan` 相同的 JSONL 格式，但打印到 stdout 而不是保存到文件。仅打印 JSONL 内容，使其可以安全地通过管道传输。

### 详细和安静模式

```bash
# Show per-file operations
>>> hf buckets sync ./data hf://buckets/username/my-bucket --verbose

# Suppress all output
>>> hf buckets sync ./data hf://buckets/username/my-bucket --quiet
```

或者通过Python：

```py
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", verbose=True)
>>> sync_bucket("./data", "hf://buckets/username/my-bucket", quiet=True)
```

## 高级

对于较低级别的用例，还可以使用以下方法：- [get_bucket_paths_info()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_bucket_paths_info)：在单个批量请求中获取有关存储桶中特定路径的信息。当您确切知道哪些文件需要元数据时很有用。

```py
>>> from huggingface_hub import get_bucket_paths_info
>>> for info in get_bucket_paths_info("username/my-bucket", ["file.txt", "models/model.safetensors"]):
...     print(info.path, info.size, info.xet_hash)
```

- [get_bucket_file_metadata()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_bucket_file_metadata)：获取单个文件的元数据（大小和 xet 数据）。由[download_bucket_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.download_bucket_files)内部使用。

```py
>>> from huggingface_hub import get_bucket_file_metadata
>>> metadata = get_bucket_file_metadata("username/my-bucket", "models/model.safetensors")
>>> metadata.size
42000
```

### 创建 CLI 扩展
https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/cli-extensions.md