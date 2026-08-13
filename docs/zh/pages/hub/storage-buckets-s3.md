<!-- huggingface-docs: machine-translated zh-CN from English source -->

# S3 兼容性

存储桶可以通过 **S3 兼容的 API** 进行访问，让您可以针对存储桶使用现有的 S3 工具（AWS CLI、`boto3`、`s5cmd` 和大多数其他 S3 SDK），而无需更改代码。 
请求通过`https://s3.hf.co`的网关服务。

S3 API 是访问存储桶数据的多种方法之一 - 对于无需单独的 S3 凭据的 Hugging Face 原生访问（`hf` CLI、`hf://` 路径和文件系统挂载），请参阅[Access Patterns](./storage-buckets-access)。

> [!注意]
> S3 API 仅适用于 [Storage Buckets](./storage-buckets)。它不公开其他 Hugging Face 存储库类型（模型、数据集、空间）。

## 生成 S3 凭证

网关使用源自 Hugging Face [User Access Token](./security-tokens) 的 AWS 风格访问密钥进行身份验证。

1. 前往您的[Access Tokens settings](https://huggingface.co/settings/tokens)。如果您还没有令牌，请使用 **创建新令牌** 按钮创建一个令牌。令牌的权限将成为 S3 凭证的权限 - 选择 **Read** 以获得对存储桶的只读访问权限，或选择 **Write** 以获得读写访问权限。
2. 在列表中找到令牌，打开其下拉菜单，然后选择 **生成 S3 凭证**。3. 将生成的 **访问密钥 ID**（前缀 `HFAK…`）和 **秘密访问密钥** 复制到安全的地方 — 秘密仅显示一次。

S3 凭证继承底层访问令牌的权限。对于细粒度令牌，请将其范围仅限于您打算使用的命名空间和存储桶。

## 配置客户端

将您的 S3 客户端指向网关端点并设置一些必需的选项。在端点 URL 中使用您的 Hugging Face **命名空间**（您的用户名或组织名称） - 请参阅下面的[Addressing buckets](#addressing-buckets)。|设置|价值|为什么 |
|--------------------------------|--------------------------------|----------------------------------------------------------------------------------------|
| `endpoint_url` | `https://s3.hf.co/<namespace>` |网关，范围仅限于您的命名空间 |
| `region` | `us-east-1` |必需的;网关目前是单区域 |
| `s3.addressing_style` | `path` |存储桶被寻址为路径段，而不是子域 |
| `request_checksum_calculation` | `when_required` |防止最近的客户端发送尾随校验和 |
| `response_checksum_validation` | `when_required` |防止最近的客户端在响应中期待校验和 |

这两个校验和设置对于最近的客户端很重要：AWS CLI ≥ 2.23 和最新的 `boto3` 版本默认发送尾随 CRC32 校验和（通过 `aws-chunked` 帧），网关不会解析该校验和。 
这些设置告诉客户端仅在操作严格要求时才发送校验和。

以下是可选的，但建议使用尽可能少的多部分部分上传：|设置|价值|
|--------------------------|-------|
| `s3.multipart_threshold` | `2GB` |
| `s3.multipart_chunksize` | `2GB` |

### 示例：AWS CLI 配置文件

添加配置文件到`~/.aws/config`：

```ini
[profile hf]
region = us-east-1
endpoint_url = https://s3.hf.co/<namespace>
s3 =
    addressing_style = path
    multipart_threshold = 2GB
    multipart_chunksize = 2GB
request_checksum_calculation = when_required
response_checksum_validation = when_required
```
注意：将上面的`<namespace>`替换为存储桶所在的用户名或组织。

将配置文件的匹配凭据添加到`~/.aws/credentials`：

```ini
[hf]
aws_access_key_id = HFAK...
aws_secret_access_key = ...
```

然后将任何 S3 命令与配置文件一起使用：

```bash
aws --profile hf s3 ls
aws --profile hf s3 mb s3://my-bucket
aws --profile hf s3 cp ./model.safetensors s3://my-bucket/models/model.safetensors
```

## 寻址桶

AWS S3 使用单一平面、全局唯一的存储桶名称空间，并且 SDK 期望存储桶名称是不带 `/` 的纯字符串。 Hugging Face 存储桶被标识为 `namespace/bucket`，其中 `namespace` 是您的用户名或组织。这个额外的级别引入了与 S3 客户端的不匹配——许多客户端不接受存储桶名称中的`/`，或者会错误地对其进行 URL 转义。有两种方法可以解决这个问题：

**1.将命名空间放入端点 URL**（大多数情况下建议）。这将每个操作的范围限定在该命名空间，因此传递给客户端的存储桶名称只是 HF 存储桶名称。它适用于您自己的存储桶或单个组织中的存储桶，但会跨命名空间分解 - 例如从个人存储桶到组织存储桶的服务器端副本。

```bash
aws --endpoint-url https://s3.hf.co/my-org s3api get-object \
  --bucket my-bucket --key some/object.txt ./object.txt
```**2.将命名空间视为存储桶**，并将 HF 存储桶名称添加到对象键之前。这适用于对象级操作（上传、下载），但对于存储桶级操作（例如创建或删除存储桶）存在问题。

```bash
aws --endpoint-url https://s3.hf.co s3api get-object \
  --bucket my-org --key my-bucket/some/object.txt ./object.txt
```

## 与 AWS S3 的限制和差异

由于存储桶并不对每个 S3 概念进行建模，因此某些行为会有所不同或不受支持。

### 对象下载

网关目前是单区域的。为了提高下载性能，`GetObject` 通常会响应 HTTP 302 重定向到最近的 Hugging Face CDN 边缘，而不是直接提供字节。

某些 SDK 不遵循来自 S3 端点的重定向，因此网关会检测标识为 `aws-cli`、`botocore`（涵盖 `boto3`）或 `aws-sdk-rust` 的客户端，并通过自身为它们代理数据。所有其他客户端（`rclone`、`s5cmd`、`curl`、AWS Go SDK 等）都会接收 302 并以本机方式遵循它，从而使网关远离数据路径以加快下载速度。

### 对象键命名

存储桶对象键比 S3 受到更多限制。密钥不得**：- 以`/`开始或结束
- 包含连续斜杠 (`//`)
- 包含`../`序列
- 以`./`开头
- 以`..`结尾
- 包含反斜杠 (`\`) 或空字节 (`\0`)

### 列表对象

- 不支持`ListObjectsV1` — 使用`ListObjectsV2`。请注意，某些客户端（例如 rclone） 
  可能需要配置为专门使用`ListObjectsV2`。
- 仅允许使用`/`作为分隔符。

### 其他 API 差异

- **对象元数据**：不存储或返回任意用户元数据（`x-amz-meta-*`）。支持`Content-Type`。
- **不支持的功能**：不支持 ACL、存储桶策略、对象标记、对象版本控制、生命周期规则、服务器端加密 (SSE) 和存储桶通知。对象始终使用 `STANDARD` 存储类别进行报告。相关请求标头和参数被接受但被忽略。
- **CopyObject**：服务器端复制仅在单个命名空间内工作。不支持跨命名空间复制和`UploadPartCopy`（将现有对象的一部分复制到分段上传）。
- **有条件请求**：`If-Match` / `If-None-Match` 前提条件在 `PutObject` 和 `CopyObject` 的复制源上得到满足，但在 `GetObject` 上不得到满足。- **分段上传过期**：从未完成或中止的正在进行的分段上传将自动过期并在 7 天后清理。

## 示例

常见任务的真实配方。每个都建立在上面的[client configuration](#configuring-a-client)之上。

### 使用`boto3`读写

[⟦T72⟧](https://docs.aws.amazon.com/boto3/latest/) 与上面的[client settings](#configuring-a-client) 网关配合使用。将 `<namespace>` 替换为您的用户名或组织：

```python
import boto3
from botocore.config import Config

s3 = boto3.client(
    "s3",
    endpoint_url="https://s3.hf.co/<namespace>",
    aws_access_key_id="HFAK...",
    aws_secret_access_key="...",
    config=Config(
        region_name="us-east-1",
        s3={"addressing_style": "path"},
        request_checksum_calculation="when_required",
        response_checksum_validation="when_required",
    ),
)

s3.upload_file("model.safetensors", "my-bucket", "models/model.safetensors")
s3.download_file("my-bucket", "models/model.safetensors", "model.safetensors")
```

### 使用 DuckDB 查询存储桶

通过 `httpfs` 扩展，[DuckDB](https://duckdb.org/) 可以直接从存储桶读取 Parquet（和其他格式）：

```sql
INSTALL httpfs;
LOAD httpfs;

CREATE SECRET hf (
    TYPE s3,
    KEY_ID 'HFAK...',
    SECRET '...',
    ENDPOINT 's3.hf.co/<namespace>',
    URL_STYLE 'path',
    REGION 'us-east-1'
);

SELECT * FROM read_parquet('s3://my-bucket/data.parquet');
```

> [!注意]
> 需要`URL_STYLE 'path'`。如果没有它，DuckDB 将使用虚拟托管式寻址 (`my-bucket.s3.hf.co`)，而网关不提供服务 — 您将看到“无法解析主机名”错误。

### 使用`rclone`导入数据

[⟦T78⟧](https://rclone.org/) 是在两个 S3 兼容存储之间复制数据的便捷方法，因此非常适合将现有 AWS S3 存储桶（或任何 S3 兼容源）移动到存储桶中。

这个想法是声明两个遥控器 - 您的源存储桶和 Hugging Face 网关 - 并让 `rclone` 在它们之间传输对象。将两个遥控器添加到`~/.config/rclone/rclone.conf`。第一个点指向您现有的 S3 存储桶；调整它以匹配您的源（此处为普通 AWS S3）：

```ini
[aws]
type = s3
provider = AWS
access_key_id = AKIA...
secret_access_key = ...
region = us-east-1
```第二个点位于拥抱脸网关。与任何其他客户端一样，将端点范围限定为 [namespace](#addressing-buckets)，使用 `path` 寻址，强制使用 ListObjectsV2（网关支持的唯一列表版本），并设置较大的多部分大小，以便上传使用尽可能少的部分：

```ini
[hf]
type = s3
provider = Other
endpoint = https://s3.hf.co/<namespace>
access_key_id = HFAK...
secret_access_key = ...
region = us-east-1
force_path_style = true
list_version = 2
upload_cutoff = 2G
chunk_size = 2G
```

注意：将上面的`<namespace>`替换为存储桶所在的用户名或组织，并使用从访问令牌生成的[S3 credentials](#generating-s3-credentials)。目标存储桶必须已存在于该命名空间下。

现在将源存储桶复制到您的存储桶中：

```bash
rclone copy aws:my-source-bucket hf:my-bucket --progress
```

`rclone copy` 仅传输在目的地丢失或更改的对象，因此重新运行以恢复中断的导入或拾取新对象是安全的。要使目标成为源的精确镜像（删除目标中不再存在于源中的对象），请改用 `rclone sync`：

```bash
rclone sync aws:my-source-bucket hf:my-bucket --progress
```

> [!提示]
> 对于大型导入，添加`--transfers`和`--checkers`以提高并发性（例如`--transfers 16 --checkers 16`），然后运行`rclone check aws:my-source-bucket hf:my-bucket`以确认每个对象都已通过。

### 带 DVC 的版本数据[DVC](https://dvc.org/) 通过在存储库中保留小指针文件并将实际数据推送到远程来在 git 中版本化数据集和模型。存储桶通过网关充当 DVC 远程，因此您的代码和 `.dvc` 指针保留在 git 中，而数据存储在您的存储桶中。安装支持 S3 的 DVC (`pip install 'dvc[s3]'`)，然后使用上面的 [client settings](#configuring-a-client) 添加遥控器：

```bash
dvc remote add -d hf-bucket s3://my-bucket/dvc-store
dvc remote modify hf-bucket endpointurl https://s3.hf.co/<namespace>
dvc remote modify hf-bucket region us-east-1
```

通过环境传递 [S3 credentials](#generating-s3-credentials)，或者使用 `dvc remote modify --local`，这样它们就不会出现在 git 中：

```bash
export AWS_ACCESS_KEY_ID=HFAK...
export AWS_SECRET_ACCESS_KEY=...
```

然后像往常一样跟踪、推送和拉取 — `dvc push` 上传到存储桶，而新克隆上的 `dvc pull` 将其下载回来：

```bash
dvc add data/
git add data.dvc .gitignore .dvc/config && git commit -m "Track data with DVC"
dvc push
```

> [!注意]
> 在 `endpointurl` 中使用您的 [namespace](#addressing-buckets) 并在 `s3://` URL 中使用裸存储桶名称。

### 优化
https://huggingface.co/docs/hub/datasets-polars-optimizations.md