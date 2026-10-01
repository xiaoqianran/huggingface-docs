<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhook 指南：使用作业处理存储桶中的新文件

[Storage Buckets](./storage-buckets) 是存放随时间推移到达的文件的好地方：从其他工具导出、日志、录音、扫描。您通常需要对这些文件执行一个工作流程，然后才能使用它们，例如转换、清理或转录它们。

存储桶上的 Webhook 和 [Job](./jobs) 可以自动完成此工作，无需服务器。当文件更改时，Webhook 会启动作业。该作业接收已更改文件的列表，对其进行处理并停止。您只需在运行时付费。

本指南解释了这一模式，然后通过一个示例进行设置：将上传的 CSV 和 JSON 文件转换为优化的 Parquet。

## 它是如何工作的

```text
file uploaded ──▶ webhook ──▶ Job runs with the event ──▶ result written to a second bucket
```

Webhook 可以 [trigger a Job](./jobs-webhooks) 而不是调用 URL。您创建作业一次，每个 Webhook 事件都会在其环境中再次运行该作业。对于存储桶，`WEBHOOK_PAYLOAD`列出了已更改的文件：

```json
{
  "event": { "action": "update", "scope": "repo.content" },
  "repo": { "type": "bucket", "name": "your-username/my-raw-files", ... },
  "updatedFiles": [
    { "path": "data.csv", "action": "add", "xetHash": "433b53b8...", "size": 10983 }
  ]
}
```

因此，作业仅处理新文件，并且它可以在下载任何内容之前决定如何处理每个文件。每次运行都会重用作业的映像、命令、环境变量、硬件风格和超时。它无法获取作业的卷数或秘密。这就是为什么以下步骤从 URL 运行脚本并将令牌放在 Webhook 上的原因。

该作业将其结果写入**第二个存储桶**。如果它写入它监视的存储桶，它自己的输出将再次触发 Webhook。

## 示例：将上传转化为优化的 Parquet

CSV 和 JSON 导出查询速度慢且难以共享。得益于 Xet 重复数据删除功能，[Optimized Parquet](./datasets-libraries#optimized-parquet-files) 的过滤和流式传输速度更快，上传和下载速度更快。在此示例中，上传到一个存储桶的每个 CSV、JSON 或 Parquet 文件在另一个存储桶中显示为优化的 Parquet。该作业运行 [⟦T10⟧](https://huggingface.co/datasets/uv-scripts/data-processing/blob/main/optimize-parquet.py)，这是来自 [uv-scripts](https://huggingface.co/uv-scripts) 的现成脚本。

您需要一个带有 [pre-paid credits](https://huggingface.co/settings/billing) 和 [⟦T11⟧ CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli#getting-started) 的 Hugging Face 帐户。

### 创建两个桶

一个存储桶接收您的上传，另一个存储桶接收 Parquet 文件：

```bash
hf buckets create my-raw-files --private
hf buckets create my-parquet --private
```

### 创建工作

创建 webhook 将运行的作业：

```bash
hf jobs run --flavor cpu-upgrade --timeout 2h \
    -e OUTPUT_BUCKET=your-username/my-parquet \
    ghcr.io/astral-sh/uv:python3.12-bookworm \
    uv run https://huggingface.co/datasets/uv-scripts/data-processing/raw/main/optimize-parquet.py
```该脚本从其带有 `hf jobs run` 的 URL 运行。 `hf jobs uv run` 是运行 UV 脚本的常用方法，但它将脚本上传到卷，这是 webhook 运行所没有的。 `cpu-upgrade` 足以完成这项工作，两个小时的超时为大文件留下了空间。

该脚本也可以位于私有存储桶或存储库中。不要使用监视的存储桶：上传脚本将启动作业。将 `--secrets HF_TOKEN` 添加到命令中，并带有可以读取脚本的令牌。如果运行失败并显示 `SyntaxError`，则令牌无法读取它。 Webhook 的令牌需要相同的访问权限，因为 Webhook 运行不会获取此秘密。

`hf jobs run`立即开始作业。第一次运行没有 webhook 事件，如果没有 webhook 事件，`optimize-parquet.py` 不会执行任何操作，因此作业会立即停止。复制它打印的作业 ID。

### 创建网络钩子

观察第一个存储桶并在它发生变化时运行您的作业：

```bash
hf webhooks create --job-id <job ID> --watch bucket:your-username/my-raw-files \
    --domain repo --secrets HF_TOKEN
```

`--domain repo` 将 Webhook 限制为文件和设置更改。存储桶仅发送这些内容，因此这是明确的而不是必要的，但如果您稍后查看模型或数据集，它会保持命令正确。`--secrets HF_TOKEN` 使用 webhook 存储加密的令牌。 Webhook 启动的每个作业都会将其作为秘密接收，并使用它来读取和写入存储桶。该值来自您环境中的`HF_TOKEN`，或者来自您登录时使用的令牌。仅具有作业所需权限的[fine-grained token](./security-tokens)是最安全的选择。

> [!提示]
> 要使用专门为此管道创建的令牌，请使用 `--secrets HF_TOKEN=hf_…` 显式传递该值，或者通过管道将其输入：`printf 'HF_TOKEN=hf_…\n' | hf webhooks create … --secrets-file -` 将其保留在 shell 历史记录之外。

该命令打印 webhook ID。要稍后停止管道，请使用 `hf webhooks delete <webhook ID>` 删除 Webhook。

### 上传文件

```bash
hf buckets cp data.csv hf://buckets/your-username/my-raw-files/data.csv
```

大约一分钟后，您的 [Jobs page](https://huggingface.co/settings/jobs) 上会出现一个作业。如果没有，请打开 [webhook settings](https://huggingface.co/settings/webhooks) 中 Webhook 的“活动”选项卡以查看交付情况，并使用 `hf jobs logs <job ID>` 读取作业的输出。当作业完成时，Parquet 文件位于第二个存储桶中：

```bash
hf buckets ls your-username/my-parquet -R
```

```text
data.csv/README.md
data.csv/data/train-00000-of-00001.parquet
```

要转换多个文件，请使用一个 `hf buckets sync` 上传它们。一起上传的文件通常作为一个事件到达，因此一项作业可转换所有文件，每个事件最多可转换 10,000 个文件。请参阅 [Webhooks](./webhooks#buckets) 了解整个铲斗有效负载以及超出该限制时会发生的情况。

### 脚本的作用

特定于 webhooks 的部分是读取事件：

```python
event = json.loads(os.environ.get("WEBHOOK_PAYLOAD", "{}"))  # empty when you run the Job yourself
input_bucket = os.environ.get("WEBHOOK_REPO_ID")

for changed_file in event.get("updatedFiles", []):
    if changed_file["action"] != "add":
        continue  # skip deleted files
    path, size = changed_file["path"], changed_file["size"]
    ...  # convert the file (see the full script)
```对于每个新文件，脚本会直接从存储桶 (`hf://buckets/...`) 中使用 `datasets` 加载它。如果文件大于作业可用磁盘的三分之一，则会进行流式传输而不是下载。然后，`push_to_hub` 将其作为 Parquet 写入输出存储桶，并具有内容定义的分块、页面索引和最多 100 MB 的行组。

## 使用该模式完成您自己的任务

保留设置并更改脚本。例如：

- **在训练前删除个人数据。** 在私有存储桶中收集原始文本，使用[GLiNER2 PII filter](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi)等模型编辑个人信息，然后仅将编辑后的文本推送到您的训练数据集。
- **评估新的检查点。** 训练作业将检查点保存到存储桶中，每个新检查点都会启动评估作业。如果检查点分多个部分上传，则仅当您最后编写的标记文件出现在`updatedFiles`中时才做出反应；事件可能会无序到达。
- **转录或嵌入新文件。** 将新录音转换为转录本，或将新文档转换为嵌入内容以供搜索。当您编写脚本时，请确定在未设置 `WEBHOOK_PAYLOAD` 时它在第一次运行时执行的操作。它可以像 `optimize-parquet.py` 一样退出，或者做实际工作，例如处理存储桶中已有的文件。如果第一次运行需要令牌，例如读取存储桶或私有脚本，请将`--secrets HF_TOKEN`添加到`hf jobs run`。 Webhook 运行不会获得此秘密。他们从 webhook 获取令牌。

## Webhook 还是时间表？

Webhook 会为每个事件启动一个作业，因此新文件会在到达后大约一分钟进行处理。如果不需要那么快地处理它们，[scheduled Job](./jobs-schedule)是一种替代方案：它每小时或每天运行一次，并处理自上次运行以来到达的所有内容。这会将工作分为更少的作业，这有助于处理启动缓慢的 GPU 任务，并且避免了每 24 小时 1,000 个触发器的 Webhook 限制。然后，脚本必须找到新文件本身，例如通过比较输入和输出存储桶。

### 秘密扫描
https://huggingface.co/docs/hub/security-secrets.md