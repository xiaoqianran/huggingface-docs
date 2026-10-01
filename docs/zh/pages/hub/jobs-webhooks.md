<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhooks 自动化

Webhooks 允许您侦听 Hugging Face 上特定存储库或存储桶或属于特定用户/组织集（不仅仅是您的存储库，而是任何存储库）的所有存储库的新更改。

在 `huggingface_hub` Python 客户端中使用 `create_webhook` 创建一个 Webhook，当 Hugging Face 存储库中发生更改时触发作业：

```python
from huggingface_hub import create_webhook

# Example: Creating a webhook that triggers a Job
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

要在 [bucket](./storage-buckets) 添加或删除文件时运行作业，请监视存储桶：

```python
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "bucket", "name": "your-username/your-bucket"}],
    domains=["repo"],
    secrets={"HF_TOKEN": "hf_***"},
)
```

有关完整示例，请参阅[Process new files in a bucket with Jobs](./webhooks-guide-bucket-jobs)。

Webhook 使用以下环境变量触发作业：

- `WEBHOOK_PAYLOAD`：JSON 字符串形式的完整 Webhook 负载
- `WEBHOOK_REPO_ID`：存储库或存储桶名称（例如，`user/repo-name`）
- `WEBHOOK_REPO_TYPE`：存储库类型（`model`、`dataset`、`space` 或 `bucket`）
- `WEBHOOK_SECRET`：webhook 秘密（如果已配置）
- `WEBHOOK_ID`：交付的唯一标识符，在该交付的重试中保持稳定

> [!警告]
> Webhook 运行不会保留作业的卷或其秘密。传递Job需要的秘密，例如`HF_TOKEN`，与`create_webhook`中的`secrets=`。要从 Webhook 运行 UV 脚本，请使用 `hf jobs run <image> uv run <url>` 定义您的作业，否则 `hf jobs uv run` 会将脚本上传到 Webhook 运行没有的卷。Webhook 负载包含多个字段，以下是一些有用的字段：

```
- event:
  - action: one of "create", "delete", "move", "update"
  - scope: string
- repo:
  - owner: string
  - headSha: string (not sent for buckets)
  - name: string
  - type: one of "dataset", "model", "space", "bucket"
- updatedFiles: for bucket file changes, the list of added and deleted files
```

您可以在 [⟦T20⟧ Webhooks documentation](https://huggingface.co/docs/huggingface_hub/en/guides/webhooks) 中找到有关 webhook 的更多信息。

### 如何使用 Okta 配置 OIDC SSO
https://huggingface.co/docs/hub/security-sso-okta-oidc.md