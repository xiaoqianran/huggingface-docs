<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhooks 自动化

Webhooks 允许您监听 Hugging Face 上特定存储库或属于特定用户/组织集（不仅仅是您的存储库，而是任何存储库）的所有存储库的新更改。

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

Webhook 使用以下环境变量触发作业：

- `WEBHOOK_PAYLOAD`：JSON 字符串形式的完整 Webhook 负载
- `WEBHOOK_REPO_ID`：存储库名称（例如，`user/repo-name`）
- `WEBHOOK_REPO_TYPE`：存储库类型（`model`、`dataset` 或 `space`）
- `WEBHOOK_SECRET`：webhook 秘密（如果已配置）

Webhook 负载包含多个字段，以下是一些有用的字段：

```
- event:
  - action: one of "create", "delete", "move", "update"
  - scope: string
- repo:
  - owner: string
  - headSha: string
  - name: string
  - type: one of "dataset", "model", "space"
```

您可以在 [⟦T12⟧ Webhooks documentation](https://huggingface.co/docs/huggingface_hub/en/guides/webhooks) 中找到有关 webhooks 的更多信息。

### 模型卡
https://huggingface.co/docs/hub/model-cards.md