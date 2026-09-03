<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 网络钩子

Webhook 是 MLOps 相关功能的基础。它们允许您监听特定存储库或属于您感兴趣的特定用户/组织的所有存储库的新更改。本指南将首先解释如何以编程方式管理 Webhooks。然后我们将了解如何利用 `huggingface_hub` 创建一个监听 webhook 的服务器并将其部署到空间。

本指南假设您熟悉 Huggingface Hub 上的 Webhook 概念。要了解有关 webhooks 本身的更多信息，您应该首先阅读此 [guide](https://huggingface.co/docs/hub/webhooks)。

## 管理 Webhooks

`huggingface_hub` 允许您以编程方式管理您的 webhook。您可以列出现有的 Webhook、创建新的 Webhook，以及更新、启用、禁用或删除它们。本节将指导您完成使用 Hugging Face Hub 的 API 函数的过程。

### 创建 Webhook

要创建新的 Webhook，请使用 [create_webhook()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_webhook) 并指定应发送有效负载的 URL、应监视哪些事件，并可选择设置域和安全密钥。

```python
from huggingface_hub import create_webhook

# Example: Creating a webhook
webhook = create_webhook(
    url="https://webhook.site/your-custom-url",
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

Webhook 还可以触发作业在 Hugging 脸部基础设施上运行，而不是将有效负载发送到 URL。
在这种情况下，您需要传递源作业的 ID。```python
from huggingface_hub import create_webhook

# Example: Creating a webhook that triggers a Job
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

Webhook 使用环境变量 `WEBHOOK_PAYLOAD` 中的 Webhook 负载触发作业。
有关拥抱脸部作业、可用硬件（CPU、GPU）和 UV 脚本的更多信息，请参阅[Jobs documentation](./jobs)。

### 列出 Webhooks

要查看您已配置的所有 Webhook，您可以使用 [list_webhooks()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_webhooks) 列出它们。这对于查看其 ID、URL 和状态很有用。

```python
from huggingface_hub import list_webhooks

# Example: Listing all webhooks
webhooks = list_webhooks()
for webhook in webhooks:
    print(webhook)
```

### 更新 Webhook

如果您需要更改现有 Webhook 的配置，例如 URL 或其监视的事件，您可以使用 [update_webhook()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_webhook) 进行更新。

```python
from huggingface_hub import update_webhook

# Example: Updating a webhook
updated_webhook = update_webhook(
    webhook_id="your-webhook-id",
    url="https://new.webhook.site/url",
    watched=[{"type": "user", "name": "new-username"}],
    domains=["repo"]
)
```

### 启用和禁用 Webhooks

您可能想要暂时禁用 Webhook 而不将其删除。这可以使用 [disable_webhook()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.disable_webhook) 来完成，并且稍后可以使用 [enable_webhook()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.enable_webhook) 重新启用 Webhook。

```python
from huggingface_hub import enable_webhook, disable_webhook

# Example: Enabling a webhook
enabled_webhook = enable_webhook("your-webhook-id")
print("Enabled:", enabled_webhook)

# Example: Disabling a webhook
disabled_webhook = disable_webhook("your-webhook-id")
print("Disabled:", disabled_webhook)
```

### 删除 Webhook

当不再需要 Webhook 时，可以使用 [delete_webhook()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_webhook) 将其永久删除。

```python
from huggingface_hub import delete_webhook

# Example: Deleting a webhook
delete_webhook("your-webhook-id")
```

### 从 CLI 管理 Webhooks

上述所有操作也可以通过 `hf` 命令行界面进行。这对于脚本编写和代理工作流程特别有用。

```bash
# List all webhooks
>>> hf webhooks ls

# Get details about a specific webhook
>>> hf webhooks info <webhook_id>

# Create a webhook that pings a URL
>>> hf webhooks create --url https://example.com/hook --watch model:bert-base-uncased --domain repo

# Create a webhook that triggers a Job
>>> hf webhooks create --job-id <job_id> --watch user:julien-c

# Update an existing webhook
>>> hf webhooks update <webhook_id> --url https://new-url.com/hook

# Enable / disable a webhook
>>> hf webhooks enable <webhook_id>
>>> hf webhooks disable <webhook_id>

# Delete a webhook (with confirmation prompt)
>>> hf webhooks delete <webhook_id>
````--watch` 选项使用格式 `type:name`，其中 type 为 `model`、`dataset`、`space`、`org` 或 `user` 之一。可以重复观看多个项目。要获得完整的选项列表，请运行 `hf webhooks <command> --help`。欲了解更多详情，请参阅[CLI guide](./cli#hf-webhooks)。

## Webhook 服务器

我们将在本指南部分中使用的基类是[WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer)。这是一个用于轻松配置服务器的类
可以从 Huggingface Hub 接收 webhooks。该服务器基于 [Gradio](https://gradio.app/) 应用程序。它有一个用户界面
为您或您的用户显示说明以及用于侦听 Webhook 的 API。

> [!提示]
> 要查看 Webhook 服务器的运行示例，请查看 [Spaces CI Bot](https://huggingface.co/spaces/spaces-ci-bot/webhook)
> 一。当在空间上打开 PR 时，它会启动临时环境。

> [!警告]
> 这是[experimental feature](../package_reference/environment_variables#hfhubdisableexperimentalwarning)。这个
> 表示我们仍在努力改进 API。未来可能会在没有事先通知的情况下引入重大变更
> 注意。确保在您的要求中固定`huggingface_hub`的版本。

### 创建端点

实现 webhook 端点就像装饰函数一样简单。让我们看第一个例子来解释主要内容
概念：

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Trigger a training job if a dataset is updated
        ...
```

将此代码片段保存在名为 `'app.py'` 的文件中，并使用 `'python app.py'` 运行它。您应该看到如下消息：

```text
Webhook secret is not defined. This means your webhook endpoints will be open to everyone.
To add a secret, set `WEBHOOK_SECRET` as environment variable or pass it at initialization:
        `app = WebhooksServer(webhook_secret='my_secret', ...)`
For more details about webhook secrets, please refer to https://huggingface.co/docs/hub/webhooks#webhook-secret.
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://1fadb0f52d8bf825fc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces

Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
Go to https://huggingface.co/settings/webhooks to setup your webhooks.
```好工作！您刚刚启动了一个 Webhook 服务器！让我们来详细分析一下到底发生了什么：

1. 通过用[webhook_endpoint()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.webhook_endpoint)修饰函数，在后台创建了一个[WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer)对象。
如您所见，该服务器是一个在 http://127.0.0.1:7860 上运行的 Gradio 应用程序。如果您在浏览器中打开此 URL，您
将看到一个登陆页面，其中包含有关已注册 Webhook 的说明。
2. Gradio 应用程序本质上是一个 FastAPI 服务器。已添加新的 POST 路由`/webhooks/trigger_training`。
这是监听 webhook 并在触发时运行 `trigger_training` 函数的路由。 FastAPI 将
自动解析有效负载并将其作为 [WebhookPayload](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhookPayload) 对象传递给函数。这是一个`pydantic`对象
其中包含有关触发 Webhook 的事件的所有信息。
3. Gradio应用程序还打开了一个隧道来接收来自互联网的请求。这是有趣的部分：你可以
在 https://huggingface.co/settings/webhooks 上配置指向本地计算机的 Webhook。这对于
调试您的 webhook 服务器并在将其部署到空间之前快速迭代。
4. 最后，日志还告诉您，您的服务器当前没有受到机密保护。这对于本地调试，但稍后要记住。

> [!警告]
> 默认情况下，服务器在脚本末尾启动。如果您在笔记本上运行它，则可以启动
> 通过调用`decorated_function.launch()`手动服务器。由于使用了唯一的服务器，因此您只需启动服务器即可
> 一次，即使您有多个端点。

### 配置 Webhook

现在您已经运行了 Webhook 服务器，您想要配置 Webhook 以开始接收消息。
转到 https://huggingface.co/settings/webhooks，单击“添加新的 Webhook”并配置您的 Webhook。设定目标
您想要观看的存储库和 Webhook URL，此处为 `https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training`。

就是这样！您现在可以通过更新目标存储库（例如推送提交）来触发该 Webhook。检查
Webhook 的“活动”选项卡可查看已触发的事件。现在您已经有了工作设置，您可以
测试它并快速迭代。如果您修改代码并重新启动服务器，您的公共 URL 可能会更改。确保
如果需要，更新集线器上的 Webhook 配置。

### 部署到空间现在您已经有了一个可用的 Webhook 服务器，目标是将其部署到空间。前往 https://huggingface.co/new-space
创建一个空间。为其命名，选择 Gradio SDK，然后单击“创建空间”。将您的代码上传到空间
在一个名为 `app.py` 的文件中。您的空间将自动启动！有关Spaces的更多详细信息，请参阅此
[guide](https://huggingface.co/docs/hub/spaces-overview)。

您的 webhook 服务器现在正在公共空间上运行。在大多数情况下，您需要用秘密来保护它。前往
您的空间设置 >“存储库机密”>“添加机密”部分。将 `WEBHOOK_SECRET` 环境变量设置为
您选择的价值。返回[Webhooks settings](https://huggingface.co/settings/webhooks)并设置
webhook 配置中的秘密。现在，只有具有正确密钥的请求才会被您的服务器接受。

就是这样！您的空间现在已准备好接收来自 Hub 的 Webhook。请记住，如果您运行 Space
在免费的“cpu-basic”硬件上，它将在 48 小时不活动后关闭。如果您需要永久空间，您
应考虑设置为[upgraded hardware](https://huggingface.co/docs/hub/spaces-gpus#hardware-specs)。

### 高级用法

上面的指南解释了设置[WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer)的最快方法。在本节中，我们将了解如何自定义
更进一步。

#### 多个端点您可以在同一服务器上注册多个端点。例如，您可能希望触发一个端点
一项训练工作和另一项触发模型评估的工作。您可以通过添加多个 `@webhook_endpoint` 来做到这一点
装饰器：

```python
# app.py
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload) -> None:
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Trigger a training job if a dataset is updated
        ...

@webhook_endpoint
async def trigger_evaluation(payload: WebhookPayload) -> None:
    if payload.repo.type == "model" and payload.event.action == "update":
        # Trigger an evaluation job if a model is updated
        ...
```

这将创建两个端点：

```text
(...)
Webhooks are correctly setup and ready to use:
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_training
  - POST https://1fadb0f52d8bf825fc.gradio.live/webhooks/trigger_evaluation
```

#### 自定义服务器

为了获得更大的灵活性，您还可以直接创建一个 [WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer) 对象。如果您愿意，这很有用
自定义服务器的登陆页面。您可以通过传递 [Gradio UI](https://gradio.app/docs/#blocks) 来做到这一点
这将覆盖默认的。例如，您可以为用户添加说明或手动添加表单
触发 webhook。创建 [WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer) 时，您可以使用以下方法注册新的 webhooks
`add_webhook()` 装饰器。

这是一个完整的例子：

```python
import gradio as gr
from fastapi import Request
from huggingface_hub import WebhooksServer, WebhookPayload

# 1. Define  UI
with gr.Blocks() as ui:
    ...

# 2. Create WebhooksServer with custom UI and secret
app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

# 3. Register webhook with explicit name
@app.add_webhook("/say_hello")
async def hello(payload: WebhookPayload):
    return {"message": "hello"}

# 4. Register webhook with implicit name
@app.add_webhook
async def goodbye(payload: WebhookPayload):
    return {"message": "goodbye"}

# 5. Start server (optional)
app.launch()
```1. 我们使用 Gradio 块定义自定义 UI。此 UI 将显示在服务器的登陆页面上。
2. 我们创建一个具有自定义 UI 和密钥的 [WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer) 对象。秘密是可选的，可以设置
`WEBHOOK_SECRET` 环境变量。
3. 我们注册一个具有显式名称的 Webhook。这将在`/webhooks/say_hello`处创建一个端点。
4. 我们使用隐式名称注册一个 Webhook。这将在`/webhooks/goodbye`处创建一个端点。
5.我们启动服务器。这是可选的，因为您的服务器将在脚本结束时自动启动。

### 引擎盖下的沙箱
https://huggingface.co/docs/huggingface_hub/v1.30.0/concepts/sandbox.md