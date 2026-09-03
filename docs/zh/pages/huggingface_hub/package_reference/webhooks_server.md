<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhook 服务器

Webhook 是 MLOps 相关功能的基础。它们允许您监听特定存储库的新更改或
属于您感兴趣的特定用户/组织的所有存储库。学习
有关 Huggingface Hub 上的 Webhooks 的更多信息，您可以阅读 Webhooks [guide](https://huggingface.co/docs/hub/webhooks)。

> [!提示]
> 查看此 [guide](../guides/webhooks_server)，了解有关如何设置 Webhooks 服务器的分步教程以及
> 将其部署为空间。

> [!警告]
> 这是一项实验性功能。这意味着我们仍在努力改进 API。重大变化可能是
> 日后推出，恕不另行通知。确保在您的要求中固定`huggingface_hub`的版本。
> 使用实验性功能时会触发警告。您可以通过将 `HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1` 设置为环境变量来禁用它。

## 服务器

服务器是一个[Gradio](https://gradio.app/)应用程序。它有一个 UI 来为您或您的用户显示说明，还有一个 API
监听网络钩子。实现 webhook 端点就像装饰函数一样简单。然后就可以调试了
在将 Webhooks 部署到空间之前，将其重定向到您的计算机（使用 Gradio 隧道）。

### WebhooksServer[[huggingface_hub.WebhooksServer]]#### Huggingface_hub.WebhooksServer[[huggingface_hub.WebhooksServer]]

```python
huggingface_hub.WebhooksServer(*args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_server.py#L43)

**参数：**

ui (`gradio.Blocks`，可选)：用作 Space 登陆页面的 Gradio UI 实例。如果是 `None`，则会创建一个显示有关已配置 Webhook 的说明的 UI。

webhook_secret（`str`，可选）：用于验证传入 Webhook 请求的密钥。您可以将此值设置为任何您想要的秘密，只要您也在 [webhooks settings panel](https://huggingface.co/settings/webhooks) 中配置它即可。您还可以将此值设置为 `WEBHOOK_SECRET` 环境变量。如果未提供密码，则打开 Webhook 端点时没有任何安全性。

[WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer) 类可让您创建可以接收 Huggingface Webhook 的 Gradio 应用程序实例。
这些 webhook 可以使用 `add_webhook()` 装饰器注册。 Webhook 端点已添加到
该应用程序作为 FastAPI 路由器的 POST 端点。注册所有 Webhook 后，必须使用 `launch` 方法
调用以启动应用程序。

建议接受 [WebhookPayload](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhookPayload) 作为 webhook 函数的第一个参数。这是一个 Pydantic
包含有关 webhook 事件的所有信息的模型。系统会自动为您解析数据。查看 [webhooks guide](../guides/webhooks_server) 了解如何设置您的分步教程
WebhooksServer 并将其部署在 Space 上。

> [!警告]
> `WebhooksServer` 是实验性的。其 API 将来可能会发生变化。

> [!警告]
> 您必须安装`gradio`才能使用`WebhooksServer` (`pip install --upgrade gradio`)。

示例：

```python
import gradio as gr
from huggingface_hub import WebhooksServer, WebhookPayload

with gr.Blocks() as ui:
    ...

app = WebhooksServer(ui=ui, webhook_secret="my_secret_key")

@app.add_webhook("/say_hello")
async def hello(payload: WebhookPayload):
    return {"message": "hello"}

app.launch()
```

### @webhook_endpoint[[huggingface_hub.webhook_endpoint]]

#### Huggingface_hub.webhook_endpoint[[huggingface_hub.webhook_endpoint]]

```python
huggingface_hub.webhook_endpoint(path: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_server.py#L226)

**参数：**

path (`str`, 可选) : 注册webhook函数的URL路径。如果未提供，函数名称将用作路径。无论如何，所有 webhook 都在 `/webhooks` 下注册。

装饰器启动 [WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer) 并将装饰函数注册为 webhook 端点。

这是快速入门的好帮手。如果您需要更多灵活性（自定义登录页面或 webhook 秘密），
您可以直接使用[WebhooksServer()](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhooksServer)。您可以使用以下方法注册多个 Webhook 端点（到同一服务器）
这个装饰器多次。

查看 [webhooks guide](../guides/webhooks_server) 了解如何设置您的分步教程
服务器并将其部署在空间上。

> [!警告]
> `webhook_endpoint` 是实验性的。其 API 将来可能会发生变化。> [!警告]
> 您必须安装`gradio`才能使用`webhook_endpoint` (`pip install --upgrade gradio`)。

示例：
默认用法是将函数注册为 Webhook 端点。函数名称将用作路径。
服务器将在退出时自动启动（即在脚本结束时）。

```python
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload):
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Trigger a training job if a dataset is updated
        ...

# Server is automatically started at the end of the script.
```

高级用法：将函数注册为 webhook 端点并手动启动服务器。如果您这样做，这很有用
在笔记本上运行它。

```python
from huggingface_hub import webhook_endpoint, WebhookPayload

@webhook_endpoint
async def trigger_training(payload: WebhookPayload):
    if payload.repo.type == "dataset" and payload.event.action == "update":
        # Trigger a training job if a dataset is updated
        ...

# Start the server manually
trigger_training.launch()
```

## 有效负载[[huggingface_hub.WebhookPayload]]

[WebhookPayload](/docs/huggingface_hub/v1.30.0/en/package_reference/webhooks_server#huggingface_hub.WebhookPayload) 是包含 Webhooks 有效负载的主要数据结构。这是
一个 `pydantic` 类，这使得它非常容易与 FastAPI 一起使用。如果将其作为参数传递给 webhook 端点，它
将被自动验证并解析为 Python 对象。

有关 Webhooks Payload 的更多信息，您可以参考 Webhooks Payload [guide](https://huggingface.co/docs/hub/webhooks#webhook-payloads)。

#### Huggingface_hub.WebhookPayload[[huggingface_hub.WebhookPayload]]

```python
huggingface_hub.WebhookPayload(event: WebhookPayloadEvent, repo: WebhookPayloadRepo, discussion: huggingface_hub._webhooks_payload.WebhookPayloadDiscussion | None = None, comment: huggingface_hub._webhooks_payload.WebhookPayloadComment | None = None, webhook: WebhookPayloadWebhook, movedTo: huggingface_hub._webhooks_payload.WebhookPayloadMovedTo | None = None, updatedRefs: list[huggingface_hub._webhooks_payload.WebhookPayloadUpdatedRef] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L129)

### WebhookPayload[[huggingface_hub.WebhookPayload]]

#### Huggingface_hub.WebhookPayload[[huggingface_hub.WebhookPayload]]

```python
huggingface_hub.WebhookPayload(event: WebhookPayloadEvent, repo: WebhookPayloadRepo, discussion: huggingface_hub._webhooks_payload.WebhookPayloadDiscussion | None = None, comment: huggingface_hub._webhooks_payload.WebhookPayloadComment | None = None, webhook: WebhookPayloadWebhook, movedTo: huggingface_hub._webhooks_payload.WebhookPayloadMovedTo | None = None, updatedRefs: list[huggingface_hub._webhooks_payload.WebhookPayloadUpdatedRef] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L129)

### WebhookPayloadComment[[huggingface_hub.WebhookPayloadComment]]

#### Huggingface_hub.WebhookPayloadComment[[huggingface_hub.WebhookPayloadComment]]```python
huggingface_hub.WebhookPayloadComment(id: str, author: ObjectId, hidden: bool, content: str | None = None, url: WebhookPayloadUrl)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L94)

### WebhookPayloadDiscussion[[huggingface_hub.WebhookPayloadDiscussion]]

#### Huggingface_hub.WebhookPayloadDiscussion[[huggingface_hub.WebhookPayloadDiscussion]]

```python
huggingface_hub.WebhookPayloadDiscussion(id: str, num: int, author: ObjectId, url: WebhookPayloadUrl, title: str, isPullRequest: bool, status: typing.Literal['closed', 'draft', 'open', 'merged'], changes: huggingface_hub._webhooks_payload.WebhookPayloadDiscussionChanges | None = None, pinned: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L101)

### WebhookPayloadDiscussionChanges[[huggingface_hub.WebhookPayloadDiscussionChanges]]

#### Huggingface_hub.WebhookPayloadDiscussionChanges[[huggingface_hub.WebhookPayloadDiscussionChanges]]

```python
huggingface_hub.WebhookPayloadDiscussionChanges(base: str, mergeCommitId: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L89)

### WebhookPayloadEvent[[huggingface_hub.WebhookPayloadEvent]]

#### Huggingface_hub.WebhookPayloadEvent[[huggingface_hub.WebhookPayloadEvent]]

```python
huggingface_hub.WebhookPayloadEvent(action: typing.Literal['create', 'delete', 'move', 'update'], scope: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L84)

### WebhookPayloadMovedTo[[huggingface_hub.WebhookPayloadMovedTo]]

#### Huggingface_hub.WebhookPayloadMovedTo[[huggingface_hub.WebhookPayloadMovedTo]]

```python
huggingface_hub.WebhookPayloadMovedTo(name: str, owner: ObjectId)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L75)

### WebhookPayloadRepo[[huggingface_hub.WebhookPayloadRepo]]

#### Huggingface_hub.WebhookPayloadRepo[[huggingface_hub.WebhookPayloadRepo]]

```python
huggingface_hub.WebhookPayloadRepo(id: str, owner: ObjectId, head_sha: str | None = None, name: str, private: bool, subdomain: str | None = None, tags: list[str] | None = None, type: typing.Literal['dataset', 'model', 'space'], url: WebhookPayloadUrl)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L112)

### WebhookPayloadUrl[[huggingface_hub.WebhookPayloadUrl]]

#### Huggingface_hub.WebhookPayloadUrl[[huggingface_hub.WebhookPayloadUrl]]

```python
huggingface_hub.WebhookPayloadUrl(web: str, api: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L70)

### WebhookPayloadWebhook[[huggingface_hub.WebhookPayloadWebhook]]

#### Huggingface_hub.WebhookPayloadWebhook[[huggingface_hub.WebhookPayloadWebhook]]

```python
huggingface_hub.WebhookPayloadWebhook(id: str, version: typing.Literal[3])
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_webhooks_payload.py#L80)### HF URI
https://huggingface.co/docs/huggingface_hub/v1.30.0/package_reference/hf_uris.md