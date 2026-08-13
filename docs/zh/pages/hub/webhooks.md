<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 网络钩子

Webhook 是 MLOps 相关功能的基础。它们允许您监听特定存储库或属于特定用户/组织集的所有存储库（不仅仅是您的存储库，而是任何存储库）的新更改。

您可以使用它们自动转换模型、构建社区机器人或为您的模型、数据集和空间（以及更多！）构建 CI/CD。 Webhook 还可以[trigger Jobs](./jobs-webhooks) 自动执行计算任务以响应存储库事件。

Webhooks 的文档如下 - 或者您也可以浏览我们的**指南**，其中展示了 Webhooks 的一些可能的用例：
- [Fine-tune a new model whenever a dataset gets updated (Python)](./webhooks-guide-auto-retrain)
- [Create a discussion bot on the Hub, using a LLM API (NodeJS)](./webhooks-guide-discussion-bot)
- [Create metadata quality reports (Python)](./webhooks-guide-metadata-review)
- 还有更多即将推出...

## 创建您的 Webhook

您可以创建新的 Webhooks 并编辑 Webhooks 中的现有 Webhook [settings](https://huggingface.co/settings/webhooks)：

![Settings of an individual webhook](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhook-settings.png)

Webhooks 可以监视存储库更新、拉取请求、讨论和新评论。甚至可以创建一个空间来对您的 Webhooks 做出反应！

## Webhook 有效负载

注册 Webhook 后，您将通过对指定目标 URL 的 `HTTP POST` 调用收到新事件通知。有效负载以 JSON 形式编码。

您可以在 Webhook 设置页面的活动选项卡中查看发送的有效负载的历史记录，还可以重播过去的 Webhook 以方便调试：

![image.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhook-activity.png)例如，以下是打开 Pull 请求时的完整负载：

```json
{
  "event": {
    "action": "create",
    "scope": "discussion"
  },
  "repo": {
    "type": "model",
    "name": "openai-community/gpt2",
    "id": "621ffdc036468d709f17434d",
    "private": false,
    "url": {
      "web": "https://huggingface.co/openai-community/gpt2",
      "api": "https://huggingface.co/api/models/openai-community/gpt2"
    },
    "owner": {
      "id": "628b753283ef59b5be89e937"
    }
  },
  "discussion": {
    "id": "6399f58518721fdd27fc9ca9",
    "title": "Update co2 emissions",
    "url": {
      "web": "https://huggingface.co/openai-community/gpt2/discussions/19",
      "api": "https://huggingface.co/api/models/openai-community/gpt2/discussions/19"
    },
    "status": "open",
    "author": {
      "id": "61d2f90c3c2083e1c08af22d"
    },
    "num": 19,
    "isPullRequest": true,
    "changes": {
      "base": "refs/heads/main"
    }
  },
  "comment": {
    "id": "6399f58518721fdd27fc9caa",
    "author": {
      "id": "61d2f90c3c2083e1c08af22d"
    },
    "content": "Add co2 emissions information to the model card",
    "hidden": false,
    // Note: when `hidden` is `true`, `content` will be undefined
    "url": {
      "web": "https://huggingface.co/openai-community/gpt2/discussions/19#6399f58518721fdd27fc9caa"
    }
  },
  "webhook": {
    "id": "6390e855e30d9209411de93b",
    "version": 3
  }
}
```

### 活动

顶级属性 `event` 始终被指定并用于确定事件的性质。

它有两个子属性：`event.action`和`event.scope`。

`event.scope` 将是以下值之一：

- `"repo"` - 回购协议上的全球事件。关联的 `action` 的可能值：`"create"`、`"delete"`、`"update"`、`"move"`。
- `"repo.content"` - 存储库内容的事件，例如新提交或标签。由于新创建的引用/提交，它也会触发新的拉取请求。关联的 `action` 始终为 `"update"`。
- `"repo.config"` - 配置上的事件：更新空间秘密、更新设置、更新 DOI、禁用与否等。关联的 `action` 始终为 `"update"`。
- `"discussion"` - 创建讨论或拉取请求，更新标题或状态，然后合并。关联的 `action` 的可能值：`"create"`、`"delete"`、`"update"`。
- `"discussion.comment"` - 创建、更新和隐藏评论。相关 `action` 的可能值：`"create"`、`"update"`。

将来可以添加更多范围。为了处理未知事件，您的 Webhook 处理程序可以将缩小范围内的任何操作视为更广泛范围内的 `"update"` 操作。例如，如果将来添加 `"repo.config.dois"` 范围，则您的 webhook 处理程序可以将具有该范围的任何事件视为 `"repo.config"` 范围上的 `"update"` 操作。

### 回购协议

在当前版本的 webhooks 中，始终指定顶级属性 `repo`，因为事件始终可以与存储库关联。例如，考虑以下值：

```json
"repo": {
	"type": "model",
	"name": "some-user/some-repo",
	"id": "6366c000a2abcdf2fd69a080",
	"private": false,
	"url": {
		"web": "https://huggingface.co/some-user/some-repo",
		"api": "https://huggingface.co/api/models/some-user/some-repo"
	},
	"headSha": "c379e821c9c95d613899e8c4343e4bfee2b0c600",
	"owner": {
		"id": "61d2000c3c2083e1c08af22d"
	}
}
```

`repo.headSha` 是存储库的 `main` 分支上最新提交的 sha。仅当`event.scope`以`"repo"`开头时发送，而不是在讨论和评论等社区活动中发送。

### 代码更改

在代码更改时，顶级属性 `updatedRefs` 在存储库事件上指定。它是一组已更新的参考文献。这是一个示例值：

```json
"updatedRefs": [
  {
    "ref": "refs/heads/main",
    "oldSha": "ce9a4674fa833a68d5a73ec355f0ea95eedd60b7",
    "newSha": "575db8b7a51b6f85eb06eee540738584589f131c"
  },
  {
    "ref": "refs/tags/test",
    "oldSha": null,
    "newSha": "575db8b7a51b6f85eb06eee540738584589f131c"
  }
]
```

新创建的引用会将 `oldSha` 设置为 `null`。删除的引用会将 `newSha` 设置为 `null`。

您可以对特定拉取请求、新标签或新分支的新提交做出反应。

### 配置更改

当顶级属性`event.scope`为`"repo.config"`时，指定`updatedConfig`属性。它是一个包含更新配置的对象。这是一个示例值：

```json
"updatedConfig": {
  "private": false
}
```

当 webhook 不支持更新的配置键时，该对象将为空：

```json
"updatedConfig": {}
```目前仅支持`private`。如果您希望从此处提供的更多配置密钥中受益，请通过 website@huggingface.co 告知我们。

### 讨论和拉取请求

顶级属性 `discussion` 在社区活动（讨论和 Pull 请求）上指定。 `discussion.isPullRequest` 属性是一个布尔值，指示讨论是否也是 Pull 请求（在 Hub 上，PR 是一种特殊类型的讨论）。这是一个示例值：

```json
"discussion": {
	"id": "639885d811ae2bad2b7ba461",
	"title": "Hello!",
	"url": {
		"web": "https://huggingface.co/some-user/some-repo/discussions/3",
		"api": "https://huggingface.co/api/models/some-user/some-repo/discussions/3"
	},
	"status": "open",
	"author": {
		"id": "61d2000c3c2083e1c08af22d"
	},
	"isPullRequest": true,
	"changes": {
		"base": "refs/heads/main"
	}
	"num": 3
}
```

### 评论

顶级属性 `comment` 在创建评论（包括创建讨论）或更新时指定。这是一个示例值：

```json
"comment": {
	"id": "6398872887bfcfb93a306f18",
	"author": {
		"id": "61d2000c3c2083e1c08af22d"
	},
	"content": "This adds an env key",
	"hidden": false,
	"url": {
		"web": "https://huggingface.co/some-user/some-repo/discussions/4#6398872887bfcfb93a306f18"
	}
}
```

## Webhook 秘密

设置 Webhook 机密有助于确保发送到 Webhook 处理程序 URL 的有效负载实际上来自 Hugging Face。

如果您为 Webhook 设置秘密，它将在每个请求上作为 `X-Webhook-Secret` HTTP 标头发送。仅支持 ASCII 字符。

> [!提示]
> 也可以直接在处理程序 URL 中添加机密。例如，将其设置为查询参数：https://example.com/webhook?secret=XXX。
>
> 如果访问请求的 HTTP 标头对于 Webhook 处理程序来说很复杂，这会很有帮助。

## 速率限制每个 Webhook 每 24 小时最多可触发 1,000 次。您可以在 Webhook 设置页面的“活动”选项卡中查看您的使用情况。

如果您需要增加 Webhook 的触发器数量，请升级到 PRO、Team 或 Enterprise，并通过 website@huggingface.co 联系我们。

## 开发您的 Webhook

如果您没有 HTTPS 端点/URL，您可以尝试使用公共工具进行 Webhook 测试。这些工具充当发送给它们的包罗万象（捕获所有请求）并给出 200 OK 状态代码。 [Beeceptor](https://beeceptor.com/) 是一种可用于创建临时 HTTP 端点并检查传入负载的工具。另一个这样的工具是[Webhook.site](https://webhook.site/)。

此外，您可以在开发过程中将真实的 Webhook 有效负载路由到计算机上本地运行的代码。这是测试和调试以实现更快集成的好方法。您可以通过将本地主机端口公开到互联网来完成此操作。为了能够走这条路，您可以使用[ngrok](https://ngrok.com/)或[localtunnel](https://theboroer.github.io/localtunnel-www/)。

## 调试 Webhooks

您可以轻松找到最近为您的 webhook 生成的事件。打开 Webhook 的活动选项卡。在那里您将看到最近事件的列表。![image.png](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhook-payload.png)
 
您可以在此处查看 HTTP 状态代码和生成的事件的负载。此外，您可以通过单击 `Replay` 按钮重播这些事件！ 

注意：更改 Webhook 的目标 URL 或机密时，重播事件会将有效负载发送到更新的 URL。

## 常见问题解答

##### 我可以在我的组织和我的用户帐户上定义 webhook 吗？

不，目前不支持此功能。

##### 我如何订阅 HF 上的所有事件（或跨整个存储库类型，如所有型号）？

目前尚未向最终用户公开，但如果您发送电子邮件至 website@huggingface.co，我们可以为您切换此功能。

### 使用🤗数据集
https://huggingface.co/docs/hub/datasets-usage.md