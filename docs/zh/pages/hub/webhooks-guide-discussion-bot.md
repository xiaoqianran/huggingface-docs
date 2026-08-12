<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhook 指南：使用 LLM 回复构建讨论机器人

以下是有关如何使用 Hugging Face Webhooks 构建机器人的简短指南，该机器人可以通过您选择的法学硕士通过 [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/en/index) 生成的回复来回复 Hub 上的讨论评论。

## 在您的用户个人资料中创建您的 Webhook

首先，让我们从您的 [settings]( https://huggingface.co/settings/webhooks) 创建一个 Webhook。

- 输入您的 Webhook 将侦听的一些目标存储库。
- 您现在可以放置一个虚拟的 Webhook URL，但是定义您的 Webhook 将让您查看将发送到它的事件（并且您可以重放它们，这对于调试非常有用）。
- 输入一个秘密，这样会更安全。
- 订阅社区（公关和讨论）活动，因为我们正在构建讨论机器人。

您的 Webhook 将如下所示：

![webhook-creation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/001-discussion-bot/webhook-creation.png)

## 创建一个新的`Bot`用户配置文件

在本指南中，我们创建一个单独的用户帐户来托管空间并发表评论：

![discussion-bot-profile](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/001-discussion-bot/discussion-bot-profile.png)

> [!提示]
> 创建将与 Hub 上的其他用户交互的机器人时，我们要求您将该帐户明确标记为“机器人”（请参阅个人资料屏幕截图）。

## 创建一个将对您的 Webhook 做出反应的空间

第三步实际上是监听Webhook事件。一个简单的方法是使用 Space。我们使用我们创建的用户帐户，但如果您愿意，也可以从主用户帐户执行此操作。

空间的代码是[here](https://huggingface.co/spaces/discussion-bot/webhook/tree/main)。

我们使用 NodeJS 和 Typescript 来实现它，但任何语言或框架都可以同样良好地工作。了解有关 Docker Spaces [here](https://huggingface.co/docs/hub/spaces-sdks-docker) 的更多信息。

**主要的`server.ts`文件是[here](https://huggingface.co/spaces/discussion-bot/webhook/blob/main/server.ts)**

让我们看看这个文件中发生了什么：

```ts
app.post("/", async (req, res) => {
	if (req.header("X-Webhook-Secret") !== process.env.WEBHOOK_SECRET) {
		console.error("incorrect secret");
		return res.status(400).json({ error: "incorrect secret" });
	}
	...
```

在这里，我们监听对`/`发出的POST请求，然后检查`X-Webhook-Secret`标头是否等于我们之前定义的秘密（您还需要在空间设置中设置`WEBHOOK_SECRET`秘密才能验证它）。

```ts
	const event = req.body.event;
	if (
		event.action === "create" &&
		event.scope === "discussion.comment" &&
		req.body.comment.content.includes(BOT_USERNAME)
	) {
		...
```

事件的有效负载被编码为 JSON。在这里，我们指定仅在以下情况下运行 Webhook：
- 事件涉及讨论评论
- 该事件是一个创作，即已发布新评论
- 评论内容包含`@discussion-bot`，即我们的机器人刚刚在评论中提到。

在这种情况下，我们将继续下一步：

```ts
	const CHAT_API_URL =
		"https://router.huggingface.co/v1/chat/completions";
	const SYSTEM_PROMPT = `You are a helpful bot that replies to discussions about machine learning. Keep your responses concise and friendly.`;

	const response = await fetch(CHAT_API_URL, {
		method: "POST",
		headers: {
			"Authorization": `Bearer ${process.env.HF_TOKEN}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify({
			model: "moonshotai/Kimi-K2.6",
			messages: [
				{ role: "system", content: SYSTEM_PROMPT },
				{ role: "user", content: req.body.comment.content },
			],
			max_tokens: 500,
		}),
	});
	if (response.ok) {
		const output = await response.json();
		const continuationText = output.choices[0].message.content;
		...
```

这是最酷的部分：使用[Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/chat-completions)，我们调用一个带有系统提示和用户评论的开放权重模型（`moonshotai/Kimi-K2.6`）。该模型生成一个回复，我们从 `choices[0].message.content` 中提取该回复。您可以从 [list of available models](https://huggingface.co/inference/models) 中选择任何型号 — 只需交换 `model` 字段即可。

最后，我们将在同一讨论线程中将其作为回复发布：

```ts
	const commentUrl = req.body.discussion.url.api + "/comment";

	const commentApiResponse = await fetch(commentUrl, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${process.env.HF_TOKEN}`,
			"Content-Type": "application/json",
		},
		body: JSON.stringify({ comment: continuationText }),
	});

	const apiOutput = await commentApiResponse.json();
```

## 配置您的 Webhook 以将事件发送到您的空间

最后但并非最不重要的一点是，您需要配置 Webhook 以将 POST 请求发送到您的空间。

首先，我们从上下文菜单中获取 Space 的“直接 URL”。单击“嵌入此空间”并复制“直接 URL”。

![embed this Space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/001-discussion-bot/embed-space.png)
![direct URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/001-discussion-bot/direct-url.png)

更新您的 webhook 以将请求发送到该 URL：

![webhook settings](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/001-discussion-bot/webhook-creation.png)

## 结果

![discussion-result](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/001-discussion-bot/discussion-result.png)

### 学术中心
https://huggingface.co/docs/hub/academia-hub.md