<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhook 指南：设置自动系统以在数据集更改时重新训练模型

本指南将帮助您完成 Hugging Face 平台上自动训练管道的设置
使用 HF 数据集、Webhooks、Spaces 和 AutoTrain。

我们将构建一个 Webhook，用于监听图像分类数据集的变化并触发微调
使用 [AutoTrain](https://huggingface.co/autotrain) 的 [microsoft/resnet-50](https://huggingface.co/microsoft/resnet-50)。

## 先决条件：将数据集上传到 Hub

我们将使用[simple image classification dataset](https://huggingface.co/datasets/huggingface-projects/auto-retrain-input-dataset)
的例子。了解有关将数据上传到中心[here](https://huggingface.co/docs/datasets/upload_dataset)的更多信息。

![dataset](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/002-auto-retrain/dataset.png)

## 创建一个 Webhook 来响应数据集的更改

首先，让我们从您的 [settings]( https://huggingface.co/settings/webhooks) 创建一个 Webhook。

- 选择您的数据集作为目标存储库。在此示例中，我们将定位[huggingface-projects/input-dataset](https://huggingface.co/datasets/huggingface-projects/input-dataset)。
- 您现在可以输入一个虚拟的 Webhook URL。定义 Webhook 将让您查看将发送给它的事件。您还可以重播它们，这对于调试很有用！
- 输入密码以使其更安全。
- 订阅“Repo update”事件，因为我们希望对数据更改做出反应

您的 Webhook 将如下所示：

![webhook-creation](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/002-auto-retrain/webhook-creation.png)

## 创建一个空间来响应您的 Webhook我们现在需要一种方法来响应您的 Webhook 事件。一个简单的方法是使用[Space](https://huggingface.co/docs/hub/spaces-overview)！

您可以找到一个示例空间[here](https://huggingface.co/spaces/huggingface-projects/auto-retrain/tree/main)。

该空间使用 Docker、Python、[FastAPI](https://fastapi.tiangolo.com/) 和 [uvicorn](https://www.uvicorn.org) 来运行一个简单的 HTTP 服务器。了解有关 Docker Spaces [here](https://huggingface.co/docs/hub/spaces-sdks-docker) 的更多信息。

入口点是[src/main.py](https://huggingface.co/spaces/huggingface-projects/auto-retrain/blob/main/src/main.py)。让我们浏览一下这个文件并详细说明它的作用：

1. 它会生成一个 FastAPI 应用程序，该应用程序将侦听 `/webhook` 上的 HTTP `POST` 请求：

```python
from fastapi import FastAPI

# [...]
@app.post("/webhook")
async def post_webhook(
	# ...
):

# ...
```

2. 2. 此路由检查 `X-Webhook-Secret` 标头是否存在，并且其值是否与您在 Webhook 设置中设置的值相同。 `WEBHOOK_SECRET` 密钥必须在空间设置中设置，并且与 Webhook 中设置的密钥相同。

```python
# [...]

WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET")

# [...]

@app.post("/webhook")
async def post_webhook(
	# [...]
	x_webhook_secret:  Optional[str] = Header(default=None),
	# ^ checks for the X-Webhook-Secret HTTP header
):
	if x_webhook_secret is None:
		raise HTTPException(401)
	if x_webhook_secret != WEBHOOK_SECRET:
		raise HTTPException(403)
	# [...]
```

3. 事件的有效负载被编码为 JSON。在这里，我们将使用 pydantic 模型来解析事件负载。我们还指定仅在以下情况下运行 Webhook：
- 事件涉及输入数据集
- 该事件是存储库内容的更新，即有一个新的提交

```python
# defined in src/models.py
class WebhookPayloadEvent(BaseModel):
	action: Literal["create", "update", "delete", "move"]
	scope: str

class WebhookPayloadRepo(BaseModel):
	type: Literal["dataset", "model", "space"]
	name: str
	id: str
	private: bool
	headSha: str

class WebhookPayload(BaseModel):
	event: WebhookPayloadEvent
	repo: WebhookPayloadRepo

# [...]

@app.post("/webhook")
async def post_webhook(
	# [...]
	payload: WebhookPayload,
	# ^ Pydantic model defining the payload format
):
	# [...]
	if not (
		payload.event.action == "update"
		and payload.event.scope.startswith("repo.content")
		and payload.repo.name == config.input_dataset
		and payload.repo.type == "dataset"
	):
		# no-op if the payload does not match our expectations
		return {"processed": False}
	#[...]
```

4. 如果有效负载有效，下一步是在 AutoTrain 上创建一个项目，安排对输入数据集的输入模型（在我们的示例中为`microsoft/resnet-50`）进行微调，并在完成后创建关于数据集的讨论！

```python
def schedule_retrain(payload: WebhookPayload):
	# Create the autotrain project
	try:
		project = AutoTrain.create_project(payload)
		AutoTrain.add_data(project_id=project["id"])
		AutoTrain.start_processing(project_id=project["id"])
	except requests.HTTPError as err:
		print("ERROR while requesting AutoTrain API:")
		print(f"  code: {err.response.status_code}")
		print(f"  {err.response.json()}")
		raise
	# Notify in the community tab
	notify_success(project["id"])
```访问评论内的链接查看训练成本估算，并开始微调模型！

![community tab notification](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/002-auto-retrain/notification.png)

在此示例中，我们使用 Hugging Face AutoTrain 快速微调我们的模型，但您当然可以插入您的训练基础设施！

请随意将空间复制到您的个人命名空间并使用它。您需要提供两个秘密：
- `WEBHOOK_SECRET`：Webhook 的秘密。
- `HF_ACCESS_TOKEN`：具有`write`权限的用户访问令牌。您可以创建一个 [from your settings](https://huggingface.co/settings/tokens)。

您还需要调整 [⟦T13⟧ file](https://huggingface.co/spaces/huggingface-projects/auto-retrain/blob/main/config.json) 以使用您选择的数据集和模型：

```json
{
	"target_namespace": "the namespace where the trained model should end up",
	"input_dataset": "the dataset on which the model will be trained",
	"input_model": "the base model to re-train",
	"autotrain_project_prefix": "A prefix for the AutoTrain project"
}
```

## 配置您的 Webhook 以将事件发送到您的空间

最后但并非最不重要的一点是，您需要配置 Webhook 以将 POST 请求发送到您的空间。

首先，我们从上下文菜单中获取 Space 的“直接 URL”。单击“嵌入此空间”并复制“直接 URL”。

![embed this Space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/002-auto-retrain/duplicate-space.png)

![direct URL](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/002-auto-retrain/direct-url.png)

更新您的 Webhook 以将请求发送到该 URL：

![webhook settings](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/002-auto-retrain/update-webhook.png)

就是这样！现在，对输入数据集的每次提交都将触发使用 AutoTrain 对 ResNet-50 进行微调 🎉

### 静态 HTML 空间
https://huggingface.co/docs/hub/spaces-sdks-static.md