<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Webhook 指南：为模型和数据集设置自动元数据质量审查 

本指南将引导您创建一个系统，该系统可以对 Hub 上用户或组织的模型或数据集的更改做出反应，并为更改的存储库创建“元数据审核”。 

## 我们正在构建什么以及为什么？

在深入探讨此特定工作流程中涉及的技术细节之前，我们将快速概述我们正在创建的内容以及原因。 

[Model cards](https://huggingface.co/docs/hub/model-cards) 和 [dataset cards](https://huggingface.co/docs/hub/datasets-cards) 是记录机器学习模型和数据集的重要工具。 Hugging Face Hub 使用包含 [YAML](https://en.wikipedia.org/wiki/YAML) 标头块的 `README.md` 文件来生成模型和数据集卡。此`YAML`部分定义与模型或数据集相关的元数据。例如： 

```yaml
---
language: 
  - "List of ISO 639-1 code for your language"
  - lang1
  - lang2
tags:
- tag1
- tag2
license: "any valid license identifier"
datasets:
- dataset1
---
```

此元数据包含有关潜在用户的模型或数据集的基本信息。例如，许可证定义了可以使用模型或数据集的条款。 Hub 用户还可以使用 `YAML` 元数据中定义的字段作为过滤器来识别符合特定条件的模型或数据集。由于此块中定义的元数据对于我们的模型和数据集的潜在用户至关重要，因此完成此部分非常重要。在团队或组织设置中，将模型和数据集推送到 Hub 的用户可能对此 YAML 元数据块的重要性有不同的熟悉程度。虽然团队中的某人可以承担审查此元数据的责任，但我们可以采取一些自动化措施来帮助我们解决此问题。结果将是当中心上的存储库发生更改时自动发布或更新的元数据审查报告。对于我们的元数据质量，该系统的工作原理与[CI/CD](https://en.wikipedia.org/wiki/CI/CD)类似。

![Metadata review](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/003-metadata-review/metadata-report-screenshot.png)

您还可以找到示例评论[here](https://huggingface.co/datasets/davanstrien/test_webhook/discussions/1#63d932fe19aa7b8ed2718b3f)。 

## 使用 Hub 客户端库创建模型审核卡 

`huggingface_hub` 是一个 Python 库，可让您与 Hub 交互。我们可以使用 `DatasetCard.load` 或 `ModelCard.load` 方法从 Hub 使用此库到 [download model and dataset cards](https://huggingface.co/docs/huggingface_hub/how-to-model-cards)。特别是，我们将使用这些方法加载 Python 字典，其中包含模型或数据集卡的 `YAML` 中定义的元数据。我们将创建一个小的 Python 函数来包装这些方法并进行一些异常处理。 

```python
from huggingface_hub import DatasetCard, ModelCard
from huggingface_hub.utils import EntryNotFoundError 

def load_repo_card_metadata(repo_type, repo_name):
    if repo_type == "dataset":
        try:
            return DatasetCard.load(repo_name).data.to_dict()
        except EntryNotFoundError:
            return {}
    if repo_type == "model":
        try:
            return ModelCard.load(repo_name).data.to_dict()
        except EntryNotFoundError:
            return {}
```此函数将返回一个包含与存储库关联的元数据的 Python 字典（如果没有元数据，则返回一个空字典）。

```python
{'license': 'afl-3.0'}
```

## 创建我们的元数据审查报告

一旦我们有了包含与存储库关联的元数据的 Python 字典，我们将为元数据审查创建一个“报告卡”。在这个特定的实例中，我们将通过定义一些我们想要值的元数据字段来检查我们的元数据。例如，我们可能希望确保 `license` 字段始终已完成。为了对元数据进行评级，我们将计算所需字段中存在哪些元数据字段，并根据我们希望查看值的所需元数据字段的覆盖率返回百分比分数。

由于我们有一个包含元数据的 Python 字典，因此我们可以循环遍历该字典来检查是否存在我们想要的键。如果缺少所需的元数据字段（字典中的键），我们将将该值指定为 `None`。

```python
def create_metadata_key_dict(card_data, repo_type: str):
    shared_keys = ["tags", "license"]
    if repo_type == "model":
        model_keys = ["library_name", "datasets", "metrics", "co2", "pipeline_tag"]
        shared_keys.extend(model_keys)
        keys = shared_keys
        return {key: card_data.get(key) for key in keys}
    if repo_type == "dataset":
        # [...]
```该函数将返回一个字典，其中包含表示模型或数据集所需的元数据字段的键。字典值将包含为该字段输入的元数据，或者如果 `YAML` 中缺少该元数据字段，则包含 `None`。 

```python
{'tags': None,
 'license': 'afl-3.0',
 'library_name': None,
 'datasets': None,
 'metrics': None,
 'co2': None,
 'pipeline_tag': None}
```

一旦我们有了这个字典，我们就可以创建我们的元数据报告。为了简洁起见，我们不会在此处包含完整的代码，但此 Webhook 的 Hugging Face Spaces [repository](https://huggingface.co/spaces/librarian-bot/webhook_metadata_reviewer/blob/main/main.py) 包含完整的代码。

我们创建一个函数，该函数创建一个 Markdown 表，该表生成元数据覆盖字典中数据的更漂亮版本。 

```python
def create_metadata_breakdown_table(desired_metadata_dictionary):
    # [...]
    return tabulate(
        table_data, tablefmt="github", headers=("Metadata Field", "Provided Value")
    )
```

我们还有一个生成分数的 Python 函数（表示存在的所需元数据字段的百分比）

```python
def calculate_grade(desired_metadata_dictionary):
    # [...]
    return round(score, 2)
```

以及一个为我们的元数据审查创建 Markdown 报告的 Python 函数。该报告包含分数和元数据表，以及报告内容的一些解释。

```python
def create_markdown_report(
    desired_metadata_dictionary, repo_name, repo_type, score, update: bool = False
):
    # [...]
    return report
```

## 如何自动发布评论？我们现在有了 Markdown 格式的元数据审查报告。我们将使用 `huggingface_hub` 库来发布此评论。我们定义一个函数，用于取回从 Hub 接收到的 Webhook 数据、解析数据并创建元数据报告。根据之前是否已创建报告，该函数会创建新报告或将新问题发布到现有元数据审核线程。

```python
def create_or_update_report(data):
    if parsed_post := parse_webhook_post(data):
        repo_type, repo_name = parsed_post
    else:
        return Response("Unable to parse webhook data", status_code=400)
    # [...]
    return True
```

> [!提示]
> `:=` 是 Python 3.8 版本中添加的赋值表达式运算符的 Python 语法（俗称海象运算符）。人们对这种语法有不同的看法，如果你不使用它，它不会改变 Python 计算代码的方式。您可以在这篇 Real Python 文章中阅读有关此运算符的更多信息。

## 创建 Webhook 来响应 Hub 上的更改

我们现在已经获得了为模型或数据集创建元数据审核报告的核心功能。下一步是使用 Webhooks 自动响应更改。 

## 在您的用户配置文件中创建 Webhook

首先，访问 https://huggingface.co/settings/webhooks 创建 Webhook。- 输入您的 Webhook 将侦听的一些目标存储库（您可能希望将其限制为您自己的存储库或您所属组织的存储库）。
- 输入一个秘密以使您的 Webhook 更安全（如果您不知道为此选择什么，您可能需要使用 [password generator](https://1password.com/password-generator/) 为您的秘密生成足够长的随机字符串）。
- 我们现在可以为 `Webhook URL` 参数传递一个虚拟 URL。 

您的 Webhook 将如下所示：

![webhook settings](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/003-metadata-review/webhook-settings.png)

## 创建新的 Bot 用户配置文件

本指南创建一个单独的用户帐户来发布元数据评论。 

![Bot user account](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/003-metadata-review/librarian-bot-profile.png)

> [!提示]
> 创建将与 Hub 上的其他用户交互的机器人时，我们要求您将该帐户明确标记为“机器人”（请参阅个人资料屏幕截图）。

## 创建 Webhook 监听器

我们现在需要某种方式来监听 Webhook 事件。您可以使用许多可能的工具来侦听 Webhook 事件。许多现有服务，例如 [Zapier](https://zapier.com/) 和 [IFTTT](https://ifttt.com)，可以使用 Webhooks 来触发操作（例如，它们可以在每次更新模型时发布一条推文）。在本例中，我们将使用 [FastAPI](https://fastapi.tiangolo.com/) 实现 Webhook 监听器。[FastAPI](https://fastapi.tiangolo.com/) 是一个 Python Web 框架。我们将使用 FastAPI 创建 Webhook 侦听器。特别是，我们需要实现一个在 `/webhook` 上接受 `POST` 请求的路由。为了进行身份验证，我们将比较 `X-Webhook-Secret` 标头与可以传递给我们的 [Docker container at runtime](./spaces-sdks-docker#runtime) 的 `WEBHOOK_SECRET` 秘密。

```python
from fastapi import FastAPI, Request, Response
import os

KEY = os.environ.get("WEBHOOK_SECRET")

app = FastAPI()

@app.post("/webhook")
async def webhook(request: Request):
    if request.method == "POST":
        if request.headers.get("X-Webhook-Secret") != KEY:
            return Response("Invalid secret", status_code=401)
        data = await request.json()
        result = create_or_update_report(data)
        return "Webhook received!" if result else result
```

上述函数将接收 Webhook 事件并为更改的存储库创建或更新元数据审核报告。

## 使用 Spaces 部署我们的 Webhook 应用程序 

我们的 [main.py](https://huggingface.co/spaces/librarian-bot/webhook_metadata_reviewer/blob/main/main.py) 文件包含 Webhook 应用程序所需的所有代码。为了部署它，我们将使用[Space](./spaces-overview)。 

对于我们的 Space，我们将使用 Docker 来运行我们的应用程序。 [Dockerfile](https://huggingface.co/spaces/librarian-bot/webhook_metadata_reviewer/blob/main/Dockerfile) 复制我们的应用程序文件，安装所需的依赖项，然后运行应用程序。为了填充 `KEY` 变量，我们还将使用我们之前生成的秘密为我们的空间设置一个 `WEBHOOK_SECRET` 秘密。您可以阅读有关 Docker Spaces [here](./spaces-sdks-docker) 的更多信息。

最后，我们需要将 Webhook 设置中的 URL 更新为我们空间的 URL。我们可以从上下文菜单中获取空间的“直接 URL”。单击“嵌入此空间”并复制“直接 URL”。

![direct url](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/webhooks-guides/003-metadata-review/direct-url.png)获得此 URL 后，我们可以将其传递给 Webhook 设置中的 `Webhook URL` 参数。当受监控的存储库发生变化时，我们的机器人现在应该开始发布评论！ 

## 结论和后续步骤

我们现在有一个自动元数据审查机器人！以下是一些关于如何在本指南的基础上进行构建的想法：

- 我们的机器人所做的元数据审查相对粗糙；您可以添加更复杂的规则来查看元数据。
- 您可以使用完整的 `README.md` 文件进行审核。
- 您可能想要定义对您的组织特别重要的“规则”，并使用 Webhook 来检查这些规则是否得到遵守。

如果您使用 Webhooks 构建元数据质量应用程序，请标记我@davanstrien；我很想知道！

### GGUF 与 llama.cpp 的用法
https://huggingface.co/docs/hub/gguf-llamacpp.md