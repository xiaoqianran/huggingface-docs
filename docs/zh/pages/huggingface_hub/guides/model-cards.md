<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 创建并分享模型卡

`huggingface_hub` 库提供了一个 Python 接口来创建、共享和更新模型卡。
访问[the dedicated documentation page](https://huggingface.co/docs/hub/models-cards)
更深入地了解 Hub 上的模型卡是什么以及它们在幕后如何工作。

## 从集线器加载模型卡

要从集线器加载现有卡，您可以使用[ModelCard.load()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.RepoCard.load)功能。在这里，我们将从[⟦T21⟧](https://huggingface.co/nateraw/vit-base-beans)加载卡片。

```python
from huggingface_hub import ModelCard

card = ModelCard.load('nateraw/vit-base-beans')
```

这张卡有一些您可能想要访问/利用的有用属性：
  - `card.data`：返回带有模型卡元数据的[ModelCardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.ModelCardData)实例。在此实例上调用`.to_dict()`以获取字典形式的表示。
  - `card.text`：返回卡片的文本，*不包括元数据标头*。
  - `card.content`：返回卡片的文本内容，*包括元数据标头*。

## 创建模型卡

### 来自文本

要从文本初始化模型卡，只需在初始化时将卡的文本内容传递给`ModelCard`即可。

```python
content = """
---
language: en
license: mit
---

# My Model Card
"""

card = ModelCard(content)
card.data.to_dict() == {'language': 'en', 'license': 'mit'}  # True
```

您可能想要执行此操作的另一种方法是使用 f 字符串。在下面的例子中，我们：

- 使用[ModelCardData.to_yaml()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.CardData.to_yaml)将我们定义的元数据转换为YAML，以便我们可以使用它在模型卡中插入YAML块。
- 展示如何通过 Python f 字符串使用模板变量。

```python
card_data = ModelCardData(language='en', license='mit', library='timm')

example_template_var = 'nateraw'
content = f"""
---
{ card_data.to_yaml() }
---

# My Model Card

This model created by [@{example_template_var}](https://github.com/{example_template_var})
"""

card = ModelCard(content)
print(card)
```上面的例子会给我们留下一张看起来像这样的卡片：

```
---
language: en
license: mit
library: timm
---

# My Model Card

This model created by [@nateraw](https://github.com/nateraw)
```

### 来自 Jinja 模板

如果您安装了`Jinja2`，您可以从 jinja 模板文件创建模型卡。让我们看一个基本的例子：

```python
from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData

# Define your jinja template
template_text = """
---
{{ card_data }}
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@{{ author }}](https://hf.co/{{author}}).
""".strip()

# Write the template to a file
Path('custom_template.md').write_text(template_text)

# Define card metadata
card_data = ModelCardData(language='en', license='mit', library_name='keras')

# Create card from template, passing it any jinja template variables you want.
# In our case, we'll pass author
card = ModelCard.from_template(card_data, template_path='custom_template.md', author='nateraw')
card.save('my_model_card_1.md')
print(card)
```

生成的卡片的降价如下所示：

```
---
language: en
license: mit
library_name: keras
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

如果您更新任何卡数据，它将反映在卡本身中。

```
card.data.library_name = 'timm'
card.data.language = 'fr'
card.data.license = 'apache-2.0'
print(card)
```

现在，如您所见，元数据标头已更新：

```
---
language: fr
license: apache-2.0
library_name: timm
---

# Model Card for MyCoolModel

This model does this and that.

This model was created by [@nateraw](https://hf.co/nateraw).
```

当您更新卡数据时，您可以通过调用[ModelCard.validate()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.RepoCard.validate)来验证该卡对集线器是否仍然有效。这可确保该卡通过 Hugging Face Hub 上设置的任何验证规则。

### 来自默认模板

您还可以使用 [default template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)，而不是使用自己的模板，这是一张功能齐全的模型卡，其中包含大量您可能需要填写的部分。在底层，它使用 [Jinja2](https://jinja.palletsprojects.com/en/3.1.x/) 来填写模板文件。

> [!提示]
> 请注意，您必须安装 Jinja2 才能使用 `from_template`。您可以使用 `pip install Jinja2` 来实现。

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
card.save('my_model_card_2.md')
print(card)
```

## 分享模型卡

如果您通过 Hugging Face Hub 进行身份验证（使用 `hf auth login` 或 [login()](/docs/huggingface_hub/v1.29.0/en/package_reference/authentication#huggingface_hub.login)），则只需调用 [ModelCard.push_to_hub()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.RepoCard.push_to_hub) 即可将卡片推送到 Hub。让我们看看如何做到这一点......首先，我们将在经过身份验证的用户的命名空间下创建一个名为“hf-hub-modelcards-pr-test”的新存储库：

```python
from huggingface_hub import whoami, create_repo

user = whoami()['name']
repo_id = f'{user}/hf-hub-modelcards-pr-test'
url = create_repo(repo_id, exist_ok=True)
```

然后，我们将从默认模板创建一张卡片（与上一节中定义的模板相同）：

```python
card_data = ModelCardData(language='en', license='mit', library_name='keras')
card = ModelCard.from_template(
    card_data,
    model_id='my-cool-model',
    model_description="this model does this and that",
    developers="Nate Raw",
    repo="https://github.com/huggingface/huggingface_hub",
)
```

最后，我们将其推送到中心

```python
card.push_to_hub(repo_id)
```

您可以查看生成的卡片[here](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/blob/main/README.md)。

如果您想将卡片作为拉取请求推送，则可以在调用 `push_to_hub` 时只说 `create_pr=True`：

```python
card.push_to_hub(repo_id, create_pr=True)
```

可以看到从此命令创建的结果 PR [here](https://huggingface.co/nateraw/hf-hub-modelcards-pr-test/discussions/3)。

## 更新元数据

在本节中，我们将了解回购卡中有哪些元数据以及如何更新它们。

`metadata` 指的是哈希映射（或键值）上下文，它提供有关模型、数据集或空间的一些高级信息。该信息可以包括模型的 `pipeline type`、`model_id` 或 `model_description` 等详细信息。有关更多详细信息，您可以查看这些指南：[Model Card](https://huggingface.co/docs/hub/model-cards#model-card-metadata)、[Dataset Card](https://huggingface.co/docs/hub/datasets-cards#dataset-card-metadata) 和 [Spaces Settings](https://huggingface.co/docs/hub/spaces-settings#spaces-settings)。
现在让我们看一些有关如何更新这些元数据的示例。

让我们从第一个例子开始：

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "image-classification"})
```

使用这两行代码，您将更新元数据以设置新的`pipeline_tag`。

默认情况下，您无法更新卡上已存在的密钥。如果你想这样做，你必须通过
`overwrite=True` 明确：

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("username/my-cool-model", {"pipeline_tag": "text-generation"}, overwrite=True)
```您经常想建议对存储库进行一些更改
您没有写入权限。您可以通过在该存储库上创建 PR 来做到这一点，这将允许所有者
检查并合并您的建议。

```python
>>> from huggingface_hub import metadata_update
>>> metadata_update("someone/model", {"pipeline_tag": "text-classification"}, create_pr=True)
```

## 包括评估结果

要将评估结果包含在元数据 `model-index` 中，您可以传递 [EvalResult](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.EvalResult) 或 `EvalResult` 列表以及关联的评估结果。当您调用 `card.data.to_dict()` 时，它会在底层创建 `model-index`。有关其工作原理的更多信息，您可以查看[this section of the Hub docs](https://huggingface.co/docs/hub/models-cards#evaluation-results)。

> [!提示]
> 请注意，使用此功能需要您在 [ModelCardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.ModelCardData) 中包含 `model_name` 属性。

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = EvalResult(
        task_type='image-classification',
        dataset_type='beans',
        dataset_name='Beans',
        metric_type='accuracy',
        metric_value=0.7
    )
)

card = ModelCard.from_template(card_data)
print(card.data)
```

生成的 `card.data` 应如下所示：

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
```

如果您想要分享多个评估结果，只需传递 `EvalResult` 列表即可：

```python
card_data = ModelCardData(
    language='en',
    license='mit',
    model_name='my-cool-model',
    eval_results = [
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='accuracy',
            metric_value=0.7
        ),
        EvalResult(
            task_type='image-classification',
            dataset_type='beans',
            dataset_name='Beans',
            metric_type='f1',
            metric_value=0.65
        )
    ]
)
card = ModelCard.from_template(card_data)
card.data
```

这应该会给你留下以下`card.data`：

```
language: en
license: mit
model-index:
- name: my-cool-model
  results:
  - task:
      type: image-classification
    dataset:
      name: Beans
      type: beans
    metrics:
    - type: accuracy
      value: 0.7
    - type: f1
      value: 0.65
```

### 推理端点
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/inference_endpoints.md