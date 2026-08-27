<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储卡

Huggingface_hub 库提供了一个 Python 接口来创建、共享和更新模型/数据集卡。
访问[dedicated documentation page](https://huggingface.co/docs/hub/models-cards)以更深入地了解什么
集线器上的模型卡以及它们在幕后的工作原理。您还可以查看我们的[Model Cards guide](../how-to-model-cards)
了解如何在自己的项目中使用这些实用程序。

## 回购卡[[huggingface_hub.RepoCard]]

`RepoCard`对象是[ModelCard](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.ModelCard)、[DatasetCard](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.DatasetCard)和`SpaceCard`的父类。

#### Huggingface_hub.RepoCard[[huggingface_hub.RepoCard]]

```python
huggingface_hub.RepoCard(content: str, ignore_metadata_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L37)

#### __init__[[huggingface_hub.RepoCard.__init__]]

```python
__init__(content: str, ignore_metadata_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L42)

**参数：**

content (`str`) : Markdown 文件的内容。

从字符串内容初始化 RepoCard。内容应该是
Markdown 文件，开头有一个 YAML 块，还有一个 Markdown 正文。

示例：
```python
>>> from huggingface_hub.repocard import RepoCard
>>> text = '''
... ---
... language: en
... license: mit
... ---
...
... # My repo
... '''
>>> card = RepoCard(text)
>>> card.data.to_dict()
{'language': 'en', 'license': 'mit'}
>>> card.text
'\n# My repo\n'

```

> [!提示]
> 引发以下错误：
>
> - [⟦T40⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 当回购卡元数据的内容不是字典时。

#### from_template[[huggingface_hub.RepoCard.from_template]]

```python
from_template(card_data: CardData, template_path: str | None = None, template_str: str | None = None, **template_kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L289)

**参数：**

card_data (`huggingface_hub.CardData`) ：huggingface_hub.CardData 实例，其中包含要包含在 Hugging Face Hub 上的存储卡的 YAML 标头中的元数据。template_path (`str`, *可选*) ：带有可选 Jinja 模板变量的 Markdown 文件的路径，可以用 `template_kwargs` 填充。默认为默认模板。

template_str (`str`, *可选*) ：带有可选变量的原始 Jinja 模板字符串。当 `template_path` 和默认模板都不合适时使用。如果还提供了`template_path`，则忽略。

**返回：** [huggingface_hub.repocard.RepoCard](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.RepoCard)

具有指定卡片数据和内容的 RepoCard 实例
模板。

从模板初始化 RepoCard。默认情况下，它使用默认模板。

模板是 Jinja2 模板，可以通过传递关键字参数进行自定义。

#### 加载[[huggingface_hub.RepoCard.load]]

```python
load(repo_id_or_path: str | pathlib.Path, repo_type: str | None = None, token: str | None = None, ignore_metadata_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L135)

**参数：**

repo_id_or_path (`Union[str, Path]`) ：与 Hugging Face Hub 存储库或本地文件路径关联的存储库 ID。

repo_type (`str`, *可选*) ：要推送到的 Hugging Face 存储库的类型。默认为 None，将使用“model”。其他选项是“数据集”和“空间”。从本地文件路径加载时不使用。如果从子类调用它，则默认值将是子类的`repo_type`。token (`str`, *可选*) : 身份验证令牌，通过 `huggingface_hub.HfApi.login` 方法获取。将默认为存储的令牌。

ignore_metadata_errors (`str`) ：如果为 True，则解析元数据部分时的错误将被忽略。在此过程中可能会丢失一些信息。使用它的风险由您自行承担。

**返回：** [huggingface_hub.repocard.RepoCard](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.RepoCard)

从存储库初始化的 RepoCard（或子类）
README.md 文件或文件路径。

从 Hugging Face Hub 存储库的 README.md 或本地文件路径初始化 RepoCard。

示例：
```python
>>> from huggingface_hub.repocard import RepoCard
>>> card = RepoCard.load("nateraw/food")
>>> assert card.data.tags == ["generated_from_trainer", "image-classification", "pytorch"]

```

####push_to_hub[[huggingface_hub.RepoCard.push_to_hub]]

```python
push_to_hub(repo_id: str, token: str | None = None, repo_type: str | None = None, commit_message: str | None = None, commit_description: str | None = None, revision: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L226)

**参数：**

repo_id (`str`) ：要推送到的 Hugging Face Hub 存储库的存储库 ID。示例：“nateraw/食物”。

token (`str`, *可选*) : 身份验证令牌，通过 `huggingface_hub.HfApi.login` 方法获取。将默认为存储的令牌。

repo_type (`str`，*可选*，默认为“model”)：要推送到的 Hugging Face 存储库的类型。选项有“模型”、“数据集”和“空间”。如果这个函数被子类调用，它将默认为子类的`repo_type`。

commit_message (`str`, *可选*) ：生成的提交的摘要/标题/第一行。commit_description (`str`, *可选*) ：生成的提交的描述。

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

create_pr (`bool`, *可选*) ：是否使用此提交创建拉取请求。默认为`False`。

Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交过多，则特别有用。

**返回：** `str`

更新卡元数据的提交的 URL。

将 RepoCard 推送到 Hugging Face Hub 存储库。

#### 保存[[huggingface_hub.RepoCard.save]]

```python
save(filepath: pathlib.Path | str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L115)

**参数：**

filepath (`Union[Path, str]`) : 要保存的 markdown 文件的文件路径。

将 RepoCard 保存到文件中。

示例：
```python
>>> from huggingface_hub.repocard import RepoCard
>>> card = RepoCard("---\nlanguage: en\n---\n# This is a test repo card")
>>> card.save("/tmp/test.md")

```

#### 验证[[huggingface_hub.RepoCard.validate]]

```python
validate(repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L189)

**参数：**repo_type (`str`，*可选*，默认为“model”)：要推送到的 Hugging Face 存储库的类型。选项有“模型”、“数据集”和“空间”。如果从子类调用此函数，则默认值将是子类的`repo_type`。

根据 Hugging Face Hub 的卡片验证逻辑验证卡片。
使用该功能需要联网，所以只调用
内部由[huggingface_hub.repocard.RepoCard.push_to_hub()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.RepoCard.push_to_hub)。

> [!提示]
> 引发以下错误：
>
> - [⟦T77⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果卡未通过验证检查。
> - [⟦T78⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果对 Hub API 的请求因任何其他原因失败。

## 卡片数据[[huggingface_hub.CardData]]

[CardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.CardData)对象是[ModelCardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.ModelCardData)和[DatasetCardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.DatasetCardData)的父类。

#### Huggingface_hub.CardData[[huggingface_hub.CardData]]

```python
huggingface_hub.CardData(ignore_metadata_errors: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L165)

包含来自 RepoCard 的元数据的结构。

[CardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.CardData)是[ModelCardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.ModelCardData)和[DatasetCardData](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.DatasetCardData)的父类。

元数据可以导出为字典或 YAML。可以自定义导出以更改数据的表示形式
（示例：扁平化评估结果）。 `CardData` 表现为字典（可以获取、弹出、设置值），但不
继承`dict`以允许此导出步骤。

#### get[[huggingface_hub.CardData.get]]

```python
get(key: str, default: typing.Any = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L228)获取给定元数据键的值。

#### 弹出[[huggingface_hub.CardData.pop]]

```python
pop(key: str, default: typing.Any = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L233)

给定元数据键的 Pop 值。

#### to_dict[[huggingface_hub.CardData.to_dict]]

```python
to_dict()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L178)

**返回：** `dict`

CardData 表示为准备转储到 YAML 的字典
包含在 README.md 文件中的块。

将 CardData 转换为字典。

#### to_yaml[[huggingface_hub.CardData.to_yaml]]

```python
to_yaml(line_break = None, original_order: list[str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L198)

**参数：**

line_break (str, *可选*) ：转储到 yaml 时使用的换行符。

Original_order (`list[str]`, *可选*) ：如果提供，则在转储之前重新排序元数据字段以匹配此列表。任何不在 `original_order` 中的键都会附加在列出的键之后，保留它们现有的相对顺序。对于在不打乱其键的情况下往返 YAML 块很有用。

**返回：** `str`

CardData 表示为 YAML 块。

将 CardData 转储到 YAML 块以包含在 README.md 文件中。

## 模型卡

### ModelCard[[huggingface_hub.ModelCard]]

#### Huggingface_hub.ModelCard[[huggingface_hub.ModelCard]]

```python
huggingface_hub.ModelCard(content: str, ignore_metadata_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L336)

#### from_template[[huggingface_hub.ModelCard.from_template]]

```python
from_template(card_data: ModelCardData, template_path: str | None = None, template_str: str | None = None, **template_kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L341)

**参数：**card_data (`huggingface_hub.ModelCardData`) ：huggingface_hub.ModelCardData 实例，其中包含要包含在 Hugging Face Hub 上模型卡的 YAML 标头中的元数据。

template_path (`str`, *可选*) ：带有可选 Jinja 模板变量的 Markdown 文件的路径，可以用 `template_kwargs` 填充。默认为默认模板。

template_str (`str`, *可选*) ：带有可选变量的原始 Jinja 模板字符串。当`template_path`和默认模板都不合适时使用。如果还提供了`template_path`，则忽略。

**返回：** [huggingface_hub.ModelCard](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.ModelCard)

具有指定卡片数据和内容的 ModelCard 实例
模板。

从模板初始化 ModelCard。默认情况下，它使用默认模板，可以在此处找到该模板：
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md

模板是 Jinja2 模板，可以通过传递关键字参数进行自定义。

示例：
```python
>>> from huggingface_hub import ModelCard, ModelCardData, EvalResult

>>> # Using the Default Template
>>> card_data = ModelCardData(
...     language='en',
...     license='mit',
...     library_name='timm',
...     tags=['image-classification', 'resnet'],
...     datasets=['beans'],
...     metrics=['accuracy'],
... )
>>> card = ModelCard.from_template(
...     card_data,
...     model_description='This model does x + y...'
... )

>>> # Including Evaluation Results
>>> card_data = ModelCardData(
...     language='en',
...     tags=['image-classification', 'resnet'],
...     eval_results=[
...         EvalResult(
...             task_type='image-classification',
...             dataset_type='beans',
...             dataset_name='Beans',
...             metric_type='accuracy',
...             metric_value=0.9,
...         ),
...     ],
...     model_name='my-cool-model',
... )
>>> card = ModelCard.from_template(card_data)

>>> # Using a Custom Template
>>> card_data = ModelCardData(
...     language='en',
...     tags=['image-classification', 'resnet']
... )
>>> card = ModelCard.from_template(
...     card_data=card_data,
...     template_path='./src/huggingface_hub/templates/modelcard_template.md',
...     custom_template_var='custom value',  # will be replaced in template if it exists
... )

```

### ModelCardData[[huggingface_hub.ModelCardData]]

#### Huggingface_hub.ModelCardData[[huggingface_hub.ModelCardData]]

```python
huggingface_hub.ModelCardData(base_model: str | list[str] | None = None, datasets: str | list[str] | None = None, eval_results: list[huggingface_hub.repocard_data.EvalResult] | None = None, language: str | list[str] | None = None, library_name: str | None = None, license: str | None = None, license_name: str | None = None, license_link: str | None = None, metrics: list[str] | None = None, model_name: str | None = None, pipeline_tag: str | None = None, tags: list[str] | None = None, ignore_metadata_errors: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L271)

**参数：**base_model（`str`或`list[str]`，*可选*）：模型派生的基本模型的标识符。例如，如果您的模型是现有模型的微调或适配器，则这适用。该值必须是中心上模型的 ID（如果您的模型派生自多个模型，则为 ID 列表）。默认为无。

数据集（`Union[str, list[str]]`，*可选*）：用于训练该模型的数据集或数据集列表。应该是在 https://hf.co/datasets 上找到的数据集 ID。默认为无。

eval_results (`Union[list[EvalResult], EvalResult]`, *可选*) ：定义模型评估结果的`huggingface_hub.EvalResult`列表。如果提供，`model_name` 将用作 PapersWithCode 排行榜上的名称。默认为`None`。

语言（`Union[str, list[str]]`，*可选*）：模型训练数据或元数据的语言。它必须是 ISO 639-1、639-2 或 639-3 代码（两个/三个字母），或特殊值，如“代码”、“多语言”。默认为 `None`。

library_name (`str`, *可选*) ：该模型使用的库的名称。示例：keras 或 https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts 中的任何库。默认为无。许可证（`str`，*可选*）：该型号的许可证。示例：apache-2.0 或 https://huggingface.co/docs/hub/repositories-licenses 中的任何许可证。默认为无。

license_name (`str`, *可选*) : 该模型的许可证名称。默认为无。与`license_link`配合使用。通用许可证（Apache-2.0、MIT、CC-BY-SA-4.0）不需要名称。在这种情况下，请改用`license`。

license_link (`str`, *可选*) : 链接到该模型的许可证。默认为无。与`license_name`配合使用。通用许可证（Apache-2.0、MIT、CC-BY-SA-4.0）不需要链接。在这种情况下，请改用`license`。

指标（`list[str]`，*可选*）：用于评估此模型的指标列表。应该是可以在 https://hf.co/metrics 找到的指标名称。例如：“准确性”。默认为无。

model_name (`str`, *可选*) ：该模型的名称。它与 `eval_results` 一起使用，在卡的元数据中构建 `model-index`。您在此处提供的名称将在 PapersWithCode 的排行榜上使用。如果没有提供，则使用存储库名称作为默认值。默认为无。pipeline_tag (`str`, *可选*) ：与模型关联的管道标签。示例：“文本分类”。

标签（`list[str]`，*可选*）：添加到模型的标签列表，可在 Hugging Face Hub 上进行过滤时使用。默认为无。

ignore_metadata_errors (`str`) ：如果为 True，则解析元数据部分时的错误将被忽略。在此过程中可能会丢失一些信息。使用它的风险由您自行承担。

kwargs（`dict`，*可选*）：将添加到模型卡的附加元数据。默认为无。

Hugging Face Hub 使用的模型卡元数据包含在 README.md 顶部

示例：
```python
>>> from huggingface_hub import ModelCardData
>>> card_data = ModelCardData(
...     language="en",
...     license="mit",
...     library_name="timm",
...     tags=['image-classification', 'resnet'],
... )
>>> card_data.to_dict()
{'language': 'en', 'license': 'mit', 'library_name': 'timm', 'tags': ['image-classification', 'resnet']}

```

## 数据集卡

数据集卡在 ML 社区中也称为数据卡。

### DatasetCard[[huggingface_hub.DatasetCard]]

#### Huggingface_hub.DatasetCard[[huggingface_hub.DatasetCard]]

```python
huggingface_hub.DatasetCard(content: str, ignore_metadata_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L420)

#### from_template[[huggingface_hub.DatasetCard.from_template]]

```python
from_template(card_data: DatasetCardData, template_path: str | None = None, template_str: str | None = None, **template_kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L425)

**参数：**

card_data (`huggingface_hub.DatasetCardData`) ：huggingface_hub.DatasetCardData 实例，其中包含要包含在 Hugging Face Hub 上数据集卡的 YAML 标头中的元数据。template_path (`str`, *可选*) ：带有可选 Jinja 模板变量的 Markdown 文件的路径，可以用 `template_kwargs` 填充。默认为默认模板。

template_str (`str`, *可选*) ：带有可选变量的原始 Jinja 模板字符串。当`template_path`和默认模板都不合适时使用。如果还提供了`template_path`，则忽略。

**返回：** [huggingface_hub.DatasetCard](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.DatasetCard)

具有指定卡片数据和内容的 DatasetCard 实例
模板。

从模板初始化 DatasetCard。默认情况下，它使用默认模板，可以在此处找到该模板：
https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md

模板是 Jinja2 模板，可以通过传递关键字参数进行自定义。

示例：
```python
>>> from huggingface_hub import DatasetCard, DatasetCardData

>>> # Using the Default Template
>>> card_data = DatasetCardData(
...     language='en',
...     license='mit',
...     annotations_creators='crowdsourced',
...     task_categories=['text-classification'],
...     task_ids=['sentiment-classification', 'text-scoring'],
...     multilinguality='monolingual',
...     pretty_name='My Text Classification Dataset',
... )
>>> card = DatasetCard.from_template(
...     card_data,
...     pretty_name=card_data.pretty_name,
... )

>>> # Using a Custom Template
>>> card_data = DatasetCardData(
...     language='en',
...     license='mit',
... )
>>> card = DatasetCard.from_template(
...     card_data=card_data,
...     template_path='./src/huggingface_hub/templates/datasetcard_template.md',
...     custom_template_var='custom value',  # will be replaced in template if it exists
... )

```

### DatasetCardData[[huggingface_hub.DatasetCardData]]

#### Huggingface_hub.DatasetCardData[[huggingface_hub.DatasetCardData]]

```python
huggingface_hub.DatasetCardData(language: str | list[str] | None = None, license: str | list[str] | None = None, annotations_creators: str | list[str] | None = None, language_creators: str | list[str] | None = None, multilinguality: str | list[str] | None = None, size_categories: str | list[str] | None = None, source_datasets: list[str] | None = None, task_categories: str | list[str] | None = None, task_ids: str | list[str] | None = None, paperswithcode_id: str | None = None, pretty_name: str | None = None, train_eval_index: dict | None = None, config_names: str | list[str] | None = None, ignore_metadata_errors: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L400)

**参数：**

语言（`list[str]`，*可选*）：数据集数据或元数据的语言。它必须是 ISO 639-1、639-2 或 639-3 代码（两个/三个字母），或特殊值，如“代码”、“多语言”。许可证（`Union[str, list[str]]`，*可选*）：该数据集的许可证。示例：apache-2.0 或 https://huggingface.co/docs/hub/repositories-licenses 中的任何许可证。

annotations_creators（`Union[str, list[str]]`，*可选*）：如何创建数据集的注释。选项包括：“找到”、“众包”、“专家生成”、“机器生成”、“无注释”、“其他”。

language_creators（`Union[str, list[str]]`，*可选*）：如何创建数据集中基于文本的数据。选项有：“找到”、“众包”、“专家生成”、“机器生成”、“其他”

多语言性（`Union[str, list[str]]`，*可选*）：数据集是否是多语言的。选项有：“单语言”、“多语言”、“翻译”、“其他”。

size_categories (`Union[str, list[str]]`, *可选*) ：数据集中的示例数量。选项有：“n<1K”、“1K1T”和“其他”。

source_datasets (`list[str]]`, *可选*) ：指示数据集是原始数据集还是从另一个现有数据集扩展而来。选项有：“原始”和“扩展”。

task_categories (`Union[str, list[str]]`, *可选*)：数据集支持哪些类别的任务？

task_ids (`Union[str, list[str]]`, *可选*)：数据集支持哪些特定任务？paperwithcode_id (`str`, *可选*) ：PapersWithCode 上数据集的 ID。

Pretty_name (`str`, *可选*) ：数据集的更易于理解的名称。 （例如“猫大战狗”）

train_eval_index (`dict`, *可选*) ：描述在 Hub 上进行评估所需规范的字典。如果未提供，它将从 kwargs 的“train-eval-index”键中收集。

config_names (`Union[str, list[str]]`, *可选*) ：数据集的可用数据集配置列表。

Hugging Face Hub 使用的数据集卡元数据包含在 README.md 顶部

## 太空卡

### SpaceCard[[huggingface_hub.SpaceCard]]

#### Huggingface_hub.SpaceCard[[huggingface_hub.SpaceCard]]

```python
huggingface_hub.SpaceCard(content: str, ignore_metadata_errors: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L488)

### SpaceCardData[[huggingface_hub.SpaceCardData]]

#### Huggingface_hub.SpaceCardData[[huggingface_hub.SpaceCardData]]

```python
huggingface_hub.SpaceCardData(title: str | None = None, sdk: str | None = None, sdk_version: str | None = None, python_version: str | None = None, app_file: str | None = None, app_port: int | None = None, license: str | None = None, duplicated_from: str | None = None, models: list[str] | None = None, datasets: list[str] | None = None, tags: list[str] | None = None, ignore_metadata_errors: bool = False, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L480)

**参数：**

title (`str`, *可选*) : 空间的标题。

sdk（`str`，*可选*）：空间的 SDK（`gradio`、`streamlit`、`docker` 或 `static` 之一）。

sdk_version (`str`, *可选*) ：所使用的 SDK 版本（如果是 Gradio/Streamlit sdk）。

python_version (`str`, *可选*) ：Space 中使用的 Python 版本（如果 Gradio/Streamlit sdk）。app_file（`str`，*可选*）：主应用程序文件的路径（包含gradio或streamlit Python代码，或静态html代码）。路径是相对于存储库的根目录的。

app_port (`str`, *可选*) ：应用程序运行的端口。仅当 sdk 为 `docker` 时使用。

许可证（`str`，*可选*）：该模型的许可证。示例：apache-2.0 或 https://huggingface.co/docs/hub/repositories-licenses 中的任何许可证。

重复的_from（`str`，*可选*）：原始空间的ID（如果这是重复的空间）。

models (`list[str]`, *可选*) : 与此空间相关的模型列表。应该是在 https://hf.co/models 上找到的数据集 ID。

datasets (`list[str]`，*可选*)：与此空间相关的数据集列表。应该是在 https://hf.co/datasets 上找到的数据集 ID。

标签（`list[str]`，*可选*）：添加到空间的标签列表，可在集线器上过滤时使用。

ignore_metadata_errors (`str`) ：如果为 True，则解析元数据部分时的错误将被忽略。在此过程中可能会丢失一些信息。使用它的风险由您自行承担。

kwargs（`dict`，*可选*）：将添加到空间卡的附加元数据。Hugging Face Hub 使用的空间卡元数据包含在 README.md 顶部

要获取 Spaces 配置的详尽参考，请访问 https://huggingface.co/docs/hub/spaces-config-reference#spaces-configuration-reference。

示例：
```python
>>> from huggingface_hub import SpaceCardData
>>> card_data = SpaceCardData(
...     title="Dreambooth Training",
...     license="mit",
...     sdk="gradio",
...     duplicated_from="multimodalart/dreambooth-training"
... )
>>> card_data.to_dict()
{'title': 'Dreambooth Training', 'sdk': 'gradio', 'license': 'mit', 'duplicated_from': 'multimodalart/dreambooth-training'}
```

## 实用程序

### EvalResult[[huggingface_hub.EvalResult]]

#### Huggingface_hub.EvalResult[[huggingface_hub.EvalResult]]

```python
huggingface_hub.EvalResult(task_type: str, dataset_type: str, dataset_name: str, metric_type: str, metric_value: typing.Any, task_name: str | None = None, dataset_config: str | None = None, dataset_split: str | None = None, dataset_revision: str | None = None, dataset_args: dict[str, typing.Any] | None = None, metric_name: str | None = None, metric_config: str | None = None, metric_args: dict[str, typing.Any] | None = None, verified: bool | None = None, verify_token: str | None = None, source_name: str | None = None, source_url: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L13)

**参数：**

task_type (`str`) ：任务标识符。示例：“图像分类”。

dataset_type (`str`) ：数据集标识符。示例：“common_voice”。使用 https://hf.co/datasets 中的数据集 ID。

dataset_name (`str`)：数据集的一个漂亮名称。示例：“通用语音（法语）”。

metric_type (`str`) ：指标标识符。例如：“我们”。使用 https://hf.co/metrics 中的指标 ID。

metric_value (`Any`) ：指标值。示例：0.9 或“20.0 ± 1.2”。

task_name (`str`, *可选*) ：任务的漂亮名称。示例：“语音识别”。dataset_config (`str`, *可选*) ：`load_dataset()`中使用的数据集配置的名称。示例：`load_dataset("common_voice", "fr")` 中的 fr。有关更多信息，请参阅 `datasets` 文档：https://hf.co/docs/datasets/package_reference/loading_methods#datasets.load_dataset.name

dataset_split (`str`, *可选*) ：`load_dataset()`中使用的分割。例如：“测试”。

dataset_revision（`str`，*可选*）：`load_dataset()`中使用的数据集的修订版（AKA Git Sha）。示例：5503434ddd753f426f4b38109466949a1217c2bb

dataset_args (`dict[str, Any]`, *可选*) ：`Metric.compute()`期间传递的参数。 `bleu` 示例：`{"max_order": 4}`

metric_name (`str`, *可选*) ：指标的漂亮名称。示例：“测试 WER”。

metric_config (`str`, *可选*) ：`load_metric()`中使用的指标配置的名称。示例：`load_metric("bleurt", "bleurt-large-512")` 中的 bleurt-large-512。有关更多信息，请参阅 `datasets` 文档：https://huggingface.co/docs/datasets/v2.1.0/en/loading#load-configurations

metric_args (`dict[str, Any]`, *可选*) ：`Metric.compute()`期间传递的参数。 `bleu` 示例：max_order: 4

已验证（`bool`，*可选*）：指示指标是否源自 Hugging Face 的 [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator)。由 Hugging Face 自动计算，无需设置。verify_token (`str`, *可选*) ：一个 JSON Web Token，用于验证指标是否源自 Hugging Face 的 [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator)。

source_name (`str`, *可选*) ：评估结果的来源名称。示例：“打开 LLM 排行榜”。

source_url (`str`, *可选*) ：评估结果来源的URL。示例：“https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard”。

在模型卡的模型索引中找到的个人评估结果的扁平化表示。

有关模型索引规范的更多信息，请参阅 https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1。

#### is_equal_ except_value[[huggingface_hub.EvalResult.is_equal_ except_value]]

```python
is_equal_except_value(other: EvalResult)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L145)

如果 `self` 和 `other` 描述完全相同的度量但具有
不同的价值。

### model_index_to_eval_results[[huggingface_hub.repocard_data.model_index_to_eval_results]]

#### Huggingface_hub.repocard_data.model_index_to_eval_results[[huggingface_hub.repocard_data.model_index_to_eval_results]]

```python
huggingface_hub.repocard_data.model_index_to_eval_results(model_index: list)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L561)

**参数：**model_index (`list[dict[str, Any]]`) ：模型索引数据结构，可能来自 Hugging Face Hub 上的 README.md 文件。

**返回：** 模型名称 (`str`)

在模型索引中找到的模型名称。这用作
排行榜（如 PapersWithCode）上模型的标识符。
评估结果（`list[EvalResult]`）：
包含指标的`huggingface_hub.EvalResult`对象列表
在提供的 model_index 中报告。

接受模型索引并返回模型名称和 `huggingface_hub.EvalResult` 对象列表。

模型索引的详细规格可以在这里找到：
https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1

示例：
```python
>>> from huggingface_hub.repocard_data import model_index_to_eval_results
>>> # Define a minimal model index
>>> model_index = [
...     {
...         "name": "my-cool-model",
...         "results": [
...             {
...                 "task": {
...                     "type": "image-classification"
...                 },
...                 "dataset": {
...                     "type": "beans",
...                     "name": "Beans"
...                 },
...                 "metrics": [
...                     {
...                         "type": "accuracy",
...                         "value": 0.9
...                     }
...                 ]
...             }
...         ]
...     }
... ]
>>> model_name, eval_results = model_index_to_eval_results(model_index)
>>> model_name
'my-cool-model'
>>> eval_results[0].task_type
'image-classification'
>>> eval_results[0].metric_type
'accuracy'

```

### eval_results_to_model_index[[huggingface_hub.repocard_data.eval_results_to_model_index]]

#### Huggingface_hub.repocard_data.eval_results_to_model_index[[huggingface_hub.repocard_data.eval_results_to_model_index]]

```python
huggingface_hub.repocard_data.eval_results_to_model_index(model_name: str, eval_results: list)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard_data.py#L677)

**参数：**

model_name (`str`) ：模型的名称（例如“my-cool-model”）。这用作 PapersWithCode 等排行榜上模型的标识符。

eval_results (`list[EvalResult]`) ：包含要在模型索引中报告的指标的`huggingface_hub.EvalResult`对象列表。

**返回：** model_index (`list[dict[str, Any]]`)

eval_results 转换为模型索引。接受给定的模型名称和 `huggingface_hub.EvalResult` 列表并返回
有效的模型索引将与预期的格式兼容
拥抱脸部中心。

示例：
```python
>>> from huggingface_hub.repocard_data import eval_results_to_model_index, EvalResult
>>> # Define minimal eval_results
>>> eval_results = [
...     EvalResult(
...         task_type="image-classification",  # Required
...         dataset_type="beans",  # Required
...         dataset_name="Beans",  # Required
...         metric_type="accuracy",  # Required
...         metric_value=0.9,  # Required
...     )
... ]
>>> eval_results_to_model_index("my-cool-model", eval_results)
[{'name': 'my-cool-model', 'results': [{'task': {'type': 'image-classification'}, 'dataset': {'name': 'Beans', 'type': 'beans'}, 'metrics': [{'type': 'accuracy', 'value': 0.9}]}]}]

```

###metadata_eval_result[[huggingface_hub.metadata_eval_result]]

#### Huggingface_hub.metadata_eval_result[[huggingface_hub.metadata_eval_result]]

```python
huggingface_hub.metadata_eval_result(model_pretty_name: str, task_pretty_name: str, task_id: str, metrics_pretty_name: str, metrics_id: str, metrics_value: typing.Any, dataset_pretty_name: str, dataset_id: str, metrics_config: str | None = None, metrics_verified: bool = False, dataset_config: str | None = None, dataset_split: str | None = None, dataset_revision: str | None = None, metrics_verification_token: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L560)

**参数：**

model_pretty_name (`str`) ：模型的自然语言名称。

task_pretty_name (`str`) ：任务的自然语言名称。

task_id (`str`) ：示例：自动语音识别。任务 ID。

metrics_pretty_name (`str`) ：自然语言中指标的名称。示例：测试 WER。

metrics_id (`str`) ：示例：wer.来自 https://hf.co/metrics 的指标 ID。

metrics_value (`Any`) ：指标的值。示例：20.0 或“20.0 ± 1.2”。

dataset_pretty_name (`str`) ：数据集的自然语言名称。

dataset_id (`str`) ：示例：common_voice。来自 https://hf.co/datasets 的数据集 ID。

metrics_config (`str`, *可选*) ：`load_metric()`中使用的指标配置的名称。示例：`load_metric("bleurt", "bleurt-large-512")` 中的 bleurt-large-512。metrics_verified（`bool`，*可选*，默认为`False`）：指示指标是否源自Hugging Face的[evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator)。由 Hugging Face 自动计算，无需设置。

dataset_config（`str`，*可选*）：示例：fr. `load_dataset()`中使用的数据集配置的名称。

dataset_split (`str`, *可选*) ：示例：测试。 `load_dataset()` 中使用的数据集分割的名称。

dataset_revision（`str`，*可选*）：示例：5503434ddd753f426f4b38109466949a1217c2bb。 `load_dataset()`中使用的数据集修订版的名称。

metrics_verification_token（`bool`，*可选*）：一个 JSON Web Token，用于验证指标是否源自 Hugging Face 的 [evaluation service](https://huggingface.co/spaces/autoevaluate/model-evaluator)。

**退货：** `dict`

元数据字典，其中包含在数据集上评估的模型的结果。

使用在数据集上评估的模型的结果创建元数据字典。

示例：
```python
>>> from huggingface_hub import metadata_eval_result
>>> results = metadata_eval_result(
...         model_pretty_name="RoBERTa fine-tuned on ReactionGIF",
...         task_pretty_name="Text Classification",
...         task_id="text-classification",
...         metrics_pretty_name="Accuracy",
...         metrics_id="accuracy",
...         metrics_value=0.2662102282047272,
...         dataset_pretty_name="ReactionJPEG",
...         dataset_id="julien-c/reactionjpeg",
...         dataset_config="default",
...         dataset_split="test",
... )
>>> results == {
...     'model-index': [
...         {
...             'name': 'RoBERTa fine-tuned on ReactionGIF',
...             'results': [
...                 {
...                     'task': {
...                         'type': 'text-classification',
...                         'name': 'Text Classification'
...                     },
...                     'dataset': {
...                         'name': 'ReactionJPEG',
...                         'type': 'julien-c/reactionjpeg',
...                         'config': 'default',
...                         'split': 'test'
...                     },
...                     'metrics': [
...                         {
...                             'type': 'accuracy',
...                             'value': 0.2662102282047272,
...                             'name': 'Accuracy',
...                             'verified': False
...                         }
...                     ]
...                 }
...             ]
...         }
...     ]
... }
True

```

### 元数据更新[[huggingface_hub.metadata_update]]

#### Huggingface_hub.metadata_update[[huggingface_hub.metadata_update]]

```python
huggingface_hub.metadata_update(repo_id: str, metadata: dict, repo_type: str | None = None, overwrite: bool = False, token: str | None = None, commit_message: str | None = None, commit_description: str | None = None, revision: str | None = None, create_pr: bool = False, parent_commit: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/repocard.py#L688)

**参数：**

repo_id (`str`) ：存储库的名称。

元数据 (`dict`) ：包含要更新的元数据的字典。repo_type (`str`, *可选*) ：如果更新到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果更新到模型，则设置为 `None` 或 `"model"`。默认为`None`。

覆盖（`bool`，*可选*，默认为`False`）：如果设置为`True`，则可以覆盖现有字段，否则尝试覆盖现有字段将导致错误。

token (`str`, *可选*) : Hugging Face 身份验证令牌。

commit_message (`str`, *可选*) ：生成的提交的摘要/标题/第一行。默认为 `f"Update metadata with huggingface_hub"`

commit_description (`str`, *可选*) : 生成的提交的描述

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

create_pr (`boolean`, *可选*) ：是否使用该提交从 `revision` 创建拉取请求。默认为`False`。Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交过多，则特别有用。

**退货：** `str`

更新卡元数据的提交的 URL。

更新 Hugging Face Hub 上存储库的 README.md 中的元数据。
如果 README.md 文件尚不存在，则会使用元数据创建一个新文件并
默认的 ModelCard 或 DatasetCard 模板。对于`space` repo，会抛出错误
因为如果没有 `README.md` 文件，空间就无法存在。

示例：
```python
>>> from huggingface_hub import metadata_update
>>> metadata = {'model-index': [{'name': 'RoBERTa fine-tuned on ReactionGIF',
...             'results': [{'dataset': {'name': 'ReactionGIF',
...                                      'type': 'julien-c/reactiongif'},
...                           'metrics': [{'name': 'Recall',
...                                        'type': 'recall',
...                                        'value': 0.7762102282047272}],
...                          'task': {'name': 'Text Classification',
...                                   'type': 'text-classification'}}]}]}
>>> url = metadata_update("hf-internal-testing/reactiongif-roberta-card", metadata)

```

### 推理类型
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/inference_types.md