<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Mixins 和序列化方法

## 混合

`huggingface_hub` 库提供了一系列 mixin，可以用作对象的父类，以便
提供简单的上传和下载功能。查看我们的[integration guide](../guides/integrations)来学习
如何将任何 ML 框架与 Hub 集成。

### 通用[[huggingface_hub.ModelHubMixin]]

#### Huggingface_hub.ModelHubMixin[[huggingface_hub.ModelHubMixin]]

```python
huggingface_hub.ModelHubMixin(*args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L77)

**参数：**

repo_url (`str`, *可选*) ：库存储库的 URL。用于生成模型卡。

paper_url (`str`, *可选*) : 图书馆论文的 URL。用于生成模型卡。

docs_url (`str`, *可选*) ：库文档的 URL。用于生成模型卡。

model_card_template (`str`, *可选*) ：模型卡的模板。用于生成模型卡。默认为通用模板。

语言（`str`或`list[str]`，*可选*）：库支持的语言。用于生成模型卡。

library_name (`str`, *可选*) : 集成 ModelHubMixin 的库的名称。用于生成模型卡。

许可证（`str`，*可选*）：集成 ModelHubMixin 的库的许可证。用于生成模型卡。例如：“apache-2.0”license_name (`str`, *可选*) : 集成 ModelHubMixin 的库的名称。用于生成模型卡。仅当 `license` 设置为 `other` 时才使用。例如：“coqui-public-model-license”。

license_link (`str`, *可选*) ：集成 ModelHubMixin 的库的许可证的 URL。用于生成模型卡。仅当 `license` 设置为 `other` 并且设置了 `license_name` 时才使用。例如：“https://coqui.ai/cpml”。

pipeline_tag (`str`, *可选*) ：管道的标签。用于生成模型卡。例如。 “文本分类”。

标签（`list[str]`，*可选*）：要添加到模型卡的标签。用于生成模型卡。例如。 [“计算机视觉”]

coders (`dict[Type, tuple[Callable, Callable]]`，*可选*)：自定义类型及其编码器/解码器的字典。用于对默认情况下不可 jsonable 的参数进行编码/解码。例如。数据类、argparse.Namespace、OmegaConf 等。

一个通用 mixin，用于将任何机器学习框架与 Hub 集成。

要集成您的框架，您的模型类必须继承自此类。用于保存/加载模型的自定义逻辑
必须在`_from_pretrained`和`_save_pretrained`中被覆盖。 [PyTorchModelHubMixin](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)就是一个很好的例子
mixin 与 Hub 的集成。查看我们的[integration guide](../guides/integrations)以获取更多说明。当继承[ModelHubMixin](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)时，可以定义类级别的属性。这些属性不会传递给
`__init__` 但对于类定义本身。这对于定义有关库集成的元数据很有用
[ModelHubMixin](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)。

有关如何将 mixin 与您的库集成的更多详细信息，请查看 [integration guide](../guides/integrations)。

示例：

```python
>>> from huggingface_hub import ModelHubMixin

# Inherit from ModelHubMixin
>>> class MyCustomModel(
...         ModelHubMixin,
...         library_name="my-library",
...         tags=["computer-vision"],
...         repo_url="https://github.com/huggingface/my-cool-library",
...         paper_url="https://arxiv.org/abs/2304.12244",
...         docs_url="https://huggingface.co/docs/my-cool-library",
...         # ^ optional metadata to generate model card
...     ):
...     def __init__(self, size: int = 512, device: str = "cpu"):
...         # define how to initialize your model
...         super().__init__()
...         ...
...
...     def _save_pretrained(self, save_directory: Path) -> None:
...         # define how to serialize your model
...         ...
...
...     @classmethod
...     def from_pretrained(
...         cls: type[T],
...         pretrained_model_name_or_path: Union[str, Path],
...         *,
...         force_download: bool = False,
...         token: Optional[Union[str, bool]] = None,
...         cache_dir: Optional[Union[str, Path]] = None,
...         local_files_only: bool = False,
...         revision: Optional[str] = None,
...         **model_kwargs,
...     ) -> T:
...         # define how to deserialize your model
...         ...

>>> model = MyCustomModel(size=256, device="gpu")

# Save model weights to local directory
>>> model.save_pretrained("my-awesome-model")

# Push model weights to the Hub
>>> model.push_to_hub("my-awesome-model")

# Download and initialize weights from the Hub
>>> reloaded_model = MyCustomModel.from_pretrained("username/my-awesome-model")
>>> reloaded_model.size
256

# Model card has been correctly populated
>>> from huggingface_hub import ModelCard
>>> card = ModelCard.load("username/my-awesome-model")
>>> card.data.tags
["x-custom-tag", "pytorch_model_hub_mixin", "model_hub_mixin"]
>>> card.data.library_name
"my-library"
```

#### _save_pretrained[[huggingface_hub.ModelHubMixin._save_pretrained]]

```python
_save_pretrained(save_directory: Path)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L451)

**参数：**

save_directory（`str`或`Path`）：保存模型权重和配置的目录路径。

在子类中覆盖此方法来定义如何保存模型。
查看我们的[integration guide](../guides/integrations)以获取说明。

#### _from_pretrained[[huggingface_hub.ModelHubMixin._from_pretrained]]

```python
_from_pretrained(model_id: str, revision: str | None, cache_dir: str | pathlib.Path | None, force_download: bool, local_files_only: bool, token: str | bool | None, **model_kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L578)

**参数：**

model_id (`str`) ：从 Huggingface Hub 加载的模型的 ID（例如 `bigscience/bloom`）。

修订版（`str`，*可选*）：集线器上模型的修订版。可以是分支名称、git 标签或任何提交 ID。默认为 `main` 分支上的最新提交。

force_download（`bool`，*可选*，默认为`False`）：是否强制（重新）从 Hub 下载模型权重和配置文件，覆盖现有缓存。token（`str` 或 `bool`，*可选*）：用作远程文件的 HTTP 承载授权的令牌。默认情况下，它将使用运行`hf auth login`时缓存的令牌。

cache_dir (`str`, `Path`, *可选*) : 存储缓存文件的文件夹路径。

local_files_only (`bool`，*可选*，默认为`False`)：如果`True`，避免下载文件，并返回本地缓存文件的路径（如果存在）。

model_kwargs ：传递给 [_from_pretrained()](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin._from_pretrained) 方法的附加关键字参数。

在子类中覆盖此方法以定义如何从预训练中加载模型。

在加载文件之前，使用 [hf_hub_download()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 或 [snapshot_download()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 从集线器下载文件。大多数
作为输入的 args 可以直接传递给这两个方法。如果需要，您可以为此添加更多参数
使用“model_kwargs”的方法。例如 `PyTorchModelHubMixin._from_pretrained()` 将 `map_location` 作为输入
参数来设置模型应加载到哪个设备上。

查看我们的[integration guide](../guides/integrations)以获取更多说明。

#### from_pretrained[[huggingface_hub.ModelHubMixin.from_pretrained]]

```python
from_pretrained(pretrained_model_name_or_path: str | pathlib.Path, force_download: bool = False, token: str | bool | None = None, cache_dir: str | pathlib.Path | None = None, local_files_only: bool = False, revision: str | None = None, **model_kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L462)

**参数：**pretrained_model_name_or_path (`str`, `Path`) ：- Hub 上托管的模型的 `model_id`（字符串），例如`bigscience/bloom`。 - 或者包含使用 [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) 保存的模型权重的 `directory` 路径，例如 `../path/to/my_model_directory/`。

修订版（`str`，*可选*）：集线器上模型的修订版。可以是分支名称、git 标签或任何提交 ID。默认为 `main` 分支上的最新提交。

force_download（`bool`，*可选*，默认为`False`）：是否强制（重新）从 Hub 下载模型权重和配置文件，覆盖现有缓存。

token（`str` 或 `bool`，*可选*）：用作远程文件的 HTTP 承载授权的令牌。默认情况下，它将使用运行`hf auth login`时缓存的令牌。

cache_dir (`str`, `Path`, *可选*) : 存储缓存文件的文件夹路径。

local_files_only (`bool`，*可选*，默认为`False`) ：如果`True`，则避免下载文件，并返回本地缓存文件的路径（如果存在）。

model_kwargs (`dict`, *可选*) ：在初始化期间传递给模型的附加 kwargs。

从 Huggingface Hub 下载模型并实例化它。

####push_to_hub[[huggingface_hub.ModelHubMixin.push_to_hub]]

```python
push_to_hub(repo_id: str, config: dict | huggingface_hub.hub_mixin.DataclassInstance | None = None, commit_message: str = 'Push model using huggingface_hub.', private: bool | None = None, token: str | None = None, branch: str | None = None, create_pr: bool | None = None, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, delete_patterns: list[str] | str | None = None, model_card_kwargs: dict[str, typing.Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L620)**参数：**

repo_id (`str`) ：要推送到的存储库的 ID（例如：`"username/my-model"`）。

config（`dict`或`DataclassInstance`，*可选*）：指定为键/值字典或数据类实例的模型配置。

commit_message (`str`, *可选*) ：推送时提交的消息。

private (`bool`, *可选*) : 创建的存储库是否应该是私有的。如果`None`（默认），则存储库将是公开的，除非组织默认为私有。

token（`str`，*可选*）：用作远程文件的 HTTP 承载授权的令牌。默认情况下，它将使用运行`hf auth login`时缓存的令牌。

分支（`str`，*可选*）：要推送模型的 git 分支。默认为 `"main"`。

create_pr (`boolean`, *可选*) ：是否使用该提交从 `branch` 创建拉取请求。默认为`False`。

allowed_pa​​tterns (`list[str]` 或 `str`, *可选*) ：如果提供，则仅推送与至少一种模式匹配的文件。

ignore_patterns（`list[str]` 或 `str`，*可选*）：如果提供，则不会推送与任何模式匹配的文件。

delete_patterns（`list[str]`或`str`，*可选*）：如果提供，匹配任何模式的远程文件将从存储库中删除。model_card_kwargs (`dict[str, Any]`, *可选*) ：传递给模型卡模板以自定义模型卡的附加参数。

**退货：**

给定存储库中模型提交的 URL。

将模型检查点上传到 Hub。

使用 `allow_patterns` 和 `ignore_patterns` 精确过滤哪些文件应推送到集线器。使用
`delete_patterns` 在同一提交中删除现有的远程文件。更多信息请参见[upload_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)参考
详细信息。

#### save_pretrained[[huggingface_hub.ModelHubMixin.save_pretrained]]

```python
save_pretrained(save_directory: str | pathlib.Path, config: dict | huggingface_hub.hub_mixin.DataclassInstance | None = None, repo_id: str | None = None, push_to_hub: bool = False, model_card_kwargs: dict[str, typing.Any] | None = None, **push_to_hub_kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L383)

**参数：**

save_directory（`str`或`Path`）：保存模型权重和配置的目录路径。

config（`dict`或`DataclassInstance`，*可选*）：指定为键/值字典或数据类实例的模型配置。

Push_to_hub（`bool`，*可选*，默认为`False`）：保存模型后是否将其推送到 Huggingface Hub。

repo_id（`str`，*可选*）：Hub 上存储库的 ID。仅当`push_to_hub=True`时使用。如果未提供，将默认为文件夹名称。

model_card_kwargs (`dict[str, Any]`, *可选*) ：传递给模型卡模板以自定义模型卡的附加参数。Push_to_hub_kwargs ：传递给 [push_to_hub()](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin.push_to_hub) 方法的附加关键字参数。

**返回：** `str` 或 `None`

如果 `push_to_hub=True`，则为 Hub 上提交的 url，否则为 `None`。

将权重保存在本地目录中。

### PyTorch[[huggingface_hub.PyTorchModelHubMixin]]

#### Huggingface_hub.PyTorchModelHubMixin[[huggingface_hub.PyTorchModelHubMixin]]

```python
huggingface_hub.PyTorchModelHubMixin(*args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hub_mixin.py#L703)

实现[ModelHubMixin](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)，为 PyTorch 模型提供模型 Hub 上传/下载功能。型号
默认情况下使用`model.eval()`设置为评估模式（辍学模块已停用）。为了训练模型，
您应该首先使用 `model.train()` 将其设置回训练模式。

有关如何使用 mixin 的更多详细信息，请参阅[ModelHubMixin](/docs/huggingface_hub/v1.27.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)。

示例：

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin

>>> class MyModel(
...         nn.Module,
...         PyTorchModelHubMixin,
...         library_name="keras-nlp",
...         repo_url="https://github.com/keras-team/keras-nlp",
...         paper_url="https://arxiv.org/abs/2304.12244",
...         docs_url="https://keras.io/keras_nlp/",
...         # ^ optional metadata to generate model card
...     ):
...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
...         super().__init__()
...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
...         self.linear = nn.Linear(output_size, vocab_size)

...     def forward(self, x):
...         return self.linear(x + self.param)
>>> model = MyModel(hidden_size=256)

# Save model weights to local directory
>>> model.save_pretrained("my-awesome-model")

# Push model weights to the Hub
>>> model.push_to_hub("my-awesome-model")

# Download and initialize weights from the Hub
>>> model = MyModel.from_pretrained("username/my-awesome-model")
>>> model.hidden_size
256
```

### Fastai[[huggingface_hub.from_pretrained_fastai]]

#### Huggingface_hub.from_pretrained_fastai[[huggingface_hub.from_pretrained_fastai]]

```python
huggingface_hub.from_pretrained_fastai(repo_id: str, revision: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/fastai_utils.py#L289)

**参数：**repo_id (`str`) : 腌制的 fastai.Learner 所在的位置。它可以是以下两者之一： - 托管在 Hugging Face Hub 上。例如：“espejelomar/fatai-pet-breeds-classification”或“distilgpt2”。您可以通过在 `repo_id` 末尾附加 `@` 来添加 `revision`。例如：`dbmdz/bert-base-german-cased@main`。修订版是要使用的特定模型版本。由于我们使用基于 git 的系统在 Hugging Face Hub 上存储模型和其他工件，因此它可以是分支名称、标签名称或提交 ID。 - 本地托管。 `repo_id` 将是一个包含 pickle 和 pyproject.toml 的目录，指示用于构建 `fastai.Learner` 的 fastai 和 fastcore 版本。例如：`./my_model_directory/`。

revision (`str`, *可选*) ：下载存储库文件的修订版本。请参阅`snapshot_download`的文档。

**退货：**

`repo_id` 存储库中的 `fastai.Learner` 模型。

从 Hub 或本地目录加载预训练的 fastai 模型。

#### Huggingface_hub.push_to_hub_fastai[[huggingface_hub.push_to_hub_fastai]]

```python
huggingface_hub.push_to_hub_fastai(learner, repo_id: str, commit_message: str = 'Push FastAI model using huggingface_hub.', private: bool | None = None, token: str | None = None, config: dict | None = None, branch: str | None = None, create_pr: bool | None = None, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, delete_patterns: list[str] | str | None = None, api_endpoint: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/fastai_utils.py#L334)

**参数：**

learner (*Learner*) ：您想要推送到 Hub 的 *fastai.Learner。repo_id (*str*) ：Hub 中模型的存储库 ID，格式为“namespace/repo_name”。命名空间可以是您的个人帐户或您具有写入权限的组织（例如“stanfordnlp/stanza-de”）。

commit_message (*str`, *optional*) : Message to commit while pushing. Will default to `"添加模型"`.

private (*bool*, *可选*) ：创建的存储库是否应该是私有的。如果“无”（默认），则默认为公开，除非组织的默认设置为私有。

token (*str*, *可选*) ：用作远程文件的 HTTP 承载授权的 Hugging Face 帐户令牌。如果`None`，将通过提示询问令牌。

config (*dict*, *可选*) ：与模型权重一起保存的配置对象。

分支 (*str*, *可选*) ：要推送模型的 git 分支。这默认为存储库中指定的默认分支，默认为*“main”*。

create_pr (*boolean*, *可选*) ：是否使用该提交从 *branch* 创建拉取请求。默认为*假*。

api_endpoint (*str*, *可选*) ：将模型推送到中心时使用的 API 端点。allow_patterns (*list[str]* 或 *str*, *可选*) ：如果提供，则仅推送至少匹配一种模式的文件。

ignore_patterns (*list[str]* 或 *str*, *可选*) ：如果提供，则不会推送与任何模式匹配的文件。

delete_patterns (*list[str]* 或 *str*, *可选*) ：如果提供，匹配任何模式的远程文件将从存储库中删除。

**退货：**

给定存储库中模型提交的 URL。

将学习者检查点文件上传到中心。

使用 *allow_patterns* 和 *ignore_patterns* 精确过滤哪些文件应推送到集线器。使用
*delete_patterns* 删除同一提交中现有的远程文件。有关更多信息，请参阅 [*upload_folder*] 参考
详细信息。

> [!提示]
> 引发以下错误：
>
> - [*ValueError*](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果用户未登录 Hugging Face Hub。

### 环境变量
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/environment_variables.md