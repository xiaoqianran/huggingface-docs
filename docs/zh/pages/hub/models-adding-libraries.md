<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 将您的库与 Hub 集成

Hugging Face Hub 旨在促进共享机器学习模型、检查点和工件。这项努力包括将 Hub 集成到社区中许多令人惊叹的第三方库中。一些已经集成的包括 [spaCy](https://spacy.io/usage/projects#huggingface_hub)、[Sentence Transformers](https://sbert.net/)、[OpenCLIP](https://github.com/mlfoundations/open_clip) 和 [timm](https://huggingface.co/docs/timm/index) 等。集成意味着用户可以直接从您的库下载文件并将文件上传到集线器。我们希望您能够整合您的图书馆，并与我们一起为每个人实现人工智能的民主化。

将 Hub 与您的图书馆集成可带来许多好处，包括：

- 为您和您的用户提供免费模型托管。
- 内置文件版本控制 - 即使对于大文件 - 通过 [Git-Xet](./xet/using-xet-storage#git-xet) 成为可能。
- 社区功能（讨论、拉取请求、点赞）。
- 与您的库一起运行的所有模型的使用指标。

本教程将帮助您将 Hub 集成到您的库中，以便您的用户可以从 Hub 提供的所有功能中受益。

在开始之前，我们建议您创建一个 [Hugging Face account](https://huggingface.co/join)，您可以从中管理您的存储库和文件。如果您需要集成方面的帮助，请随时打开[issue](https://github.com/huggingface/huggingface_hub/issues/new/choose)，我们将非常乐意为您提供帮助。

## 实施

实现库与 Hub 的集成通常意味着提供内置方法来从 Hub 加载模型并允许用户将新模型推送到 Hub。本节将介绍如何使用 `huggingface_hub` 库执行此操作的基础知识。如需更深入的指导，请查看[this guide](https://huggingface.co/docs/huggingface_hub/guides/integrations)。 

### 安装

要将您的库与 Hub 集成，您需要添加 `huggingface_hub` 库作为依赖项：

```bash
pip install huggingface_hub
```

有关`huggingface_hub`安装的更多详细信息，请查看[this guide](https://huggingface.co/docs/huggingface_hub/installation)。

> [!提示]
> 在本指南中，我们将重点关注 Python 库。如果您已经用 JavaScript 实现了库，则可以使用 [⟦T13⟧](https://www.npmjs.com/package/@huggingface/hub) 来代替。其余逻辑（即托管文件、代码示例等）不依赖于代码语言。
>
> ```
> npm add @huggingface/hub
> ```

成功安装`huggingface_hub`库后，用户将需要进行身份验证。最简单的身份验证方法是将令牌保存在计算机上。用户可以使用 `login()` 命令从终端执行此操作：

```
hf auth login
```该命令告诉他们是否已经登录并提示他们输入令牌。然后验证令牌并将其保存在其`HF_HOME`目录中（默认为`~/.cache/huggingface/token`）。与集线器交互的任何脚本或库在发送请求时都将使用此令牌。

或者，用户可以在笔记本或脚本中使用 `login()` 以编程方式登录：

```py
from huggingface_hub import login
login()
```

从 Hub 上的公共存储库下载文件时，身份验证是可选的。

### 从集线器下载文件

集成允许用户从中心下载模型并直接从您的库实例化它。这通常可以通过提供特定于您的库的方法（通常称为 `from_pretrained` 或 `load_from_hf`）来实现。要从 Hub 实例化模型，您的库必须：
- 从集线器下载文件。这就是我们现在要讨论的内容。
- 从这些文件实例化 Python 模型。使用 [⟦T21⟧](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.hf_hub_download) 方法从 Hub 上的存储库下载文件。下载的文件存储在缓存中：`~/.cache/huggingface/hub`。用户下次使用文件时无需重新下载该文件，这为大文件节省了大量时间。此外，如果存储库更新了文件的新版本，`huggingface_hub`将自动下载最新版本并将其存储在缓存中。用户不必担心手动更新文件。

例如，从 [lysandre/arxiv-nlp](https://huggingface.co/lysandre/arxiv-nlp) 存储库下载 `config.json` 文件：

```python
>>> from huggingface_hub import hf_hub_download
>>> config_path = hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
>>> config_path
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'
```

`config_path` 现在包含下载文件的路径。您可以保证该文件存在并且是最新的。

如果您的图书馆需要下载整个存储库，请使用[⟦T26⟧](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download)。它将负责并行下载所有文件。返回值是包含下载文件的目录的路径。

```py
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'
```

有许多选项可用于从特定版本下载文件、过滤要下载的文件、提供自定义缓存目录、下载到本地目录等。查看 [download guide](https://huggingface.co/docs/huggingface_hub/en/guides/download) 了解更多详细信息。

### 将文件上传到集线器您可能还想提供一种方法，以便用户可以将自己的模型推送到 Hub。这使得社区能够构建与您的库兼容的模型生态系统。 `huggingface_hub`库提供了创建存储库和上传文件的方法：

- `create_repo` 在集线器上创建一个存储库。
- `upload_file` 和 `upload_folder` 将文件上传到 Hub 上的存储库。

`create_repo` 方法在 Hub 上创建一个存储库。使用 `repo_id` 参数为您的存储库提供名称：

```python
>>> from huggingface_hub import create_repo
>>> create_repo(repo_id="test-model")
'https://huggingface.co/lysandre/test-model'
```

当您检查 Hugging Face 帐户时，您现在应该在您的命名空间下看到一个 `test-model` 存储库。

[⟦T34⟧](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 方法将文件上传到集线器。该方法需要满足以下条件：

- 要上传的文件的路径。
- 存储库中的最终路径。
- 您希望将文件推送到的存储库。

例如：

```python
>>> from huggingface_hub import upload_file
>>> upload_file(
...    path_or_fileobj="/home/lysandre/dummy-test/README.md", 
...    path_in_repo="README.md", 
...    repo_id="lysandre/test-model"
... )
'https://huggingface.co/lysandre/test-model/blob/main/README.md'
```

如果您检查您的 Hugging Face 帐户，您应该会在存储库中看到该文件。

通常，库会将模型序列化到本地目录，然后立即将整个文件夹上传到 Hub。这可以使用 [⟦T35⟧](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 来完成：

```py
>>> from huggingface_hub import upload_folder
>>> upload_folder(
...     folder_path="/home/lysandre/dummy-test",
...     repo_id="lysandre/test-model",
... )
```

有关如何上传文件的更多详细信息，请查看[upload guide](https://huggingface.co/docs/huggingface_hub/en/guides/upload)。

## 模型卡模型卡是模型随附的文件，可提供方便的信息。在底层，模型卡是带有附加元数据的简单 Markdown 文件。模型卡对于可发现性、可重复性和共享至关重要！您可以在任何模型存储库中找到模型卡作为 README.md 文件。有关如何创建好的模型卡的更多详细信息，请参阅[model cards guide](./model-cards)。

如果您的库允许将模型推送到中心，建议生成一个最小的模型卡，其中包含预填充的元数据（通常为`library_name`、`pipeline_tag`或`tags`）以及有关模型如何训练的信息。这将有助于为使用您的库构建的所有模型提供标准化描述。

## 注册你的库干得好！您现在应该拥有一个能够从 Hub 加载模型并最终推送新模型的库。下一步是确保 Hub 上的模型有完整的文档记录并与平台集成。为此，可以在 Hub 上注册库，这为用户带来了一些好处：
- 模型页面上可以显示漂亮的标签（例如`KerasNLP`而不是`keras-nlp`）
- 每个模型页面中添加了指向您的库存储库和文档的链接
- 可以定义自定义下载计数规则
- 可以生成代码片段来展示如何使用您的库加载模型要注册新库，请按照以下说明打开拉取请求[here](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts)：
- 库 ID 应小写并用连字符分隔（例如：`"adapter-transformers"`）。打开 PR 时请确保保留字母顺序。
- 使用用户友好的外壳设置`repoName`和`prettyLabel`（例如：`DeepForest`）。
- 设置 `repoUrl` 并包含库源代码（通常是 GitHub 存储库）的链接。
- （可选）设置 `docsUrl` 并包含库文档的链接。如果文档位于上面引用的 GitHub 存储库中，则无需设置两次。
- 将 `filter` 设置为 `false`。
- （可选）通过设置 `countDownload` 定义如何计算下载量。可以通过文件扩展名或文件名跟踪下载。确保不要重复计数。例如，如果加载模型需要 3 个文件，则下载计数规则必须仅对 3 个文件中的 1 个文件进行下载计数。否则，下载次数将会被高估。
**注意：**如果库使用默认配置文件之一（`config.json`、`config.yaml`、`hyperparams.yaml`、`params.json`和`meta.yaml`，请参阅[here](https://huggingface.co/docs/hub/models-download-stats#which-are-the-query-files-for-different-libraries)），则无需手动定义下载计数规则。
-（可选）定义`snippets`，让用户知道如何快速实例化模型。更多详细信息如下。在打开 PR 之前，请确保 https://huggingface.co/models?other=my-library-name 上至少引用了一个模型。如果没有，则必须使用`library_name: my-library-name`更新相关模型的模型卡元数据（参见[example](https://huggingface.co/google/gemma-scope/blob/main/README.md?code=true#L3)）。如果您不是 Hub 上模型的所有者，请打开 PR（请参阅[example](https://huggingface.co/MCG-NJU/VFIMamba/discussions/1)）。

这是为 VFIMamba 添加集成的最小 [example](https://github.com/huggingface/huggingface.js/pull/885/files)。

### 代码片段

我们建议添加代码片段来解释如何在下游库中使用模型。 

要添加代码片段，您应该使用适用于您的模型的说明更新 [model-libraries-snippets.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts) 文件。例如，[Asteroid](https://huggingface.co/asteroid-team)集成包括一个简短的代码片段，用于说明如何加载和使用 Asteroid 模型：

```typescript
const asteroid = (model: ModelData) =>
`from asteroid.models import BaseModel
  
model = BaseModel.from_pretrained("${model.id}")`;
```

这样做还会向您的模型添加标签，以便用户可以快速识别您的库中的模型。

将您的代码片段添加到[model-libraries-snippets.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries-snippets.ts)后，您可以如上所述在[model-libraries.ts](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/model-libraries.ts)中引用它。

## 记录你的库

最后，您可以将您的库添加到中心的文档中。例如，检查将 [SetFit](./setfit) 添加到文档中的 [Setfit PR](https://github.com/huggingface/hub-docs/pull/1150)。

### 在拥抱面使用稳定基线3
https://huggingface.co/docs/hub/stable-baselines3.md