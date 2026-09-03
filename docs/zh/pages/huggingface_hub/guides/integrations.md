<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 将任何 ML 框架与 Hub 集成

Hugging Face Hub 让社区托管和共享模型变得简单。它支持
开源生态系统中的[dozens of libraries](https://huggingface.co/docs/hub/models-libraries)。我们总是
致力于扩大这种支持，以推动协作机器学习向前发展。 `huggingface_hub` 库起着
在此过程中发挥着关键作用，允许任何Python脚本轻松推送和加载文件。

将库与 Hub 集成有四种主要方法：
1. **推送到Hub：**实现将模型上传到Hub的方法。这包括模型权重，以及
   [the model card](https://huggingface.co/docs/huggingface_hub/how-to-model-cards) 以及任何其他相关信息
   或运行模型所需的数据（例如训练日志）。这种方法通常被称为`push_to_hub()`。
2. **从Hub下载：** 实现从Hub加载模型的方法。该方法应该下载模型
   配置/权重并加载模型。这种方法通常被称为`from_pretrained`或`load_from_hub()`。
3. **小部件：** 在 Hub 上模型的登陆页面上显示小部件。它允许用户快速尝试模型
   从浏览器。在本指南中，我们将重点关注前两个主题。我们将介绍您可以用来集成的两种主要方法
图书馆，各有优点和缺点。指南末尾总结了所有内容，以帮助您选择
两者之间。请记住，这些只是指导方针，您可以根据自己的要求自由调整。

如果您对 Inference 和 Widgets 感兴趣，可以关注[this guide](https://huggingface.co/docs/hub/models-adding-libraries#set-up-the-inference-api)。
在这两种情况下，如果您正在将图书馆与中心集成并希望被列出，您可以联系我们
[in our docs](https://huggingface.co/docs/hub/models-libraries)。

## 灵活的方法：助手

将库集成到 Hub 的第一种方法是实际实现 `push_to_hub` 和 `from_pretrained`
自己的方法。这使您可以完全灵活地选择需要上传/下载的文件以及如何处理输入
具体到您的框架。您可以参考[upload files](./upload)和[download files](./download)两个指南
详细了解如何做到这一点。例如，这就是 FastAI 集成的实现方式（参见 [push_to_hub_fastai()](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.push_to_hub_fastai)
和[from_pretrained_fastai()](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.from_pretrained_fastai)）。

库之间的实现可能有所不同，但工作流程通常相似。

### from_pretrained

`from_pretrained` 方法通常如下所示：

```python
def from_pretrained(model_id: str) -> MyModelClass:
   # Download model from Hub
   cached_model = hf_hub_download(
      repo_id=repo_id,
      filename="model.pkl",
      library_name="fastai",
      library_version=get_fastai_version(),
   )

   # Load model
    return load_model(cached_model)
```

### 推送到集线器`push_to_hub` 方法通常需要更多的复杂性来处理存储库创建、生成模型卡和保存权重。
一种常见的方法是将所有这些文件保存在临时文件夹中，上传然后删除。

```python
def push_to_hub(model: MyModelClass, repo_name: str) -> None:
   api = HfApi()

   # Create repo if not existing yet and get the associated repo_id
   repo_id = api.create_repo(repo_name, exist_ok=True)

   # Save all files in a temporary directory and push them in a single commit
   with TemporaryDirectory() as tmpdir:
      tmpdir = Path(tmpdir)

      # Save weights
      save_model(model, tmpdir / "model.safetensors")

      # Generate model card
      card = generate_model_card(model)
      (tmpdir / "README.md").write_text(card)

      # Save logs
      # Save figures
      # Save evaluation metrics
      # ...

      # Push to hub
      return api.upload_folder(repo_id=repo_id, folder_path=tmpdir)
```

这当然只是一个例子。如果您对更复杂的操作感兴趣（删除远程文件、上传
动态权重、本地持久权重等）请参阅[upload files](./upload)指南。

### 限制

虽然很灵活，但这种方法也有一些缺点，特别是在维护方面。拥抱脸用户经常
习惯于使用 `huggingface_hub` 时的附加功能。例如，当从 Hub 加载文件时，
通常提供如下参数：
- `token`：从私人仓库下载
- `revision`：从特定分支下载
- `cache_dir`：将文件缓存到特定目录中
- `force_download`/`local_files_only`：是否重用缓存
- `proxies`：配置HTTP会话推送模型时，支持类似的参数：
- `commit_message`：自定义提交消息
- `private`：如果缺少，则创建一个私有仓库
- `create_pr`：创建 PR 而不是推送到 `main`
- `branch`：推送到分支而不是`main`分支
- `allow_patterns`/`ignore_patterns`：过滤要上传的文件
- `token`
- ...

所有这些参数都可以添加到我们上面看到的实现中并传递给 `huggingface_hub` 方法。
但是，如果参数更改或添加新功能，您将需要更新您的软件包。支持那些
参数还意味着您需要维护更多文档。要了解如何减轻这些限制，让我们跳转
到我们的下一节**类继承**。

## 更复杂的方法：类继承

正如我们在上面看到的，有两种主要方法可以包含在您的库中以将其与 Hub 集成： 上传文件
(`push_to_hub`) 并下载文件(`from_pretrained`)。您可以自己实现这些方法，但它附带
注意事项。为了解决这个问题，`huggingface_hub`提供了一个使用类继承的工具。让我们看看它是如何工作的！在很多情况下，库已经使用 Python 类实现了其模型。该类包含以下属性
模型以及加载、运行、训练和评估它的方法。我们的方法是扩展此类以包括上传和
使用 mixins 下载功能。 [Mixin](https://stackoverflow.com/a/547714) 是一个旨在扩展
具有一组使用多重继承的特定功能的现有类。 `huggingface_hub`提供了自己的mixin，
[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)。这里的关键是理解它的行为以及如何定制它。

[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 类实现 3 个 *public* 方法（`push_to_hub`、`save_pretrained` 和 `from_pretrained`）。那些
是您的用户将调用以使用您的库加载/保存模型的方法。 [ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)也定义了2
*私有*方法（`_save_pretrained`和`_from_pretrained`）。这些是您必须实施的。所以要整合
您的图书馆，您应该：1. 让你的Model类继承[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)。
2. 实现私有方法：
    - [_save_pretrained()](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin._save_pretrained)：方法将目录路径作为输入并将模型保存到其中。
    您必须在此方法中编写转储模型的所有逻辑：模型卡、模型权重、配置文件、
    培训日志和数据。该模型的任何相关信息都必须通过该方法处理。
    [Model Cards](https://huggingface.co/docs/hub/model-cards) 对于描述您的模型特别重要。检查
    出[our implementation guide](./model-cards)了解更多详情。
    - [_from_pretrained()](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin._from_pretrained)：**类方法** 将 `model_id` 作为输入并返回实例化
    模型。该方法必须下载相关文件并加载它们。
3. 你完成了！

使用 [ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 的优点是，一旦你处理好文件的序列化/加载，你就可以开始了。您无需担心存储库创建、提交、PR 或修订等问题。 [ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 还确保公共方法被记录并带有类型注释，并且您将能够在 Hub 上查看模型的下载计数。所有这些都由 [ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 处理并可供您的用户使用。

### 一个具体的例子：PyTorch我们上面看到的一个很好的例子是[PyTorchModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)，我们对 PyTorch 框架的集成。这是一个即用型集成。

#### 如何使用它？

以下是任何用户如何从 Hub 加载 PyTorch 模型或将 PyTorch 模型保存到 Hub 的方法：

```python
>>> import torch
>>> import torch.nn as nn
>>> from huggingface_hub import PyTorchModelHubMixin

# Define your Pytorch model exactly the same way you are used to
>>> class MyModel(
...         nn.Module,
...         PyTorchModelHubMixin, # multiple inheritance
...         library_name="keras-nlp",
...         tags=["keras"],
...         repo_url="https://github.com/keras-team/keras-nlp",
...         docs_url="https://keras.io/keras_nlp/",
...         # ^ optional metadata to generate model card
...     ):
...     def __init__(self, hidden_size: int = 512, vocab_size: int = 30000, output_size: int = 4):
...         super().__init__()
...         self.param = nn.Parameter(torch.rand(hidden_size, vocab_size))
...         self.linear = nn.Linear(output_size, vocab_size)

...     def forward(self, x):
...         return self.linear(x + self.param)

# 1. Create model
>>> model = MyModel(hidden_size=128)

# Config is automatically created based on input + default values
>>> model.param.shape[0]
128

# 2. (optional) Save model to local directory
>>> model.save_pretrained("path/to/my-awesome-model")

# 3. Push model weights to the Hub
>>> model.push_to_hub("my-awesome-model")

# 4. Initialize model from the Hub => config has been preserved
>>> model = MyModel.from_pretrained("username/my-awesome-model")
>>> model.param.shape[0]
128

# Model card has been correctly populated
>>> from huggingface_hub import ModelCard
>>> card = ModelCard.load("username/my-awesome-model")
>>> card.data.tags
["keras", "pytorch_model_hub_mixin", "model_hub_mixin"]
>>> card.data.library_name
"keras-nlp"
```

#### 实施

实现实际上非常简单，完整的实现可以找到[here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/hub_mixin.py)。

1. 首先，从`ModelHubMixin`继承你的类：

```python
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)
```

2. 实现`_save_pretrained`方法：

```py
from huggingface_hub import ModelHubMixin

class PyTorchModelHubMixin(ModelHubMixin):
   (...)

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        save_model_as_safetensor(self.module, str(save_directory / SAFETENSORS_SINGLE_FILE))

```

3. 实现`_from_pretrained`方法：

```python
class PyTorchModelHubMixin(ModelHubMixin):
   (...)

   @classmethod # Must be a classmethod!
   def _from_pretrained(
      cls,
      *,
      model_id: str,
      revision: str,
      cache_dir: str,
      force_download: bool,
      local_files_only: bool,
      token: Union[str, bool, None],
      map_location: str = "cpu", # additional argument
      strict: bool = False, # additional argument
      **model_kwargs,
   ):
      """Load Pytorch pretrained weights and return the loaded model."""
        model = cls(**model_kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            return cls._load_as_safetensor(model, model_file, map_location, strict)

         model_file = hf_hub_download(
            repo_id=model_id,
            filename=SAFETENSORS_SINGLE_FILE,
            revision=revision,
            cache_dir=cache_dir,
            force_download=force_download,
            token=token,
            local_files_only=local_files_only,
            )
         return cls._load_as_safetensor(model, model_file, map_location, strict)
```

就是这样！您的库现在允许用户在 Hub 上上传和下载文件。

### 高级用法

在上面的部分中，我们快速讨论了 [ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 的工作原理。在本节中，我们将看到它的一些更高级的功能，以改善您的库与 Hugging Face Hub 的集成。

#### 模型卡

[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 为您生成模型卡。模型卡是模型附带的文件，提供有关模型的重要信息。在底层，模型卡是带有附加元数据的简单 Markdown 文件。模型卡对于可发现性、可重复性和共享至关重要！查看[Model Cards guide](https://huggingface.co/docs/hub/model-cards)了解更多详情。半自动生成模型卡是确保您的库推送的所有模型共享通用元数据的好方法：`library_name`、`tags`、`license`、`pipeline_tag` 等。这使得您的库支持的所有模型都可以在 Hub 上轻松搜索，并为登录 Hub 的用户提供一些资源链接。继承[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)时可以直接定义元数据：

```py
class UniDepthV1(
   nn.Module,
   PyTorchModelHubMixin,
   library_name="unidepth",
   repo_url="https://github.com/lpiccinelli-eth/UniDepth",
   docs_url=...,
   pipeline_tag="depth-estimation",
   license="cc-by-nc-4.0",
   tags=["monocular-metric-depth-estimation", "arxiv:1234.56789"]
):
   ...
```

默认情况下，将使用您提供的信息生成通用模型卡（例如：[pyp1/VoiceCraft_giga830M](https://huggingface.co/pyp1/VoiceCraft_giga830M)）。但您也可以定义自己的模型卡模板！

在此示例中，使用 `VoiceCraft` 类推送的所有模型将自动包含引用部分和许可证详细信息。有关如何定义模型卡模板的更多详细信息，请查看[Model Cards guide](./model-cards)。

```py
MODEL_CARD_TEMPLATE = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This is a VoiceCraft model. For more details, please check out the official Github repo: https://github.com/jasonppy/VoiceCraft. This model is shared under a Attribution-NonCommercial-ShareAlike 4.0 International license.

## Citation

@article{peng2024voicecraft,
  author    = {Peng, Puyuan and Huang, Po-Yao and Li, Daniel and Mohamed, Abdelrahman and Harwath, David},
  title     = {VoiceCraft: Zero-Shot Speech Editing and Text-to-Speech in the Wild},
  journal   = {arXiv},
  year      = {2024},
}
"""

class VoiceCraft(
   nn.Module,
   PyTorchModelHubMixin,
   library_name="voicecraft",
   model_card_template=MODEL_CARD_TEMPLATE,
   ...
):
   ...
```

最后，如果你想用动态值扩展模型卡生成过程，你可以重写 `generate_model_card()` 方法：

```py
from huggingface_hub import ModelCard, PyTorchModelHubMixin

class UniDepthV1(nn.Module, PyTorchModelHubMixin, ...):
   (...)

   def generate_model_card(self, *args, **kwargs) -> ModelCard:
      card = super().generate_model_card(*args, **kwargs)
      card.data.metrics = ...  # add metrics to the metadata
      card.text += ... # append section to the modelcard
      return card
```

#### 配置[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin) 为您处理模型配置。当您实例化模型并将其序列化到 `config.json` 文件中时，它会自动检查输入值。这有两个好处：
1. 用户将能够使用与您完全相同的参数重新加载模型。
2. 拥有 `config.json` 文件会自动启用集线器上的分析（即“下载”计数）。

但它在实践中是如何运作的呢？从用户的角度来看，有几条规则可以使该过程尽可能顺利：
- 如果您的 `__init__` 方法需要 `config` 输入，它将自动在存储库中保存为 `config.json`。
- 如果 `config` 输入参数用数据类类型注释（例如 `config: Optional[MyConfigClass] = None`），则 `config` 值将为您正确反序列化。
- 初始化时传递的所有值也将存储在配置文件中。这意味着您不一定需要期望 `config` 输入才能从中受益。

示例：

```py
class MyModel(ModelHubMixin):
   def __init__(value: str, size: int = 3):
      self.value = value
      self.size = size

   (...) # implement _save_pretrained / _from_pretrained

model = MyModel(value="my_value")
model.save_pretrained(...)

# config.json contains passed and default values
{"value": "my_value", "size": 3}
```但是如果一个值无法序列化为 JSON 该怎么办？默认情况下，保存配置文件时将忽略该值。但是，在某些情况下，您的库已经期望自定义对象作为无法序列化的输入，并且您不希望更新内部逻辑来更新其类型。不用担心！当继承[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)时，您可以传递任何类型的自定义编码器/解码器。这需要更多的工作，但可以确保在将库与 Hub 集成时，内部逻辑不会受到影响。

这是一个具体示例，其中类需要 `argparse.Namespace` 配置作为输入：

```py
class VoiceCraft(nn.Module):
    def __init__(self, args):
      self.pattern = self.args.pattern
      self.hidden_size = self.args.hidden_size
      ...
```

一种解决方案是将 `__init__` 签名更新为 `def __init__(self, pattern: str, hidden_size: int)` 并更新实例化类的所有片段。这是一种完全有效的修复方法，但它可能会破坏使用您的库的下游应用程序。

另一个解决方案是提供一个简单的编码器/解码器来将`argparse.Namespace`转换为字典。

```py
from argparse import Namespace

class VoiceCraft(
   nn.Module,
   PyTorchModelHubMixin,  # inherit from mixin
   coders={
      Namespace : (
         lambda x: vars(x),  # Encoder: how to convert a `Namespace` to a valid jsonable value?
         lambda data: Namespace(**data),  # Decoder: how to reconstruct a `Namespace` from a dictionary?
      )
   }
):
    def __init__(self, args: Namespace): # annotate `args`
      self.pattern = self.args.pattern
      self.hidden_size = self.args.hidden_size
      ...
```在上面的代码片段中，类的内部逻辑和`__init__`签名都没有改变。这意味着您的库的所有现有代码片段将继续工作。为了实现这一目标，我们必须：
1. 继承自mixin（本例中为`PytorchModelHubMixin`）。
2. 在继承中传递一个`coders`参数。这是一本字典，其中键是您要处理的自定义类型。值是一个元组`(encoder, decoder)`。
   - 编码器期望指定类型的对象作为输入并返回 jsonable 值。这将在使用 `save_pretrained` 保存模型时使用。
   - 解码器期望原始数据（通常是字典）作为输入并重建初始对象。这将在使用 `from_pretrained` 加载模型时使用。
3. 在`__init__`签名中添加类型注释。这对于让 mixin 知道类需要哪种类型以及使用哪个解码器非常重要。

为了简单起见，上例中的编码器/解码器功能并不鲁棒。对于具体的实现，您很可能必须正确处理极端情况。

## 快速比较让我们快速总结一下我们看到的两种方法及其优点和缺点。下表仅供参考。
您的框架可能有一些需要解决的特殊性。本指南仅提供指导和
关于如何处理集成的想法。无论如何，如果您有任何疑问，请随时与我们联系！

|           整合|                                                      使用助手|                                     使用[ModelHubMixin](/docs/huggingface_hub/v1.30.0/en/package_reference/mixins#huggingface_hub.ModelHubMixin)|
| :--------------------------: | :--------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------: |
|         用户体验 |                                `model = load_from_hub(...)``push_to_hub(model, ...)` |               `model = MyModel.from_pretrained(...)``model.push_to_hub(...)` ||           灵活性 |                                 非常灵活。您完全控制实施。                                  |                    灵活性较差。您的框架必须有一个模型类。                    |
|           保养|更多维护以添加对配置和新功能的支持。可能还需要修复用户报告的问题。 |维护较少，因为与 Hub 的大部分交互都是在 `huggingface_hub` 中实现的。 |
|文档/类型注释 |                                                 需要手动编写。                                                  |                             部分由`huggingface_hub`处理。                             |
|        下载计数器|                                                 需手动处理。                                                  |                      如果类具有 `config` 属性，则默认启用。                      ||           型号卡|                                                  需要手动处理 |                       默认生成带有library_name、标签等。

### 管理您的空间
https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/manage-spaces.md