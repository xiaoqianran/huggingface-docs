<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 上传模型

要将模型上传到 Hub，您需要在 [Hugging Face](https://huggingface.co/join) 创建一个帐户。 Hub 上的模型是 [Git-based repositories](./repositories)，它为您提供版本控制、分支、可发现性和共享功能、与数十个库的集成等等！您可以控制要上传到存储库的内容，其中可能包括检查点、配置和任何其他文件。

您可以将存储库与个人用户（例如 [osanseviero/fashion_brands_patterns](https://huggingface.co/osanseviero/fashion_brands_patterns)）或组织（例如 [facebook/bart-large-xsum](https://huggingface.co/facebook/bart-large-xsum)）链接。组织可以收集与公司、社区或图书馆相关的模型！如果您选择一个组织，该模型将显示在该组织的页面上，并且该组织的每个成员都将能够为存储库做出贡献。您可以创建一个新组织[here](https://huggingface.co/organizations/new)。

> **_注意：_** 模型不需要与 Transformers/Diffusers 库兼容即可获取下载指标。支持任何自定义模型。阅读下文了解更多内容！

有多种方法可以上传模型，以便将它们很好地集成到 Hub 中并获得 [download metrics](models-download-stats)，如下所述。- 如果您的模型是为具有[built-in support](#upload-from-a-library-with-built-in-support)的库设计的，您可以使用该库提供的方法。使用 `trust_remote_code=True` 的自定义模型也可以利用这些方法。
- 如果您的模型是自定义 PyTorch 模型，则可以利用 [⟦T3⟧ class](#upload-a-pytorch-model-using-huggingfacehub)，因为它允许将 `from_pretrained`、`push_to_hub` 添加到任何 `nn.Module` 类，就像 Transformers、Diffusers 和 Timm 库中的模型一样。
- 除了程序化上传之外，您始终可以使用[web interface](#using-the-web-interface)或[the git command line](#using-git)。

上传模型后，我们建议将 [Model Card](./model-cards) 添加到您的存储库中以记录您的模型并使其更容易被发现。

<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/example_repository.png"
alt="drawing" width="600"/>

 利用 [PyTorchModelHubMixin](#upload-a-pytorch-model-using-huggingfacehub) 的示例 [repository](https://huggingface.co/LiheYoung/depth_anything_vitl14)。下载内容显示在右侧。

## 使用网络界面

要创建全新的模型存储库，请访问[huggingface.co/new](http://huggingface.co/new)。然后按照下列步骤操作：

1. 在“文件和版本”选项卡中，选择“添加文件”并指定“上传文件”：

2. 从那里，从您的计算机中选择要上传的文件，并留下有用的提交消息以了解您要上传的内容：

3. 然后，单击“**提交更改**”将您的模型上传到中心！

4.检查文件和历史记录

您可以检查您的存储库以及所有最近添加的文件！UI 允许您探索模型文件和提交，并查看每个提交引入的差异：

5. 添加元数据

您可以将元数据添加到模型卡中。您可以指定：
* 该模型适用的任务类型，启用小部件和推理提供程序。
* 使用的库（`transformers`、`spaCy`等）
* 语言
* 数据集
* 指标
* 许可证
* 还有更多！

了解有关型号标签 [here](./model-cards#model-card-metadata) 的更多信息。

6.添加TensorBoard痕迹

任何包含 TensorBoard 跟踪（包含 `tfevents` 的文件名）的存储库都被分类为 [⟦T10⟧ tag](https://huggingface.co/models?filter=tensorboard)。作为惯例，我们建议您将跟踪保存在 `runs/` 子文件夹下。然后，“训练指标”选项卡可以轻松查看记录变量的图表，例如损失或准确性。

如果安装了 [⟦T12⟧](https://pypi.org/project/tensorboard/)，使用 🤗 Transformers 训练的模型将默认生成 [TensorBoard traces](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.TensorBoardCallback)。

## 从具有内置支持的库上传

首先检查您的模型是否来自具有内置支持从 Hub 推送/加载的库，例如 Transformers、Diffusers、Timm、Asteroid 等：https://huggingface.co/docs/hub/models-libraries。下面我们将展示这对于像 Transformers 这样的库来说是多么容易：

```python
from transformers import BertConfig, BertModel

config = BertConfig()
model = BertModel(config)

model.push_to_hub("nielsr/my-awesome-bert-model")

# reload
model = BertModel.from_pretrained("nielsr/my-awesome-bert-model")
```有些库，例如 Transformers，支持加载 [code from the Hub](https://huggingface.co/docs/transformers/custom_models)。这是一种使用 `trust_remote_code=True` 标志使您的模型与 Transformer 配合使用的方法。您可能需要考虑此选项而不是成熟的库集成。

## 使用 Huggingface_hub 上传 PyTorch 模型

如果您的模型是（自定义）PyTorch 模型，您可以利用 [huggingface_hub](https://github.com/huggingface/huggingface_hub) Python 库中提供的 `PyTorchModelHubMixin` [class](https://huggingface.co/docs/huggingface_hub/package_reference/mixins#huggingface_hub.PyTorchModelHubMixin)。它是一个最小的类，为任何 `nn.Module` 添加了 `from_pretrained` 和 `push_to_hub` 功能以及下载指标。

以下是如何使用它（假设你已经运行了`pip install huggingface_hub`）：

```python
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

class MyModel(
    nn.Module,
    PyTorchModelHubMixin, 
    # optionally, you can add metadata which gets pushed to the model card
    repo_url="your-repo-url",
    pipeline_tag="text-to-image",
    license="mit",
):
    def __init__(self, num_channels: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.param = nn.Parameter(torch.rand(num_channels, hidden_size))
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        return self.linear(x + self.param)

# create model
config = {"num_channels": 3, "hidden_size": 32, "num_classes": 10}
model = MyModel(**config)

# save locally
model.save_pretrained("my-awesome-model")

# push to the hub
model.push_to_hub("your-hf-username/my-awesome-model")

# reload
model = MyModel.from_pretrained("your-hf-username/my-awesome-model")
```

如您所见，唯一的要求是您的模型继承自`PyTorchModelHubMixin`。所有实例属性将自动序列化为`config.json`文件。请注意，`init` 方法只能采用 JSON 可序列化的参数。支持 Python 数据类。

它带有自动下载指标，这意味着您将能够查看模型的下载次数，就像它们可用于 Transformers、Diffusers 或 Timm 库中本地集成的模型一样。使用此 mixin 类，每个单独的检查点都存储在 Hub 上由 2 个文件组成的单个存储库中：- 包含权重的`pytorch_model.bin`或`model.safetensors`文件
- `config.json` 文件，它是模型配置的序列化版本。该类用于计算下载指标：每次用户调用 `from_pretrained` 加载 `config.json` 时，计数就会加一。有关自动下载指标，请参阅[this guide](https://huggingface.co/docs/hub/models-download-stats)。

建议在每个检查点添加一张模型卡，以便人们可以阅读模型的内容、获得论文的链接等。

访问[the huggingface_hub's documentation](https://huggingface.co/docs/huggingface_hub/guides/integrations)了解更多信息。

或者，也可以简单地以编程方式将文件或文件夹上传到中心：https://huggingface.co/docs/huggingface_hub/guides/upload。

## 使用 Git

最后，由于模型存储库只是 Git 存储库，因此您还可以使用 Git 将模型文件推送到 Hub。按照 [Getting Started with Repositories](repositories-getting-started#terminal) 上的指南了解如何使用 `git` CLI 提交和推送模型。

### 空间作为 API 端点
https://huggingface.co/docs/hub/spaces-api-endpoints.md