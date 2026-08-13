<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 吉斯卡德谈空间

**Giskard** 是适用于法学硕士、表格和 NLP 模型的 AI 模型质量测试工具包。它由一个开源的Python组成 
用于扫描和测试 AI 模型的库以及 AI 模型质量测试应用程序，现在可以使用 Hugging Face 进行部署 
Docker 空间。 AI 模型质量测试应用程序扩展了开源库的功能，使您能够：

- 调试测试来诊断您的问题

- 通过自动模型洞察创建特定领域的测试

- 比较模型来决定推广哪个模型

- 收集模型结果的业务反馈

- 与同事分享您的结果以进行调整

- 将所有 QA 对象（测试、数据切片、评估标准等）存储在一个地方，以提高工作效率

参观[Giskard's documentation](https://docs.giskard.ai/)和[Quickstart Guides](https://docs.giskard.ai/en/latest/getting_started/quickstart/index.html) 
了解如何使用 Giskard 提供的全套工具。

在接下来的部分中，您将学习部署自己的 Giskard AI 模型质量测试应用程序并直接使用它 
拥抱脸部空间。这个 Giskard 应用程序是一个**独立的应用程序，完全使用 Docker 托管在 Spaces 上**。

## 在 Spaces 上部署 Giskard

只需点击几下，您就可以在 Spaces 上部署 Giskard：> [!警告]
> 关于数据持久性的重要说明：
> 您可以按原样使用 Giskard Space 进行初步探索和实验。 **为了更长时间地使用 
> 小型项目，附上[Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)**。这可以防止 Space 重新启动期间数据丢失 
> 每 24 小时发生一次。

您需要定义**所有者**（您的个人帐户或组织）、**空间名称**和**可见性**。 
如果您不想公开分享您的模型和质量测试，请将您的空间设置为 **私人**。

创建空间后，您将看到`Building`状态。一旦变成`Running`，您的空间就准备好了。 
如果您在屏幕上没有看到变化，请刷新页面。

## 申请免费许可证

Giskard Space 启动并运行后，您需要申请免费许可证才能开始使用该应用程序。 
然后您将自动收到一封包含许可证文件的电子邮件。 

## 创建一个新的 Giskard 项目

进入应用程序后，首先从欢迎屏幕创建一个新项目。

## 生成 Hugging Face Giskard 空间令牌和 Giskard API 密钥Giskard API 密钥用于在运行 AI 模型的环境与 
Hugging Face Spaces 上的 Giskard 应用程序。如果您已将空间的**可见性**设置为**私人**，则需要提供拥抱的面孔 
用户访问令牌来生成 Hugging Face Giskard 空间令牌并建立访问您的私人空间的通信。为此，请按照说明进行操作 
显示在 Giskard 应用程序的设置页面中。

## 启动 ML Worker

Giskard 使用直接在 Python 环境中运行模型的工作器来执行您的模型，其中包含所有 
您的模型所需的依赖项。您可以执行 ML Worker：

- 来自内核中的本地笔记本，其中包含模型的所有依赖项

- 来自内核中的 Google Colab，其中包含模型的所有依赖项

- 或者从包含模型所有依赖项的 Python 环境中的终端

只需在包含模型所有依赖项的 Python 环境中运行以下命令：

```bash
giskard worker start -d -k GISKARD-API-KEY -u https://XXX.hf.space --hf-token GISKARD-SPACE-TOKEN
```

## 上传您的测试套件、模型和数据集为了开始为项目构建质量测试，您需要上传模型和数据集对象，然后创建或 
从 Giskard Python 库上传测试套件。 

> [!提示]
> 有关如何从 Giskard 的 Python 库的自动模型扫描工具创建测试套件的更多信息，请访问 head 
> 转到 Giskard 的 [Quickstart Guides](https://docs.giskard.ai/en/latest/getting_started/quickstart/index.html)。

这些操作都需要您的 Python 环境与 
吉斯卡德空间。通过初始化 Giskard 客户端来实现此目的：只需复制“创建 Giskard 客户端”片段即可 
从 Giskard 应用程序的设置页面中，并在您的 Python 环境中运行它。这看起来像这样：

```python
from giskard import GiskardClient

url = "https://user_name-space_name.hf.space"
api_key = "gsk-xxx"
hf_token = "xxx"

# Create a giskard client to communicate with Giskard
client = GiskardClient(url, api_key, hf_token)
```

如果您遇到问题，请前往 Giskard 的 [upload object documentation page](https://docs.giskard.ai/en/latest/giskard_hub/upload/index.html)。

## 反馈和支持

如果您有建议或需要具体支持，请加入[Giskard's Discord community](https://discord.com/invite/ABvfpbu69R)或联系[Giskard's GitHub repository](https://github.com/Giskard-AI/giskard)。

### 使用 Hugging Face 中的 Unity Sentis 模型
https://huggingface.co/docs/hub/unity-sentis.md