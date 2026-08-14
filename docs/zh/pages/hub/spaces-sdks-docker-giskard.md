<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 吉斯卡德谈空间

**Giskard** 是适用于法学硕士、表格和 NLP 模型的 AI 模型质量测试工具包。它由一个开源的Python组成 
library for scanning and testing AI models and an AI Model Quality Testing app, which can now be deployed using Hugging Face's 
Docker 空间。 Extending the features of the open-source library, the AI Model Quality Testing app enables you to:

- 调试测试来诊断您的问题

- Create domain-specific tests thanks to automatic model insights

- 比较模型来决定推广哪个模型

- Collect business feedback of your model results

- 与同事分享您的结果以进行调整

- Store all your QA objects (tests, data slices, evaluation criteria, etc.) in one place to work more efficiently

参观[Giskard's documentation](https://docs.giskard.ai/)和[Quickstart Guides](https://docs.giskard.ai/en/latest/getting_started/quickstart/index.html) 
to learn how to use the full range of tools provided by Giskard.

In the next sections, you'll learn to deploy your own Giskard AI Model Quality Testing app and use it right from 
拥抱脸部空间。 This Giskard app is a **self-contained application completely hosted on Spaces using Docker**.

## 在 Spaces 上部署 Giskard

You can deploy Giskard on Spaces with just a few clicks:> [!警告]
> 关于数据持久性的重要说明：
> You can use the Giskard Space as is for initial exploration and experimentation. **为了更长时间地使用 
> 小型项目，附上[Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)**。 This prevents data loss during Space restarts which 
> 每 24 小时发生一次。

您需要定义**所有者**（您的个人帐户或组织）、**空间名称**和**可见性**。 
If you don’t want to publicly share your models and quality tests, set your Space to **Private**.

Once you have created the Space, you'll see the `Building` status. Once it becomes `Running`, your Space is ready to go. 
If you don't see a change in the screen, refresh the page.

## 申请免费许可证

Giskard Space 启动并运行后，您需要申请免费许可证才能开始使用该应用程序。 
You will then automatically receive an email with the license file. 

## 创建一个新的 Giskard 项目

Once inside the app, start by creating a new project from the welcome screen.

## 生成 Hugging Face Giskard 空间令牌和 Giskard API 密钥Giskard API 密钥用于在运行 AI 模型的环境与 
the Giskard app on Hugging Face Spaces.如果您已将空间的**可见性**设置为**私人**，则需要提供拥抱的面孔 
用户访问令牌以生成 Hugging Face Giskard 空间令牌并建立访问您的私人空间的通信。 To do so, follow the instructions 
显示在 Giskard 应用程序的设置页面中。

## 启动 ML Worker

Giskard 使用直接在 Python 环境中运行模型的工作器来执行您的模型，其中包含所有 
dependencies required by your model. You can either execute the ML worker:

- 来自内核中的本地笔记本，其中包含模型的所有依赖项

- 来自内核中的 Google Colab，其中包含模型的所有依赖项

- 或者从包含模型所有依赖项的 Python 环境中的终端

只需在包含模型所有依赖项的 Python 环境中运行以下命令：

```bash
giskard worker start -d -k GISKARD-API-KEY -u https://XXX.hf.space --hf-token GISKARD-SPACE-TOKEN
```

## Upload your test suite, models and datasets为了开始为项目构建质量测试，您需要上传模型和数据集对象，然后创建或 
从 Giskard Python 库上传测试套件。 

> [!提示]
> 有关如何从 Giskard 的 Python 库的自动模型扫描工具创建测试套件的更多信息，请访问 head 
> 转到 Giskard 的 [Quickstart Guides](https://docs.giskard.ai/en/latest/getting_started/quickstart/index.html)。

这些操作都需要您的 Python 环境与 
吉斯卡德空间。通过初始化 Giskard 客户端来实现此目的：只需复制“创建 Giskard 客户端”片段即可 
从 Giskard 应用程序的设置页面并在您的 Python 环境中运行它。这看起来像这样：

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

### 空间中的 Cookie 限制
https://huggingface.co/docs/hub/spaces-cookie-limitations.md