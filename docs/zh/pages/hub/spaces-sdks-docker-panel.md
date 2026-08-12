<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间小组

[Panel](https://panel.holoviz.org/) 是一个开源 Python 库，可让您完全使用 Python 轻松构建强大的工具、仪表板和复杂的应用程序。它具有包含电池的理念，使 PyData 生态系统、强大的数据表等触手可及。高级反应式 API 和基于低级回调的 API 确保您可以快速构建探索性应用程序，但如果您构建具有丰富交互性的复杂、多页面应用程序，则不受限制。 Panel 是 [HoloViz](https://holoviz.org/) 生态系统的成员，是您进入数据探索工具互联生态系统的门户。

访问 [Panel documentation](https://panel.holoviz.org/) 了解有关制作强大应用程序的更多信息。 

## 🚀 在空间上部署面板

您只需点击几下即可在 Spaces 上部署面板：

  

您需要定义几个关键参数：所有者（您的个人帐户或组织）、空间名称和可见性。如果您打算执行计算密集型深度学习模型，请考虑升级到 GPU 以提高性能。 

创建空间后，它将以“正在构建”状态开始，一旦您的空间准备就绪，该状态将更改为“正在运行”。 

## ⚡️ 你会看到什么？当您的空间构建并准备就绪时，您将看到此图像分类面板应用程序，它可以让您获取随机图像并在其上运行 OpenAI CLIP 分类器模型。查看我们的 [blog post](https://blog.holoviz.org/building_an_interactive_ml_dashboard_in_panel.html) 了解此应用程序的演练。 

## 🛠️ 如何定制和制作自己的应用程序？

空间模板将填充一些文件来启动您的应用程序： 

三个文件很重要：

### 1.app.py

该文件定义您的面板应用程序代码。您可以从修改现有应用程序开始，或者完全替换它来构建您自己的应用程序。要了解有关编写自己的面板应用程序的更多信息，请参阅[Panel documentation](https://panel.holoviz.org/)。

### 2.Dockerfile

Dockerfile 包含 Docker 将执行的一系列命令，以构建和启动图像作为面板应用程序将在其中运行的容器。通常，为了提供面板应用程序，我们使用命令 `panel serve app.py`。在这个特定的文件中，我们将命令划分为字符串列表。此外，我们必须定义地址和端口，因为 Hugging Face 期望在端口 7860 上为您的应用程序提供服务。此外，我们需要指定 `allow-websocket-origin` 标志以启用与服务器 Websocket 的连接。 

### 3.需求.txt该文件定义了我们的面板应用程序所需的包。使用Space时，将自动安装requirements.txt文件中列出的依赖项。您可以通过删除不必要的包或添加应用程序所需的其他包来自由修改此文件。请随意进行必要的更改，以确保您的应用程序安装了适当的软件包。

## 🌐 加入我们的社区
小组社区充满活力和支持，经验丰富的开发人员和数据科学家渴望提供帮助和分享他们的知识。加入我们并与我们联系：

- [Discord](https://discord.gg/aRFhC3Dz9w)
- [Discourse](https://discourse.holoviz.org/)
- [Twitter](https://twitter.com/Panel_Org)
- [LinkedIn](https://www.linkedin.com/company/panel-org)
- [Github](https://github.com/holoviz/panel)

### 空间上的磁盘使用情况
https://huggingface.co/docs/hub/spaces-storage.md