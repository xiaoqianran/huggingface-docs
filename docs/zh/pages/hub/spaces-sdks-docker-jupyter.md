<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间上的 JupyterLab

[JupyterLab](https://jupyter.org/) 是一个基于 Web 的 Jupyter 笔记本、代码和数据的交互式开发环境。它是数据科学和机器学习的一个很好的工具，被社区广泛使用。通过 Hugging Face Spaces，您可以部署自己的 JupyterLab 实例，并直接从 Hugging Face 网站使用它进行开发。

## ⚡️ 在 Spaces 上部署 JupyterLab 实例

只需点击几下，您就可以在 Spaces 上部署 JupyterLab。首先，前往[this link](https://huggingface.co/new-space?template=SpacesExamples/jupyterlab)或点击以下按钮：

  

空间要求您定义：

* **所有者**：您的个人帐户或您所属的组织
  的一部分。 

* A **空间名称**：帐户内空间的名称
  你正在创造空间。

* **可见性**：_私人_如果您想要
  仅您或您的组织可见的空间，如果您愿意，也可以_公开_
  它对其他用户可见。 

* **硬件**：您想要用于 JupyterLab 实例的硬件。从 CPU 到 H100。* 您可以选择配置 `JUPYTER_TOKEN` 密码来保护您的 JupyterLab 工作区。未指定时，默认为`huggingface`。如果您的空间是公共的或者空间位于组织中，我们强烈建议您进行此设置。

Hugging Face Spaces 中的存储是短暂的，默认配置中存储的数据可能会在空间重新启动或重置时丢失。我们建议将您的工作保存到远程位置或将[Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)附加到您的空间以获取持久数据。

## 阅读更多

- [HF Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)

如果您有任何反馈或更改请求，请随时通过[Feedback Discussion](https://huggingface.co/spaces/SpacesExamples/jupyterlab/discussions/3)联系所有者。

## 致谢

该模板由[camenduru](https://twitter.com/camenduru)和[nateraw](https://huggingface.co/nateraw)创建，并得到[osanseviero](https://huggingface.co/osanseviero)和[azzr](https://huggingface.co/azzr)的贡献。

###达斯克
https://huggingface.co/docs/hub/datasets-dask.md