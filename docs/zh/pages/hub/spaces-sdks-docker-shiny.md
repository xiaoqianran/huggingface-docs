<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间闪亮

[Shiny](https://shiny.posit.co/)是一个开源框架，用于构建简单、美观且高性能的数据应用程序。 
开发 Shiny 的目标是构建足够简单的东西，可以在一个下午教别人，但又足够可扩展，可以为大型关键任务应用程序提供支持。 
您可以在几分钟内创建一个有用的 Shiny 应用程序，但如果您的项目范围扩大，您可以确定 Shiny 可以容纳该应用程序。

Shiny 与其他框架的主要区别在于其反应式执行模型。 
当您编写 Shiny 应用程序时，框架会推断输入、输出和中间计算之间的关系，并使用这些关系仅呈现因用户操作而需要更改的内容。 
结果是用户可以轻松开发高效、可扩展的应用程序，而无需显式缓存数据或编写回调函数。

## Python 的闪亮[Shiny for Python](https://shiny.rstudio.com/py/)是Shiny的纯Python实现。 
这使您可以访问 Shiny 的所有强大功能，例如反应性、复杂布局和模块，而无需使用 R。 
Shiny for Python 非常适合 Hugging Face 应用程序，因为它与其他 Hugging Face 工具顺利集成。

要开始部署空间，请单击此按钮选择您的硬件并指定您需要公共空间还是私有空间。
空间模板将填充一些文件以启动您的应用程序。

  

_应用程序.py_

该文件定义您的应用程序的逻辑。要了解有关如何修改此文件的更多信息，请参阅[the Shiny for Python documentation](https://shiny.rstudio.com/py/docs/overview.html)。 
随着您的应用程序变得越来越复杂，最好将应用程序逻辑分解为 [modules](https://shiny.rstudio.com/py/docs/workflow-modules.html)。

_Dockerfile_

Shiny for Python 应用程序的 Dockerfile 非常小，因为该库没有很多系统依赖项，但如果您的应用程序有其他系统依赖项，您可能需要修改此文件。 
该文件的一个基本功能是，它在空间 README 文件中指定的端口（默认情况下为 7860）上公开并运行应用程序。

__需求.txt__空间将自动安装requirements.txt 文件中列出的依赖项。 
请注意，您必须在此文件中包含闪亮。

## R 的闪亮

[Shiny for R](https://shiny.rstudio.com/) 是 R 社区中流行且完善的应用程序框架，如果您想在 Hugging Face 基础架构上托管 R 应用程序或利用一些出色的 [Shiny R extensions](https://github.com/nanxstats/awesome-shiny-extensions)，那么它是一个不错的选择。 
要将 Hugging Face 工具集成到 R 应用程序中，您可以使用 [httr2](https://httr2.r-lib.org/) 调用 Hugging Face API，或使用 [reticulate](https://rstudio.github.io/reticulate/) 调用 Hugging Face Python SDK 之一。

要部署 R Shiny Space，请单击此按钮并填写空间元数据。 
这将使用您开始使用所需的所有文件填充空间。

  

_app.R_
该文件包含所有应用程序逻辑。如果您愿意，可以将此文件分成 `ui.R` 和 `server.R`。

_Dockerfile_

Dockerfile 基于 [rocker shiny](https://hub.docker.com/r/rocker/shiny) 镜像构建。您需要修改此文件才能使用其他包。 
如果您使用大量 tidyverse 软件包，我们建议将基础镜像切换为 [rocker/shinyverse](https://hub.docker.com/r/rocker/shiny-verse)。
您可以通过在 dockerfile 的 `RUN install2.r` 部分下添加其他 R 软件包来安装它们，并且可以通过在 `RUN installGithub.r` 下添加存储库来安装 github 软件包。这个 Dockerfile 有两个主要要求：

- 首先，该文件必须公开您在自述文件中列出的端口。默认值为 7860，除非有理由，否则我们建议不要更改此端口。

- 其次，目前您必须使用[httpuv](https://github.com/rstudio/httpuv)的开发版本，它解决了Hugging Face上应用程序超时的问题。

### 如何使用 Microsoft Entra ID (Azure AD) 配置 SAML SSO
https://huggingface.co/docs/hub/security-sso-azure-saml.md