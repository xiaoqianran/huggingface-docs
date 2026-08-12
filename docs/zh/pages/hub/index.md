<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 拥抱 Face Hub 文档

Hugging Face Hub 是开放式机器学习的参考 AI 平台。它托管超过 200 万个模型、150 万个数据集和 150 万个 AI 应用程序 (Spaces)，全部开放且公开可用。除了开放人工智能之外，该中心还是内部和私人团队的绝佳协作平台。探索、实验、协作和构建，尽在一个地方！ 🤗

 订阅和计划
专业版订阅
团队和企业计划
单点登录 (SSO)
审核日志
存储区域
用于私有数据集的 Data Studio
资源组
高级安全性
代币管理
网络安全
速率限制

 存储库
开始使用
存储库设置
存储限制
存储后端 (Xet)
本地缓存
拉取请求和讨论
通知
收藏
网络钩子
下一步
许可证

     型号
模型中心
模型卡
评估结果
门控模型
上传模型
下载模型
图书馆
任务
小部件
推理提供者
下载统计数据

 数据集
简介
数据集概述
数据集卡
门控数据集
上传数据集
摄取数据集
下载数据集
流数据集
编辑数据集
图书馆
数据工作室
下载统计数据
数据文件配置空间
简介
空间概览
渐变空间
静态 HTML 空间
Docker 空间
零GPU空间
嵌入您的空间
使用 Docker 运行
参考
高级主题
使用 HF 登录

 存储桶新
简介
存储桶与 Git 存储库
创建桶
管理文件
使用案例
安全与合规性

 工作机会
简介
职位概览
快速入门
定价
管理职位
职位配置
热门图片
安排工作
Webhooks 自动化
参考

 代理商
简介
代理商概览
用于 AI 代理的 Hugging Face CLI
抱脸MCP服务器
抱脸特工技能
使用 HF SDK 构建代理
当地代理
代理库

 其他
组织机构
计费
安全性
适度
纸页
学术中心
博客文章
搜索
数字对象标识符 (DOI)
集线器 API 端点
使用 HF 登录
贡献者行为准则
内容指南

## 拥抱脸部中心是什么？

我们正在帮助社区共同努力实现推进机器学习的目标。包括科技巨头在内的任何一家公司都无法独自“解决人工智能问题”——我们实现这一目标的唯一方法是以社区为中心的方式共享知识和资源。我们正在 Hugging Face Hub 上构建最大的开源模型、数据集和演示集合，以便为每个人实现 ML 的民主化和进步。

我们鼓励您阅读[Code of Conduct](https://huggingface.co/code-of-conduct)和[Content Guidelines](https://huggingface.co/content-guidelines)，以熟悉我们期望社区成员坚持的价值观🤗。

## 您可以在 Hub 上找到什么？

Hugging Face Hub 托管基于 Git 的存储库，这些存储库是版本控制的文件夹，可以包含您的所有文件。对于非版本化、可变对象存储，Hub 还提供 [Storage Buckets](./storage-buckets)。

在上面，您将能够上传并发现...

- 模型：_托管法学硕士、文本、视觉和音频任务的最新最先进模型_
- 数据集：_具有不同领域和模式的各种数据_
- Spaces：_用于直接在浏览器中演示 ML 模型的交互式应用程序_该中心提供**版本控制、提交历史记录、差异、分支和十多个库集成**！ 
所有存储库均基于 [Xet](./xet/index) 构建，这是一项新技术，可在 Git 内高效存储大文件，智能地将文件拆分为独特的块并加速上传和下载。

您可以在[**Repositories documentation**](./repositories)中了解有关所有存储库共享的功能的更多信息。

## 型号

您可以发现和使用社区共享的数十万个开源 ML 模型。为了促进负责任的模型使用和开发，模型存储库配备了[Model Cards](./model-cards)，以告知用户每个模型的局限性和偏差。可以包含有关任务、语言和评估结果等信息的附加[metadata](./model-cards#model-card-metadata)，如果存储库包含[TensorBoard traces](./tensorboard)，甚至还可以添加训练指标图表。向您的模型添加 [**inference widget**](./models-widgets) 也很容易，让任何人都可以直接在浏览器中使用该模型！对于编程访问，[**Inference Providers**](./models-inference) 提供了无服务器 API。

要将模型上传到 Hub，或下载模型并将其集成到您的工作中，请探索 [**Models documentation**](./models)。您还可以选择支持 Hub 的[**over a dozen libraries**](./models-libraries)，例如🤗 Transformers、Asteroids 和 ESPnet。

## 数据集该中心拥有超过 50 万个公共数据集，涵盖超过 8000 种语言，可用于 NLP、计算机视觉和音频等广泛的任务。该中心使查找、下载和上传数据集变得简单。数据集附有[**Dataset Cards**](./datasets-cards)和[**Data Studio**](./datasets-viewer)形式的大量文档，让您可以直接在浏览器中探索数据。虽然许多数据集是公开的，但[**organizations**](./organizations)和个人可以创建私有数据集以遵守许可或隐私问题。您可以了解更多关于[**Datasets here on the Hugging Face Hub documentation**](./datasets-overview)的信息。

[🤗 ⟦T0⟧](https://huggingface.co/docs/datasets/index) 库允许您以编程方式与数据集交互，因此您可以在项目中轻松使用 Hub 中的数据集。只需一行代码，您就可以访问数据集；即使它们太大而无法放入您的计算机，您也可以使用流式传输来有效地访问数据。

## 空格

[Spaces](https://huggingface.co/spaces) 是在 Hub 上托管 ML 演示应用程序的简单方法。它们允许您构建 ML 产品组合、在会议上或向利益相关者展示您的项目，以及与 ML 生态系统中的其他人员协作。我们目前支持两个很棒的 Python SDK（**[Gradio](https://gradio.app/)** 和 **[Streamlit](./spaces-sdks-streamlit)**），让您可以在几分钟内构建很酷的应用程序。用户还可以创建静态空间，即简单的 HTML/CSS/JavaScript 页面，或部署任何基于 Docker 的应用程序。

如果您的演示需要 GPU 能力，请尝试[**ZeroGPU**](./spaces-zerogpu)：它仅在需要时实时动态提供 NVIDIA RTX Pro 6000 Blackwell GPU。

探索了几个空间后（看看我们的 [Space of the Week!](https://huggingface.co/spaces)），请深入了解 [**Spaces documentation**](./spaces-overview) 了解如何创建自己的空间。您还可以升级 Space 以在 GPU 或其他加速硬件上运行。 ⚡️

## 存储桶

[Storage Buckets](./storage-buckets) 在 Hugging Face 上提供类似 S3 的对象存储，由 Xet 存储后端提供支持。与存储库（基于 git 并跟踪文件历史记录）不同，存储桶是远程对象存储容器，专为具有内容可寻址重复数据删除功能的大型文件而设计。它们专为需要简单、快速、可变存储的用例而设计，例如存储训练检查点、日志、中间工件或任何不需要版本控制的大型文件集合。

## 组织公司、大学和非营利组织是 Hugging Face 社区的重要组成部分！ Hub 提供[**Organizations**](./organizations)，可用于对帐户进行分组并管理数据集、模型和空间。教育工作者还可以使用[Hugging Face for Classrooms](https://huggingface.co/classrooms)为学生创建协作组织。组织的存储库将显示在组织的页面上，并且组织的每个成员都将能够为存储库做出贡献。除了方便地对组织的所有工作进行分组之外，该中心还允许管理员将角色设置为[**control access to repositories**](./organizations-security)，并管理其组织的[payment method and billing info](https://huggingface.co/pricing)。协作时机器学习会更有趣！ 🔥

[Explore existing organizations](https://huggingface.co/organizations)，创建新组织[here](https://huggingface.co/organizations/new)，然后访问[**Organizations documentation**](./organizations)了解更多信息。

## 安全

Hugging Face Hub 支持安全和访问控制功能，让您放心，您的代码、模型和数据都是安全的。请访问这些文档中的 [**Security**](./security) 部分以了解：

- 用户访问令牌
- 组织的访问控制
- 使用 GPG 签署提交
- 恶意软件扫描

### Spaces 开发模式：Spaces 中的无缝开发
https://huggingface.co/docs/hub/spaces-dev-mode.md