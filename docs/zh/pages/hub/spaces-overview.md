<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间概述

Hugging Face Spaces 让您可以在几分钟内轻松创建和部署 ML 支持的演示。观看以下视频，快速了解 Spaces：

在以下部分中，您将学习创建空间、配置空间以及将代码部署到空间的基础知识。

## 创建一个新空间

**要创建新空间**，请访问[Spaces main page](https://huggingface.co/spaces)并单击**创建新空间**。除了为您的空间选择名称、选择可选许可证以及设置您的空间的 [visibility](#space-visibility)（公共、受保护或私有）之外，系统还会提示您为您的空间选择 **SD​​K**。 Hub 提供三种 SDK 选项：Gradio、Docker 和静态 HTML。如果您选择“Gradio”作为 SDK，您将导航到显示以下页面的新存储库：

> [!警告]
> 静态空间对所有人免费。 Gradio 和 Docker Spaces 在计算上运行，需要付费计划才能创建：用于个人帐户的 PRO、用于组织的团队或企业帐户。信誉良好的免费个人帐户仍然可以托管最多 2 个在 [ZeroGPU](./spaces-zerogpu) 上运行的 Gradio Space。在底层，Spaces 将您的代码存储在 git 存储库中，就像模型和数据集存储库一样。因此，我们用于所有 [other repositories on the Hub](./repositories)（`git` 和 `git-xet`）的相同工具也适用于空间。按照与[Getting Started with Repositories](./repositories-getting-started)中相同的流程将文件添加到您的空间。每次推送新的提交时，空间都会自动重建并重新启动。

有关创建第一个空间的分步教程，请参阅以下指南：
* [Creating a Gradio Space](./spaces-sdks-gradio)
* [Creating a Docker Space](./spaces-sdks-docker-first-demo)

## 空间可见性

您可以使用可见性下拉列表从 **设置** 选项卡设置空间的可见性。空间支持三种可见性级别：**公共**、**受保护**和**私有**。

> [!警告]
> 受保护的可见性是 PRO 或团队和企业计划的一部分。

| |公共|受保护 |私人|
|---|---|---|---|
| Hub 上的源代码 |所有人都可以看到 |私人（仅限所有者/合作者）|私人（仅限所有者/合作者）|
|可通过嵌入 URL 访问应用程序 |是的 |是的 |没有 |
|通过 [custom domain](./spaces-custom-domain) 访问应用程序 |是的 |是的 |没有 |
|可以被其他人克隆|是的 |没有 |没有 |

**公共**空间完全开放：任何人都可以查看源代码、访问正在运行的应用程序以及克隆存储库。**受保护** 空间在 Hub 上保留其源代码的私密性 - 只有所有者和协作者可以查看或克隆存储库。但是，正在运行的应用程序可以通过其嵌入 URL (`https://<space-subdomain>.hf.space`) 或在配置后通过 [custom domain](./spaces-custom-domain) 公开访问。这对于在不发布源代码的情况下托管网站或应用程序特别有用。

**私人** 空间是完全私人的：源代码和正在运行的应用程序只能由所有者和协作者访问。该空间不会出现在搜索结果中，其他用户在访问其 URL 时将收到 `404` 错误。

## 硬件资源

默认情况下，每个 Spaces 环境仅限于 16GB RAM、2 个 CPU 核心和 50GB（非持久）磁盘空间。默认的 CPU Basic 硬件没有每小时成本，但创建在计算（Gradio 或 Docker）上运行的空间需要付费计划，而静态空间对所有人免费。您可以为 [competitive price](https://huggingface.co/pricing#spaces) 升级到更好的硬件，包括各种 GPU 加速器。要请求升级，请单击您空间中的_设置_按钮并选择您喜欢的硬件环境。| **硬件** | **CPU** | **内存** | **GPU 内存** | **每小时价格** |
|------------------------ |-------------- |------------- |---------------- | ----------------- |
| CPU基础| 2 个虚拟CPU | 16GB|                 |免费|
| CPU升级 | 8 个 vCPU | 32GB|                 | 0.03 美元 | 
| Nvidia T4 - 小| 4 个虚拟CPU | 15GB| 16GB| 0.40 美元 |
| Nvidia T4 - 中 | 8 个 vCPU | 30GB| 16GB| 0.60 美元 |
| 1x Nvidia L4 | 8 个 vCPU | 30GB| 24GB| 0.80 美元 |
| 4x Nvidia L4 | 48 个 vCPU | 186 GB | 186 GB 96GB | 3.80 美元 |
| 1x Nvidia L40S | 8 个 vCPU | 62GB| 48GB| 1.80 美元 |
| 4x Nvidia L40S | 48 个 vCPU | 382GB| 192 GB | 192 GB 8.30 美元 |
| 8x Nvidia L40S | 192 个 vCPU | 1534GB| 384GB| 23.50 美元 |
| Nvidia A10G - 小| 4 个虚拟CPU | 15GB| 24GB| 1.00 美元 || Nvidia A10G - 大| 12 个 vCPU | 46GB| 24GB| 1.50 美元 |
| 2x Nvidia A10G - 大 | 24 个 vCPU | 92GB| 48GB| $3.00 |
| 4x Nvidia A10G - 大 | 48 个 vCPU | 184 GB | 184 GB 96GB | 5.00 美元 |
| Nvidia A100 - 大| 12 个 vCPU | 142 GB | 142 GB 80GB| 2.50 美元 |
| 4x Nvidia A100 | 48 个 vCPU | 568GB| 320GB| 10.00 美元 |
| 8x Nvidia A100 | 96 个 vCPU | 1136 GB | 1136 GB 640GB| 20.00 美元 |

注意：在 [our pricing page](https://huggingface.co/pricing) 上查找更详细、更全面的定价信息。

您是否拥有很棒的空间，但需要帮助支付硬件升级费用？我们热衷于帮助那些拥有创新空间的人，因此请随时使用您空间的_设置_选项卡中的链接申请社区 GPU 拨款，看看您的空间是否符合要求！

请阅读我们关于 [Spaces GPU Upgrades](./spaces-gpus) 和 [Spaces Disk Usage & Storage](./spaces-storage) 的专门部分了解更多信息。

## 管理机密和环境变量[[managing-secrets]]如果您的应用程序需要环境变量（例如密钥或令牌），请勿在应用程序中对它们进行硬编码！相反，请转到空间存储库的“设置”页面并添加新的**变量**或**秘密**。如果您需要存储非敏感配置值和秘密以存储访问令牌、API 密钥或任何敏感值或凭据，请使用变量。

	
	

您可以使用：

* **变量** 如果您需要存储非敏感配置值。它们可以公开访问和查看，并将自动添加到与您的空间重复的空间中。
* **秘密** 用于存储访问令牌、API 密钥或任何敏感值或凭据。它们是私有的，一旦设置，就无法从空间的设置页面读取它们的值。它们不会添加到从您的存储库复制的空间中。

根据您的 Space SDK，访问机密和变量会有所不同：

- 对于静态空间，两者都可以通过 `window.huggingface.variables` 中的客户端 JavaScript 获得
- 对于 Docker 空间，请查看 [environment management with Docker](./spaces-sdks-docker#secrets-and-variables-management)对于其他空间，两者都会作为环境变量暴露给您的应用程序。这是一个在 Python 中访问先前声明的 `MODEL_REPO_ID` 变量的非常简单的示例（对于秘密来说是相同的）：
```py
import os
print(os.getenv('MODEL_REPO_ID'))
```

当我们的 `Spaces Secrets Scanner` [finds hard-coded secrets](./security-secrets) 时，空间所有者会收到警告。

## 复制空间

如果您想使用另一个演示作为初始模板来构建新演示，则复制空间会很有用。如果您想拥有一个单独的升级空间供您快速推理使用，那么重复空间也很有用。

如果您想复制一个空间，可以单击该空间右上角的三个点，然后单击“**复制此空间**”。完成此操作后，您将能够更改以下属性：* 所有者：复制的空间可以位于您的帐户或您具有写入权限的任何组织下
* 空间名称
* 可见性：空间默认是私有的。了解有关可见性选项 [here](./repositories-settings#repository-visibility) 的更多信息。
* 硬件：您可以选择运行空间的硬件。了解有关硬件升级的更多信息[here](./spaces-gpus)。
* 存储：如果原始仓库使用存储桶，则会提示您配置存储。阅读有关磁盘使用和存储的更多信息[here](./spaces-storage)。
* 秘密和变量：如果原始存储库设置了一些秘密和变量，您将能够在复制存储库时设置它们。

某些空间可能具有您可能需要设置的环境变量。在这些情况下，重复的工作流程将自动填充源空间中的公共变量，并向您发出有关设置 Secret 的警告。默认情况下，复制的空间将使用 CPU Basic 硬件，但您可以根据需要稍后进行升级。复制遵循与创建新空间相同的规则：复制 Gradio 或 Docker Space 需要为目标帐户或组织提供付费计划（个人帐户具有相同的 [ZeroGPU free tier](./spaces-zerogpu) 例外）。

## 网络如果您的空间需要发出任何网络请求，您可以通过标准 HTTP 和 HTTPS 端口（80 和 443）以及端口 8080 发出请求。任何发送到其他端口的请求都将被阻止。

## 生命周期管理

在免费硬件上，如果未使用，您的 Space 将在一段时间后“进入睡眠状态”并停止执行。如果您希望您的空间无限期运行，请考虑[upgrading to paid hardware](./spaces-gpus)。您还可以从 **设置** 选项卡手动暂停您的空间。暂停的空间将停止执行，直到由其所有者手动重新启动。
暂停时间不计费。

## 内置环境变量

在某些情况下，您可能有兴趣以编程方式访问空间作者或存储库名称。当您希望用户复制您的空间时，此功能特别有用。为了解决这个问题，Spaces 在运行时公开了不同的环境变量（另请参见[built-in environment variables in Jobs](./jobs-configuration#built-in-environment-variables)）。给定一个空间[⟦T9⟧](https://huggingface.co/spaces/osanseviero/i-like-flan)：* `ACCELERATOR`：可用加速器的类型（例如，`t4-medium`、`a10g-small`），或`none`（仅限 CPU 空间）。
* `CPU_CORES`: 4
* `MEMORY`：15Gi
* `SPACE_AUTHOR_NAME`: 奥桑维罗
* `SPACE_REPO_NAME`: 我喜欢果馅饼
* `SPACE_TITLE`：我喜欢果馅饼（自述文件中指定）
* `SPACE_ID`: `osanseviero/i-like-flan`
* `SPACE_HOST`: `osanseviero-i-like-flan.hf.space`
* `SPACE_CREATOR_USER_ID`: `6032802e1f993496bc14d9e3` - 这是最初创建空间的用户的 ID。如果空间属于某个组织，则此功能很有用。您可以通过API调用`https://huggingface.co/api/users/{SPACE_CREATOR_USER_ID}/overview`获取用户信息。

如果您的空间启用了[OAuth](./spaces-oauth)，则以下变量也将可用：

* `OAUTH_CLIENT_ID`：您的 OAuth 应用程序的客户端 ID（公共）
* `OAUTH_CLIENT_SECRET`：OAuth 应用程序的客户端密钥
* `OAUTH_SCOPES`：OAuth 应用程序可访问的范围。目前，这始终是`"openid profile"`。
* `OPENID_PROVIDER_URL`：OpenID 提供商的 URL。 OpenID 元数据将于 [⟦T31⟧](https://huggingface.co/.well-known/openid-configuration) 提供。

## 克隆存储库

您可以轻松地在本地克隆您的 Space 存储库。首先单击空间页面右上角的下拉菜单：

选择“克隆存储库”，然后您将能够按照说明使用 HTTPS 或 SSH 将 Space 存储库克隆到本地计算机。

## 在 Hub 上链接模型和数据集您可以通过在空间的自述文件元数据中添加标识符来展示您的空间链接到的所有模型和数据集。为此，您可以在 `models` 和 `datasets` 键下定义它们。除了在 README 文件中列出工件之外，您还可以将它们记录在任何 `.py`、`.ini` 或 `.html` 文件中。我们会神奇地自动解析它！

以下是从空间链接两个模型的示例：

```
title: My lovely space
emoji: 🤗
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
models:
- reach-vb/musicgen-large-fp16-endpoint
- reach-vb/wav2vec2-large-xls-r-1B-common_voice7-lt-ft
```

### 迪迪标签
https://huggingface.co/docs/hub/datasets-distilabel.md