<!-- huggingface-docs: machine-translated zh-CN from English source -->

# ZenML 的空间

[ZenML](https://github.com/zenml-io/zenml) 是一个可扩展的开源 MLOps 框架，用于创建可移植的、可用于生产的 MLOps 管道。它专为数据科学家、ML 工程师和 MLOps 开发人员构建，以便他们在开发到生产时进行协作。

ZenML 提供简单灵活的语法，与云和工具无关，并且具有
接口/抽象迎合机器学习工作流程。有了 ZenML，您将拥有一切
您最喜欢的工具集中在一处，因此您可以定制适合的工作流程
您的具体需求。

ZenML Huggingface Space 允许您启动并运行已部署的版本
只需点击几下即可使用 ZenML。几分钟之内，您将获得此默认值
ZenML 仪表板已部署并可供您从本地连接
机。

在接下来的部分中，您将学习部署自己的 ZenML 实例并使用
它可以直接从中心查看和管理您的机器学习管道。 ZenML
Huggingface Spaces 上是一个**独立的应用程序，完全托管在
使用 Docker 的集线器**。下图展示了完整的过程。

![ZenML on HuggingFace Spaces -- default deployment](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/zenml/hf_spaces_chart.png)访问[the ZenML documentation](https://docs.zenml.io/)了解更多
功能以及如何开始运行机器学习管道
通过您的 Huggingface Spaces 部署。您可以查看 ZenML 管道的 [some small sample
examples](https://github.com/zenml-io/zenml/tree/main/examples) 来开始使用，或者选择更多管道
[the ZenML Projects
repository](https://github.com/zenml-io/zenml-projects) 的复杂生产级项目。 ZenML 集成
许多您最喜欢的开箱即用工具，当然是[including
Huggingface](https://zenml.io/integrations/huggingface)！如果有
如果您想使用其他东西，我们的设计是可扩展的，您可以轻松地
让它与您的自定义工具或工作流程一起工作。

## ⚡️ 在 Spaces 上部署 ZenML

只需点击几下，您就可以在 Spaces 上部署 ZenML：

    

要设置 ZenML 应用程序，您需要指定三个主要组件：
（您的个人帐户或组织）、空间名称和
可见性（页面下方一点）。请注意，空间可见性需要
如果您希望从本地连接到 ZenML 服务器，请设置为“公共”
机。

![Choose the ZenML Docker template](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/zenml/choose_space.png)您可以在此处选择更高层的计算机用于您的服务器。
选择付费CPU实例的好处是不受
自动关闭策略，因此只要您不关闭它就会保持打开状态。在
为了使用持久的 CPU，您可能需要创建并设置一个
要连接的 MySQL 数据库（见下文）。

要个性化您的空间外观，例如标题、表情符号和颜色，
导航到“文件和版本”并修改 README.md 文件中的元数据。
有关 Spaces 配置参数的完整信息可以在
拥抱脸[documentation reference guide](https://huggingface.co/docs/hub/spaces-config-reference)。

创建空间后，您会注意到“建筑物”状态以及日志
显示在屏幕上。当切换到“正在运行”时，您的空间就可以使用了。如果
ZenML 登录 UI 不可见，请尝试刷新页面。在空间的右上角，您会看到一个带有三个点的按钮
当您单击它时，将为您提供“嵌入此空间”的菜单选项。
（有关更多详细信息，请参阅[the HuggingFace
documentation](https://huggingface.co/docs/hub/spaces-embed)
此功能。）复制您现在可以看到的框中显示的“直接 URL”
屏幕。这应该看起来像这样：
`https://<YOUR_USERNAME>-<SPACE_NAME>.hf.space`。打开该 URL 并使用我们的默认值 
登录以访问仪表板（用户名：'default'，密码：（留空））。

## 从本地计算机连接到 ZenML 服务器

一旦您的 ZenML 服务器启动并运行，您就可以从您的计算机连接到它
本地机器。为此，您需要获取空间的“直接 URL”（见上文）。

> [!警告]
> 您的空间 URL 仅可用于从您的连接
> 本地计算机（如果空间的可见性设置为“公共”）。

您可以使用“直接 URL”从本地计算机连接到 ZenML 服务器
使用以下 CLI 命令（安装 ZenML 并使用您的自定义
URL 而不是占位符）：

```shell
zenml connect --url '<YOUR_HF_SPACES_DIRECT_URL>' --username='default' --password=''
```

您还可以使用浏览器中的直接 URL 将 ZenML 仪表板用作
全屏应用程序（即没有 HuggingFace Spaces 包装器）。> [!警告]
> 从 Huggingface 内部查看时，ZenML 仪表板当前无法工作 
> 网页（即包含在主 `https://huggingface.co/...` 网站中）。这是在 
> 说明 ZenML 和 Huggingface 之间处理 cookie 的方式存在限制。 
> 您**必须**从“直接 URL”查看仪表板（见上文）。

## 额外的配置选项

默认情况下，ZenML 应用程序将配置为使用 SQLite
非持久性数据库。如果你想使用持久数据库，你可以
通过修改空间根目录中的`Dockerfile`来配置它。对于
有关您可以更改的各种参数的完整详细信息，请参阅有关配置的[our reference
documentation](https://docs.zenml.io/getting-started/deploying-zenml/docker#zenml-server-configuration-options)
使用 Docker 部署时的 ZenML。

> [!提示]
> 如果您仅使用该空间进行测试和实验，则不需要
> 对配置进行任何更改。一切都会开箱即用。

您还可以将外部机密后端与 HuggingFace 一起使用
[our
documentation](https://docs.zenml.io/getting-started/deploying-zenml/docker#zenml-server-configuration-options) 中描述的空间。你应该是
确保使用 HuggingFace 的内置“存储库秘密”功能
配置您需要在`Dockerfile`配置中使用的任何秘密。 [See the
documentation](https://huggingface.co/docs/hub/spaces-sdks-docker#secret-management)
有关如何设置的更多详细信息。> [!警告]
> 如果您希望将云机密后端与 ZenML 一起用于机密
> 管理，**您必须在您的 ZenML 服务器上采取以下最低限度的安全预防措施**
> 仪表板：
>
> - 更改您开始时获得的`default`帐户的密码。你
> 可以从仪表板或通过 CLI 执行此操作。
> - 创建一个带有密码的新用户帐户并为其分配 `admin` 角色。这个
> 也可以从仪表板（通过“邀请”新用户）或通过 CLI 完成。
> - 按照描述使用新用户帐户和密码重新连接到服务器
> 并使用这个新用户帐户作为您的工作帐户。
>
> 这是因为创建的默认用户
> HuggingFace Spaces 部署过程没有分配密码，并且作为
> 空间可公开访问（因为空间是公共的）*可能任何人
> 无需此额外步骤即可访问您的秘密*。更改您的密码
> 单击右上角的按钮导航至“设置”页面
> 仪表板一角，然后单击“更新密码”。

## 在 HF Spaces 上升级 ZenML 服务器默认空间将自动使用最新版本的 ZenML。如果你
想要更新您的版本，您只需选择“工厂重启”选项即可
在空间的“设置”选项卡中。请注意，这将擦除所有数据
包含在该空间内，因此如果您没有使用 MySQL 持久性
数据库（如上所述）您将丢失 ZenML 中包含的所有数据
空间上的部署。您还可以配置空间以使用较早的
通过更新最顶部的 `Dockerfile` 的 `FROM` 导入语句来更新版本。

## 后续步骤

下一步，请查看我们的 [Starter Guide to MLOps with
ZenML](https://docs.zenml.io/starter-guide/pipelines)，这是一系列短片
有关如何快速入门的实用页面。或者，查看 [our
⟦T9⟧
example](https://github.com/zenml-io/zenml/tree/main/examples/quickstart)
是 ZenML 许多功能的完整端到端示例。

## 🤗 反馈和支持

如果您在 HuggingFace Spaces 上的 ZenML 服务器遇到问题，您可以
单击空间顶部的“打开日志”按钮查看日志。
这将为您提供有关服务器所发生情况的更多背景信息。

如果您有任何建议或需要其他任何不属于的具体支持
工作中，请[join the ZenML Slack community](https://zenml.io/slack-invite/)
我们很乐意为您提供帮助！### DDUF
https://huggingface.co/docs/hub/dduf.md