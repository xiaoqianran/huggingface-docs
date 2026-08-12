<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 虎斑在空间

[Tabby](https://tabby.tabbyml.com)是一款开源、自托管的人工智能编码助手。借助 Tabby，每个团队都可以轻松设置自己的 LLM 支持的代码完成服务器。

在本指南中，您将学习如何部署自己的 Tabby 实例并直接从 Hugging Face 网站使用它进行开发。

## 你的第一个虎斑空间

在本节中，您将了解如何部署 Tabby Space 并将其用于您自己或您的组织。

### 在空间上部署 Tabby

只需点击几下，您就可以在 Spaces 上部署 Tabby：

[⟦T4⟧](https://huggingface.co/spaces/TabbyML/tabby-template-space?duplicate=true)

您需要定义所有者（您的个人帐户或组织）、空间名称和可见性。为了保护 api 端点，我们将可见性配置为私有。

![Duplicate Space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tabby/duplicate-space.png)

您将看到*建筑状态*。一旦它变成“正在运行”，您的空间就准备好了。如果您没有看到 Tabby Swagger UI，请尝试刷新页面。

![Swagger UI](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tabby/swagger-ui.png)

> [!提示]
> 如果您想自定义空间的标题、表情符号和颜色，请转到“文件和版本”并编辑 README.md 文件的元数据。

### 您的虎斑空间网址Tabby 启动并运行后，对于 https://huggingface.com/spaces/TabbyML/tabby 等空间链接，直接 URL 将为 https://tabbyml-tabby.hf.space。
此 URL 提供对全屏模式下稳定 Tabby 实例的访问，并用作 IDE/编辑器扩展与之通信的 API 端点。

### 将 VSCode 扩展连接到 Space 后端

1. 安装[VSCode Extension](https://marketplace.visualstudio.com/items?itemName=TabbyML.vscode-tabby)。
2. 打开位于`~/.tabby-client/agent/config.toml`的文件。取消注释 `[server]` 部分和 `[server.requestHeaders]` 部分。
   * 将端点设置为您在上一步中找到的直接 URL，它应该类似于 `https://UserName-SpaceName.hf.space`。
   * 由于空间设置为**私有**，因此必须配置访问端点的授权标头。您可以从[Access Tokens](https://huggingface.co/settings/tokens)页面获取代币。

![Agent Config](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tabby/agent-config.png)

3. 您会注意到一个 ✓ 图标，表示连接成功。
![Tabby Connected](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tabby/tabby-connected.png)

4. 您已完成设置，现在可以开始使用 Tab 了！

![Code Completion](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/tabby/code-completion.png)

您还可以在其他 IDE 中使用 Tabby 扩展，例如 [JetBrains](https://plugins.jetbrains.com/plugin/22379-tabby)。

## 反馈和支持

如果您有改进建议或需要具体支持，请加入[Tabby Slack community](https://join.slack.com/t/tabbycommunity/shared_invite/zt-1xeiddizp-bciR2RtFTaJ37RBxr8VxpA)或联系[Tabby’s GitHub repository](https://github.com/TabbyML/tabby)。

### 配置数据集查看器
https://huggingface.co/docs/hub/datasets-viewer-configure.md