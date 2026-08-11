<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间 Livebook

**Livebook** 是一个开源工具，用于在 [Elixir](https://elixir-lang.org/) 中编写交互式代码笔记本。它是不断增长的适用于 [numerical computing](https://github.com/elixir-nx/nx)、[data science](https://github.com/elixir-nx/explorer) 和 [Machine Learning](https://github.com/elixir-nx/bumblebee) 的 Elixir 工具集合的一部分。

Livebook 的一些最令人兴奋的功能包括：

- **可重复的工作流程**：Livebook 以可预测的顺序运行您的代码，一直到包管理
- **智能单元**：使用 Livebook 的可扩展笔记本单元，只需点击几下即可执行复杂的任务，例如数据操作和运行机器学习模型
- **Elixir 支持**：使用 Elixir 编程语言的强大功能来编写可扩展到您的机器之外的并发和分布式笔记本

要了解更多信息，请观看此[15-minute video](https://www.youtube.com/watch?v=EhSNXWkji6o)。或访问[Livebook's website](https://livebook.dev/)。或者关注其[Twitter](https://twitter.com/livebookdev)和[blog](https://news.livebook.dev/)以跟上新功能和更新。

## 你的第一个 Livebook 空间

只需点击几下，您就可以在 Space 中启动并运行 Livebook。单击下面的按钮开始使用 Livebook 的 Docker 模板创建新空间：

    

然后：

1. 为您的空间命名
2. 设置Livebook的密码
3. 设置其对公众的可见性
4. 创建你的空间

![Creating a Livebok Space ](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/spaces-livebook-new-space.png)

这将开始使用 Livebook 的 Docker 镜像构建您的空间。空间的可见性必须设置为公开，Livebook 中的智能单元功能才能正常运行。但是，您的 Livebook 实例将受到 Livebook 身份验证的保护。

> [!提示]
> 智能单元是一种 Livebook 单元，它提供用于完成特定任务的 UI 组件。任务的代码是根据用户与 UI 的交互自动生成的，从而可以更快地完成高级任务，而无需从头开始编写代码。

应用程序构建完成后，转到空间中的“应用程序”选项卡并使用您之前设置的密码登录您的 Livebook：

![Livebook authentication](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/spaces-livebook-authentication.png)

就是这样！现在您可以开始在您的空间内使用 Livebook。

如果这是您第一次使用 Livebook，您可以在 Livebook 本身中了解如何将其与交互式笔记本一起使用：

![Livebook's learn notebooks](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/spaces-livebook-learn-section.png)

## Livebook 与拥抱脸部模型的集成

Livebook 有一个 [official integration with Hugging Face models](https://livebook.dev/integrations/hugging-face)。借助此功能，您只需点击几下即可在 Livebook 中运行各种机器学习模型。

这是一个演示如何执行此操作的快速视频：

## 如何更新Livebook的版本要将 Livebook 更新到最新版本，请转到空间的“设置”页面，然后单击“恢复出厂设置”：

![Factory reboot a Space](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/spaces-livebook-factory-reboot.png)

## 注意事项

以下注意事项适用于在空间内运行 Livebook：

- 空间的可见性设置必须是公开的。否则，智能电池将无法工作。也就是说，您的 Livebook 实例仍将支持 Livebook 身份验证，因为您已经设置了 `LIVEBOOK_PASSWORD` 秘密。
- Space重启后Livebook全局配置将会丢失。如果您发现自己需要跨部署保留配置，请考虑使用[desktop app](https://livebook.dev/#install)。

## 反馈和支持

如果您有改进建议或需要具体支持，请加入[Livebook community on GitHub](https://github.com/livebook-dev/livebook/discussions)。

### 如何使用 Google Workspace 配置 SAML SSO
https://huggingface.co/docs/hub/security-sso-google-saml.md