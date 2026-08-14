<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 静态 HTML 空间

Spaces 还可以为您的应用程序提供自定义 HTML，而不是使用 Streamlit 或 Gradio。在 Spaces **README.md** 文件顶部的 `YAML` 块内设置 `sdk: static`。然后，您可以将 HTML 代码放入 **index.html** 文件中。

静态空间对所有人免费：它们直接提供服务，无需在计算上运行，因此与 [Gradio](./spaces-sdks-gradio) 和 [Docker](./spaces-sdks-docker) 空间不同，创建它们不需要付费计划。

以下是使用自定义 HTML 的 Spaces 的一些示例：

* [Smarter NPC](https://huggingface.co/spaces/mishig/smarter_npc)：在 Spaces 中显示带有 iframe 的 PlayCanvas 项目。
* [Huggingfab](https://huggingface.co/spaces/pierreant-p/huggingfab)：在空间中显示 Sketchfab 模型。
* [Diffuse the rest](https://huggingface.co/spaces/huggingface-projects/diffuse-the-rest)：绘制并扩散其余部分

## 在服务之前添加构建步骤

静态空间支持在提供静态资产之前添加自定义构建步骤。这对于像 React、Svelte 和 Vue 这样在提供应用程序之前需要构建过程的前端框架非常有用。当您的空间更新时，构建命令会自动运行。

在 Spaces **README.md** 文件顶部的 `YAML` 块内添加 `app_build_command` 和 `app_file`。

例如：
- `app_build_command: npm run build`
- `app_file: dist/index.html`

空间示例： 

- [Svelte App](https://huggingface.co/spaces/julien-c/vite-svelte)
- [React App](https://huggingface.co/spaces/coyotte508/static-vite)

在底层，它将[launch a build](https://huggingface.co/spaces/huggingface/space-build)，将生成的文件存储在特殊的`refs/convert/build`引用中。

## 空间变量自定义[environment variables](./spaces-overview#managing-secrets)可以传递到您的空间。如果您的空间有 [enabled OAuth](./spaces-oauth)，则 OAuth 信息（例如客户端 ID 和范围）也可用作环境变量。

要在 JavaScript 中使用这些变量，您可以使用 `window.huggingface.variables` 对象。例如，要访问 `OAUTH_CLIENT_ID` 变量，您可以使用 `window.huggingface.variables.OAUTH_CLIENT_ID`。

以下是使用自定义环境变量并启用 oauth 并在 HTML 中显示变量的空间示例：

* [Static Variables](https://huggingface.co/spaces/huggingfacejs/static-variables)

### 数据集概述
https://huggingface.co/docs/hub/datasets-overview.md