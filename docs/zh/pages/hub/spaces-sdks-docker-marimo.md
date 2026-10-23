<!-- huggingface-docs: machine-translated zh-CN from English source -->

# marimo 上的空间

[marimo](https://github.com/marimo-team/marimo) 是 Python 的反应式笔记本，它将笔记本建模为数据流图。当您运行单元格或与 UI 元素交互时，marimo 会自动运行受影响的单元格（或将它们标记为过时），保持代码和输出一致并防止错误发生。每个 marimo 笔记本都以纯 Python 形式存储，可以作为脚本执行，也可以作为应用程序部署。

主要特点：

- ⚡️ **反应式：** 运行一个单元，marimo 反应性地运行所有依赖单元或将它们标记为过时
- 🖐️ **交互式：** 将滑块、表格、绘图等绑定到 Python - 无需回调
- 🔬 **可重复：** 无隐藏状态、确定性执行、内置包管理
- 🏃 **可执行文件：** 作为 Python 脚本执行，由 CLI 参数参数化
- 🛜 **可共享：** 部署为交互式 Web 应用程序或幻灯片，通过 WASM 在浏览器中运行
- 🛢️ **专为数据设计：** 使用 SQL 查询数据框和数据库，过滤和搜索数据框

## 在 Spaces 上部署 marimo 应用程序

要开始在 Spaces 上使用 marimo，请单击下面的按钮：这将开始使用 marimo 的 Docker 模板构建您的空间。如果成功，您应该会看到与[marimo introduction notebook](https://huggingface.co/spaces/marimo-team/marimo-app-template)类似的应用程序。

## 定制您的 marimo 应用程序

创建 marimo 空间时，您将获得一些关键文件来帮助您入门：

### 1.app.py

这是您的主要 marimo 笔记本文件，用于定义应用程序的逻辑。 marimo 笔记本是纯 Python 文件，使用 `@app.cell` 装饰器来定义单元格。要了解有关构建笔记本和应用程序的更多信息，请参阅[the marimo documentation](https://docs.marimo.io)。随着应用程序的发展，您可以将代码组织到模块中并将它们导入到主笔记本中。

### 2.Dockerfile

marimo 应用程序的 Dockerfile 很小，因为 marimo 几乎没有系统依赖项。关键要求是：

- 它安装`requirements.txt`中列出的依赖项（使用`uv`）
- 为了安全起见，它创建一个非root用户
- 它使用 `marimo run app.py` 运行应用程序

如果您的应用程序需要额外的系统依赖项、权限或其他 CLI 标志，您可能需要修改此文件。

### 3.需求.txt

Space 将自动安装`requirements.txt` 文件中列出的依赖项。至少，您必须在此文件中包含 `marimo`。您将需要添加应用程序所需的任何其他必需包。marimo Space 模板提供了一个基本设置，您可以根据需要进行扩展。部署后，您的笔记本将在“应用程序模式”下运行，该模式隐藏代码单元并仅显示交互式输出 - 非常适合与最终用户共享。您可以通过设置将 `--include-code` 添加到 Dockerfile 中的 `marimo run` 命令来选择在应用程序中包含代码单元。

## 其他资源和支持

- [marimo documentation](https://docs.marimo.io)
- [marimo GitHub repository](https://github.com/marimo-team/marimo)
- [marimo Discord](https://marimo.io/discord)
- [marimo template Space](https://huggingface.co/spaces/marimo-team/marimo-app-template)

## 故障排除

如果您遇到问题：

1. 使用 `marimo run app.py` 确保您的笔记本在应用程序模式下本地运行
2. 检查`requirements.txt`中是否列出了所有需要的包
3. 验证端口配置匹配（7860 是 Spaces 的默认值）
4. 检查 Space 日志中是否有任何 Python 错误

如需更多帮助，请访问[marimo Discord](https://marimo.io/discord)或[open an issue](https://github.com/marimo-team/marimo/issues)。

### 模型卡
https://huggingface.co/docs/hub/model-cards.md