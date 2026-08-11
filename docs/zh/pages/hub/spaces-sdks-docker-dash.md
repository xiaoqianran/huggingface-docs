<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 冲向空格

借助 Dash Open Source，您可以使用纯 Python 在笔记本电脑上创建数据应用程序，无需 JavaScript。

通过使用开源构建 [sample app](https://dash.plotly.com/tutorial) 来熟悉 Dash。当您的 Dash 应用程序准备好供部门或公司范围使用时，可使用 [Dash Enterprise](https://plotly.com/dash/) 进行扩展。或者，从一开始就使用 Dash Enterprise 启动您的计划，以解锁开发人员生产力的提高并从 Plotly 团队获得实际加速。

## 在 Spaces 上部署 Dash

要开始使用 Dash on Spaces，请单击下面的按钮：

    

这将开始使用 Plotly 的 Dash Docker 模板构建您的空间。如果成功，您应该会看到与[Dash template app](https://huggingface.co/spaces/dash/dash-app-template)类似的应用程序。

## 定制您的 Dash 应用

如果您以前从未使用过 Dash 构建过，我们建议您开始使用我们的 [Dash in 20 minutes tutorial](https://dash.plotly.com/tutorial)。

创建 Dash Space 时，您将获得一些关键文件来帮助您入门：

### 1.app.py

这是定义项目核心逻辑的主应用程序文件。 Dash 应用程序通常被构造为模块，您可以选择将布局、回调和数据分离到其他文件中，例如 `layout.py` 等。

在`app.py`里面你会看到：1. `from dash import Dash, html`
   我们导入 `Dash` 对象来定义我们的应用程序，以及 `html` 库，它为我们提供了构建项目的构建块。

2.`app = Dash()`
   在这里，我们定义我们的应用程序。布局、服务器和回调都_绑定_到`app`对象。

3.`server = app.server`
   在这里，我们定义服务器变量，用于在生产中运行应用程序。

4.`app.layout = `
   入门应用程序布局定义为 Dash 组件列表、单个 Dash 组件或返回其中任一组件的函数。

   `app.layout` 是您的初始布局，它将通过项目中的回调和其他逻辑更新为单页应用程序。

5.`if __name__ == '__main__': app.run(debug=True)`
   如果您使用 `python app.py` 在本地运行项目，`app.run(...)` 将执行并启动开发服务器来处理您的项目，其功能包括热重载、回调图等。

   在生产中，我们推荐`gunicorn`，它是生产级服务器。使用 `gunicorn` 运行项目时，不会启用调试功能，因此永远不会到达此行。

### 2.Dockerfile

由于 Dash 几乎没有系统依赖性，因此 Dash 应用程序的 Dockerfile 很小。关键要求是：- 它安装`requirements.txt`中列出的依赖项（使用`uv`）
- 为了安全起见，它创建一个非root用户
- 它使用 `gunicorn app:server --workers 4` 运行应用程序 `gunicorn`

如果您的应用程序需要额外的系统依赖项、权限或其他 CLI 标志，您可能需要修改此文件。

### 3.需求.txt

Space 将自动安装`requirements.txt` 文件中列出的依赖项。您至少必须在此文件中包含 `dash` 和 `gunicorn`。您将需要添加应用程序所需的任何其他必需包。

Dash Space 模板提供了一个基本设置，您可以根据需要进行扩展。

## 其他资源和支持

- [Dash documentation](https://dash.plotly.com)
- [Dash GitHub repository](https://github.com/plotly/dash)
- [Dash Community Forums](https://community.plotly.com)
- [Dash Enterprise](https://plotly.com/dash)
- [Dash template Space](https://huggingface.co/spaces/plotly/dash-app-template)

## 故障排除

如果您遇到问题：

1. 使用 `python app.py` 确保您的笔记本在应用程序模式下本地运行
2. 检查`requirements.txt`中是否列出了所有需要的包
3. 验证端口配置匹配（7860 是 Spaces 的默认值）
4. 检查 Space 日志中是否有任何 Python 错误

如需更多帮助，请访问[Plotly Community Forums](https://community.plotly.com)或[open an issue](https://github.com/plotly/dash/issues)。

### 组织、安全性和 Hub API
https://huggingface.co/docs/hub/other.md