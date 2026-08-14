<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Dash on Spaces

With Dash Open Source, you can create data apps on your laptop in pure Python, no JavaScript required.

Get familiar with Dash by building a [sample app](https://dash.plotly.com/tutorial) with open source. Scale up with [Dash Enterprise](https://plotly.com/dash/) when your Dash app is ready for department or company-wide consumption.或者，从一开始就使用 Dash Enterprise 启动您的计划，以解锁开发人员生产力的提高并从 Plotly 团队获得实际加速。

## Deploy Dash on Spaces

To get started with Dash on Spaces, click the button below:

    

This will start building your Space using Plotly's Dash Docker template. If successful, you should see a similar application to the [Dash template app](https://huggingface.co/spaces/dash/dash-app-template).

## Customizing your Dash app

If you have never built with Dash before, we recommend getting started with our [Dash in 20 minutes tutorial](https://dash.plotly.com/tutorial).

When you create a Dash Space, you'll get a few key files to help you get started:

### 1. app.py

This is the main app file that defines the core logic of your project. Dash 应用程序通常被构造为模块，您可以选择将布局、回调和数据分离到其他文件中，例如 `layout.py` 等。

Inside of `app.py` you will see:1. `from dash import Dash, html`
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

由于 Dash 几乎没有系统依赖性，因此 Dash 应用程序的 Dockerfile 很小。 The key requirements are:- 它安装`requirements.txt`中列出的依赖项（使用`uv`）
- It creates a non-root user for security
- It runs the app with `gunicorn` using `gunicorn app:server --workers 4`

You may need to modify this file if your application requires additional system dependencies, permissions, or other CLI flags.

### 3.需求.txt

The Space will automatically install dependencies listed in the `requirements.txt` file. At minimum, you must include `dash` and `gunicorn` in this file.您将需要添加应用程序所需的任何其他必需包。

The Dash Space template provides a basic setup that you can extend based on your needs.

## Additional Resources and Support

- [Dash documentation](https://dash.plotly.com)
- [Dash GitHub repository](https://github.com/plotly/dash)
- [Dash Community Forums](https://community.plotly.com)
- [Dash Enterprise](https://plotly.com/dash)
- [Dash template Space](https://huggingface.co/spaces/plotly/dash-app-template)

## 故障排除

如果您遇到问题：

1. Make sure your notebook runs locally in app mode using `python app.py`
2. Check that all required packages are listed in `requirements.txt`
3. 验证端口配置匹配（7860 是 Spaces 的默认值）
4. Check Space logs for any Python errors

For more help, visit the [Plotly Community Forums](https://community.plotly.com) or [open an issue](https://github.com/plotly/dash/issues).

### 使用 TensorBoard
https://huggingface.co/docs/hub/tensorboard.md