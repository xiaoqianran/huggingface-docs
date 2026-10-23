<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 处理渐变空间中的空间依赖性

## 默认依赖

默认的 Gradio Spaces 环境附带了几个预安装的依赖项：

* [⟦T0⟧](https://huggingface.co/docs/huggingface_hub/index) 客户端库允许您使用 Python 管理 Hub 上的存储库和文件，并以编程方式从您的空间访问 [Inference Providers](./models-inference)。如果您选择使用推理提供程序在应用程序中实例化模型，您可以从内置的加速优化中受益。此选项还消耗更少的计算资源，这对环境总是有利的！ 🌎 

  有关如何以编程方式访问推理提供程序的更多信息，请参阅此[page](https://huggingface.co/docs/huggingface_hub/how-to-inference)。

* [⟦T1⟧](https://docs.python-requests.org/en/master/) 对于从您的应用程序调用第三方 API 非常有用。

* [⟦T2⟧](https://github.com/huggingface/datasets) 允许您从应用程序内的 Hub 获取或显示任何数据集。

* [⟦T3⟧](https://github.com/gradio-app/gradio)。您可以选择使用 [⟦T4⟧ in the ⟦T5⟧ file](spaces-config-reference) 来要求特定版本。

* 常见的 Debian 软件包，例如 `ffmpeg`、`cmake`、`libsm6` 等。

## 添加您自己的依赖项如果您需要其他 Python 包来运行您的应用程序，请将它们添加到存储库根目录的 **requirements.txt** 文件中。 Spaces 运行时引擎将动态创建自定义环境。您还可以添加一个 **pre-requirements.txt** 文件，描述将在主要依赖项之前安装的依赖项。如果您需要更新 pip 本身，它会很有用。

还支持 Debian 依赖项。在存储库的根目录中添加 **packages.txt** 文件，并在其中列出所有依赖项。每个依赖项都应该位于单独的行上，并且每行都将由 `apt-get install` 读取并安装。

### 组织博客文章
https://huggingface.co/docs/hub/enterprise-blog-articles.md