<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 渐变空间

**Gradio** 提供了一个简单直观的界面，用于从输入列表运行模型并以图像、音频、3D 对象等格式显示输出。 Gradio 现在甚至拥有 [Plot output component](https://gradio.app/docs/#o_plot)，用于使用 Matplotlib、Bokeh 和 Plotly 创建数据可视化！有关更多详细信息，请查看 Gradio 团队的 [Getting started](https://gradio.app/getting_started/) 指南。

当 [creating a new Space](https://huggingface.co/new-space) 选择 **Gradio** 作为 SDK 时，将通过在 `README.md` 文件的 YAML 块中将 `sdk` 属性设置为 `gradio`，使用最新版本的 Gradio 初始化您的空间。如果您想更改 Gradio 版本，可以编辑 `sdk_version` 属性。

访问 [Gradio documentation](https://gradio.app/docs/) 了解其所有功能，并查看 [Gradio Guides](https://gradio.app/guides/) 获取一些方便的教程来帮助您入门！

## 您的第一个渐变空间：热狗分类器

在以下部分中，您将学习创建空间、配置空间以及将代码部署到空间的基础知识。我们将创建一个带有 Gradio 的 **热狗分类器** 空间，用于演示 [julien-c/hotdog-not-hotdog](https://huggingface.co/julien-c/hotdog-not-hotdog) 模型，该模型可以检测给定图片是否包含热狗 🌭

您可以在[NimaBoscarino/hotdog-gradio](https://huggingface.co/spaces/NimaBoscarino/hotdog-gradio)找到完整版本。

## 创建一个新的渐变空间我们将从 [creating a brand new Space](https://huggingface.co/new-space) 开始，并选择 **Gradio** 作为我们的 SDK。 Hugging Face Spaces 是 Git 存储库，这意味着您可以通过推送提交来增量（协作）地处理您的空间。在继续之前，请查看 [Getting Started with Repositories](./repositories-getting-started) 指南，了解如何创建和编辑文件。

## 添加依赖

对于 **Hot Dog Classifier**，我们将使用 [🤗 Transformers pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) 来使用模型，因此我们需要首先安装一些依赖项。这可以通过在我们的存储库中创建 **requirements.txt** 文件并向其添加以下依赖项来完成：

```
transformers
torch
```

Spaces 运行时将处理依赖项的安装！

## 创建Gradio界面

要创建 Gradio 应用程序，请在存储库中创建一个名为 **app.py** 的新文件，并添加以下代码：

```python
import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

def predict(input_img):
    predictions = pipeline(input_img)
    return input_img, {p["label"]: p["score"] for p in predictions} 

gradio_app = gr.Interface(
    predict,
    inputs=gr.Image(label="Select hot dog candidate", sources=['upload', 'webcam'], type="pil"),
    outputs=[gr.Image(label="Processed Image"), gr.Label(label="Result", num_top_classes=2)],
    title="Hot Dog? Or Not?",
)

if __name__ == "__main__":
    gradio_app.launch()
```

此 Python 脚本使用 [🤗 Transformers pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) 加载 [julien-c/hotdog-not-hotdog](https://huggingface.co/julien-c/hotdog-not-hotdog) 模型，该模型由 Gradio 接口使用。 Gradio 应用程序会要求您上传图像，然后将其分类为“热狗”或“非热狗”。将代码保存到 **app.py** 文件后，请访问 **App** 选项卡以查看您的应用程序的运行情况！

## 在其他网页上嵌入渐变空间您可以使用 Web 组件或 HTML `<iframe>` 标签将 Gradio Space 嵌入其他网页。查看 [our documentation](./spaces-embed) 或 [Gradio documentation](https://gradio.app/sharing_your_app/#embedding-hosted-spaces) 了解更多详情。

### 如何使用 Okta 配置 SAML SSO
https://huggingface.co/docs/hub/security-sso-okta-saml.md