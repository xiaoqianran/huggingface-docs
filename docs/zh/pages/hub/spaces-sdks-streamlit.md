<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 流光空间

**Streamlit** 让用户可以自由地以“反应式”方式使用 Python 构建功能齐全的 Web 应用程序。每次应用程序状态发生变化时，您的代码都会重新运行。 Streamlit 也非常适合数据可视化，并支持多个图表库，例如 Bokeh、Plotly 和 Altair。阅读这篇关于在 Spaces 中构建和托管 Streamlit 应用程序的 [blog post](https://huggingface.co/blog/streamlit-spaces)。

当 [creating a new Space](https://huggingface.co/new-space) 选择 **Streamlit** 作为 SDK 时，将通过在 `README.md` 文件的 YAML 块中将 `sdk` 属性设置为 `streamlit`，使用最新版本的 Streamlit 初始化您的空间。如果您想更改 Streamlit 版本，可以编辑 `sdk_version` 属性。

要在空间中使用 Streamlit，请在通过 [**New Space** form](https://huggingface.co/new-space) 创建空间时选择 **Streamlit** 作为 SDK。这将创建一个带有 `README.md` 的存储库，其中在 YAML 配置块中包含以下属性：

```yaml
sdk: streamlit
sdk_version: 1.25.0 # The latest supported version
```

您可以编辑 `sdk_version`，但请注意，当您使用不受支持的 Streamlit 版本时可能会出现问题。并非所有 Streamlit 版本均受支持，因此请参阅 [reference section](./spaces-config-reference) 查看可用的版本。

有关 Streamlit 的深入信息，请参阅[Streamlit documentation](https://docs.streamlit.io/)。> [!警告]
> Streamlit Spaces 仅允许使用端口 8501（默认端口）。因此，如果您为空间提供 `config.toml` 文件，请确保默认端口未被覆盖。

## 您的第一个 Streamlit 空间：热狗分类器

在以下部分中，您将学习创建空间、配置空间以及将代码部署到空间的基础知识。我们将使用 Streamlit 创建一个 **热狗分类器** 空间，用于演示 [julien-c/hotdog-not-hotdog](https://huggingface.co/julien-c/hotdog-not-hotdog) 模型，该模型可以检测给定图片是否包含热狗 🌭

您可以在[NimaBoscarino/hotdog-streamlit](https://huggingface.co/spaces/NimaBoscarino/hotdog-streamlit)找到该内容的完整版本。

## 创建一个新的 Streamlit 空间

我们将从 [creating a brand new Space](https://huggingface.co/new-space) 开始并选择 **Streamlit** 作为我们的 SDK。 Hugging Face Spaces 是 Git 存储库，这意味着您可以通过推送提交来增量（协作）地处理您的空间。在继续之前，请查看 [Getting Started with Repositories](./repositories-getting-started) 指南，了解如何创建和编辑文件。

## 添加依赖

对于 **Hot Dog Classifier**，我们将使用 [🤗 Transformers pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) 来使用模型，因此我们需要首先安装一些依赖项。这可以通过在我们的存储库中创建 **requirements.txt** 文件并向其添加以下依赖项来完成：

```
transformers
torch
```Spaces 运行时将处理依赖项的安装！

## 创建 Streamlit 应用程序

要创建 Streamlit 应用程序，请在存储库中创建一个名为 **app.py** 的新文件，并添加以下代码：

```python
import streamlit as st
from transformers import pipeline
from PIL import Image

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")

st.title("Hot Dog? Or Not?")

file_name = st.file_uploader("Upload a hot dog candidate image")

if file_name is not None:
    col1, col2 = st.columns(2)

    image = Image.open(file_name)
    col1.image(image, use_column_width=True)
    predictions = pipeline(image)

    col2.header("Probabilities")
    for p in predictions:
        col2.subheader(f"{ p['label'] }: { round(p['score'] * 100, 1)}%")
```

此 Python 脚本使用 [🤗 Transformers pipeline](https://huggingface.co/docs/transformers/pipeline_tutorial) 加载 [julien-c/hotdog-not-hotdog](https://huggingface.co/julien-c/hotdog-not-hotdog) 模型，该模型由 Streamlit 接口使用。 Streamlit 应用程序将要求您上传图像，然后将其分类为“热狗”或“非热狗”。将代码保存到 **app.py** 文件后，请访问 **App** 选项卡以查看您的应用程序的运行情况！

## 在其他网页上嵌入 Streamlit Spaces

您可以使用 HTML `<iframe>` 标签将 Streamlit Space 作为内联框架嵌入到其他网页上。只需包含您的空间的 URL，以 `.hf.space` 后缀结尾。要查找您的空间的 URL，您可以使用空间选项中的“嵌入此空间”按钮。

例如，上面的演示可以使用以下标签嵌入到这些文档中：

```
<iframe
  src="https://NimaBoscarino-hotdog-streamlit.hf.space?embed=true"
  title="My awesome Streamlit Space"
></iframe>
```

<iframe
  src="https://NimaBoscarino-hotdog-streamlit.hf.space?embed=true"
  frameborder="0"
  height="364"
  title="Streamlit app"
  class="container p-0 grow space-iframe"
  allow="accelerometer; ambient-light-sensor; autoplay; battery; camera; document-domain; encrypted-media; fullscreen; geolocation; gyroscope; layout-animations; legacy-image-formats; magnetometer; microphone; midi; oversized-images; payment; picture-in-picture; publickey-credentials-get; sync-xhr; usb; vr ; wake-lock; xr-spatial-tracking"
  sandbox="allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox allow-same-origin allow-scripts allow-downloads"
>

请注意，我们已将 `?embed=true` 添加到 URL，这会激活 Streamlit 应用程序的嵌入模式，删除一些间隔符和页脚以实现超薄嵌入。

## 使用自动调整大小的 IFrame 嵌入 Streamlit SpacesStreamlit 自 [1.17.0](https://docs.streamlit.io/library/changelog#version-1170) 起支持自动调整 iframe 大小，以便自动调整父 iframe 的大小以适应嵌入式 Streamlit 应用程序的内容量。

它依赖于[⟦T15⟧](https://github.com/davidjbradshaw/iframe-resizer)库，您需要为此添加几行代码，如下例所示

- `id` 设置为 `<iframe />`，用于指定自动调整大小目标。
- `iFrame Resizer` 通过`script` 标签加载。
- 使用目标`iframe`元素的ID调用`iFrameResize()`函数，使其大小自动改变。

我们可以将选项传递给`iFrameResize()`的第一个参数。详情请参阅[the document](https://github.com/davidjbradshaw/iframe-resizer/blob/master/docs/parent_page/options.md)。

```html
<iframe
	id="your-iframe-id"
	src="https://<space-subdomain>.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>
<script src="https://cdn.jsdelivr.net/npm/iframe-resizer@4.3.4/js/iframeResizer.min.js"></script>
<script>
  iFrameResize({}, "#your-iframe-id")
</script>
```

此外，您还可以查看[our documentation](./spaces-embed)。

### 存储桶
https://huggingface.co/docs/hub/storage-buckets.md