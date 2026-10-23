<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 自定义 Python 空间

> [!提示]
> Spaces 现在支持任意 Dockerfile，因此您可以使用 [Docker Spaces](./spaces-sdks-docker) 直接托管任何 Python 应用程序。

虽然不是官方工作流程，但您可以通过选择 Gradio 作为您的 SDK 并在端口 `7860` 上提供前端服务，在 Spaces 中运行您自己的 Python + 界面堆栈。有关示例，请参阅[templates](https://huggingface.co/templates#spaces)。

空间在 iframe 中提供，默认情况下限制链接在父页面中打开。最简单的解决方案是在新窗口中打开它们：

```HTML
<a href="https://hf.space" rel="noopener" target="_blank">Spaces</a>
```

通常，使用Gradio库界面时，Spaces的高度会自动调整。但是，如果您在 Gradio SDK 中提供自己的前端并且内容高度大于视口，则需要添加 [iFrame Resizer script](https://cdnjs.com/libraries/iframe-resizer)，以便内容在 iframe 中可滚动：

```HTML
<script src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.2/iframeResizer.contentWindow.min.js"></script>
```
作为示例，以下是带有和不带有脚本的相同空间：
- https://huggingface.co/spaces/ronvolutional/http-server
- https://huggingface.co/spaces/ronvolutional/iframe-test

### 在 Hugging Face 中使用 fastai
https://huggingface.co/docs/hub/fastai.md