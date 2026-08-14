<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 将您的空间嵌入另一个网站

一旦您的空间启动并运行，您可能希望将其嵌入到网站或博客中。 
嵌入或共享您的空间是一种很好的方式，可以让您的观众与您的作品和演示进行互动，而无需他们进行任何设置。
要嵌入空间，其可见性需要**公开**或**受保护**。受保护的空间在 Hub 上保持其源代码的私密性，同时通过其嵌入 URL（以及 [custom domains](./spaces-custom-domain)，配置后）保持公开访问。更多详情请参见[Space visibility](./spaces-overview#space-visibility)。

## 直接网址

空间分配有一个唯一的 URL，您可以使用它来共享您的空间或将其嵌入网站中。

该 URL 的格式为：`"https://<space-subdomain>.hf.space"`。例如，空间[NimaBoscarino/hotdog-gradio](https://huggingface.co/spaces/NimaBoscarino/hotdog-gradio)对应的URL为`"https://nimaboscarino-hotdog-gradio.hf.space"`。子域是唯一的，并且仅在您移动或重命名空间时才会更改。

您的空间始终由该子域的根目录提供服务。

您可以找到空间 URL 以及如何直接从选项菜单嵌入它的示例片段：

## 嵌入 IFrame

空间的默认嵌入方法是使用 IFrame。在要嵌入空间的 HTML 位置中添加以下元素：

```html
<iframe
    src="https://<space-subdomain>.hf.space"
    frameborder="0"
    width="850"
    height="450"
></iframe>
```

例如使用 [NimaBoscarino/hotdog-gradio](https://huggingface.co/spaces/NimaBoscarino/hotdog-gradio) 空间：## 使用 WebComponent 嵌入

如果您要嵌入的空间是基于 Gradio 的，您可以使用 Web 组件来嵌入您的空间。 WebComponent 比 IFrame 更快，并且会自动调整到您的网页，因此您无需为元素配置 `width` 或 `height`。
首先，您需要通过将以下脚本添加到 HTML 中来导入与 Space 中的 Gradio 版本相对应的 Gradio JS 库。

然后，在要嵌入空间的位置添加一个 `gradio-app` 元素。
```html
<gradio-app src="https://<space-subdomain>.hf.space"></gradio-app>
```

查看[Gradio documentation](https://www.gradio.app/guides/sharing-your-app#embedding-hosted-spaces)了解更多详情。

### 数据集卡
https://huggingface.co/docs/hub/datasets-cards.md