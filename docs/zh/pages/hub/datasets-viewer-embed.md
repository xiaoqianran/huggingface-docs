<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在网页中嵌入数据集查看器

您可以使用 iframe 将数据集查看器嵌入到您自己的网页中。

要使用的 URL 为 `https://huggingface.co/datasets/<namespace>/<dataset-name>/embed/viewer`，其中 `<namespace>` 是数据集的所有者（用户或组织），`<dataset-name>` 是数据集的名称。您还可以传递其他参数，例如子集、拆分、过滤器、搜索或选定行。

例如，以下 iframe 嵌入来自 `nyu-mll` 组织的 `glue` 数据集的数据集查看器：

```html
<iframe
  src="https://huggingface.co/datasets/nyu-mll/glue/embed/viewer"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
```

您还可以直接从数据集查看器界面获取嵌入代码。单击数据集查看器右上角的`Embed`按钮：

它将打开一个包含 iframe 代码的模式，您可以将其复制并粘贴到网页中：

## 参数

数据集查看器页面的所有参数也可以通过将它们添加到 iframe URL 来传递到嵌入式查看器（过滤器、搜索、特定分割等）。例如，要显示 `nyu-mll/glue` 数据集的 `rte` 子集的 `test` 分割中的 `mangrove` 搜索结果，您可以使用以下 URL：

```html
<iframe
  src="https://huggingface.co/datasets/nyu-mll/glue/embed/viewer/rte/split?search=mangrove"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
```

您可以通过执行搜索直接从数据集查看器界面获取此代码，单击`⋮`按钮，然后单击`Embed`：它将打开一个包含 iframe 代码的模式，您可以将其复制并粘贴到网页中：

## 示例

嵌入式数据集查看器在多种机器学习工具和平台中用于显示数据集。这里有几个例子。 

如果你想出现在这个版块，请打开[pull request](https://github.com/huggingface/hub-docs/blob/main/docs/hub/datasets-viewer-embed.md)！

### 工具：ZenML

[⟦T14⟧](https://huggingface.co/htahir1) 分享了 [blog post](https://www.zenml.io/blog/embedding-huggingface-datasets-visualizations-with-zenml)，展示了如何使用 [ZenML](https://huggingface.co/zenml) 与数据集查看器集成来可视化 ZenML 管道中的 Hugging Face 数据集。

### 工具：Metaflow + Outerbounds

[⟦T15⟧](https://huggingface.co/eddie-OB) 在 [demo video](https://www.linkedin.com/posts/eddie-mattia_the-team-at-hugging-facerecently-released-activity-7219416449084272641-swIu) 中展示了如何在 [Outerbounds](https://huggingface.co/outerbounds) 上的 Metaflow 卡中包含数据集查看器。

### 工具：AutoTrain

[⟦T16⟧](https://huggingface.co/abhishek) 展示了数据集查看器如何集成到 [demo video](https://x.com/abhi1thakur/status/1813892464144798171) 中的 [AutoTrain](https://huggingface.co/autotrain)。

### 数据集：羊驼风格数据集库

[⟦T17⟧](https://huggingface.co/davanstrien) 在 [space](https://huggingface.co/spaces/davanstrien/collection_dataset_viewer) 中展示了 [collection of Alpaca-style datasets](https://huggingface.co/collections/librarian-bots/alpaca-style-datasets-66964d3e490f463859002588)。

### 数据集：Docmatix

[⟦T18⟧](https://huggingface.co/andito)在[blog post](https://huggingface.co/blog/docmatix)中使用嵌入式查看器，宣布发布[Docmatix](https://huggingface.co/datasets/HuggingFaceM4/Docmatix)，这是一个用于文档视觉问答（DocVQA）的巨大数据集。

### App：Masader - 阿拉伯语 NLP 数据目录

[⟦T19⟧](https://huggingface.co/Zaid) [showcases](https://x.com/zaidalyafeai/status/1815365207775932576) [Masader - the Arabic NLP data catalogue0](https://arbml.github.io/masader//) 中的数据集查看器。

### 虎斑在空间
https://huggingface.co/docs/hub/spaces-sdks-docker-tabby.md