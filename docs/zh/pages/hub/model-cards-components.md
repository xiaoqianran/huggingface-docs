<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型卡组件

**模型卡组件**是特殊元素，您可以将其直接注入到模型卡降价中，以在模型页面中显示强大的自定义组件。这些组件由我们编写，欢迎分享关于[this discussion](https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/17)中新模型卡组件的想法。

## 图库组件

`<Gallery />` 组件可在模型卡中使用来展示生成的图像和视频。

### 如何使用它？

1. 更新您的模型卡[widget metadata](/docs/hub/models-widgets-examples#text-to-image)以添加您想要展示的媒体。 

```yaml
widget:
  - text: a girl wandering through the forest
    output:
      url: images/6CD03C101B7F6545EB60E9F48D60B8B3C2D31D42D20F8B7B9B149DD0C646C0C2.jpeg
  - text: a tiny witch child
    output:
      url: images/7B482E1FDB39DA5A102B9CD041F4A2902A8395B3835105C736C5AD9C1D905157.jpeg
  - text: an artist leaning over to draw something
    output:
      url: images/7CCEA11F1B74C8D8992C47C1C5DEA9BD6F75940B380E9E6EC7D01D85863AF718.jpeg
```

2. 将 `<Gallery />` 组件添加到您的卡中。 `<Gallery />` 组件将使用小部件元数据来显示带有每个关联提示的媒体。
```md

<Gallery />

## Model description

A very classic hand drawn cartoon style.

```

查看结果[here](https://huggingface.co/alvdansen/littletinies#little-tinies)。

> 提示：通过 GUI 编辑器对卡组件的支持即将推出...

### 模型常见问题解答
https://huggingface.co/docs/hub/models-faq.md