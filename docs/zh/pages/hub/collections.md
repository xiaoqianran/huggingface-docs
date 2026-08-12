<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 收藏

使用集合在专用页面上对中心的存储库（模型、数据集、空间和论文）进行分组。

![Collection page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-intro.webp)

集合有很多用例：

- 在您的个人或组织资料中突出显示特定存储库。
- 将您的个人资料访问者的密钥存储库与其他存储库分开。
- 展示并分享一个完整的项目及其论文、数据集、模型和空间。
- 将您在 Hub 上找到的内容按类别添加为书签。
- 有一个专门的页面来与他人分享精选的内容。
- 门控一组模型/数据集（团队和企业）

这只是可能用途的列表，但请记住，集合只是一种对事物进行分组的方式，因此请以最适合您的用例的方式使用它们。

## 创建一个新集合

创建集合有多种方法：

- 对于个人收藏：使用登录主页上的 **+ 新建** 按钮 (1)。
- 对于组织集合：使用组织页面 (2) 上的 **+ 新建** 按钮。

![New collection](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-new.webp)从存储库页面添加第一个项目时，也可以动态创建集合，从下拉菜单中选择 **+ 创建新集合**。
您需要输入要创建的集合的标题和简短说明。

## 将项目添加到集合中

有 2 种方法可以将项目添加到集合中：

- 从任何存储库页面：使用任何存储库页面上可用的上下文菜单，然后选择 **添加到集合** 将其添加到集合 (1)。
- 从集合页面：如果您知道要添加的存储库的名称，请使用右侧菜单 (2) 中的 **+ 添加到集合** 选项。

![Add items to collections](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-add.webp)

可以将外部存储库添加到您的集合中，而不仅仅是您自己的集合中。

## 合作收藏

组织馆藏是共同构建馆藏的好方法。具有只读访问权限的成员可以查看集合，但只有具有写入（或更高）组织权限的成员才能创建集合或添加、编辑和删除项目。
使用**历史记录功能**来跟踪谁编辑了该集合。

![Collection history](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-history.webp)

## 收集选项

### 集合可见性

![Collections on profiles](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-profile.webp)**公共**集合显示在您的个人资料或组织页面的顶部，任何人都可以查看。每个集合中的前 3 项在集合预览 (1) 中直接可见。要查看更多内容，用户必须点击进入收藏页面。

如果您不希望通过其 URL 访问您的收藏（它将不会显示在您的个人资料/组织页面上），请将您的收藏设置为**私有**。对于组织而言，私人收藏仅可供该组织的成员使用。

### Gating Group Collections（团队和企业）

您可以使用集合来[gate](https://huggingface.co/docs/hub/en/models-gated)属于它的所有模型/数据集，从而允许您一次授予（或拒绝）对所有模型/数据集的访问权限。

此功能是为[Team & Enterprise](https://huggingface.co/docs/hub/en/enterprise)订户保留的：有关门控组集合的更多信息可以在[our dedicated doc](https://huggingface.co/docs/hub/en/enterprise-gating-group-collections)中找到。

### 将集合分配给资源组（团队和企业）组织集合可以分配给[Resource Group](./security-resource-groups)来控制哪些成员可以访问它们，就像存储库一样。您可以在创建组织集合时选择一个资源组，并且具有适当权限的成员可以稍后从集合的菜单将集合移动到不同的资源组。属于资源组的私有集合仅对该组的成员可见。

此功能专为 [Team & Enterprise](https://huggingface.co/docs/hub/en/enterprise) 订户保留。

### 订购您的收藏品及其物品

您可以使用集合列表（位于集合页面左侧）中的拖放手柄来更改集合的顺序 (1)。前两个集合将直接在您的个人资料/组织页面上可见。

您还可以通过拖动每个项目旁边的手柄 (2) 对集合中的存储库进行排序。

![Collections sort](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-sort.webp)

### 从集合中删除项目

要从集合中删除项目，请单击将鼠标悬停在项目上时右侧显示的菜单中的垃圾桶图标 (1)。
要删除整个集合，请单击右侧菜单 (2) 上的删除 - 您需要确认此操作。

![Collection delete](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-delete.webp)

### 为集合的项目添加注释可以向集合中的任何项目添加注释，以为其提供更多上下文（对于其他人，或者作为对自己的提醒）。当您将鼠标悬停在某个项目上时，可以通过单击铅笔图标来添加注释。注释是纯文本，不支持 Markdown，以保持简洁。注释中的 URL 会转换为可点击的链接。

![Collection note](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-note.webp)

### 将图像添加到集合项

同样，您可以将图像附加到集合项目。这对于展示模型的输出、数据集的内容、附加上下文信息图等非常有用。

要开始将图像添加到您的收藏中，您可以单击项目的上下文菜单中的图像图标。当您将鼠标悬停在某个项目上时，会显示该菜单。

![Collection image icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-button.webp)

然后，通过从计算机拖放图像来添加图像。您还可以单击灰色区域从计算机的文件系统中选择图像文件。

![Collection image drop zone with images](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-gallery.webp)

您可以通过拖放图像来重新排序图像。单击图像将以全屏模式打开它。

![Collection image viewer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-viewer.webp)

## 您对收藏的反馈我们正在努力改进集合，因此如果您有任何错误、疑问或希望添加新功能，请在 [dedicated discussion](https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/12) 中发布消息。

### TF-Keras（旧版）
https://huggingface.co/docs/hub/tf-keras.md

## TF-Keras（旧版）

`tf-keras` 是 Keras 2.x 版本的名称。它现在作为单独的 GitHub 存储库 [here](https://github.com/keras-team/tf-keras) 托管。尽管它是一个遗留框架，但 Hub 上仍然托管着 [4.5k+ models](https://huggingface.co/models?library=tf-keras&sort=trending)。这些模型可以使用 `huggingface_hub` 库加载。您**必须**在计算机上安装 `tf-keras` 或 `keras<3.x`。

如果您对 Keras 3.x 支持感兴趣，请查看 [this guide](./keras)。

安装完成后，您只需使用`from_pretrained_keras`方法从Hub加载模型即可。了解更多关于 `from_pretrained_keras` [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_keras) 的信息。

```py
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("keras-io/mobile-vit-xxs")
prediction = model.predict(image)
prediction = tf.squeeze(tf.round(prediction))
print(f'The image is a {classes[(np.argmax(prediction))]}!')

<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# The image is a sunflower!
```

您还可以在 Hub 上托管您的 `tf-keras` 模型。但是，请记住，`tf-keras` 是一个遗留框架。为了达到最大数量的用户，我们建议使用 Keras 3.x 创建模型并如上所述在本机共享。有关上传`tf-keras`模型的更多详细信息，请查看[⟦T11⟧ documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.push_to_hub_keras)。

```py
from huggingface_hub import push_to_hub_keras

push_to_hub_keras(model,
    "your-username/your-model-name",
    "your-tensorboard-log-directory",
    tags = ["object-detection", "some_other_tag"],
    **model_save_kwargs,
)
```

## 其他资源

- [GitHub repo](https://github.com/keras-team/tf-keras)
* 博文[Putting Keras on 🤗 Hub for Collaborative Training and Reproducibility](https://merveenoyan.medium.com/putting-keras-on-hub-for-collaborative-training-and-reproducibility-9018301de877)（2022 年 4 月）

### 快速入门
https://huggingface.co/docs/hub/jobs-quickstart.md