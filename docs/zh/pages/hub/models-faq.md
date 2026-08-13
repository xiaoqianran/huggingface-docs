<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型常见问题解答

## 我如何查看使用什么数据集来训练模型？

由上传模型的人来包含训练信息！用户可以[specify](./model-cards#specifying-a-dataset)用于训练模型的数据集。如果模型使用的数据集位于 Hub 上，则上传者可能已将它们包含在 [model card's metadata](https://huggingface.co/Jiva/xlm-roberta-large-it-mnli/blob/main/README.md#L7-L9) 中。在这种情况下，数据集将与模型页面右侧的方便卡片链接：

## 如何查看模型的实际运行示例？

模型可以具有推理小部件，让您可以在浏览器中尝试模型！推理小部件易于配置，并且有许多不同的选项可供您使用。访问[Widgets documentation](models-widgets)了解更多信息。

Hugging Face Hub 也是 Spaces 的所在地，这是用于展示模型的交互式演示。如果模型有任何与其关联的空间，您会发现它们在模型页面上链接如下：

空间是展示您制作的模型或探索使用现有模型的新方法的好方法！访问[Spaces documentation](./spaces)学习如何制作自己的。

## 社区博客文章是否链接到模型？是的 - 如果您发布的 [blog article](./blog-articles) 提及您拥有的模型（作为用户或组织），它将自动显示在模型页面侧边栏中。这同样适用于数据集页面。详情请参阅[Linking to Models and Datasets](./blog-articles#linking-to-models-and-datasets)。

## 如何上传模型的更新/新版本？

可以通过将新提交推送到模型的存储库来发布对已发布模型的更新。为此，请执行与上传初始模型相同的过程。您以前的模型版本将保留在存储库的提交历史记录中，因此您仍然可以从特定的 git 提交或标记下载以前的模型版本，或者根据需要恢复到以前的版本。

## 如果我在不同数据集上训练的模型有不同的检查点怎么办？按照惯例，每个模型存储库应包含一个检查点。您应该将在不同数据集上训练的任何新检查点上传到新模型存储库中的中心。您可以通过使用 [model card's metadata](./model-cards) 中的 `tags` 键中指定的标签、使用 [Collections](./collections) 将不同的相关存储库分组在一起或通过在模型卡中链接到它们来将模型链接在一起。例如，[akiyamasho/AnimeBackgroundGAN-Shinkai](https://huggingface.co/akiyamasho/AnimeBackgroundGAN-Shinkai#other-pre-trained-model-versions) 模型在“其他预训练模型版本”* 下引用模型卡中的其他检查点。

## 我可以将我的模型链接到 arXiv 上的论文吗？

如果模型卡包含 arXiv 上论文的链接，Hugging Face Hub 将提取 arXiv ID 并将其包含在格式为 `arxiv:<PAPER ID>` 的模型标签中。单击该标签将使您：

* 访问纸质页面
* 过滤 Hub 上引用同一论文的其他模型。

了解有关纸质页面的更多信息[here](./paper-pages)。

### 纸页
https://huggingface.co/docs/hub/paper-pages.md