<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数字对象标识符 (DOI)

Hugging Face Hub 可以为您的模型或数据集生成 DOI。 DOI（数字对象标识符）是唯一标识数字对象的字符串，从文章到图形，包括数据集和模型。 DOI 与对象元数据绑定，包括对象的 URL、版本、创建日期、描述等。它们是研究和学术界普遍接受的数字资源参考；它们类似于书籍的 ISBN。

## 如何生成 DOI？

为此，您必须转到模型或数据集的设置。在 DOI 部分中，应出现一个名为“生成 DOI”的按钮：

要为此模型或数据集生成 DOI，您需要单击此按钮并确认中心上的某些功能将受到限制，并且您的一些信息（您的全名）将传输到我们的合作伙伴 DataCite。生成 DOI 时，您可以选择个性化作者姓名列表，以便将所有贡献者归功于您的模型或数据集。同意这些条款后，您的模型或数据集将获得分配的 DOI，并且新标签应出现在您的模型或数据集标题中，允许您引用它。

## 如果我的模型或数据集发生变化，我可以重新生成新的 DOI 吗？

如果模型或数据集出现新版本，则可以轻松分配新的 DOI，而之前版本的 DOI 就会过时。这使得引用对象的特定版本变得很容易，即使它已经发生了变化。

您只需点击“生成新的 DOI”即可！🎉 将为您的模型或数据集的当前版本分配一个新的 DOI。

## 为什么在我的模型或数据集上进行删除、重命名和更改可见性操作时会出现“被 DOI 锁定”消息？

DOI 使查找有关模型或数据集的信息变得更加容易，并通过永不过期或不会更改的永久链接与全世界共享它们。因此，带有 DOI 的数据集/模型旨在永久保留，并且只能在向我们的支持人员提出请求后才能删除、重命名和更改其可见性（网站 Huggingface.co）

## 进一步阅读

- [Introducing DOI: the Digital Object Identifier to Datasets and Models](https://huggingface.co/blog/introducing-doi)

### 手动配置
https://huggingface.co/docs/hub/datasets-manual-configuration.md