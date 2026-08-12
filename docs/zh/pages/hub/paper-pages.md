<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 页数

纸质页面允许人们查找与论文相关的工件，例如模型、数据集和应用程序/演示（空间）。纸质页面还使社区能够讨论该纸质。

## 将论文链接到模型、数据集或空间

如果存储库卡 (`README.md`) 包含指向 Paper 页面的链接（HF 或 Arxiv 摘要/PDF），Hugging Face Hub 将提取 arXiv ID 并将其包含在存储库的标签中。单击 arxiv 标签将使您：

* 访问论文页面。
* 过滤 Hub 上引用同一论文的其他模型或数据集。

## 声明论文作者身份

该中心将尝试根据用户的电子邮件自动将纸张与用户匹配。 

如果您的论文未链接到您的帐户，您可以在相应的论文页面中单击您的姓名，然后单击“声明作者身份”。这将自动重定向到您的纸张设置，您可以在其中确认请求。管理团队将很快验证您的请求。确认后，Paper 页面将显示为已验证。

如果您还没有任何关于 Hugging Face 的论文，您可以按照 [here](#can-i-have-a-paper-page-even-if-i-have-no-modeldatasetspace) 的说明对第一篇论文进行索引。一旦可用，您就可以声明作者身份。

## 常见问题### 我可以控制在我的个人资料中显示哪些 Paper 页面吗？

是的！您可以在[settings](https://huggingface.co/settings/papers)中访问您的论文，在那里您将看到经过验证的论文列表。在那里，您可以单击“在个人资料上显示”复选框以在您的个人资料中隐藏/显示它。 

### 你支持 ACL 选集吗？

我们从 Arxiv 开始，因为它占 Hugging Face 用户在其存储库中有机链接的论文 URL 的 95%。我们将检查它是如何演变的，并可能在未来扩展到其他纸质主机。

### 即使我没有模型/数据集/空间，我也可以拥有纸质页面吗？

是的。您可以前往[the main Papers page](https://huggingface.co/papers)，点击搜索并输入论文名称或完整的Arxiv id。如果该论文不存在，您将可以选择为其建立索引。您也可以访问页面 `hf.co/papers/xxxx.yyyyy` 并替换为您想要索引的论文的 arxiv id。

### 使用 SetFit 和拥抱脸部
https://huggingface.co/docs/hub/setfit.md