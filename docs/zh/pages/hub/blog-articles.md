<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 博客文章

博客文章可让您直接在 Hub 上发布长篇内容 - 模型发布、研究更新、教程和公告 - 并与更广泛的社区分享。文章可以在您的个人命名空间或您所属的[organization](./organizations)下发布。

## 谁可以发布博客文章

要在**您的个人命名空间**下发布，您需要一封确认的电子邮件，并且必须至少满足以下条件之一：

- 您有有效的 [PRO](./pro) 订阅。
- 您是 [Team or Enterprise](https://huggingface.co/enterprise) 组织的成员（在该组织中具有 `write` 或 `admin` 角色）。

托管用户（通过企业 IdP 配置的用户）无法在其个人命名空间下发布。

要在**组织命名空间**下发布，两者都必须包含：

- 该组织处于[Team or Enterprise](https://huggingface.co/enterprise)计划中。
- 您在该组织中具有 `write` 或 `admin` 角色。有关角色的更多信息，请参阅[Access Control in Organizations](./organizations-security)。

有关组织特定的详细信息，请参阅[Blog Articles for Organizations](./enterprise-blog-articles)。

## 创建博客文章

前往[huggingface.co/new-blog](https://huggingface.co/new-blog)开始一篇新文章。您可以使用 Markdown 进行编写、嵌入媒体以及 Hub 上托管的参考模型、数据集和空间。

创建文章时，从下拉列表中选择应在其下发布的命名空间：- **您的用户名** — 文章出现在您的用户个人资料中。
- **组织** — 文章出现在该组织的个人资料页面上。

## 内容指南

发表的文章应属于以下两类之一：

- **探索人工智能科学或工程概念。**例如，[Uncensor any LLM with abliteration](https://huggingface.co/blog/mlabonne/abliteration)和[KV Caching Explained: Optimizing Transformer Inference Efficiency](https://huggingface.co/blog/not-lain/kv-caching)。
- **宣布发布开源工件**，例如模型、数据集或工具。例如，[Welcome NVIDIA Cosmos 3](https://huggingface.co/blog/nvidia/cosmos-3-for-physical-ai)和[OlmoEarth v1.1](https://huggingface.co/blog/allenai/olmoearth-v1-1)。

## 编辑文章

- 在用户命名空间下发布的文章可以由原始作者和文章中列出的任何共同作者进行编辑。
- 具有`write`或`admin`角色的任何组织成员都可以编辑在组织命名空间下发布的文章。

## 链接到模型和数据集

当博客文章提到模型或数据集，并且文章的作者（用户或组织）与存储库的所有者相同时，该文章将自动显示在该模型或数据集页面的侧边栏中 **“提及 [repo-id] 的文章”** 下。最多显示三篇最新的匹配文章。

这使得访问者可以轻松地在存储库本身旁边发现相关的文章、发布公告和研究笔记。如果文章引用 [Collection](./collections)，则该集合中的每个模型和数据集都被视为链接 - 文章将显示在每个成员存储库的页面上（遵循相同的所有权规则）。

### 使用小行星拥抱脸部
https://huggingface.co/docs/hub/asteroid.md