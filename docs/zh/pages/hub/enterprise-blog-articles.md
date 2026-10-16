<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 组织博客文章

> [!警告]
> 此功能是团队和企业计划的一部分。

博客文章允许团队和企业组织直接在您的组织资料下发布长格式内容，使您能够与更广泛的社区共享模型版本、研究更新和公告。

## 作为一个组织进行出版

在 [huggingface.co/new-blog](https://huggingface.co/new-blog) 创建新文章时，从下拉列表中选择您的组织，以组织而非个人的身份发布。发布后，文章将显示在您组织的个人资料页面上。

## 博客主页上的特色内容

符合条件的组织发表的文章将自动出现在[huggingface.co/blog](https://huggingface.co/blog)的主要部分；其他文章出现在社区部分。一篇文章在以下情况下会被推荐：

- 该组织采用付费团队或企业计划，**至少 30 个席位**。
- 文章有**缩略图**。没有缩略图发布的文章不会被推荐。

文章首次发表时会检查资格。之后添加缩略图不会追溯该文章。为了保持主要部分的多样性，每个组织可以每两周一次发布一篇专题文章。发布新的精选文章会取消该窗口中该组织的其他精选文章的特色。

## 权限

要在组织命名空间下发布博客文章，成员需要组织级别的 `write` 或 `admin` 角色。有关角色的更多详细信息，请参阅[Access Control in Organizations](./organizations-security)。

> [!注意]
> 在企业计划及以上版本中，组织管理员可以通过资源组 [Granular feature access](./security-resource-groups#granular-feature-access) 设置限制谁可以撰写和发布组织博客文章。

## 链接到模型和数据集

提及同一组织拥有的模型或数据集的文章将自动显示在该存储库的页面上。详情请参阅[Linking to Models and Datasets](./blog-articles#linking-to-models-and-datasets)。

有关在 Hub 上创作博客文章的一般信息，请参阅 [Blog Articles](./blog-articles)。

### 执行SQL操作
https://huggingface.co/docs/hub/datasets-duckdb-sql.md