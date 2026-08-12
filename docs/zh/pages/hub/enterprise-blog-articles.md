<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 组织博客文章

> [!警告]
> 此功能是团队和企业计划的一部分。

博客文章允许团队和企业组织直接在您的组织资料下发布长格式内容，使您能够与更广泛的社区共享模型版本、研究更新和公告。

## 作为一个组织进行出版

在 [huggingface.co/new-blog](https://huggingface.co/new-blog) 创建新文章时，从下拉列表中选择您的组织，以组织而非个人的身份发布。发布后，文章将显示在您组织的个人资料页面上。

## 权限

要在组织命名空间下发布博客文章，成员需要组织级别的 `write` 或 `admin` 角色。有关角色的更多详细信息，请参阅[Access Control in Organizations](./organizations-security)。

> [!注意]
> 博客文章权限当前与组织级别角色相关联，无法使用 [Resource Groups](./security-resource-groups) 确定范围。资源组仅控制对存储库（模型、数据集和空间）的访问，而不控制对博客文章的访问。

## 链接到模型和数据集

提及同一组织拥有的模型或数据集的文章将自动显示在该存储库的页面上。详情请参见[Linking to Models and Datasets](./blog-articles#linking-to-models-and-datasets)。有关在 Hub 上创作博客文章的一般信息，请参阅 [Blog Articles](./blog-articles)。

### 执行SQL操作
https://huggingface.co/docs/hub/datasets-duckdb-sql.md