<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 拉取请求和讨论

Hub Pull 请求和讨论允许用户对存储库进行社区贡献。拉取请求和讨论对于所有存储库类型都是相同的。

在较高层面上，目标是构建其他 git 主机（如 GitHub）的 PR 和问题的更简单版本：
- 不涉及分叉：贡献者直接推送到源存储库上的特殊 `ref` 分支。
- 讨论和 PR 之间没有严格的区别：它们本质上是相同的，因此它们显示在相同的列表中。
- 它们针对 ML（即模型/数据集/空间存储库）进行了简化，而不是任意存储库。

_注意，拉取请求和讨论可以从 [repository settings](./repositories-settings#disabling-discussions--pull-requests)_ 启用或禁用

## 列表

通过转到任何存储库中的社区选项卡，您可以查看所有讨论和拉取请求。您还可以过滤以仅查看打开的内容。

## 查看

讨论页面允许您查看不同用户的评论。如果是拉取请求，您可以通过转到“文件已更改”选项卡来查看所有更改。

## 编辑讨论/拉取请求标题如果您打开了 PR 或讨论，是存储库的作者，或者对其具有写入权限，则可以通过单击铅笔按钮来编辑讨论标题。

## 固定讨论/拉取请求

如果您拥有存储库的写入权限，则可以固定讨论和拉取请求。固定讨论出现在所有讨论的顶部。

## 锁定讨论/拉取请求

如果您对存储库具有写入权限，则可以锁定讨论或拉取请求。讨论锁定后，之前的评论仍然可见，并且用户将无法添加新评论。

## 评论编辑和审核

如果您撰写了评论或对存储库具有写入权限，则可以从评论框右上角的上下文菜单编辑评论内容。

编辑评论后，评论上方将出现一个新链接。此链接显示编辑历史记录。 

您还可以隐藏评论。隐藏评论是不可逆的，任何人都无法看到其内容，也无法再对其进行编辑。

另请阅读[moderation](./moderation)，了解如何举报辱骂性评论。

## 我可以在评论和讨论中使用 Markdown 和 LaTeX 吗？是的！您可以使用 Markdown 为评论添加格式。此外，您可以使用 LaTeX 进行数学排版，您的公式将在 Markdown 中解析之前使用 [KaTeX](https://katex.org/) 进行渲染。

对于 LaTeX 方程，您必须使用以下分隔符：
- `$$ ... $$` 显示模式
- `&#92;&#92;(...\\)` 用于内联模式（斜杠和括号之间没有空格）。

## 如何在本地管理 Pull 请求？

假设您的 PR 号是 42。 

```bash
git fetch origin refs/pr/42:pr/42
git checkout pr/42
# Do your changes
git add .
git commit -m "Add your change"
git push origin pr/42:refs/pr/42
```

### 草稿模式

草稿模式是在“高级模式”下从头开始打开新的 Pull 请求时的默认状态。在此状态下，其他贡献者知道您的 Pull 请求正在处理中并且无法合并。当您的分支准备就绪时，只需点击“发布”按钮即可将拉取请求的状态更改为“打开”。请注意，一旦发布，您将无法返回草稿模式。 

## 删除 Pull 请求引用

当 Pull 请求关闭或合并时，您可以删除其关联的 git ref （存储 PR 提交的分支）以释放存储空间。关闭或合并 PR 后，您将在讨论底部看到一条通知，显示通过删除引用可以释放的估计存储空间。单击“删除引用”按钮可永久删除 PR 的 git 引用并回收存储。

> [!提示]
> 当主分支被压缩并且文件稍后被删除时，这特别有用。即使这些文件不是由 PR 本身添加的，它们也会保留在 PR 分支历史记录中，从而占用可以释放的存储空间。

> [!警告]
> 删除 PR 参考是不可逆的。删除后，您将无法再在本地获取或签出 PR 的提交。

## Pull请求高级用法

### git 存储库中的更改存储在哪里？

我们的 Pull 请求不使用分叉和分支，而是使用名为 `refs` 的自定义“分支”，它们直接存储在源存储库上。

[Git References](https://git-scm.com/book/en/v2/Git-Internals-Git-References) 是 git 的内部机制，它已经存储了标签和分支。

使用自定义引用（例如 `refs/pr/42`）而不是分支的优点是，克隆存储库的人（包括存储库“所有者”）不会（默认情况下）获取它们，但仍然可以根据需要获取它们。

### 获取所有 Pull 请求：适用于 git 魔术师🧙‍♀️您可以调整本地 **refspec** 以获取所有 Pull 请求：

1. 获取

```bash
git fetch origin refs/pr/*:refs/remotes/origin/pr/*
```

2. 创建一个本地分支来跟踪 ref

```bash
git checkout pr/{PR_NUMBER}
# for example: git checkout pr/42
```

3. 如果您进行本地更改，请推送到 PR 参考：

```bash
git push origin pr/{PR_NUMBER}:refs/pr/{PR_NUMBER}
# for example: git push origin pr/42:refs/pr/42
```

### 空间设置
https://huggingface.co/docs/hub/spaces-settings.md