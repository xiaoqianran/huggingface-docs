<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 后续步骤

接下来的部分重点介绍您可能会发现对于充分利用 Hugging Face Hub 上的 Git 存储库有用的功能和其他信息。

## 如何以编程方式管理存储库

Hugging Face 支持通过 [⟦T5⟧ library](https://huggingface.co/docs/huggingface_hub/index) 使用 Python 访问存储库。我们探索过的操作，例如下载存储库和上传文件，都可以通过该库以及其他有用的功能来实现！

如果您更喜欢直接使用 git，请阅读以下部分。

## 了解有关 Git 的更多信息

如果您想继续学习 Git，[this Git tutorial](https://learngitbranching.js.org/) 是一个不错的选择。有关 Git 的更多背景知识，您可以查看 [GitHub's Git Guides](https://github.com/git-guides)。 

## 如何使用分支

为了有效地协作使用 Git 存储库并在不发布过早代码的情况下处理功能，您可以使用**分支**。分支允许您将“正在进行的工作”代码与“生产就绪”代码分开，还有一个额外的好处，即让多人在一个项目上工作，而不会经常与彼此的贡献发生冲突。您可以使用分支来隔离自己分支中的实验，甚至是[adopt team-wide practices for managing branches](https://ericmjl.github.io/essays-on-data-science/workflow/gitflow/)。要了解 Git 分支，您可以尝试 [Learn Git Branching interactive tutorial](https://learngitbranching.js.org/)。

## 使用标签

Git 允许您*标记*提交，以便您可以轻松地记录项目中的里程碑。因此，您可以使用标签来标记 Hub 存储库中的提交！要了解如何使用标签，您可以访问[this DevConnected post](https://devconnected.com/how-to-create-git-tags/)。

除了可以轻松识别存储库历史记录中的重要提交之外，使用 Git 标签还允许您进行 A/B 测试、[clone a repository at a specific tag](https://www.techiedelight.com/clone-specific-tag-with-git/) 等等！ `huggingface_hub` 库还支持使用标签，例如 [downloading files from a specific tagged commit](https://huggingface.co/docs/huggingface_hub/main/en/how-to-downstream#hfhuburl)。

## 如何复制存储库

复制存储库的方法有多种，具体取决于您是否需要保留 Git 历史记录。

### 从集线器复制

单击任何存储库页面右上角的三个点，然后选择“**复制此模型**”、“**复制此数据集**”或“**复制此空间**”。由于使用了[Xet deduplication technology](./xet/deduplication)，该操作几乎是即时的。您将能够选择：* **所有者**：您的帐户或您拥有写入权限的任何组织。
* **存储库名称**：复制的存储库的名称。默认情况下，它在您的命名空间下保留与源相同的名称（例如，复制 `bigscience/bloom-560m` 创建 `your-username/bloom-560m`）。
* **可见性**：您可以选择将重复的存储库设为公开或私有。了解有关私有存储库的更多信息[here](./repositories-settings#private-repositories)。

对于模型和数据集，Git 历史记录被压缩为单个提交。对于 Spaces，会保留完整的 Git 历史记录。公共变量被复制到空间中，但秘密必须手动重新输入。

#### 限制

有些存储库无法复制：
- **门控存储库**（启用访问请求的模型或数据集）。
- 作者已**禁用复制**的存储库。
- **不支持跨区域复制**（例如，存储在美国区域的存储库无法复制到欧盟组织）。

### 以编程方式复制

您还可以使用 `huggingface_hub` 库或 CLI 复制存储库。它们使用与上面的 Hub 按钮相同的服务器端 API（模型和数据集的 Git 历史记录被压缩，为空间保留）。

使用Python：

```python
from huggingface_hub import duplicate_repo

duplicate_repo("bigscience/bloom-560m", private=False)
duplicate_repo("openai/gdpval", repo_type="dataset")
duplicate_repo("multimodalart/dreambooth-training", repo_type="space", private=False)
```

或者使用 CLI：

```bash
hf repos duplicate bigscience/bloom-560m
hf repos duplicate openai/gdpval --type dataset
```对于 Spaces，您仍然需要配置自己的设置（硬件、睡眠时间、存储、变量和秘密）。查看 [Manage your Space](https://huggingface.co/docs/huggingface_hub/guides/manage-spaces) 指南了解更多详细信息。

或者，如果您想保留存储库的本地副本，可以使用 `hf download` 后跟 `hf upload` 到不同的命名空间。这也不会保留 Git 历史记录。

### 使用 Git 手动分叉

如果您需要保留模型/数据集的 Git 历史记录，或者想要对流程进行更多控制（例如，根据您自己的更改进行变基），您可以使用 Git 手动分叉存储库。

您需要安装[⟦T12⟧](https://huggingface.co/docs/hub/xet/using-xet-storage#git)。分叉可能需要一些时间，具体取决于您的带宽，因为您必须获取并重新上传所有 LFS 文件（尽管由于 Xet，重新上传会很快）。

1. 在 https://huggingface.co 上创建目标存储库（例如 `me/myfork`）

2. 克隆它并将源存储库添加为远程存储库：

```bash
git clone git@hf.co:me/myfork
cd myfork
git xet install
git remote add upstream git@hf.co:friend/upstream
git fetch upstream
git lfs fetch --all upstream
```

3. 用上游历史记录替换 fork 内容：

```bash
git reset --hard upstream/main
```

4. 推送：

```bash
git push --force origin main
```

### Langfuse 空间
https://huggingface.co/docs/hub/spaces-sdks-docker-langfuse.md