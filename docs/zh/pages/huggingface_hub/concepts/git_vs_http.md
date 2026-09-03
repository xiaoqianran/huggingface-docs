<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Git 与 HTTP 范例

`huggingface_hub` 库是一个用于与 Hugging Face Hub 交互的库，它是基于 git 的存储库（模型、数据集或空间）的集合。使用 `huggingface_hub` 访问 Hub 有两种主要方式。

第一种方法，即所谓的“基于 git”的方法，依赖于直接在终端中使用标准 `git` 命令。此方法允许您克隆存储库、创建提交并手动推送更改。第二种选项称为“基于 HTTP”的方法，涉及使用 [HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 客户端发出 HTTP 请求。让我们来看看每种方法的优缺点。

## Git：历史上基于 CLI 的方法

起初，大多数用户使用简单的 `git` 命令（例如 `git clone`、`git add`、`git commit`、`git push`、`git tag` 或 `git checkout`）与 Hugging Face Hub 进行交互。这种方法使您可以使用计算机上存储库的完整本地副本，就像传统的软件开发一样。当您需要离线访问或想要使用存储库的完整历史记录时，这可能是一个优势。然而，它也有缺点：您负责在本地保持存储库最新、处理凭据以及管理大型文件（通过`git-lfs`），这在处理大型机器学习模型或数据集时可能会变得很麻烦。

在许多机器学习工作流程中，您可能只需要下载一些文件进行推理或转换权重，而不需要克隆整个存储库。在这种情况下，使用 `git` 可能会太过分，并带来不必要的复杂性。

## HfApi：灵活便捷的HTTP客户端

开发 [HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 类是为了提供使用本地 git 存储库的替代方案，本地 git 存储库维护起来很麻烦，尤其是在处理大型模型或数据集时。 [HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 类提供与基于 git 的工作流程相同的功能 - 例如下载和推送文件以及创建分支和标签 - 但不需要需要保持同步的本地文件夹。除了 `git` 已经提供的功能之外，[HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 类还提供了其他功能，例如管理存储库的能力、使用缓存下载文件以实现高效重用、搜索 Hub 中的存储库和元数据、访问社区功能（例如讨论、PR 和评论）以及配置 Spaces 硬件和机密。

## 我应该使用什么？什么时候？

总的来说，**基于 HTTP 的方法是在所有情况下推荐使用** `huggingface_hub` 的方法。 [HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 允许您拉动和推送更改、使用 PR、标签和分支、与讨论互动等等。

然而，并非所有 git 命令都可以通过 [HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 使用。有些可能永远不会实现，但我们一直在努力改进并缩小差距。如果您没有看到您的用例，请打开[an issue on GitHub](https://github.com/huggingface/huggingface_hub)！我们欢迎反馈意见，以帮助我们与用户一起构建 HF 生态系统。

这种对基于 HTTP 的 [HfApi](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi) 而非直接 `git` 命令的偏好并不意味着 git 版本控制很快就会从 Hugging Face Hub 中消失。在有意义的工作流程中，始终可以在本地使用 `git`。### 迁移到huggingface_hub v1.0
https://huggingface.co/docs/huggingface_hub/v1.30.0/concepts/migration.md