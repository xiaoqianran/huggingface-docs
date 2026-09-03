<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 创建并管理存储库

Hugging Face Hub 是 git 存储库的集合。 [Git](https://git-scm.com/)是软件中广泛使用的工具
协作工作时轻松进行项目版本开发。本指南将向您展示如何与
Hub 上的存储库，特别是：

- 创建和删除存储库。
- 管理分支和标签。
- 重命名您的存储库。
- 更新您的存储库可见性。
- 管理存储库的本地副本。

> [!警告]
> 如果您习惯使用 GitLab/GitHub/Bitbucket 等平台，您的第一直觉
> 可能是使用 `git` CLI 克隆您的存储库 (`git clone`)，提交更改 (`git add, git commit`) 并推送它们
> (`git push`)。这在使用 Hugging Face Hub 时有效。然而，软件工程和机器学习确实
> 不共享相同的要求和工作流程。模型存储库可能会维护不同的大型模型权重文件
> 框架和工具，因此克隆存储库可能会导致您维护大量的本地文件夹。作为
> 因此，使用我们自定义的 HTTP 方法可能会更有效。您可以阅读我们的[Git vs HTTP paradigm](../concepts/git_vs_http)
> 说明页面了解更多详细信息。如果您想在Hub上创建和管理存储库，您的机器必须登录。如果您没有登录，请参阅
[this section](../quick-start#authentication)。在本指南的其余部分中，我们将假设您的计算机已登录。

## 列出您的存储库

您可以使用 [list_user_repos()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_user_repos) 列出您的账户或组织的所有存储库（模型、数据集、空间和存储桶）。结果包括存储信息并按存储使用情况排序。

```py
>>> from huggingface_hub import list_user_repos

# List repos for the authenticated user
>>> repos = list(list_user_repos())
>>> for repo in repos[:3]:
...     print(f"{repo.id} ({repo.type}) - {repo.storage} bytes")
username/my-model (model) - 4828692480 bytes
username/my-dataset (dataset) - 598427559 bytes
username/my-space (space) - 120620146 bytes

# List repos from an organization
>>> repos = list(list_user_repos(namespace="my-org"))
```

或者通过 CLI（默认显示 30 个存储库，使用 `--limit 0` 列出所有）：

```bash
>>> hf repos ls
>>> hf repos ls --namespace my-org --type model
>>> hf repos ls --limit 0 --format json | jq '.[].id'
```

## 仓库创建和删除

第一步是了解如何创建和删除存储库。您只能管理您拥有的存储库（在
您的用户名命名空间）或来自您具有写入权限的组织。

### 创建存储库

使用 [create_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建一个空存储库，并使用 `repo_id` 参数为其命名。 `repo_id` 是您的命名空间，后跟存储库名称：`username_or_org/repo_name`。

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model")
'https://huggingface.co/lysandre/test-model'
```

或者通过 CLI：

```bash
>>> hf repos create lysandre/test-model
Successfully created lysandre/test-model on the Hub.
Your repo is now available at https://huggingface.co/lysandre/test-model
```

默认情况下，[create_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建模型存储库。但您可以使用 `repo_type` 参数来指定其他存储库类型。例如，如果您想创建一个数据集存储库：

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-dataset", repo_type="dataset")
'https://huggingface.co/datasets/lysandre/test-dataset'
```

或者通过 CLI：

```bash
>>> hf repos create lysandre/test-dataset --repo-type dataset
```创建存储库时，可以使用 `visibility` 参数设置存储库可见性：

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-private", visibility="private")
```

或者通过 CLI：

```bash
>>> hf repos create lysandre/test-private --private
```

如果您想稍后更改存储库可见性，可以使用 [update_repo_settings()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_repo_settings) 函数。

> [!提示]
> 如果您属于具有企业计划的组织，则可以通过将 `resource_group_id` 作为参数传递给 [create_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 在特定资源组中创建存储库。资源组是一项安全功能，用于控制组织中的哪些成员可以访问给定资源。您可以通过从 Hub 上的组织设置页面 URL 复制资源组 ID 来获取资源组 ID（例如 `"https://huggingface.co/organizations/huggingface/settings/resource-groups/66670e5163145ca562cb1988"` => `"66670e5163145ca562cb1988"`）。有关资源组的更多详细信息，请查看此[guide](https://huggingface.co/docs/hub/en/security-resource-groups)。

您还可以通过传递 `region` 作为参数在特定云区域中创建存储库：

```py
>>> from huggingface_hub import create_repo
>>> create_repo("lysandre/test-model", region="us")
```

### 删除存储库

使用 [delete_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_repo) 删除存储库。确保您要删除存储库，因为这是一个不可逆的过程！

指定要删除的存储库的 `repo_id`：

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset")
```

如果存储库不存在，则传递 `missing_ok=True` 以静默忽略调用：

```py
>>> delete_repo(repo_id="lysandre/my-corrupted-dataset", repo_type="dataset", missing_ok=True)
```

或者通过 CLI：

```bash
>>> hf repos delete lysandre/my-corrupted-dataset --repo-type dataset
```

### 复制存储库在某些情况下，您想要复制其他人的存储库以使其适应您的用例。
使用 [duplicate_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.duplicate_repo) 方法可以实现这一点。它将复制整个存储库，保留完整的 git 历史记录。
这适用于模型、数据集和空间。对于 Spaces，您仍然需要配置自己的设置（硬件、睡眠时间、存储、变量和秘密）。请查看我们的 [Manage your Space](./manage-spaces) 指南了解更多详细信息。

```py
>>> from huggingface_hub import duplicate_repo

# Duplicate a Space
>>> duplicate_repo("multimodalart/dreambooth-training", repo_type="space", private=False)
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)

# Duplicate a dataset
>>> duplicate_repo("openai/gdpval", repo_type="dataset")
RepoUrl('https://huggingface.co/datasets/nateraw/gdpval',...)
```

## 搜索空间

Hub 提供了用于发现空间的语义搜索 API。您可以使用 [search_spaces()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.search_spaces) 使用自然语言查询进行搜索：

```py
>>> from huggingface_hub import search_spaces
>>> results = list(search_spaces("generate image"))
>>> results[0].id
'mrfakename/Z-Image-Turbo'
```

有关更多详细信息和过滤选项，请参阅 [Manage your Spaces](./manage-spaces#search-for-spaces) 指南。

## 上传和下载文件

现在您已经创建了存储库，您有兴趣将更改推送到它并从中下载文件。

这两个主题值得有自己的指南。请参阅[upload](./upload)和[download](./download)指南
了解如何使用您的存储库。

## 复制文件使用 [copy_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.copy_files) 将集线器上已托管的文件从一个存储库复制到另一个存储库（甚至在同一存储库中），而无需下载和重新上传它们。支持单个文件和整个文件夹，并且使用 Xet 或 LFS 跟踪的文件通过哈希值复制到服务器端。

```py
>>> from huggingface_hub import copy_files

# Copy a single file from one repo to another
>>> copy_files(
...     "hf://username/source-model/config.json",
...     "hf://username/dest-model/config.json",
... )

# Copy an entire folder (a trailing "/" copies the folder *contents*, rsync-style)
>>> copy_files(
...     "hf://datasets/username/my-dataset/data/",
...     "hf://datasets/username/my-dataset-copy/data/",
... )
```

或者通过 CLI，使用统一的 `hf cp` 命令（也可用作 `hf repos cp`）：

```bash
# Copy a single file between repositories
>>> hf cp hf://username/source-model/config.json hf://username/dest-model/config.json

# Copy a file from a repo to your local machine
>>> hf repos cp hf://username/my-model/config.json ./config.json

# Upload a local file to a repository
>>> hf repos cp ./model.safetensors hf://username/my-model/model.safetensors
```

> [!提示]
> `copy_files`（和`hf cp`）还可以将文件从存储库复制到[Bucket](./buckets)。不支持从存储桶复制到存储库。有关更多详细信息，请参阅 [Buckets](./buckets) 指南。

> [!警告]
> 服务器端副本只能在同一个[storage region](https://huggingface.co/docs/hub/storage-regions)内工作。

## 分支和标签

Git 存储库通常使用分支来存储同一存储库的不同版本。
标签还可以用于标记存储库的特定状态，例如在发布版本时。
更一般地，分支和标签被称为[git references](https://git-scm.com/book/en/v2/Git-Internals-Git-References)。

### 创建分支和标签

您可以使用 [create_branch()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_branch) 和 [create_tag()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_tag) 创建新分支和标签：

```py
>>> from huggingface_hub import create_branch, create_tag

# Create a branch on a Space repo from `main` branch
>>> create_branch("Matthijs/speecht5-tts-demo", repo_type="space", branch="handle-dog-speaker")

# Create a tag on a Dataset repo from `v0.1-release` branch
>>> create_tag("bigcode/the-stack", repo_type="dataset", revision="v0.1-release", tag="v0.1.1", tag_message="Bump release version.")
```

或者通过 CLI：

```bash
>>> hf repos branch create Matthijs/speecht5-tts-demo handle-dog-speaker --repo-type space
>>> hf repos tag create bigcode/the-stack v0.1.1 --repo-type dataset --revision v0.1-release -m "Bump release version."
```

您可以以相同的方式使用[delete_branch()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_branch)和[delete_tag()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_tag)函数来删除分支或标签，或者在CLI中分别使用`hf repos branch delete`和`hf repos tag delete`。

### 列出所有分支和标签您还可以使用 [list_repo_refs()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_refs) 列出存储库中现有的 git 引用：

```py
>>> from huggingface_hub import list_repo_refs
>>> list_repo_refs("bigcode/the-stack", repo_type="dataset")
GitRefs(
   branches=[
         GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
         GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
   ],
   converts=[],
   tags=[
         GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
   ]
)
```

## 更改存储库设置

存储库附带一些您可以配置的设置。大多数时候，您需要在
浏览器中的存储库设置页面。您必须具有对存储库的写访问权限才能配置它（要么拥有它，要么成为该存储库的一部分）
一个组织）。在本节中，我们将看到您也可以使用 `huggingface_hub` 以编程方式配置的设置。

某些设置特定于空间（硬件、环境变量……）。要配置这些，请参阅我们的[Manage your Spaces](../guides/manage-spaces)指南。

### 更新可见性

存储库可以是公共的或私有的。私有存储库仅对您或存储库所在组织的成员可见。将存储库更改为私有，如下所示：

```py
>>> from huggingface_hub import update_repo_settings
>>> update_repo_settings(repo_id=repo_id, private=True)
```

或者通过 CLI：

```bash
>>> hf repos settings lysandre/test-private --private true
```

### 设置门控访问为了更好地控制存储库的使用方式，中心允许存储库作者为其存储库启用**访问请求**。用户必须同意与存储库作者共享其联系信息（用户名和电子邮件地址），才能在启用后访问文件。启用访问请求的存储库称为**门控存储库**。

您可以使用 [update_repo_settings()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_repo_settings) 将存储库设置为门禁：

```py
>>> from huggingface_hub import HfApi

>>> api = HfApi()
>>> api.update_repo_settings(repo_id=repo_id, gated="auto")  # Set automatic gating for a model
```

或者通过 CLI：

```bash
>>> hf repos settings lysandre/test-private --gated auto
```

### 重命名你的存储库

您可以使用 [move_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.move_repo) 重命名 Hub 上的存储库。使用此方法，您还可以将存储库从用户移动到
一个组织。这样做时，会出现一个[few limitations](https://hf.cos/docs/hub/repositories-settings#renaming-or-transferring-a-repo)
您应该知道这一点。例如，您无法将您的存储库转移给其他用户。

```py
>>> from huggingface_hub import move_repo
>>> move_repo(from_id="Wauplin/cool-model", to_id="huggingface/cool-model")
```

或者通过 CLI：

```bash
>>> hf repos move Wauplin/cool-model huggingface/cool-model
```

## 内核存储库

Hub 支持用于托管计算内核的 `"kernel"` 存储库类型。这不是完全兼容的存储库类型。仅有限的一组方法经过测试并得到官方支持：

- [kernel_info()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.kernel_info)
- [hf_hub_download()](/docs/huggingface_hub/v1.30.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)
- [snapshot_download()](/docs/huggingface_hub/v1.30.0/en/package_reference/file_download#huggingface_hub.snapshot_download)
- [list_repo_refs()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_refs)
- [list_repo_files()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_files)
- [list_repo_tree()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree)

请注意，[create_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 和 [delete_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_repo) 也兼容，但仅限于 Hub 上允许的一小部分用户和组织。对于构建、发布和使用内核存储库，请改用专用的 [⟦T49⟧](https://github.com/huggingface/kernels) 包。更多详情请参阅[Kernels documentation](https://huggingface.co/docs/kernels/index)。

### 沙箱
https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/sandbox.md