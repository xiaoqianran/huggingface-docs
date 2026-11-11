<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 从存储库中删除敏感数据

> [!注意]
> 此页面仅适用于 git 支持的存储库（模型、数据集和空间）。

如果您在 Hub 上提交了您不想要的秘密、私有数据或任何文件，在新提交中删除它是不够的：旧提交仍然包含它，并且 `/resolve/<old_sha>/<path>`、`/blob/<old_sha>/<path>` 或 `/commit/<old_sha>` 等 URL 继续提供它。要完全删除文件，请从可以到达它的每个引用中删除它：分支、标签和拉取请求。

> [!警告]
> 如果文件包含机密（令牌、API 密钥、密码），请首先撤销或轮换它，并将其视为已泄露：CDN 缓存的副本或用户下载的副本可能比删除后的寿命更长。对于拥抱脸部令牌，请参阅[Revoking a leaked token](./security-tokens#revoking-a-leaked-token)。

＃＃ 概述|行动|删除 |叶子|
|---|---|---|
|删除新提交中的文件 |最新版本的文件 |所有以前的提交，因此本身还不够 |
| [List Large Files](#lfs-and-xet-files) → 使用“重写历史记录”删除 |文件内容（立即）及其来自所有分支、标签和 PR 引用的指针 |旧提交包含文件路径、sha256 和大小，[until cleanup](#when-old-commits-disappear) |
| [⟦T5⟧](#regular-git-files) + 强力推 |来自每个分支的文件和您推送的标签 |您不推送的参考文献，历史记录由拉取请求固定 |
| [⟦T6⟧](#regular-git-files) |一枝的历史|由拉取请求固定的其他分支、标签、历史记录 |
| [Delete the pull request](#pull-requests) |公关裁判，仅发布其保留的历史|那段历史，[until cleanup](#when-old-commits-disappear)|

重写历史记录（除新提交之外的每个选项）都是不可逆的，并且会更改提交 SHA。使用 git 的协作者应该重新克隆或将其分支重置到重写的分支上，并且不得将其旧历史记录推回原处，否则会重新引入该文件。

## 推荐顺序

1. 撤销或轮换任何泄露的秘密。
2. 使用“重写历史记录”删除[LFS and Xet files](#lfs-and-xet-files)。
3. 从每个分支和标签中删除[regular git files](#regular-git-files)。
4. 对于常规git文件，删除文件提交后打开的[pull requests](#pull-requests)。
5. 清理你拥有的[duplicated repositories](#duplicated-repositories)。
6. [Contact us](#when-to-contact-us) 如果需要。

## LFS 和 Xet 文件如果该文件出现在存储库的“列出大文件”中（这包括 Xet 支持的文件）：

1. 转到存储库的设置页面。
2. 单击“存储”部分中的“列出大文件”。
3. 删除文件，同时选中“重写历史记录”（默认）。

使用“重写历史记录”删除大文件会立即删除其内容，并从每个分支、标签和 PR 引用中删除其指针。这会更改包含该文件及其所有后代的第一个提交的 SHA，并删除留空的提交（例如仅添加该文件的提交）。旧的提交仍然显示文件的路径、sha256 和大小[until they're cleaned up](#when-old-commits-disappear)。更多详情请参见[Storage Limits](./storage-limits#deleting-individual-lfs-files)。

> [!注意]
> 如果文件的路径或名称本身敏感，它会保留在那些旧的提交中[until they're cleaned up](#when-old-commits-disappear)。 “重写历史”将其与他们一起删除。

## 常规 git 文件

**选项 1：使用 [⟦T7⟧](https://github.com/newren/git-filter-repo)** 重写历史记录，并强制推送包含该文件的每个分支和标记。具有写访问权限的用户允许强制推送。

```bash
git clone https://huggingface.co/<namespace>/<repo>  # use /datasets/ or /spaces/ for those repo types
cd <repo>
git filter-repo --invert-paths --path path/to/file
git remote add origin https://huggingface.co/<namespace>/<repo>  # filter-repo removes the remote
git push --force --all origin
git push --force --tags origin
```

**选项 2：删除文件并使用 [⟦T8⟧](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.super_squash_history) 压缩历史记录**。每次调用都会将单个分支压缩为一次提交，因此对每个受影响的分支重复此操作并删除不需要的分支和标签。

```python
from huggingface_hub import HfApi

api = HfApi()
api.delete_file("path/to/file", repo_id="<namespace>/<repo>")
api.super_squash_history(repo_id="<namespace>/<repo>", branch="main")
api.delete_tag(repo_id="<namespace>/<repo>", tag="v1.0")
```

## 拉取请求这仅适用于常规 git 文件：删除具有“重写历史记录”的大文件已经重写了拉取请求。

提交文件后打开的任何拉取请求都会使 SHA 可以访问该历史记录，即使您重写或压缩`main`，即使 PR 已合并或关闭。 [Deleting the pull request](./storage-limits#deleting-pull-request-refs) 释放该历史记录，即 [cleaned up](#when-old-commits-disappear)。

## 重复的存储库

[Duplicated repositories](./repositories-next-steps#how-to-duplicate-a-repo) 是独立的，因此请以同样的方式清理您拥有的。模型和数据集副本仅包含复制时`main`的文件（作为单个提交），而空间副本保留`main`的完整历史记录。其他分支和标签不会被复制。

## 当旧的提交消失时

旧提交通过 SHA（在 Web UI、API 和 `git fetch <sha>` 中）保持可访问性，直到它们被清理为止，这通常发生在强制推送、`super_squash_history`、“重写历史记录”删除或删除保留它们的拉取请求后大约一小时内。

## 何时联系我们

- 您需要保证返回 404 的旧提交 URL
- 您需要缓存的下载无效
- 您需要重置派生副本，例如 Data Studio Parquet 转换 (`refs/convert/*`) 或 Space 构建### GGUF 与 llama.cpp 的用法
https://huggingface.co/docs/hub/gguf-llamacpp.md