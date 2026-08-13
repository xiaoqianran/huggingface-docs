<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储限制

在 Hugging Face，我们的目标是为人工智能社区提供大量的**公共存储库免费存储空间**，并可以在必要时选择购买更多存储空间。我们还对免费层之上的**私有存储库**的存储空间进行计费（见下表）。

> [!提示]
> 存储限制和策略适用于 Hub 上所有类型的存储库（模型、数据集、存储桶等）。

我们从[optimize our infrastructure](https://huggingface.co/blog/xethub-joins-hf)持续到[scale our storage](https://x.com/julien_c/status/1821540661973160339)，以迎接未来几年人工智能和机器学习的发展。

我们确实采取了缓解措施来防止滥用免费公共存储，一般来说，我们要求用户和组织确保任何上传的大型模型或数据集**对社区尽可能有用**（例如，以点赞或下载的数量表示）。升级到付费组织或用户 (PRO) 帐户以解锁更高的限制。

## 存储计划|账户类型 |公共存储|私人存储|
| ------------------------ | ---------------------------------------------------------------------------------- | ---------------------------- |
|免费用户或组织 |尽力而为\* | 100GB |
|专业版 |包含高达 10TB\* + [add-on](#public-storage-add-on) ✅ 可为有影响力的工作提供资助† | 1TB + 按量付费 |
|团队组织| 12TB 底座 + 每个座位 1TB + [add-on](#public-storage-add-on) ✅ |每个席位 1TB + 即用即付 |
|企业组织| 200TB 基础 + 每个席位 1TB + [add-on](#public-storage-add-on) 🏆 大合同高达 1,000TB |每个席位 1TB + 即用即付 |

💡 [Team or Enterprise Organizations](https://huggingface.co/enterprise) 订阅中包含每个席位 1TB 的私人存储空间：例如，如果您的组织有 40 名成员，则您拥有 40TB 的包含私人存储空间。\* 我们的目标是继续为人工智能社区提供慷慨的免费公共存储空间。除了最初的几千兆字节之外，请通过上传为其他用户提供真正价值的内容来负责任地使用此资源。如果您需要大量存储空间，则需要升级到[PRO, Team or Enterprise](https://huggingface.co/pricing)。

† 在某些情况下，额外的存储补助可用于高影响力的开源工作，而付费计划确实无法满足需求。联系我们并提供社区影响的证据（点赞、下载、引用）。

### 公共存储附加组件

付费计划（PRO、团队或企业）的用户可以订阅**公共存储附加组件**，以在其计划的基本限制之上获得额外的公共存储。

|存储附加组件 |价格|每 TB |
| -------------- | -------------- | ---------------- |
| 1TB | 12 美元/月 | 12 美元/TB/月 |
| 5TB | 60 美元/月 | 12 美元/TB/月 |
| 10TB | 120 美元/月 | 12 美元/TB/月 |
| 20TB | 240 美元/月 | 12 美元/TB/月 |
| 50TB | 500 美元/月 | 10 美元/TB/月 |您可以从您的帐户或组织的**计费**设置页面订阅或更改您的级别。升级立即生效；降级计划于下月初生效。如果您需要更多存储空间，可以通过[contact us](https://huggingface.co/contact/sales)来利用[custom large-scale pricing](https://huggingface.co/pricing#storage)。

### 私人存储 按量付费

除了 [PRO](https://huggingface.co/subscribe/pro) 和 [Team or Enterprise Organizations](https://huggingface.co/enterprise) 中包含的 1TB（或每个席位 1TB）私人存储之外，额外的私人存储将以即用即付模式通过您的付款方式收取费用，基本价格为 18 美元/TB/月。
通过我们的客户经理，大批量可享受额外折扣：

|卷 |价格（私人回购）|
| ------ | -------------------- |
|基地| 18 美元/TB/月 |
| 50TB+ | 16 美元/TB/月 |
| 200TB+ | 14 美元/TB/月 |
| 500TB+ | 12 美元/TB/月 |

请参阅我们的[billing doc](./billing)了解更多详情，或在[huggingface.co/pricing](https://huggingface.co/pricing#storage)查看最新定价。

## 存储库限制和建议

> [!注意]
> 本节不适用于[Storage Buckets](./storage-buckets)除了帐户（用户或组织）级别的存储限制之外，在处理特定 Git 支持的存储库中的大量数据时，还需要注意一些限制。考虑到传输数据所需的时间，
无论是在 hf.co 上还是在本地工作时，上传/推送在流程结束时失败或体验降级可能会非常烦人。在下一节中，我们将描述有关如何最好地构建大型存储库的建议。

### 建议

我们收集了一系列有关构建存储库的提示和建议。如果您正在寻找更实用的技巧，请查看[this guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#tips-and-tricks-for-large-uploads)了解如何使用Python库上传大量数据。

|特点|推荐|提示 |
| ---------------- | ------------------ | ------------------------------------------------------ |
|回购规模 | - |升级您的 [storage plan](#storage-plans) 或联系我们获取大型存储库（TB 级数据）|
|每个存储库的文件 | <100k              | merge data into fewer files                            |
| Entries per folder | <10k               | use subdirectories in repo                             |
| File size          | <200GB             | split data into chunked files                          |
| Commit size        | <100 files*        | ⟦T2⟧/⟦T3⟧ auto-split large folders into multiple commits                       |
| Commits per repo   | -                  | upload multiple files per commit and/or squash history |

_\* Not relevant when using ⟦T4⟧ CLI directly_

Please read the next section to better understand those limits and how to deal with them.

### Explanations

What are we talking about when we say "large uploads", and what are their associated limitations? Large uploads can be
very diverse, from repositories with a few huge files (e.g. model weights) to repositories with thousands of small files
(e.g. an image dataset).

Under the hood, the Hub uses Git to version the data, which has structural implications on what you can do in your repo.
If your repo is crossing some of the numbers mentioned in the previous section, **we strongly encourage you to check out ⟦T30⟧**,
which has very detailed documentation about the different factors that will impact your experience. Here is a TL;DR of factors to consider:

- **Repository size**: The total size of the data you're planning to upload. There is no per-repo size limit for models and datasets, but uploads count against your account's total storage quota (see ⟦T31⟧ above). If you need more storage, ⟦T32⟧ or purchase a ⟦T33⟧.
- **Number of files**:
    - For optimal experience, we recommend keeping the total number of files under 100k, and ideally much less. Try merging the data into fewer files if you have more.
      For example, json files can be merged into a single jsonl file, or large datasets can be exported as Parquet files or in ⟦T34⟧ format.
    - The maximum number of files per folder cannot exceed 10k files per folder. A simple solution is to
      create a repository structure that uses subdirectories. For example, a repo with 1k folders from ⟦T6⟧ to ⟦T7⟧, each containing at most 1000 files, is already enough.
- **File size**: In the case of uploading large files (e.g. model weights), we strongly recommend splitting them **into chunks <200GB each.**.
There are a few reasons for this:
    - Uploading and downloading smaller files is much easier both for you and the other users. Connection issues can always
      happen when streaming data and smaller files avoid resuming from the beginning in case of errors.
    - Files are served to the users using CloudFront. From our experience, huge files are not cached by this service
      leading to a slower download speed.
In all cases, no single file will exceed 500GB. I.e. 500GB is the hard limit for a single file size.
- **Number of commits**: There is no hard limit for the total number of commits on your repo history. However, from
our experience, the user experience on the Hub starts to degrade after a few thousand commits. We are constantly working to
improve the service, but one must always remember that a git repository is not meant to work as a database with a lot of
writes. If your repo's history gets very large, it is always possible to squash all the commits to get a
fresh start using ⟦T8⟧'s ⟦T35⟧. Be aware that this is a non-revertible operation.
- **Number of operations per commit**: Once again, there is no hard limit here. When a commit is uploaded on the Hub, each
git operation (addition or delete) is checked by the server. When a hundred Large Files are committed at once,
each file is checked individually to ensure it's been correctly uploaded. When pushing data through HTTP,
a timeout of 60s is set on the request, meaning that if the process takes more time, an error is raised. However, it can
happen (in rare cases) that even if the timeout is raised client-side, the process is still
completed server-side. This can be checked manually by browsing the repo on the Hub. The ⟦T36⟧ method and ⟦T11⟧ command avoid this automatically by splitting large folders into multiple commits. If you commit manually, keep around 50-100 files per commit.

### Sharing large datasets on the Hub

One key way Hugging Face supports the machine learning ecosystem is by hosting datasets on the Hub, including very large ones. Large datasets count against your account's total storage quota, so make sure your ⟦T37⟧ has sufficient capacity before uploading. Additional public storage can be purchased as an ⟦T38⟧.

For hosting large datasets on the Hub, we require the following:

- A dataset card: we want to ensure that your dataset can be used effectively by the community and one of the key ways of enabling this is via a dataset card. This ⟦T39⟧ provides an overview of how to write a dataset card.
- You are sharing the dataset to enable community reuse. If you plan to upload a dataset you anticipate won't have any further reuse, other platforms are likely more suitable.
- You must follow the repository limitations outlined above.
- Using file formats that are well integrated with the Hugging Face ecosystem. We have good support for ⟦T40⟧ and ⟦T41⟧ formats, which are often good options for sharing large datasets efficiently. This will also ensure the dataset viewer works for your dataset.
- Avoid the use of custom loading scripts when using datasets. In our experience, datasets that require custom code to use often end up with limited reuse.

Please get in touch with us if any of these requirements are difficult for you to meet because of the type of data or domain you are working in.

### Sharing large volumes of models on the Hub

Similarly to datasets, large models or large volumes of models (for instance, hundreds of automated quants) count against your account's total storage quota. Make sure your ⟦T42⟧ has sufficient capacity, or purchase a ⟦T43⟧.

### Grants for research teams and non-profits

We recommend that academic and research institutions upgrade to ⟦T44⟧ for guaranteed storage limits. In some cases, storage grants may be available for high-impact open-source work where a paid plan genuinely cannot cover the need. These are evaluated on a case-by-case basis and require demonstrated community impact (downloads, citations, community adoption, etc.). Contact datasets@huggingface.co or models@huggingface.co with a detailed proposal.

## How can I free up storage space in my account/organization?

There are several ways to manage and free some storage space in your account or organization. First, if you need more storage space, upgrade to a PRO, Team or Enterprise plan for increased storage limits.

⚠️ **Important**: Deleting Large Files is a destructive operation that cannot be undone. Make sure to backup your files before proceeding.

Key points to remember:
- Deleting only LFS pointers doesn't free up space
- If you do not rewrite the Git history, future checkouts of branches/tags containing deleted LFS files with existing lfs pointers will fail (to avoid errors, add the following line to your ⟦T12⟧ file: ⟦T13⟧)

### Deleting individual LFS files

1. Navigate to your repository's Settings page
2. Click on "List LFS files" in the "Storage" section
3. Use the actions menu to delete specific files

### Deleting Pull request refs

⟦T45⟧ create git refs that store their commits. After closing or merging a PR, you can delete its ref to free up storage space. This is especially useful when:
- PRs contain large files that were never merged
- You've squashed the main branch and removed files later on — those files remain in the PR branch history even if they weren't added by the PR itself

To delete a PR ref, open the closed or merged PR and look for the storage notice at the bottom showing the estimated space that could be freed. Click "Delete ref" to permanently remove it.

> [！注意]
> 删除 PR 引用是不可逆的，并且将阻止任何人在本地获取或检查这些提交。### 使用 API 超级压缩您的存储库

super-squash 操作将整个 Git 历史记录压缩为一次提交。当您需要从不使用的旧 LFS 版本回收存储时，请考虑使用 super-squash。此操作只能通过[Hub Python Library](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/hf_api#huggingface_hub.HfApi.super_squash_history)或API进行。

⚠️ **重要**：这是一个无法撤消的破坏性操作，提交历史记录将永久丢失，并且 **LFS 文件历史记录将被删除**

压缩操作不会立即对您的存储配额产生影响，并将在 36 小时内反映在您的配额中。

### 高级：跟踪 LFS 文件引用

当您在存储库的“列出 LFS 文件”中找到 LFS 文件但不知道它来自哪里时，您可以使用 git log 命令使用其 SHA-256 OID 跟踪其历史记录： 

```bash
git log --all -p -S <SHA-256-OID>
```

例如：

```bash
git log --all -p -S 68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3

commit 5af368743e3f1d81c2a846f7c8d4a028ad9fb021
Date:   Sun Apr 28 02:01:18 2024 +0200

    Update LayerNorm tensor names to weight and bias

diff --git a/model.safetensors b/model.safetensors
index a090ee7..e79c80e 100644
--- a/model.safetensors
+++ b/model.safetensors
@@ -1,3 +1,3 @@
 version https://git-lfs.github.com/spec/v1
-oid sha256:68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
+oid sha256:0bb7a1683251b832d6f4644e523b325adcf485b7193379f5515e6083b5ed174b
 size 440449768

commit 0a6aa9128b6194f4f3c4db429b6cb4891cdb421b (origin/pr/28)
Date:   Wed Nov 16 15:15:39 2022 +0000

    Adding `safetensors` variant of this model (#15)
    
    
    - Adding `safetensors` variant of this model (18c87780b5e54825a2454d5855a354ad46c5b87e)
    
    
    Co-authored-by: Nicolas Patry <Narsil@users.noreply.huggingface.co>

diff --git a/model.safetensors b/model.safetensors
new file mode 100644
index 0000000..a090ee7
--- /dev/null
+++ b/model.safetensors
@@ -0,0 +1,3 @@
+version https://git-lfs.github.com/spec/v1
+oid sha256:68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
+size 440449768

commit 18c87780b5e54825a2454d5855a354ad46c5b87e (origin/pr/15)
Date:   Thu Nov 10 09:35:55 2022 +0000

    Adding `safetensors` variant of this model

diff --git a/model.safetensors b/model.safetensors
new file mode 100644
index 0000000..a090ee7
--- /dev/null
+++ b/model.safetensors
@@ -0,0 +1,3 @@
+version https://git-lfs.github.com/spec/v1
+oid sha256:68d45e234eb4a928074dfd868cead0219ab85354cc53d20e772753c6bb9169d3
+size 440449768

```

### 如何使用 Okta 配置 OIDC SSO
https://huggingface.co/docs/hub/security-sso-okta-oidc.md