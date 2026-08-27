<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 与讨论和 Pull 请求互动

`huggingface_hub` 库提供了一个 Python 接口来与 Hub 上的 Pull 请求和讨论进行交互。
参观[the dedicated documentation page](https://huggingface.co/docs/hub/repositories-pull-requests-discussions)
更深入地了解 Hub 上的讨论和拉取请求是什么，以及它们在幕后如何工作。

## 从中心检索讨论和拉取请求

`HfApi` 类允许您检索给定存储库上的讨论和拉取请求：

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(repo_id="bigscience/bloom"):
...     print(f"{discussion.num} - {discussion.title}, pr: {discussion.is_pull_request}")

# 11 - Add Flax weights, pr: True
# 10 - Update README.md, pr: True
# 9 - Training languages in the model card, pr: True
# 8 - Update tokenizer_config.json, pr: True
# 7 - Slurm training script, pr: False
[...]
```

`HfApi.get_repo_discussions` 支持按作者、类型（Pull Request 或 Discussion）和状态（`open` 或 `closed`）进行过滤：

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(
...    repo_id="bigscience/bloom",
...    author="ArthurZ",
...    discussion_type="pull_request",
...    discussion_status="open",
... ):
...     print(f"{discussion.num} - {discussion.title} by {discussion.author}, pr: {discussion.is_pull_request}")

# 19 - Add Flax weights by ArthurZ, pr: True
```

`HfApi.get_repo_discussions` 返回一个 [generator](https://docs.python.org/3.7/howto/functional.html#generators)，产生
[Discussion](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.Discussion) 对象。要将所有讨论放在一个列表中，请运行：

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

[HfApi.get_repo_discussions()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_repo_discussions) 返回的 [Discussion](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.Discussion) 对象包含以下内容的高级概述：
讨论或拉取请求。您还可以使用[HfApi.get_discussion_details()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_discussion_details)获取更详细的信息：

```python
>>> from huggingface_hub import get_discussion_details

>>> get_discussion_details(
...     repo_id="bigscience/bloom-1b3",
...     discussion_num=2
... )
DiscussionWithDetails(
    num=2,
    author='cakiki',
    title='Update VRAM memory for the V100s',
    status='open',
    is_pull_request=True,
    events=[
        DiscussionComment(type='comment', author='cakiki', ...),
        DiscussionCommit(type='commit', author='cakiki', summary='Update VRAM memory for the V100s', oid='1256f9d9a33fa8887e1c1bf0e09b4713da96773a', ...),
    ],
    conflicting_files=[],
    target_branch='refs/heads/main',
    merge_commit_oid=None,
    diff='diff --git a/README.md b/README.md\nindex a6ae3b9294edf8d0eda0d67c7780a10241242a7e..3a1814f212bc3f0d3cc8f74bdbd316de4ae7b9e3 100644\n--- a/README.md\n+++ b/README.md\n@@ -132,7 +132,7 [...]',
)
```

[HfApi.get_discussion_details()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_discussion_details) 返回一个 [DiscussionWithDetails](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionWithDetails) 对象，它是 [Discussion](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.Discussion) 的子类
有关讨论或拉取请求的更多详细信息。信息包括所有评论、状态更改、
并通过 `DiscussionWithDetails.events` 重命名讨论。

如果是 Pull 请求，您可以使用 `DiscussionWithDetails.diff` 检索原始 git diff。所有的提交
Pull 请求列在`DiscussionWithDetails.events` 中。## 以编程方式创建和编辑讨论或拉取请求

[HfApi](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi) 类还提供了创建和编辑讨论和拉取请求的方法。
您将需要一个[access token](https://huggingface.co/docs/hub/security-tokens)来创建和编辑讨论
或拉取请求。

对 Hub 上的存储库提出更改的最简单方法是通过 [create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) API：只需
将`create_pr`参数设置为`True`。此参数也可用于包装 [create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 的其他方法：

    * [upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)
    * [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)
    * [delete_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_file)
    * [delete_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_folder)
    * [metadata_update()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.metadata_update)

```python
>>> from huggingface_hub import metadata_update

>>> metadata_update(
...     repo_id="username/repo_name",
...     metadata={"tags": ["computer-vision", "awesome-model"]},
...     create_pr=True,
... )
```

您还可以使用 [HfApi.create_discussion()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_discussion)（分别为 [HfApi.create_pull_request()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_pull_request)）在存储库上创建讨论（分别为 Pull 请求）。
如果您需要在本地处理更改，以这种方式打开拉取请求可能会很有用。以这种方式打开的 Pull 请求将处于 `"draft"` 模式。

```python
>>> from huggingface_hub import create_discussion, create_pull_request

>>> create_discussion(
...     repo_id="username/repo-name",
...     title="Hi from the huggingface_hub library!",
...     token="<insert your access token here>",
... )
DiscussionWithDetails(...)

>>> create_pull_request(
...     repo_id="username/repo-name",
...     title="Hi from the huggingface_hub library!",
...     token="<insert your access token here>",
... )
DiscussionWithDetails(..., is_pull_request=True)
```

管理 Pull 请求和讨论可以完全使用 [HfApi](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi) 类来完成。例如：

    * [comment_discussion()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.comment_discussion) 添加评论
    * [edit_discussion_comment()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.edit_discussion_comment) 编辑评论
    * [rename_discussion()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.rename_discussion) 重命名讨论或 Pull 请求
    * [change_discussion_status()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.change_discussion_status) 打开或关闭讨论/拉取请求
    * [merge_pull_request()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.merge_pull_request) 合并 Pull 请求

请访问 [HfApi](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi) 文档页面，获取所有可用方法的详尽参考。

## 从 CLI 管理讨论和拉取请求上述所有操作也可以通过 `hf discussions` 从命令行进行。这对于
脚本、CI 管道或快速交互，无需编写 Python 代码。

```bash
# List open discussions and PRs on a repo
hf discussions list bigscience/bloom

# List discussions on a dataset repo
hf discussions list nebius/SWE-rebench-V2 --type dataset

# Get info for a specific discussion with comments
hf discussions info bigscience/bloom 2 --comments

# Create a new discussion
hf discussions create username/repo-name --title "Bug report" --body "Description here"

# Create a pull request
hf discussions create username/repo-name --title "Fix typo" --pull-request

# Comment on a discussion or PR
hf discussions comment username/repo-name 5 --body "LGTM!"

# Merge a pull request
hf discussions merge username/repo-name 5 --yes

# Show the diff of a pull request
hf discussions diff username/repo-name 5
```

有关选项的完整列表，请运行 `hf discussions --help` 或参阅 [CLI reference](./cli#hf-discussions)。

## 将更改推送到拉取请求

*即将推出！*

## 另请参阅

如需更详细的参考，请访问 [Discussions and Pull Requests](../package_reference/community) 和 [hf_api](../package_reference/hf_api) 文档页面。

### 将文件上传到集线器
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/upload.md