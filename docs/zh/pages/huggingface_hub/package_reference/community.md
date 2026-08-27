<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 与讨论和 Pull 请求交互

查看[HfApi](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi)文档页面以获取启用方法的参考
与 Hub 上的 Pull 请求和讨论进行交互。

- [get_repo_discussions()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_repo_discussions)
- [get_discussion_details()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_discussion_details)
- [create_discussion()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_discussion)
- [create_pull_request()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_pull_request)
- [rename_discussion()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.rename_discussion)
- [comment_discussion()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.comment_discussion)
- [edit_discussion_comment()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.edit_discussion_comment)
- [change_discussion_status()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.change_discussion_status)
- [merge_pull_request()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.merge_pull_request)

## 数据结构[[huggingface_hub.Discussion]]

#### Huggingface_hub.Discussion[[huggingface_hub.Discussion]]

```python
huggingface_hub.Discussion(title: str, status: typing.Literal['open', 'closed', 'merged', 'draft'], num: int, repo_id: str, repo_type: str, author: str, is_pull_request: bool, created_at: datetime, endpoint: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L20)

**参数：**

title (`str`) : 讨论/拉取请求的标题

status (`str`) ：讨论/拉取请求的状态。它必须是以下之一： * `"open"` * `"closed"` * `"merged"` （仅适用于 Pull 请求） * `"draft"` （仅适用于 Pull 请求）

num (`int`) ：讨论/拉取请求的数量。

repo_id (`str`) ：打开讨论/拉取请求的存储库的 ID (`"{namespace}/{repo_name}"`)。

repo_type (`str`) ：打开讨论/拉取请求的存储库的类型。可能的值为：`"model"`、`"dataset"`、`"space"`。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果此后用户已被删除，则可以是`"deleted"`。

is_pull_request (`bool`) ：这是否是拉取请求。

created_at (`datetime`) ：创建讨论/拉取请求的`datetime`。端点（`str`）：集线器的端点。默认为 https://huggingface.co。

git_reference (`str`, *可选*) ：（属性）如果这是拉取请求，则可以推送更改的 Git 引用，否则为 `None`。

url (`str`) ：（属性）Hub 上讨论的 URL。

集线器上的讨论或拉取请求。

该数据类不打算直接实例化。

#### Huggingface_hub.DiscussionWithDetails[[huggingface_hub.DiscussionWithDetails]]

```python
huggingface_hub.DiscussionWithDetails(title: str, status: typing.Literal['open', 'closed', 'merged', 'draft'], num: int, repo_id: str, repo_type: str, author: str, is_pull_request: bool, created_at: datetime, endpoint: str, events: list, conflicting_files: list[str] | bool | None, target_branch: str | None, merge_commit_oid: str | None, diff: str | None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L88)

**参数：**

title (`str`) : 讨论/拉取请求的标题

status (`str`) ：讨论/拉取请求的状态。它可以是以下之一： * `"open"` * `"closed"` * `"merged"` （仅适用于 Pull 请求） * `"draft"` （仅适用于 Pull 请求）

num (`int`) ：讨论/拉取请求的数量。

repo_id (`str`) ：打开讨论/拉取请求的存储库的 ID (`"{namespace}/{repo_name}"`)。

repo_type (`str`) ：打开讨论/拉取请求的存储库的类型。可能的值为：`"model"`、`"dataset"`、`"space"`。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果用户此后已被删除，则可以是`"deleted"`。

is_pull_request (`bool`) ：这是否是拉取请求。created_at (`datetime`) ：创建讨论/拉取请求的`datetime`。

events (`list` of [DiscussionEvent](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionEvent)) ：此讨论或 Pull 请求中的 `DiscussionEvents` 列表。

冲突文件（`Union[list[str], bool, None]`，*可选*）：冲突文件列表（如果这是拉取请求）。 `None` 如果 `self.is_pull_request` 是 `False`。 `True` 如果存在冲突文件但无法检索列表。

target_branch (`str`, *可选*) ：如果这是拉取请求，则要将更改合并到的分支。 `None` 如果 `self.is_pull_request` 是 `False`。

merge_commit_oid (`str`, *可选*) ：如果这是合并的拉取请求，则将其设置为合并提交的 OID / SHA，否则为 `None`。

diff (`str`, *可选*) ：如果这是 Pull Request 则为 git diff，否则为 `None`。

端点（`str`）：集线器的端点。默认为 https://huggingface.co。

git_reference (`str`, *可选*) ：（属性）如果这是拉取请求，则可以推送更改的 Git 引用，否则为 `None`。

url (`str`) ：（属性）Hub 上讨论的 URL。

[Discussion](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.Discussion) 的子类。

#### Huggingface_hub.DiscussionEvent[[huggingface_hub.DiscussionEvent]]

```python
huggingface_hub.DiscussionEvent(id: str, type: str, created_at: datetime, author: str, _event: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L155)

**参数：**

id (`str`) ：事件的ID。十六进制字符串。type (`str`) ：事件的类型。

created_at (`datetime`) ：一个[⟦T69⟧](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)对象，保存事件的创建时间戳。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果此后用户已被删除，则可以是`"deleted"`。

讨论或拉取请求中的事件。

使用具体类：
* [DiscussionComment](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionComment)
* [DiscussionStatusChange](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionStatusChange)
* [DiscussionCommit](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionCommit)
* [DiscussionTitleChange](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionTitleChange)

#### Huggingface_hub.DiscussionComment[[huggingface_hub.DiscussionComment]]

```python
huggingface_hub.DiscussionComment(id: str, type: str, created_at: datetime, author: str, _event: dict, content: str, edited: bool, hidden: bool)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L188)

**参数：**

id (`str`) ：事件的 ID。十六进制字符串。

type (`str`) ：事件的类型。

created_at (`datetime`) ：一个[⟦T75⟧](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)对象，保存事件的创建时间戳。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果此后用户已被删除，则可以是`"deleted"`。

content (`str`) : 评论的原始 Markdown 内容。提及、链接和图像不会呈现。

已编辑(`bool`)：此评论是否已被编辑。

hidden (`bool`) : 该评论是否被隐藏。

讨论/拉取请求中的评论。

[DiscussionEvent](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionEvent) 的子类。

#### Huggingface_hub.DiscussionStatusChange[[huggingface_hub.DiscussionStatusChange]]

```python
huggingface_hub.DiscussionStatusChange(id: str, type: str, created_at: datetime, author: str, _event: dict, new_status: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L243)

**参数：**id (`str`) ：事件的ID。十六进制字符串。

type (`str`) ：事件的类型。

created_at (`datetime`) ：一个[⟦T84⟧](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)对象，保存事件的创建时间戳。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果此后用户已被删除，则可以是`"deleted"`。

new_status (`str`) ：更改后讨论/拉取请求的状态。它可以是以下之一： * `"open"` * `"closed"` * `"merged"` （仅适用于 Pull 请求）

讨论/拉取请求中的状态更改。

[DiscussionEvent](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionEvent) 的子类。

#### Huggingface_hub.DiscussionCommit[[huggingface_hub.DiscussionCommit]]

```python
huggingface_hub.DiscussionCommit(id: str, type: str, created_at: datetime, author: str, _event: dict, summary: str, oid: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L271)

**参数：**

id (`str`) ：事件的 ID。十六进制字符串。

type (`str`) ：事件的类型。

created_at (`datetime`) ：一个[⟦T94⟧](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)对象，保存事件的创建时间戳。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果此后用户已被删除，则可以是`"deleted"`。

摘要 (`str`) ：提交的摘要。

oid (`str`) ：提交的 OID / SHA，作为十六进制字符串。

拉取请求中的提交。

[DiscussionEvent](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionEvent) 的子类。

#### Huggingface_hub.DiscussionTitleChange[[huggingface_hub.DiscussionTitleChange]]

```python
huggingface_hub.DiscussionTitleChange(id: str, type: str, created_at: datetime, author: str, _event: dict, old_title: str, new_title: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.29.0/src/huggingface_hub/community.py#L298)**参数：**

id (`str`) ：事件的 ID。十六进制字符串。

type (`str`) ：事件的类型。

created_at (`datetime`) ：一个[⟦T102⟧](https://docs.python.org/3/library/datetime.html?highlight=datetime#datetime.datetime)对象，保存事件的创建时间戳。

作者 (`str`) ：讨论/拉取请求作者的用户名。如果用户此后已被删除，则可以是`"deleted"`。

old_title (`str`) ：讨论/拉取请求的先前标题。

new_title (`str`) ：新标题。

讨论/拉取请求中的重命名事件。

[DiscussionEvent](/docs/huggingface_hub/v1.29.0/en/package_reference/community#huggingface_hub.DiscussionEvent) 的子类。

### Webhook 服务器
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/webhooks_server.md