<!-- huggingface-docs: machine-translated zh-CN from English source -->

# HfApi 客户端

以下是 `HfApi` 类的文档，该类用作 Hugging Face Hub API 的 Python 包装器。

`HfApi` 中的所有方法也可以直接从包的根目录访问。下面详细介绍这两种方法。

使用 root 方法更简单，但 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 类为您提供了更大的灵活性。
特别是，您可以传递一个将在所有 HTTP 调用中重复使用的令牌。这是不同的
来自 `hf auth login` 或 [login()](/docs/huggingface_hub/v1.27.0/en/package_reference/authentication#huggingface_hub.login)，因为令牌未保留在计算机上。
还可以提供不同的端点或配置自定义用户代理。

```python
from huggingface_hub import HfApi, list_models

# Use root method
models = list_models()

# Or configure a HfApi client
hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token="hf_xxx", # Token is not persisted on the machine.
)
models = hf_api.list_models()
```

## HfApi[[huggingface_hub.HfApi]]

#### Huggingface_hub.HfApi[[huggingface_hub.HfApi]]

```python
huggingface_hub.HfApi(endpoint: str | None = None, token: str | bool | None = None, library_name: str | None = None, library_version: str | None = None, user_agent: dict | str | None = None, headers: dict[str, str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2231)

**参数：**

端点（`str`，*可选*）：集线器的端点。默认为 .

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。library_name (`str`, *可选*) ：发出 HTTP 请求的库的名称。将添加到用户代理标头中。示例：`"transformers"`。

library_version (`str`, *可选*) ：发出 HTTP 请求的库的版本。将添加到用户代理标头中。示例：`"4.24.0"`。

user_agent (`str`, `dict`, *可选*) ：字典或单个字符串形式的用户代理信息。它将包含有关已安装软件包的信息来完成。

headers (`dict`, *可选*) ：随每个请求发送的附加标头。示例：`{"X-My-Header": "value"}`。此处传递的标头优先于默认标头。

客户端通过 HTTP 与 Hugging Face Hub 交互。

客户端使用所有请求中使用的一些高级设置进行初始化
发送到集线器（HF 端点、身份验证、用户代理...）。使用`HfApi`
客户端是首选，但不是强制性的，因为它的所有公共方法都是公开的
直接位于`huggingface_hub`的根部。

####接受_访问_请求[[huggingface_hub.HfApi.accept_access_request]]

```python
accept_access_request(repo_id: str, user: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10822)

**参数：**

repo_id (`str`) : 接受访问请求的repo的id。user (`str`) ：应接受访问请求的用户的用户名。

repo_type (`str`, *可选*) ：接受访问请求的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为 `model`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。
- `HfHubHTTPError` -- 
  如果集线器上不存在用户，则返回 HTTP 404。
- `HfHubHTTPError` -- 
  如果找不到用户访问请求，则返回 HTTP 404。
- `HfHubHTTPError` -- 
  如果用户访问请求已在接受列表中，则返回 HTTP 404。

接受用户对给定门控存储库的访问请求。一旦请求被接受，用户将能够下载存储库的任何文件并访问社区
选项卡。如果审批模式为自动，则无需手动接受请求。接受的请求可以是
使用[cancel_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request)和[reject_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request)随时取消或拒绝。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

#### add_collection_item[[huggingface_hub.HfApi.add_collection_item]]

```python
add_collection_item(collection_slug: str, item_id: str, item_type: CollectionItemType_T, note: str | None = None, exists_ok: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10375)

**参数：**

collection_slug (`str`) ：要更新的集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

item_id (`str`) ：要添加到集合中的项目的 ID。使用存储库/空间/数据集的 repo_id，论文的论文 id，另一个集合的 slug（例如 `"moonshotai/kimi-k2"`）或存储桶 id（例如 `"namespace/bucket-name"`）。

item_type (`str`) ：要添加的项目的类型。可以是 `"model"`、`"dataset"`、`"space"`、`"paper"`、`"collection"` 或 `"bucket"` 之一。

note (`str`, *可选*) ：附加到集合中的项目的注释。注释的最大长度为 500 个字符。

contains_ok (`bool`, *可选*) ：如果`True`，如果项目已存在，则不会引发错误。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。
- `HfHubHTTPError` -- 
  如果您尝试添加到集合中的项目在集线器上不存在，则会出现 HTTP 404。
- `HfHubHTTPError` -- 
  如果您尝试添加到集合中的项目已在集合中（且存在_ok = False），则 HTTP 409

将项目添加到 Hub 上的集合中。

返回：[Collection](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.Collection)

示例：

```py
>>> from huggingface_hub import add_collection_item
>>> collection = add_collection_item(
...     collection_slug="davanstrien/climate-64f99dc2a5067f6b65531bab",
...     item_id="pierre-loic/climate-news-articles",
...     item_type="dataset"
... )
>>> collection.items[-1].item_id
"pierre-loic/climate-news-articles"
# ^item got added to the collection on last position

# Add item with a note
>>> add_collection_item(
...     collection_slug="davanstrien/climate-64f99dc2a5067f6b65531bab",
...     item_id="datasets/climate_fever",
...     item_type="dataset"
...     note="This dataset adopts the FEVER methodology that consists of 1,535 real-world claims regarding climate-change collected on the internet."
... )
(...)
```

#### add_space_secret[[huggingface_hub.HfApi.add_space_secret]]

```python
add_space_secret(repo_id: str, key: str, value: str, description: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8096)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

key (`str`) ：秘密密钥。示例：`"GITHUB_API_KEY"`

value (`str`) ：秘密值。示例：`"your_github_api_key"`。

描述（`str`，*可选*）：秘密描述。示例：`"Github API key to access the Github API"`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

添加或更新空间中的秘密。

秘密允许为空间设置秘密密钥或令牌，而无需对其进行硬编码。
有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets。

#### add_space_variable[[huggingface_hub.HfApi.add_space_variable]]

```python
add_space_variable(repo_id: str, key: str, value: str, description: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8222)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

key (`str`) ：可变键。示例：`"MODEL_REPO_ID"`

value (`str`) ：变量值。示例：`"the_model_repo_id"`。

描述 (`str`) ：变量的描述。示例：`"Model Repo ID of the implemented model"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

添加或更新空间中的变量。变量允许将环境变量设置为空间，而无需对其进行硬编码。
有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables

#### auth_check[[huggingface_hub.HfApi.auth_check]]

```python
auth_check(repo_id: str, repo_type: str | None = None, token: bool | str | None = None, write: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11946)

**参数：**

repo_id (`str`) ：用于检查访问权限的存储库。格式应为`"user/repo_name"`。示例：`"user/my-cool-model"`。 

repo_type (`str`, *可选*) ：存储库的类型。应为 `"model"`、`"dataset"`​​ 或 `"space"` 之一。如果不指定，默认为`"model"`。 

令牌（`Union[bool, str, None]`，*可选*）：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。 

write (`bool`, *可选*) ：如果`True`，则检查用户是否具有存储库的内容写入权限。如果`False`（默认），则仅检查读取访问权限。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [GatedRepoError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.GatedRepoError)- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果存储库不存在、是私有的或用户无权访问，则引发。这个可以
  如果 `repo_id` 或 `repo_type` 不正确，或者存储库是私有的但用户
  未经过身份验证。

- [GatedRepoError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.GatedRepoError) -- 
  如果存储库存在但被限制且用户无权访问它，则会引发该异常。

检查提供的用户令牌是否有权访问 Hugging Face Hub 上的特定存储库。

此方法验证通过提供的令牌进行身份验证的用户是否有权访问指定的
存储库。如果找不到存储库或者用户缺乏访问它所需的权限，
该方法引发适当的异常。

示例：

检查用户是否有权访问存储库：

```python
>>> from huggingface_hub import auth_check
>>> from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

try:
    auth_check("user/my-cool-model")
except GatedRepoError:
    # Handle gated repository error
    print("You do not have permission to access this gated repository.")
except RepositoryNotFoundError:
    # Handle repository not found error
    print("The repository was not found or you do not have access.")
```

在这个例子中：
- 如果用户具有访问权限，则该方法成功完成。
- 如果存储库被限制或不存在，则会引发适当的异常，允许用户
来相应地处理它们。

####batch_bucket_files[[huggingface_hub.HfApi.batch_bucket_files]]

```python
batch_bucket_files(bucket_id: str, add: list[tuple[str | Path | bytes, str]] | None = None, copy: list[tuple[str, str, str, str]] | None = None, delete: list[str] | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L14327)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。add (`list` of `tuple`, *可选*) ：要上传的文件。每个元素都是一个 `(source, destination)` 元组，其中 `source` 是本地文件（`str` 或 `Path`）或原始 `bytes` 内容的路径，`destination` 是存储桶中的路径。

copy (`list` of `tuple`, *可选*) ：通过 xet 哈希复制的文件。每个元素都是一个 `(source_repo_type, source_repo_id, xet_hash, destination)` 元组，其中： - `source_repo_type` 是源存储库的类型：`"model"`、`"dataset"`、`"space"` 或 `"bucket"`。 - `source_repo_id` 是源存储库或存储桶的 ID（例如 `"username/my-model"`）。 - `xet_hash` 是要复制的文件的 xet 哈希值。 - `destination` 是桶中的目标路径。这是服务器端操作 - 不会下载或重新上传数据。

delete (`list` of `str`, *可选*) : 要从存储桶中删除的文件的路径。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

添加、复制和/或删除存储桶中的文件。

这是一个非事务性操作。如果过程中出现错误，可能是部分文件已经上传，
复制或删除，而其他人则没有。示例：
```python
>>> from huggingface_hub import batch_bucket_files

# Upload files
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[
...         ("./model.safetensors", "models/model.safetensors"),
...         (b'{{"key": "value"}}', "config.json"),
...     ],
... )

# Copy xet files from another bucket or repo (server-side, no data transfer)
>>> batch_bucket_files(
...     "username/my-bucket",
...     copy=[
...         ("bucket", "username/source-bucket", "<xethash_1>", "models/model.safetensors"),
...         ("model", "username/my-model", "<xethash_2>", "models/config.safetensors"),
...     ],
... )

# Delete files
>>> batch_bucket_files("username/my-bucket", delete=["old-model.bin"])

# Upload and delete in one batch
>>> batch_bucket_files(
...     "username/my-bucket",
...     add=[("./new.txt", "new.txt")],
...     delete=["old.txt"],
... )
```

####bucket_info[[huggingface_hub.HfApi.bucket_info]]

```python
bucket_info(bucket_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13702)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [BucketInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketInfo)

桶信息。

**加薪：** `BucketNotFoundError` 或 ``or``

- `BucketNotFoundError` -- 如果找不到桶。这可能是因为它不存在，
- ``or`` -- because it is set to `private` 并且您无权访问。

获取有关 Hub 上特定存储桶的信息。

示例：
```python
>>> from huggingface_hub import bucket_info
>>> info = bucket_info(bucket_id="Wauplin/first-bucket")
>>> info.id
'Wauplin/first-bucket'
>>> info.private
False
>>> info.created_at
datetime.datetime(2026, 2, 6, 17, 37, 57, tzinfo=datetime.timezone.utc)
>>> info.size
551879671
>>> info.total_files
12
```

#### cancel_access_request[[huggingface_hub.HfApi.cancel_access_request]]

```python
cancel_access_request(repo_id: str, user: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10782)

**参数：**

repo_id (`str`) : 取消访问请求的repo的id。

user (`str`) ：应取消访问请求的用户的用户名。

repo_type (`str`, *可选*) ：要取消访问请求的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为`model`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。
- `HfHubHTTPError` -- 
  如果集线器上不存在用户，则返回 HTTP 404。
- `HfHubHTTPError` -- 
  如果找不到用户访问请求，则返回 HTTP 404。
- `HfHubHTTPError` -- 
  如果用户访问请求已在待处理列表中，则 HTTP 404。

取消用户对给定门控存储库的访问请求。

取消的请求将返回到待处理列表，用户将失去对存储库的访问权限。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

#### cancel_job[[huggingface_hub.HfApi.cancel_job]]

```python
cancel_job(job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12585)

**参数：**

job_id (`str`) ：作业的 ID。命名空间（`str`，*可选*）：作业运行的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

取消 Hugging Face 基础设施上的计算作业。

####change_discussion_status[[huggingface_hub.HfApi.change_discussion_status]]

```python
change_discussion_status(repo_id: str, discussion_num: int, new_status: Literal['open', 'closed'], token: bool | str | None = None, comment: str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7851)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。

new_status (`str`) ：讨论的新状态，`"open"`或`"closed"`。

comment (`str`, *可选*) ：随状态更改一起发布的可选评论。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [DiscussionStatusChange](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionStatusChange)

状态改变事件

关闭或重新打开讨论或拉取请求。

示例：
```python
>>> new_title = "New title, fixing a typo"
>>> HfApi().rename_discussion(
...     repo_id="username/repo_name",
...     discussion_num=34
...     new_title=new_title
... )
# DiscussionStatusChange(id='deadbeef0000000', type='status-change', ...)

```

> [!提示]
> 引发以下错误：
>
> - [⟦T486⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T487⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### comment_discussion[[huggingface_hub.HfApi.comment_discussion]]

```python
comment_discussion(repo_id: str, discussion_num: int, comment: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7708)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。

comment (`str`) ：要创建的评论内容。评论支持Markdown格式。repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [DiscussionComment](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionComment)

新创建的评论

对给定的讨论创建新评论。

示例：
```python

>>> comment = """
... Hello @otheruser!
...
... # This is a title
...
... **This is bold**, *this is italic* and ~this is strikethrough~
... And [this](http://url) is a link
... """

>>> HfApi().comment_discussion(
...     repo_id="username/repo_name",
...     discussion_num=34
...     comment=comment
... )
# DiscussionComment(id='deadbeef0000000', type='comment', ...)

```

> [!提示]
> 引发以下错误：
>
> - [⟦T502⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T503⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### copy_files[[huggingface_hub.HfApi.copy_files]]

```python
copy_files(source: str, destination: str, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L14001)

**参数：**

源 (`str`) ：作为 `hf://` URI 的源位置。可以是存储桶路径（例如`"hf://buckets/my-bucket/path/to/file"`）或存储库路径（例如`"hf://username/my-model/weights.bin"`、`"hf://datasets/username/my-dataset/data/"`）。目的地 (`str`) ：目标位置作为指向存储桶（例如`"hf://buckets/my-bucket/target/path"`）或存储库（例如`"hf://username/my-model/target/path"`）的`hf://` URI。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** ``ValueError``

- [⟦T518⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果源/目标 URI 无效或者从存储桶复制到存储库。

在集线器上的位置之间复制文件。

将文件从存储桶或存储库（模型、数据集、空间）复制到存储桶或另一个存储库。
支持单个文件和整个文件夹。

复制文件夹时，源路径上的尾随 `/` 使用 rsync 样式语义：复制 *内容*
文件夹到目标中，而不嵌套源文件夹本身。没有尾随 `/`，
源文件夹嵌套在目标文件夹内（如`cp -r`）。

从存储库复制到存储桶时，`.gitattributes` 文件会被自动排除，因为它们
是 git 特定的元数据，与存储桶上下文无关。存储库到存储库的副本在底层使用 [CommitOperationCopy](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy) 并在目标上创建提交
存储库。不支持存储桶到存储库的副本。

> [!警告]
> 服务器端副本只能在同一个[storage region](https://huggingface.co/docs/hub/storage-regions)内工作。

示例：
```python
>>> from huggingface_hub import copy_files

# Copy a single file between buckets
>>> copy_files("hf://buckets/my-bucket/data.bin", "hf://buckets/other-bucket/data.bin")

# Copy a folder into another bucket (nests: backup/models/...)
>>> copy_files("hf://buckets/my-bucket/models", "hf://buckets/other-bucket/backup/")

# Copy folder contents (trailing /): files go directly into backup/
>>> copy_files("hf://buckets/my-bucket/models/", "hf://buckets/other-bucket/backup/")

# Copy a file from a model repo to a bucket
>>> copy_files("hf://username/my-model/model.safetensors", "hf://buckets/my-bucket/")

# Copy an entire dataset to a bucket
>>> copy_files("hf://datasets/username/my-dataset/", "hf://buckets/my-bucket/datasets/")

# Copy files between repositories
>>> copy_files("hf://username/source-model/", "hf://username/dest-model/")

# Copy a file from one repo to another
>>> copy_files("hf://username/source-model/config.json", "hf://username/dest-model/config.json")
```

#### create_branch[[huggingface_hub.HfApi.create_branch]]

```python
create_branch(repo_id: str, branch: str, revision: str | None = None, token: bool | str | None = None, repo_type: str | None = None, exist_ok: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7058)

**参数：**

repo_id (`str`) ：将在其中创建分支的存储库。示例：`"user/my-cool-model"`。 

分支 (`str`) ：要创建的分支的名称。 

revision (`str`, *可选*) : 用于创建分支的 git 修订版本。它可以是分支名称或提交的 OID/SHA，作为十六进制字符串。默认为 `"main"` 分支的头部。 

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请通过`False`。 

repo_type (`str`, *可选*) ：如果在数据集或空间上创建分支，则设置为 `"dataset"` 或 `"space"`；如果标记模型，则设置为 `None` 或 `"model"`。默认为`None`。存在_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果分支已存在，则不会引发错误。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type，私有
  但未经过身份验证或存储库不存在。
- [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) -- 
  如果分支的引用无效。例如：`refs/pr/5` 或“refs/foo/bar”。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  如果分支已存在于存储库中（错误 409）并且 `exist_ok` 是
  设置为`False`。

从指定的修订版开始（默认为 `main`），为 Hub 上的存储库创建一个新分支。
要查找适合您需要的修订版本，您可以使用 [list_repo_refs()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_refs) 或 [list_repo_commits()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_commits)。

#### create_bucket[[huggingface_hub.HfApi.create_bucket]]

```python
create_bucket(bucket_id: str, private: bool | None = None, resource_group_id: str | None = None, region: REPO_REGIONS | None = None, exist_ok: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13602)

**参数：**

bucket_id (`str`) ：命名空间（用户或组织）和存储桶名称，由`/`分隔。如果未提供命名空间，则将在当前用户的命名空间中创建存储桶。

private (`bool`, *可选*) ：是否将存储桶设为私有。如果`None`（默认），则存储桶将是公共的，除非组织默认为私有。resource_group_id (`str`, *可选*) ：要在其中创建存储桶的资源组。资源组仅适用于企业中心组织，并允许定义组织的哪些成员可以访问资源。资源组的 ID 可以在 Hub 上资源页面的 URL 中找到（例如 `"66670e5163145ca562cb1988"`）。要了解有关资源组的更多信息，请参阅 https://huggingface.co/docs/hub/en/security-resource-groups。

区域（`Literal["us", "eu"]`，*可选*）：要在其中创建存储桶的云区域。可以是 `"us"` 或 `"eu"` 之一。如果不指定，则将在默认地域创建桶。需要团队计划或以上。

存在_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果存储桶已存在，则不会引发错误。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [BucketUrl](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketUrl)

新创建的存储桶的 URL，其中包含
`endpoint`、`namespace` 和 `bucket_id` 等属性。

在 Hub 上创建一个存储桶。

示例：
```python
>>> from huggingface_hub import create_bucket

>>> url = create_bucket(bucket_id="my-bucket")
>>> url.bucket_id
'user/my-bucket'
>>> url.url
'https://huggingface.co/buckets/user/my-bucket'
>>> url.uri.to_uri()
'hf://buckets/user/my-bucket'

>>> create_bucket(bucket_id="my-bucket", private=True, exist_ok=True)
BucketUrl(...)

>>> create_bucket(bucket_id="my-bucket", region="us")
BucketUrl(...)
```#### create_collection[[huggingface_hub.HfApi.create_collection]]

```python
create_collection(title: str, namespace: str | None = None, description: str | None = None, private: bool = False, resource_group_id: str | None = None, exists_ok: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10155)

**参数：**

title (`str`) ：要创建的集合的标题。示例：`"Recent models"`。

命名空间（`str`，*可选*）：要创建的集合的命名空间（用户名或组织）。将默认为所有者名称。

描述（`str`，*可选*）：要创建的集合的描述。描述的最大长度为 150 个字符。

private (`bool`, *可选*) : 集合是否应该是私有的。默认为`False`（即公共收藏）。

resource_group_id (`str`, *可选*) ：将集合分配给所属组织的资源组。仅对组织拥有的馆藏有效。资源组ID是24个字符的十六进制字符串。

contains_ok (`bool`, *可选*) ：如果`True`，如果集合已存在，则不会引发错误。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

在 Hub 上创建一个新集合。退货：[Collection](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.Collection)

示例：

```py
>>> from huggingface_hub import create_collection
>>> collection = create_collection(
...     title="ICCV 2023",
...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
... )
>>> collection.slug
"username/iccv-2023-64f9a55bb3115b4f513ec026"
```

#### create_commit[[huggingface_hub.HfApi.create_commit]]

```python
create_commit(repo_id: str, operations: Iterable[CommitOperation], commit_message: str, commit_description: str | None = None, token: str | bool | None = None, repo_type: str | None = None, revision: str | None = None, create_pr: bool | None = None, num_threads: int = 5, parent_commit: str | None = None, run_as_future: bool = False, _hot_reload: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L5081)

**参数：**

repo_id (`str`) ：将在其中创建提交的存储库，例如：`"username/custom_transformers"` 

操作（`Iterable` of `CommitOperation()`）：要包含在提交中的可迭代操作，可以是： - [CommitOperationAdd](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd) 上传文件 - [CommitOperationDelete](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationDelete) 删除文件 - [CommitOperationCopy](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy) 复制文件 操作对象将发生变化，以包含与上传相关的信息。不要在多次提交中重复使用相同的对象。 

commit_message (`str`) ：将创建的提交的摘要（第一行）。 

commit_description (`str`, *可选*) : 将创建的提交的描述 

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。 

create_pr (`boolean`, *可选*) ：是否使用该提交创建拉取请求。默认为`False`。如果未设置 `revision`，则针对 `"main"` 分支打开 PR。如果 `revision` 已设置并且是一个分支，则针对该分支打开 PR。如果设置了 `revision` 并且不是分支名称（例如：提交 oid），则服务器会返回 `RevisionNotFoundError`。 

num_threads (`int`, *可选*) : 上传文件的并发线程数。默认为5。设置为2表示最多同时上传2个文件。 

Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交，则特别有用。run_as_future (`bool`, *可选*) : 是否在后台运行此方法。后台作业按顺序运行，不会阻塞主线程。传递 `run_as_future=True` 将返回一个 [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) 对象。默认为`False`。

**返回：** [CommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitInfo) 或 `Future`

[CommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitInfo) 的实例，包含有关新创建的提交的信息（提交哈希、提交
url、pr url、提交消息...）。如果 `run_as_future=True` 被传递，则返回一个 Future 对象，该对象将
包含执行时的结果。

**加薪：** ``ValueError`` 或 [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

- [⟦T614⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果提交消息为空。
- [⟦T615⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果父提交不是有效的提交 OID。
- [⟦T616⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果提交了包含无效元数据部分的 README.md 文件。在这种情况下，提交将会失败
  尽早，在尝试上传任何文件之前。
- [⟦T617⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果 `create_pr` 是 `True` 并且修订版本既不是 `None` 也不是 `"main"`。
- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type，私有
  但未经过身份验证或存储库不存在。

在给定的存储库中创建提交，根据需要删除和上传文件。> [!警告]
> `CommitOperation`的输入列表将在提交过程中发生变化。不要重复使用相同的对象
> 用于多次提交。

> [!警告]
> `create_commit` 假设该存储库已存在于 Hub 上。如果你得到一个
> 客户端错误404，请确保您已通过身份验证，您的令牌具有所需的权限，
> 并且 `repo_id` 和 `repo_type` 设置正确。如果回购不存在，
> 首先使用 [create_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建它。

> [!警告]
> `create_commit` 仅限于 25k LFS 文件和 1GB 常规文件有效负载。

#### create_discussion[[huggingface_hub.HfApi.create_discussion]]

```python
create_discussion(repo_id: str, title: str, token: bool | str | None = None, description: str | None = None, repo_type: str | None = None, pull_request: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7535)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

title (`str`) ：讨论的标题。它的长度最多为 200 个字符，并且必须至少为 3 个字符。前导和尾随空格将被删除。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。描述（`str`，*可选*）：拉取请求的可选描述。默认为 `"Discussion opened with the huggingface_hub Python library"`

pull_request (`bool`, *可选*) ：是否创建拉取请求或讨论。如果`True`，则创建拉取请求。如果`False`，则创建讨论。默认为 `False`。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

创建讨论或拉取请求。

以编程方式创建的 Pull 请求将处于 `"draft"` 状态。

也可以使用 [HfApi.create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 立即创建包含更改的拉取请求。

返回：[DiscussionWithDetails](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionWithDetails)

> [!提示]
> 引发以下错误：
>
> - [⟦T646⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T647⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### create_inference_endpoint[[huggingface_hub.HfApi.create_inference_endpoint]]

```python
create_inference_endpoint(name: str, repository: str, framework: str, accelerator: str, instance_size: str, instance_type: str, region: str, vendor: str, account_id: str | None = None, min_replica: int = 1, max_replica: int = 1, scaling_metric: InferenceEndpointScalingMetric | None = None, scaling_threshold: float | None = None, scale_to_zero_timeout: int | None = None, revision: str | None = None, task: str | None = None, custom_image: dict | None = None, container_command: list[str] | None = None, container_args: list[str] | None = None, env: dict[str, str] | None = None, secrets: dict[str, str] | None = None, type: InferenceEndpointType | str = <InferenceEndpointType.AUTHENTICATED: 'authenticated'>, domain: str | None = None, path: str | None = None, cache_http_responses: bool | None = None, tags: list[str] | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9343)

**参数：**

name (`str`) ：新推理端点的唯一名称。存储库 (`str`) ：与推理端点关联的模型存储库的名称（例如`"gpt2"`）。

框架（`str`）：用于模型的机器学习框架（例如`"custom"`）。

加速器（`str`）：用于推理的硬件加速器（例如`"cpu"`）。

instance_size (`str`) ：用于托管模型的实例的大小或类型（例如`"x4"`）。

instance_type (`str`) ：将部署推理端点的云实例类型（例如`"intel-icl"`）。

区域 (`str`) ：将在其中创建推理端点的云区域（例如`"us-east-1"`）。

供应商 (`str`) ：将托管推理端点的云提供商或供应商（例如 `"aws"`）。

account_id（`str`，*可选*）：用于将 VPC 链接到私有推理端点（如果适用）的账户 ID。

min_replica (`int`, *可选*) ：推理端点保持运行的最小副本（实例）数量。要启用缩放为零，请将此值设置为 0 并相应调整 `scale_to_zero_timeout`。默认为 1。

max_replica（`int`，*可选*）：推理端点可扩展的最大副本（实例）数量。默认为 1。scaling_metric（`str`或`InferenceEndpointScalingMetric `，*可选*）：缩放的度量参考。提供“pendingRequests”或“hardwareUsage”时。默认为“无”（含义：让 HF Endpoints 服务指定指标）。

scaling_threshold (`float`, *可选*) ：用于触发扩展的扩展指标阈值。未提供缩放指标时将被忽略。默认为“无”（含义：让 HF Endpoints 服务指定阈值）。

scale_to_zero_timeout（`int`，*可选*）：非活动端点缩放到零之前的持续时间（以分钟为单位），或者如果设置为 None 并且 `min_replica` 不为 0，则不会缩放到零。默认为 None。

修订版（`str`，*可选*）：要在推理端点上部署的特定模型修订版（例如`"6c0e6080953db56375760c0471a8c5f2929baf11"`）。

任务（`str`，*可选*）：部署模型的任务（例如`"text-classification"`）。

custom_image（`dict`，*可选*）：用于推理端点的自定义 Docker 映像。如果您想部署在 `text-generation-inference` (TGI) 框架或自定义容器上运行的推理端点（请参阅示例），这非常有用。container_command（`list[str]`，*可选*）：覆盖容器入口点命令（映射到API负载中的`model.command`）。适用于托管引擎映像（例如 vLLM、SGLang）和自定义映像。

container_args (`list[str]`, *可选*) ：附加到容器入口点的参数（映射到 API 负载中的 `model.args`）。适用于托管引擎映像（例如 vLLM、SGLang）和自定义映像。

env (`dict[str, str]`, *可选*) ：要注入到容器环境中的非秘密环境变量。

Secrets (`dict[str, str]`, *可选*) ：要注入到容器环境中的秘密值。

类型（[⟦T685⟧, *optional*) : The type of the Inference Endpoint, which can be ⟦T686⟧ (default), ⟦T687⟧ or ⟦T688⟧. ⟦T689⟧ is deprecated in favor of ⟦T690⟧ and will be removed in a future release.

domain (⟦T691⟧, *optional*) : The custom domain for the Inference Endpoint deployment, if setup the inference endpoint will be available at this domain (e.g. ⟦T692⟧).

path (⟦T693⟧, *optional*) : The custom path to the deployed model, should start with a ⟦T694⟧ (e.g. ⟦T695⟧).

cache_http_responses (⟦T696⟧, *optional*) : Whether to cache HTTP responses from the Inference Endpoint. Defaults to ⟦T697⟧.

tags (⟦T698⟧, *optional*) : A list of tags to associate with the Inference Endpoint.

namespace (⟦T699⟧, *optional*) : The namespace where the Inference Endpoint will be created. Defaults to the current user's namespace.

token (⟦T700⟧ or ⟦T701⟧, *optional*) : A valid user access token (string). Defaults to the locally saved token, which is the recommended method for authentication (see https://huggingface.co/docs/huggingface_hub/quick-start#authentication). To disable authentication, pass ⟦T702⟧.

**Returns:** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关更新的推理端点的信息。

创建一个新的推理端点。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> endpoint = api.create_inference_endpoint(
...     "my-endpoint-name",
...     repository="gpt2",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="cpu",
...     vendor="aws",
...     region="us-east-1",
...     type="authenticated",
...     instance_size="x2",
...     instance_type="intel-icl",
... )
>>> endpoint
InferenceEndpoint(name='my-endpoint-name', status="pending",...)

# Run inference on the endpoint
>>> endpoint.client.text_generation(...)
"..."
```

```python
# Start an Inference Endpoint running Zephyr-7b-beta on TGI
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> endpoint = api.create_inference_endpoint(
...     "aws-zephyr-7b-beta-0486",
...     repository="HuggingFaceH4/zephyr-7b-beta",
...     framework="pytorch",
...     task="text-generation",
...     accelerator="gpu",
...     vendor="aws",
...     region="us-east-1",
...     type="authenticated",
...     instance_size="x1",
...     instance_type="nvidia-a10g",
...     env={
...           "MAX_BATCH_PREFILL_TOKENS": "2048",
...           "MAX_INPUT_LENGTH": "1024",
...           "MAX_TOTAL_TOKENS": "1512",
...           "MODEL_ID": "/repository"
...         },
...     custom_image={
...         "healthRoute": "/health",
...         "url": "ghcr.io/huggingface/text-generation-inference:1.1.0",
...     },
...    secrets={"MY_SECRET_KEY": "secret_value"},
...    tags=["dev", "text-generation"],
... )
```

```python
# Start an Inference Endpoint running ProsusAI/finbert while scaling to zero in 15 minutes
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> endpoint = api.create_inference_endpoint(
...     "finbert-classifier",
...     repository="ProsusAI/finbert",
...     framework="pytorch",
...     task="text-classification",
...     min_replica=0,
...     scale_to_zero_timeout=15,
...     accelerator="cpu",
...     vendor="aws",
...     region="us-east-1",
...     type="authenticated",
...     instance_size="x2",
...     instance_type="intel-icl",
... )
>>> endpoint.wait(timeout=300)
# Run inference on the endpoint
>>> endpoint.client.text_generation(...)
TextClassificationOutputElement(label='positive', score=0.8983615040779114)
```

#### create_inference_endpoint_from_catalog[[huggingface_hub.HfApi.create_inference_endpoint_from_catalog]]

```python
create_inference_endpoint_from_catalog(repo_id: str, name: str | None = None, accelerator: Literal['cpu', 'gpu', 'neuron'] | str | None = None, token: bool | str | None = None, namespace: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9605)

**参数：**

repo_id (`str`) ：目录中要部署为推理端点的模型的 ID。

name（`str`，*可选*）：新推理端点的唯一名称。如果未提供，将生成一个随机名称。加速器（`str`，*可选*）：用于推理的硬件加速器。可能的值包括 `"cpu"`、`"gpu"` 和 `"neuron"`。如果未提供，服务器将使用适合该模型的默认值。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。

命名空间（`str`，*可选*）：将在其中创建推理端点的命名空间。默认为当前用户的命名空间。

**返回：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关新推理端点的信息。

从拥抱面部推理目录中的模型创建新的推理端点。

推理目录的目标是提供针对推理优化的模型的精选列表
并对其默认配置进行了测试。请参阅 https://endpoints.huggingface.co/catalog 获取列表
目录中的可用型号。

> [!警告]
> `create_inference_endpoint_from_catalog` 处于实验阶段。其 API 将来可能会发生变化。请提供反馈
> 如果您有任何建议或要求。#### create_pull_request[[huggingface_hub.HfApi.create_pull_request]]

```python
create_pull_request(repo_id: str, title: str, token: bool | str | None = None, description: str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7624)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

title (`str`) ：讨论的标题。它的长度最多为 200 个字符，并且必须至少为 3 个字符。前导和尾随空格将被删除。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

描述（`str`，*可选*）：拉取请求的可选描述。默认为 `"Discussion opened with the huggingface_hub Python library"`

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

创建拉取请求。以编程方式创建的 Pull 请求将处于 `"draft"` 状态。

也可以使用 [HfApi.create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 立即创建包含更改的 Pull 请求；

这是 [HfApi.create_discussion()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_discussion) 的包装。

返回：[DiscussionWithDetails](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionWithDetails)> [!提示]
> 引发以下错误：
>
> - [⟦T728⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T729⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### create_repo[[huggingface_hub.HfApi.create_repo]]

```python
create_repo(repo_id: str, token: str | bool | None = None, private: bool | None = None, visibility: RepoVisibility_T | None = None, repo_type: str | None = None, exist_ok: bool = False, resource_group_id: str | None = None, region: REPO_REGIONS | None = None, space_sdk: str | None = None, space_hardware: SpaceHardware | None = None, space_storage: SpaceStorage | None = None, space_sleep_time: int | None = None, space_secrets: list[dict[str, str]] | None = None, space_variables: list[dict[str, str]] | None = None, space_volumes: list[Volume] | None = None, space_template: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4635)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

private (`bool`, *可选*) ：是否将存储库设为私有。如果`None`（默认），则存储库将是公开的，除非组织默认为私有。如果存储库已存在，则忽略此值。不能与`visibility`一起通过。可见性（`Literal["public", "private", "protected"]`，*可选*）：存储库的可见性。对于空间，可以是 `"public"` 或 `"private"`，或者 `"protected"`。如果`None`（默认），则存储库将是公开的，除非组织默认为私有。如果存储库已存在，则忽略此值。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

存在_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果存储库已存在，则不会引发错误。

resource_group_id (`str`, *可选*) ：在其中创建存储库的资源组。资源组仅适用于企业中心组织，并允许定义组织的哪些成员可以访问资源。资源组的 ID 可以在 Hub 上资源页面的 URL 中找到（例如 `"66670e5163145ca562cb1988"`）。要了解有关资源组的更多信息，请参阅 https://huggingface.co/docs/hub/en/security-resource-groups。

区域（`Literal["us", "eu"]`，*可选*）：在其中创建存储库的云区域。可以是 `"us"` 或 `"eu"` 之一。如果未指定，则将在默认区域中创建存储库。需要团队计划或以上。space_sdk (`str`, *可选*) ：如果 repo_type 为“space”，则选择要使用的 SDK。可以是“streamlit”、“gradio”、“docker”或“static”。

space_hardware（`SpaceHardware`或`str`，*可选*）：如果repo_type为“space”，则选择硬件。完整列表请参见[SpaceHardware](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceHardware)。

space_storage（`SpaceStorage`或`str`，*可选*）：持久存储层的选择。示例：`"small"`。完整列表请参见[SpaceStorage](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceStorage)。

space_sleep_time (`int`, *可选*) ：空间进入睡眠状态之前等待的不活动秒数。如果您不希望 Space 休眠（升级硬件的默认行为），请设置为 `-1`。对于免费硬件，您无法配置睡眠时间（值固定为不活动的 48 小时）。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-gpus#sleep-time。

space_secrets (`list[dict[str, str]]`, *可选*) ：要在空间中设置的密钥列表。每个项目的格式为 `{"key": ..., "value": ..., "description": ...}`，其中描述是可选的。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets。space_variables (`list[dict[str, str]]`, *可选*) ：要在空间中设置的公共环境变量列表。每个项目的格式为 `{"key": ..., "value": ..., "description": ...}`，其中描述是可选的。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables。

space_volumes (`list[Volume]`, *可选*) ：创建时要挂载到空间中的 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 对象列表。每个卷都有一个 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、`source`（存储库或存储桶 ID）、`mount_path`（容器内的路径）和可选`revision`、`read_only` 和 `path` 字段。仅当 repo_type 为“space”时适用。

space_template (`str`, *可选*) ：从官方模板中播种新空间。可以是模板存储库 ID（例如 `"SpacesExamples/jupyterlab"`）或其短名称（例如 `"JupyterLab"`）。使用 [HfApi.list_space_templates()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_space_templates) 列出可用的模板。仅当 repo_type 为“space”时适用。如果建议模板为私有且未显式设置可见性，则空间将创建为私有。

**返回：** [RepoUrl](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.RepoUrl)

新创建的存储库的 URL。 Value 是 `str` 的子类，包含
`endpoint`、`repo_type` 和 `repo_id` 等属性。

在 HuggingFace Hub 上创建一个空存储库。#### create_scheduled_job[[huggingface_hub.HfApi.create_scheduled_job]]

```python
create_scheduled_job(image: str, command: list[str], schedule: str, suspend: bool | None = None, concurrency: bool | None = None, env: dict[str, Any] | None = None, secrets: dict[str, Any] | None = None, flavor: JobHardware | str | None = None, timeout: int | float | str | None = None, name: str | None = None, labels: dict[str, str] | None = None, volumes: list[Volume] | None = None, expose: list[int] | None = None, resource_group_id: str | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12827)

**参数：**

image (`str`) ：要使用的 Docker 镜像。示例：`"ubuntu"`、`"python:3.12"`、`"pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"`。来自空间的图像示例：`"hf.co/spaces/lhoestq/duckdb"`。 

命令 (`list[str]`) ：要运行的命令。示例：`["echo", "hello"]`。 

Schedule (`str`) ：“@annually”、“@yearly”、“@monthly”、“@weekly”、“@daily”、“@hourly”或 CRON 计划表达式之一（例如，“0 9 * * 1”表示每周一上午 9 点）。 

暂停（`bool`，*可选*）：如果为 True，则计划的作业将暂停（暂停）。  默认为 False。 

concurrency (`bool`, *可选*) ：如果为 True，则此作业的多个实例可以同时运行。默认为 False。 

env (`dict[str, Any]`, *可选*) ：定义作业的环境变量。 

Secrets (`dict[str, Any]`, *可选*) ：定义作业的秘密环境变量。 

风味（`str`，*可选*）：硬件风味。请参阅`JobHardware`了解可能的值。默认为`"cpu-basic"`。 

timeout (`Union[int, float, str]`，*可选*)：作业的最大持续时间：int，包含 s（秒，默认）、m（分钟）、h（小时）或 d（天）。示例：`300` 或 `"5m"` 5 分钟。name (`str`, *可选*) ：计划作业的名称。存储为 `name` 标签。不能与`labels`中的`name`密钥一起传递。名称不必是唯一的。默认为从图像和命令派生的名称（带有短哈希后缀）。 

labels (`dict[str, str]`，*可选*)：附加到作业的标签（键值对）。 

卷（`list[Volume]`，*可选*）：拥抱 Face Buckets 或 Repos 以作为卷安装在作业容器中。每个卷都是 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 和 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、`source`（例如 `"username/my-bucket"`），以及`mount_path`（例如`"/data"`）。 

hide (`list[int]`, *可选*) ：通过作业代理公开的容器端口。每个列出的端口都可以在公共作业域上访问（例如`https://<job_id>--8000.hf.jobs`）。访问始终需要 HF 令牌，该令牌具有对作业命名空间的读取访问权限。 

resource_group_id（`str`，*可选*）：在其中创建计划作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。如果未提供，则计划的作业将在任何资源组之外创建。 

命名空间（`str`，*可选*）：将在其中创建作业的命名空间。默认为当前用户的命名空间。token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

在 Hugging Face 基础设施上创建计划的计算作业。

示例：

创建您的第一个预定作业：

```python
>>> from huggingface_hub import create_scheduled_job
>>> create_scheduled_job(image="python:3.12", command=["python", "-c" ,"print('Hello from HF compute!')"], schedule="@hourly")
```

使用 CRON 计划表达式：

```python
>>> from huggingface_hub import create_scheduled_job
>>> create_scheduled_job(image="python:3.12", command=["python", "-c" ,"print('this runs every 5min')"], schedule="*/5 * * * *")
```

创建计划的 GPU 作业：

```python
>>> from huggingface_hub import create_scheduled_job
>>> image = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
>>> command = ["python", "-c", "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"]
>>> create_scheduled_job(image, command, flavor="a10g-small", schedule="@hourly")
```

#### create_scheduled_uv_job[[huggingface_hub.HfApi.create_scheduled_uv_job]]

```python
create_scheduled_uv_job(script: str, script_args: list[str] | None = None, schedule: str, suspend: bool | None = None, concurrency: bool | None = None, dependencies: list[str] | None = None, python: str | None = None, image: str | None = None, env: dict[str, Any] | None = None, secrets: dict[str, Any] | None = None, flavor: JobHardware | str | None = None, timeout: int | float | str | None = None, name: str | None = None, labels: dict[str, str] | None = None, volumes: list[Volume] | None = None, expose: list[int] | None = None, resource_group_id: str | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13220)

**参数：**

script (`str`) ：UV脚本的路径或URL，或者命令。 

script_args (`list[str]`, *可选*) ：传递给脚本或命令的参数。 

Schedule (`str`) ：“@annually”、“@yearly”、“@monthly”、“@weekly”、“@daily”、“@hourly”或 CRON 计划表达式之一（例如，“0 9 * * 1”表示每周一上午 9 点）。 

暂停（`bool`，*可选*）：如果为 True，则计划的作业将暂停（暂停）。  默认为 False。 

concurrency (`bool`, *可选*) ：如果为 True，则此作业的多个实例可以同时运行。默认为 False。依赖项（`list[str]`，*可选*）：用于运行 UV 脚本的依赖项。 

python (`str`, *可选*) ：使用特定的Python版本。默认值为 3.12。 

镜像（`str`，*可选*，默认为“ghcr.io/astral-sh/uv --python3.12-bookworm”）：使用安装了`uv`的自定义 Docker 镜像。 

env (`dict[str, Any]`, *可选*) ：定义作业的环境变量。 

Secrets (`dict[str, Any]`, *可选*) ：定义作业的秘密环境变量。 

风味（`str`，*可选*）：硬件风味。请参阅 `JobHardware` 了解可能的值。默认为 `"cpu-basic"`。 

超时（`Union[int, float, str]`，*可选*）：作业的最大持续时间：int，包含 s（秒，默认）、m（分钟）、h（小时）或 d（天）。示例：`300` 或 `"5m"` 5 分钟。 

name (`str`, *可选*) ：计划作业的名称。存储为 `name` 标签。不能与 `labels` 中的 `name` 密钥一起传递。名称不必是唯一的。默认为从脚本及其参数派生的名称（带有短哈希后缀）。 

labels (`dict[str, str]`，*可选*)：附加到作业的标签（键值对）。卷（`list[Volume]`，*可选*）：拥抱 Face Buckets 或 Repos 以作为卷安装在作业容器中。每个卷都是 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 和 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、`source`（例如 `"username/my-bucket"`），以及`mount_path`（例如`"/data"`）。 

hide (`list[int]`, *可选*) ：通过作业代理公开的容器端口。每个列出的端口都可以在公共作业域上访问（例如`https://<job_id>--8000.hf.jobs`）。访问始终需要 HF 令牌，该令牌具有对作业命名空间的读取访问权限。 

resource_group_id（`str`，*可选*）：在其中创建计划作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。如果未提供，则计划的作业将在任何资源组之外创建。 

命名空间（`str`，*可选*）：将在其中创建作业的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。在 Hugging Face 基础设施上运行 UV 脚本作业。

示例：

从 URL 安排脚本：

```python
>>> from huggingface_hub import create_scheduled_uv_job
>>> script = "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py"
>>> script_args = ["--model_name_or_path", "Qwen/Qwen2-0.5B", "--dataset_name", "trl-lib/Capybara", "--push_to_hub"]
>>> create_scheduled_uv_job(script, script_args=script_args, dependencies=["trl"], flavor="a10g-small", schedule="@weekly")
```

安排本地脚本：

```python
>>> from huggingface_hub import create_scheduled_uv_job
>>> script = "my_sft.py"
>>> script_args = ["--model_name_or_path", "Qwen/Qwen2-0.5B", "--dataset_name", "trl-lib/Capybara", "--push_to_hub"]
>>> create_scheduled_uv_job(script, script_args=script_args, dependencies=["trl"], flavor="a10g-small", schedule="@weekly")
```

安排命令：

```python
>>> from huggingface_hub import create_scheduled_uv_job
>>> script = "lighteval"
>>> script_args= ["endpoint", "inference-providers", "model_name=openai/gpt-oss-20b,provider=auto", "lighteval|gsm8k|0|0"]
>>> create_scheduled_uv_job(script, script_args=script_args, dependencies=["lighteval"], flavor="a10g-small", schedule="@weekly")
```

#### create_tag[[huggingface_hub.HfApi.create_tag]]

```python
create_tag(repo_id: str, tag: str, tag_message: str | None = None, revision: str | None = None, token: bool | str | None = None, repo_type: str | None = None, exist_ok: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7190)

**参数：**

repo_id (`str`) ：将在其中标记提交的存储库。示例：`"user/my-cool-model"`。 

tag (`str`) ：要创建的标签的名称。 

tag_message (`str`, *可选*) ：要创建的标签的描述。 

revision (`str`, *可选*) : 标签的 git 修订版。它可以是分支名称或提交的 OID/SHA，作为十六进制字符串。还支持简写（前 7 个字符）。默认为 `"main"` 分支的头部。 

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：如果标记数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果标记模型，则设置为 `None` 或 `"model"`。默认为`None`。存在_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果标签已存在，则不会引发错误。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type，私有
  但未经过身份验证或存储库不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果在存储库中未找到修订版（错误 404）。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  如果分支已存在于存储库中（错误 409）并且 `exist_ok` 是
  设置为`False`。

在 Hub 上标记存储库的给定提交。

#### create_webhook[[huggingface_hub.HfApi.create_webhook]]

```python
create_webhook(url: str | None = None, job_id: str | None = None, watched: list[dict | WebhookWatchedItem], domains: list[constants.WEBHOOK_DOMAIN_T] | None = None, secret: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11104)

**参数：**

url (`str`) ：将有效负载发送到的 URL。

job_id (`str`) ：使用环境变量 WEBHOOK_PAYLOAD 中的 Webhook 负载触发的源作业的 ID。为了方便起见，还提供了其他环境变量：WEBHOOK_REPO_ID、WEBHOOK_REPO_TYPE 和 WEBHOOK_SECRET。

Watched (`list[WebhookWatchedItem]`) ：Webhook 监视的 [WebhookWatchedItem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookWatchedItem) 列表。它可以是用户、组织、模型、数据集或空间。观看的项目也可以作为普通字典提供。

域名（`list[Literal["repo", "discussion"]]`，可选）：要观看的域名列表。它可以是“repo”、“discussion”或两者兼而有之。Secret（`str`，可选）：用于签署有效负载的秘密。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [WebhookInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

有关新创建的 Webhook 的信息。

创建一个新的网络钩子。

Webhook 可以将有效负载发送到 URL，也可以触发作业在 Hugging Face 基础设施上运行。
应使用 `url` 或 `job_id` 之一调用此函数，但不能同时使用两者。

示例：

创建一个将有效负载发送到 URL 的 Webhook

```python
>>> from huggingface_hub import create_webhook
>>> payload = create_webhook(
...     watched=[{"type": "user", "name": "julien-c"}, {"type": "org", "name": "HuggingFaceH4"}],
...     url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
...     domains=["repo", "discussion"],
...     secret="my-secret",
... )
>>> print(payload)
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    job=None,
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    domains=["repo", "discussion"],
    secret="my-secret",
    disabled=False,
)
```

运行作业，然后创建触发此作业的 Webhook

```python
>>> from huggingface_hub import create_webhook, run_job
>>> job = run_job(
...     image="ubuntu",
...     command=["bash", "-c", r"echo An event occurred in $WEBHOOK_REPO_ID: $WEBHOOK_PAYLOAD"],
... )
>>> payload = create_webhook(
...     watched=[{"type": "user", "name": "julien-c"}, {"type": "org", "name": "HuggingFaceH4"}],
...     job_id=job.id,
...     domains=["repo", "discussion"],
...     secret="my-secret",
... )
>>> print(payload)
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    url=None,
    job=JobSpec(
        docker_image='ubuntu',
        space_id=None,
        command=['bash', '-c', 'echo An event occurred in $WEBHOOK_REPO_ID: $WEBHOOK_PAYLOAD'],
        arguments=[],
        environment={},
        secrets=[],
        flavor='cpu-basic',
        timeout=None,
        tags=None,
        arch=None
    ),
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    domains=["repo", "discussion"],
    secret="my-secret",
    disabled=False,
)
```

#### dataset_info[[huggingface_hub.HfApi.dataset_info]]

```python
dataset_info(repo_id: str, revision: str | None = None, timeout: float | None = None, files_metadata: bool = False, expand: list[ExpandDatasetProperty_T] | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3332)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

revision (`str`, *可选*) ：从中获取信息的数据集存储库的修订版本。

timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。files_metadata (`bool`, *可选*) ：是否检索存储库中文件的元数据（大小、LFS 元数据等）。默认为 `False`。

Expand (`list[ExpandDatasetProperty_T]`, *可选*) ：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递`files_metadata`，则无法使用此参数。可能的值为 `"author"`、`"cardData"`、`"citation"`、`"createdAt"`、`"disabled"`、`"description"`、`"downloads"`、`"downloadsAllTime"`、`"gated"`、 `"lastModified"`、`"likes"`、`"mainSize"`、`"paperswithcode_id"`、`"private"`、`"siblings"`、`"sha"`、`"tags"`、`"trendingScore"`、 `"usedStorage"`和`"resourceGroup"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [hf_api.DatasetInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DatasetInfo)

数据集存储库信息。

在 Huggingface.co 上获取有关某一特定数据集的信息。

如果您传递可接受的令牌，则数据集可以是私有的。> [!提示]
> 引发以下错误：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果找不到要下载的版本。

#### 删除分支[[huggingface_hub.HfApi.delete_branch]]

```python
delete_branch(repo_id: str, branch: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7138)

**参数：**

repo_id (`str`) ：将删除其中分支的存储库。示例：`"user/my-cool-model"`。 

分支 (`str`) ：要删除的分支的名称。 

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：如果在数据集或空间上创建分支，则设置为 `"dataset"` 或 `"space"`；如果标记模型，则设置为 `None` 或 `"model"`。默认为`None`。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type，私有
  但未经过身份验证或存储库不存在。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  如果尝试删除受保护的分支。例如：`main`无法删除。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  如果尝试删除不存在的分支。

从 Hub 上的存储库中删除分支。

#### 删除桶[[huggingface_hub.HfApi.delete_bucket]]

```python
delete_bucket(bucket_id: str, missing_ok: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13797)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。

missing_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果存储桶不存在，则不会引发错误。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `BucketNotFoundError`

- `BucketNotFoundError` -- 如果找不到存储桶且`missing_ok` 设置为`False`（默认）。

从 Hub 中删除存储桶。

示例：
```python
>>> from huggingface_hub import delete_bucket
>>> delete_bucket(bucket_id="Wauplin/first-bucket")
>>> delete_bucket(bucket_id="Wauplin/first-bucket", missing_ok=True)
```

#### delete_collection[[huggingface_hub.HfApi.delete_collection]]

```python
delete_collection(collection_slug: str, missing_ok: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10337)

**参数：**collection_slug (`str`) ：要删除的集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

missing_ok (`bool`, *可选*) ：如果`True`，如果集合不存在，则不会引发错误。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

删除 Hub 上的集合。

示例：

```py
>>> from huggingface_hub import delete_collection
>>> collection = delete_collection("username/useless-collection-64f9a55bb3115b4f513ec026", missing_ok=True)
```

> [!警告]
> 这是不可恢复的操作。已删除的集合无法恢复。

#### delete_collection_item[[huggingface_hub.HfApi.delete_collection_item]]

```python
delete_collection_item(collection_slug: str, item_object_id: str, missing_ok: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10512)

**参数：**

collection_slug (`str`) ：要更新的集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

item_object_id (`str`) ：集合中项目的 ID。这不是 Hub 上项目的 ID（repo_id 或论文 ID）。它必须从 [CollectionItem](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.CollectionItem) 对象中检索。示例：`collection.items[0].item_object_id`。

missing_ok (`bool`, *可选*) ：如果`True`，如果该项目不存在，则不会引发错误。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

从集合中删除项目。

示例：

```py
>>> from huggingface_hub import get_collection, delete_collection_item

# Get collection first
>>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")

# Delete item based on its ID
>>> delete_collection_item(
...     collection_slug="TheBloke/recent-models-64f9a55bb3115b4f513ec026",
...     item_object_id=collection.items[-1].item_object_id,
... )
```

####删除文件[[huggingface_hub.HfApi.delete_file]]

```python
delete_file(path_in_repo: str, repo_id: str, token: str | bool | None = None, repo_type: str | None = None, revision: str | None = None, commit_message: str | None = None, commit_description: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6082)

**参数：**

path_in_repo (`str`) ：存储库中的相对文件路径，例如：`"checkpoints/1fec34a/weights.bin"`

repo_id (`str`) ：要从中删除文件的存储库，例如：`"username/custom_transformers"`

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

repo_type (`str`, *可选*) ：如果文件位于数据集或空间中，则设置为 `"dataset"` 或 `"space"`；如果文件位于模型中，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。commit_message (`str`, *可选*) ：生成的提交的摘要/标题/第一行。默认为`f"Delete {path_in_repo} with huggingface_hub"`。

commit_description (`str` *可选*) : 生成的提交的描述

create_pr (`boolean`, *可选*) ：是否使用该提交创建拉取请求。默认为`False`。如果未设置 `revision`，则针对 `"main"` 分支打开 PR。如果 `revision` 已设置并且是一个分支，则针对该分支打开 PR。如果设置了 `revision` 并且不是分支名称（例如：提交 oid），则服务器返回 `RevisionNotFoundError`。

Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交，则特别有用。

删除给定存储库中的文件。> [!提示]
> 引发以下错误：
>
> - [⟦T1002⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T1003⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果找不到要下载的版本。
> - [EntryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError)
> 如果找不到要下载的文件。

####删除文件[[huggingface_hub.HfApi.delete_files]]

```python
delete_files(repo_id: str, delete_patterns: list[str], token: bool | str | None = None, repo_type: str | None = None, revision: str | None = None, commit_message: str | None = None, commit_description: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6169)

**参数：**

repo_id (`str`) ：要从中删除文件夹的存储库，例如：`"username/custom_transformers"`

delete_patterns (`list[str]`) ：要删除的文件或文件夹列表。每个字符串可以是文件路径、文件夹路径或通配符模式。模式是标准通配符（通配符模式），如文档[here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)所述。模式匹配基于[⟦T1008⟧](https://docs.python.org/3/library/fnmatch.html)。请注意，`fnmatch` 跨路径边界匹配 `*`，这与传统的 Unix shell 通配不同。例如。 `["file.txt", "folder/", "data/*.parquet"]`令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。到存储的令牌。

repo_type (`str`, *可选*) ：要从中删除文件的存储库的类型。可以是 `"model"`、`"dataset"` 或 `"space"`。默认为`"model"`。

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

commit_message (`str`, *可选*) ：生成的提交的摘要（第一行）。默认为`f"Delete files using huggingface_hub"`。

commit_description (`str` *可选*) ：生成的提交的描述。

create_pr (`boolean`, *可选*) ：是否使用该提交创建拉取请求。默认为`False`。如果未设置`revision`，则针对`"main"`分支打开PR。如果 `revision` 已设置并且是一个分支，则针对该分支打开 PR。如果设置了 `revision` 并且不是分支名称（例如：提交 oid），则服务器会返回 `RevisionNotFoundError`。Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` ​​确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交，则特别有用。

从 Hub 上的存储库中删除文件。

如果提供了文件夹路径，则整个文件夹将被删除
它包含的所有文件。

#### 删除文件夹[[huggingface_hub.HfApi.delete_folder]]

```python
delete_folder(path_in_repo: str, repo_id: str, token: bool | str | None = None, repo_type: str | None = None, revision: str | None = None, commit_message: str | None = None, commit_description: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6248)

**参数：**

path_in_repo (`str`) ：存储库中的相对文件夹路径，例如：`"checkpoints/1fec34a"`。

repo_id (`str`) ：要从中删除文件夹的存储库，例如：`"username/custom_transformers"`令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。到存储的令牌。

repo_type（`str`，*可选*）：如果文件夹位于数据集或空间中，则设置为 `"dataset"` 或 `"space"`；如果位于模型中，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

commit_message (`str`, *可选*) ：生成的提交的摘要/标题/第一行。默认为`f"Delete folder {path_in_repo} with huggingface_hub"`。

commit_description (`str` *可选*) ：生成的提交的描述。

create_pr (`boolean`, *可选*) ：是否使用该提交创建拉取请求。默认为 `False`。如果未设置`revision`，则针对`"main"`分支打开PR。如果 `revision` 已设置并且是一个分支，则针对该分支打开 PR。如果设置了 `revision` 并且不是分支名称（例如：提交 oid），则服务器会返回 `RevisionNotFoundError`。Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交，则特别有用。

删除给定存储库中的文件夹。

[create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 方法的简单包装。

#### delete_inference_endpoint[[huggingface_hub.HfApi.delete_inference_endpoint]]

```python
delete_inference_endpoint(name: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9902)

**参数：**

name (`str`) ：要删除的推理端点的名称。

命名空间（`str`，*可选*）：推理端点所在的命名空间。默认为当前用户。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

删除推理端点。此操作不可逆。如果您不想为推理端点付费，最好选择
使用[pause_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint)暂停它或使用[scale_to_zero_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint)将其缩放到零。

为了方便起见，您还可以使用 [InferenceEndpoint.delete()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.delete) 删除推理端点。

#### delete_repo[[huggingface_hub.HfApi.delete_repo]]

```python
delete_repo(repo_id: str, token: str | bool | None = None, repo_type: str | None = None, missing_ok: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4853)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。

missing_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果repo不存在，则不会引发错误。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到要删除的存储库并且 `missing_ok` 设置为 False（默认）。

从 HuggingFace Hub 中删除存储库。注意：这是不可逆转的。#### 删除_scheduled_job[[huggingface_hub.HfApi.delete_scheduled_job]]

```python
delete_scheduled_job(scheduled_job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13045)

**参数：**

Scheduled_job_id (`str`) ：计划作业的 ID。 

namespace (`str`, *可选*) ：计划作业所在的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

删除 Hugging Face 基础设施上计划的计算作业。

#### delete_space_secret[[huggingface_hub.HfApi.delete_space_secret]]

```python
delete_space_secret(repo_id: str, key: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8136)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

key (`str`) ：秘密密钥。示例：`"GITHUB_API_KEY"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

从空间中删除秘密。秘密允许为空间设置秘密密钥或令牌，而无需对其进行硬编码。
有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets。

#### 删除_空间_存储[[huggingface_hub.HfApi.删除_空间_存储]]

```python
delete_space_storage(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9161)

**参数：**

repo_id (`str`) ：要更新的空间的 ID。示例：`"open-llm-leaderboard/open_llm_leaderboard"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关空间的运行时信息，包括空间阶段和硬件。

**加薪：** `BadRequestError`

- `BadRequestError` -- 
  如果空间没有持久存储。

删除空间的持久存储。

> [!警告]
> `delete_space_storage` 已弃用，并将在 2.0 版本中删除。请使用 [delete_space_volumes()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_space_volumes) 代替。

#### delete_space_variable[[huggingface_hub.HfApi.delete_space_variable]]

```python
delete_space_variable(repo_id: str, key: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8263)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

key (`str`) : 可变键。示例：`"MODEL_REPO_ID"`令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

从空间中删除变量。

变量允许将环境变量设置为空间，而无需对其进行硬编码。
有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables

#### 删除_空间_卷[[huggingface_hub.HfApi.删除_空间_卷]]

```python
delete_space_volumes(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9246)

**参数：**

repo_id (`str`) ：要更新的空间的 ID。示例：`"username/my-space"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `BadRequestError`

- `BadRequestError` -- 
  如果空间没有附加卷。

从空间中删除所有卷。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.delete_space_volumes("username/my-space")
```

####删除_标签[[huggingface_hub.HfApi.delete_tag]]

```python
delete_tag(repo_id: str, tag: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7264)

**参数：**repo_id (`str`) ：将删除标签的存储库。示例：`"user/my-cool-model"`。 

tag (`str`) ：要删除的标签的名称。 

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：如果标记数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果标记模型，则设置为 `None` 或 `"model"`。默认为`None`。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type，私有
  但未经过身份验证或存储库不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果没有找到标签。

从 Hub 上的存储库中删除标签。

#### delete_webhook[[huggingface_hub.HfApi.delete_webhook]]

```python
delete_webhook(webhook_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11433)

**参数：**

webhook_id (`str`) ：要删除的 webhook 的唯一标识符。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `None`

删除网络钩子。

示例：
```python
>>> from huggingface_hub import delete_webhook
>>> delete_webhook("654bbbc16f2ec14d77f109cc")
```

####disable_space_dev_mode[[huggingface_hub.HfApi.disable_space_dev_mode]]

```python
disable_space_dev_mode(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8510)

**参数：**

repo_id (`str`) ：禁用开发模式的空间 ID。示例：`"Salesforce/BLIP2"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关您的空间的运行时信息。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) 或 [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到您的空间（错误 404）。很可能是错误的 repo_id 或您的空间是私人的，但您
  未经过身份验证。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  403 Forbidden：只有空间的所有者才能设置开发模式。如果你想处理一个你不想处理的空间
  自己的，可以通过打开讨论来询问所有者或复制空间。
- [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) -- 
  如果您的空间是静态空间。静态空间始终运行且从不计费。如果你想隐藏
  静态空间，可以设置为私有。

在空间上禁用开发模式。

Spaces 开发模式可简化应用程序的调试，并允许您更快地迭代 Spaces
重新启动应用程序而不停止 Space 容器本身。此功能可作为
PRO 或团队和企业计划。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-dev-mode。

####disable_webhook[[huggingface_hub.HfApi.disable_webhook]]

```python
disable_webhook(webhook_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11380)

**参数：**

webhook_id (`str`) ：要禁用的 webhook 的唯一标识符。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [WebhookInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

有关禁用的 Webhook 的信息。

禁用 Webhook（使其“禁用”）。

示例：
```python
>>> from huggingface_hub import disable_webhook
>>> disabled_webhook = disable_webhook("654bbbc16f2ec14d77f109cc")
>>> disabled_webhook
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    jon=None,
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    domains=["repo", "discussion"],
    secret="my-secret",
    disabled=True,
)
```

#### download_bucket_files[[huggingface_hub.HfApi.download_bucket_files]]

```python
download_bucket_files(bucket_id: str, files: list[tuple[str | BucketFile, str | Path]], raise_on_missing_files: bool = False, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L14629)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。

files (`list[tuple[Union[str, BucketFile], Union[str, Path]]]`) ：作为元组列表下载的文件（源，目标）。有关格式详细信息，请参阅上面的描述。

raise_on_missing_files (`bool`, *可选*) ：如果`True`，则当存储桶中不存在请求的文件时引发`EntryNotFoundError`。如果 `False`（默认），则会跳过丢失的文件并发出警告。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

从存储桶下载文件。文件输入是 `(remote file, local file)` 元组的列表，其中 `remote file` 是文件的路径
在存储桶或[BucketFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketFile)对象中，`local file`是本地文件系统上的目标路径。
当传递[BucketFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketFile)对象（从[list_bucket_tree()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_bucket_tree)获得）时，该方法将跳过元数据
获取步骤并直接下载文件。

示例：
```python
>>> from huggingface_hub import download_bucket_files

>>> download_bucket_files(
...     bucket_id="username/my-bucket",
...     files=[
...         ("models/model.safetensors", "./local/model.safetensors"),
...         ("config.json", "./local/config.json"),
...     ],
... )
```

```python
>>> from huggingface_hub import download_bucket_files

>>> parquet_files = [file for file in list_bucket_tree(bucket_id="username/my-bucket") if file.path.endswith(".parquet")]
>>> download_bucket_files(
...     bucket_id="username/my-bucket",
...     files=[(file, f"./local/{file.path}") for file in parquet_files],
... )
```

####duplicate_repo[[huggingface_hub.HfApi.duplicate_repo]]

```python
duplicate_repo(from_id: str, to_id: str | None = None, repo_type: str | None = None, private: bool | None = None, visibility: RepoVisibility_T | None = None, token: bool | str | None = None, exist_ok: bool = False, space_hardware: SpaceHardware | None = None, space_storage: SpaceStorage | None = None, space_sleep_time: int | None = None, space_secrets: list[dict[str, str]] | None = None, space_variables: list[dict[str, str]] | None = None, space_volumes: list[Volume] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8838)

**参数：**

from_id (`str`) ：要复制的存储库的 ID。示例：`"openai/gdpval"`。

to_id (`str`, *可选*) ：新存储库的 ID。示例：`"myorg/my-gdpval"`。如果未提供，新存储库将与原始存储库具有相同的名称，但位于您的帐户中。

repo_type (`str`, *可选*) ：如果复制数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果复制模型，则设置为 `None` 或 `"model"`。默认为`None`。

private (`bool`, *可选*) : 新的存储库是否应该是私有的。默认与原始存储库具有相同的隐私。不能与`visibility`一起传递。

可见性（`Literal["public", "private", "protected"]`，*可选*）：新存储库的可见性。对于空间，可以是 `"public"` 或 `"private"`，或者 `"protected"`。默认与原始存储库具有相同的可见性。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

存在_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果存储库已存在，则不会引发错误。

space_hardware（`SpaceHardware`或`str`，*可选*）：如果repo_type为“space”，则选择硬件。示例：`"t4-medium"`。完整列表请参见[SpaceHardware](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceHardware)。

space_storage（`SpaceStorage`或`str`，*可选*）：如果repo_type为“space”，则选择持久存储层。示例：`"small"`。完整列表请参见[SpaceStorage](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceStorage)。

space_sleep_time (`int`, *可选*) ：空间进入睡眠状态之前等待的不活动秒数。如果您不希望 Space 休眠（升级硬件的默认行为），请设置为 `-1`。对于免费硬件，您无法配置睡眠时间（值固定为不活动的 48 小时）。仅当 repo_type 为“space”时适用。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-gpus#sleep-time。space_secrets (`list[dict[str, str]]`, *可选*) ：要在空间中设置的密钥列表。每个项目的格式为 `{"key": ..., "value": ..., "description": ...}`，其中描述是可选的。仅当 repo_type 为“space”时适用。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets。

space_variables (`list[dict[str, str]]`, *可选*) ：要在空间中设置的公共环境变量列表。每个项目的格式为 `{"key": ..., "value": ..., "description": ...}`，其中描述是可选的。仅当 repo_type 为“space”时适用。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables。

space_volumes (`list[Volume]`, *可选*) ：复制时要在空间中挂载的 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 对象列表。每个卷都有一个 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、一个 `source`（存储库或存储桶 ID）、一个 `mount_path`（容器内的路径）和可选`revision`、`read_only` 和 `path` 字段。仅当 repo_type 为“space”时适用。

**退货：** [RepoUrl](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.RepoUrl)

新创建的存储库的 URL。值是`str`的子类，包含
`endpoint`、`repo_type` 和 `repo_id` 等属性。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 `HfHubHTTPError`- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到`from_id`或`to_id`之一。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。
- `HfHubHTTPError` -- 
  如果 HuggingFace API 返回错误

在 Hub（模型、数据集或空间）上复制存储库。

这将执行服务器端复制，保留完整的 git 历史记录和 LFS 对象
无需本地下载/上传往返。

示例：
```python
>>> from huggingface_hub import duplicate_repo

# Duplicate a model to your account
>>> duplicate_repo("google/gemma-7b")
RepoUrl('https://huggingface.co/nateraw/gemma-7b',...)

# Duplicate a dataset with a custom name
>>> duplicate_repo("openai/gdpval", to_id="myorg/my-gdpval", repo_type="dataset")
RepoUrl('https://huggingface.co/datasets/myorg/my-gdpval',...)

# Duplicate a Space with custom hardware
>>> duplicate_repo("multimodalart/dreambooth-training", repo_type="space", space_hardware="t4-medium")
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)
```

####重复空间[[huggingface_hub.HfApi.duplicate_space]]

```python
duplicate_space(from_id: str, to_id: str | None = None, private: bool | None = None, visibility: RepoVisibility_T | None = None, token: bool | str | None = None, exist_ok: bool = False, hardware: SpaceHardware | None = None, storage: SpaceStorage | None = None, sleep_time: int | None = None, secrets: list[dict[str, str]] | None = None, variables: list[dict[str, str]] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9024)

**参数：**

from_id (`str`) ：要复制的空间的 ID。示例：`"pharma/CLIP-Interrogator"`。

to_id (`str`, *可选*) : 新空间的 ID。示例：`"dog/CLIP-Interrogator"`。如果未提供，新空间将与原始空间具有相同的名称，但在您的帐户中。

private (`bool`, *可选*) ：新空间是否应该是私有的。默认与原始空间具有相同的隐私。不能与`visibility`一起通过。

可见性（`Literal["public", "private", "protected"]`，*可选*）：新空间的可见性。可以是 `"public"`、`"private"` 或 `"protected"`。默认与原始空间具有相同的可见性。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

存在_ok（`bool`，*可选*，默认为`False`）：如果`True`，如果存储库已存在，则不会引发错误。

硬件（`SpaceHardware` 或 `str`，*可选*）：硬件选择。示例：`"t4-medium"`。完整列表请参见[SpaceHardware](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceHardware)。

存储（`SpaceStorage`或`str`，*可选*）：持久存储层的选择。示例：`"small"`。完整列表请参见[SpaceStorage](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceStorage)。

sleep_time (`int`, *可选*) ：空间进入休眠状态之前等待的不活动秒数。如果您不希望 Space 休眠（升级硬件的默认行为），请设置为 `-1`。对于免费硬件，您无法配置睡眠时间（值固定为不活动的 48 小时）。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-gpus#sleep-time。Secrets (`list[dict[str, str]]`，*可选*)：要在您的空间中设置的密钥列表。每个项目的格式为 `{"key": ..., "value": ..., "description": ...}`，其中描述是可选的。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets。

变量（`list[dict[str, str]]`，*可选*）：要在空间中设置的公共环境变量列表。每个项目的格式为 `{"key": ..., "value": ..., "description": ...}`，其中描述是可选的。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables。

**退货：** [RepoUrl](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.RepoUrl)

新创建的存储库的 URL。值是`str`的子类，包含
`endpoint`、`repo_type` 和 `repo_id` 等属性。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 `HfHubHTTPError`

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到`from_id`或`to_id`之一。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。
- `HfHubHTTPError` -- 
  如果 HuggingFace API 返回错误

复制空间。

以编程方式复制空间。新空间将在您的帐户中创建，并将处于相同的状态
作为原始空间（运行或暂停）。无论空间的当前状态如何，您都可以复制空间。

示例：
```python
>>> from huggingface_hub import duplicate_space

# Duplicate a Space to your account
>>> duplicate_space("multimodalart/dreambooth-training")
RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)

# Can set custom destination id and visibility flag.
>>> duplicate_space("multimodalart/dreambooth-training", to_id="my-dreambooth", visibility="private")
RepoUrl('https://huggingface.co/spaces/nateraw/my-dreambooth',...)
```> [!警告]
> `duplicate_space` 已弃用，并将在 2.0 版本中删除。请使用 [duplicate_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.duplicate_repo) 代替。

#### edit_discussion_comment[[huggingface_hub.HfApi.edit_discussion_comment]]

```python
edit_discussion_comment(repo_id: str, discussion_num: int, comment_id: str, new_content: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7979)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。

comment_id (`str`) ：要编辑的评论的 ID。

new_content (`str`) ：评论的新内容。评论支持Markdown格式。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [DiscussionComment](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionComment)

编辑后的评论

编辑对讨论/拉取请求的评论。> [!提示]
> 引发以下错误：
>
> - [⟦T1272⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T1273⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

####enable_space_dev_mode[[huggingface_hub.HfApi.enable_space_dev_mode]]

```python
enable_space_dev_mode(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8471)

**参数：**

repo_id (`str`) ：启用开发模式的空间 ID。示例：`"Salesforce/BLIP2"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关您的空间的运行时信息。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) 或 [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到您的空间（错误 404）。很可能是错误的 repo_id 或您的空间是私人的，但您
  未经过身份验证。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  403 Forbidden：只有空间的所有者才能设置开发模式。如果你想处理一个你不想处理的空间
  自己的，可以通过打开讨论来询问所有者或复制空间。
- [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) -- 
  如果您的空间是静态空间。静态空间始终运行且从不计费。如果你想隐藏
  静态空间，可以设置为私有。

在空间上启用开发模式。

Spaces 开发模式可简化应用程序的调试，并允许您更快地迭代 Spaces
重新启动应用程序而不停止 Space 容器本身。此功能可作为
PRO 或团队和企业计划。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-dev-mode。

####启用_webhook[[huggingface_hub.HfApi.enable_webhook]]

```python
enable_webhook(webhook_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11327)

**参数：**

webhook_id (`str`) ：要启用的 webhook 的唯一标识符。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [WebhookInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

有关已启用的 Webhook 的信息。

启用 Webhook（使其“活动”）。

示例：
```python
>>> from huggingface_hub import enable_webhook
>>> enabled_webhook = enable_webhook("654bbbc16f2ec14d77f109cc")
>>> enabled_webhook
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    job=None,
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    domains=["repo", "discussion"],
    secret="my-secret",
    disabled=False,
)
```

#### fetch_job_logs[[huggingface_hub.HfApi.fetch_job_logs]]

```python
fetch_job_logs(job_id: str, namespace: str | None = None, follow: bool = False, tail: int | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12198)

**参数：**

job_id (`str`) ：作业的 ID。 

命名空间（`str`，*可选*）：作业运行的命名空间。默认为当前用户的命名空间。 

follow (`bool`, *可选*) ：如果`True`，则实时流式传输日志，直到作业完成（阻塞）。如果`False`（默认），则仅获取当前可用的日志并立即返回（非阻塞）。 

tail (`int`, *可选*) ：从日志返回的最大行数。与 `follow=True` 结合使用时，从最后 N 行开始并继续传输新日志。当 `follow=False` 时，仅返回当前可用日志的最后 N 行。token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

从 Hugging Face 基础设施上的计算作业获取所有日志。

示例：

```python
>>> from huggingface_hub import fetch_job_logs, run_job
>>> job = run_job(image="python:3.12", command=["python", "-c" ,"print('Hello from HF compute!')"])
>>> for log in fetch_job_logs(job_id=job.id):
...     print(log)
Hello from HF compute!

>>> # Non-blocking: fetch only currently available logs
>>> for log in fetch_job_logs(job_id=job.id, follow=False):
...     print(log)

>>> # Stream logs starting from the last 100 lines
>>> for log in fetch_job_logs(job_id=job.id, follow=True, tail=100):
...     print(log)
```

#### fetch_job_metrics[[huggingface_hub.HfApi.fetch_job_metrics]]

```python
fetch_job_metrics(job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12278)

**参数：**

job_id (`str`) ：作业的 ID。 

命名空间（`str`，*可选*）：作业运行的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

从 Hugging Face 基础设施上的计算作业中获取所有实时指标。

示例：

```python
>>> from huggingface_hub import fetch_job_metrics, run_job
>>> job = run_job(image="python:3.12", command=["python", "-c" ,"print('Hello from HF compute!')"], flavor="a10g-small")
>>> for metrics in fetch_job_metrics(job_id=job.id):
...     print(metrics)
{
    "cpu_usage_pct": 0,
    "cpu_millicores": 3500,
    "memory_used_bytes": 1306624,
    "memory_total_bytes": 15032385536,
    "rx_bps": 0,
    "tx_bps": 0,
    "gpus": {
        "882fa930": {
            "utilization": 0,
            "memory_used_bytes": 0,
            "memory_total_bytes": 22836000000
        }
    },
    "replica": "57vr7"
}
```

#### fetch_space_logs[[huggingface_hub.HfApi.fetch_space_logs]]

```python
fetch_space_logs(repo_id: str, build: bool = False, follow: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8712)

**参数：**

repo_id (`str`) ：空间的 ID。示例：`"bigcode/in-the-stack"`。build (`bool`，*可选*，默认为`False`)：如果`True`，则获取容器构建日志（当空间卡在`BUILD_ERROR`时很有用）。如果`False`（默认），则获取运行日志，即正在运行的应用程序的stdout/stderr。

follow (`bool`，*可选*，默认为`False`)：如果`True`，则流实时记录（阻塞），直到服务器关闭流或引发`KeyboardInterrupt`。如果`False`（默认），则仅获取当前缓冲的日志并立即返回（非阻塞，如`docker logs`）。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌。默认为本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** `Iterable[str]`

当日志行可用时生成日志行的生成器。

获取 Hub 上空间的运行或构建日志。

对于调试无法构建或在运行时崩溃的空间很有用，
特别是在脚本或代理工作流程中，在浏览器中读取日志
不是一个选择。

示例：

```python
>>> from huggingface_hub import fetch_space_logs
>>> # Non-blocking: print currently available run logs and exit.
>>> for line in fetch_space_logs("username/my-space"):
...     print(line, end="")

>>> # Debug a build failure:
>>> for line in fetch_space_logs("username/my-space", build=True):
...     print(line, end="")

>>> # Stream run logs until the server closes the stream.
>>> for line in fetch_space_logs("username/my-space", follow=True):
...     print(line, end="")
```

#### file_exists[[huggingface_hub.HfApi.file_exists]]

```python
file_exists(repo_id: str, filename: str, repo_type: str | None = None, revision: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3869)

**参数：**repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

filename (`str`) : 要检查的文件名，例如：`"config.json"`

repo_type (`str`, *可选*) ：如果从数据集或空间获取存储库信息，则设置为 `"dataset"` 或 `"space"`；如果从模型获取存储库信息，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：从中获取信息的存储库的修订版本。默认为 `"main"` 分支。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：**

如果文件存在则为 True，否则为 False。

检查 Hugging Face Hub 上的存储库中是否存在文件。

示例：
```py
>>> from huggingface_hub import file_exists
>>> file_exists("bigcode/starcoder", "config.json")
True
>>> file_exists("bigcode/starcoder", "not-a-file")
False
>>> file_exists("bigcode/not-a-repo", "config.json")
False
```

#### get_bucket_file_metadata[[huggingface_hub.HfApi.get_bucket_file_metadata]]

```python
get_bucket_file_metadata(bucket_id: str, remote_path: str, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L14577)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。

remote_path (`str`) ：文件在存储桶中的路径。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [BucketFileMetadata](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketFileMetadata)

包含大小和 xet 信息的文件元数据。

获取存储桶中文件的元数据。

示例：
```python
>>> from huggingface_hub import get_bucket_file_metadata
>>> metadata = get_bucket_file_metadata(
...     bucket_id="username/my-bucket",
...     remote_path="models/model.safetensors",
... )
>>> metadata.size
42000
```

#### get_bucket_paths_info[[huggingface_hub.HfApi.get_bucket_paths_info]]

```python
get_bucket_paths_info(bucket_id: str, paths: Iterable[str], token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13949)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。

paths (`Iterable[str]`) ：获取信息的路径。如果路径不存在，则会忽略该路径而不引发异常。仅支持文件路径。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[BucketFile]`

有关路径的信息，作为 [BucketFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketFile) 对象的可迭代对象。

获取有关存储桶路径的信息。以 1000 个路径为一组进行批量调用。结果在收到时即产生。

示例：
```py
>>> from huggingface_hub import get_bucket_paths_info
>>> paths_info = get_bucket_paths_info("username/my-bucket", ["file.txt", "checkpoints/model.safetensors"])
>>> for info in paths_info:
...     print(info)
BucketFile(type='file', path='file.txt', size=2379, xet_hash='96e637d9665bd35477b1908a23f2e254edfba0618dbd2d62f90a6baee7d139cf', mtime=datetime.datetime(2024, 9, 25, 15, 31, 2, 346000, tzinfo=datetime.timezone.utc))
BucketFile(type='file', path='checkpoints/model.safetensors', size=2408828, xet_hash='3ed0e9fefe788ddd61d1e26eba67057e9740a064b009256fbafadf6bb95785ca', mtime=datetime.datetime(2024, 9, 25, 15, 31, 2, 346000, tzinfo=datetime.timezone.utc))
```

#### get_collection[[huggingface_hub.HfApi.get_collection]]

```python
get_collection(collection_slug: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10116)

**参数：**

collection_slug (`str`) : Hub 集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

获取有关 Hub 上集合的信息。

退货：[Collection](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.Collection)

示例：

```py
>>> from huggingface_hub import get_collection
>>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")
>>> collection.title
'Recent models'
>>> len(collection.items)
37
>>> collection.items[0]
CollectionItem(
    item_object_id='651446103cd773a050bf64c2',
    item_id='TheBloke/U-Amethyst-20B-AWQ',
    item_type='model',
    position=88,
    note=None
)
```

#### get_dataset_leaderboard[[huggingface_hub.HfApi.get_dataset_leaderboard]]

```python
get_dataset_leaderboard(repo_id: str, base_model_only: bool | None = None, token: bool | str | None = None, timeout: float | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3402)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。例如：`"allenai/olmOCR-bench"`。base_model_only （`bool` 或 `None`，*可选*）：默认情况下，排行榜仅包含未声明 `base_model` 关系的模型（即规范/根存储库），与 Hub 的默认排行榜视图相匹配。声明父模型的微调或衍生存储库被排除在外。通过 `base_model_only=False` 禁用此过滤器并包含每个提交的结果，无论模型是否声明基本模型关系。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。

**退货：** `list[DatasetLeaderboardEntry]`

代表 [DatasetLeaderboardEntry](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DatasetLeaderboardEntry) 对象的列表
排行榜条目，按排名排序。

获取 Hub 上数据集的排行榜。排行榜根据给定基准的评估分数对模型进行排名
数据集。并非所有数据集都有排行榜——只有带有评估的基准数据集
结果提交给他们。这提供了以数据集为中心的分数视图；对于以模型为中心的
查看，使用 [model_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.model_info) 和 `expand=["evalResults"]`。

> [!提示]
> 引发以下错误：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)
> 如果数据集没有排行榜。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> leaderboard = api.get_dataset_leaderboard("allenai/olmOCR-bench")
>>> leaderboard[0].model_id
'datalab-to/chandra-ocr-2'
>>> leaderboard[0].rank
1

# Include fine-tuned / derivative models too
>>> full_leaderboard = api.get_dataset_leaderboard("allenai/olmOCR-bench", base_model_only=False)
```

#### get_dataset_tags[[huggingface_hub.HfApi.get_dataset_tags]]

```python
get_dataset_tags()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2405)

将所有有效的数据集标签列为嵌套命名空间对象。

#### get_discussion_details[[huggingface_hub.HfApi.get_discussion_details]]

```python
get_discussion_details(repo_id: str, discussion_num: int, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7459)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

从中心获取讨论/拉取请求的详细信息。

退货：[DiscussionWithDetails](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionWithDetails)

> [!提示]
> 引发以下错误：
>
> - [⟦T1374⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T1375⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### get_full_repo_name[[huggingface_hub.HfApi.get_full_repo_name]]

```python
get_full_repo_name(model_id: str, organization: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7313)

**参数：**

model_id (`str`) ：模型的名称。

组织（`str`，*可选*）：如果通过，存储库名称将位于组织命名空间而不是用户命名空间中。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `str`

用户命名空间中的存储库名称
({username}/{model_id}) 如果没有传递任何组织，则在
否则为组织名称空间 ({organization}/{model_id})。

返回给定模型 ID 和可选的存储库名称
组织。

#### get_hf_file_metadata[[huggingface_hub.HfApi.get_hf_file_metadata]]

```python
get_hf_file_metadata(url: str, token: bool | str | None = None, timeout: float | None = 10)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6448)

**参数：**

url (`str`) ：文件url，例如由[hf_hub_url()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.hf_hub_url)返回。

令牌（⟦T1384​​⟧ 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

timeout (`float`, *可选*, 默认为10) : 在放弃之前等待服务器发送元数据的秒数。

**退货：**一个 [HfFileMetadata](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.HfFileMetadata) 对象，包含位置、etag、大小和 commit_hash 等元数据。

获取 Hub 上给定 url 版本控制的文件的元数据。

#### get_inference_endpoint[[huggingface_hub.HfApi.get_inference_endpoint]]

```python
get_inference_endpoint(name: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9697)

**参数：**

name (`str`) ：要检索信息的推理端点的名称。

命名空间（`str`，*可选*）：推理端点所在的命名空间。默认为当前用户。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关请求的推理端点的信息。

获取有关推理端点的信息。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> endpoint = api.get_inference_endpoint("my-text-to-image")
>>> endpoint
InferenceEndpoint(name='my-text-to-image', ...)

# Get status
>>> endpoint.status
'running'
>>> endpoint.url
'https://my-text-to-image.region.vendor.endpoints.huggingface.cloud'

# Run inference
>>> endpoint.client.text_to_image(...)
```

#### get_model_tags[[huggingface_hub.HfApi.get_model_tags]]

```python
get_model_tags()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2396)

将所有有效的模型标签列为嵌套命名空间对象

#### get_organization_overview[[huggingface_hub.HfApi.get_organization_overview]]

```python
get_organization_overview(organization: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11646)

**参数：**组织 (`str`) ：要获取概述的组织名称。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Organization`

包含组织概述的 `Organization` 对象。

**加薪：** ``HTTPError``

- [⟦T1400⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError) -- 
  HTTP 404 如果集线器上不存在该组织。

获取 Hub 上组织的概览。

#### get_paths_info[[huggingface_hub.HfApi.get_paths_info]]

```python
get_paths_info(repo_id: str, paths: list[str] | str, expand: bool = False, revision: str | None = None, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4328)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

paths (`Union[list[str], str]`, *可选*) ：获取相关信息的路径。如果路径不存在，则会忽略该路径而不引发异常。Expand (`bool`，*可选*，默认为`False`)：是否获取有关路径的更多信息（例如上次提交和文件的安全扫描结果）。此操作对于服务器来说成本更高，因此每页仅返回 50 个结果（而不是 1000 个）。由于分页是在`huggingface_hub`中实现的，因此除了获取结果所需的时间之外，这对您来说是透明的。

revision (`str`, *可选*) ：从中获取信息的存储库的修订版本。默认为 `"main"` 分支。

repo_type (`str`, *可选*) ：从中获取信息的存储库的类型（`"model"`、`"dataset"`或`"space"`。默认为`"model"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `list[Union[RepoFile, RepoFolder]]`

有关路径的信息，作为 [RepoFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.RepoFile) 和 `RepoFolder` 对象的列表。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type、私有但未经身份验证或存储库
  不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果在存储库中未找到修订版（错误 404）。

获取有关存储库路径的信息。

示例：
```py
>>> from huggingface_hub import get_paths_info
>>> paths_info = get_paths_info("allenai/c4", ["README.md", "en"], repo_type="dataset")
>>> paths_info
[
    RepoFile(path='README.md', size=2379, blob_id='f84cb4c97182890fc1dbdeaf1a6a468fd27b4fff', lfs=None, last_commit=None, security=None),
    RepoFolder(path='en', tree_id='dc943c4c40f53d02b31ced1defa7e5f438d5862e', last_commit=None)
]
```

#### get_repo_discussions[[huggingface_hub.HfApi.get_repo_discussions]]

```python
get_repo_discussions(repo_id: str, author: str | None = None, discussion_type: constants.DiscussionTypeFilter | None = None, discussion_status: constants.DiscussionStatusFilter | None = None, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7351)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

作者（`str`，*可选*）：传递一个值以按讨论作者进行过滤。 `None`表示无过滤器。默认为`None`。

Discussion_type (`str`, *可选*) ：设置为 `"pull_request"` 仅获取拉取请求，`"discussion"` 仅获取讨论。设置为 `"all"` 或 `None` 以获取两者。默认为`None`。

Discussion_status (`str`, *可选*) ：设置为 `"open"` （分别为 `"closed"`）以仅获取打开（分别为关闭）的讨论。设置为 `"all"` 或 `None` 以获取两者。默认为`None`。

repo_type (`str`, *可选*) ：如果从数据集或空间获取，则设置为 `"dataset"` 或 `"space"`；如果从模型获取，则设置为 `None` 或 `"model"`。默认为`None`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterator[Discussion]`

[Discussion](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.Discussion) 对象的迭代器。

获取给定存储库的讨论和拉取请求。

示例：

将存储库的所有讨论收集到列表中：

```python
>>> from huggingface_hub import get_repo_discussions
>>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
```

迭代回购的讨论：

```python
>>> from huggingface_hub import get_repo_discussions
>>> for discussion in get_repo_discussions(repo_id="bert-base-uncased"):
...     print(discussion.num, discussion.title)
```

#### get_safetensors_metadata[[huggingface_hub.HfApi.get_safetensors_metadata]]

```python
get_safetensors_metadata(repo_id: str, repo_type: str | None = None, revision: str | None = None, token: bool | str | None = None, timeout: float | None = 10)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6829)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

repo_type (`str`, *可选*) ：如果文件位于数据集或空间中，则设置为 `"dataset"` 或 `"space"`；如果文件位于模型中，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：从中获取文件的 git 版本。可以是分支名称、标签或提交哈希。默认为 `"main"` 分支的头部。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

timeout (`float`，*可选*，默认为 10) ：放弃之前等待服务器发送数据的秒数，传递给每个获取 safetensors 文件头的请求。设置为 `None` 以禁用超时（不推荐，因为停滞的连接可能会无限期地挂起呼叫）。

**返回：** `SafetensorsRepoMetadata`

与 safetensors 存储库相关的信息。

**加薪：** `NotASafetensorsRepoError` 或 `SafetensorsParsingError`

- `NotASafetensorsRepoError` -- 
  如果仓库不是 safetensors 仓库，即没有
  `model.safetensors` 或`model.safetensors.index.json` 文件。
- `SafetensorsParsingError` -- 
  如果无法正确解析 safetensors 文件头。

解析 Hub 上 safetensors 存储库的元数据。

我们首先检查存储库是否有单个 safetensors 文件或分片的 safetensors 存储库。如果是单人的话
safetensors 文件，我们解析该文件中的元数据。如果它是分片安全张量存储库，我们解析
从索引文件中获取元数据，然后解析每个分片中的元数据。要从单个安全张量文件解析元数据，请使用[parse_safetensors_file_metadata()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.parse_safetensors_file_metadata)。

有关 safetensors 格式的更多详细信息，请查看 https://huggingface.co/docs/safetensors/index#format。

示例：
```py
# Parse repo with single weights file
>>> metadata = get_safetensors_metadata("bigscience/bloomz-560m")
>>> metadata
SafetensorsRepoMetadata(
    metadata=None,
    sharded=False,
    weight_map={'h.0.input_layernorm.bias': 'model.safetensors', ...},
    files_metadata={'model.safetensors': SafetensorsFileMetadata(...)}
)
>>> metadata.files_metadata["model.safetensors"].metadata
{'format': 'pt'}

# Parse repo with sharded model
>>> metadata = get_safetensors_metadata("bigscience/bloom")
Parse safetensors files: 100%|██████████████████████████████████████████| 72/72 [00:12<00:00,  5.78it/s]
>>> metadata
SafetensorsRepoMetadata(metadata={'total_size': 352494542848}, sharded=True, weight_map={...}, files_metadata={...})
>>> len(metadata.files_metadata)
72  # All safetensors files have been fetched

# Parse repo with sharded model
>>> get_safetensors_metadata("runwayml/stable-diffusion-v1-5")
NotASafetensorsRepoError: 'runwayml/stable-diffusion-v1-5' is not a safetensors repo. Couldn't find 'model.safetensors.index.json' or 'model.safetensors' files.
```

#### get_space_runtime[[huggingface_hub.HfApi.get_space_runtime]]

```python
get_space_runtime(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8292)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关空间的运行时信息，包括空间阶段和硬件。

获取有关空间的运行时信息。

#### get_space_secrets[[huggingface_hub.HfApi.get_space_secrets]]

```python
get_space_secrets(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8162)

**参数：**

repo_id (`str`) ：要查询的存储库的 ID。示例：`"bigcode/in-the-stack"`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `dict[str, SpaceSecret]`

由秘密名称键入的 `SpaceSecret` 对象字典。

获取空间中的所有秘密。

秘密值是只写的，无法读回。仅包含密钥、描述和最后更新时间
被退回。

秘密允许为空间设置秘密密钥或令牌，而无需对其进行硬编码。
有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.get_space_secrets("username/my-space")
{'HF_TOKEN': SpaceSecret(key='HF_TOKEN', description='...', updated_at=datetime.datetime(...))}
```

#### get_space_variables[[huggingface_hub.HfApi.get_space_variables]]

```python
get_space_variables(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8199)

**参数：**

repo_id (`str`) ：要查询的存储库的 ID。示例：`"bigcode/in-the-stack"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

获取空间中的所有变量。变量允许将环境变量设置为空间，而无需对其进行硬编码。
有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables

#### get_user_overview[[huggingface_hub.HfApi.get_user_overview]]

```python
get_user_overview(username: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11620)

**参数：**

username (`str`) ：要获取概述的用户的用户名。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `User`

包含用户概览的 [User](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.User) 对象。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  HTTP 404 如果集线器上不存在该用户。

获取 Hub 上用户的概览。

#### get_webhook[[huggingface_hub.HfApi.get_webhook]]

```python
get_webhook(webhook_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11000)

**参数：**

webhook_id (`str`) ：要获取的 webhook 的唯一标识符。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [WebhookInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

有关网络钩子的信息。

通过 ID 获取 Webhook。

示例：
```python
>>> from huggingface_hub import get_webhook
>>> webhook = get_webhook("654bbbc16f2ec14d77f109cc")
>>> print(webhook)
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    job=None,
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    secret="my-secret",
    domains=["repo", "discussion"],
    disabled=False,
)
```

#### grant_access[[huggingface_hub.HfApi.grant_access]]

```python
grant_access(repo_id: str, user: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10945)

**参数：**

repo_id (`str`) ：要授予访问权限的存储库的 ID。

user (`str`) ：授予访问权限的用户的用户名。

repo_type (`str`, *可选*) ：要授予访问权限的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为`model`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `HfHubHTTPError`- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果用户已经有权访问该存储库，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。
- `HfHubHTTPError` -- 
  如果集线器上不存在用户，则返回 HTTP 404。

授予用户对给定门控存储库的访问权限。

授予访问权限不需要用户自己发送访问请求。用户自动
添加到接受列表意味着他们可以下载文件您可以随时撤销授予的访问权限
使用 [cancel_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request) 或 [reject_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request)。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

#### hf_hub_download[[huggingface_hub.HfApi.hf_hub_download]]

```python
hf_hub_download(repo_id: str, filename: str, subfolder: str | None = None, repo_type: str | None = None, revision: str | None = None, cache_dir: str | Path | None = None, local_dir: str | Path | None = None, force_download: bool = False, etag_timeout: float = 10, token: bool | str | None = None, local_files_only: bool = False, tqdm_class: type[base_tqdm] | None = None, dry_run: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6524)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

filename (`str`) ：存储库中文件的名称。

子文件夹（`str`，*可选*）：与存储库内的文件夹相对应的可选值。repo_type (`str`, *可选*) ：如果从数据集或空间下载，则设置为 `"dataset"` 或 `"space"`；如果从模型下载，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：可选的 Git 修订 ID，可以是分支名称、标签或提交哈希。

cache_dir (`str`, `Path`, *可选*) : 存储缓存文件的文件夹路径。

local_dir (`str` 或 `Path`, *可选*) ：如果提供，下载的文件将放置在此目录下。

force_download (`bool`，*可选*，默认为`False`)：即使文件已存在于本地缓存中，是否也应该下载该文件。

etag_timeout (`float`, *可选*, 默认为`10`) : 获取ETag时，等待服务器发送数据多少秒后放弃，传递给`httpx.request`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。local_files_only (`bool`，*可选*，默认为`False`)：如果`True`，则避免下载文件，并返回本地缓存文件的路径（如果存在）。

tqdm_class (`tqdm`, *可选*) ：如果提供，则覆盖进度条的默认行为。传递的参数必须继承自 `tqdm.auto.tqdm` 或至少模仿其行为。默认为自定义 HF 进度条，可以通过设置 `HF_HUB_DISABLE_PROGRESS_BARS` 环境变量来禁用。

dry_run（`bool`，*可选*，默认为`False`）：如果`True`，则执行试运行而不实际下载文件。返回一个 [DryRunFileInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 对象，其中包含有关将下载的内容的信息。

**返回：** `str` 或 [DryRunFileInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo)

- 如果`dry_run=False`：文件的本地路径，或者如果网络关闭，则文件的最新版本缓存在磁盘上。
- 如果`dry_run=True`：包含下载信息的[DryRunFileInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo)对象。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 `~utils.RemoteEntryNotFoundError` 或 [LocalEntryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.LocalEntryNotFoundError) 或 ``EnvironmentError`` or ``OSError`` or ``ValueError``- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到要下载的存储库。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果找不到要下载的修订版本。
- `~utils.RemoteEntryNotFoundError` -- 
  如果找不到要下载的文件。
- [LocalEntryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.LocalEntryNotFoundError) -- 
  如果网络被禁用或不可用并且在缓存中找不到文件。
- [⟦T1557⟧](https://docs.python.org/3/library/exceptions.html#EnvironmentError) -- 
  如果`token=True`但是找不到token。
- [⟦T1559⟧](https://docs.python.org/3/library/exceptions.html#OSError) -- 
  如果无法确定 ETag。
- [⟦T1560⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果某些参数值无效。

如果本地缓存中尚不存在给定文件，则下载该文件。

新的缓存文件布局如下所示：
- 缓存目录包含每个 repo_id 一个子文件夹（按存储库类型命名）
- 在每个 repo 文件夹内：
  - refs 是最新已知修订版的列表 => commit_hash 对
  - blobs 包含实际的文件 blob（由其 git-sha 或 sha256 标识，具体取决于
  无论它们是否是 LFS 文件）
  - 快照每次提交包含一个子文件夹，每个“提交”包含文件的子集
  已在该特定提交中解决。每个文件名都是指向 blob 的符号链接
  在那个特定的提交。

```
[  96]  .
└── [ 160]  models--julien-c--EsperBERTo-small
    ├── [ 160]  blobs
    │   ├── [321M]  403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
    │   ├── [ 398]  7cb18dc9bafbfcf74629a4b760af1b160957a83e
    │   └── [1.4K]  d7edf6bd2a681fb0175f7735299831ee1b22b812
    ├── [  96]  refs
    │   └── [  40]  main
    └── [ 128]  snapshots
        ├── [ 128]  2439f60ef33a0d46d85da5001d52aeda5b00ce9f
        │   ├── [  52]  README.md -> ../../blobs/d7edf6bd2a681fb0175f7735299831ee1b22b812
        │   └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
        └── [ 128]  bbc77c8132af1cc5cf678da3f1ddf2de43606d48
            ├── [  52]  README.md -> ../../blobs/7cb18dc9bafbfcf74629a4b760af1b160957a83e
            └── [  76]  pytorch_model.bin -> ../../blobs/403450e234d65943a7dcf7e05a771ce3c92faa84dd07db4ac20f592037a1e4bd
```如果提供了`local_dir`，则存储库中的文件结构将被复制到此位置。使用此功能时
选项，`cache_dir`将不会被使用，并且将在`local_dir`的根目录下创建一个`.cache/huggingface/`文件夹
存储与下载文件相关的一些元数据。虽然这种机制不如主要机制那么强大
缓存系统，它针对定期拉取存储库的最新版本进行了优化。

#### hide_discussion_comment[[huggingface_hub.HfApi.hide_discussion_comment]]

```python
hide_discussion_comment(repo_id: str, discussion_num: int, comment_id: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8036)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。

comment_id (`str`) ：要编辑的评论的ID。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [DiscussionComment](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionComment)

隐藏的评论

隐藏对讨论/拉取请求的评论。

> [!警告]
> 隐藏评论内容无法再检索。隐藏评论是不可逆转的。

> [!提示]
> 引发以下错误：
>
> - [⟦T1578⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T1579⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

####检查作业[[huggingface_hub.HfApi.inspect_job]]

```python
inspect_job(job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12420)

**参数：**

job_id (`str`) ：作业的 ID。 

命名空间（`str`，*可选*）：作业运行的命名空间。默认为当前用户的命名空间。token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为`False`以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

检查 Hugging Face 基础设施上的计算作业。

示例：

```python
>>> from huggingface_hub import inspect_job, run_job
>>> job = run_job(image="python:3.12", command=["python", "-c" ,"print('Hello from HF compute!')"])
>>> inspect_job(job.id)
JobInfo(
    id='68780d00bbe36d38803f645f',
    created_at=datetime.datetime(2025, 7, 16, 20, 35, 12, 808000, tzinfo=datetime.timezone.utc),
    docker_image='python:3.12',
    space_id=None,
    command=['python', '-c', "print('Hello from HF compute!')"],
    arguments=[],
    environment={},
    secrets={},
    flavor='cpu-basic',
    status=JobStatus(stage='RUNNING', message=None)
)
```

#### Inspect_scheduled_job[[huggingface_hub.HfApi.inspect_scheduled_job]]

```python
inspect_scheduled_job(scheduled_job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13006)

**参数：**

Scheduled_job_id (`str`) ：计划作业的 ID。 

namespace (`str`, *可选*) ：计划作业所在的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

检查 Hugging Face 基础设施上计划的计算作业。

示例：

```python
>>> from huggingface_hub import inspect_job, create_scheduled_job
>>> scheduled_job = create_scheduled_job(image="python:3.12", command=["python", "-c" ,"print('Hello from HF compute!')"], schedule="@hourly")
>>> inspect_scheduled_job(scheduled_job.id)
```

#### kernel_info[[huggingface_hub.HfApi.kernel_info]]

```python
kernel_info(repo_id: str, revision: str | None = None, timeout: float | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3542)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。revision (`str`, *可选*) ：从中获取信息的内核存储库的修订版本。

timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [ModelInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.ModelInfo)

内核存储库信息。

在 Huggingface.co 上获取有关某一特定内核的信息。

#### list_accepted_access_requests[[huggingface_hub.HfApi.list_accepted_access_requests]]

```python
list_accepted_access_requests(repo_id: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10632)

**参数：**

repo_id (`str`) ：要获取访问请求的存储库的 id。

repo_type (`str`, *可选*) ：要获取访问请求的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为`model`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterable[AccessRequest]`

`AccessRequest` 对象的可迭代。每个时间包含一个`username`，`email`，
`status` 和 `timestamp` 属性。如果门控存储库具有自定义表单，则 `fields` 属性将
填充用户的答案。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。

获取给定门控存储库已接受的访问请求。已接受的请求意味着用户已请求访问存储库并且该请求已被接受。用户
可以下载存储库的任何文件。如果审批模式是自动，则该列表默认包含所有
请求。已接受的请求可以随时使用 [cancel_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request) 取消或拒绝，并且
[reject_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request)。取消的请求将返回待处理列表，而拒绝的请求将返回待处理列表。
转到拒绝列表。在这两种情况下，用户都将失去对存储库的访问权限。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

示例：
```py
>>> from huggingface_hub import list_accepted_access_requests

>>> requests = list(list_accepted_access_requests("meta-llama/Llama-2-7b"))
>>> len(requests)
411
>>> requests[0]
[
    AccessRequest(
        username='clem',
        fullname='Clem 🤗',
        email='***',
        timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
        status='accepted',
        fields=None,
    ),
    ...
]
```

#### list_bucket_tree[[huggingface_hub.HfApi.list_bucket_tree]]

```python
list_bucket_tree(bucket_id: str, prefix: str | None = None, recursive: bool | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13896)

**参数：**

bucket_id (`str`) ：存储桶的 ID（例如 `"username/my-bucket"`）。

prefix (`str`, *可选*) ：将结果过滤到路径以此前缀开头的文件。

递归（`bool`，*可选*）：如果`True`，则递归列出文件。如果`False`（默认），则仅列出根目录下的文件和目录。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[Union[BucketFile, BucketFolder]]`

[BucketFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketFile) 和 `BucketFolder` 对象的可迭代对象
包含文件和目录信息（路径等）。

列出存储桶中的文件。

示例：
```python
>>> from huggingface_hub import list_bucket_tree
>>> for file_info in list_bucket_tree(bucket_id="username/my-bucket"):
...     print(file_info.path)

>>> # Filter by prefix
>>> for file_info in list_bucket_tree(bucket_id="username/my-bucket", prefix="models/"):
...     print(file_info.path)
```

#### list_buckets[[huggingface_hub.HfApi.list_buckets]]

```python
list_buckets(namespace: str | None = None, search: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13750)

**参数：**

命名空间（`str`，*可选*）：列出此命名空间（用户或组织）下的存储桶。默认列出用户的存储桶。

search (`str`, *可选*) ：用于过滤存储桶名称的搜索字符串。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[BucketInfo]`

[BucketInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.BucketInfo) 对象的可迭代。

列出 Hub 上某个命名空间下的存储桶。

示例：
```python
>>> from huggingface_hub import list_buckets
>>> for bucket in list_buckets(): # lists buckets in the user's namespace
...     print(bucket)

>>> for bucket in list_buckets(namespace="huggingface"): # lists buckets in the "huggingface" organization
...     print(bucket)

>>> for bucket in list_buckets(search="my-prefix"): # filter buckets by name
...     print(bucket)
```#### list_collections[[huggingface_hub.HfApi.list_collections]]

```python
list_collections(owner: list[str] | str | None = None, item: list[str] | str | None = None, sort: CollectionSort_T | None = None, limit: int | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10060)

**参数：**

所有者（`list[str]` 或 `str`，*可选*）：按所有者的用户名过滤。

item（`list[str]` 或 `str`，*可选*）：过滤包含特定项目的集合。例如：`"models/teknium/OpenHermes-2.5-Mistral-7B"`、`"datasets/squad"` 或 `"papers/2311.12983"`。

排序（`Literal["lastModified", "trending", "upvotes"]`，*可选*）：按上次修改、趋势或投票对集合进行排序。

limit (`int`, *可选*) ：要返回的最大集合数。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[Collection]`

[Collection](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.Collection) 对象的可迭代。

列出 Huggingface Hub 上的集合，并提供一些过滤器。

> [!警告]
> 列出集合时，每个集合的项目列表将被截断为最多 4 个项目。检索所有项目
> 从集合中，您必须使用[get_collection()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_collection)。

#### list_daily_papers[[huggingface_hub.HfApi.list_daily_papers]]

```python
list_daily_papers(date: str | None = None, token: bool | str | None = None, week: str | None = None, month: str | None = None, submitter: str | None = None, sort: DailyPapersSort_T | None = None, p: int | None = None, limit: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11877)

**参数：**日期（`str`，*可选*）：ISO 格式（YYYY-MM-DD）的日期，用于获取日报。默认为最新的。

token (Union[bool, str, None], *可选*) ：有效的用户访问令牌（字符串）。默认为本地保存的令牌。要禁用身份验证，请传递`False`。

week (`str`，*可选*) ：ISO 格式 (YYYY-Www) 中的周，用于获取日报。例如，`2025-W09`。

月份（`str`，*可选*）：获取日报的 ISO 格式 (YYYY-MM) 的月份。例如，`2025-02`。

提交者（`str`，*可选*）：过滤日报的提交者的用户名。

sort (`Literal["publishedAt", "trending"]`, *可选*) : 日报的排序顺序。可以通过 `publishedAt` 或通过 `trending`。默认为 `"publishedAt"`

p（`int`，*可选*）：分页的页码。默认为 0。

limit（`int`，*可选*）：要获取的论文限制。默认为 50。

**返回：** `Iterable[PaperInfo]`

`huggingface_hub.hf_api.PaperInfo` 对象的可迭代。

列出 Hugging Face Hub 上给定日期发表的日报。

示例：

```python
>>> from huggingface_hub import HfApi

>>> api = HfApi()
>>> list(api.list_daily_papers(date="2025-10-29"))
```

#### list_dataset_parquet_files[[huggingface_hub.HfApi.list_dataset_parquet_files]]

```python
list_dataset_parquet_files(repo_id: str, config: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2823)

**参数：**

repo_id (`str`) ：数据集存储库 ID（例如 `"username/dataset-name"`）。config (`str`, *可选*) ：按特定配置/子集名称过滤。提供后，仅返回该配置的镶木地板文件。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `list[DatasetParquetEntry]`

`DatasetParquetEntry` 对象的列表
包含每个 parquet 文件的配置、分割、url 和大小。

列出可用于 Hub 上的数据集的 parquet 文件。

Hub 上托管的所有数据集都会由
[Dataset Viewer](https://huggingface.co/docs/dataset-viewer/parquet)。
此方法返回镶木地板文件列表及其 URL、配置、
分割和大小。

示例：
```python
>>> from huggingface_hub import list_dataset_parquet_files
>>> list_dataset_parquet_files("lhoestq/demo1")
>>> entries[0]
DatasetParquetEntry(config='default', split='train', url='https://huggingface.co/...', size=5038)
```

#### list_datasets[[huggingface_hub.HfApi.list_datasets]]

```python
list_datasets(filter: str | Iterable[str] | None = None, author: str | None = None, benchmark: Literal[True] | Literal['official'] | str | None = None, dataset_name: str | None = None, gated: bool | None = None, language_creators: str | list[str] | None = None, language: str | list[str] | None = None, multilinguality: str | list[str] | None = None, size_categories: str | list[str] | None = None, task_categories: str | list[str] | None = None, task_ids: str | list[str] | None = None, search: str | None = None, sort: DatasetSort_T | None = None, limit: int | None = None, expand: list[ExpandDatasetProperty_T] | None = None, full: bool | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2616)

**参数：**

过滤器（`str`或`Iterable[str]`，*可选*）：用于过滤集线器上的数据集的字符串或字符串列表。

作者（`str`，*可选*）：标识返回数据集作者的字符串。benchmark（`True`、`"official"`、`str`、*可选*）：按基准筛选数据集。可以是 `True` 或 `"official"` 返回官方基准数据集。为了未来的兼容性，也可以是表示基准名称的字符串（目前仅支持“官方”）。

dataset_name (`str`, *可选*) : 一个字符串或字符串列表，可用于通过名称识别集线器上的数据集，例如 `SQAC` 或 `wikineural`

门控（`bool`，*可选*）：一个布尔值，用于过滤集线器上门控或非门控的数据集。默认情况下，返回所有数据集。如果传递`gated=True`，则仅返回门控数据集。如果传递`gated=False`，则仅返回非门控数据集。

language_creators（`str` 或 `List`，*可选*）：可用于标识 Hub 上的数据集以及数据的整理方式的字符串或字符串列表，例如 `crowdsourced` 或 `machine_generated`。

language （`str` 或 `List`，*可选*）：表示用于在 Hub 上过滤数据集的两字符语言的字符串或字符串列表。

多语言性（`str`或`List`，*可选*）：表示包含多种语言的数据集的过滤器的字符串或字符串列表。size_categories（`str`或`List`，*可选*）：可用于根据数据集大小识别集线器上的数据集的字符串或字符串列表，例如`100K<n<1M`或`1M<n<10M`。

标签（`str` 或 `List`，*可选*）：已弃用。在`filter`中传递标签以按标签过滤数据集。

task_categories (`str` 或 `List`, *可选*) ：可用于通过设计的任务识别 Hub 上的数据集的字符串或字符串列表，例如 `audio_classification` 或 `named_entity_recognition`。

task_ids（`str`或`List`，*可选*）：可用于通过特定任务（例如`speech_emotion_recognition`或`paraphrase`）识别集线器上的数据集的字符串或字符串列表。

search (`str`, *可选*) ：将包含在返回的数据集中的字符串。

sort (`DatasetSort_T`, *可选*) ：用于对结果数据集进行排序的键。可能的值为“created_at”、“downloads”、“last_modified”、“likes”和“trending_score”。

limit (`int`, *可选*) ：获取数据集数量的限制。将此选项保留为 `None` 将获取所有数据集。Expand (`list[ExpandDatasetProperty_T]`, *可选*) ：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递`full`，则无法使用此参数。可能的值为 `"author"`、`"cardData"`、`"citation"`、`"createdAt"`、`"disabled"`、`"description"`、`"downloads"`、`"downloadsAllTime"`、 `"gated"`、`"lastModified"`、`"likes"`、`"mainSize"`、`"paperswithcode_id"`、`"private"`、`"siblings"`、`"sha"`、`"tags"`、 `"trendingScore"`、`"usedStorage"`、`"resourceGroup"`。

full (`bool`, *可选*) : 是否获取所有数据集数据，包括`last_modified`、`card_data`和文件。可以包含有用的信息，例如 PapersWithCode ID。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterable[DatasetInfo]`

[huggingface_hub.hf_api.DatasetInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DatasetInfo) 对象的可迭代。

列出 Huggingface Hub 上托管的数据集（给定一些过滤器）。

`filter` 参数的用法示例：

```python
>>> from huggingface_hub import HfApi

>>> api = HfApi()

# List all datasets
>>> api.list_datasets()

# List only the text classification datasets
>>> api.list_datasets(filter="task_categories:text-classification")

# List only the datasets in russian for language modeling
>>> api.list_datasets(
...     filter=("language:ru", "task_ids:language-modeling")
... )

# List FiftyOne datasets (identified by the tag "fiftyone" in dataset card)
>>> api.list_datasets(tags="fiftyone")
```

`search` 参数的用法示例：

```python
>>> from huggingface_hub import HfApi

>>> api = HfApi()

# List all datasets with "text" in their name
>>> api.list_datasets(search="text")

# List all datasets with "text" in their name made by google
>>> api.list_datasets(search="text", author="google")
```

#### list_inference_catalog[[huggingface_hub.HfApi.list_inference_catalog]]

```python
list_inference_catalog(token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9667)

**参数：**令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。

**返回：** 列表`str`

目录中可用的型号 ID 列表。

列出拥抱面部推理目录中可用的模型。

推理目录的目标是提供针对推理优化的模型的精选列表
并对其默认配置进行了测试。请参阅 https://endpoints.huggingface.co/catalog 获取列表
目录中的可用型号。

使用 [create_inference_endpoint_from_catalog()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_inference_endpoint_from_catalog) 从目录中部署模型。

> [!警告]
> `list_inference_catalog` 是实验性的。其 API 将来可能会发生变化。请提供反馈
> 如果您有任何建议或要求。

#### list_inference_endpoints[[huggingface_hub.HfApi.list_inference_endpoints]]

```python
list_inference_endpoints(namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9285)

**参数：**

命名空间（`str`，*可选*）：列出端点的命名空间。默认为当前用户。设置为 `"*"` 以列出所有命名空间（即个人命名空间和用户所属的所有组织）的所有端点。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：**列表[InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

给定命名空间的所有推理端点的列表。

列出给定命名空间的所有推理端点。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.list_inference_endpoints()
[InferenceEndpoint(name='my-endpoint', ...), ...]
```

#### list_jobs[[huggingface_hub.HfApi.list_jobs]]

```python
list_jobs(status: list[JobStage | str] | JobStage | str | None = None, labels: dict[str, str] | None = None, timeout: int | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12344)

**参数：**

status (`JobStage`、`str` 或 `list`，*可选*)：仅返回具有给定状态的作业，例如`"RUNNING"` 或 `[JobStage.RUNNING, JobStage.SCHEDULING]`。请参阅 [JobStage](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobStage) 了解可能的值。 

labels (`dict[str, str]`, *可选*) ：仅返回具有所有给定 `key=value` 标签的作业，例如`{"env": "prod", "team": "ml"}`。 

timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。 

命名空间（`str`，*可选*）：列出作业的命名空间。默认为当前用户的命名空间。token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** `Iterable[JobInfo]`

[JobInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobInfo) 对象的可迭代。

列出 Hugging Face 基础设施上的计算作业。

#### list_jobs_hardware[[huggingface_hub.HfApi.list_jobs_hardware]]

```python
list_jobs_hardware(token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12392)

**退货：** `list[JobHardwareInfo]`

可用硬件配置的列表。

列出 Hugging Face 基础设施上作业的可用硬件选项。

示例：

```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> hardware_list = api.list_jobs_hardware()
>>> hardware_list[0]
JobHardwareInfo(name='cpu-basic', pretty_name='CPU Basic', cpu='2 vCPU', ram='16 GB', ephemeral_storage='20 GB', accelerator=None, unit_cost_micro_usd=167, unit_cost_usd=0.000167, unit_label='minute')
>>> hardware_list[0].name
'cpu-basic'

# Filter GPU options
>>> gpu_hardware = [hw for hw in hardware_list if hw.accelerator is not None]
>>> gpu_hardware[0].accelerator.model
'T4'
```

#### list_lfs_files[[huggingface_hub.HfApi.list_lfs_files]]

```python
list_lfs_files(repo_id: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4485)

**参数：**

repo_id (`str`) ：您要列出 LFS 文件的存储库。

repo_type (`str`, *可选*) ：存储库的类型。如果从数据集或空间列出，则设置为 `"dataset"` 或 `"space"`；如果从模型列出，则设置为 `None` 或 `"model"`。默认为`None`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[LFSFileInfo]`

`LFSFileInfo` 对象的迭代器。

列出 Hub 上存储库中的所有 LFS 文件。

这主要用于计算存储库使用了多少存储空间并最终清理大文件
与[permanently_delete_lfs_files()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.permanently_delete_lfs_files)。请注意，这将是一个永久操作，将影响所有提交
引用此已删除的文件并且无法撤消。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> lfs_files = api.list_lfs_files("username/my-cool-repo")

# Filter files files to delete based on a combination of `filename`, `pushed_at`, `ref` or `size`.
# e.g. select only LFS files in the "checkpoints" folder
>>> lfs_files_to_delete = (lfs_file for lfs_file in lfs_files if lfs_file.filename.startswith("checkpoints/"))

# Permanently delete LFS files
>>> api.permanently_delete_lfs_files("username/my-cool-repo", lfs_files_to_delete)
```

#### list_liked_repos[[huggingface_hub.HfApi.list_liked_repos]]

```python
list_liked_repos(user: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3101)

**参数：**

user (`str`, *可选*) ：您要为其获取点赞的用户的名称。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [UserLikes](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.UserLikes)包含用户名和 3 个存储库 ID 列表的对象（1 个用于
模型，1 个用于数据集，1 个用于空间）。

**加薪：** ``ValueError``

- [⟦T1783⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果未传递 `user` 且未找到令牌（无论是来自参数还是来自机器）。

列出huggingface.co 上用户喜欢的所有公共存储库。

该列表是公开的，因此令牌是可选的。如果不传递`user`，则默认为
登录的用户。

另请参阅[unlike()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.unlike)。

示例：
```python
>>> from huggingface_hub import list_liked_repos

>>> likes = list_liked_repos("julien-c")

>>> likes.user
"julien-c"

>>> likes.models
["osanseviero/streamlit_1.15", "Xhaheen/ChatGPT_HF", ...]
```

#### list_models[[huggingface_hub.HfApi.list_models]]

```python
list_models(filter: str | Iterable[str] | None = None, author: str | None = None, apps: str | list[str] | None = None, gated: bool | None = None, inference: Literal['warm'] | None = None, inference_provider: Literal['all'] | PROVIDER_T | list[PROVIDER_T] | None = None, model_name: str | None = None, trained_dataset: str | list[str] | None = None, search: str | None = None, pipeline_tag: str | None = None, num_parameters: str | None = None, emissions_thresholds: tuple[float, float] | None = None, sort: ModelSort_T | None = None, limit: int | None = None, expand: list[ExpandModelProperty_T] | None = None, full: bool | None = None, cardData: bool = False, fetch_config: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2414)

**参数：**

过滤器（`str`或`Iterable[str]`，*可选*）：用于过滤集线器上模型的字符串或字符串列表。模型可以按库、语言、任务、标签等进行过滤。

作者（`str`，*可选*）：标识返回模型的作者（用户或组织）的字符串。

apps （`str` 或 `List`，*可选*）：用于过滤 Hub 上支持指定应用程序的模型的字符串或字符串列表。示例值包括 `"ollama"` 或 `["ollama", "vllm"]`。门控（`bool`，*可选*）：一个布尔值，用于过滤集线器上门控或非门控的模型。默认情况下，返回所有模型。如果通过了`gated=True`，则仅返回门控模型。如果通过了`gated=False`，则仅返回非门控模型。

推理（`Literal["warm"]`，*可选*）：如果“暖”，则过滤集线器上当前由至少一个提供商提供服务的模型。

inference_provider（`Literal["all"]` 或 `str`，*可选*）：用于过滤集线器上由特定提供商提供服务的模型的字符串。通过 `"all"` 以获得至少一个提供商提供服务的所有模型。

训练数据集（`str`或`List`，*可选*）：Hub上模型的训练数据集的字符串标签或字符串标签列表。

search (`str`, *可选*) ：将包含在返回的模型 ID 中的字符串。

pipeline_tag (`str`, *可选*) ：用于过滤 Hub 上模型的字符串管道标签，例如 `summarization`。

num_parameters (`str`, *可选*) ：按参数计数过滤模型。接受与 Hub UI 和 API 相同的范围语法，例如 `"min:6B,max:128B"`、`"min:6B"` 或 `"max:128B"`。emissions_thresholds（`Tuple`，*可选*）：两个整数或浮点数的元组，表示用于过滤结果模型的最小和最大碳足迹（以克为单位）。

sort (`ModelSort_T`, *可选*) ：用于对结果模型进行排序的键。可能的值为“created_at”、“downloads”、“last_modified”、“likes”和“trending_score”。

limit (`int`, *可选*) ：获取模型数量的限制。将此选项保留为 `None` 将获取所有模型。

Expand (`list[ExpandModelProperty_T]`, *可选*) ：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递了`full`、`cardData`或`fetch_config`，则无法使用此参数。可能的值为 `"author"`、`"cardData"`、`"config"`、`"createdAt"`、`"disabled"`、`"downloads"`、`"downloadsAllTime"`、`"evalResults"`、 `"gated"`、`"gguf"`、`"inference"`、`"inferenceProviderMapping"`、`"lastModified"`、`"library_name"`、`"likes"`、`"mask_token"`、`"model-index"`、 `"pipeline_tag"`、`"private"`、`"safetensors"`、`"sha"`、`"siblings"`、`"spaces"`、`"tags"`、`"transformersInfo"`、`"trendingScore"`、 `"widgetData"`和`"resourceGroup"`。

full (`bool`, *可选*) : 是否获取所有模型数据，包括`last_modified`、`sha`、文件和`tags`。使用过滤器时，默认设置为 `True`。cardData (`bool`, *可选*) ：是否也获取模型的元数据。可以包含有用的信息，例如碳排放、指标和训练数据集。

fetch_config (`bool`, *可选*) ：是否也获取模型配置。由于其尺寸，它不包含在 `full` 中。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

model_name（`str`，*可选*）：（已弃用）。请使用 `search` 代替。

**退货：** `Iterable[ModelInfo]`

[huggingface_hub.hf_api.ModelInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.ModelInfo) 对象的可迭代。

列出 Huggingface Hub 上托管的模型，并给出一些过滤器。

示例：

```python
>>> from huggingface_hub import HfApi

>>> api = HfApi()

# List all models
>>> api.list_models()

# List text classification models
>>> api.list_models(filter="text-classification")

# List models from the KerasHub library
>>> api.list_models(filter="keras-hub")

# List models served by Cohere
>>> api.list_models(inference_provider="cohere")

# List models with "bert" in their name
>>> api.list_models(search="bert")

# List models with "bert" in their name and pushed by google
>>> api.list_models(search="bert", author="google")

# List models with 6B to 128B parameters
>>> api.list_models(num_parameters="min:6B,max:128B", sort="likes")
```

#### list_organization_followers[[huggingface_hub.HfApi.list_organization_followers]]

```python
list_organization_followers(organization: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11673)

**参数：**

组织 (`str`) ：要获取关注者的组织名称。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[User]`

包含该组织追随者的 [User](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.User) 对象列表。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  HTTP 404 如果集线器上不存在该组织。

列出 Hub 上某个组织的关注者。

#### list_organization_members[[huggingface_hub.HfApi.list_organization_members]]

```python
list_organization_members(organization: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11702)

**参数：**

组织 (`str`) ：要获取其成员的组织名称。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterable[User]`

包含组织成员的 [User](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.User) 对象列表。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  HTTP 404 如果集线器上不存在该组织。Hub 上组织的成员列表。

#### list_papers[[huggingface_hub.HfApi.list_papers]]

```python
list_papers(query: str | None = None, limit: int | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11786)

**参数：**

查询（`str`，*可选*）：用于查找论文的搜索查询字符串。如果提供，则返回与查询匹配的论文。

limit（`int`，*可选*）：返回的最大论文数。

token (Union[bool, str, None], *可选*) ：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[PaperInfo]`

`huggingface_hub.hf_api.PaperInfo` 对象的可迭代。

根据搜索查询列出 Hugging Face Hub 上的每日报纸。

示例：

```python
>>> from huggingface_hub import HfApi

>>> api = HfApi()

# List all papers with "attention" in their title
>>> api.list_papers(query="attention")
```

#### list_pending_access_requests[[huggingface_hub.HfApi.list_pending_access_requests]]

```python
list_pending_access_requests(repo_id: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10568)

**参数：**

repo_id (`str`) ：要获取访问请求的存储库的 id。

repo_type (`str`, *可选*) ：要获取访问请求的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为`model`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterable[AccessRequest]`

`AccessRequest` 对象的可迭代。每个时间包含一个`username`，`email`，
`status` 和 `timestamp` 属性。如果门控存储库具有自定义表单，则 `fields` 属性将
填充用户的答案。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或 `admin` 存储库所属组织中的角色，或者如果您传递了 `read` 令牌。

获取给定门控存储库的待处理访问请求。

待处理的请求意味着用户已请求访问存储库，但该请求尚未得到处理。
如果审批模式为自动，则此列表应为空。可以接受或拒绝待处理的请求
使用 [accept_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request) 和 [reject_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.reject_access_request)。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

示例：
```py
>>> from huggingface_hub import list_pending_access_requests, accept_access_request

# List pending requests
>>> requests = list(list_pending_access_requests("meta-llama/Llama-2-7b"))
>>> len(requests)
411
>>> requests[0]
[
    AccessRequest(
        username='clem',
        fullname='Clem 🤗',
        email='***',
        timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
        status='pending',
        fields=None,
    ),
    ...
]

# Accept Clem's request
>>> accept_access_request("meta-llama/Llama-2-7b", "clem")
```#### list_rejected_access_requests[[huggingface_hub.HfApi.list_rejected_access_requests]]

```python
list_rejected_access_requests(repo_id: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10694)

**参数：**

repo_id (`str`) ：要获取访问请求的存储库的 id。

repo_type (`str`, *可选*) ：要获取访问请求的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为`model`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterable[AccessRequest]`

`AccessRequest` 对象的可迭代。每个时间包含一个`username`，`email`，
`status` 和 `timestamp` 属性。如果门控存储库具有自定义表单，则 `fields` 属性将
填充用户的答案。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。

获取对给定门控存储库的拒绝访问请求。拒绝的请求意味着用户已请求访问存储库并且该请求已被明确拒绝
由存储库所有者（您或您组织中的其他用户）。用户无法下载任何文件
回购。被拒绝的请求可以随时使用 [accept_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request) 接受或取消，并且
[cancel_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request)。取消的请求将返回待处理列表，而接受的请求将返回待处理列表。
转到已接受的列表。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

示例：
```py
>>> from huggingface_hub import list_rejected_access_requests

>>> requests = list(list_rejected_access_requests("meta-llama/Llama-2-7b"))
>>> len(requests)
411
>>> requests[0]
[
    AccessRequest(
        username='clem',
        fullname='Clem 🤗',
        email='***',
        timestamp=datetime.datetime(2023, 11, 23, 18, 4, 53, 828000, tzinfo=datetime.timezone.utc),
        status='rejected',
        fields=None,
    ),
    ...
]
```

#### list_repo_commits[[huggingface_hub.HfApi.list_repo_commits]]

```python
list_repo_commits(repo_id: str, repo_type: str | None = None, token: bool | str | None = None, revision: str | None = None, formatted: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4242)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

repo_type (`str`, *可选*) ：如果从数据集或空间列出提交，则设置为 `"dataset"` 或 `"space"`，如果从模型列出，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

formatted (`bool`) ：是否返回 HTML 格式的提交标题和描述。默认为 False。

**返回：**列表[[GitCommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.GitCommitInfo)]

包含有关 Hub 上存储库提交信息的对象列表。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type、私有但未经身份验证或存储库
  不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果在存储库中未找到修订版（错误 404）。

获取 Hub 上存储库的给定修订版的提交列表。

提交按日期排序（最后一次提交在前）。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Commits are sorted by date (last commit first)
>>> initial_commit = api.list_repo_commits("gpt2")[-1]

# Initial commit is always a system commit containing the `.gitattributes` file.
>>> initial_commit
GitCommitInfo(
    commit_id='9b865efde13a30c13e0a33e536cf3e4a5a9d71d8',
    authors=['system'],
    created_at=datetime.datetime(2019, 2, 18, 10, 36, 15, tzinfo=datetime.timezone.utc),
    title='initial commit',
    message='',
    formatted_title=None,
    formatted_message=None
)

# Create an empty branch by deriving from initial commit
>>> api.create_branch("gpt2", "new_empty_branch", revision=initial_commit.commit_id)
```

#### list_repo_files[[huggingface_hub.HfApi.list_repo_files]]

```python
list_repo_files(repo_id: str, revision: str | None = None, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3927)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

revision (`str`, *可选*) ：从中获取信息的存储库的修订版本。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `list[str]`

给定存储库中的文件列表。

获取给定存储库中的文件列表。

#### list_repo_likers[[huggingface_hub.HfApi.list_repo_likers]]

```python
list_repo_likers(repo_id: str, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3218)

**参数：**

repo_id (`str`) ：要检索的存储库。示例：`"user/my-cool-model"`。 

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

**退货：** `Iterable[User]`

[huggingface_hub.hf_api.User](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.User) 对象的可迭代。

列出在拥抱 Face Hub 上喜欢给定存储库的所有用户。

另请参阅[list_liked_repos()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_liked_repos)。#### list_repo_refs[[huggingface_hub.HfApi.list_repo_refs]]

```python
list_repo_refs(repo_id: str, repo_type: str | None = None, include_pull_requests: bool = False, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4170)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

repo_type (`str`, *可选*) ：如果从数据集、空间或内核中列出引用，则设置为 `"dataset"`、`"space"` 或 `"kernel"`；如果从模型中列出，则设置为 `None` 或 `"model"`。默认为`None`。

include_pull_requests (`bool`, *可选*) ：是否在列表中包含来自拉取请求的引用。默认为`False`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [GitRefs](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.GitRefs)

包含有关分支和标签的所有信息的对象
集线器上的仓库。

获取给定存储库的引用列表（标签和分支）。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.list_repo_refs("gpt2")
GitRefs(branches=[GitRefInfo(name='main', ref='refs/heads/main', target_commit='e7da7f221d5bf496a48136c0cd264e630fe9fcc8')], converts=[], tags=[])

>>> api.list_repo_refs("bigcode/the-stack", repo_type='dataset')
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

#### list_repo_tree[[huggingface_hub.HfApi.list_repo_tree]]

```python
list_repo_tree(repo_id: str, path_in_repo: str | None = None, recursive: bool = False, expand: bool = False, revision: str | None = None, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3964)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。path_in_repo (`str`, *可选*) ：存储库中树（文件夹）的相对路径，例如：`"checkpoints/1fec34a/results"`。将默认为存储库的根树（文件夹）。

recursive (`bool`，*可选*，默认为`False`) : 是否递归列出树的文件和文件夹。

Expand (`bool`，*可选*，默认为`False`)：是否获取有关树的文件和文件夹的更多信息（例如上次提交和文件的安全扫描结果）。此操作对于服务器来说成本更高，因此每页仅返回 50 个结果（而不是 1000 个）。由于分页是在`huggingface_hub`中实现的，因此除了获取结果所需的时间之外，这对您来说是透明的。

revision (`str`, *可选*) ：从中获取树的存储库的修订版本。默认为 `"main"` 分支。

repo_type (`str`, *可选*) ：从中获取树的存储库的类型（`"model"`、`"dataset"`、`"space"` 或 `"kernel"`）。默认为`"model"`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[Union[RepoFile, RepoFolder]]`

有关树的文件和文件夹的信息，作为 [RepoFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.RepoFile) 和 `RepoFolder` 对象的可迭代对象。文件和文件夹的顺序是
不保证。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 `~utils.RemoteEntryNotFoundError`

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type、私有但未经身份验证或存储库
  不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果在存储库中未找到修订版（错误 404）。
- `~utils.RemoteEntryNotFoundError` -- 
  如果存储库上不存在树（文件夹）（错误 404）。

列出存储库树的文件和文件夹并获取有关它们的信息。

示例：

获取有关存储库树的信息。

```py
>>> from huggingface_hub import list_repo_tree
>>> repo_tree = list_repo_tree("lysandre/arxiv-nlp")
>>> repo_tree
<generator object HfApi.list_repo_tree at 0x7fa4088e1ac0>
>>> list(repo_tree)
[
    RepoFile(path='.gitattributes', size=391, blob_id='ae8c63daedbd4206d7d40126955d4e6ab1c80f8f', lfs=None, last_commit=None, security=None),
    RepoFile(path='README.md', size=391, blob_id='43bd404b159de6fba7c2f4d3264347668d43af25', lfs=None, last_commit=None, security=None),
    RepoFile(path='config.json', size=554, blob_id='2f9618c3a19b9a61add74f70bfb121335aeef666', lfs=None, last_commit=None, security=None),
    RepoFile(
        path='flax_model.msgpack', size=497764107, blob_id='8095a62ccb4d806da7666fcda07467e2d150218e',
        lfs={'size': 497764107, 'sha256': 'd88b0d6a6ff9c3f8151f9d3228f57092aaea997f09af009eefd7373a77b5abb9', 'pointer_size': 134}, last_commit=None, security=None
    ),
    RepoFile(path='merges.txt', size=456318, blob_id='226b0752cac7789c48f0cb3ec53eda48b7be36cc', lfs=None, last_commit=None, security=None),
    RepoFile(
        path='pytorch_model.bin', size=548123560, blob_id='64eaa9c526867e404b68f2c5d66fd78e27026523',
        lfs={'size': 548123560, 'sha256': '9be78edb5b928eba33aa88f431551348f7466ba9f5ef3daf1d552398722a5436', 'pointer_size': 134}, last_commit=None, security=None
    ),
    RepoFile(path='vocab.json', size=898669, blob_id='b00361fece0387ca34b4b8b8539ed830d644dbeb', lfs=None, last_commit=None, security=None)]
]
```

获取有关存储库树的更多信息（上次提交和文件的安全扫描结果）

```py
>>> from huggingface_hub import list_repo_tree
>>> repo_tree = list_repo_tree("prompthero/openjourney-v4", expand=True)
>>> list(repo_tree)
[
    RepoFolder(
        path='feature_extractor',
        tree_id='aa536c4ea18073388b5b0bc791057a7296a00398',
        last_commit={
            'oid': '47b62b20b20e06b9de610e840282b7e6c3d51190',
            'title': 'Upload diffusers weights (#48)',
            'date': datetime.datetime(2023, 3, 21, 9, 5, 27, tzinfo=datetime.timezone.utc)
        }
    ),
    RepoFolder(
        path='safety_checker',
        tree_id='65aef9d787e5557373fdf714d6c34d4fcdd70440',
        last_commit={
            'oid': '47b62b20b20e06b9de610e840282b7e6c3d51190',
            'title': 'Upload diffusers weights (#48)',
            'date': datetime.datetime(2023, 3, 21, 9, 5, 27, tzinfo=datetime.timezone.utc)
        }
    ),
    RepoFile(
        path='model_index.json',
        size=582,
        blob_id='d3d7c1e8c3e78eeb1640b8e2041ee256e24c9ee1',
        lfs=None,
        last_commit={
            'oid': 'b195ed2d503f3eb29637050a886d77bd81d35f0e',
            'title': 'Fix deprecation warning by changing `CLIPFeatureExtractor` to `CLIPImageProcessor`. (#54)',
            'date': datetime.datetime(2023, 5, 15, 21, 41, 59, tzinfo=datetime.timezone.utc)
        },
        security={
            'safe': True,
            'av_scan': {'virusFound': False, 'virusNames': None},
            'pickle_import_scan': None
        }
    )
    ...
]
```

#### list_scheduled_jobs[[huggingface_hub.HfApi.list_scheduled_jobs]]

```python
list_scheduled_jobs(timeout: int | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12974)

**参数：**timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。 

命名空间（`str`，*可选*）：列出作业的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

列出 Hugging Face 基础设施上计划的计算作业。

#### list_space_templates[[huggingface_hub.HfApi.list_space_templates]]

```python
list_space_templates(token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4607)

**参数：**

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `list[SpaceTemplate]`

可用空间模板的列表。

列出 Hub 上可用的官方空间模板。

返回模板的`repo_id`（或其缩写`name`）可以作为`space_template`传递
到 [HfApi.create_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 从该模板播种一个新空间。示例：
```py
>>> from huggingface_hub import list_space_templates
>>> templates = list_space_templates()
>>> templates[0]
SpaceTemplate(name='Streamlit', repo_id='streamlit/streamlit-template-space', sdk='docker', preferred_private=False)
```

#### list_spaces[[huggingface_hub.HfApi.list_spaces]]

```python
list_spaces(filter: str | Iterable[str] | None = None, author: str | None = None, search: str | None = None, datasets: str | Iterable[str] | None = None, models: str | Iterable[str] | None = None, linked: bool = False, sort: SpaceSort_T | None = None, limit: int | None = None, expand: list[ExpandSpaceProperty_T] | None = None, full: bool | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2887)

**参数：**

过滤器（`str`或`Iterable`，*可选*）：可用于识别集线器上的空间的字符串标签或标签列表。

作者（`str`，*可选*）：标识返回空间的作者的字符串。

search (`str`, *可选*) ：将包含在返回的 Spaces 中的字符串。

datasets （`str` 或 `Iterable`，*可选*）：是否返回使用数据集的空间。特定数据集的名称可以作为字符串传递。

models (`str` 或 `Iterable`, *可选*) ：是否返回使用模型的空间。特定模型的名称可以作为字符串传递。

linked (`bool`, *可选*) ：是否返回使用模型或数据集的空间。

sort (`SpaceSort_T`, *可选*) ：用于对结果空间进行排序的键。可能的值为“created_at”、“last_modified”、“likes”和“trending_score”。

limit (`int`, *可选*) : 获取的空间数量的限制。将此选项保留为 `None` 会获取所有空间。Expand (`list[ExpandSpaceProperty_T]`, *可选*) ：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递了`full`，则无法使用此参数。可能的值为 `"author"`、`"cardData"`、`"datasets"`、`"disabled"`、`"lastModified"`、`"createdAt"`、`"likes"`、`"models"`、 `"private"`、`"region"`、`"runtime"`、`"sdk"`、`"siblings"`、`"sha"`、`"subdomain"`、`"tags"`、`"trendingScore"`、 `"usedStorage"`和`"resourceGroup"`。

full (`bool`, *可选*) : 是否获取所有Spaces数据，包括`last_modified`、`siblings`和`card_data`字段。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Iterable[SpaceInfo]`

[huggingface_hub.hf_api.SpaceInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.SpaceInfo) 对象的可迭代。

列出 Huggingface Hub 上托管的空间，并提供一些过滤器。

#### list_spaces_hardware[[huggingface_hub.HfApi.list_spaces_hardware]]

```python
list_spaces_hardware(token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8313)

**退货：** `list[JobHardwareInfo]`

可用硬件配置的列表。

列出 Spaces 的可用硬件选项。

示例：

```python
>>> from huggingface_hub import list_spaces_hardware
>>> hardware_list = list_spaces_hardware()
>>> hardware_list[0]
JobHardwareInfo(name='cpu-basic', pretty_name='CPU Basic', cpu='2 vCPU', ram='16 GB', ...)
>>> hardware_list[0].name
'cpu-basic'
```

#### list_user_followers[[huggingface_hub.HfApi.list_user_followers]]```python
list_user_followers(username: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11730)

**参数：**

username (`str`) ：要获取关注者的用户的用户名。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[User]`

包含用户关注者的 [User](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.User) 对象列表。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  HTTP 404 如果集线器上不存在该用户。

获取 Hub 上用户的关注者列表。

#### list_user_following[[huggingface_hub.HfApi.list_user_following]]

```python
list_user_following(username: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11758)

**参数：**

username (`str`) : 用户的用户名，以获取后面跟随的用户。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[User]`

[User](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.User) 对象的列表，其中包含用户及其后的用户。

**加薪：** `HfHubHTTPError`- `HfHubHTTPError` -- 
  HTTP 404 如果集线器上不存在该用户。

获取 Hub 上某个用户关注的用户列表。

#### list_user_repos[[huggingface_hub.HfApi.list_user_repos]]

```python
list_user_repos(namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3178)

**参数：**

命名空间（`str`，*可选*）：组织名称。如果未提供，则列出经过身份验证的用户的存储库。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌。默认为本地保存的令牌。

**返回：** `Iterable[RepoStorageInfo]`

`RepoStorageInfo` 对象的可迭代。

列出用户或组织的所有存储库（模型、数据集、空间、存储桶）以及存储信息。

使用经过身份验证的用户的 `/api/settings/repositories` 端点或
`/api/organizations/{namespace}/settings/repositories` 对于组织。

示例：
```python
>>> from huggingface_hub import list_user_repos

>>> repos = list(list_user_repos())
>>> repos[0]
RepoStorageInfo(id='username/my-model', type='model', ...)

>>> # List repos from an organization
>>> repos = list(list_user_repos(namespace="my-org"))
```

#### list_webhooks[[huggingface_hub.HfApi.list_webhooks]]

```python
list_webhooks(token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11053)

**参数：**

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `list[WebhookInfo]`

Webhook 信息对象列表。

列出所有配置的 webhook。

示例：
```python
>>> from huggingface_hub import list_webhooks
>>> webhooks = list_webhooks()
>>> len(webhooks)
2
>>> webhooks[0]
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    url="https://webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    secret="my-secret",
    domains=["repo", "discussion"],
    disabled=False,
)
```#### merge_pull_request[[huggingface_hub.HfApi.merge_pull_request]]

```python
merge_pull_request(repo_id: str, discussion_num: int, token: bool | str | None = None, comment: str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7926)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。

comment (`str`, *可选*) ：随状态更改一起发布的可选评论。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [DiscussionStatusChange](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionStatusChange)

状态改变事件

合并拉取请求。> [!提示]
> 引发以下错误：
>
> - [⟦T2090⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T2091⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### model_info[[huggingface_hub.HfApi.model_info]]

```python
model_info(repo_id: str, revision: str | None = None, timeout: float | None = None, securityStatus: bool | None = None, files_metadata: bool = False, expand: list[ExpandModelProperty_T] | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3257)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由`/`分隔。

revision (`str`, *可选*) ：从中获取信息的模型存储库的修订版本。

timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。

securityStatus (`bool`, *可选*) ：是否也从模型存储库中检索安全状态。安全状态将在`security_repo_status`字段中返回。

files_metadata (`bool`, *可选*) : 是否检索存储库中文件的元数据（大小、LFS 元数据等）。默认为 `False`。Expand (`list[ExpandModelProperty_T]`, *可选*) ：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递了`securityStatus`或`files_metadata`，则不能使用此参数。可能的值为 `"author"`、`"baseModels"`、`"cardData"`、`"childrenModelCount"`、`"config"`、`"createdAt"`、`"disabled"`、`"downloads"`、 `"downloadsAllTime"`、`"evalResults"`、`"gated"`、`"gguf"`、`"inference"`、`"inferenceProviderMapping"`、`"lastModified"`、`"library_name"`、`"likes"`、 `"mask_token"`、`"model-index"`、`"pipeline_tag"`、`"private"`、`"safetensors"`、`"sha"`、`"siblings"`、`"spaces"`、`"tags"`、 `"transformersInfo"`、`"trendingScore"`、`"widgetData"`、`"usedStorage"` 和 `"resourceGroup"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [huggingface_hub.hf_api.ModelInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.ModelInfo)

模型存储库信息。

在 Huggingface.co 上获取某一特定型号的信息

如果您传递可接受的令牌或已登录，模型可以是私有的。> [!提示]
> 引发以下错误：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果找不到要下载的版本。

#### move_bucket[[huggingface_hub.HfApi.move_bucket]]

```python
move_bucket(from_id: str, to_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13839)

**参数：**

from_id (`str`) ：命名空间（用户或组织）和存储桶名称，由`/`分隔。原始存储桶标识符（例如`"username/my-bucket"`）。

to_id (`str`) ：命名空间（用户或组织）和存储桶名称，由`/`分隔。最终存储桶标识符（例如`"username/new-bucket-name"`或`"organization/my-bucket"`）。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `BucketNotFoundError`

- `BucketNotFoundError` -- 
  如果找不到源存储桶。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。将存储桶从“namespace1/repo_name1”移动到“namespace2/repo_name2”

请注意，存在某些限制。有关搬家的更多信息
存储库，请参阅
https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo。

示例：
```python
>>> from huggingface_hub import move_bucket

>>> # Rename a bucket within the same namespace
>>> move_bucket(from_id="username/old-name", to_id="username/new-name")

>>> # Transfer a bucket to an organization
>>> move_bucket(from_id="username/my-bucket", to_id="my-org/my-bucket")
```

#### move_repo[[huggingface_hub.HfApi.move_repo]]

```python
move_repo(from_id: str, to_id: str, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4983)

**参数：**

from_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。原始存储库标识符。

to_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。最终存储库标识符。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

将存储库从namespace1/repo_name1移动到namespace2/repo_name2请注意，存在某些限制。有关搬家的更多信息
存储库，请参阅
https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo。

> [!提示]
> 引发以下错误：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### paper_info[[huggingface_hub.HfApi.paper_info]]

```python
paper_info(id: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11837)

**参数：**

id (`str`，**可选**)：论文的 ArXiv id。

**返回：** `PaperInfo`

一个`PaperInfo`对象。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  HTTP 404 如果 Hub 上不存在该文件。

获取有关 Hub 上的论文的信息。

#### parse_safetensors_file_metadata[[huggingface_hub.HfApi.parse_safetensors_file_metadata]]

```python
parse_safetensors_file_metadata(repo_id: str, filename: str, repo_type: str | None = None, revision: str | None = None, token: bool | str | None = None, timeout: float | None = 10)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6980)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

filename (`str`) ：存储库中文件的名称。

repo_type (`str`, *可选*) ：如果文件位于数据集或空间中，则设置为 `"dataset"` 或 `"space"`；如果位于模型中，则设置为 `None` 或 `"model"`。默认为`None`。revision (`str`, *可选*) ：从中获取文件的 git 版本。可以是分支名称、标签或提交哈希。默认为 `"main"` 分支的头部。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

timeout (`float`, *可选*, 默认为10) : 放弃之前等待服务器发送数据的秒数。设置为 `None` 以禁用超时（不推荐，因为停滞的连接可能会无限期地挂起呼叫）。

**返回：** `SafetensorsFileMetadata`

与安全张量文件相关的信息。

**加薪：** `NotASafetensorsRepoError` 或 `SafetensorsParsingError`

- `NotASafetensorsRepoError` -- 
  如果仓库不是 safetensors 仓库，即没有
  `model.safetensors` 或 `model.safetensors.index.json` 文件。
- `SafetensorsParsingError` -- 
  如果无法正确解析 safetensors 文件头。

从 Hub 上的 safetensors 文件中解析元数据。

要立即解析存储库中所有安全张量文件的元数据，请使用[get_safetensors_metadata()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_safetensors_metadata)。有关 safetensors 格式的更多详细信息，请查看 https://huggingface.co/docs/safetensors/index#format。

####pause_inference_endpoint[[huggingface_hub.HfApi.pause_inference_endpoint]]

```python
pause_inference_endpoint(name: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9930)

**参数：**

name (`str`) ：要暂停的推理端点的名称。

命名空间（`str`，*可选*）：推理端点所在的命名空间。默认为当前用户。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关暂停的推理端点的信息。

暂停推理端点。

暂停的推理端点不会被计费。它可以随时使用[resume_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint)恢复。
这与使用 [scale_to_zero_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.scale_to_zero_inference_endpoint) 将推理端点缩放为零不同，后者
当有请求时会自动重新启动。

为了方便起见，您还可以使用 [pause_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint) 暂停推理端点。

#### 暂停空间[[huggingface_hub.HfApi.pause_space]]

```python
pause_space(repo_id: str, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8432)**参数：**

repo_id (`str`) ：要暂停的空间的 ID。示例：`"Salesforce/BLIP2"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关您的空间的运行时信息，包括 `stage=PAUSED` 和请求的硬件。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) 或 [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到您的空间（错误 404）。很可能是错误的 repo_id 或您的空间是私人的，但您
  未经过身份验证。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  403 Forbidden：只有空间的所有者才能暂停它。如果您想管理您不想管理的空间
  自己的，可以通过打开讨论来询问所有者或复制空间。
- [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) -- 
  如果您的空间是静态空间。静态空间始终运行且从不计费。如果你想隐藏
  静态空间，可以设置为私有。

暂停你的空间。暂停的空间将停止执行，直到由其所有者手动重新启动。这和睡觉不一样
空闲空间在 48 小时不活动后消失的状态。无论暂停时间如何，都不会记入您的帐户
您选择的硬件。要重新启动您的空间，请使用 [restart_space()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.restart_space) 并转到您的空间设置页面。

欲了解更多详情，请访问[the docs](https://huggingface.co/docs/hub/spaces-gpus#pause)。

#### permanent_delete_lfs_files[[huggingface_hub.HfApi.permanently_delete_lfs_files]]

```python
permanently_delete_lfs_files(repo_id: str, lfs_files: Iterable[LFSFileInfo], rewrite_history: bool = True, repo_type: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4539)

**参数：**

repo_id (`str`) ：您要为其列出 LFS 文件的存储库。

lfs_files (`Iterable[LFSFileInfo]`) ：要从存储库中永久删除的 `LFSFileInfo` 项目的迭代。使用 [list_lfs_files()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_lfs_files) 列出存储库中的所有 LFS 文件。

rewrite_history (`bool`，*可选*，默认为`True`)：是否重写存储库历史记录以删除引用已删除的LFS文件的文件指针（推荐）。

repo_type (`str`, *可选*) ：存储库的类型。如果从数据集或空间列出，则设置为 `"dataset"` 或 `"space"`；如果从模型列出，则设置为 `None` 或 `"model"`。默认为`None`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

从 Hub 上的存储库中永久删除 LFS 文件。

> [!警告]
> 这是一项永久性操作，将影响引用已删除文件的所有提交，并可能损坏您的
> 存储库。这是不可恢复的操作。仅当您知道自己在做什么时才使用它。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> lfs_files = api.list_lfs_files("username/my-cool-repo")

# Filter files files to delete based on a combination of `filename`, `pushed_at`, `ref` or `size`.
# e.g. select only LFS files in the "checkpoints" folder
>>> lfs_files_to_delete = (lfs_file for lfs_file in lfs_files if lfs_file.filename.startswith("checkpoints/"))

# Permanently delete LFS files
>>> api.permanently_delete_lfs_files("username/my-cool-repo", lfs_files_to_delete)
```

#### preupload_lfs_files[[huggingface_hub.HfApi.preupload_lfs_files]]

```python
preupload_lfs_files(repo_id: str, additions: Iterable[CommitOperationAdd], token: str | bool | None = None, repo_type: str | None = None, revision: str | None = None, create_pr: bool | None = None, num_threads: int = 5, free_memory: bool = True, gitignore_content: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L5359)

**参数：**

repo_id (`str`) ：您将在其中提交文件的存储库，例如：`"username/custom_transformers"`。 

additions (`Iterable` of [CommitOperationAdd](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd)) : 要上传的文件列表。警告：此列表中的对象将发生变化，以包含与上传相关的信息。不要在多次提交中重复使用相同的对象。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：要上传到的存储库类型（例如 `"model"` -default-、`"dataset"` 或 `"space"`）。 

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。 

create_pr (`boolean`, *可选*) ：您是否计划使用该提交创建拉取请求。默认为`False`。 

num_threads (`int`, *可选*) : 上传文件的并发线程数。默认为5。设置为2表示最多同时上传2个文件。 

free_memory (`bool`, *可选*, 默认为`True`) : 如果是`True`，则每个`CommitOperationAdd`的`path_or_fileobj`属性在上传后会被替换为空的`bytes`对象以节省内存。如果您需要在后续 [create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 调用之外重用操作对象，请设置为 `False`。gitignore_content (`str`, *可选*) ：`.gitignore`文件的内容，以了解哪些文件应该被忽略。优先顺序是首先检查`gitignore_content`是否通过，然后检查`.gitignore`文件是否存在于要提交的文件列表中，最后默认为已托管在Hub上的`.gitignore`文件（如果有）。

将 LFS 文件预先上传到 S3，为将来的提交做好准备。

如果您要生成要即时上传的文件并且不想存储它们，则此方法非常有用
在一次性上传它们之前先将其存储在内存中。

> [!警告]
> 这是高级用户方法。您不需要直接调用它来进行正常提交。
> 直接使用[create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)代替。

> [!警告]
> 提交操作将在此过程中发生变化。特别是，所附的`path_or_fileobj`将是
> 上传后删除以节省内存（并替换为空的`bytes`对象）。请勿重复使用相同的
> 对象，除了将它们传递给 [create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)。如果您不想删除附加内容
> 提交操作对象，传递`free_memory=False`。

示例：
```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

# Generate and preupload LFS files one by one
>>> operations = [] # List of all `CommitOperationAdd` objects that will be generated
>>> for i in range(5):
...     content = ... # generate binary content
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition]) # upload + free memory
...     operations.append(addition)

# Create commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```

#### read_paper[[huggingface_hub.HfApi.read_paper]]

```python
read_paper(id: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11857)

**参数：**

id (`str`) ：论文的 ArXiv id。**退货：** `str`

纸质页面内容为 markdown。

**加薪：** `HfHubHTTPError`

- `HfHubHTTPError` -- 
  HTTP 404 如果 Hub 上不存在该文件。

获取 Hub 上纸质页面的 Markdown 内容。

####拒绝_访问_请求[[huggingface_hub.HfApi.reject_access_request]]

```python
reject_access_request(repo_id: str, user: str, repo_type: str | None = None, rejection_reason: str | None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10864)

**参数：**

repo_id (`str`) : 拒绝访问请求的repo的id。

user (`str`) ：应拒绝访问请求的用户的用户名。

repo_type (`str`, *可选*) ：拒绝访问请求的存储库的类型。必须是 `model`、`dataset` 或 `space` 之一。默认为`model`。

Rejection_reason (`str`, *可选*) ：用户可见的可选拒绝原因（最多 200 个字符）。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `HfHubHTTPError`- `HfHubHTTPError` -- 
  如果存储库没有门控，则为 HTTP 400。
- `HfHubHTTPError` -- 
  如果您对存储库只有只读访问权限，则为 HTTP 403。如果您没有 `write`，可能会出现这种情况
  或存储库所属组织中的`admin`角色，或者如果您传递了`read`令牌。
- `HfHubHTTPError` -- 
  如果集线器上不存在用户，则返回 HTTP 404。
- `HfHubHTTPError` -- 
  如果找不到用户访问请求，则返回 HTTP 404。
- `HfHubHTTPError` -- 
  如果用户访问请求已在拒绝列表中，则返回 HTTP 404。

拒绝用户对给定门控存储库的访问请求。

被拒绝的请求将进入被拒绝列表。用户无法下载存储库的任何文件。被拒绝
使用[accept_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.accept_access_request)和[cancel_access_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_access_request)可以随时接受或取消请求。
取消的请求将返回待处理列表，而已接受的请求将返回已接受列表。

有关门控存储库的更多信息，请参阅 https://huggingface.co/docs/hub/models-ated。

#### rename_discussion[[huggingface_hub.HfApi.rename_discussion]]

```python
rename_discussion(repo_id: str, discussion_num: int, new_title: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L7784)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。Discussion_num (`int`) : 讨论或拉取请求的数量。必须是严格正整数。

new_title (`str`) : 讨论的新标题

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [DiscussionTitleChange](/docs/huggingface_hub/v1.27.0/en/package_reference/community#huggingface_hub.DiscussionTitleChange)

标题更改事件

重命名讨论。

示例：
```python
>>> new_title = "New title, fixing a typo"
>>> HfApi().rename_discussion(
...     repo_id="username/repo_name",
...     discussion_num=34
...     new_title=new_title
... )
# DiscussionTitleChange(id='deadbeef0000000', type='title-change', ...)

```

> [!提示]
> 引发以下错误：
>
> - [⟦T2286⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T2287⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。

#### repo_exists[[huggingface_hub.HfApi.repo_exists]]

```python
repo_exists(repo_id: str, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3778)

**参数：**repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

repo_type (`str`, *可选*) ：如果从数据集或空间获取存储库信息，则设置为 `"dataset"` 或 `"space"`；如果从模型获取存储库信息，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：**

如果存储库存在则为 True，否则为 False。

检查 Hugging Face Hub 上是否存在存储库。

示例：
```py
>>> from huggingface_hub import repo_exists
>>> repo_exists("google/gemma-7b")
True
>>> repo_exists("google/not-a-repo")
False
```

#### repo_info[[huggingface_hub.HfApi.repo_info]]

```python
repo_info(repo_id: str, revision: str | None = None, repo_type: str | None = None, timeout: float | None = None, files_metadata: bool = False, expand: ExpandModelProperty_T | ExpandDatasetProperty_T | ExpandSpaceProperty_T | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3582)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

revision (`str`, *可选*) ：从中获取信息的存储库的修订版本。

repo_type (`str`, *可选*) ：如果从数据集或空间获取存储库信息，则设置为 `"dataset"` 或 `"space"`；如果从模型获取存储库信息，则设置为 `None` 或 `"model"`。默认为`None`。timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。

Expand（`ExpandModelProperty_T`或`ExpandDatasetProperty_T`或`ExpandSpaceProperty_T`，*可选*）：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递`files_metadata`，则无法使用此参数。有关可用属性的详尽列表，请查看 [model_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)、[dataset_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info) 或 [space_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.space_info)。

files_metadata (`bool`, *可选*) ：是否检索存储库中文件的元数据（大小、LFS 元数据等）。默认为`False`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `Union[SpaceInfo, DatasetInfo, ModelInfo]`

存储库信息，作为
[huggingface_hub.hf_api.DatasetInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DatasetInfo)、[huggingface_hub.hf_api.ModelInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.ModelInfo)
或 [huggingface_hub.hf_api.SpaceInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.SpaceInfo) 对象。

获取给定类型的给定存储库的信息对象。> [!提示]
> 引发以下错误：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果找不到要下载的版本。

#### request_space_hardware[[huggingface_hub.HfApi.request_space_hardware]]

```python
request_space_hardware(repo_id: str, hardware: SpaceHardware, token: bool | str | None = None, sleep_time: int | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8336)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

硬件（`str`或[SpaceHardware](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceHardware)）：运行空间的硬件。示例：`"t4-medium"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。sleep_time (`int`, *可选*) ：空间进入休眠状态之前等待的不活动秒数。如果您不希望 Space 休眠（升级硬件的默认行为），请设置为 `-1`。对于免费硬件，您无法配置睡眠时间（值固定为不活动的 48 小时）。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-gpus#sleep-time。

**返回：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关空间的运行时信息，包括空间阶段和硬件。

为空间请求新硬件。

> [!提示]
> 创建 Space 存储库时也可以直接请求硬件！详情请参阅[create_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo)。

#### request_space_storage[[huggingface_hub.HfApi.request_space_storage]]

```python
request_space_storage(repo_id: str, storage: SpaceStorage, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9125)

**参数：**

repo_id (`str`) ：要更新的空间的 ID。示例：`"open-llm-leaderboard/open_llm_leaderboard"`。

存储（`str`或[SpaceStorage](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceStorage)）：存储层。 “小”、“中”或“大”。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)有关空间的运行时信息，包括空间阶段和硬件。

请求空间的持久存储。

> [!警告]
> `request_space_storage` 已弃用，并将在 2.0 版本中删除。请使用 [set_space_volumes()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.set_space_volumes) 代替。

####resolve_revision[[huggingface_hub.HfApi.resolve_revision]]

```python
resolve_revision(repo_id: str, repo_type: str | None = None, revision: str | None = None, cache_dir: str | Path | None = None, local_files_only: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3657)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。

repo_type (`str`, *可选*) ：如果存储库是数据集、空间或内核存储库，则设置为 `"dataset"`、`"space"` 或 `"kernel"`；如果是模型，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：要解决的修订。可以是分支名称、标签、PR 引用或提交哈希。默认为默认分支。如果传递了[ResolvedRevision](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.ResolvedRevision)，则按原样返回。

cache_dir (`str`, `Path`, *可选*) ：存储缓存文件的文件夹的路径。默认为 `HF_HUB_CACHE` 的值。

local_files_only（`bool`，*可选*，默认为`False`）：如果`True`，则仅从本地缓存解析修订版本，而不联系集线器。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [ResolvedRevision](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.ResolvedRevision)

一个 `str` 子类，包含所请求的修订版本及其解析的提交哈希值。

**加薪：** [RevisionResolutionError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionResolutionError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

- [RevisionResolutionError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionResolutionError) -- 
  如果无法解析修订版本：无法访问集线器并且本地不会缓存任何内容。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果集线器上不存在该修订版本。
- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到存储库。这可能是因为它不存在，或者因为它被设置为
  `private` 并且您无权访问。

将修订（分支、标签、PR 引用）解析为提交哈希。这适用于分别下载和加载存储库的多个组件的库（配置、
权重、分词器……）。解决一次修订并传递返回的[ResolvedRevision](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.ResolvedRevision)
保证每个后续调用都针对完全相同的提交，即使存储库在
同时。它还可以节省 HTTP 调用，因为可以从本地提供使用提交哈希进行的下载
缓存而不联系集线器。

`revision` -> `commit hash` 映射缓存在磁盘上（在缓存的 `refs/` 文件夹中），位于
尽力而为的基础。如果稍后无法访问 Hub（离线模式、连接错误、超时、Hub
停机时间，...），缓存的值用作后备。

> [!提示]
> 如果您只需要下载完整的存储库快照，则一次 [snapshot_download()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 调用就足够了
> 已经做了正确的事。 `resolve_revision` 仅在单独下载文件时有用。

示例：
```py
>>> from huggingface_hub import hf_hub_download, resolve_revision
>>> revision = resolve_revision("openai-community/gpt2")
>>> revision
ResolvedRevision(initial=None, resolved='607a30d783dfa663caf39e06633721c8d4cfcd7e')

# Pass it around: every download is pinned to the same commit
>>> config = hf_hub_download("openai-community/gpt2", "config.json", revision=revision)
>>> weights = hf_hub_download("openai-community/gpt2", "model.safetensors", revision=revision)
```

#### restart_space[[huggingface_hub.HfApi.restart_space]]

```python
restart_space(repo_id: str, token: bool | str | None = None, factory_reboot: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8554)

**参数：**

repo_id (`str`) ：要重启的空间的 ID。示例：`"Salesforce/BLIP2"`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

factory_reboot (`bool`, *可选*) : 如果`True`，空间将从头开始重建，而不缓存任何需求。

**退货：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关您的空间的运行时信息。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) 或 [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到您的空间（错误 404）。很可能是错误的 repo_id 或您的空间是私人的，但您
  未经过身份验证。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  403 Forbidden：只有空间的所有者才能重新启动空间。如果您想重新启动您不想重新启动的空间
  自己的，可以通过打开讨论来询问所有者或复制空间。
- [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) -- 
  如果您的空间是静态空间。静态空间始终运行且从不计费。如果你想隐藏
  静态空间，可以设置为私有。

重新启动您的空间。如果您已将空间置于暂停状态，这是以编程方式重新启动空间的唯一方法（请参阅[pause_space()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_space)）。你
必须是空间的所有者才能重新启动它。如果您使用升级的硬件，您的帐户将
空间重新启动后立即计费。无论空间的当前状态如何，您都可以触发重新启动。

欲了解更多详情，请访问[the docs](https://huggingface.co/docs/hub/spaces-gpus#pause)。

####resume_inference_endpoint[[huggingface_hub.HfApi.resume_inference_endpoint]]

```python
resume_inference_endpoint(name: str, namespace: str | None = None, running_ok: bool = True, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9965)

**参数：**

name (`str`) ：要恢复的推理端点的名称。

命名空间（`str`，*可选*）：推理端点所在的命名空间。默认为当前用户。

running_ok (`bool`, *可选*) ：如果`True`，如果推理端点已在运行，则该方法不会引发错误。默认为`True`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关已恢复的推理端点的信息。恢复推理端点。

为了方便起见，您还可以使用 [InferenceEndpoint.resume()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.resume) 恢复推理端点。

####简历_scheduled_job[[huggingface_hub.HfApi.resume_scheduled_job]]

```python
resume_scheduled_job(scheduled_job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13105)

**参数：**

Scheduled_job_id (`str`) ：计划作业的 ID。 

namespace (`str`, *可选*) ：计划作业所在的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

恢复（取消暂停）Hugging Face 基础设施上计划的计算作业。

#### revision_exists[[huggingface_hub.HfApi.revision_exists]]

```python
revision_exists(repo_id: str, revision: str, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3822)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

revision (`str`) ：要检查的存储库的修订版本。repo_type (`str`, *可选*) ：如果从数据集或空间获取存储库信息，则设置为 `"dataset"` 或 `"space"`；如果从模型获取存储库信息，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：**

如果存储库和修订版本存在，则为 True，否则为 False。

检查 Hugging Face Hub 上的存储库中是否存在特定修订版本。

示例：
```py
>>> from huggingface_hub import revision_exists
>>> revision_exists("google/gemma-7b", "float16")
True
>>> revision_exists("google/gemma-7b", "not-a-revision")
False
```

#### run_as_future[[huggingface_hub.HfApi.run_as_future]]

```python
run_as_future(fn: Callable[..., R], *args, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2282)

**参数：**

fn (`Callable`) ：在后台运行的方法。

- ***args,** **kwargs ：调用方法时使用的参数。

**返回：** `Future`

一个 [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) 实例
得到任务的结果。

在后台运行一个方法并返回一个 Future 实例。主要目标是在不阻塞主线程的情况下运行方法（例如在训练期间推送数据）。
后台作业排队以保持顺序，但不并行运行。如果您需要加快脚本速度
通过并行化大量 API 调用，您必须设置并使用您自己的 [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)。

注意：最常用的方法，如 [upload_file()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)、[upload_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 和 [create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 都有一个 `run_as_future: bool`
参数直接在后台调用它们。这相当于对它们调用`api.run_as_future(...)`
但不太冗长。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.run_as_future(api.whoami) # instant
>>> future.done()
False
>>> future.result() # wait until complete and return result
(...)
>>> future.done()
True
```

#### run_job[[huggingface_hub.HfApi.run_job]]

```python
run_job(image: str, command: list[str], env: dict[str, Any] | None = None, secrets: dict[str, Any] | None = None, flavor: JobHardware | str | None = None, timeout: int | float | str | None = None, name: str | None = None, labels: dict[str, str] | None = None, volumes: list[Volume] | None = None, expose: list[int] | None = None, ssh: bool = False, resource_group_id: str | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12021)

**参数：**

image (`str`) ：要使用的 Docker 镜像。示例：`"ubuntu"`、`"python:3.12"`、`"pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"`。来自空间的图像示例：`"hf.co/spaces/lhoestq/duckdb"`。 

命令 (`list[str]`) ：要运行的命令。示例：`["echo", "hello"]`。 

env (`dict[str, Any]`, *可选*) ：定义作业的环境变量。 

Secrets (`dict[str, Any]`, *可选*) ：定义作业的秘密环境变量。 

风味（`str`，*可选*）：硬件风味。请参阅 `JobHardware` 了解可能的值。默认为`"cpu-basic"`。timeout (`Union[int, float, str]`，*可选*)：作业的最大持续时间：int，包含 s（秒，默认）、m（分钟）、h（小时）或 d（天）。示例：`300` 或 `"5m"` 5 分钟。 

name (`str`, *可选*) ：作业的名称。存储为 `name` 标签。不能与 `labels` 中的 `name` 键一起传递。名称不必是唯一的。默认为从图像和命令派生的名称（带有短哈希后缀）。 

labels (`dict[str, str]`，*可选*)：附加到作业的标签（键值对）。 

卷（`list[Volume]`，*可选*）：拥抱 Face Buckets 或 Repos 以作为卷安装在作业容器中。每个卷都是 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 和 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、`source`（例如 `"username/my-bucket"`），以及`mount_path`（例如`"/data"`）。 

hide (`list[int]`, *可选*) ：通过作业代理公开的容器端口。每个列出的端口都可以在公共作业域上访问（例如`https://<job_id>--8000.hf.jobs`）。访问始终需要 HF 令牌，该令牌具有对作业命名空间的读取访问权限。ssh (`bool`, *可选*) ：如果为 True，则可以通过 SSH 通过 `job.status.ssh_url` 指定的 URL 访问作业的容器（例如，从 CLI 中使用 `ssh <job_id>@ssh.hf.jobs` 或 `hf jobs ssh <job_id>`）。连接需要对作业的命名空间的写入权限以及在集线器上注册的 SSH 公钥 (https://huggingface.co/settings/keys)。默认为 False。 

resource_group_id（`str`，*可选*）：要在其中创建作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。如果未提供，则作业将在任何资源组之外创建。 

命名空间（`str`，*可选*）：将在其中创建作业的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

在 Hugging Face 基础设施上运行计算作业。

示例：

运行你的第一个作业：

```python
>>> from huggingface_hub import run_job
>>> run_job(image="python:3.12", command=["python", "-c" ,"print('Hello from HF compute!')"])
```

运行 GPU 作业：

```python
>>> from huggingface_hub import run_job
>>> image = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"
>>> command = ["python", "-c", "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"]
>>> run_job(image=image, command=command, flavor="a10g-small")
```

运行带有卷的作业：

```python
>>> from huggingface_hub import Volume, run_job
>>> dataset_volume = Volume(type="dataset", source="HuggingFaceFW/fineweb", mount_path="/data")
>>> output_bucket_volume = Volume(type="bucket", source="username/my-bucket", mount_path="/output")
>>> image = "duckdb/duckdb"
>>> command = ["duckdb", "-c", "COPY (SELECT * FROM '/data/**/*.parquet' LIMIT 5) TO '/output/first-rows.parquet'"]
>>> run_job(image=image, command=command, volumes=[dataset_volume, output_bucket_volume])
```

#### run_uv_job[[huggingface_hub.HfApi.run_uv_job]]

```python
run_uv_job(script: str, script_args: list[str] | None = None, dependencies: list[str] | None = None, python: str | None = None, image: str | None = None, env: dict[str, Any] | None = None, secrets: dict[str, Any] | None = None, flavor: JobHardware | str | None = None, timeout: int | float | str | None = None, name: str | None = None, labels: dict[str, str] | None = None, volumes: list[Volume] | None = None, expose: list[int] | None = None, ssh: bool = False, resource_group_id: str | None = None, namespace: str | None = None, token: bool | str | None = None)
```[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12658)

**参数：**

script (`str`) ：UV脚本的路径或URL，或者命令。 

script_args (`list[str]`, *可选*) ：传递给脚本或命令的参数。 

依赖项（`list[str]`，*可选*）：用于运行 UV 脚本的依赖项。 

python (`str`, *可选*) ：使用特定的Python版本。默认值为 3.12。 

镜像（`str`，*可选*，默认为“ghcr.io/astral-sh/uv --python3.12-bookworm”）：使用安装了`uv`的自定义 Docker 镜像。 

env (`dict[str, Any]`, *可选*) ：定义作业的环境变量。 

Secrets (`dict[str, Any]`, *可选*) ：定义作业的秘密环境变量。 

风味（`str`，*可选*）：硬件风味。请参阅 `JobHardware` 了解可能的值。默认为`"cpu-basic"`。 

timeout (`Union[int, float, str]`，*可选*)：作业的最大持续时间：int，包含 s（秒，默认）、m（分钟）、h（小时）或 d（天）。示例：`300` 或 `"5m"` 5 分钟。 

name (`str`, *可选*) ：作业的名称。存储为 `name` 标签。不能与 `labels` 中的 `name` 键一起传递。名称不必是唯一的。默认为从脚本及其参数派生的名称（带有短哈希后缀）。labels (`dict[str, str]`，*可选*)：附加到作业的标签（键值对）。 

卷（`list[Volume]`，*可选*）：拥抱 Face Buckets 或 Repos 以作为卷安装在作业容器中。每个卷都是 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 和 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、`source`（例如 `"username/my-bucket"`）和`mount_path`（例如`"/data"`）。 

hide (`list[int]`, *可选*) ：通过作业代理公开的容器端口。每个列出的端口都可以在公共作业域上访问（例如`https://<job_id>--8000.hf.jobs`）。访问始终需要 HF 令牌，该令牌具有对作业命名空间的读取访问权限。 

ssh (`bool`, *可选*) ：如果为 True，则可以通过 SSH 通过 `job.status.ssh_url` 指定的 URL 访问作业的容器（例如，从 CLI 中使用 `ssh <job_id>@ssh.hf.jobs` 或 `hf jobs ssh <job_id>`）。连接需要对作业的命名空间的写入权限以及在集线器上注册的 SSH 公钥 (https://huggingface.co/settings/keys)。默认为 False。 

resource_group_id（`str`，*可选*）：要在其中创建作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。如果未提供，则作业将在任何资源组之外创建。命名空间（`str`，*可选*）：将在其中创建作业的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

在 Hugging Face 基础设施上运行 UV 脚本作业。

示例：

从 URL 运行脚本：

```python
>>> from huggingface_hub import run_uv_job
>>> script = "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py"
>>> script_args = ["--model_name_or_path", "Qwen/Qwen2-0.5B", "--dataset_name", "trl-lib/Capybara", "--push_to_hub"]
>>> run_uv_job(script, script_args=script_args, dependencies=["trl"], flavor="a10g-small")
```

运行本地脚本：

```python
>>> from huggingface_hub import run_uv_job
>>> script = "my_sft.py"
>>> script_args = ["--model_name_or_path", "Qwen/Qwen2-0.5B", "--dataset_name", "trl-lib/Capybara", "--push_to_hub"]
>>> run_uv_job(script, script_args=script_args, dependencies=["trl"], flavor="a10g-small")
```

运行命令：

```python
>>> from huggingface_hub import run_uv_job
>>> script = "lighteval"
>>> script_args= ["endpoint", "inference-providers", "model_name=openai/gpt-oss-20b,provider=auto", "lighteval|gsm8k|0|0"]
>>> run_uv_job(script, script_args=script_args, dependencies=["lighteval"], flavor="a10g-small")
```

安装卷，例如在训练期间保存模型检查点：

```python
>>> from huggingface_hub import Volume, run_uv_job
>>> script = "my_sft.py"
>>> script_args = ["--output_dir", "/training-outputs/training-v3-final", ...]
>>> checkpoints_bucket = Volume(type="bucket", source="username/my-bucket", mount_path="/training-outputs")
>>> run_uv_job(script, script_args=script_args, volumes=[checkpoints_bucket])
```

####scale_to_zero_inference_endpoint[[huggingface_hub.HfApi.scale_to_zero_inference_endpoint]]

```python
scale_to_zero_inference_endpoint(name: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10011)

**参数：**

name (`str`) ：要缩放到零的推理端点的名称。

命名空间（`str`，*可选*）：推理端点所在的命名空间。默认为当前用户。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请通过`False`。

**退货：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关缩放至零推理端点的信息。

将推理端点缩放为零。

缩放为零的推理端点不会被收取费用。它将在下一个请求时恢复，并带有
冷启动延迟。这与使用 [pause_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_inference_endpoint) 暂停推理端点不同，后者
需要使用[resume_inference_endpoint()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.resume_inference_endpoint)手动恢复。

为了方便起见，您还可以使用 [InferenceEndpoint.scale_to_zero()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.scale_to_zero) 将推理端点缩放为零。

#### search_spaces[[huggingface_hub.HfApi.search_spaces]]

```python
search_spaces(query: str, filter: str | Iterable[str] | None = None, sdk: str | list[str] | None = None, include_non_running: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2991)

**参数：**

query (`str`) ：搜索查询字符串。

过滤器（`str`或`Iterable[str]`，*可选*）：要过滤的字符串标签或标签列表。

sdk（`str` 或 `list[str]`，*可选*）：按 SDK 过滤（例如 `"gradio"`、`"docker"`、`"static"`）。

include_non_running (`bool`, *可选*) ：是否在结果中包含非运行空格。默认为 `False`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** `Iterable[SpaceSearchResult]`

[SpaceSearchResult](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.SpaceSearchResult) 对象的可迭代。

使用语义搜索在 Hub 上搜索空间。

该端点使用语义搜索（基于嵌入）进行多词查询
以及单个单词查询的全文搜索。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> results = list(api.search_spaces("generate image"))
>>> results[0].id
'mrfakename/Z-Image-Turbo'
>>> results[0].ai_category
'Image Generation'
```

#### set_space_sleep_time[[huggingface_hub.HfApi.set_space_sleep_time]]

```python
set_space_sleep_time(repo_id: str, sleep_time: int, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8386)

**参数：**

repo_id (`str`) ：要更新的存储库的 ID。示例：`"bigcode/in-the-stack"`。

sleep_time (`int`, *可选*) ：空间进入休眠状态之前等待的不活动秒数。如果您不希望 Space 暂停（升级硬件的默认行为），请设置为 `-1`。对于免费硬件，您无法配置睡眠时间（值固定为不活动的 48 小时）。有关更多详细信息，请参阅 https://huggingface.co/docs/hub/spaces-gpus#sleep-time。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**返回：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

有关空间的运行时信息，包括空间阶段和硬件。

为在升级硬件上运行的空间设置自定义睡眠时间。

您的空间将在 X 秒不活动后进入睡眠状态。当您的空间处于“睡眠”状态时，您无需付费
模式。如果有新访客登陆您的空间，它会“唤醒它”。只有升级硬件才能拥有
可配置的睡眠时间。要了解更多有关睡眠阶段的信息，请参阅
https://huggingface.co/docs/hub/spaces-gpus#sleep-time。

> [!提示]
> 当使用 [request_space_hardware()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.request_space_hardware) 请求硬件时，还可以设置自定义睡眠时间。

#### set_space_volumes[[huggingface_hub.HfApi.set_space_volumes]]

```python
set_space_volumes(repo_id: str, volumes: list[Volume], token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9195)

**参数：**

repo_id (`str`) ：要更新的空间的 ID。示例：`"username/my-space"`。Volume (`list[Volume]`) ：要挂载的 [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume) 对象列表。每个卷都有一个 `type`（`"bucket"`、`"model"`、`"dataset"` 或 `"space"`）、一个 `source`（存储库或存储桶 ID）、一个 `mount_path`（容器内的路径）和可选`revision`、`read_only` 和 `path` 字段。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** `BadRequestError`

- `BadRequestError` -- 
  如果空间是静态空间（静态空间不支持卷）。

设置空间的体积。

设置（或替换）空间中安装的卷列表。每个卷都提供空间的容器访问权限
到 Hub 资源（模型、数据集或存储桶）。

示例：
```python
>>> from huggingface_hub import HfApi, Volume
>>> api = HfApi()
>>> api.set_space_volumes(
...     "username/my-space",
...     volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/data"),
...     ],
... )
```

#### snapshot_download[[huggingface_hub.HfApi.snapshot_download]]

```python
snapshot_download(repo_id: str, repo_type: str | None = None, revision: str | None = None, cache_dir: str | Path | None = None, local_dir: str | Path | None = None, etag_timeout: float = 10, force_download: bool = False, token: bool | str | None = None, local_files_only: bool = False, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, max_workers: int = 8, tqdm_class: type[base_tqdm] | None = None, dry_run: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6706)

**参数：**

repo_id (`str`) ：用户或组织名称以及存储库名称，以 `/` 分隔。repo_type (`str`, *可选*) ：如果从数据集或空间下载，则设置为 `"dataset"` 或 `"space"`；如果从模型下载，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：可选的 Git 修订 ID，可以是分支名称、标签或提交哈希。

cache_dir (`str`, `Path`, *可选*) : 存储缓存文件的文件夹路径。

local_dir (`str` 或 `Path`, *可选*) ：如果提供，下载的文件将放置在此目录下。

etag_timeout (`float`, *可选*, 默认为`10`) : 获取ETag时，等待服务器发送数据多少秒后放弃，传递给`httpx.request`。

force_download (`bool`，*可选*，默认为`False`)：即使文件已存在于本地缓存中，是否也应该下载该文件。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。local_files_only (`bool`，*可选*，默认为`False`)：如果`True`，则避免下载文件，并返回本地缓存文件的路径（如果存在）。

allowed_pa​​tterns（`list[str]`或`str`，*可选*）：如果提供，则仅下载至少匹配一种模式的文件。

ignore_patterns（`list[str]` 或 `str`，*可选*）：如果提供，则不会下载与任何模式匹配的文件。

max_workers (`int`, *可选*) ：下载文件的并发线程数（1个线程 = 1个文件下载）。默认为 8。

tqdm_class (`tqdm`, *可选*) ：如果提供，则覆盖进度条的默认行为。传递的参数必须继承自 `tqdm.auto.tqdm` 或至少模仿其行为。请注意，`tqdm_class` 不会传递给每个单独的下载。默认为自定义 HF 进度条，可以通过设置 `HF_HUB_DISABLE_PROGRESS_BARS` 环境变量来禁用。

dry_run（`bool`，*可选*，默认为`False`）：如果`True`，则执行试运行而不实际下载文件。返回 [DryRunFileInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 对象列表，其中包含有关将下载的内容的信息。

**返回：** `str` 或[DryRunFileInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 列表- 如果`dry_run=False`：存储库快照的文件夹路径。
- 如果`dry_run=True`：包含下载信息的[DryRunFileInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo)对象列表。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 ``EnvironmentError`` or ``OSError`` or ``ValueError``

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到要下载的存储库。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果找不到要下载的修订版本。
- [⟦T2565⟧](https://docs.python.org/3/library/exceptions.html#EnvironmentError) -- 
  如果`token=True`并且找不到令牌。
- [⟦T2567⟧](https://docs.python.org/3/library/exceptions.html#OSError) -- 如果
  无法确定 ETag。
- [⟦T2568⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果某些参数值无效。

下载存储库文件。

下载指定版本的存储库文件的完整快照。当您想要来自的所有文件时，这很有用
一个回购协议，因为你事先不知道需要哪些。所有文件都按顺序嵌套在一个文件夹中
以保持其实际文件名相对于该文件夹。您还可以使用过滤要下载的文件
`allow_patterns` 和 `ignore_patterns`。如果提供了`local_dir`，则存储库中的文件结构将被复制到此位置。使用此功能时
选项，`cache_dir`将不会被使用，并且将在`local_dir`的根目录下创建一个`.cache/huggingface/`文件夹
存储一些与下载文件相关的元数据。虽然这种机制不如主要机制那么健壮
缓存系统，它针对定期拉取存储库的最新版本进行了优化。

另一种方法是克隆存储库，但这需要正确安装 git 和 git-lfs
配置。使用 git 克隆存储库时也不可能过滤要下载的文件。

#### space_info[[huggingface_hub.HfApi.space_info]]

```python
space_info(repo_id: str, revision: str | None = None, timeout: float | None = None, files_metadata: bool = False, expand: list[ExpandSpaceProperty_T] | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3472)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

revision (`str`, *可选*) ：从中获取信息的空间存储库的修订版本。

timeout (`float`, *可选*) : 是否为向 Hub 的请求设置超时。

files_metadata (`bool`, *可选*) : 是否检索存储库中文件的元数据（大小、LFS 元数据等）。默认为 `False`。Expand (`list[ExpandSpaceProperty_T]`, *可选*) ：列出要在响应中返回的属性。使用时，只会返回列表中的属性。如果传递`full`，则无法使用此参数。可能的值为 `"author"`、`"cardData"`、`"createdAt"`、`"datasets"`、`"disabled"`、`"lastModified"`、`"likes"`、`"models"`、 `"private"`、`"region"`、`"runtime"`、`"sdk"`、`"siblings"`、`"sha"`、`"subdomain"`、`"tags"`、`"trendingScore"`、 `"usedStorage"`和`"resourceGroup"`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [SpaceInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.SpaceInfo)

空间存储库信息。

在 Huggingface.co 上获取某一特定空间的信息。

如果您传递可接受的令牌，则空间可以是私有的。

> [!提示]
> 引发以下错误：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果找不到要下载的版本。#### super_squash_history[[huggingface_hub.HfApi.super_squash_history]]

```python
super_squash_history(repo_id: str, branch: str | None = None, commit_message: str | None = None, repo_type: str | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4405)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由 `/` 分隔。

分支（`str`，*可选*）：要压缩的分支。默认为 `"main"` 分支的头部。

commit_message (`str`, *可选*) ：用于压缩提交的提交消息。

repo_type (`str`, *可选*) ：如果从数据集或空间列出提交，则设置为 `"dataset"` 或 `"space"`；如果从模型列出，则设置为 `None` 或 `"model"`。默认为`None`。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) 或 [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type、私有但未经身份验证或存储库
  不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果找不到要挤压的分支。
- [BadRequestError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError) -- 
  如果分支的引用无效。您不能压缩标签上的历史记录。压缩 Hub 上存储库的分支上的提交历史记录。

当您知道自己将进行数百次提交但又不想这样做时，压缩存储库历史记录会很有用
扰乱历史。压缩提交只能从分支的头部执行。

> [!警告]
> 一旦被压扁，提交历史记录将无法检索。这是不可恢复的操作。

> [!警告]
> 一旦分支的历史被压缩，就不可能将其合并回另一个分支，因为
> 他们的历史将会有所不同。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Create repo
>>> repo_id = api.create_repo("test-squash").repo_id

# Make a lot of commits.
>>> api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"content")
>>> api.upload_file(repo_id=repo_id, path_in_repo="lfs.bin", path_or_fileobj=b"content")
>>> api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"another_content")

# Squash history
>>> api.super_squash_history(repo_id=repo_id)
```

#### 挂起_scheduled_job[[huggingface_hub.HfApi.挂起_scheduled_job]]

```python
suspend_scheduled_job(scheduled_job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13075)

**参数：**

Scheduled_job_id (`str`) ：计划作业的 ID。 

namespace (`str`, *可选*) ：计划作业所在的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

暂停（暂停）Hugging Face 基础设施上计划的计算作业。####sync_bucket[[huggingface_hub.HfApi.sync_bucket]]

```python
sync_bucket(source: str | None = None, dest: str | None = None, delete: bool = False, ignore_times: bool = False, ignore_sizes: bool = False, existing: bool = False, ignore_existing: bool = False, include: list[str] | None = None, exclude: list[str] | None = None, filter_from: str | None = None, plan: str | None = None, apply: str | None = None, dry_run: bool = False, verbose: bool = False, quiet: bool = False, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L14766)

**参数：**

source (*str*, *可选*) ：源路径：本地目录或`hf://buckets/namespace/bucket_name(/prefix)`。除非使用`apply`，否则是必需的。

dest (*str*, *可选*) ：目标路径：本地目录或`hf://buckets/namespace/bucket_name(/prefix)`。除非使用`apply`，否则是必需的。

delete (*bool*, *可选*, 默认为 *False*) ：删除源中不存在的目标文件。

ignore_times（*bool*，*可选*，默认为*False*）：仅根据大小跳过文件，忽略修改时间。

ignore_sizes（*bool*，*可选*，默认为*False*）：仅根据修改时间跳过文件，忽略大小。

现有（*bool*，*可选*，默认为*False*）：跳过在接收器上创建新文件（仅更新现有文件）。

ignore_existing（*bool*，*可选*，默认为*False*）：跳过更新接收器上存在的文件（仅创建新文件）。

include (*list[str]*, *可选*) ：包含匹配模式的文件（fnmatch-style）。

排除（*list[str]*，*可选*）：排除匹配模式（fnmatch-style）的文件。

filter_from (*str*, *可选*) ：具有包含/排除规则的过滤器文件的路径。plan (*str*, *可选*) ：将同步计划保存到此 JSONL 文件而不是执行。

apply (*str*, *可选*) ：应用以前保存的计划文件。设置后，不需要`source`和`dest`。

dry_run (*bool*, *可选*, 默认为 *False*) ：将同步计划作为 JSONL 打印到标准输出而不执行。

verbose（*bool*，*可选*，默认为*False*）：显示详细的每个文件操作。

Quiet（*bool*，*可选*，默认为*False*）：禁止所有输出和进度条。

token (Union[bool, str, None], 可选) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌。

**返回：** [*SyncPlan*]

计算的（或加载的）同步计划。

在本地目录和存储桶之间同步文件。

这相当于 `hf buckets sync` CLI 命令。 `source` 或 `dest` 之一必须是存储桶路径
(`hf://buckets/...`)另一个必须是本地目录路径。

示例：
```python
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Upload local directory to bucket
>>> api.sync_bucket("./data", "hf://buckets/username/my-bucket")

# Download bucket to local directory
>>> api.sync_bucket("hf://buckets/username/my-bucket", "./data")

# Sync with delete and filtering
>>> api.sync_bucket(
...     "./data",
...     "hf://buckets/username/my-bucket",
...     delete=True,
...     include=["*.safetensors"],
... )

# Dry run: preview what would be synced
>>> plan = api.sync_bucket("./data", "hf://buckets/username/my-bucket", dry_run=True)
>>> plan.summary()
&amp;lcub;'uploads': 3, 'downloads': 0, 'deletes': 0, 'skips': 1, 'total_size': 4096}

# Save plan for review, then apply
>>> api.sync_bucket("./data", "hf://buckets/username/my-bucket", plan="sync-plan.jsonl")
>>> api.sync_bucket(apply="sync-plan.jsonl")
```

####sync_job_volume[[huggingface_hub.HfApi.sync_job_volume]]

```python
sync_job_volume(source: str | Path, mount_path: str, remote_name: str | None = None, read_only: bool = True, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13509)

**参数：**

source (`str` 或 `Path`) ：要同步的本地目录的路径。

mount_path (`str`) ：作业容器内的挂载路径，例如`"/inputs"`。必须以 `/` 开头。remote_name (`str`, *可选*) ：要同步到的存储桶子文件夹的名称。默认为从源路径和计算机主机名派生的 `{dirname}-{hash}` 名称。

read_only (`bool`，*可选*，默认为`True`)：在作业中以只读方式挂载卷。传递 `False` 让作业写回存储桶文件夹（例如，随后使用 [sync_bucket()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_bucket) 检索输出）。

命名空间（`str`，*可选*）：拥有`jobs-artifacts`存储桶的命名空间。默认为当前用户的命名空间。使用与将挂载卷的作业相同的命名空间。

令牌（`Union[bool, str, None]`，*可选*）：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** [Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume)

范围为同步子文件夹的存储桶卷，以传入 `volumes` 列表
[run_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_job)、[run_uv_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_uv_job)、[create_scheduled_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_scheduled_job) 或 [create_scheduled_uv_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_scheduled_uv_job)。

将本地目录同步到存储桶并返回准备挂载到作业中的[Volume](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.Volume)。文件上传到`{namespace}/jobs-artifacts`存储桶的子文件夹（自动创建为
私人；如果它已经存在并且是公开的，则使用相同的同步逻辑发出警告
as [sync_bucket()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_bucket)：仅重新同步同一目录
上传新的或修改的文件。默认情况下，子文件夹名称源自目录
路径和机器的主机名，因此来自同一目录的重复调用会重用相同的内容
远程文件夹。通过 `remote_name` 来使用固定名称。

请注意，数据被*复制*到存储桶，而不是实时挂载：之后在本地进行更改
同步对作业不可见（重新运行 `sync_job_volume` 进行更新），并且卷是
默认情况下以只读方式安装。要检索作业写入读写卷的数据，请同步
存储桶文件夹后面带有[sync_bucket()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_bucket)。如果源目录为空（例如输出
目录），上传占位符 `.keep` 文件，以便仍然可以挂载该卷。

示例：
```python
>>> from huggingface_hub import run_uv_job, sync_job_volume

# Upload ./training-data once, then run multiple jobs against it
>>> volume = sync_job_volume("./training-data", "/data")
>>> run_uv_job("train.py", script_args=["--learning-rate", "0.01"], volumes=[volume])
>>> run_uv_job("train.py", script_args=["--learning-rate", "0.05"], volumes=[volume])

# Read-write volume to retrieve outputs after the job completes
>>> volume = sync_job_volume("./outputs", "/outputs", read_only=False)
>>> job = run_uv_job("process.py", volumes=[volume])
```

####trigger_scheduled_job[[huggingface_hub.HfApi.trigger_scheduled_job]]

```python
trigger_scheduled_job(scheduled_job_id: str, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13135)

**参数：**

Scheduled_job_id (`str`) ：计划作业的 ID。 

namespace (`str`, *可选*) ：计划作业所在的命名空间。默认为当前用户的命名空间。token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** [JobInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobInfo)

有关触发运行的信息。

**加薪：** [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)

- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  如果另一个实例已在运行并且在计划作业上禁用了 `concurrency`，则为 HTTP 409。

触发计划的作业立即运行。

这会触发立即运行预定作业的规范。它**不**修改时间表
并且**不会**影响下一次计划的运行。如果实例已经在运行并且已计划
作业不允许并发运行，请求被拒绝 (HTTP 409)。

#### 不像[[huggingface_hub.HfApi.unlike]]

```python
unlike(repo_id: str, token: bool | str | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L3050)

**参数：**

repo_id (`str`) : 不同的存储库。示例：`"user/my-cool-model"`。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。 

repo_type (`str`, *可选*) ：如果不喜欢数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果不喜欢模型，则设置为 `None` 或 `"model"`。默认为`None`。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type，私有
  但未经过身份验证或存储库不存在。

与 Hub 上的给定存储库不同（例如从收藏夹列表中删除）。

为了防止垃圾邮件的使用，不可能从脚本中`like`存储库。

另请参阅[list_liked_repos()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_liked_repos)。

示例：
```python
>>> from huggingface_hub import list_liked_repos, unlike
>>> "gpt2" in list_liked_repos().models # we assume you have already liked gpt2
True
>>> unlike("gpt2")
>>> "gpt2" in list_liked_repos().models
False
```

#### update_collection_item[[huggingface_hub.HfApi.update_collection_item]]

```python
update_collection_item(collection_slug: str, item_object_id: str, note: str | None = None, position: int | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10459)

**参数：**

collection_slug (`str`) ：要更新的集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

item_object_id (`str`) ：集合中项目的 ID。这不是 Hub 上项目的 ID（repo_id 或论文 ID）。它必须从 [CollectionItem](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.CollectionItem) 对象中检索。示例：`collection.items[0].item_object_id`。note (`str`, *可选*) ：附加到集合中的项目的注释。注释的最大长度为 500 个字符。

位置（`int`，*可选*）：集合中项目的新位置。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

更新集合中的项目。

示例：

```py
>>> from huggingface_hub import get_collection, update_collection_item

# Get collection first
>>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")

# Update item based on its ID (add note + update position)
>>> update_collection_item(
...     collection_slug="TheBloke/recent-models-64f9a55bb3115b4f513ec026",
...     item_object_id=collection.items[-1].item_object_id,
...     note="Newly updated model!"
...     position=0,
... )
```

#### update_collection_metadata[[huggingface_hub.HfApi.update_collection_metadata]]

```python
update_collection_metadata(collection_slug: str, title: str | None = None, description: str | None = None, position: int | None = None, private: bool | None = None, theme: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10229)

**参数：**

collection_slug (`str`) ：要更新的集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

title (`str`) ：要更新的集合的标题。

描述（`str`，*可选*）：要更新的集合的描述。描述的最大长度为 150 个字符。

位置（`int`，*可选*）：集合在用户集合列表中的新位置。

private (`bool`, *可选*) : 集合是否应该是私有的。

主题（`str`，*可选*）：Hub 上集合的主题。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

更新 Hub 上集合的元数据。

所有参数都是可选的。仅更新提供的元数据。

返回：[Collection](/docs/huggingface_hub/v1.27.0/en/package_reference/collections#huggingface_hub.Collection)

示例：

```py
>>> from huggingface_hub import update_collection_metadata
>>> collection = update_collection_metadata(
...     collection_slug="username/iccv-2023-64f9a55bb3115b4f513ec026",
...     title="ICCV Oct. 2023"
...     description="Portfolio of models, datasets, papers and demos I presented at ICCV Oct. 2023",
...     private=False,
...     theme="pink",
... )
>>> collection.slug
"username/iccv-oct-2023-64f9a55bb3115b4f513ec026"
# ^collection slug got updated but not the trailing ID
```

#### update_collection_resource_group[[huggingface_hub.HfApi.update_collection_resource_group]]

```python
update_collection_resource_group(collection_slug: str, resource_group_id: str | None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L10297)

**参数：**

collection_slug (`str`) ：要更新的集合的 Slug。示例：`"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`。

resource_group_id（`str` 或 `None`）：要将集合分配到的资源组，作为 24 个字符的十六进制字符串。如果 `None`，则将从任何资源组中删除该集合。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。将集合分配给资源组，或将其从任何资源组中删除。

仅对组织拥有的馆藏有效。

示例：

```py
>>> from huggingface_hub import update_collection_resource_group
>>> update_collection_resource_group(
...     collection_slug="my-org/iccv-2023-64f9a55bb3115b4f513ec026",
...     resource_group_id="66980ecfc1e12a49c8f0e42d",
... )
```

#### update_inference_endpoint[[huggingface_hub.HfApi.update_inference_endpoint]]

```python
update_inference_endpoint(name: str, accelerator: str | None = None, instance_size: str | None = None, instance_type: str | None = None, min_replica: int | None = None, max_replica: int | None = None, scale_to_zero_timeout: int | None = None, scaling_metric: InferenceEndpointScalingMetric | None = None, scaling_threshold: float | None = None, repository: str | None = None, framework: str | None = None, revision: str | None = None, task: str | None = None, custom_image: dict | None = None, container_command: list[str] | None = None, container_args: list[str] | None = None, env: dict[str, str] | None = None, secrets: dict[str, str] | None = None, domain: str | None = None, path: str | None = None, cache_http_responses: bool | None = None, tags: list[str] | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L9744)

**参数：**

name (`str`) ：要更新的推理端点的名称。 

加速器（`str`，*可选*）：用于推理的硬件加速器（例如`"cpu"`）。

instance_size (`str`, *可选*) ：用于托管模型的实例的大小或类型（例如`"x4"`）。

instance_type（`str`，*可选*）：将部署推理端点的云实例类型（例如`"intel-icl"`）。

min_replica (`int`, *可选*) ：推理端点保持运行的最小副本（实例）数量。

max_replica（`int`，*可选*）：推理端点可扩展的最大副本（实例）数量。

scale_to_zero_timeout（`int`，*可选*）：非活动端点缩放为零之前的持续时间（以分钟为单位）。scaling_metric（`str`或`InferenceEndpointScalingMetric `，*可选*）：缩放的度量参考。提供“pendingRequests”或“hardwareUsage”时。默认为无。

scaling_threshold (`float`, *可选*) ：用于触发扩展的扩展指标阈值。未提供缩放指标时将被忽略。默认为无。

存储库（`str`，*可选*）：与推理端点关联的模型存储库的名称（例如`"gpt2"`）。

框架（`str`，*可选*）：用于模型的机器学习框架（例如`"custom"`）。

修订版（`str`，*可选*）：要在推理端点上部署的特定模型修订版（例如`"6c0e6080953db56375760c0471a8c5f2929baf11"`）。

任务（`str`，*可选*）：部署模型的任务（例如`"text-classification"`）。

custom_image（`dict`，*可选*）：用于推理端点的自定义 Docker 映像。如果您想部署在 `text-generation-inference` (TGI) 框架上运行的推理端点（请参阅示例），这非常有用。

container_command (`list[str]`, *可选*) ：覆盖容器入口点命令（映射到 API 负载中的 `model.command`）。适用于托管引擎映像（例如 vLLM、SGLang）和自定义映像。container_args (`list[str]`, *可选*) ：附加到容器入口点的参数（映射到 API 负载中的 `model.args`）。适用于托管引擎映像（例如 vLLM、SGLang）和自定义映像。

env (`dict[str, str]`, *可选*) ：要注入到容器环境中的非秘密环境变量

Secrets (`dict[str, str]`, *可选*) ：要注入到容器环境中的秘密值。 

域（`str`，*可选*）：推理端点部署的自定义域，如果设置，推理端点将在此域中可用（例如`"my-new-domain.cool-website.woof"`）。

路径（`str`，*可选*）：已部署模型的自定义路径，应以`/`开头（例如`"/models/google-bert/bert-base-uncased"`）。 

cache_http_responses (`bool`, *可选*) ：是否缓存来自推理端点的 HTTP 响应。

标签（`list[str]`，*可选*）：与推理端点关联的标签列表。 

命名空间（`str`，*可选*）：将更新推理端点的命名空间。默认为当前用户的命名空间。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [InferenceEndpoint](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint)

有关更新的推理端点的信息。

更新推理端点。

此方法允许更新计算配置、部署的模型、路由或任何组合。
所有参数都是可选的，但至少必须提供一个。

为了方便起见，您还可以使用 [InferenceEndpoint.update()](/docs/huggingface_hub/v1.27.0/en/package_reference/inference_endpoints#huggingface_hub.InferenceEndpoint.update) 更新推理端点。

#### update_job_labels[[huggingface_hub.HfApi.update_job_labels]]

```python
update_job_labels(job_id: str, labels: dict[str, str], namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12615)

**参数：**

job_id (`str`) ：作业的 ID。 

labels (`dict[str, str]`)：要在作业上设置的新标签。替换所有现有标签。键和值的长度不得超过 100 个字符，并且仅包含字母数字字符、点、破折号和下划线。 

命名空间（`str`，*可选*）：作业运行的命名空间。默认为当前用户的命名空间。token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**退货：** [JobInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobInfo)

更新后的职位信息。

更新现有作业的标签。

将所有现有的用户提供的标签替换为新标签。

#### update_repo_settings[[huggingface_hub.HfApi.update_repo_settings]]

```python
update_repo_settings(repo_id: str, gated: Literal['auto', 'manual', False] | None = None, private: bool | None = None, visibility: RepoVisibility_T | None = None, token: str | bool | None = None, repo_type: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4903)

**参数：**

repo_id (`str`) ：名称空间（用户或组织）和存储库名称，以 / 分隔。

门控（`Literal["auto", "manual", False]`，*可选*）：存储库的门控状态。如果设置为 `None`（默认），则存储库的 `gated` 设置将不会更新。 *“自动”：存储库是门控的，访问请求根据预定义的标准自动批准或拒绝。 *“手动”：存储库是封闭的，访问请求需要手动批准。 * False ：存储库没有门禁，任何人都可以访问它。

private (`bool`, *可选*) : 存储库是否应该是私有的。不能与`visibility`一起传递。可见性（`Literal["public", "private", "protected"]`，*可选*）：存储库的可见性。对于空间，可以是 `"public"` 或 `"private"`，或者 `"protected"`。

token (`Union[str, bool, None]`, *可选*) ：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递 False。

repo_type (`str`, *可选*) ：要从中更新设置的存储库类型（`"model"`、`"dataset"` 或 `"space"`）。默认为`"model"`。

**加薪：** ``ValueError`` 或 [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) 或 [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)

- [⟦T2759⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果 gate 不是“自动”、“手动”或 False 之一。
- [⟦T2760⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果 repo_type 不是 Constants.REPO_TYPES 中的值之一。
- [HfHubHTTPError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) -- 
  如果对 Hugging Face Hub API 的请求失败。
- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果找不到要下载的存储库。这可能是因为它不存在，
  或者因为它设置为 `private` 并且您无权访问。

更新存储库的设置，包括门控访问和可见性。

为了更好地控制存储库的使用方式，中心允许存储库作者启用
访问对其存储库的请求，以及更改存储库的可见性。#### update_scheduled_job_labels[[huggingface_hub.HfApi.update_scheduled_job_labels]]

```python
update_scheduled_job_labels(scheduled_job_id: str, labels: dict[str, str], namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L13177)

**参数：**

Scheduled_job_id (`str`) ：计划作业的 ID。 

labels (`dict[str, str]`) ：在计划作业上设置的新标签。替换所有现有标签。键和值的长度不得超过 100 个字符，并且仅包含字母数字字符、点、破折号和下划线。 

namespace (`str`, *可选*) ：计划作业所在的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** `ScheduledJobInfo`

更新的预定作业信息。

更新现有计划作业的标签。

将所有现有的用户提供的标签替换为新标签。

#### update_webhook[[huggingface_hub.HfApi.update_webhook]]

```python
update_webhook(webhook_id: str, url: str | None = None, watched: list[dict | WebhookWatchedItem] | None = None, domains: list[constants.WEBHOOK_DOMAIN_T] | None = None, secret: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L11238)

**参数：**

webhook_id (`str`) ：要更新的 webhook 的唯一标识符。

url (`str`，可选)：有效负载将发送到的 URL。观看（`list[WebhookWatchedItem]`，可选）：要观看的项目列表。它可以是用户、组织、模型、数据集或空间。更多详情请参阅[WebhookWatchedItem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookWatchedItem)。观看的项目也可以作为普通字典提供。

域（`list[Literal["repo", "discussion"]]`，可选）：要监视的域。这可以包括“repo”、“讨论”或两者。

秘密（`str`，可选）：用于签署有效负载的秘密，提供额外的安全层。

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** [WebhookInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookInfo)

有关更新的 Webhook 的信息。

更新现有的 Webhook。

示例：
```python
>>> from huggingface_hub import update_webhook
>>> updated_payload = update_webhook(
...     webhook_id="654bbbc16f2ec14d77f109cc",
...     url="https://new.webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
...     watched=[{"type": "user", "name": "julien-c"}, {"type": "org", "name": "HuggingFaceH4"}],
...     domains=["repo"],
...     secret="my-secret",
... )
>>> print(updated_payload)
WebhookInfo(
    id="654bbbc16f2ec14d77f109cc",
    job=None,
    url="https://new.webhook.site/a2176e82-5720-43ee-9e06-f91cb4c91548",
    watched=[WebhookWatchedItem(type="user", name="julien-c"), WebhookWatchedItem(type="org", name="HuggingFaceH4")],
    domains=["repo"],
    secret="my-secret",
    disabled=False,
```

#### upload_file[[huggingface_hub.HfApi.upload_file]]

```python
upload_file(path_or_fileobj: str | Path | bytes | BinaryIO, path_in_repo: str, repo_id: str, token: str | bool | None = None, repo_type: str | None = None, revision: str | None = None, commit_message: str | None = None, commit_description: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None, run_as_future: bool = False, _hot_reload: bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L5663)

**参数：**

path_or_fileobj (`str`、`Path`、`bytes`或`IO`)：本地计算机上的文件或二进制数据流/fileobj/缓冲区的路径。

path_in_repo (`str`) ：存储库中的相对文件路径，例如：`"checkpoints/1fec34a/weights.bin"`

repo_id (`str`) : 文件将上传到的存储库，例如：`"username/custom_transformers"`令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

commit_message (`str`, *可选*) : 生成的提交的摘要/标题/第一行

commit_description (`str` *可选*) : 生成的提交的描述

create_pr (`boolean`, *可选*) ：是否使用该提交创建拉取请求。默认为`False`。如果未设置`revision`，则针对`"main"`分支打开PR。如果 `revision` 被设置并且是一个分支，则针对该分支打开 PR。如果设置了 `revision` 并且不是分支名称（例如：提交 oid），则服务器会返回 `RevisionNotFoundError`。Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。如果指定且 `create_pr` 为 `True`，则将从 `parent_commit` 创建拉取请求。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交，则特别有用。

run_as_future (`bool`, *可选*) : 是否在后台运行此方法。后台作业按顺序运行，不会阻塞主线程。传递 `run_as_future=True` 将返回一个 [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) 对象。默认为`False`。

**返回：** [CommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitInfo) 或 `Future`

[CommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitInfo) 的实例，包含有关新创建的提交的信息（提交哈希、提交
url、pr url、提交消息...）。如果 `run_as_future=True` 被传递，则返回一个 Future 对象，该对象将
包含执行时的结果。

将本地文件（最多 50 GB）上传到给定的存储库。上传完成
通过 HTTP post 请求，并且不需要 git 或 git-lfs
安装。> [!提示]
> 引发以下错误：
>
> - [⟦T2818⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T2819⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它不存在，
> 或者因为它设置为 `private` 并且您无权访问。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果找不到要下载的版本。

> [!警告]
> `upload_file` 假设该存储库已存在于 Hub 上。如果你得到一个
> 客户端错误404，请确保您已通过身份验证，您的令牌具有所需的权限，
> 并且 `repo_id` 和 `repo_type` 设置正确。如果回购不存在，
> 首先使用 [create_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建它。

示例：

```python
>>> from huggingface_hub import upload_file

>>> with open("./local/filepath", "rb") as fobj:
...     upload_file(
...         path_or_fileobj=fileobj,
...         path_in_repo="remote/file/path.h5",
...         repo_id="username/my-dataset",
...         repo_type="dataset",
...         token="my_token",
...     )

>>> upload_file(
...     path_or_fileobj=".\\local\\file\\path",
...     path_in_repo="remote/file/path.h5",
...     repo_id="username/my-model",
...     token="my_token",
... )

>>> upload_file(
...     path_or_fileobj=".\\local\\file\\path",
...     path_in_repo="remote/file/path.h5",
...     repo_id="username/my-model",
...     token="my_token",
...     create_pr=True,
... )
```

#### upload_folder[[huggingface_hub.HfApi.upload_folder]]

```python
upload_folder(repo_id: str, folder_path: str | Path, path_in_repo: str | None = None, commit_message: str | None = None, commit_description: str | None = None, token: str | bool | None = None, repo_type: str | None = None, revision: str | None = None, create_pr: bool | None = None, parent_commit: str | None = None, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, delete_patterns: list[str] | str | None = None, run_as_future: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L5847)

**参数：**

repo_id (`str`) : 文件将上传到的存储库，例如：`"username/custom_transformers"`

folder_path（`str`或`Path`）：要上传到本地文件系统的文件夹的路径

path_in_repo (`str`, *可选*) ：存储库中目录的相对路径，例如：`"checkpoints/1fec34a/results"`。将默认为存储库的根文件夹。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

repo_type (`str`, *可选*) ：如果上传到数据集或空间，则设置为 `"dataset"` 或 `"space"`；如果上传到模型，则设置为 `None` 或 `"model"`。默认为`None`。

revision (`str`, *可选*) ：要提交的 git 修订版本。默认为 `"main"` 分支的头部。

commit_message (`str`, *可选*) ：生成的提交的摘要/标题/第一行。默认为：`f"Upload {path_in_repo} with huggingface_hub"`

commit_description (`str` *可选*) : 生成的提交的描述

create_pr (`boolean`, *可选*) ：是否使用该提交创建拉取请求。默认为 `False`。 PR 始终针对默认分支打开：同时设置 `create_pr=True` 和 `revision` 会引发 `ValueError`。请注意，每次调用 `create_pr=True` 都会打开一个新的拉取请求：要恢复中断的上传到现有 PR，请使用 `revision="refs/pr/N"` 重新运行。Parent_commit (`str`, *可选*) ：父提交的 OID / SHA，作为十六进制字符串。还支持简写（前 7 个字符）。如果指定并且`create_pr`是`False`，则如果`revision`不指向`parent_commit`，提交将会失败。指定 `parent_commit` 确保存储库在提交更改之前没有更改，并且如果存储库同时更新/提交，则特别有用。如果上传被分成多个提交（大文件夹），`parent_commit`仅适用于第一个提交。

allowed_pa​​tterns（`list[str]`或`str`，*可选*）：如果提供，则仅上传至少匹配一种模式的文件。

ignore_patterns（`list[str]` 或 `str`，*可选*）：如果提供，则不会上传与任何模式匹配的文件。

delete_patterns （`list[str]` 或 `str`，*可选*）：如果提供，则在提交新文件时，匹配任何模式的远程文件将从存储库中删除。如果您不知道哪些文件已上传，这非常有用。注意：为了避免差异，即使 `.gitattributes` 文件与模式匹配，也不会删除它。run_as_future (`bool`, *可选*) : 是否在后台运行此方法。后台作业按顺序运行，不会阻塞主线程。传递 `run_as_future=True` 将返回一个 [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) 对象。默认为`False`。

**返回：** [CommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitInfo) 或 `Future`

[CommitInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitInfo) 的实例，包含有关新创建的提交的信息（提交哈希、提交
url、pr url、提交消息...）。如果 `run_as_future=True` 被传递，则返回一个 Future 对象，该对象将
包含执行时的结果。

将本地文件夹上传到给定的存储库。上传是通过 HTTP 请求完成的，不需要 git 或
要安装 git-lfs。

文件夹的结构将被保留。存储库中已存在的同名文件将
被覆盖。其他人将保持不变。

使用 `allow_patterns` 和 `ignore_patterns` 参数指定要上传的文件。这些参数
接受单个模式或模式列表。模式是标准通配符（通配符模式）为
记录为[here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)。如果 `allow_patterns` 和
提供了`ignore_patterns`，两个约束都适用。默认情况下，会上传该文件夹中的所有文件。使用 `delete_patterns` 参数指定要删除的远程文件。输入类型与
`allow_patterns`（见上文）。如果还提供了 `path_in_repo`，则模式将与路径匹配
相对于此文件夹。例如，`upload_folder(..., path_in_repo="experiment", delete_patterns="logs/*")`
将删除`./experiment/logs/`下的任何远程文件。注意`.gitattributes`文件不会被删除
即使它与模式匹配。

任何子目录中存在的任何 `.git/` 文件夹都将被忽略。但是，请注意`.gitignore`
不考虑文件。

安装`hf_xet`（默认）后，文件将通过流式管道上传：上传开始时
该文件夹仍在根据集线器进行检查，文件在分块上传时进行哈希处理（单个
读取通行证），并且大文件夹会自动分批提交以保持在服务器限制以下
（后续提交在提交消息上获得 ` (part N)` 后缀）。如果上传中断，重新运行
相同的调用将恢复它：跳过已提交的文件并对已上传的数据进行重复数据删除。当
`hf_xet` 未安装，回退到使用 [create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 创建的单个提交。> [!提示]
> 引发以下错误：
>
> - [⟦T2885⟧](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
> 如果 HuggingFace API 返回错误
> - [⟦T2886⟧](https://docs.python.org/3/library/exceptions.html#ValueError)
> 如果某些参数值无效

> [!警告]
> `upload_folder` 假设该存储库已存在于 Hub 上。如果您收到客户端错误 404，请
> 确保您已通过身份验证，您的令牌具有所需的权限，并且 `repo_id` 和 `repo_type`
> 设置正确。如果 repo 不存在，请先使用 [create_repo()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建它。

示例：

```python
# Upload checkpoints folder except the log files
>>> upload_folder(
...     folder_path="local/checkpoints",
...     path_in_repo="remote/experiment/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="datasets",
...     token="my_token",
...     ignore_patterns="**/logs/*.txt",
... )

# Upload checkpoints folder including logs while deleting existing logs from the repo
# Useful if you don't know exactly which log files have already being pushed
>>> upload_folder(
...     folder_path="local/checkpoints",
...     path_in_repo="remote/experiment/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="datasets",
...     token="my_token",
...     delete_patterns="**/logs/*.txt",
... )

# Upload checkpoints folder while creating a PR
>>> upload_folder(
...     folder_path="local/checkpoints",
...     path_in_repo="remote/experiment/checkpoints",
...     repo_id="username/my-dataset",
...     repo_type="datasets",
...     token="my_token",
...     create_pr=True,
... )
```

#### upload_large_folder[[huggingface_hub.HfApi.upload_large_folder]]

```python
upload_large_folder(repo_id: str, folder_path: str | Path, repo_type: str, revision: str | None = None, private: bool | None = None, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, num_workers: int | None = None, print_report: bool = True, print_report_every: int = 60)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L6316)

**参数：**

repo_id (`str`) ：文件将上传到的存储库。例如。 `"HuggingFaceTB/smollm-corpus"`。

folder_path（`str`或`Path`）：要上传到本地文件系统的文件夹的路径。

repo_type (`str`) ：存储库的类型。必须是 `"model"`、`"dataset"` 或 `"space"` 之一。与所有其他 `HfApi` 方法不同，此处明确需要 `repo_type`。这是为了避免将大文件夹上传到集线器时出现任何错误，从而避免必须重新上传所有内容。

revision (`str`, `optional`) ：要提交的分支。如果未提供，将使用`main`分支。private (`bool`, `optional`) ：存储库是否应该是私有的。如果`None`（默认），则存储库将是公开的，除非组织默认为私有。

allowed_pa​​tterns (`list[str]` 或 `str`, *可选*) ：如果提供，则仅上传至少匹配一种模式的文件。

ignore_patterns（`list[str]` 或 `str`，*可选*）：如果提供，则不会上传与任何模式匹配的文件。

num_workers (`int`, *可选*) ：要启动的工人数量。默认为 CPU 核心的一半（最少 1 个）。如果您的机器允许，更多的工人可能会加快该过程。但是，在连接速度较慢的计算机上，建议保持较低的工作线程数量以确保更好的可恢复性。事实上，如果过程中断，则必须完全重新上传部分上传的文件。

print_report (`bool`, *可选*) : 是否打印上传进度报告。默认为 True。报告每 X 秒（默认为 60）打印到 `sys.stdout` 并覆盖之前的报告。

print_report_every (`int`, *可选*) ：打印报告的频率。默认为 60 秒。以最具弹性的方式将大文件夹上传到集线器。

> [!警告]
> `upload_large_folder` 已弃用，并将在未来版本中删除。 [upload_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 现在是多次提交
> 默认情况下，可以抵抗中断，因此这是上传大型文件夹的推荐方法。

几个worker开始以优化的方式上传文件。在提交到存储库之前，文件必须
如果它们是 LFS 文件，则进行哈希处理并预上传。工作人员将为文件夹中的每个文件执行这些任务。
在每一步中，有关上传过程的一些元数据信息都会保存在`.cache/.huggingface/`下的文件夹中
如果中断的话能够恢复该过程。整个过程可能会导致多次提交。

> [!提示]
> 请记住以下几点：
> - 存储库限制仍然适用：https://huggingface.co/docs/hub/repositories-recommendations
> - 不要并行启动多个进程。
> - 您可以随时中断和恢复该过程。
> - 不要将同一文件夹上传到多个存储库。如果需要这样做，必须先删除本地`.cache/.huggingface/`文件夹。> [!警告]
> 虽然在上传大型文件夹方面更加强大，但 `upload_large_folder` 在功能方面比 [upload_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 更有限。在实践中：
> - 您无法设置自定义`path_in_repo`。如果要上传到子文件夹，则需要在本地设置正确的结构。
> - 您无法设置自定义 `commit_message` 和 `commit_description`，因为会创建多个提交。
> - 上传时无法从存储库中删除。请先单独提交。
> - 您无法直接创建 PR。请先创建一个 PR（从 UI 或使用 [create_pull_request()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_pull_request)），然后通过传递 `revision` 来提交。

**技术细节：**`upload_large_folder`流程如下：
1.（检查参数和设置。）
2. 如果缺少，请创建存储库。
3. 列出要上传的本地文件。
4. 运行验证检查并在可能超出存储库限制时显示警告：
   - 如果文件总数超过 100k（建议限制），则会发出警告。
   - 如果任何文件夹包含超过 10k 个文件（建议限制），则会发出警告。
   - 对大于 20GB（推荐）或 50GB（硬限制）的文件发出警告。
5. 启动工人。工人可以执行以下任务：
   - 对文件进行哈希处理。
   - 获取文件列表的上传模式（常规或 LFS）。
   - 预先上传 LFS 文件。
   - 提交一堆文件。
一旦工作人员完成一项任务，它将根据优先级列表（见下文）继续执行下一个任务，直到
所有文件均已上传并提交。
6. 当工作人员启动时，定期将报告打印到 sys.stdout。优先顺序：
1. 如果距离上次提交尝试（且至少 1 个文件）超过 5 分钟，则提交。
2. 如果至少有 150 个文件准备好提交，则提交。
3. 如果至少 10 个文件已被散列，则获取上传模式。
4. 如果至少有 1 个文件且没有工作人员预上传，则预上传 LFS 文件。
5. 如果至少有 1 个文件并且没有工作人员正在散列，则散列文件。
6. 如果至少有 1 个文件并且没有工作人员正在获取上传模式，则获取上传模式。
7. 如果至少有 1 个文件，则预上传 LFS 文件。
8. 哈希文件（如果至少有 1 个文件要哈希）。
9. 如果至少有 1 个文件要获取上传模式，则获取上传模式。
10. 如果至少有 1 个文件需要提交，并且距离上次提交尝试至少已经过去 1 分钟，则提交。
11. 如果至少有 1 个文件要提交且所有其他队列为空，则提交。

特别规则：
- 一次只有一名工人可以提交。
- 如果没有可用任务，工作人员会等待 10 秒，然后再次检查。

#### verify_repo_checksums[[huggingface_hub.HfApi.verify_repo_checksums]]

```python
verify_repo_checksums(repo_id: str, repo_type: str | None = None, revision: str | None = None, local_dir: str | Path | None = None, cache_dir: str | Path | None = None, token: str | bool | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L4096)

**参数：**

repo_id (`str`) ：命名空间（用户或组织）和存储库名称，由`/`分隔。repo_type (`str`, *可选*) ：从中获取树的存储库的类型（`"model"`，`"dataset"`或`"space"`。默认为`"model"`。

revision (`str`, *可选*) ：从中获取树的存储库的修订版本。默认为 `"main"` 分支。

local_dir（`str`或`Path`，*可选*）：要验证的本地目录。

cache_dir （`str` 或 `Path`，*可选*）：要验证的缓存目录。

token (Union[bool, str, None], 可选) ：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。

**退货：** `FolderVerification`

包含验证详细信息的结构化结果。

**加薪：** [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) 或 [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)

- [RepositoryNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError) -- 
  如果未找到存储库（错误 404）：错误的 repo_id/repo_type、私有但未经身份验证或存储库
  不存在。
- [RevisionNotFoundError](/docs/huggingface_hub/v1.27.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError) -- 
  如果在存储库中未找到修订版（错误 404）。

根据集线器校验和验证存储库的本地文件。

#### wait_for_job[[huggingface_hub.HfApi.wait_for_job]]

```python
wait_for_job(job_id: str | list[str], timeout: float | None = None, poll_interval: float = 1.0, stages: list[JobStage] | None = None, namespace: str | None = None, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L12495)

**参数：**job_id (`str` 或 `list[str]`) ：作业的 ID，或要等待的作业 ID 列表。如果传递了列表，则返回[JobInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobInfo)的列表（顺序相同）。 

timeout (`float`, *可选*) ：等待作业完成的最长时间，以秒为单位。如果`None`，将无限期地等待。 

poll_interval (`float`, *可选*) ：每次状态检查之间等待的时间，以秒为单位。默认为 1 秒。 

stage (`list[JobStage]`, *可选*) ：要等待的阶段。默认为终端阶段（`"COMPLETED"`、`"CANCELED"`、`"ERROR"`、`"DELETED"`）。通过例如`[JobStage.RUNNING]`等待作业开始运行。无论该值如何，终端阶段始终停止等待。 

命名空间（`str`，*可选*）：运行作业的命名空间。默认为当前用户的命名空间。 

token `(Union[bool, str, None]`，*可选*) ：有效的用户访问令牌。如果未提供，将使用本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅：https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** [JobInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobInfo) 或 `list[JobInfo]`

最终的职位信息。

**加薪：** ``TimeoutError``- ``TimeoutError`` -- 
  如果至少一项作业在 `timeout` 秒后尚未到达目标阶段之一。

等待 Hugging Face 基础设施上的一个或多个计算作业达到给定阶段。

每个作业状态每 `poll_interval` 秒轮询一次（使用 [inspect_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.inspect_job)），直到其阶段为 1
`stages`（默认情况下，终端阶段：`"COMPLETED"`、`"CANCELED"`、`"ERROR"` 或 `"DELETED"`）。的
在所有情况下都会返回最终的[JobInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobInfo)：失败或取消的作业**不会**引发异常 -
检查`job.status.stage`以根据结果采取行动。

终端阶段始终会停止等待，即使未在 `stages` 中列出。这可以避免永远等待
作业永远不会到达的阶段（例如，在调度期间失败的作业上等待`"RUNNING"`）。

示例：

```python
>>> from huggingface_hub import run_job, wait_for_job
>>> job = run_job(image="python:3.12", command=["python", "-c", "print('Hello from HF compute!')"])
>>> wait_for_job(job_id=job.id).status.stage
'COMPLETED'
```

#### wait_for_space[[huggingface_hub.HfApi.wait_for_space]]

```python
wait_for_space(repo_id: str, timeout: float | None = None, poll_interval: float = 1.0, token: bool | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L8778)

**参数：**

repo_id (`str`) : 等待的空间ID。示例：`"username/my-space"`。

超时（`float`，*可选*）：等待的最长时间（以秒为单位）。如果`None`，则无限期等待。

poll_interval (`float`, *可选*) ：状态检查之间的秒数。默认为 1 秒。令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌。默认为本地保存的令牌，这是推荐的身份验证方法。设置为 `False` 以禁用身份验证。请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication。

**返回：** [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)

空间到达终点阶段后的最终运行时间信息。

**加薪：** ``TimeoutError``

- ``TimeoutError`` -- 
  如果空间在`timeout`秒后仍未到达终点。

等待空间到达最终阶段（不是构建/启动）。

每 `poll_interval` 秒进行一次投票 [get_space_runtime()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_space_runtime)，直至空间舞台
不再是中间 (`BUILDING`, `RUNNING_BUILDING`, `APP_STARTING`,
`RUNNING_APP_STARTING`）。在所有情况下返回最终的 [SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime) — 检查
`runtime.stage` 根据结果采取行动（例如 `RUNNING` 与 `BUILD_ERROR`）。

示例：

```python
>>> from huggingface_hub import restart_space, wait_for_space
>>> restart_space("username/my-space")
>>> runtime = wait_for_space("username/my-space")
>>> runtime.stage
'RUNNING'
```

#### whoami[[huggingface_hub.HfApi.whoami]]

```python
whoami(token: bool | str | None = None, cache: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2322)

**参数：**

令牌（`bool` 或 `str`，*可选*）：有效的用户访问令牌（字符串）。默认为本地保存的令牌，这是推荐的身份验证方法（请参阅 https://huggingface.co/docs/huggingface_hub/quick-start#authentication）。要禁用身份验证，请传递`False`。cache (`bool`, *可选*) : 是否缓存`whoami`调用的结果以供后续调用。如果在第一次调用期间发生错误，则不会缓存该错误。默认为 `False`。

调用 HF API 即可了解“whoami”。

如果传递 `cache=True`，结果将被缓存以供 Python 进程期间的后续调用。如果您打算打电话，这很有用
`whoami` 多次，因为出于安全原因，此端点受到严格的速率限制。

## API 数据类

### AccessRequest[[huggingface_hub.hf_api.AccessRequest]]

#### Huggingface_hub.hf_api.AccessRequest[[huggingface_hub.hf_api.AccessRequest]]

```python
huggingface_hub.hf_api.AccessRequest(username: str, fullname: str, email: str | None, timestamp: datetime, status: Literal['pending', 'accepted', 'rejected'], fields: dict[str, Any] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L573)

**参数：**

用户名 (`str`) ：请求访问的用户的用户名。

fullname (`str`) ：请求访问的用户的全名。

email (`Optional[str]`) : 请求访问的用户的电子邮件。如果用户被手动授予访问权限，则只能在 /accepted 列表中包含 `None`。

timestamp (`datetime`) : 请求的时间戳。

status (`Literal["pending", "accepted", "rejected"]`) ：请求的状态。可以是 `["pending", "accepted", "rejected"]` 之一。

fields (`dict[str, Any]`, *可选*) ：用户在门表单中填写的附加字段。

包含有关用户访问请求的信息的数据结构。### BucketFile[[huggingface_hub.BucketFile]]

#### Huggingface_hub.BucketFile[[huggingface_hub.BucketFile]]

```python
huggingface_hub.BucketFile(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_buckets.py#L195)

包含有关集线器上存储桶中的文件的信息。该对象由[list_bucket_tree()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_bucket_tree)返回。

与 [RepoFile](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.RepoFile) 类似，但适用于存储桶中的文件。

### BucketFileMetadata[[huggingface_hub.BucketFileMetadata]]

#### Huggingface_hub.BucketFileMetadata[[huggingface_hub.BucketFileMetadata]]

```python
huggingface_hub.BucketFileMetadata(size: int, xet_file_data: XetFileData)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_buckets.py#L138)

**参数：**

size (`int`) ：文件的大小（以字节为单位）。

xet_file_data (`XetFileData`) ：文件的 Xet 信息（哈希和刷新路由）。

包含有关存储桶中文件信息的数据结构。

由[get_bucket_file_metadata()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_bucket_file_metadata)返回。

### BucketInfo[[huggingface_hub.BucketInfo]]

#### Huggingface_hub.BucketInfo[[huggingface_hub.BucketInfo]]

```python
huggingface_hub.BucketInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_buckets.py#L63)

**参数：**

id (`str`) ：桶的ID。

private (`bool`) ：存储桶是否私有。

created_at (`datetime`) ：在 Hub 上创建存储桶的日期。

size (`int`) ：存储桶的大小（以字节为单位）。

Total_files (`int`) ：存储桶中的文件总数。

包含有关 Hub 上存储桶的信息。该对象由 [bucket_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.bucket_info) 和 [list_buckets()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_buckets) 返回。

### BucketUrl[[huggingface_hub.BucketUrl]]#### Huggingface_hub.BucketUrl[[huggingface_hub.BucketUrl]]

```python
huggingface_hub.BucketUrl(url: str, endpoint: str = '')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_buckets.py#L155)

**参数：**

url (`str`) ：存储桶 url 的字符串值。

端点（`str`，*可选*）：集线器的端点。默认为 .

描述 Hub 上的存储桶 URL。

`BucketUrl` 由[create_bucket()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_bucket) 返回。在初始化时，URL 被解析以填充属性：
- 端点（`str`）
- 命名空间（`str`）
-bucket_id (`str`)
- 网址 (`str`)
- uri (`HfUri`)

### DatasetLeaderboardEntry[[huggingface_hub.DatasetLeaderboardEntry]]

#### Huggingface_hub.DatasetLeaderboardEntry[[huggingface_hub.DatasetLeaderboardEntry]]

```python
huggingface_hub.DatasetLeaderboardEntry(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L2058)

**参数：**

排名 (`int`) ：模型在排行榜上的排名（1-索引）。

model_id (`str`) ：模型的 ID（例如 `"meta-llama/Llama-3-8b"`）。

value (`float`) ：评估分数值。

filename (`str`) ：包含评估数据的结果文件的名称。

已验证(`bool`)：结果是否已验证。

source (`dict[str, Any]`, *可选*) ：有关评估结果来源的信息。包含 `"url"`、`"name"` 和 `"isExternal"` 等键。并非所有条目都有来源。作者（`User`或`Organization`）：模型作者，根据API响应中的`"type"`字段解析。

pull_request (`int`, *可选*) ：与排行榜条目关联的拉取请求编号（如果有）。

注释（`str`，*可选*）：与排行榜条目关联的注释（如果有）。

包含有关 Hub 上数据集排行榜中单个条目的信息。

排行榜根据给定基准数据集的评估分数对模型进行排名。
该对象由[get_dataset_leaderboard()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_dataset_leaderboard)返回。获取评估结果
跨基准测试的具体模型，请参阅`ModelInfo.eval_results`（通过[model_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)）
`expand=["evalResults"]`）和[EvalResultEntry](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.EvalResultEntry)。

### EvalResultEntry[[huggingface_hub.EvalResultEntry]]

#### Huggingface_hub.EvalResultEntry[[huggingface_hub.EvalResultEntry]]

```python
huggingface_hub.EvalResultEntry(dataset_id: str, task_id: str, value: typing.Any, dataset_revision: str | None = None, verify_token: str | None = None, date: str | None = None, source_url: str | None = None, source_name: str | None = None, source_user: str | None = None, source_org: str | None = None, notes: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_eval_results.py#L12)

**参数：**

dataset_id (`str`) ：来自 Hub 的基准数据集 ID。示例：“cais/hle”、“Idavidrein/gpqa”。

task_id (`str`) ：基准测试中的任务标识符。示例：“gpqa_diamond”。

value (`Any`) ：度量值。示例：20.90。

dataset_revision (`str`, *可选*) ：基准数据集的 Git SHA。verify_token (`str`, *可选*) ：可用于证明评估可证明可审计和可重现的签名。

日期（`str`，*可选*）：运行评估的时间（ISO-8601 日期时间）。默认为 git 提交时间。

source_url（`str`，*可选*）：链接到评估源（例如，https://huggingface.co/spaces/SaylorTwift/smollm3-mmlu-pro）。如果提供了 `source_name`、`source_user` ​​或 `source_org`，则为必需。

source_name (`str`, *可选*) ：源的显示名称。示例：“评估日志”。

source_user (`str`, *可选*) ：用于归因的 HF 用户名。例如：“celinah”。

source_org（`str`，*可选*）：用于归因的 HF 组织名称。示例：“cais”。

注释（`str`，*可选*）：有关评估设置的详细信息。例如：“工具”、“无工具”、“思想链”。

`.eval_results/*.yaml` 格式的评估结果条目。

表示存储在模型存储库中的评估分数，该分数自动出现在
模型页面和基准数据集的排行榜。

对于 `README.md` 中的旧版 `model-index` 格式，请改用 [EvalResult](/docs/huggingface_hub/v1.27.0/en/package_reference/cards#huggingface_hub.EvalResult)。

有关更多详细信息，请参阅 https://huggingface.co/docs/hub/eval-results。

示例：
```python
>>> from huggingface_hub import EvalResultEntry
>>> # Minimal example with required fields only
>>> result = EvalResultEntry(
...     dataset_id="Idavidrein/gpqa",
...     task_id="gpqa_diamond",
...     value=0.412,
... )
>>> # Full example with all fields
>>> result = EvalResultEntry(
...     dataset_id="cais/hle",
...     task_id="default",
...     value=20.90,
...     dataset_revision="5503434ddd753f426f4b38109466949a1217c2bb",
...     verify_token="REDACTED",
...     date="2025-01-15T10:30:00Z",
...     source_url="https://huggingface.co/datasets/cais/hle",
...     source_name="CAIS HLE",
...     source_org="cais",
...     notes="no-tools",
... )

```

### SyncOperation[[huggingface_hub.SyncOperation]]#### Huggingface_hub.SyncOperation[[huggingface_hub.SyncOperation]]

```python
huggingface_hub.SyncOperation(action: typing.Literal['upload', 'download', 'delete', 'skip'], path: str, size: int | None = None, reason: str = '', local_mtime: str | None = None, remote_mtime: str | None = None, bucket_file: huggingface_hub._buckets.BucketFile | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_buckets.py#L287)

表示要执行的同步操作。

### SyncPlan[[huggingface_hub.SyncPlan]]

#### Huggingface_hub.SyncPlan[[huggingface_hub.SyncPlan]]

```python
huggingface_hub.SyncPlan(source: str, dest: str, timestamp: str, operations: list = <factory>)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_buckets.py#L300)

代表一个完整的同步计划。

### CommitInfo[[huggingface_hub.CommitInfo]]

#### Huggingface_hub.CommitInfo[[huggingface_hub.CommitInfo]]

```python
huggingface_hub.CommitInfo(*args, commit_url: str, **kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L501)

**参数：**

commit_url (`str`) ：查找提交的 URL。 

commit_message (`str`) ：已创建的提交的摘要（第一行）。 

commit_description (`str`) ：已创建的提交的描述。可以为空。 

oid (`str`) : 提交哈希 ID。示例：`"91c54ad1727ee830252e457677f467be0bfd8a57"`。 

pr_url (`str`, *可选*) ：已创建的 PR 的 URL（如果有）。当`create_pr=True`通过时填充。 

pr_revision (`str`, *可选*) ：已创建的 PR 的修订版本（如果有）。当`create_pr=True`通过时填充。示例：`"refs/pr/1"`。 

pr_num (`int`, *可选*) ：已创建的 PR 讨论的数量（如果有）。当`create_pr=True`通过时填充。可以在[get_discussion_details()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_discussion_details)中作为`discussion_num`传递。示例：`1`。repo_url (`RepoUrl`) ：包含 repo_id、repo_type 等信息的提交的 Repo URL。

包含有关新创建的提交的信息的数据结构。

由在集线器上创建提交的任何方法返回：[create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)、[upload_file()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)、[upload_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)、
[delete_file()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_file)，[delete_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_folder)。它继承自`str`以实现向后兼容性，但使用特定的方法
`str` 已弃用。

### DatasetInfo[[huggingface_hub.DatasetInfo]]

#### Huggingface_hub.DatasetInfo[[huggingface_hub.DatasetInfo]]

```python
huggingface_hub.DatasetInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1123)

**参数：**

id (`str`) ：数据集的ID。

作者 (`str`) ：数据集的作者。

card_data (`DatasetCardData`, *可选*) ：作为[huggingface_hub.repocard_data.DatasetCardData](/docs/huggingface_hub/v1.27.0/en/package_reference/cards#huggingface_hub.DatasetCardData)对象的数据集卡元数据。

引用（`str`，*可选*）：数据集的引用信息。

created_at（`datetime`，*可选*）：在 Hub 上创建存储库的日期。请注意，最小值是`2022-03-02T23:29:04.000Z`，对应于我们开始存储创建日期的日期。

描述（`str`，*可选*）：数据集的描述。

禁用（`bool`，*可选*）：存储库是否已禁用。

downloads (`int`) ：过去 30 天内数据集的下载次数。downloads_all_time (`int`) ：数据集自创建以来的累计下载次数。

gate (`Literal["auto", "manual", False]`, *可选*) : 仓库是否有门控。如果是，是否有手动或自动批准。

last_modified (`datetime`, *可选*) ：上次提交到存储库的日期。

喜欢 (`int`) ：数据集的喜欢数量。

main_size (`int`, *可选*) ：数据集主分支的大小（以字节为单位）。

paperwithcode_id (`str`, *可选*) ：带有数据集代码 ID 的论文。

private (`bool`) : 仓库是私有的。

resource_group（`dict`，*可选*）：数据集的资源组信息。

sha (`str`)：此特定修订版的 Repo SHA。

兄弟姐妹 (`list[RepoSibling]`) ：构成数据集的 [huggingface_hub.hf_api.RepoSibling](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.hf_api.RepoSibling) 对象列表。

Tags (`list[str]`) ：数据集的标签列表。

trending_score (`int`, *可选*) ：数据集的趋势分数。

used_storage (`int`, *可选*) ：集线器上数据集的大小（以字节为单位）。

包含有关 Hub 上数据集的信息。该对象由 [dataset_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info) 和 [list_datasets()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) 返回。> [!提示]
> 该类的大多数属性都是可选的。这是因为 Hub 返回的数据取决于所做的查询。
> 一般来说，查询越具体，返回的信息就越多。相反，当列出数据集时
> 使用 [list_datasets()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) 仅返回属性的子集。

### DryRunFileInfo[[huggingface_hub.DryRunFileInfo]]

#### Huggingface_hub.DryRunFileInfo[[huggingface_hub.DryRunFileInfo]]

```python
huggingface_hub.DryRunFileInfo(commit_hash: str, file_size: int, filename: str, local_path: str, is_cached: bool, will_download: bool)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/file_download.py#L174)

**参数：**

commit_hash (`str`) ：与文件相关的commit_hash。

file_size (`int`) ：文件的大小。如果是 LFS 文件，则包含实际 LFS 文件的大小，而不是指针。

filename (`str`) ：存储库中文件的名称。

is_cached (`bool`) : 文件是否已经缓存在本地。

will_download (`bool`) : 如果使用`dry_run=False`调用`hf_hub_download`，是否会下载文件。实际上，如果文件未缓存或`force_download=True`，则will_download为`True`。

执行文件下载试运行时返回的信息。

由 [hf_hub_download()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 在 `dry_run=True` 时返回。

### GitRefInfo[[huggingface_hub.GitRefInfo]]

#### Huggingface_hub.GitRefInfo[[huggingface_hub.GitRefInfo]]

```python
huggingface_hub.GitRefInfo(name: str, ref: str, target_commit: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1554)

**参数：**name (`str`) ：引用的名称（例如标签名称或分支名称）。

ref (`str`) ：Hub 上的完整 git 参考（例如 `"refs/heads/main"` 或 `"refs/tags/v1.0"`）。

target_commit (`str`) ：引用的目标提交的 OID（例如 `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`）

包含有关 Hub 上存储库的 git 参考的信息。

### GitCommitInfo[[huggingface_hub.GitCommitInfo]]

#### Huggingface_hub.GitCommitInfo[[huggingface_hub.GitCommitInfo]]

```python
huggingface_hub.GitCommitInfo(commit_id: str, authors: list[str], created_at: datetime, title: str, message: str, formatted_title: str | None, formatted_message: str | None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1599)

**参数：**

commit_id (`str`) ：提交的 OID（例如 `"e7da7f221d5bf496a48136c0cd264e630fe9fcc8"`）

作者 (`list[str]`) ：提交的作者列表。

created_at (`datetime`) ：创建提交的日期时间。

title (`str`) ：提交的标题。这是作者输入的自由文本值。

message (`str`) ：提交的描述。这是作者输入的自由文本值。

formatted_title (`str`) ：格式化为 HTML 的提交标题。仅当设置了 `formatted=True` 时才返回。

formatted_message (`str`) ：格式化为 HTML 的提交描述。仅当设置了 `formatted=True` 时才返回。

包含有关 Hub 上存储库的 git 提交的信息。查看[list_repo_commits()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_commits)了解更多详情。

### GitRefs[[huggingface_hub.GitRefs]]#### Huggingface_hub.GitRefs[[huggingface_hub.GitRefs]]

```python
huggingface_hub.GitRefs(branches: list[GitRefInfo], converts: list[GitRefInfo], tags: list[GitRefInfo], pull_requests: list[GitRefInfo] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1573)

**参数：**

分支 (`list[GitRefInfo]`) ：[GitRefInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.GitRefInfo) 的列表，包含有关存储库上分支的信息。

Converts (`list[GitRefInfo]`) ：[GitRefInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.GitRefInfo) 的列表，包含有关存储库上“转换”引用的信息。转换是（内部）用于将预处理数据推送到数据集存储库中的引用。

Tags (`list[GitRefInfo]`) ：[GitRefInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.GitRefInfo) 的列表，包含有关存储库上标签的信息。

pull_requests (`list[GitRefInfo]`, *可选*) ：[GitRefInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.GitRefInfo)的列表，包含有关存储库上拉取请求的信息。仅当设置了 `include_prs=True` 时才返回。

包含有关 Hub 上存储库的所有 git 引用的信息。

对象由[list_repo_refs()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_refs)返回。

### InferenceProviderMapping[[huggingface_hub.hf_api.InferenceProviderMapping]]

#### Huggingface_hub.hf_api.InferenceProviderMapping[[huggingface_hub.hf_api.InferenceProviderMapping]]

```python
huggingface_hub.hf_api.InferenceProviderMapping(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L883)

### KernelInfo[[huggingface_hub.KernelInfo]]

#### Huggingface_hub.KernelInfo[[huggingface_hub.KernelInfo]]

```python
huggingface_hub.KernelInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1395)

**参数：**

id (`str`) ：内核存储库的 ID。

作者（`str`，*可选*）：内核存储库的作者。downloads (`int`，*可选*)：过去 30 天内内核存储库的下载次数。

gate (`Literal["auto", "manual", False]`, *可选*) : 仓库是否有门控。如果是，是否有手动或自动批准。

last_modified (`datetime`, *可选*) ：上次提交到存储库的日期。

喜欢（`int`，*可选*）：内核存储库的喜欢数量。

private (`bool`, *可选*) : 仓库是否私有。

sha（`str`，*可选*）：此特定修订版的 Repo SHA。

包含有关 Hub 上的内核存储库的信息。该对象由[kernel_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.kernel_info)返回。

### LFSFileInfo[[huggingface_hub.hf_api.LFSFileInfo]]

#### Huggingface_hub.hf_api.LFSFileInfo[[huggingface_hub.hf_api.LFSFileInfo]]

```python
huggingface_hub.hf_api.LFSFileInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1997)

**参数：**

file_oid (`str`) ：文件的 SHA-256 对象 ID。这是永久删除文件时要传递的标识符。

filename (`str`) ：LFS 对象的可能文件名。请参阅上面的注释了解更多信息。

oid (`str`) ：LFS 对象的 OID。

Push_at (`datetime`) ：LFS 对象推送到存储库的日期。

ref (`str`, *可选*) ：LFS 对象已被推送的位置的引用（如果有）。

size (`int`) ：LFS 对象的大小。包含有关在 Hub 上的存储库上以 LFS 形式存储的文件的信息。

在从存储库中列出和永久删除 LFS 文件以释放空间的上下文中使用。
有关更多详细信息，请参阅[list_lfs_files()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_lfs_files) 和 [permanently_delete_lfs_files()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.permanently_delete_lfs_files)。

Git LFS 文件使用 SHA-256 对象 ID（而不是文件路径）进行跟踪，以优化性能
这种方法是必要的，因为单个对象可以被不同提交的多个路径引用，
使得搜索和解析这些连接变得不切实际。查看[our documentation](https://huggingface.co/docs/hub/storage-limits#advanced-track-lfs-file-references)
了解如何知道哪些文件名与每个 SHA 相关联。

示例：
```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> lfs_files = api.list_lfs_files("username/my-cool-repo")

# Filter files files to delete based on a combination of `filename`, `pushed_at`, `ref` or `size`.
# e.g. select only LFS files in the "checkpoints" folder
>>> lfs_files_to_delete = (lfs_file for lfs_file in lfs_files if lfs_file.filename.startswith("checkpoints/"))

# Permanently delete LFS files
>>> api.permanently_delete_lfs_files("username/my-cool-repo", lfs_files_to_delete)
```

### 模型信息[[huggingface_hub.ModelInfo]]

#### Huggingface_hub.ModelInfo[[huggingface_hub.ModelInfo]]

```python
huggingface_hub.ModelInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L908)

**参数：**

id (`str`) ：模型 ID。

作者（`str`，*可选*）：模型的作者。

base_models (`list[str]`, *可选*) ：该模型派生自的基本模型列表。

card_data (`ModelCardData`, *可选*) ：将卡元数据建模为 [huggingface_hub.repocard_data.ModelCardData](/docs/huggingface_hub/v1.27.0/en/package_reference/cards#huggingface_hub.ModelCardData) 对象。

Children_model_count (`int`, *可选*) ：从此模型派生的子模型的数量。

config（`dict`，*可选*）：模型配置。created_at（`datetime`，*可选*）：在 Hub 上创建存储库的日期。请注意，最小值是`2022-03-02T23:29:04.000Z`，对应于我们开始存储创建日期的日期。

禁用（`bool`，*可选*）：存储库是否已禁用。

downloads (`int`)：过去 30 天内模型的下载次数。

downloads_all_time (`int`) ：模型自创建以来的累计下载次数。

eval_results (`list[EvalResultEntry]`, *可选*) ：模型的评估结果。

gate (`Literal["auto", "manual", False]`, *可选*) : 仓库是否有门控。如果是，是否有手动或自动批准。

gguf (`dict`, *可选*) ：模型的 GGUF 信息。

推理（`Literal["warm"]`，*可选*）：推理提供程序上模型的状态。如果模型由至少一个提供商提供服务，则为温暖。

inference_provider_mapping (`list[InferenceProviderMapping]`, *可选*) ：在用户的提供商订单之后排序的 `InferenceProviderMapping` 列表。

last_modified (`datetime`, *可选*) ：上次提交到存储库的日期。

library_name (`str`, *可选*) ：与模型关联的库。

喜欢 (`int`) ：模型的喜欢数量。

mask_token (`str`, *可选*) ：模型使用的掩码令牌。model_index (`dict`, *可选*) ：用于评估的模型索引。

pipeline_tag (`str`, *可选*) ：与模型关联的管道标签。

private (`bool`) : 仓库是私有的。

resource_group（`dict`，*可选*）：模型的资源组信息。

safetensors (`SafeTensorsInfo`，*可选*)：模型的safetensors信息。

security_repo_status（`dict`，*可选*）：模型的安全扫描状态。

sha（`str`，*可选*）：此特定修订版的 Repo SHA。

兄弟姐妹 (`list[RepoSibling]`) ：构成模型的 [huggingface_hub.hf_api.RepoSibling](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.hf_api.RepoSibling) 对象列表。

空间（`list[str]`，*可选*）：使用模型的空间列表。

Tags (`list[str]`) ：模型的标签列表。与`card_data.tags`相比，包含由Hub计算的额外标签（例如支持的库、模型的arXiv）。

Transformers_info (`TransformersInfo`, *可选*) ：与模型关联的 Transformers 特定信息（汽车类、处理器等）。

trending_score (`int`, *可选*) ：模型的趋势分数。

used_storage (`int`, *可选*) ：集线器上模型的大小（以字节为单位）。

widget_data (`Any`, *可选*) ：与模型关联的小部件数据。包含有关 Hub 上模型的信息。该对象由[model_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)和[list_models()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_models)返回。

> [!提示]
> 该类的大多数属性都是可选的。这是因为 Hub 返回的数据取决于所做的查询。
> 一般来说，查询越具体，返回的信息就越多。相反，在列出型号时
> 使用 [list_models()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_models) 仅返回属性的子集。

### RepoSibling[[huggingface_hub.hf_api.RepoSibling]]

#### Huggingface_hub.hf_api.RepoSibling[[huggingface_hub.hf_api.RepoSibling]]

```python
huggingface_hub.hf_api.RepoSibling(rfilename: str, size: int | None = None, blob_id: str | None = None, lfs: BlobLfsInfo | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L758)

**参数：**

rfilename (str) ：文件名，相对于存储库根目录。

size (`int`, *可选*) ：文件的大小，以字节为单位。当[repo_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.repo_info)的`files_metadata`参数设置为`True`时，定义该属性。否则是`None`。

blob_id (`str`, *可选*) ：文件的 git OID。当[repo_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.repo_info)的`files_metadata`参数设置为`True`时，定义该属性。否则是`None`。

lfs (`BlobLfsInfo`, *可选*) ：文件的 LFS 元数据。当[repo_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.repo_info)的`files_metadata`参数设置为`True`并且文件使用Git LFS存储时，定义此属性。否则就是`None`。包含有关 Hub 上存储库内存储库文件的基本信息。

> [!提示]
> 除了 `rfilename` 之外，该类的所有属性都是可选的。这是因为仅返回文件名
> 在 Hub 上列出存储库（使用 [list_models()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_models)、[list_datasets()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) 或 [list_spaces()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces)）。如果您需要更多
> 文件大小、blob id 或 lfs 详细信息等信息，您必须一次专门从一个存储库请求它们
> （使用 [model_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.model_info)、[dataset_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.dataset_info) 或 [space_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.space_info)），因为它对后端服务器添加了更多约束
> 检索这些。

### RepoFile[[huggingface_hub.RepoFile]]

#### Huggingface_hub.RepoFile[[huggingface_hub.RepoFile]]

```python
huggingface_hub.RepoFile(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L790)

**参数：**

path (str) ：相对于存储库根目录的文件路径。

size (`int`) ：文件的大小，以字节为单位。

blob_id (`str`) ：文件的 git OID。

lfs (`BlobLfsInfo`, *可选*) ：文件的 LFS 元数据。

xet_hash (`str`, *可选*) ：文件的 Xet 哈希值。

last_commit (`LastCommitInfo`, *可选*) ：文件的上次提交元数据。仅当使用 `expand=True` 调用 [list_repo_tree()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree) 和 [get_paths_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_paths_info) 时才定义。

security（`BlobSecurityInfo`，*可选*）：文件的安全扫描元数据。仅当使用 `expand=True` 调用 [list_repo_tree()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_repo_tree) 和 [get_paths_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_paths_info) 时才定义。

包含有关集线器上文件的信息。### RepoUrl[[huggingface_hub.RepoUrl]]

#### Huggingface_hub.RepoUrl[[huggingface_hub.RepoUrl]]

```python
huggingface_hub.RepoUrl(url: Any, endpoint: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L649)

**参数：**

url (`Any`) ：存储库 URL 的字符串值。

端点（`str`，*可选*）：集线器的端点。默认为 .

**加薪：** `HfUriError`

- `HfUriError` -- 
  如果无法解析 URL（例如规范的单段存储库，或未知的`repo_type`）。

`str` 的子类，描述 Hub 上的存储库 URL。

`RepoUrl` 由`HfApi.create_repo` 返回。它继承于`str`向后
兼容性。在初始化时，URL 被解析以填充属性：
- 端点（`str`）
- 命名空间（`str`）
- 仓库名称 (`str`)
- repo_id (`str`)
- repo_type (`Literal["model", "dataset", "space"]`)
- 网址 (`str`)

示例：
```py
>>> RepoUrl('https://huggingface.co/openai-community/gpt2')
RepoUrl('https://huggingface.co/openai-community/gpt2', endpoint='https://huggingface.co', repo_type='model', repo_id='openai-community/gpt2')

>>> RepoUrl('https://hub-ci.huggingface.co/datasets/dummy_user/dummy_dataset', endpoint='https://hub-ci.huggingface.co')
RepoUrl('https://hub-ci.huggingface.co/datasets/dummy_user/dummy_dataset', endpoint='https://hub-ci.huggingface.co', repo_type='dataset', repo_id='dummy_user/dummy_dataset')

>>> RepoUrl('hf://datasets/my-user/my-dataset')
RepoUrl('hf://datasets/my-user/my-dataset', endpoint='https://huggingface.co', repo_type='dataset', repo_id='user/dataset')

>>> HfApi.create_repo("dummy_model")
RepoUrl('https://huggingface.co/Wauplin/dummy_model', endpoint='https://huggingface.co', repo_type='model', repo_id='Wauplin/dummy_model')
```

### SafetensorsRepoMetadata[[huggingface_hub.utils.SafetensorsRepoMetadata]]

#### Huggingface_hub.utils.SafetensorsRepoMetadata[[huggingface_hub.utils.SafetensorsRepoMetadata]]

```python
huggingface_hub.utils.SafetensorsRepoMetadata(metadata: dict | None, sharded: bool, weight_map: dict, files_metadata: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_safetensors.py#L74)

**参数：**

元数据（`dict`，*可选*）：“model.safetensors.index.json”文件中包含的元数据（如果存在）。仅填充分片模型。

sharded (`bool`) ：存储库是否包含分片模型。Weight_map (`dict[str, str]`) ：所有权重的映射。键是张量名称，值是包含张量的文件的文件名。

files_metadata (`dict[str, SafetensorsFileMetadata]`) ：所有文件元数据的映射。键是文件名，值是相应文件的元数据，作为 `SafetensorsFileMetadata` 对象。

parameter_count (`dict[str, int]`) ：每种数据类型的参数数量的映射。键是数据类型，值是该数据类型的参数数量。

Safetensors 存储库的元数据。

如果一个存储库包含“model.safetensors”权重文件（非共享），则该存储库被视为 Safetensors 存储库
model）或其根目录下的“model.safetensors.index.json”索引文件（分片模型）。

该类由[get_safetensors_metadata()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.get_safetensors_metadata)返回。

有关 safetensors 格式的更多详细信息，请查看 https://huggingface.co/docs/safetensors/index#format。

### SafetensorsFileMetadata[[huggingface_hub.utils.SafetensorsFileMetadata]]

#### Huggingface_hub.utils.SafetensorsFileMetadata[[huggingface_hub.utils.SafetensorsFileMetadata]]

```python
huggingface_hub.utils.SafetensorsFileMetadata(metadata: dict, tensors: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_safetensors.py#L44)

**参数：**

元数据 (`dict`) ：文件中包含的元数据。张量 (`dict[str, TensorInfo]`) ：所有张量的映射。键是张量名称，值是有关相应张量的信息，作为 `TensorInfo` 对象。

parameter_count (`dict[str, int]`) ：每种数据类型的参数数量的映射。键是数据类型，值是该数据类型的参数数量。

Hub 上托管的 Safetensors 文件的元数据。

该类由[parse_safetensors_file_metadata()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.parse_safetensors_file_metadata)返回。

有关 safetensors 格式的更多详细信息，请查看 https://huggingface.co/docs/safetensors/index#format。

### SpaceInfo[[huggingface_hub.SpaceInfo]]

#### Huggingface_hub.SpaceInfo[[huggingface_hub.SpaceInfo]]

```python
huggingface_hub.SpaceInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1257)

**参数：**

id (`str`) : 空间的ID。

作者（`str`，*可选*）：空间的作者。

card_data (`SpaceCardData`, *可选*) ：空间卡元数据作为 [huggingface_hub.repocard_data.SpaceCardData](/docs/huggingface_hub/v1.27.0/en/package_reference/cards#huggingface_hub.SpaceCardData) 对象。

created_at（`datetime`，*可选*）：在 Hub 上创建存储库的日期。请注意，最小值是`2022-03-02T23:29:04.000Z`，对应于我们开始存储创建日期的日期。

数据集（`list[str]`，*可选*）：空间使用的数据集列表。

禁用（`bool`，*可选*）：空间是否已禁用。gate (`Literal["auto", "manual", False]`, *可选*) ：存储库是否有门控。如果是，是否有手动或自动批准。

host (`str`, *可选*) : 空间的主机 URL。

last_modified (`datetime`, *可选*) ：上次提交到存储库的日期。

喜欢 (`int`) : 空间的喜欢数量。

models（`list[str]`，*可选*）：空间使用的模型列表。

private (`bool`) : 仓库是私有的。

区域（`Literal["us", "eu"]`，*可选*）：存储空间的云区域。

resource_group（`dict`，*可选*）：空间的资源组信息。

运行时（`SpaceRuntime`，*可选*）：空间运行时信息作为[huggingface_hub.hf_api.SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)对象。

sdk（`str`，*可选*）：空间使用的SDK。

sha（`str`，*可选*）：此特定修订版的 Repo SHA。

兄弟姐妹 (`list[RepoSibling]`) ：构成 Space 的 [huggingface_hub.hf_api.RepoSibling](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.hf_api.RepoSibling) 对象列表。

子域（`str`，*可选*）：空间的子域。

Tags (`list[str]`) ：空间的标签列表。

trending_score (`int`, *可选*) : 空间的趋势分数。

used_storage (`int`, *可选*) ：集线器上空间的大小（以字节为单位）。

包含有关 Hub 上空间的信息。该对象由[space_info()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.space_info)和[list_spaces()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces)返回。> [!提示]
> 该类的大多数属性都是可选的。这是因为 Hub 返回的数据取决于所做的查询。
> 一般来说，查询越具体，返回的信息就越多。相反，当列出空间时
> 使用 [list_spaces()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces) 仅返回属性的子集。

### SpaceSearchResult[[huggingface_hub.SpaceSearchResult]]

#### Huggingface_hub.SpaceSearchResult[[huggingface_hub.SpaceSearchResult]]

```python
huggingface_hub.SpaceSearchResult(data: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L302)

**参数：**

id (`str`) ：空间 ID（例如 `"username/repo-name"`）。

作者（`str`）：空间的作者。

title (`str`) : 显示空间的标题。

表情符号（`str`或`None`）：空间的表情符号图标。

sdk（`str`或`None`）：空间使用的SDK（例如`"gradio"`、`"docker"`、`"static"`）。

喜欢 (`int`) ：喜欢的数量。

private (`bool`) : 空间是否私有。

标签（`list[str]`或`None`）：标签列表。

运行时（[SpaceRuntime](/docs/huggingface_hub/v1.27.0/en/package_reference/space_runtime#huggingface_hub.SpaceRuntime)或`None`）：运行时信息（阶段、硬件等）。

ai_short_description（`str`或`None`）：AI生成的简短描述。

ai_category（`str`或`None`）：AI生成的类别（例如`"Image Generation"`）。语义相关性得分（`float`或`None`）：相对于搜索查询的语义相关性得分（0-1）。

trending_score（`int`或`None`）：趋势分数。

来自 Spaces 语义搜索 API 的单个结果。

由[HfApi.search_spaces()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.search_spaces)返回。

### TensorInfo[[huggingface_hub.utils.TensorInfo]]

#### Huggingface_hub.utils.TensorInfo[[huggingface_hub.utils.TensorInfo]]

```python
huggingface_hub.utils.TensorInfo(dtype: typing.Literal['F64', 'F32', 'F16', 'BF16', 'I64', 'I32', 'I16', 'I8', 'U8', 'BOOL'], shape: list, data_offsets: tuple)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_safetensors.py#L14)

**参数：**

dtype (`str`) ：张量的数据类型（“F64”、“F32”、“F16”、“BF16”、“I64”、“I32”、“I16”、“I8”、“U8”、“BOOL”）。

shape (`list[int]`) ：张量的形状。

data_offsets (`tuple[int, int]`) ：文件中数据的偏移量作为元组`[BEGIN, END]`。

parameter_count (`int`) ：张量中参数的数量。

有关张量的信息。

有关 safetensors 格式的更多详细信息，请查看 https://huggingface.co/docs/safetensors/index#format。

### 用户[[huggingface_hub.User]]

#### Huggingface_hub.User[[huggingface_hub.User]]

```python
huggingface_hub.User(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1768)

**参数：**

用户名 (`str`) ：集线器上用户的名称（唯一）。

fullname (`str`) ：用户的全名。

avatar_url (`str`) : 用户头像的 URL。

详细信息（`str`，*可选*）：用户的详细信息。is_following (`bool`, *可选*) ：经过身份验证的用户是否关注该用户。

is_pro (`bool`, *可选*) : 用户是否是专业用户。

num_models (`int`, *可选*) ：用户创建的模型数量。

num_datasets (`int`, *可选*) ：用户创建的数据集数量。

num_spaces (`int`, *可选*) : 用户创建的空间数量。

num_discussions (`int`, *可选*) ：用户发起的讨论数量。

num_papers (`int`, *可选*) ：用户撰写的论文数量。

num_upvotes (`int`, *可选*) : 用户收到的赞成票数。

num_likes (`int`, *可选*) ：用户点赞的数量。

num_following (`int`, *可选*) : 该用户关注的用户数量。

num_followers (`int`, *可选*) : 关注该用户的用户数量。

orgs（`Organization`列表）：用户所属组织的列表。

包含有关 Hub 上用户的信息。

### UserLikes[[huggingface_hub.UserLikes]]

#### Huggingface_hub.UserLikes[[huggingface_hub.UserLikes]]

```python
huggingface_hub.UserLikes(user: str, total: int, datasets: list[str], kernels: list[str], models: list[str], spaces: list[str])
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L1632)

**参数：**

user (`str`) ：我们为其获取点赞的用户的名称。总计 (`int`) : 点赞总数。

datasets (`list[str]`) ：用户喜欢的数据集列表（如repo_ids）。

kernels (`list[str]`) ：用户喜欢的内核列表（如 repo_ids）。

models (`list[str]`) ：用户喜欢的模型列表（如repo_ids）。

space (`list[str]`) ：用户喜欢的空间列表（如 repo_ids）。

包含有关用户在 Hub 上点赞的信息。

### WebhookInfo[[huggingface_hub.WebhookInfo]]

#### Huggingface_hub.WebhookInfo[[huggingface_hub.WebhookInfo]]

```python
huggingface_hub.WebhookInfo(id: str, url: str | None, job: JobSpec | None, watched: list[WebhookWatchedItem], domains: list[constants.WEBHOOK_DOMAIN_T], secret: str | None, disabled: bool)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L618)

**参数：**

id (`str`) ：webhook 的 ID。

url (`str`, *可选*) : webhook 的 URL。

job (`JobSpec`，*可选*)：要触发的作业的规范。

Watched (`list[WebhookWatchedItem]`) ：Webhook 监视的项目列表，请参阅 [WebhookWatchedItem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.WebhookWatchedItem)。

域 (`list[WEBHOOK_DOMAIN_T]`) ：webhook 正在监视的域列表。可以是 `["repo", "discussions"]` 之一。

Secret (`str`，*可选*)：webhook 的秘密。

禁用 (`bool`) : webhook 是否禁用。

包含有关 Webhook 的信息的数据结构。

指定了 `url` 或 `job` 之一，但不能同时指定两者。

### WebhookWatchedItem[[huggingface_hub.WebhookWatchedItem]]

#### Huggingface_hub.WebhookWatchedItem[[huggingface_hub.WebhookWatchedItem]]```python
huggingface_hub.WebhookWatchedItem(type: Literal['dataset', 'model', 'org', 'space', 'user'], name: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/hf_api.py#L603)

**参数：**

type (`Literal["dataset", "model", "org", "space", "user"]`) ：要观看的项目的类型。可以是`["dataset", "model", "org", "space", "user"]`之一。

name (`str`) : 要观看的项目的名称。可以是用户名、组织名称、模型名称、数据集名称或空间名称。

包含有关 Webhook 监视的项目信息的数据结构。

## CommitOperation[[huggingface_hub.CommitOperationAdd]]

以下是 `CommitOperation()` 支持的值：

#### Huggingface_hub.CommitOperationAdd[[huggingface_hub.CommitOperationAdd]]

```python
huggingface_hub.CommitOperationAdd(path_in_repo: str, path_or_fileobj: str | pathlib.Path | bytes | typing.BinaryIO)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_api.py#L145)

**参数：**

path_in_repo (`str`) ：存储库中的相对文件路径，例如：`"checkpoints/1fec34a/weights.bin"`

path_or_fileobj (`str`、`Path`、`bytes` 或 `BinaryIO`) ：任一： - 要上传的本地文件的路径（如 `str` 或 `pathlib.Path`） - 字节缓冲区(`bytes`) 保存要上传的文件的内容 - 一个“文件对象”（`io.BufferedIOBase`的子类），通常通过`open(path, "rb")`获得。它必须支持 `seek()` 和 `tell()` 方法。

**加薪：** ``ValueError``- [⟦T3318⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果`path_or_fileobj`不是`str`、`Path`、`bytes`或`io.BufferedIOBase`之一。
- [⟦T3324⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果 `path_or_fileobj` 是 `str` 或 `Path` 但不是现有文件的路径。
- [⟦T3328⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果 `path_or_fileobj` 是 `io.BufferedIOBase` 但它不支持两者
  `seek()` 和 `tell()`。

数据结构保存将文件上传到集线器上的存储库所需的信息。

#### as_file[[huggingface_hub.CommitOperationAdd.as_file]]

```python
as_file(with_tqdm: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_api.py#L227)

**参数：**

with_tqdm（`bool`，*可选*，默认为`False`）：如果为 True，则迭代文件对象将显示进度条。仅当类文件对象是文件路径时才有效。不支持纯字节和缓冲区。

一个上下文管理器，生成一个类似文件的对象，允许读取底层
`path_or_fileobj`背后的数据。

示例：

```python
>>> operation = CommitOperationAdd(
...        path_in_repo="remote/dir/weights.h5",
...        path_or_fileobj="./local/weights.h5",
... )
CommitOperationAdd(path_in_repo='remote/dir/weights.h5', path_or_fileobj='./local/weights.h5')

>>> with operation.as_file() as file:
...     content = file.read()

>>> with operation.as_file(with_tqdm=True) as file:
...     while True:
...         data = file.read(1024)
...         if not data:
...              break
config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]

>>> with operation.as_file(with_tqdm=True) as file:
...     httpx.put(..., data=file)
config.json: 100%|█████████████████████████| 8.19k/8.19k [00:02<00:00, 3.72kB/s]
```

#### b64content[[huggingface_hub.CommitOperationAdd.b64content]]

```python
b64content()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_api.py#L277)

`path_or_fileobj`的base64编码内容

返回：`bytes`

#### Huggingface_hub.CommitOperationDelete[[huggingface_hub.CommitOperationDelete]]

```python
huggingface_hub.CommitOperationDelete(path_in_repo: str, is_folder: typing.Union[bool, typing.Literal['auto']] = 'auto')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_api.py#L60)

**参数：**

path_in_repo (`str`) ：存储库中的相对文件路径，例如：`"checkpoints/1fec34a/weights.bin"`表示文件或`"checkpoints/1fec34a/"`表示文件夹。is_folder（`bool`或`Literal["auto"]`，*可选*）：删除操作是否适用于文件夹。如果为“auto”，则通过查看路径是否以“/”（文件夹）或不以“/”（文件）结尾来自动猜测路径类型（文件或文件夹）。要显式设置路径类型，可以设置`is_folder=True`或`is_folder=False`。

包含从存储库中删除文件或文件夹所需信息的数据结构
在集线器上。

#### Huggingface_hub.CommitOperationCopy[[huggingface_hub.CommitOperationCopy]]

```python
huggingface_hub.CommitOperationCopy(src_path_in_repo: str, path_in_repo: str, src_revision: str | None = None, src_repo_id: str | None = None, src_repo_type: str | None = None, _src_oid: str | None = None, _dest_oid: str | None = None, _is_duplicated: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_api.py#L91)

**参数：**

src_path_in_repo (`str`) ：要复制的文件的存储库中的相对文件路径，例如`"checkpoints/1fec34a/weights.bin"`。

path_in_repo (`str`) ：存储库中复制文件的相对文件路径，例如`"checkpoints/1fec34a/weights_copy.bin"`。

src_revision (`str`, *可选*) ：要复制的文件的 git 版本。可以是任何有效的 git 修订版。默认为目标提交修订版。

src_repo_id (`str`, *可选*) ：要从中复制的源存储库（例如`"username/source-model"`）。默认为目标存储库（存储库内副本）。

src_repo_type (`str`, *可选*) ：源存储库的类型（`"model"`、`"dataset"` 或 `"space"`）。当`src_repo_id`设置时需要。数据结构保存复制集线器存储库中的文件所需的信息。

LFS 文件和常规文件均受支持。 LFS 文件在服务器端复制，而常规文件则在服务器端复制
作为提交的一部分下载并重新上传。

通过设置 `src_repo_id` 和 `src_repo_type` 支持跨存储库副本。对于跨存储库 LFS 副本，
在创建提交之前，LFS 对象将被复制到目标存储库。这是处理的
由 [create_commit()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit) 自动生成。请注意，跨存储库副本仅在同一存储库中有效
[storage region](https://huggingface.co/docs/hub/storage-regions)；不支持跨区域复制。

注意：您可以组合使用[CommitOperationCopy](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy)和[CommitOperationDelete](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitOperationDelete)来重命名Hub上的LFS文件。

## CommitScheduler[[huggingface_hub.CommitScheduler]]

#### Huggingface_hub.CommitScheduler[[huggingface_hub.CommitScheduler]]

```python
huggingface_hub.CommitScheduler(repo_id: str, folder_path: str | pathlib.Path, every: int | float = 5, path_in_repo: str | None = None, repo_type: str | None = None, revision: str | None = None, private: bool | None = None, token: str | None = None, allow_patterns: list[str] | str | None = None, ignore_patterns: list[str] | str | None = None, squash_history: bool = False, hf_api: typing.Optional[ForwardRef('HfApi')] = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_scheduler.py#L29)

**参数：**

repo_id (`str`) ：要提交的存储库的 id。

folder_path（`str`或`Path`）：定期上传的本地文件夹路径。

every (`int` 或 `float`, *可选*) ：每次提交之间的分钟数。默认为 5 分钟。path_in_repo (`str`, *可选*) ：存储库中目录的相对路径，例如：`"checkpoints/"`。默认为存储库的根文件夹。

repo_type (`str`, *可选*) ：要提交的存储库的类型。默认为`model`。

revision (`str`, *可选*) ：要提交的存储库的修订版本。默认为`main`。

private (`bool`, *可选*) ：是否将存储库设为私有。如果`None`（默认），则存储库将是公开的，除非组织默认为私有。如果存储库已存在，则忽略此值。

token (`str`, *可选*) ：用于提交到存储库的令牌。默认为保存在机器上的令牌。

allowed_pa​​tterns（`list[str]`或`str`，*可选*）：如果提供，则仅上传至少匹配一种模式的文件。

ignore_patterns（`list[str]` 或 `str`，*可选*）：如果提供，则不会上传与任何模式匹配的文件。

squash_history (`bool`, *可选*) ：是否在每次提交后压缩存储库的历史记录。默认为`False`。压缩提交对于避免存储库变得太大时性能下降很有用。hf_api（`HfApi`，*可选*）：用于提交到 Hub 的 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 客户端。可以使用自定义设置（用户代理、令牌...）进行设置。

调度程序定期将本地文件夹上传到集线器（例如每 5 分钟推送到集线器）。

使用调度程序的推荐方法是将其用作上下文管理器。这确保了调度程序
正确停止并在脚本结束时触发最后一次提交。调度程序也可以手动停止
使用`stop`方法。查看[upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload#scheduled-uploads)
了解有关如何使用它的更多信息。

示例：
```py
>>> from pathlib import Path
>>> from huggingface_hub import CommitScheduler

# Scheduler uploads every 10 minutes
>>> csv_path = Path("watched_folder/data.csv")
>>> CommitScheduler(repo_id="test_scheduler", repo_type="dataset", folder_path=csv_path.parent, every=10)

>>> with csv_path.open("a") as f:
...     f.write("first line")

# Some time later (...)
>>> with csv_path.open("a") as f:
...     f.write("second line")
```

使用上下文管理器的示例：
```py
>>> from pathlib import Path
>>> from huggingface_hub import CommitScheduler

>>> with CommitScheduler(repo_id="test_scheduler", repo_type="dataset", folder_path="watched_folder", every=10) as scheduler:
...     csv_path = Path("watched_folder/data.csv")
...     with csv_path.open("a") as f:
...         f.write("first line")
...     (...)
...     with csv_path.open("a") as f:
...         f.write("second line")

# Scheduler is now stopped and last commit have been triggered
```

####push_to_hub[[huggingface_hub.CommitScheduler.push_to_hub]]

```python
push_to_hub()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_scheduler.py#L204)

将文件夹推送到集线器并返回提交信息。

> [!警告]
> 该方法不应该直接调用。它由调度程序在后台运行，尊重
> 队列机制以避免并发提交。直接调用该方法可能会导致并发
> 问题。`push_to_hub` 的默认行为是采用仅附加文件夹。它列出了文件夹中的所有文件并
仅上传更改的文件。如果未发现任何更改，该方法将返回而不提交任何内容。如果你想要
要更改此行为，您可以继承[CommitScheduler](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler)并重写此方法。这可能很有用
例如，在提交之前将数据压缩到一个文件中。有关更多详细信息和示例，请检查
出我们的[integration guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#scheduled-uploads)。

#### 停止[[huggingface_hub.CommitScheduler.stop]]

```python
stop()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_scheduler.py#L157)

停止调度程序。

停止的调度程序无法重新启动。主要用于测试目的。

#### 触发器[[huggingface_hub.CommitScheduler.trigger]]

```python
trigger()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_commit_scheduler.py#L181)

触发 `push_to_hub` 并返回 future。

该方法每`every`分钟自动调用一次。您也可以手动调用它来触发提交
立即执行，无需等待下一次计划的提交。

### 沙箱
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/sandbox.md