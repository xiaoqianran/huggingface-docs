<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 实用程序

## 配置日志记录[[huggingface_hub.utils.logging.get_verbosity]]

`huggingface_hub` 包公开了一个 `logging` 实用程序来控制包本身的日志记录级别。
您可以这样导入它：

```py
from huggingface_hub import logging
```

然后，您可以定义详细程度以更新您将看到的日志量：

```python
from huggingface_hub import logging

logging.set_verbosity_error()
logging.set_verbosity_warning()
logging.set_verbosity_info()
logging.set_verbosity_debug()

logging.set_verbosity(...)
```

各级别应理解如下：

- `error`：仅显示有关使用情况的关键日志，这可能会导致错误或意外行为。
- `warning`：显示不重要但使用可能会导致意外行为的日志。
  此外，可能会显示重要的信息日志。
- `info`：显示大多数日志，包括一些有关幕后发生的情况的详细日志记录。
  如果某些行为出现异常，我们建议将详细级别切换到此顺序
  以获得更多信息。
- `debug`：显示所有日志，包括一些可用于准确跟踪正在发生的情况的内部日志
  在引擎盖下。

#### Huggingface_hub.utils.logging.get_verbosity[[huggingface_hub.utils.logging.get_verbosity]]

```python
huggingface_hub.utils.logging.get_verbosity()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L103)

**退货：**

日志记录级别，例如 `huggingface_hub.logging.DEBUG` 和
`huggingface_hub.logging.INFO`。

返回 HuggingFace Hub 根记录器的当前级别。> [!提示]
> HuggingFace Hub 具有以下日志记录级别：
>
> - `huggingface_hub.logging.CRITICAL`、`huggingface_hub.logging.FATAL`
> - `huggingface_hub.logging.ERROR`
> - `huggingface_hub.logging.WARNING`、`huggingface_hub.logging.WARN`
> - `huggingface_hub.logging.INFO`
> - `huggingface_hub.logging.DEBUG`

#### Huggingface_hub.utils.logging.set_verbosity[[huggingface_hub.utils.logging.set_verbosity]]

```python
huggingface_hub.utils.logging.set_verbosity(verbosity: int)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L122)

**参数：**

详细程度 (`int`) ：日志记录级别，例如 `huggingface_hub.logging.DEBUG` 和 `huggingface_hub.logging.INFO`。

设置 HuggingFace Hub 根记录器的级别。

#### Huggingface_hub.utils.logging.set_verbosity_info[[huggingface_hub.utils.logging.set_verbosity_info]]

```python
huggingface_hub.utils.logging.set_verbosity_info()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L134)

将详细程度设置为 `logging.INFO`。

#### Huggingface_hub.utils.logging.set_verbosity_debug[[huggingface_hub.utils.logging.set_verbosity_debug]]

```python
huggingface_hub.utils.logging.set_verbosity_debug()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L148)

将详细程度设置为 `logging.DEBUG`。

#### Huggingface_hub.utils.logging.set_verbosity_warning[[huggingface_hub.utils.logging.set_verbosity_warning]]

```python
huggingface_hub.utils.logging.set_verbosity_warning()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L141)

将详细程度设置为 `logging.WARNING`。

#### Huggingface_hub.utils.logging.set_verbosity_error[[huggingface_hub.utils.logging.set_verbosity_error]]

```python
huggingface_hub.utils.logging.set_verbosity_error()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L155)

将详细程度设置为 `logging.ERROR`。

#### Huggingface_hub.utils.logging.disable_propagation[[huggingface_hub.utils.logging.disable_propagation]]

```python
huggingface_hub.utils.logging.disable_propagation()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L162)禁用库日志输出的传播。请注意，日志传播是
默认禁用。

#### Huggingface_hub.utils.logging.enable_propagation[[huggingface_hub.utils.logging.enable_propagation]]

```python
huggingface_hub.utils.logging.enable_propagation()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L170)

启用库日志输出的传播。请禁用
HuggingFace Hub 的默认处理程序可防止根用户重复记录
记录器已配置。

### 特定于存储库的帮助器方法[[huggingface_hub.utils.logging.get_logger]]

下面公开的方法与修改 `huggingface_hub` 库本身的模块相关。
如果您使用 `huggingface_hub` 并且不修改它们，则不需要使用这些。

#### Huggingface_hub.utils.logging.get_logger[[huggingface_hub.utils.logging.get_logger]]

```python
huggingface_hub.utils.logging.get_logger(name: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/logging.py#L78)

**参数：**

name (`str`, *可选*) : 要获取的记录器的名称，通常是文件名

返回具有指定名称的记录器。这个功能不应该
供图书馆用户直接访问。

示例：

```python
>>> from huggingface_hub import get_logger

>>> logger = get_logger(__file__)
>>> logger.set_verbosity_info()
```

## 配置进度条进度条是一个有用的工具，可以在执行长时间运行的任务时向用户显示信息（例如
下载或上传文件时）。 `huggingface_hub` 公开了 `tqdm` 包装器以在
整个图书馆的一致方式。

默认情况下，进度条处于启用状态。您可以通过设置 `HF_HUB_DISABLE_PROGRESS_BARS` 全局禁用它们
环境变量。您还可以使用 [enable_progress_bars()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.utils.enable_progress_bars) 启用/禁用它们
[disable_progress_bars](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.utils.disable_progress_bars)。如果设置，环境变量优先于帮助程序。

```py
>>> from huggingface_hub import snapshot_download
>>> from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars

>>> # Disable progress bars globally
>>> disable_progress_bars()

>>> # Progress bar will not be shown !
>>> snapshot_download("gpt2")

>>> are_progress_bars_disabled()
True

>>> # Re-enable progress bars globally
>>> enable_progress_bars()
```

### 特定于组的进度条控制

您还可以启用或禁用特定组的进度栏。这使您可以在应用程序或库的不同部分中更精细地管理进度条可见性。当某个组的进度条被禁用时，其下的所有子组也会受到影响，除非明确覆盖。

```py
# Disable progress bars for a specific group
>>> disable_progress_bars("peft.foo")
>>> assert not are_progress_bars_disabled("peft")
>>> assert not are_progress_bars_disabled("peft.something")
>>> assert are_progress_bars_disabled("peft.foo")
>>> assert are_progress_bars_disabled("peft.foo.bar")

# Re-enable progress bars for a subgroup
>>> enable_progress_bars("peft.foo.bar")
>>> assert are_progress_bars_disabled("peft.foo")
>>> assert not are_progress_bars_disabled("peft.foo.bar")

# Use groups with tqdm
# No progress bar for `name="peft.foo"`
>>> for _ in tqdm(range(5), name="peft.foo"):
...     pass

# Progress bar will be shown for `name="peft.foo.bar"`
>>> for _ in tqdm(range(5), name="peft.foo.bar"):
...     pass
100%|███████████████████████████████████████| 5/5 [00:00<00:00, 117817.53it/s]
```

### are_progress_bars_disabled[[huggingface_hub.utils.are_progress_bars_disabled]]

#### Huggingface_hub.utils.are_progress_bars_disabled[[huggingface_hub.utils.are_progress_bars_disabled]]

```python
huggingface_hub.utils.are_progress_bars_disabled(name: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/tqdm.py#L191)

**参数：**

name (`str`, *可选*) ：要检查的组名称；如果没有，则检查全局设置。

**退货：** `bool`如果进度条被禁用则为 True，否则为 False。

检查进度条是否全局禁用或针对特定组禁用。

此函数返回是否对给定组或全局禁用进度条。
它首先检查 `HF_HUB_DISABLE_PROGRESS_BARS` 环境变量，然后检查编程
设置。

###disable_progress_bars[[huggingface_hub.utils.disable_progress_bars]]

#### Huggingface_hub.utils.disable_progress_bars[[huggingface_hub.utils.disable_progress_bars]]

```python
huggingface_hub.utils.disable_progress_bars(name: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/tqdm.py#L109)

**参数：**

name (`str`, *可选*) ：要禁用进度条的组的名称。如果无，则全局禁用进度条。

**加薪：** ``Warning``

- ``Warning`` -- 如果环境变量阻止更改。

全局或指定组禁用进度条。

此函数根据组名称更新进度条的状态。
如果未提供组名称，则所有进度条均被禁用。操作
尊重 `HF_HUB_DISABLE_PROGRESS_BARS` 环境变量的设置。

既可以用作常规调用，也可以用作上下文管理器：
disable_progress_bars() # 禁用直到enable_progress_bars()
withdisable_progress_bars(): # 禁用该块，退出时重新启用
...### enable_progress_bars[[huggingface_hub.utils.enable_progress_bars]]

#### Huggingface_hub.utils.enable_progress_bars[[huggingface_hub.utils.enable_progress_bars]]

```python
huggingface_hub.utils.enable_progress_bars(name: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/tqdm.py#L159)

**参数：**

name (`str`, *可选*) : 启用进度条的组的名称。如果无，则全局启用进度条。

**加薪：** ``Warning``

- ``Warning`` -- 如果环境变量阻止更改。

全局或指定组启用进度条。

此函数将进度条设置为对指定组或全局启用
如果没有指定组。操作须遵守`HF_HUB_DISABLE_PROGRESS_BARS`
环境设置。

## 配置 HTTP 后端[[huggingface_hub.set_client_factory]]

在`huggingface_hub` v0.x 中，HTTP 请求通过`requests` 处理，配置通过`configure_http_backend` 完成。由于我们现在使用`httpx`，配置工作方式有所不同：您必须提供一个不带参数并返回`httpx.Client`的工厂函数。您可以查看[default implementation here](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/utils/_http.py)以查看默认使用哪些参数。在某些设置中，您可能需要控制如何发出 HTTP 请求，例如在代理后面工作时。 `huggingface_hub`库允许您使用[set_client_factory()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.set_client_factory)进行全局配置。配置后，对 Hub 的所有请求都将使用您的自定义设置。由于 `huggingface_hub` 在底层依赖于 `httpx.Client`，因此您可以检查 [⟦T98⟧ documentation](https://www.python-httpx.org/advanced/clients/) 以获取有关可用参数的详细信息。

如果您正在构建第三方库并需要直接向 Hub 发出请求，请使用 [get_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.get_session) 获取正确配置的 `httpx` 客户端。将任何直接 `httpx.get(...)` 调用替换为 `get_session().get(...)` 以确保正确的行为。

#### Huggingface_hub.set_client_factory[[huggingface_hub.set_client_factory]]

```python
huggingface_hub.set_client_factory(client_factory: Callable)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_http.py#L335)

设置 `huggingface_hub` 使用的 HTTP 客户端工厂。

客户端工厂是一个返回 `httpx.Client` 对象的方法。第一次调用 [get_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.get_session) 客户端工厂时
将用于创建一个新的 `httpx.Client` 对象，该对象将在 `huggingface_hub` 进行的所有调用之间共享。

如果您在需要自定义配置（例如自定义代理或认证）的特定环境中运行脚本，这可能会很有用。

使用 [get_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.get_session) 获得正确配置的 `httpx.Client`。

#### Huggingface_hub.get_session[[huggingface_hub.get_session]]

```python
huggingface_hub.get_session()
```[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_http.py#L368)

使用用户的传输工厂获取 `httpx.Client` 对象。

该客户端在 `huggingface_hub` 发起的所有调用之间共享。因此您不应该手动关闭它。

使用[set_client_factory()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.set_client_factory)自定义`httpx.Client`。

在极少数情况下，您可能需要手动关闭当前会话（例如，在短暂的`SSLError`之后）。您可以使用 [close_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.close_session) 来做到这一点。下次调用[get_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.get_session)时将自动创建一个新会话。

当进程退出时，会话总是自动关闭。

#### Huggingface_hub.close_session[[huggingface_hub.close_session]]

```python
huggingface_hub.close_session()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_http.py#L396)

关闭`huggingface_hub`使用的全局`httpx.Client`。

如果客户端关闭，它将在下次调用[get_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.get_session)时重新创建。

可能有用，例如SSL 证书已更新。

对于异步代码，使用 [set_async_client_factory()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.set_async_client_factory) 配置 `httpx.AsyncClient` 并使用 [get_async_session()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.get_async_session) 检索一个。

#### Huggingface_hub.set_async_client_factory[[huggingface_hub.set_async_client_factory]]

```python
huggingface_hub.set_async_client_factory(async_client_factory: Callable)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_http.py#L352)

设置 `huggingface_hub` 使用的 HTTP 异步客户端工厂。异步客户端工厂是一个返回 `httpx.AsyncClient` 对象的方法。
如果您在需要自定义配置（例如自定义代理或认证）的特定环境中运行脚本，这可能会很有用。
使用 `get_async_client` 获取正确配置的 `httpx.AsyncClient`。

> [!警告]
> 与在 `huggingface_hub` 发出的所有呼叫之间共享的 `httpx.Client` 相反，`httpx.AsyncClient` 不共享。
> 建议使用异步上下文管理器来确保退出上下文时正确关闭客户端。

#### Huggingface_hub.get_async_session[[huggingface_hub.get_async_session]]

```python
huggingface_hub.get_async_session()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_http.py#L383)

使用用户的传输工厂返回一个 `httpx.AsyncClient` 对象。

使用[set_async_client_factory()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.set_async_client_factory)自定义`httpx.AsyncClient`。

> [!警告]
> 与在 `huggingface_hub` 发出的所有调用之间共享的 `httpx.Client` 相反，`httpx.AsyncClient` 不共享。
> 建议使用异步上下文管理器来确保退出上下文时正确关闭客户端。

与同步客户端不同，异步客户端的生命周期不会自动管理。使用异步上下文管理器来正确处理它。

## 处理HTTP错误`huggingface_hub` 定义了自己的 HTTP 错误，以细化由 引发的 `HTTPError`
`httpx` 以及服务器发回的附加信息。

### 提高状态[[huggingface_hub.hf_raise_for_status]]

[hf_raise_for_status()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.hf_raise_for_status) 旨在成为从任何人“提升地位”的核心方法
向中心提出请求。它包裹底座`httpx.Response.raise_for_status`以提供
附加信息。任何抛出的 `HTTPError` 都会转换为 `HfHubHTTPError`。

```py
from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

response = get_session().post(...)
try:
    hf_raise_for_status(response)
except HfHubHTTPError as e:
    print(str(e)) # formatted message
    e.request_id, e.server_message # details returned by server

    # Complete the error message with additional information once it's raised
    e.append_to_message("\n`create_commit` expects the repository to exist.")
    raise
```

#### Huggingface_hub.hf_raise_for_status[[huggingface_hub.hf_raise_for_status]]

```python
huggingface_hub.hf_raise_for_status(response: Response, endpoint_name: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_http.py#L768)

**参数：**

响应 (`Response`) ：来自服务器的响应。

endpoint_name (`str`, *可选*) ：已调用的端点的名称。如果提供，错误消息将更加完整。

`response.raise_for_status()` 的内部版本将改进潜在的 HTTPError。
引发的异常将是 [HfHubHTTPError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError) 的实例。

这个助手是调用 Hugging Face Hub 时 raise_for_status 的唯一方法。> [!警告]
> 请求失败时引发：
>
> - [RepositoryNotFoundError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.RepositoryNotFoundError)
> 如果找不到要下载的存储库。这可能是因为它
> 不存在，因为 `repo_type` 设置不正确，或者因为 repo
> 是 `private`，您无权访问。
> - [GatedRepoError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.GatedRepoError)
> 如果存储库存在但被限制并且用户不在授权范围内
> 列表。
> - [RevisionNotFoundError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.RevisionNotFoundError)
> 如果存储库存在但找不到修订版本。
> - [EntryNotFoundError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.EntryNotFoundError)
> 如果存储库存在但无法获取条目（例如请求的文件）
> 找到。
> - [BadRequestError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.BadRequestError)
> 如果请求失败并出现 HTTP 400 BadRequest 错误。
> - [HfHubHTTPError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.HfHubHTTPError)
> 如果请求因上面未列出的原因而失败。

### 检查离线模式[[huggingface_hub.is_offline_mode]]

您可以使用`is_offline_mode`以编程方式检查是否启用了离线模式。通过将 `HF_HUB_OFFLINE=1` 设置为环境变量来启用离线模式。

#### Huggingface_hub.is_offline_mode[[huggingface_hub.is_offline_mode]]

```python
huggingface_hub.is_offline_mode()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/constants.py#L195)

返回 Hub 是否处于离线模式。

启用离线模式后，所有使用 `get_session` 发出的 HTTP 请求都会引发 `OfflineModeIsEnabled` 异常。示例：
```py
from huggingface_hub import is_offline_mode

def list_files(repo_id: str):
    if is_offline_mode():
        ... # list files from local cache (degraded experience but still functional)
    else:
        ... # list files from Hub (complete experience)
```

### HTTP 错误

以下是 `huggingface_hub` 中抛出的 HTTP 错误列表。

#### HfHubHTTPError[[huggingface_hub.errors.HfHubHTTPError]]

`HfHubHTTPError` 是任何 HF Hub HTTP 错误的父类。它负责解析
服务器响应并格式化错误消息以向服务器提供尽可能多的信息
用户尽可能。

#### Huggingface_hub.errors.HfHubHTTPError[[huggingface_hub.errors.HfHubHTTPError]]

```python
huggingface_hub.errors.HfHubHTTPError(message: str, response: Response, server_message: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L90)

HF Hub 中引发的任何自定义 HTTP 错误的继承 HTTPError。

任何 HTTPError 至少都会转换为 `HfHubHTTPError`。如果某些信息是
由服务器发回，它会被添加到错误消息中。

添加的详细信息：
- 请求 ID 源自标头，按优先顺序排列：“X-Request-Id”、“X-Amzn-Trace-Id”、“X-Amz-Cf-Id”。
- 来自标头“X-Error-Message”的服务器错误消息。
- 如果我们可以在响应正文中找到服务器错误消息。

示例：
```py
    import httpx
    from huggingface_hub.utils import get_session, hf_raise_for_status, HfHubHTTPError

    response = get_session().post(...)
    try:
        hf_raise_for_status(response)
    except HfHubHTTPError as e:
        print(str(e)) # formatted message
        e.request_id, e.server_message # details returned by server

        # Complete the error message with additional information once it's raised
        e.append_to_message("
ate_commit` expects the repository to exist.")
        raise
```

####append_to_message[[huggingface_hub.errors.HfHubHTTPError.append_to_message]]

```python
append_to_message(additional_message: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L137)

将附加信息附加到`HfHubHTTPError`初始消息。

#### RepositoryNotFoundError[[huggingface_hub.errors.RepositoryNotFoundError]]#### Huggingface_hub.errors.RepositoryNotFoundError[[huggingface_hub.errors.RepositoryNotFoundError]]

```python
huggingface_hub.errors.RepositoryNotFoundError(message: str, response: Response, server_message: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L301)

**参数：**

repo_id (`str` 或 `None`) ：未找到的存储库 ID（如果可以从请求 URL 确定）。

repo_type (`str` 或 `None`) ：存储库类型（“模型”、“数据集”或“空间”）（如果可以从请求 URL 确定）。

尝试使用无效存储库名称访问 hf.co URL 时引发，或者
具有用户无权访问的私有存储库名称。

示例：

```py
>>> from huggingface_hub import model_info
>>> model_info("<non_existent_repository>")
(...)
huggingface_hub.errors.RepositoryNotFoundError: 401 Client Error. (Request ID: PvMw_VjBMjVdMz53WKIzP)

Repository Not Found for url: https://huggingface.co/api/models/%3Cnon_existent_repository%3E.
Please make sure you specified the correct `repo_id` and `repo_type`.
If the repo is private, make sure you are authenticated and your token has the required permissions.
Invalid username or password.
```

#### GatedRepoError[[huggingface_hub.errors.GatedRepoError]]

#### Huggingface_hub.errors.GatedRepoError[[huggingface_hub.errors.GatedRepoError]]

```python
huggingface_hub.errors.GatedRepoError(message: str, response: Response, server_message: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L331)

当尝试访问用户不在其上的封闭存储库时引发
授权名单。

注意：源自`RepositoryNotFoundError`以确保向后兼容性。

示例：

```py
>>> from huggingface_hub import model_info
>>> model_info("<gated_repository>")
(...)
huggingface_hub.errors.GatedRepoError: 403 Client Error. (Request ID: ViT1Bf7O_026LGSQuVqfa)

Cannot access gated repo for url https://huggingface.co/api/models/ardent-figment/gated-model.
Access to model ardent-figment/gated-model is restricted and you are not in the authorized list.
Visit https://huggingface.co/ardent-figment/gated-model to ask for access.
```

#### RevisionNotFoundError[[huggingface_hub.errors.RevisionNotFoundError]]

#### Huggingface_hub.errors.RevisionNotFoundError[[huggingface_hub.errors.RevisionNotFoundError]]

```python
huggingface_hub.errors.RevisionNotFoundError(message: str, response: Response, server_message: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L374)

**参数：**

repo_id (`str` 或 `None`) ：repo id（如果可以从请求 URL 确定）。repo_type (`str` 或 `None`) ：存储库类型（“模型”、“数据集”或“空间”）（如果可以从请求 URL 确定）。

尝试使用有效存储库但无效的 hf.co URL 访问时引发
修订。

示例：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download('bert-base-cased', 'config.json', revision='<non-existent-revision>')
(...)
huggingface_hub.errors.RevisionNotFoundError: 404 Client Error. (Request ID: Mwhe_c3Kt650GcdKEFomX)

Revision Not Found for url: https://huggingface.co/bert-base-cased/resolve/%3Cnon-existent-revision%3E/config.json.
```

#### RevisionResolutionError[[huggingface_hub.errors.RevisionResolutionError]]

#### Huggingface_hub.errors.RevisionResolutionError[[huggingface_hub.errors.RevisionResolutionError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L401)

当修订无法解析为提交哈希时，由[HfApi.resolve_revision()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.resolve_revision)引发：集线器无法
已达到（离线模式、连接错误、超时、集线器停机时间...）并且在
本地缓存。

#### BadRequestError[[huggingface_hub.errors.BadRequestError]]

#### Huggingface_hub.errors.BadRequestError[[huggingface_hub.errors.BadRequestError]]

```python
huggingface_hub.errors.BadRequestError(message: str, response: Response, server_message: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L492)

当服务器返回 HTTP 400 错误时，由 `hf_raise_for_status` 引发。

示例：

```py
>>> resp = httpx.post("hf.co/api/check", ...)
>>> hf_raise_for_status(resp, endpoint_name="check")
huggingface_hub.errors.BadRequestError: Bad request for check endpoint: {details} (Request ID: XXX)
```

#### EntryNotFoundError[[huggingface_hub.errors.EntryNotFoundError]]

#### Huggingface_hub.errors.EntryNotFoundError[[huggingface_hub.errors.EntryNotFoundError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L410)

当本地或远程未找到条目时引发。

示例：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download('bert-base-cased', '<non-existent-file>')
(...)
huggingface_hub.errors.RemoteEntryNotFoundError (...)
>>> hf_hub_download('bert-base-cased', '<non-existent-file>', local_files_only=True)
(...)
huggingface_hub.utils.errors.LocalEntryNotFoundError (...)
```#### RemoteEntryNotFoundError[[huggingface_hub.errors.RemoteEntryNotFoundError]]

#### Huggingface_hub.errors.RemoteEntryNotFoundError[[huggingface_hub.errors.RemoteEntryNotFoundError]]

```python
huggingface_hub.errors.RemoteEntryNotFoundError(message: str, response: Response, server_message: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L428)

**参数：**

repo_id (`str` 或 `None`) ：repo id（如果可以从请求 URL 确定）。

repo_type (`str` 或 `None`) ：存储库类型（“模型”、“数据集”或“空间”）（如果可以从请求 URL 确定）。

尝试使用有效的存储库和修订访问 hf.co URL 时引发
but an invalid filename.

示例：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download('bert-base-cased', '<non-existent-file>')
(...)
huggingface_hub.errors.EntryNotFoundError: 404 Client Error. (Request ID: 53pNl6M0MxsnG5Sw8JA6x)

Entry Not Found for url: https://huggingface.co/bert-base-cased/resolve/main/%3Cnon-existent-file%3E.
```

#### LocalEntryNotFoundError[[huggingface_hub.errors.LocalEntryNotFoundError]]

#### Huggingface_hub.errors.LocalEntryNotFoundError[[huggingface_hub.errors.LocalEntryNotFoundError]]

```python
huggingface_hub.errors.LocalEntryNotFoundError(message: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L455)

当网络连接时尝试访问不在磁盘上的文件或快照时引发
已禁用或不可用（连接问题）。该条目可能存在于集线器上。

示例：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download('bert-base-cased', '<non-cached-file>',  local_files_only=True)
(...)
huggingface_hub.errors.LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.
```

#### IncompleteSnapshotError[[huggingface_hub.errors.IncompleteSnapshotError]]

#### Huggingface_hub.errors.IncompleteSnapshotError[[huggingface_hub.errors.IncompleteSnapshotError]]

```python
huggingface_hub.errors.IncompleteSnapshotError(message: str, snapshot_path: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L474)当无法到达集线器时（离线、连接问题或
`local_files_only=True`）并且已知缓存的快照不完整：中列出的一些文件
本地快照中缺少存储库的缓存树列表。

这是 `LocalEntryNotFoundError` 的子类，用于向后兼容。

`snapshot_path`属性保存了不完整的本地快照的路径，因此下游库可以定位
最新的缓存文件，即使已知它们不完整。

#### CachedRepoTreeNotFoundError[[huggingface_hub.errors.CachedRepoTreeNotFoundError]]

#### Huggingface_hub.errors.CachedRepoTreeNotFoundError[[huggingface_hub.errors.CachedRepoTreeNotFoundError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L26)

当没有为请求的修订缓存树列表时，由 [get_cached_repo_tree()](/docs/huggingface_hub/v1.30.0/en/package_reference/file_download#huggingface_hub.get_cached_repo_tree) 引发。

树列表的填充是 [snapshot_download()](/docs/huggingface_hub/v1.30.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 的副作用。

#### OfflineModeIsEnabled[[huggingface_hub.errors.OfflineModeIsEnabled]]

#### Huggingface_hub.errors.OfflineModeIsEnabled[[huggingface_hub.errors.OfflineModeIsEnabled]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L86)

当发出请求但 `HF_HUB_OFFLINE=1` 设置为环境变量时引发。

## 遥测[[huggingface_hub.utils.send_telemetry]]`huggingface_hub` 包括一个发送遥测数据的助手。这些信息有助于我们调试问题并确定新功能的优先级。
用户可以随时通过设置 `HF_HUB_DISABLE_TELEMETRY=1` 环境变量来禁用遥测收集。
遥测在离线模式下也会被禁用（即设置 HF_HUB_OFFLINE=1 时）。

如果您是第三方库的维护者，发送遥测数据就像调用`send_telemetry`一样简单。
数据在单独的线程中发送，以尽可能减少对用户的影响。

#### Huggingface_hub.utils.send_telemetry[[huggingface_hub.utils.send_telemetry]]

```python
huggingface_hub.utils.send_telemetry(topic: str, library_name: str | None = None, library_version: str | None = None, user_agent: dict | str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_telemetry.py#L19)

**参数：**

topic (`str`) ：监控的主题名称。该主题直接用于构建 URL。如果要监控子主题，只需使用“/”分隔即可。示例：“gradio”、“变压器/示例”、...

library_name (`str`, *可选*) ：发出 HTTP 请求的库的名称。将添加到用户代理标头中。

library_version (`str`, *可选*) ：发出 HTTP 请求的库的版本。将添加到用户代理标头中。user_agent (`str`, `dict`, *可选*) ：字典或单个字符串形式的用户代理信息。它将包含有关已安装软件包的信息来完成。

发送遥测数据，帮助跟踪不同 HF 库的使用情况。

这些使用数据可以帮助我们调试问题并确定新功能的优先级。然而，我们明白并不是每个人都想要
分享更多信息，我们尊重您的隐私。您可以通过设置来禁用遥测收集
`HF_HUB_DISABLE_TELEMETRY=1` 作为环境变量。遥测在离线模式下也会被禁用（即当设置
`HF_HUB_OFFLINE=1`）。

遥测收集在单独的线程中运行，以尽量减少对用户的影响。

示例：
```py
>>> from huggingface_hub.utils import send_telemetry

# Send telemetry without library information
>>> send_telemetry("ping")

# Send telemetry to subtopic with library information
>>> send_telemetry("gradio/local_link", library_name="gradio", library_version="3.22.1")

# Send telemetry with additional data
>>> send_telemetry(
...     topic="examples",
...     library_name="transformers",
...     library_version="4.26.0",
...     user_agent={"pipeline": "text_classification", "framework": "flax"},
... )
```

## 验证器

`huggingface_hub` 包括自定义验证器来自动验证方法参数。
验证的灵感来自于[Pydantic](https://pydantic-docs.helpmanual.io/)中所做的工作
验证类型提示，但功能更有限。

### 通用装饰器

[validate_hf_hub_args()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.utils.validate_hf_hub_args)是一个通用的装饰器来封装
具有遵循 `huggingface_hub` 命名的参数的方法。默认情况下，所有
已实现验证器的参数将被验证。如果输入无效，则会抛出 [HFValidationError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.HFValidationError)。仅
第一个无效值会引发错误并停止验证过程。

用途：

```py
>>> from huggingface_hub.utils import validate_hf_hub_args

>>> @validate_hf_hub_args
... def my_cool_method(repo_id: str):
...     print(repo_id)

>>> my_cool_method(repo_id="valid_repo_id")
valid_repo_id

>>> my_cool_method("other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> my_cool_method(repo_id="other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
```

#### validate_hf_hub_args[[huggingface_hub.utils.validate_hf_hub_args]]

#### Huggingface_hub.utils.validate_hf_hub_args[[huggingface_hub.utils.validate_hf_hub_args]]

```python
huggingface_hub.utils.validate_hf_hub_args(fn: ~CallableT)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_validators.py#L41)

**加薪：** [HFValidationError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.HFValidationError)

- [HFValidationError](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.errors.HFValidationError) -- 
  如果输入无效。

验证作为 `huggingface_hub` 的任何公共方法的参数收到的值。

该装饰器的目标是协调重用参数的验证
无处不在。默认情况下，所有定义的验证器都会被测试。

验证者：
- [validate_repo_id()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.utils.validate_repo_id)：`repo_id` 必须是 `"repo_name"`
  或`"namespace/repo_name"`。命名空间是用户名或组织。
- `~utils.smoothly_deprecate_legacy_arguments`：下载文件时忽略`proxies`（应全局设置）。

示例：
```py
>>> from huggingface_hub.utils import validate_hf_hub_args

>>> @validate_hf_hub_args
... def my_cool_method(repo_id: str):
...     print(repo_id)

>>> my_cool_method(repo_id="valid_repo_id")
valid_repo_id

>>> my_cool_method("other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.

>>> my_cool_method(repo_id="other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
```

#### HFValidationError[[huggingface_hub.errors.HFValidationError]]

#### Huggingface_hub.errors.HFValidationError[[huggingface_hub.errors.HFValidationError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L217)

`huggingface_hub` 验证器抛出的通用异常。

继承于[⟦T182⟧](https://docs.python.org/3/library/exceptions.html#ValueError)。

### 参数验证器

验证器也可以单独使用。这是可以的所有参数的列表
已验证。#### repo_id[[huggingface_hub.utils.validate_repo_id]]

#### Huggingface_hub.utils.validate_repo_id[[huggingface_hub.utils.validate_repo_id]]

```python
huggingface_hub.utils.validate_repo_id(repo_id: str | None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_validators.py#L93)

验证`repo_id`有效。

这并不是要取代在集线器上进行的正确验证，而是为了
尽可能避免局部不一致（例如：在
`repo_id` 被禁止）。

规则：
- 1 到 96 个字符之间。
- “repo_name”或“命名空间/repo_name”
- [a-zA-Z0-9] 或“-”、“_”、“.”
- 禁止使用“--”和“..”

有效：`"foo"`、`"foo/bar"`、`"123"`、`"Foo-BAR_foo.bar123"`

无效：`"datasets/foo/bar"`、`".repo_id"`、`"foo--bar"`、`"foo.git"`

示例：
```py
>>> from huggingface_hub.utils import validate_repo_id
>>> validate_repo_id(repo_id="valid_repo_id")
>>> validate_repo_id(repo_id="other..repo..id")
huggingface_hub.utils._validators.HFValidationError: Cannot have -- or .. in repo_id: 'other..repo..id'.
```

在 https://github.com/huggingface/huggingface_hub/issues/1008 中讨论。
在登月（内部存储库）中：
- https://github.com/huggingface/moon-landing/blob/main/server/lib/Names.ts#L27
- https://github.com/huggingface/moon-landing/blob/main/server/views/components/NewRepoForm/NewRepoForm.svelte#L138

#### smooth_deprecate_legacy_arguments[[huggingface_hub.utils._validators.smoothly_deprecate_legacy_arguments]]

不完全是验证器，但也运行。#### Huggingface_hub.utils._validators.smoothly_deprecate_legacy_arguments[[huggingface_hub.utils._validators.smoothly_deprecate_legacy_arguments]]

```python
huggingface_hub.utils._validators.smoothly_deprecate_legacy_arguments(fn_name: str, kwargs: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/utils/_validators.py#L151)

顺利弃用 `huggingface_hub` 代码库中的遗留参数。

此函数忽略 kwargs 中的一些已弃用的参数，并警告用户它们被忽略。
目标是避免破坏现有代码，同时引导用户采用新的做事方式。

已弃用的参数列表：
- `proxies`：
  要设置代理，用户必须使用 HTTP_PROXY 环境变量或配置 `httpx.Client`
  手动使用[set_client_factory()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.set_client_factory)功能。

  在huggingface_hub 0.x中，`proxies`是直接传递给`requests.request`的字典。
  在huggingface_hub 1.x中，我们迁移到`httpx`，它不以同样的方式支持`proxies`。
  特别是，不可能根据每个请求配置代理。解决方案是配置
  它全局使用 [set_client_factory()](/docs/huggingface_hub/v1.30.0/en/package_reference/utilities#huggingface_hub.set_client_factory) 函数或使用 HTTP_PROXY 环境变量。

  有关更多详细信息，请参阅：
  - https://www.python-httpx.org/advanced/proxies/
  - https://www.python-httpx.org/compatibility/#proxy-keys。- `resume_download`：已弃用且无需替换。 `huggingface_hub` 总是尽可能恢复下载。
- `force_filename`：已弃用且无需替换。文件名始终与集线器上的相同。
- `local_dir_use_symlinks`：已弃用且无需替换。下载到本地目录不再使用符号链接。

### 文件系统 API
https://huggingface.co/docs/huggingface_hub/v1.30.0/package_reference/hf_file_system.md