<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 认证

`huggingface_hub` 库允许用户以编程方式管理 Hub 的身份验证。这包括登录、注销、令牌之间切换以及列出可用令牌。

有关身份验证的更多详细信息，请查看[this section](../quick-start#authentication)。

## 登录[[huggingface_hub.login]]

#### Huggingface_hub.login[[huggingface_hub.login]]

```python
huggingface_hub.login(token: str | None = None, add_to_git_credential: bool = False, skip_if_logged_in: bool = True)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_login.py#L57)

**参数：**

令牌（`str`，*可选*）：从 https://huggingface.co/settings/token 生成的用户访问令牌。

add_to_git_credential (`bool`, 默认为`False`) : 如果`True`，token 将被设置为 git 凭证。如果没有配置 git credential helper，则会向用户显示警告。仅在提供`token`时使用；被基于浏览器的流程忽略。

Skip_if_logged_in (`bool`，默认为`True`)：如果`True`，如果用户已经登录，则不提示输入令牌。设置为`False`强制重新登录。在 CLI 中，请改用 `--force`。

**提高：** ``ValueError`` or `DeviceCodeError`

- [⟦T19⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果传递了组织令牌。仅个人账户令牌有效
  登录。
- [⟦T20⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果令牌无效。
- `DeviceCodeError` -- 
  如果基于浏览器的登录失败（授权被拒绝、代码过期……）。登录机器以访问 Hub。

`token` 保存在缓存中并设置为 git 凭证。完成后，机器
已登录，访问令牌将在所有 `huggingface_hub` 中可用
组件。如果未提供 `token`，则使用基于浏览器的 OAuth 流程
验证：打开 URL，输入短代码，然后检索并保存令牌。
在终端中，您还可以选择粘贴现有的访问令牌。

要从脚本外部登录，还可以使用 `hf auth login`，即
包装 [login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login) 的 cli 命令。

> [!提示]
> 当token不传递时，[login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login)会自动检测脚本是否运行
> 是否在笔记本中。然而，这种检测可能不准确，因为
> 当今存在的各种笔记本电脑。如果是这种情况，您可以随时强制
> 使用 [notebook_login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.notebook_login) 或 [interpreter_login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.interpreter_login) 的 UI。

##terpreter_login[[huggingface_hub.interpreter_login]]

#### Huggingface_hub.interpreter_login[[huggingface_hub.interpreter_login]]

```python
huggingface_hub.interpreter_login(skip_if_logged_in: bool = True)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_login.py#L284)

**参数：**

Skip_if_logged_in (`bool`，默认为`True`)：如果`True`，如果用户已经登录，则不提示输入令牌。设置为`False`强制重新登录。在 CLI 中，请改用 `--force`。显示登录 HF 网站并存储令牌的提示。

这相当于不在笔记本中运行时不传递令牌的[login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login)。
如果你想强制使用终端提示符，[interpreter_login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.interpreter_login)很有用
而不是笔记本流程。

欲了解更多详情，请参阅[login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login)。

## 笔记本登录[[huggingface_hub.notebook_login]]

#### Huggingface_hub.notebook_login[[huggingface_hub.notebook_login]]

```python
huggingface_hub.notebook_login(skip_if_logged_in: bool = True)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_login.py#L335)

**参数：**

Skip_if_logged_in (`bool`，默认为`True`)：如果`True`，如果用户已经登录，则不提示输入令牌。设置为`False`强制重新登录。在 CLI 中，请改用 `--force`。

显示登录 HF 网站并存储令牌的提示。

这相当于在笔记本中运行时不传递令牌的[login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login)。
如果你想强制使用笔记本流程，[notebook_login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.notebook_login)很有用
而不是终端中的提示。

欲了解更多详情，请参阅[login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login)。

## 注销[[huggingface_hub.logout]]

#### Huggingface_hub.logout[[huggingface_hub.logout]]

```python
huggingface_hub.logout(token_name: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_login.py#L120)

**参数：**

token_name (`str`, *可选*) ：要注销的访问令牌的名称。如果`None`，将从所有保存的访问令牌中注销。

**加薪：** ``ValueError``- [⟦T39⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果未找到访问令牌名称。

从集线器注销计算机。

令牌将从计算机中删除并从 git 凭证中删除。

## auth_switch[[huggingface_hub.auth_switch]]

#### Huggingface_hub.auth_switch[[huggingface_hub.auth_switch]]

```python
huggingface_hub.auth_switch(token_name: str, add_to_git_credential: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_login.py#L162)

**参数：**

token_name (`str`) ：要切换到的访问令牌的名称。

add_to_git_credential (`bool`, 默认为`False`) : 如果`True`，token 将被设置为 git 凭证。如果没有配置 git credential helper，则会向用户显示警告。如果 `token` 是 `None`，则`add_to_git_credential` 的值将被忽略，并将再次提示给最终用户。

**加薪：** ``ValueError``

- [⟦T48⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果未找到访问令牌名称。

切换到不同的访问令牌。

## auth_list[[huggingface_hub.auth_list]]

#### Huggingface_hub.auth_list[[huggingface_hub.auth_list]]

```python
huggingface_hub.auth_list()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/_login.py#L191)

列出所有存储的访问令牌。

### 概述
https://huggingface.co/docs/huggingface_hub/v1.30.0/package_reference/overview.md