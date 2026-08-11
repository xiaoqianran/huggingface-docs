<!-- huggingface-docs: machine-translated zh-CN from English source -->

# OAuth 和 FastAPI

OAuth 是访问委托的开放标准，通常用于授予应用程序对用户信息的有限访问权限，而无需暴露其凭据。与 FastAPI 结合使用时，您可以构建安全的 API，允许用户使用 Google 或 GitHub 等外部身份提供商登录。
在通常的场景中：
- FastAPI 将定义 API 端点并处理 HTTP 请求。
- OAuth 使用 fastapi.security 等库或 Authlib 等外部工具进行集成。
- 当用户想要登录时，FastAPI 将其重定向到 OAuth 提供商的登录页面。
- 成功登录后，提供商会使用令牌重定向回来。
- FastAPI 验证此令牌并使用它来授权用户或获取用户配置文件数据。

这种方法有助于避免直接处理密码，并将身份管理工作交给受信任的提供商。

# FastAPI 中的 Hugging Face OAuth 集成

该模块提供将 Hugging Face OAuth 集成到 FastAPI 应用程序中的工具。它支持使用 Hugging Face 平台进行用户身份验证，包括本地开发的模拟行为和 Spaces 的真实 OAuth 流程。

## OAuth 概述`attach_huggingface_oauth` 函数向您的 FastAPI 应用程序添加登录、注销和回调端点。在空间中使用时，它会连接到 Hugging Face OAuth 系统。当在本地使用时，它将注入一个模拟用户。点击这里了解更多关于[adding a Sign-In with HF option to your Space](https://huggingface.co/docs/hub/en/spaces-oauth)的信息

### 如何使用它？

```python
from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth
from fastapi import FastAPI, Request

app = FastAPI()
attach_huggingface_oauth(app)

@app.get("/")
def greet_json(request: Request):
    oauth_info = parse_huggingface_oauth(request)
    if oauth_info is None:
        return {"msg": "Not logged in!"}
    return {"msg": f"Hello, {oauth_info.user_info.preferred_username}!"}
```

> [!提示]
> 您可能还对 [a practical example that demonstrates OAuth in action](https://huggingface.co/spaces/Wauplin/fastapi-oauth/blob/main/app.py) 感兴趣。
> 如需更全面的实施，请查看[medoidai/GiveBackGPT](https://huggingface.co/spaces/medoidai/GiveBackGPT) Space，它在全面的应用程序中实施了 HF OAuth。

### Attach_huggingface_oauth[[huggingface_hub.attach_huggingface_oauth]]

#### Huggingface_hub.attach_huggingface_oauth[[huggingface_hub.attach_huggingface_oauth]]

```python
huggingface_hub.attach_huggingface_oauth(app: fastapi.FastAPI, route_prefix: str = '/')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_oauth.py#L124)

将 OAuth 端点添加到 FastAPI 应用程序以启用使用 Hugging Face 的 OAuth 登录。

使用方法：
- 在 FastAPI 应用程序上调用此方法以添加 OAuth 端点。
- 在您的路由处理程序中，调用 `parse_huggingface_oauth(request)` 来检索 OAuth 信息。
- 如果用户登录，则会返回一个包含用户信息的[OAuthInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/oauth#huggingface_hub.OAuthInfo)对象。如果不是，则返回`None`。
- 在您的应用程序中，确保添加指向 `/oauth/huggingface/login` 和 `/oauth/huggingface/logout` 的链接，以便用户登录和退出。

示例：
```py
from huggingface_hub import attach_huggingface_oauth, parse_huggingface_oauth

# Create a FastAPI app
app = FastAPI()

# Add OAuth endpoints to the FastAPI app
attach_huggingface_oauth(app)

# Add a route that greets the user if they are logged in
@app.get("/")
def greet_json(request: Request):
    # Retrieve the OAuth info from the request
    oauth_info = parse_huggingface_oauth(request)  # e.g. OAuthInfo dataclass
    if oauth_info is None:
        return {"msg": "Not logged in!"}
    return {"msg": f"Hello, {oauth_info.user_info.preferred_username}!"}
```

### parse_huggingface_oauth[[huggingface_hub.parse_huggingface_oauth]]#### Huggingface_hub.parse_huggingface_oauth[[huggingface_hub.parse_huggingface_oauth]]

```python
huggingface_hub.parse_huggingface_oauth(request: fastapi.Request)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_oauth.py#L191)

以 [OAuthInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/oauth#huggingface_hub.OAuthInfo) 对象的形式返回登录用户的信息。

为了灵活性和面向未来，这种方法的解析非常宽松，不会引发错误。
缺失字段设置为 `None`，且不发出警告。

如果用户未登录（会话 cookie 中没有信息），则返回`None`。

有关如何使用此方法的示例，请参阅[attach_huggingface_oauth()](/docs/huggingface_hub/v1.27.0/en/package_reference/oauth#huggingface_hub.attach_huggingface_oauth)。

### OAuthOrgInfo[[huggingface_hub.OAuthOrgInfo]]

#### Huggingface_hub.OAuthOrgInfo[[huggingface_hub.OAuthOrgInfo]]

```python
huggingface_hub.OAuthOrgInfo(sub: str, name: str, preferred_username: str, picture: str, plan: str | None = None, can_pay: bool | None = None, role_in_org: str | None = None, security_restrictions: list[typing.Literal['ip', 'token-policy', 'mfa', 'sso']] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_oauth.py#L23)

**参数：**

sub (`str`) ：组织的唯一标识符。 OpenID 连接字段。

name (`str`) ：组织的全名。 OpenID 连接字段。

Preferred_username (`str`) ：组织的用户名。 OpenID 连接字段。

图片 (`str`) ：组织的个人资料图片 URL。 OpenID 连接字段。

计划（`str`，*可选*）：组织的计划（例如“企业”、“团队”）。拥抱脸场。

can_pay (`Optional[bool]`, *可选*) ：组织是否设置了付款方式。拥抱脸场。

role_in_org (`Optional[str]`, *可选*) ：用户在组织中的角色。拥抱脸场。security_restrictions（`Optional[list[Literal["ip", "token-policy", "mfa", "sso"]]]`，*可选*）：用户尚未为此组织完成的安全限制数组。可能的值：“ip”、“令牌策略”、“mfa”、“sso”。拥抱脸场。

有关链接到使用 OAuth 登录的用户的组织的信息。

### OAuthUserInfo[[huggingface_hub.OAuthUserInfo]]

#### Huggingface_hub.OAuthUserInfo[[huggingface_hub.OAuthUserInfo]]

```python
huggingface_hub.OAuthUserInfo(sub: str, name: str, preferred_username: str, email_verified: bool | None, email: str | None, picture: str, profile: str, website: str | None, is_pro: bool, can_pay: bool | None, orgs: list[huggingface_hub._oauth.OAuthOrgInfo] | None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_oauth.py#L57)

**参数：**

sub (`str`) ：用户的唯一标识符，即使在重命名的情况下也是如此。 OpenID 连接字段。

name (`str`) ：用户的全名。 OpenID 连接字段。

Preferred_username (`str`) ：用户的用户名。 OpenID 连接字段。

email_verified (`Optional[bool]`, *可选*) ：指示用户的电子邮件是否经过验证。 OpenID 连接字段。

email（`Optional[str]`，*可选*）：用户的电子邮件地址。 OpenID 连接字段。

图片 (`str`) ：用户的个人资料图片 URL。 OpenID 连接字段。

profile (`str`) ：用户的个人资料 URL。 OpenID 连接字段。

website (`Optional[str]`，*可选*)：用户的网站 URL。 OpenID 连接字段。

is_pro (`bool`) : 用户是否是专业用户。拥抱脸场。can_pay (`Optional[bool]`, *可选*) ：用户是否设置了付款方式。拥抱脸场。

orgs (`Optional[list[OrgInfo]]`，*可选*)：用户所属的组织列表。拥抱脸场。

有关使用 OAuth 登录的用户的信息。

### OAuthInfo[[huggingface_hub.OAuthInfo]]

#### Huggingface_hub.OAuthInfo[[huggingface_hub.OAuthInfo]]

```python
huggingface_hub.OAuthInfo(access_token: str, access_token_expires_at: datetime, user_info: OAuthUserInfo, state: str | None, scope: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_oauth.py#L100)

**参数：**

access_token (`str`) ：访问令牌。

access_token_expires_at (`datetime.datetime`) ：访问令牌的到期日期。

user_info ([OAuthUserInfo](/docs/huggingface_hub/v1.27.0/en/package_reference/oauth#huggingface_hub.OAuthUserInfo)) ：用户信息。

state（`str`，*可选*）：在向 OAuth 提供者发出的原始请求中传递给 OAuth 提供者的状态。

范围 (`str`) ：授予范围。

有关 OAuth 登录的信息。

### 概述
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/overview.md