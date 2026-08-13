<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在您的空间中添加使用 HF 登录按钮

您可以通过无缝创建和关联 [OAuth/OpenID connect](https://developer.okta.com/blog/2019/10/21/illustrated-guide-to-oauth-and-oidc) 应用程序来启用空间中的内置登录流程，以便用户可以使用其 HF 帐户登录。

这将为您的空间带来新的用例。例如，当与[Storage Buckets](https://huggingface.co/docs/hub/storage-buckets)结合使用时，生成式AI空间可以允许用户登录以访问他们的前几代，只有他们可以访问。

> [!提示]
> 本指南将引导您完成将*使用 HF 登录*按钮集成到任何空间的过程。如果您正在寻找一种快速而简单的方法来在 **Gradio** 空间中实现此目的，请查看它的 [built-in integration](https://www.gradio.app/guides/sharing-your-app#o-auth-login-via-hugging-face)。

> [!提示]
> 您还可以使用 HF OAuth 流程在 Spaces 之外的任何网站或应用程序中创建“使用 HF 登录”流程。 [Read our general OAuth page](./oauth)。

## 创建一个 OAuth 应用程序

您需要做的就是将 `hf_oauth: true` 添加到 `README.md` 文件内的空间元数据中。

以下是渐变空间的元数据示例：

```yaml
title: Gradio Oauth Test
emoji: 🏆
colorFrom: pink
colorTo: pink
sdk: gradio
sdk_version: 3.40.0
python_version: 3.10.6
app_file: app.py

hf_oauth: true
# optional, default duration is 8 hours/480 minutes. Max duration is 30 days/43200 minutes.
hf_oauth_expiration_minutes: 480
# optional, see "Scopes" below. "openid profile" is always included.
hf_oauth_scopes:
 - read-repos
 - gated-repos
 - write-repos
 - manage-repos
 - inference-api
# optional, restrict access to members of specific organizations
hf_oauth_authorized_org: ORG_NAME
hf_oauth_authorized_org:
  - ORG_NAME1
  - ORG_NAME2
```

您可以查看[configuration reference docs](./spaces-config-reference)了解更多信息。

这会将以下 [environment variables](https://huggingface.co/docs/hub/spaces-overview#helper-environment-variables) 添加到您的空间：- `OAUTH_CLIENT_ID`：OAuth 应用程序的客户端 ID（公共）
- `OAUTH_CLIENT_SECRET`：OAuth 应用程序的客户端密钥
- `OAUTH_SCOPES`：OAuth 应用程序可访问的范围。
- `OPENID_PROVIDER_URL`：OpenID 提供商的 URL。 OpenID 元数据将在 [⟦T8⟧](https://huggingface.co/.well-known/openid-configuration) 提供。

对于任何其他环境变量，您可以通过使用 `os.getenv("OAUTH_CLIENT_ID")` 在代码中使用它们。

## 重定向 URL

您可以使用任何您想要的重定向 URL，只要它针对您的空间即可。

请注意，`SPACE_HOST` 是 [available](https://huggingface.co/docs/hub/spaces-overview#helper-environment-variables) 作为环境变量。

例如，您可以使用 `https://{SPACE_HOST}/login/callback` 作为重定向 URI。

## 范围

空间始终包含以下范围：

- `openid`：除了访问令牌之外，还接收 ID 令牌。
- `profile`：读取用户的个人资料信息（用户名、头像等）

这些范围是可选的，可以通过在空间的元数据中设置 `hf_oauth_scopes` 来添加：- `email`：读取用户的电子邮件地址。
- `read-billing`：了解用户是否设置了支付方式。
- `read-repos`：阅读用户的个人存储库。
- `gated-repos`：读取用户已被授予访问权限的公共门控存储库的内容。与 `read-repos` 不同，这不会授予对私人存储库的访问权限。
- `contribute-repos`：创建存储库并访问由此应用程序创建的存储库。除非授予额外权限，否则无法访问任何其他存储库。
- `write-repos`：读写用户的个人存储库。
- `manage-repos`：全面管理用户的个人仓库，包括创建和删除它们。
- `read-collections`：阅读用户的个人收藏。
- `write-collections`：读写用户的个人收藏，包括创建和删除它们。
- `inference-api`：代表用户向[Inference Providers](https://huggingface.co/docs/inference-providers/index)提出推理请求。
- `read-endpoints`：查看用户的[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)并代表用户向其发出推理请求。
- `write-endpoints`：管理用户的推理端点，包括创建和删除它们。包括 `read-endpoints` 访问权限。
- `jobs`：运行[jobs](https://huggingface.co/docs/huggingface_hub/main/en/guides/jobs)
- `webhooks`：管理[webhooks](https://huggingface.co/docs/huggingface_hub/main/en/guides/webhooks)- `write-discussions`：代表用户打开讨论和拉取请求，并与讨论互动（包括反应、发布/编辑评论、结束讨论……）。要在私有存储库上打开拉取请求，您还需要请求 `read-repos` 范围。

## 访问组织资源

默认情况下，oauth 应用程序不需要访问组织资源。

但某些范围（例如 `read-repos` 或 `read-billing`）也适用于组织。

用户在授权应用程序时可以选择向哪些组织授予访问权限。如果您需要访问特定组织，可以将 `orgIds=ORG_ID` 作为查询参数添加到 OAuth 授权 URL。您必须将 `ORG_ID` 替换为组织 ID，该 ID 可在 userinfo 响应的 `organizations.sub` 字段中找到。

## 将按钮添加到您的空间

现在，您已掌握向您的空间添加“使用 HF 登录”按钮的所有信息。一些库（[Python](https://github.com/lepture/authlib)、[NodeJS](https://github.com/panva/node-openid-client)）可以帮助您实现 OpenID/OAuth 协议。

Gradio 和 Huggingface.js 还提供**内置支持**，使得使用 HF 按钮实现登录变得轻而易举；您可以通过[gradio](https://www.gradio.app/guides/sharing-your-app#o-auth-login-via-hugging-face)和[huggingface.js](https://huggingface.co/docs/huggingface.js/hub/README#oauth-login)查看相关指南。

基本上，您需要：- 将用户重定向到`https://huggingface.co/oauth/authorize?redirect_uri={REDIRECT_URI}&scope=openid%20profile&client_id={CLIENT_ID}&state={STATE}`，其中`STATE`是一个随机字符串，您稍后需要验证。
- 处理`/auth/callback`或`/login/callback`（或您自己的自定义回调URL）上的回调并验证`state`参数。
- 使用 `code` 查询参数从 `https://huggingface.co/oauth/token` 获取访问令牌和 id 令牌（使用 `client_id`、`code`、`grant_type=authorization_code` 和 `redirect_uri` 作为表单数据，并使用 `Authorization: Basic {base64(client_id:client_secret)}` 作为标头的 POST 请求）。

> [!警告]
> 您应该在按钮上使用 `target=_blank` 在新选项卡中打开登录页面，除非您运行其 `iframe` 之外的空间。否则，您可能会在某些浏览器上遇到 cookie 问题。

## 示例：

- [Gradio test app](https://huggingface.co/spaces/Wauplin/gradio-oauth-test)
- [HuggingChat (NodeJS/SvelteKit)](https://huggingface.co/spaces/huggingchat/chat-ui)
- [Inference Widgets (Auth.js/SvelteKit)](https://huggingface.co/spaces/huggingfacejs/inference-widgets)，使用`inference-api`范围代表用户发出推理请求。
- [Client-Side in a Static Space (huggingface.js)](https://huggingface.co/spaces/huggingfacejs/client-side-oauth) - 非常简单的 JavaScript 示例。

JS代码示例：

```js
import { oauthLoginUrl, oauthHandleRedirectIfPresent } from "@huggingface/hub";

const oauthResult = await oauthHandleRedirectIfPresent();

if (!oauthResult) {
  // If the user is not logged in, redirect to the login page
  window.location.href = await oauthLoginUrl();
}

// You can use oauthResult.accessToken, oauthResult.userInfo among other things
console.log(oauthResult);
```

### 附录
https://huggingface.co/docs/hub/model-card-appendix.md