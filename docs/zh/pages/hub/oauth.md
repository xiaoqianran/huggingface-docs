<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 抱脸登录

您可以使用 HF OAuth / OpenID 连接流程在任何网站或应用程序中创建**“使用 HF 登录”**流程。

这将允许用户通过单击与此类似的按钮，使用其 HF 帐户登录您的网站或应用程序：

![Sign in with Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/sign-in-with-huggingface-xl-dark.svg)

单击此按钮后，您的用户将看到一个权限模式来授权您的应用程序：

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/oauth-accept-application.png)

## 创建一个 oauth 应用程序

您可以在[settings](https://huggingface.co/settings/applications/new)中创建您的应用程序：

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/oauth-create-application.png)

### 公共 OAuth 应用程序（没有秘密）

您可以在没有客户端密钥的情况下创建或使用 OAuth 应用程序。这对于本机应用程序、CLI 或其他无法保守秘密的环境非常有用。

- **创建应用程序时**：创建新的 OAuth 应用程序时，您可以选择在不使用密钥的情况下创建它。
- **创建后**：对于现有应用程序，您可以在应用程序设置中删除客户端密钥。然后该应用程序将作为公共应用程序运行。

公共应用程序仅使用客户端 ID 进行身份验证（例如，在带有 PKCE 的设备代码或授权代码流中）。具有机密的应用程序仍然可以在需要时使用该机密（例如用于令牌请求的`Authorization: Basic`）。

### 重定向 URI授权请求中发送的 `redirect_uri` 必须与您的应用程序上注册的重定向 URI 之一完全匹配。

环回重定向 URI 是一个例外：对于在环回主机上使用 `http` 方案的 URI（`localhost`、`127.0.0.1` 或 `[::1]`），只要所有其他组件都与已注册的 URI 匹配，在请求时就会接受任何端口。这遵循 [RFC 8252 §7.3](https://datatracker.ietf.org/doc/html/rfc8252#section-7.3) / OAuth 2.1，并允许本机应用程序、CLI 和 MCP 客户端注册无端口 URI，例如 `http://localhost/callback`，然后侦听操作系统分配给它们的任何临时端口（例如 `http://localhost:49282/callback`）。

需要记住以下几点：

- 例外情况仅适用于`http`方案。 `https` 和自定义方案（例如 `myapp://callback`）重定向 URI 始终需要完全匹配，包括端口。
- 环回主机单独匹配：对`http://localhost:49282/callback`的请求与已注册的`http://127.0.0.1/callback`不匹配。如果您的应用程序可能使用其中一个，请注册两者。
- 仅端口可能不同。路径、查询和用户信息仍然必须完全匹配。

### 如果您在 Spaces 中托管

> [!提示]
> 如果您将应用程序托管在 Spaces 上，那么流程将更容易实现（并且直接内置到 Gradio 中）；检查我们的[Spaces OAuth guide](https://huggingface.co/docs/hub/spaces-oauth)。

### 自动创建 oauth 应用程序Hugging Face 支持 CIMD 又名 [Client ID Metadata Documents](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)，它允许您以自动方式为您的网站创建 oauth 应用程序：

- 将端点添加到您的网站`/.well-known/oauth-cimd`，它返回以下 JSON：

```json
{
  client_id:                  "[your website url]/.well-known/oauth-cimd",
  client_name:                "Your Website",
  redirect_uris:              ["[your website url]/oauth/callback/huggingface"],
  token_endpoint_auth_method: "none",
  logo_uri:                  "https://....", // optional
  client_uri:                 "[your website url]", // optional
}
```

- 使用`"[your website url]/.well-known/oauth-cimd"`作为客户端ID，并使用PCKE作为身份验证机制

在本地端口上接收授权代码的本机应用程序和 MCP 客户端可以在 `redirect_uris` 中列出无端口环回 URI，例如`["http://localhost/callback", "http://127.0.0.1/callback"]` — 参见[Redirect URIs](#redirect-uris)。

这对于临时环境或 MCP 客户端特别有用。在拥抱聊天中查看[implementation example](https://github.com/huggingface/chat-ui/pull/1978)。

## 设备代码 OAuth

设备代码流允许用户通过在另一台设备（例如手机或浏览器）上输入短代码来授权一台设备（例如 CLI）上的应用程序。运行应用程序的设备上不需要重定向 URI 或浏览器。

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/oauth-device-first-step.png)

![](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/oauth-device-second-step.png)

### 使用示例脚本进行测试

您可以使用以下脚本测试设备代码 OAuth 应用程序。将 `<Client ID>` 替换为您应用程序的客户端 ID。对于**公共应用程序**（没有秘密），脚本按原样运行。对于 **具有秘密的应用程序**，请将 `Authorization: Basic` 标头（`client_id:client_secret` 的 Base64）添加到设备和令牌请求中。

```sh
#!/bin/bash
CLIENT_ID="<Client ID>"

# Step 1: Get device code
RESPONSE=$(curl -s -X POST https://huggingface.co/oauth/device \
  -d "client_id=$CLIENT_ID")

DEVICE_CODE=$(echo $RESPONSE | jq -r '.device_code')
USER_CODE=$(echo $RESPONSE | jq -r '.user_code')
VERIFICATION_URI=$(echo $RESPONSE | jq -r '.verification_uri')

echo "Device Code: $DEVICE_CODE"
echo "User Code: $USER_CODE"
echo ""
echo "Open: ${VERIFICATION_URI}"
echo "Enter the user code: $USER_CODE"
echo ""
read -p "Press Enter after authorizing..."

# Step 3: Get token
curl -X POST https://huggingface.co/oauth/token \
  -d "grant_type=urn:ietf:params:oauth:grant-type:device_code" \
  -d "device_code=$DEVICE_CODE" \
  -d "client_id=$CLIENT_ID"
```> [!注意]
> 对于具有客户端密钥的 OAuth 应用程序，请在设备代码请求和令牌请求上包含 `Authorization: Basic` 标头（带有 Base64 编码的 `client_id:client_secret`）。

## 目前支持的范围

目前支持的范围有：

- `openid`：除了访问令牌之外，还接收 ID 令牌。
- `profile`：读取用户的个人资料信息（用户名、头像等）
- `email`：读取用户的电子邮件地址。
- `read-billing`：了解用户是否设置了支付方式。
- `read-repos`：阅读用户的个人存储库。
- `gated-repos`：读取用户已被授予访问权限的公共门控存储库的内容。与 `read-repos` 不同，这不会授予对私人存储库的访问权限。
- `contribute-repos`：创建存储库并访问此应用程序创建的存储库。除非授予额外权限，否则无法访问任何其他存储库。
- `write-repos`：读写用户的个人存储库。
- `manage-repos`：全面管理用户的个人仓库，包括创建和删除它们。
- `read-collections`：阅读用户的个人收藏。
- `write-collections`：读写用户的个人收藏，包括创建和删除它们。
- `inference-api`：代表用户向[Inference Providers](https://huggingface.co/docs/inference-providers/index)提出推理请求。- `read-endpoints`：查看用户的[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)并代表用户向其发出推理请求。
- `write-endpoints`：管理用户的推理端点，包括创建和删除它们。包括 `read-endpoints` 访问权限。
- `jobs`：运行[jobs](https://huggingface.co/docs/huggingface_hub/main/en/guides/jobs)
- `webhooks`：管理[webhooks](https://huggingface.co/docs/huggingface_hub/main/en/guides/webhooks)
- `write-discussions`：代表用户打开讨论和拉取请求，并与讨论互动（包括反应、发布/编辑评论、结束讨论等）。要在私有存储库上打开拉取请求，您还需要请求 `read-repos` 范围。

所有其他信息均可在 [OpenID metadata](https://huggingface.co/.well-known/openid-configuration) 中找到。

> [!警告]
> 如果您需要任何额外的范围，请联系我们。

## 访问组织资源

默认情况下，oauth 应用程序不需要访问组织资源。

但某些范围（例如 `read-repos` 或 `read-billing`）也适用于组织。

用户在授权应用程序时可以选择向哪些组织授予访问权限。如果您需要访问特定组织，可以将 `orgIds=ORG_ID` 作为查询参数添加到 OAuth 授权 URL。您必须将 `ORG_ID` 替换为组织 ID，该 ID 可在 userinfo 响应的 `organizations.sub` 字段中找到。

## 品牌推广您可以自由地使用自己设计的按钮。下面是一些有用的 SVG 图像。

查看 [our badges](https://huggingface.co/datasets/huggingface/badges#sign-in-with-hugging-face) 以及将它们集成到 Markdown 或 HTML 中的说明。

[⟦T76⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)
[⟦T77⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)

[⟦T78⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)
[⟦T79⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)

[⟦T80⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)
[⟦T81⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)

[⟦T82⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)
[⟦T83⟧](https://huggingface.co/oauth/authorize?client_id=CLIENT_ID&redirect_uri=REDIRECT_URI&scope=openid%20profile&state=STATE)

## 组织令牌交换 (RFC 8693)

> [!警告]
> 此功能是企业计划的一部分。

令牌交换允许组织以编程方式为其成员颁发访问令牌，而无需交互式用户同意。这对于构建需要代表组织成员访问 Hugging Face 资源的内部工具、自动化管道和企业集成特别有用。

> [!提示]
> 如果您只需要 CI/CD 工作流程（GitHub Actions、GitLab CI、CircleCI 等）中的无密钥身份验证（无需为每个成员颁发令牌），请参阅[Trusted Publishers](./trusted-publishers)，它也使用 `/oauth/token`，但采用 CI 提供商铸造的 OIDC `id_token` 作为主题令牌（不需要企业计划，不需要客户端凭据）。

该功能实现了[RFC 8693 - OAuth 2.0 Token Exchange](https://www.rfc-editor.org/rfc/rfc8693.html)，一种用于代币交换场景的标准协议。

### 用例

令牌交换专为您的组织需要以下的场景而设计：- **构建内部平台**：创建代表您的团队成员访问 Hugging Face 资源的仪表板或门户，无需每个用户手动进行身份验证。
- **自动化 CI/CD 管道**：为需要将模型或数据集推送到组织存储库的自动化工作流程发出短期的、有范围的令牌。
- **与企业身份系统集成**：通过根据您的内部用户目录颁发令牌，将您现有的身份提供商与 Hugging Face 连接起来。
- **实施自定义访问控制**：构建中间件，根据组织的内部策略颁发具有特定范围的令牌。

### 它是如何工作的

1. 您的组织有一个 OAuth 应用程序绑定到您的组织，并具有 `token-exchange` 权限。
2. 您的后端服务使用客户端凭据对此 OAuth 应用程序进行身份验证。
3. 您的服务请求特定组织成员（通过电子邮件标识）的访问令牌。
4. Hugging Face 验证用户是否是您组织的成员并颁发范围令牌。
5. 发行的Token只能访问您组织范围内的资源。

### 先决条件要使用令牌交换，您需要具有 `token-exchange` 权限的组织绑定 OAuth 应用程序。请联系 Hugging Face 支持人员，为您的组织设置符合条件的 OAuth 应用程序。

配置完成后，您将收到：
- **客户端 ID**（例如，`a1b2c3d4-e5f6-7890-abcd-ef1234567890`）
- **客户秘密**（确保安全！）

> [!警告]
> 组织管理员可以在创建后管理 OAuth 应用程序，包括刷新客户端密钥和配置令牌持续时间。

### 身份验证

令牌交换将 HTTP 基本身份验证与您的 OAuth 应用程序凭据结合使用。通过对 `client_id:client_secret` 进行 Base64 编码来创建授权标头：

```bash
# Create the authorization header
export CLIENT_ID="your-client-id"
export CLIENT_SECRET=REDACTED
export AUTH_HEADER=$(echo -n "${CLIENT_ID}:${CLIENT_SECRET}" | base64)
```

### 通过电子邮件发行代币

要使用组织成员的电子邮件地址为其颁发访问令牌：

```bash
curl -X POST "https://huggingface.co/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "Authorization: Basic ${AUTH_HEADER}" \
  -d "grant_type=urn:ietf:params:oauth:grant-type:token-exchange" \
  -d "subject_token=user@yourorg.com" \
  -d "subject_token_type=urn:huggingface:token-type:user-email"
```

### 回应

成功的请求返回访问令牌：

```json
{
  "access_token": "hf_oauth_...",
  "token_type": "bearer",
  "expires_in": 28800,
  "scope": "openid profile email read-repos",
  "id_token": "eyJhbGciOiJS...",
  "issued_token_type": "urn:ietf:params:oauth:token-type:access_token"
}
```

当请求 `openid` 范围时，会包含 `id_token` 字段。

然后，您可以使用此令牌代表用户发出 API 请求：

```bash
curl "https://huggingface.co/api/whoami-v2" \
  -H "Authorization: Bearer ${ACCESS_TOKEN}"
```

### 范围控制

默认情况下，颁发的令牌继承 OAuth 应用程序上配置的所有范围。您可以通过添加 `scope` 参数来请求特定范围。有关可用值，请参阅[Currently supported scopes](#currently-supported-scopes)。令牌的有效权限受到请求的范围和用户在组织内的角色的限制。

```bash
curl -X POST "https://huggingface.co/oauth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -H "Authorization: Basic ${AUTH_HEADER}" \
  -d "grant_type=urn:ietf:params:oauth:grant-type:token-exchange" \
  -d "subject_token=user@yourorg.com" \
  -d "subject_token_type=urn:huggingface:token-type:user-email" \
  -d "scope=openid profile"
```

> [!提示]
> 遵循最小权限原则：仅请求应用程序实际需要的范围。

### 安全考虑

通过代币交易所发行的代币具有内置的安全限制：

- **组织范围**：令牌只能访问组织内的资源（组织拥有的模型、数据集、空间和集合）。在组织外部，访问权限是只读的，并且仅限于：来自任何用户或组织的公共集合，以及用户已单独授予访问权限的公共门控存储库。
- **无个人访问**：代币无法访问用户的个人私有存储库或来自其他组织的私有存储库。
- **短暂**：令牌默认在 8 小时后过期。组织管理员可以在 OAuth 应用程序设置中配置令牌持续时间（最长 30 天）。不提供刷新令牌。
- **可审计**：所有代币交换都会记录在您组织的[audit logs](./audit-logs)中并可见。> [!警告]
> 小心保护您的 OAuth 应用程序凭据。有权访问您的客户端密钥的任何人都可以为您组织的任何成员颁发令牌。

### 错误响应

|错误|描述 |
|--------|-------------|
| `invalid_client` |客户端无权使用令牌交换，或应用程序未绑定组织 |
| `invalid_grant` |在绑定的组织中找不到用户 |
| `invalid_scope` |请求的范围无效 |

### 参考

**拨款类型：**
```
urn:ietf:params:oauth:grant-type:token-exchange
```

**请求参数（`subject_token_type`）：**

|价值|描述 |
|--------|-------------|
| `urn:huggingface:token-type:user-email` |通过电子邮件地址识别用户 |

**响应字段（`issued_token_type`）：**

|价值|描述 |
|--------|-------------|
| `urn:ietf:params:oauth:token-type:access_token` |表示已颁发访问令牌 |

**相关文档：**
- [RFC 8693 - OAuth 2.0 Token Exchange](https://www.rfc-editor.org/rfc/rfc8693.html)
- [Audit Logs](./audit-logs)

### 门控模型
https://huggingface.co/docs/hub/models-gate.md