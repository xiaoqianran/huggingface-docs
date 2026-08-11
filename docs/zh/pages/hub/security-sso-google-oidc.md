<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Google Workspace 配置 OIDC SSO

在本指南中，我们将使用 Google Workspace 作为 SSO 提供商，并使用 OpenID Connect (OIDC) 协议作为我们的首选身份协议。

我们目前支持 SP 发起的身份验证。有关用户配置，请参阅[SCIM](./enterprise-scim)。

> [!警告]
> 此功能是团队和企业计划的一部分。

## 第 1 步：在 Google Workspace 中创建 OIDC 应用

- 在 Google Cloud 控制台中，搜索并导航到 `Google Auth Platform` > `Clients`。
- 单击`Create Client`。
- 对于应用程序类型，选择`Web Application`。
- 为您的应用程序提供一个名称。
- 从 Hugging Face 组织设置中检索 `Redirection URI`，转到 `SSO` 选项卡并选择 `OIDC` 协议。
- 点击`Create`。
- 将出现一个弹出窗口，其中包含 `Client ID` 和 `Client Secret`，复制它们并将其粘贴到您的 Hugging Face 组织设置中。在 `SSO` 选项卡中（确保选择 `OIDC`），粘贴 `Client Identifier` 和 `Client Secret` 的相应值。

## 步骤 2：使用 Google 的 OIDC 详细信息配置拥抱脸部

- 此时，应在 Hugging Face 组织设置 `SSO` 选项卡中设置 **Client ID** 和 **Client Secret**。
- 将 **发行者 URL** 设置为 `https://accounts.google.com`。

## 步骤 3：测试并启用 SSO> [!警告]
> 测试之前，请确保您已授予适当用户访问应用程序的权限。执行测试的管理员必须具有访问权限。

- 现在，在 Hugging Face SSO 设置中，单击 **“更新并测试 OIDC 配置”**。
- 您应该被重定向到您的 Google 登录提示。登录后，您将被重定向到组织的设置页面。
- OIDC 选择器附近的绿色复选标记将确认测试成功。
- 测试成功后，您可以通过单击“启用”按钮为您的组织启用 SSO。
- 启用后，组织成员必须完成 [How it works](./security-sso-basic#how-it-works) 部分中描述的 SSO 身份验证流程。

### 技能
https://huggingface.co/docs/hub/agents-skills.md