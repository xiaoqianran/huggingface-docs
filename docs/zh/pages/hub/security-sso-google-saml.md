<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Google Workspace 配置 SAML SSO

在本指南中，我们将使用 Google Workspace 作为 SSO 提供商，并使用安全断言标记语言 (SAML) 协议作为我们的首选身份协议。

我们目前支持 SP 发起和 IdP 发起的身份验证。有关用户配置，请参阅[SCIM](./enterprise-scim)。

> [!警告]
> 此功能是团队和企业计划的一部分。

## 第 1 步：在 Google Workspace 中创建 SAML 应用

- 在 Google Workspace 管理控制台中，导航至 `Admin` > `Apps` > `Web and mobile apps`。
- 单击`Add app`，然后单击`Add custom SAML app`。
- 您必须在“应用程序名称”字段中提供应用程序的名称。
- 单击`Continue`。

## 步骤 2：使用 Google 的 IdP 详细信息配置拥抱脸部

- Google 设置中的下一个屏幕包含您的应用程序的 SSO 信息。
- 在 Hugging Face 组织设置中，转到 `SSO` 选项卡并选择 `SAML` 协议。
- 将 Google 中的 **SSO URL** 复制到 Hugging Face 上的 **登录 URL** 字段中。
- 将 Google 的 **证书** 复制到 Hugging Face 上的相应字段中。公共证书必须具有以下格式：
  ```
  -----BEGIN CERTIFICATE-----
  {certificate}
  -----END CERTIFICATE-----
  ```

- 在 Google Workspace 设置中，点击 `Continue`。## 步骤 3：使用 Hugging Face 的 SP 详细信息配置 Google

- 在“服务提供商详细信息”屏幕中，您需要 Hugging Face SSO 设置中的 `Assertion Consumer Service URL` 和 `SP Entity ID`。将它们复制到 Google 中相应的 `ACS URL` 和 `Entity ID` 字段中。
- 确保设置以下内容：
    - 勾选**签名回复**框。
    - 姓名 ID 格式：`EMAIL`
    - 姓名 ID：`Basic Information > Primary email`

- 单击`Continue`。

## 步骤 4：属性映射

- 在“属性映射”屏幕上，单击`Add mapping`并配置要发送的属性。此步骤是可选的，取决于您要在 Hugging Face 上使用 [Role Mapping](./security-sso-user-management#role-mapping) 还是 [Resource Group Mapping](./security-sso-user-management#resource-group-mapping)。

- 点击`Finish`。

## 步骤 5：测试并启用 SSO

> [!警告]
> 测试之前，请确保您已在 Google Workspace 管理控制台中应用的“用户访问权限”设置下为相应用户授予了应用访问权限。执行测试的管理员必须具有访问权限。用户访问权限更改可能需要几分钟才能应用到 Google Workspace。- 现在，在 Hugging Face SSO 设置中，单击 **“更新和测试 SAML 配置”**。
- 您应该被重定向到您的 Google 登录提示。登录后，您将被重定向到组织的设置页面。
- SAML 选择器附近的绿色复选标记将确认测试成功。
- 测试成功后，您可以通过单击“启用”按钮为您的组织启用 SSO。
- 启用后，您组织的成员必须完成 [How it works](./security-sso-basic#how-it-works) 部分中描述的 SSO 身份验证流程。

### 小部件示例
https://huggingface.co/docs/hub/models-widgets-examples.md