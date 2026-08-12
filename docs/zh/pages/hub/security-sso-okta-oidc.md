<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Okta 配置 OIDC SSO

在本指南中，我们将使用 Okta 作为 SSO 提供商，并使用 Open ID Connect (OIDC) 协议作为我们的首选身份协议。

> [!警告]
> 此功能是团队和企业计划的一部分。

## 第 1 步：在您的身份提供商中创建一个新应用程序

在浏览器中打开新选项卡/窗口并登录您的 Okta 帐户。

导航到“管理/应用程序”，然后单击“创建应用程序集成”按钮。

然后选择“OIDC - OpenID Connect”应用程序，选择应用程序类型“Web 应用程序”并单击“创建”。

## 步骤 2：在 Okta 中配置您的应用程序

在浏览器中打开一个新选项卡/窗口，然后导航到组织设置的 SSO 部分。选择 OIDC 协议。

从 Hugging Face 上的组织设置中复制“重定向 URI”，并将其粘贴到 Okta 上的“登录重定向 URI”字段中。
URL 如下所示：`https://huggingface.co/organizations/[organizationIdentifier]/oidc/consume`。

您可以将可选的注销重定向 URI 留空。

保存您的新申请。

## 步骤 3：完成 Hugging Face 上的配置

在 Okta 应用程序的“常规”下，找到以下字段：- 客户ID
- 客户秘密
- 发行人网址
  您将需要这些来完成 Hugging Face 上的 SSO 设置。

Okta 发行人 URL 通常是类似 `https://tenantId.okta.com` 的 URL；你可以参考他们的[guide](https://support.okta.com/help/s/article/What-is-theIssuerlocated-under-the-OpenID-Connect-ID-Token-app-settings-used-for?language=en_US)了解更多详情。

在 Hugging Face 上组织设置的 SSO 部分中，从 Okta 复制粘贴以下值：

- 客户ID
- 客户秘密

您现在可以单击“更新并测试 OIDC 配置”来保存设置。

您应该被重定向到 SSO 提供商 (IdP) 登录提示。登录后，您将被重定向到组织的设置页面。

OIDC 选择器附近的绿色复选标记将证明测试成功。

## 步骤 4：在您的组织中启用 SSO

现在单点登录已配置并测试完毕，您可以通过单击“启用”按钮为组织的成员启用它。

启用后，您组织的成员必须完成 [How it works](./security-sso-basic#how-it-works) 部分中描述的 SSO 身份验证流程。

### 高级主题
https://huggingface.co/docs/hub/models-advanced.md