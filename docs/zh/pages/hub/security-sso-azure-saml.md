<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Microsoft Entra ID (Azure AD) 配置 SAML SSO

在本指南中，我们将使用 Microsoft Entra ID 作为 SSO 提供程序，并使用安全断言标记语言 (SAML) 协议作为我们的首选身份协议。

我们目前支持 SP 发起和 IdP 发起的身份验证。有关用户配置，请参阅[SCIM](./enterprise-scim)。

> [!警告]
> 此功能是团队和企业计划的一部分。

## 第 1 步：在您的身份提供商中创建一个新应用程序

在浏览器中打开一个新选项卡/窗口并登录到组织的 Azure 门户。

导航到“企业应用程序”并单击“新建应用程序”按钮。

您将被重定向到此页面，单击“创建您自己的应用程序”，填写应用程序的名称，然后“创建”应用程序。

然后选择“单点登录”，并选择 SAML

## 步骤 2：在 Azure 上配置您的应用程序

在浏览器中打开一个新选项卡/窗口，然后导航到组织设置的 SSO 部分。选择 SAML 协议。

从 Hugging Face 上的组织设置中复制“SP 实体 Id”，并将其粘贴到 Azure 上的“标识符（实体 Id）”字段中 (1)。从 Hugging Face 上的组织设置中复制“断言消费者服务 URL”，并将其粘贴到 Azure 上的“回复 URL”字段中 (2)。

URL 如下所示：`https://huggingface.co/organizations/[organizationIdentifier]/saml/consume`。

然后在“SAML 证书”下，验证“登录选项”是否设置为“签署 SAML 响应和断言”。

保存您的新申请。

## 步骤 3：完成 Hugging Face 上的配置

在 Azure 应用程序中的“设置”下，找到以下字段：

- 登录网址

在“SAML 证书”下：

- 下载“证书（base64）”

您将需要它们来完成 Hugging Face 上的 SSO 设置。

在组织设置的 SSO 部分中，从 Azure 复制粘贴以下值：

- 登录 URL -> 登录 URL
- 证书 -> 公共证书

公共证书必须具有以下格式：

```
-----BEGIN CERTIFICATE-----
{certificate}
-----END CERTIFICATE-----
```

您现在可以单击“更新并测试 SAML 配置”来保存设置。

您应该被重定向到 SSO 提供商 (IdP) 登录提示。登录后，您将被重定向到组织的设置页面。

SAML 选择器附近的绿色复选标记将证明测试成功。

	
	

## 步骤 4：在您的组织中启用 SSO现在单点登录已配置并测试完毕，您可以通过单击“启用”按钮为组织的成员启用它。

启用后，您组织的成员必须完成 [How it works](./security-sso-basic#how-it-works) 部分中描述的 SSO 身份验证流程。

### 集线器本地缓存
https://huggingface.co/docs/hub/local-cache.md