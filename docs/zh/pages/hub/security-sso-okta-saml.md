<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Okta 配置 SAML SSO

在本指南中，我们将使用 Okta 作为 SSO 提供商，并使用安全断言标记语言 (SAML) 协议作为我们的首选身份协议。

我们目前支持 SP 发起和 IdP 发起的身份验证。有关用户配置，请参阅[SCIM](./enterprise-scim)。

> [!警告]
> 此功能是团队和企业计划的一部分。

## 第 1 步：在您的身份提供商中创建一个新应用程序

在浏览器中打开新选项卡/窗口并登录您的 Okta 帐户。

导航到“管理/应用程序”，然后单击“创建应用程序集成”按钮。

然后选择“SAML 2.0”应用程序并单击“创建”。

## 步骤 2：在 Okta 上配置您的应用程序

在浏览器中打开一个新选项卡/窗口，然后导航到组织设置的 SSO 部分。选择 SAML 协议。

从 Hugging Face 上的组织设置中复制“断言消费者服务 URL”，并将其粘贴到 Okta 上的“单点登录 URL”字段中。
URL 如下所示：`https://huggingface.co/organizations/[organizationIdentifier]/saml/consume`。

在 Okta 上，进行以下设置：- 设置受众 URI（SP 实体 ID）以匹配 Hugging Face 上的“SP 实体 ID”值。
- 将名称 ID 格式设置为电子邮件地址。
- 在“显示高级设置”下，验证响应和断言签名是否设置为：已签名。

保存您的新申请。

## 步骤 3：完成 Hugging Face 上的配置

在 Okta 应用程序的“登录/设置/更多详细信息”下，找到以下字段：

- 登录网址
- 公共证书
- SP实体ID
  您将需要它们来完成 Hugging Face 上的 SSO 设置。

在组织设置的 SSO 部分中，从 Okta 复制粘贴以下值：

- 登录网址
- SP实体ID
- 公共证书

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

### 流数据集
https://huggingface.co/docs/hub/datasets-streaming.md