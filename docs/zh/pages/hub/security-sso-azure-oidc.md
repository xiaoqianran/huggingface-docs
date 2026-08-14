<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Microsoft Entra ID (Azure AD) 配置 OIDC SSO

本指南将使用 Microsoft Entra ID 作为 SSO 提供程序，并使用 Open ID Connect (OIDC) 协议作为我们的首选身份协议。

> [!警告]
> 此功能是团队和企业计划的一部分。

## 第 1 步：在您的身份提供商中创建一个新应用程序

在浏览器中打开新选项卡/窗口并登录到组织的 Azure 门户。

导航到 Microsoft Entra ID 管理中心并单击“企业应用程序”

您将被重定向到此页面。然后单击顶部的“新应用程序”和“创建您自己的应用程序”。

输入应用程序的名称（例如 Hugging Face SSO），然后选择“注册应用程序以与 Microsoft Entra ID 集成（您正在开发的应用程序）”。

## 步骤 2：在 Azure 上配置您的应用程序

在浏览器中打开一个新选项卡/窗口，然后导航到组织设置的 SSO 部分。选择 OIDC 协议。

从 Hugging Face 上的组织设置中复制“重定向 URI”，并将其粘贴到 Azure Entra ID 上的“重定向 URI”字段中。确保在下拉菜单中选择“Web”。
URL 如下所示：`https://huggingface.co/organizations/[organizationIdentifier]/oidc/consume`。保存您的新申请。

## 步骤 3：完成 Hugging Face 上的配置

我们需要收集以下信息来完成 Hugging Face 上的设置：

- OIDC 应用程序的客户端 ID
- OIDC 应用程序的客户端密钥
- OIDC 应用程序的发行者 URL

在 Microsoft Entra ID 中，导航到企业应用程序，然后单击列表中新创建的应用程序。

在应用程序概述中，单击“单点登录”，然后单击“转到应用程序”

在 OIDC 应用程序概述中，您将找到一个名为“应用程序（客户端）ID”的可复制字段。
将该 ID 复制到剪贴板并将其粘贴到 Huggingface 上的“客户端 ID”字段中。

接下来，单击 Microsoft Entra 顶部菜单中的“端点”。
复制“OpenID 连接元数据文档”字段中的值并将其粘贴到 Hugging Face 中的“问题 URL”字段中。

返回 Microsoft Entra，导航到“证书和机密”，然后通过单击“新客户端机密”生成新机密。

创建秘密后，复制秘密值并将其粘贴到 Hugging Face 上的“客户端秘密”字段中。

您现在可以单击“更新并测试 OIDC 配置”来保存设置。您应该被重定向到 SSO 提供商 (IdP) 登录提示。登录后，您将被重定向到组织的设置页面。

OIDC 选择器附近的绿色复选标记将证明测试成功。

## 步骤 4：在您的组织中启用 SSO

现在单点登录已配置并测试完毕，您可以通过单击“启用”按钮为组织的成员启用它。

启用后，您组织的成员必须完成 [How it works](./security-sso-basic#how-it-works) 部分中描述的 SSO 身份验证流程。

### 用于 AI 代理的 Hugging Face CLI
https://huggingface.co/docs/hub/agents-cli.md