<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 基本单点登录

> [!警告]
> 此功能是团队和企业计划的一部分。

基本 SSO 在标准 Hugging Face 登录之上添加了一个访问控制层。当成员访问组织命名空间下的资源（例如私有模型、数据集和空间）时，它允许您通过身份提供商 (IdP) 强制执行身份验证。

有关与托管 SSO 的比较，请参阅 [SSO overview](./enterprise-sso)。

## 它是如何工作的

> [!注意]
> **基本 SSO 不会取代 Hugging Face 登录。** 您的成员仍需要使用自己的凭据（电子邮件/密码、Google 或 GitHub）登录 Hugging Face，然后系统会提示完成 SSO 身份验证以访问组织的资源。这是设计使然：基本 SSO 可保护对组织的访问，而无需接管用户的 Hugging Face 身份。

启用单点登录后，组织成员通过您的身份提供商 (IdP) 进行身份验证。您可以选择 SSO 是**强制**还是**可选**：- **强制**（默认）：成员必须先完成 SSO 身份验证，然后才能访问组织命名空间下的任何内容。
- **可选**：页面顶部的横幅会提示成员设置 SSO，但可以跳过它并仍然访问组织。当您要迁移大量用户并希望在明确执行 SSO 之前给他们时间整理帐户时，这会很方便。

	
	

所有人（包括非会员）仍然可以访问公共内容。

**我们使用电子邮件地址来识别 SSO 用户。作为用户，请确保您的组织电子邮件地址（例如您的公司电子邮件）已添加到 [your user account](https://huggingface.co/settings/account)。**

当用户登录时，系统将提示他们完成单点登录身份验证流程，并显示类似以下内容的横幅：

	
	

单点登录仅适用于您的组织。成员可能属于 Hugging Face 上的其他组织。

## 开始使用

可以直接从组织的设置中配置基本 SSO。 Hugging Face Hub 可以与任何符合 OIDC 要求的或 SAML 身份提供商配合使用，包括 Okta、OneLogin 和 Microsoft Entra ID (Azure AD)。

请参阅我们的 [Configuration Guides](./security-sso-configuration-guides) 了解分步设置说明。

## 用户配置在您的组织中启用 SSO 后，可以复制直接加入链接并与新成员共享。此 SSO 加入链接在 **SSO** 和 **成员** 设置选项卡中均可用。由于启用了 SSO 的组织无法使用经典邀请链接，因此 SSO 加入链接是邀请团队成员加入组织的主要方法。只需单击复制按钮即可将链接复制到剪贴板并与您想要邀请的成员共享。当收件人单击共享链接时，他们将能够通过 SSO 进行身份验证并直接加入您的组织。

企业计划中的组织还可以使用 [SCIM](./enterprise-scim) 自动执行来自身份提供商的基于邀请的配置。更多详情请参阅[SCIM guide](./enterprise-scim)。

## 单点登录功能

基本 SSO 支持[role mapping, resource group mapping, session timeout, matching email domains, and external collaborators](./security-sso-user-management)。这些功能可通过您组织的设置进行配置。

### 数据集下载统计
https://huggingface.co/docs/hub/datasets-download-stats.md