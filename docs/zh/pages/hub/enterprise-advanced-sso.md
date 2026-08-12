<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 托管单点登录

> [!警告]
> 此功能是 Enterprise Plus 计划的一部分。

托管 SSO **完全取代 Hugging Face 登录**。您的身份提供商将成为整个 Hugging Face 平台上组织成员的唯一身份验证方法。该组织控制整个用户生命周期，从帐户创建到停用。

有关与基本 SSO 的比较，请参阅 [SSO overview](./enterprise-sso)。

## 它是如何工作的

> [!注意]
> **托管 SSO 取代了 Hugging Face 登录。** 您的 IdP 是托管用户在 Hugging Face 上进行身份验证的唯一方式 — 没有单独的 Hugging Face 登录。与基本 SSO 不同，会员不需要预先存在的 Hugging Face 帐户。当用户首次通过 IdP 进行身份验证时，系统会自动为其创建一个帐户。

您的 IdP 是组织所有成员与 Hugging Face 平台任何部分交互的强制身份验证路由。成员需要通过您的 IdP 对所有 Hugging Face 服务进行身份验证，而不仅仅是在访问私人或组织存储库时。当用户在您的 IdP 中被停用时，他们的 Hugging Face 帐户也会被停用。这使您的组织可以完全控制身份、访问和数据治理。

## 开始使用

托管 SSO 无法自行配置。要为您的组织启用托管 SSO，请联系 Hugging Face 团队。该设置是与我们的技术团队合作完成的，以确保您的组织顺利过渡。

支持 SAML 2.0 和 OIDC 协议，并且可以与 Okta、Microsoft Entra ID (Azure AD) 和 Google Workspace 等流行的身份提供商集成。

## 用户配置

托管 SSO 通过 [SCIM](./enterprise-scim) 引入自动化用户配置，管理 Hugging Face 上的整个用户生命周期。 SCIM 允许您的 IdP 将用户身份信息传达给 Hugging Face，从而在您的 IdP 中发生更改时实现用户帐户的自动创建、更新（例如，名称更改、角色更改）和停用。

在我们的[dedicated guide](./enterprise-scim)中了解有关如何设置和管理 SCIM 的更多信息。

## 单点登录功能

托管 SSO 支持 [role mapping, resource group mapping, session timeout, and external collaborators](./security-sso-user-management)。这些功能可通过您组织的设置进行配置。

## 对管理帐户的限制> [!警告]
> 管理帐户的重要注意事项。

为了确保组织控制和数据治理，托管用户帐户具有特定的限制：

* **禁止创建个人内容**：托管用户无法在其个人用户命名空间中创建任何内容（模型、数据集或空间）。所有内容都必须在组织内创建。
* **组织范围内的协作**：托管用户仅限于在其管理组织内进行协作。他们不能加入其他组织或为其管理组织之外的存储库做出贡献。
* **内容可见性**：托管用户创建的内容驻留在组织内。虽然托管用户无法在其个人配置文件中创建公共内容，但如果组织的设置允许，他们可以**在组织内创建公共内容**。

这些限制维护了您企业的安全边界。对于个人项目或更广泛的组织外部协作，成员应使用单独的、非托管的 Hugging Face 帐户。

### 将使用 HF 按钮登录到您的空间
https://huggingface.co/docs/hub/spaces-oauth.md