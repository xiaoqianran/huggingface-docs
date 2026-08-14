<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 单点登录 (SSO)

> [!警告]
> 此功能是团队和企业计划的一部分。

Hugging Face 提供两种不同的 SSO 模型，每种模型都针对不同的组织需求而设计。了解这两种方法之间的差异是为您的团队选择正确设置的关键。

## 概览

|  | **基本单点登录** | **托管单点登录** |
| --- | --- | --- |
| **计划** |团队与企业 |企业增强版 |
| **范围** |仅组织资源|整个抱脸平台|
| **取代拥抱脸登录** |否 — 用户保留其现有的 Hugging Face 凭据 |是的 — 您的 IdP 成为唯一的登录方法 |
| **用户帐户** |用户保留自己的个人 Hugging Face 帐户 |帐户由组织拥有和管理 |
| **个人内容** |用户可以在他们的个人命名空间中创建内容 |用户只能在组织内创建内容 |
| **多组织会员资格** |用户可以属于多个组织 |用户仅限于其管理组织 |
| **用户配置** |手动（SSO 加入链接）— 或企业版上基于邀请的 [SCIM](./enterprise-scim) |完整生命周期（[SCIM](./enterprise-scim)）|| **设置** |组织设置中的自助服务 |需要 Hugging Face 团队进行设置 |
| **外部合作者** |是的 |是的 |
| **协议** | SAML 2.0 和 OIDC | SAML 2.0 和 OIDC |
| **角色映射** |是的 |是的 |
| **资源组映射** |是的 |是的 |

## 基本单点登录

基本 SSO 在标准 Hugging Face 登录之上添加了一个访问控制层。它**不会**取代 Hugging Face 登录 — 成员保留其现有凭据，并且仅在访问组织的资源时才会提示完成 SSO。

这非常适合希望**安全地访问其组织资源，同时保留个人 Hugging Face 帐户的灵活性**的团队。设置是通过您组织的设置进行的自助服务。

[Getting started with Basic SSO →](./security-sso-basic)

## 托管单点登录

托管 SSO **完全取代 Hugging Face 登录**。您的身份提供商将成为整个 Hugging Face 平台的唯一身份验证方法。该组织控制整个用户生命周期，从帐户创建到停用。这是专为需要**完全控制身份、访问和数据治理**的公司而设计的。托管帐户具有[specific restrictions](./enterprise-advanced-sso#restrictions-on-managed-accounts)（无个人内容，组织范围内的协作）。设置需要与 Hugging Face 团队协调。

[Getting started with Managed SSO →](./enterprise-advanced-sso)

## 用户配置 (SCIM)

两种 SSO 模型都支持[SCIM](./enterprise-scim)（跨域身份管理系统），以自动从身份提供商进行用户配置。这两个模型使用 SCIM 的方式不同，但与各自的理念一致：

- **基本 SSO**（企业计划）：SCIM 自动邀请**现有 Hugging Face 用户加入您的组织。用户必须接受加入邀请。
- **托管 SSO**（Enterprise Plus 计划）：SCIM 管理**整个用户生命周期** — 帐户创建、配置文件更新和停用。

在[User Provisioning (SCIM) guide](./enterprise-scim)了解更多信息。

## 您应该选择哪种型号？

**如果您的团队需要安全地访问组织资源，同时允许成员维护自己的 Hugging Face 帐户并参与更广泛的社区，请选择基本 SSO**。**如果您的企业需要集中控制所有用户帐户、自动配置和取消配置以及防止在组织外部创建任何内容的严格数据治理策略，请选择托管 SSO**。

两种模型都支持 SAML 2.0 和 OIDC 协议，并且可以与 Okta、Microsoft Entra ID (Azure AD) 和 Google Workspace 等流行的身份提供商集成。

## 进一步阅读

- [User Management](./security-sso-user-management) — 角色映射、资源组映射、会话超时等
- [Configuration Guides](./security-sso-configuration-guides) — Okta、Microsoft Entra ID 和 Google Workspace 的分步设置说明

### 纸页
https://huggingface.co/docs/hub/paper-pages.md