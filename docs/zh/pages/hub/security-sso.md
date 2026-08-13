<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 单点登录 (SSO)

> [!警告]
> 此功能是团队和企业计划的一部分。

Hugging Face 支持单点登录 (SSO)，让组织可以通过自己的身份提供商 (IdP) 管理用户身份验证。支持 SAML 2.0 和 OpenID Connect (OIDC) 协议。

有两种 SSO 模型可用，具体取决于您的计划和需求。详细比较请参见[SSO overview](./enterprise-sso)。

- **[Basic SSO](./security-sso-basic)** — 适用于团队和企业计划。在标准 Hugging Face 登录之上添加访问控制层，以保护组织的资源。
- **[Managed SSO](./enterprise-advanced-sso)** — 适用于 Enterprise Plus 计划。完全取代 Hugging Face 登录，让您的组织能够完全控制用户帐户和访问权限。需要与 Hugging Face 团队进行设置 - 请联系我们开始。

## 进一步阅读

- [User Management](./security-sso-user-management) — 角色映射、资源组映射、会话超时等
- [Configuration Guides](./security-sso-configuration-guides) — Okta、Microsoft Entra ID 和 Google Workspace 的分步设置说明
- [User Provisioning (SCIM)](./enterprise-scim) — 来自身份提供商的自动用户配置

### 查询数据集
https://huggingface.co/docs/hub/datasets-duckdb-select.md