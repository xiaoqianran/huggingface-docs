<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 单点登录配置指南

> [!警告]
> 此功能是团队和企业计划的一部分。

这些指南可帮助您使用 [Basic SSO](./security-sso-basic) 的身份提供商配置 SAML 2.0 和 OpenID Connect (OIDC)。 Hugging Face Hub 可以与任何 SAML 或 OIDC 兼容的身份提供商一起使用。

> [!注意]
> 如果您想要设置 [Managed SSO](./enterprise-advanced-sso)，配置是与 Hugging Face 团队合作完成的。请联系我们开始使用。

> [!提示]
> **OIDC 令牌端点身份验证方法。** 交换授权代码时，Hugging Face 使用 `client_secret_post` 方法对身份提供商的令牌端点进行身份验证。如果您的 IdP 应用程序配置为需要不同的方法（例如 `client_secret_basic`），则登录将失败并出现 `invalid_client` 错误。要解决此问题，请将 IdP 应用程序的令牌端点身份验证方法设置为 `client_secret_post`。

## 奥克塔

- [How to configure OIDC with Okta](./security-sso-okta-oidc)
- [How to configure SAML with Okta](./security-sso-okta-saml)
- [How to configure SCIM with Okta](./security-sso-okta-scim)

## Microsoft Entra ID (Azure AD)

- [How to configure SAML with Entra ID](./security-sso-azure-saml)
- [How to configure OIDC with Entra ID](./security-sso-azure-oidc)
- [How to configure SCIM with Entra ID](./security-sso-entra-id-scim)

## Google 工作区

- [How to configure SAML with Google Workspace](./security-sso-google-saml)
- [How to configure OIDC with Google Workspace](./security-sso-google-oidc)

### 下载模型
https://huggingface.co/docs/hub/models-downloading.md