<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 迁移到托管 SSO

> [!警告]
> 此功能是 Enterprise Plus 计划的一部分。

本指南适用于将现有组织从标准组织转换为 [Managed SSO](./enterprise-advanced-sso) 组织的组织管理员。

开始之前要了解的最重要的一点是：**会员现有的个人 Hugging Face 帐户不会转换为管理帐户**。每个成员都会获得一个全新的托管帐户，由您的身份提供商 (IdP) 提供。在配置成员之前，他们必须释放其个人帐户上的工作电子邮件地址。

在本指南中，我们使用 `my-org` 作为示例组织名称。

## 先决条件

[SSO](./security-sso-basic) 必须已在您的组织中配置、测试并启用。迁移会重用您现有的 OIDC 或 SAML 应用程序。

## 迁移步骤1. **让您的会员做好准备。** 与您的会员一起完成[pre-migration checklist](#before-you-migrate)。
2. **联系 Hugging Face。** 与 Hugging Face 团队联系并确认切换。我们转换组织并向您发送 `my-org:owner` Break-glass 帐户的凭据。
3. **让成员释放他们的工作电子邮件。** 请参阅[Freeing up work email addresses](#freeing-up-work-email-addresses)。在启用配置之前执行此操作：一旦打开 SCIM，就会配置已分配给 SSO 应用程序的用户。 Hugging Face 团队可以向您发送其个人帐户仍保留工作电子邮件地址的成员列表，以便您可以跟进他们。
4. **在 IdP 中设置 SCIM。** 在现有 OIDC 或 SAML 应用程序中使用组织的 **SSO → SCIM** 设置中的 SCIM 租户 URL 和令牌，然后按照 IdP 的指南进行操作。
   - **Okta**：遵循[Okta guide](./security-sso-okta-scim)，包括[Step 3](./security-sso-okta-scim#step-3-check-your-saml-settings)。名称 ID 必须是 Okta 用户 ID。
   - **Microsoft Entra ID**：遵循[Entra ID guide](./security-sso-entra-id-scim)。 `userName` 必须位于 [username rules](./security-sso-entra-id-scim#step-4-configure-attribute-mappings) 之后，因此诸如 `jdoe@example.com` 之类的电子邮件样式值将被拒绝。
   - **其他 IdP**：请参阅[User Provisioning (SCIM)](./enterprise-scim)，具有相同的用户名规则。
5. **在 IdP 中分配用户和组。** 托管帐户是在配置用户时创建的。6. **配置每个成员的管理帐户后，立即从组织中删除剩余的个人帐户**。为了安全起见，此步骤是必需的，从[personal accounts keep access](#existing-personal-accounts-in-the-organization)开始直到它们被删除。删除它们也会将它们从组织的资源组中删除。
7. **（可选）将组映射到资源组和角色。** [Link your SCIM groups to Resource Groups](./enterprise-scim#linking-a-scim-group-to-a-resource-group)，并设置 [Role Mapping](./security-sso-user-management#role-mapping) 以从 IdP 分配组织角色。仅当资源组没有成员且自动加入已关闭时，才能将其链接到 SCIM 组。如果打开自动加入，新配置的管理帐户也会添加到组中，因此在配置用户之前将其关闭，或者在链接之前删除任何剩余成员。

## 有什么变化

### 转换的作用- **转换由 Hugging Face 团队完成。** 没有自助开关。一旦您提出请求，Hugging Face 团队就会将您的组织切换到托管模式。
- **您组织的内容和配置保持原样。** 存储库、设置、[Resource Groups](./enterprise-resource-groups) 和您现有的 SSO 配置（OIDC 或 SAML 应用程序）将保留。您保留相同的 IdP 应用程序，但某些设置可能需要针对 SCIM 进行调整（请参阅[step 4](#migration-steps)）。
- **创建了一个打破玻璃的管理员帐户**，名为 `my-org:owner`。它使用用户名和密码（不通过 SSO）登录，并且是组织的管理员。 Hugging Face 团队与您分享其凭据。安全地存储它们：如果您的 IdP 不可用或配置错误，此帐户可让您访问您的组织。
- **SSO 不再是可选的。** 它仍然可以完全禁用。

> [!警告]
> 在托管组织上禁用 SSO 会锁定每个托管用户，因为托管帐户没有密码。

### 管理帐户是新帐户- 现有的个人 Hugging Face 帐户**不会**转换、合并或链接到管理帐户中。每个成员都会获得一个新帐户，该帐户是在 IdP 中分配用户时由 IdP 通过 [SCIM](./enterprise-scim) 创建的。
- 托管用户名以组织名称为前缀：`<org-name>:<username>`，例如`my-org:jdoe`。
- 托管用户仅通过您的 IdP 登录。他们没有拥抱面部密码，“忘记密码”对他们不起作用。
- 托管帐户受 [restrictions on managed accounts](./enterprise-advanced-sso#restrictions-on-managed-accounts) 的约束。

### 会员资格

- 无法再通过邀请添加普通会员。内部成员必须通过 SCIM 进行配置。
- 邀请仅适用于[external collaborators](./security-sso-user-management#external-collaborators)。

### 组织中现有的个人帐户

转换后仍属于组织成员的个人帐户将保留其角色。他们的访问令牌、已授权的 OAuth 应用程序以及通过 SSH 的 git 继续工作，因为这些不会针对 SSO 进行检查。他们无法再启动组织的新 SSO 登录：

- 如果用户使用其个人帐户登录，或者该帐户已链接到您的 IdP，他们会看到：
  `You cannot login with SSO with this account because it's not managed by this organization. Contact your organization administrators.`
- 否则，他们会看到`User not found`。

## 迁移之前在请求转换之前，请与您的会员一起查看此清单：

- **个人存储库**：如果成员在个人命名空间中拥有存储库，并且想要继续使用新的管理帐户访问这些存储库，则应在切换之前向组织发送[transfer them](./repositories-settings#renaming-or-transferring-a-repo)。托管帐户无法拥有个人命名空间中的内容。成员仍然可以通过其个人（非托管）帐户访问其个人命名空间下的任何存储库。
- **令牌和自动化**：个人帐户中的令牌保留对组织的访问权限，直到帐户从组织中删除为止。将脚本、CI 作业和管道迁移到[service accounts](./enterprise-service-accounts)，或迁移到切换后从托管帐户创建的令牌，然后删除旧帐户。
- **其他组织**：管理帐户无法加入其他组织。需要继续在其他组织工作的会员应保留单独的个人账户（见下文[option 2](#option-2-keep-the-personal-account)）。
- **工作电子邮件地址**：每个成员在配置之前都必须释放其工作电子邮件。参见[Freeing up work email addresses](#freeing-up-work-email-addresses)。

## 释放工作电子邮件地址一个电子邮件地址只能属于一个 Hugging Face 帐户。这适用于帐户的**所有**电子邮件地址（主要和辅助）。匹配不区分大小写。

当 SCIM 尝试使用个人帐户已使用的工作电子邮件创建托管帐户时，配置会失败并出现 `409` 错误：

```
The email address (jdoe@example.com) is already associated with another Hugging Face account. Please delete the account.
```

您的 IdP 将其显示为该用户的配置错误。

为了避免这种情况，每个成员都必须在配置之前**使用以下两个选项之一释放其工作电子邮件地址。

> [!提示]
> **我应该选择哪个选项？**
> - 如果您仅使用 Hugging Face 帐户在该组织中工作，请选择**选项 1**：删除个人帐户。
> - 如果您还将其用于开源工作、个人项目或其他组织，请选择**选项 2**：保留个人帐户并从中删除工作电子邮件。

### 选项 1：删除个人帐户

从您的[account settings](https://huggingface.co/settings/account)删除您的个人帐户。这将永久删除您个人命名空间中的所有存储库，并将您从每个组织中删除。

> [!警告]
> 删除帐户是永久性的。首先将您想要保留的任何存储库转移到组织。如果您是另一个组织的唯一管理员，则删除将被阻止。首先向该组织添加或删除另一个管理员。正在迁移的组织不是问题，因为 `my-org:owner` 也是管理员。

### 选项 2：保留个人帐户

1. 在您的[account settings](https://huggingface.co/settings/account)中，添加个人电子邮件地址并进行验证。
2. 将个人电子邮件地址设为您的主要电子邮件地址。
3. **完全删除工作电子邮件地址。** 将其设为辅助电子邮件是不够的。

将您的个人帐户保留在组织中，直到配置您的管理帐户为止。然后您的管理员将其删除。

> [!注意]
> 如果您的组织已启用托管 SSO 并具有 [verified email domain](./organizations-managing#organization-domain-name)，则使用该域上的电子邮件地址登录会将您重定向到组织的 SSO 页面。要登录您的个人帐户，请使用您的用户名。

一旦工作电子邮件空闲，管理员可以在 IdP 中重试为该用户进行配置，或者等待下一次同步来获取它。

## 迁移后

用户生命周期由您的 IdP 管理：- **停用** IdP 中的用户会将其从组织中删除。如果稍后重新激活同一用户，则会重复使用其现有的管理帐户。
- 通过 SCIM **删除**用户会永久删除其管理帐户。参见[Deprovisioning](./enterprise-scim#deprovisioning)。

### 秘密扫描
https://huggingface.co/docs/hub/security-secrets.md