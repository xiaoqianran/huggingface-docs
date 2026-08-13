<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 用户配置 (SCIM)

> [!警告]
> 此功能是 Enterprise 和 Enterprise Plus 计划的一部分。

SCIM（跨域身份管理系统）是自动化用户配置的标准。它允许您将身份提供商 (IdP) 连接到 Hugging Face 以管理组织的成员。

SCIM 的工作方式有所不同，具体取决于您的 SSO 模型。详细比较请参见[SSO overview](./enterprise-sso#user-provisioning-scim)。

## 基本 SSO：基于邀请的配置

通过 [Basic SSO](./security-sso-basic)（企业计划），SCIM 可以自动**现有 Hugging Face 用户加入您的组织。

- 用户**必须已经拥有 Hugging Face 帐户**，然后才能通过 SCIM 进行配置
- 当您的 IdP 配置用户时，Hugging Face 会向他们发送一封 **邀请电子邮件** 以加入组织
- 用户必须**接受邀请**才能成为会员 - 配置不会授予立即访问权限
- SCIM **无法修改**用户个人资料信息（姓名、电子邮件、用户名）——用户保留对其 Hugging Face 帐户的完全控制权
- 当用户在您的 IdP 中取消配置时，他们的邀请将被停用，并且他们对组织的访问权限将被撤销

## 托管 SSO：全生命周期配置通过 [Managed SSO](./enterprise-advanced-sso)（企业 Plus 计划），SCIM 管理 Hugging Face 上的**整个用户生命周期**。

- SCIM **在配置用户时创建一个新的 Hugging Face 帐户** — 不需要预先存在的帐户
- 用户将作为成员**立即添加**到组织，无需邀请步骤
- SCIM **可以在您的 IdP 发生更改时更新**用户个人资料信息（姓名、电子邮件、用户名）
- 当用户在您的 IdP 中取消配置时，他们的 Hugging Face 帐户将被停用，并且他们的访问权限将被撤销

## 如何启用 SCIM

要启用 SCIM，请转到组织的设置，导航到 **SSO** 选项卡，然后选择 **SCIM** 子选项卡。

您将找到 **SCIM 租户 URL** 和用于生成 **SCIM 令牌**的按钮。您将需要这两个来配置您的 IdP。 SCIM 令牌是一个秘密，应安全地存储在 IdP 的配置中。

    
    

在 IdP 中启用 SCIM 后，配置的用户将显示在组织设置的 **用户管理** 选项卡中，配置的组将显示在 **SCIM** 选项卡中。

## 组配置除了用户配置之外，SCIM 还支持**组配置**。从 IdP 推送的组将作为 SCIM 组存储在 Hugging Face 上，并且可以从组织设置中的 **SCIM** 选项卡链接到 [Resource Groups](./enterprise-resource-groups)。

### 将 SCIM 组链接到资源组

要链接 SCIM 组，请转至您组织的 **SSO → SCIM** 选项卡。表中列出了配置的组。在 **资源组** 列中，每个组显示 **链接资源组** 按钮（如果尚不存在链接）或当前链接的资源组的数量（例如“2 个资源组”）。单击任一会打开一个模式，您可以在其中添加一个或多个资源组，每个资源组都有自己的角色分配。您还可以更改或删除同一模式中的现有链接。

链接之前，请确保满足以下条件：

- 资源组必须**没有现有成员**。不允许链接到非空资源组。
- 资源组必须**未启用自动加入**。自动加入（自动将每个新组织成员添加到 RG）与 SCIM 管理是互斥的。链接前禁用 RG 上的自动加入。一个 SCIM 组可以链接到多个资源组，每个资源组都有自己的角色。

### 链接后会发生什么

SCIM 组链接到资源组后：

- **回填**：SCIM 组中已有的任何成员都会立即添加到已配置角色的资源组中。
- **持续同步**：您的 IdP 中的成员资格更改会自动反映：
  - 当用户被**添加**到您的 IdP 中的组时，他们将被添加到所有链接的资源组中。
  - 当用户从 IdP 的组中**删除**时，他们将从所有链接的资源组中删除，用户通过其他 SCIM 组链接到的资源组除外。对于这些用户，用户的角色将更新为其他 SCIM 组授予的“最高”角色。
  - 当 IdP 中的 SCIM 组被**删除**时，其所有成员都会从链接的资源组中删除，通过其他 SCIM 组属于这些资源组的用户除外。对于每个资源组，用户的角色都会更新为其他 SCIM 组授予的“最高”角色。
- **角色更改**：如果您更新链接上的角色，则该资源组中所有当前组成员的角色都会立即更新。### SCIM 管理的资源组

链接到 SCIM 组的资源组被视为 **SCIM 管理**。 IdP 是其普通用户的唯一事实来源。结果：

- 对于普通用户来说，通过 Hub UI 或 API 进行的手动成员资格更改被**阻止** - 任何在 SCIM 管理的资源组上添加、删除或更改普通用户角色的尝试都将返回 `403` 错误。
- [Service accounts](./enterprise-service-accounts) 是例外：由于它们从未通过 SCIM 进行配置，因此您仍然可以在 SCIM 管理的资源组上手动添加、删除或更改服务帐户的角色。
- 自动加入 **无法在 SCIM 管理的资源组上启用**。要重新启用自动加入，请首先删除 SCIM 链接。

在资源组设置中，SCIM 管理的资源组显示 **SCIM 同步** 徽章，并且成员资格表单指出只能手动添加或删除服务帐户。

基本 SSO 和托管 SSO 的组配置工作方式相同。

## 支持的用户属性

Hugging Face SCIM 端点支持以下用户属性：|属性|描述 |基本单点登录 |托管单点登录 |
| --- | --- | --- | --- |
| `userName` |拥抱脸用户名|只读 |读/写|
| `name.givenName` |名字 |只读 |读/写|
| `name.familyName` |姓氏 |只读 |读/写|
| `emails[type eq "work"].value` |电子邮件地址 |只读 |读/写|
| `externalId` | IdP 分配的标识符 |读/写|读/写|
| `active` |用户是否为活跃会员 |读/写|读/写|

使用基本 SSO，只有 `active` 和 `externalId` 可以通过 SCIM 进行修改 — 所有其他属性均由用户在其 Hugging Face 帐户上控制。

对于组配置，支持的属性为 `displayName`、`members` 和 `externalId`。

## 取消配置

取消配置行为取决于删除用户的方式以及您使用的 SSO 模型。

**将 `active` 设置为 `false`**（软取消配置）：

- 用户失去对组织的访问权限
- 使用基本 SSO：邀请已停用
- 使用托管 SSO：用户将从组织中删除，但保留其帐户和内容 — 通过将 `active` 设置回 `true`，这是**可逆的**

**通过 SCIM 删除用户**（硬取消配置）：- 使用基本 SSO：用户将从组织及其所有资源组中删除。他们的 Hugging Face 帐户和个人内容**不受影响** - 他们只是失去了您组织中的成员资格。
- 使用托管 SSO：用户的 Hugging Face 帐户及其创建的所有内容将被**永久删除**。此操作**不可逆转**。

## 支持的身份提供商

我们支持 SCIM 与任何实施 SCIM 2.0 协议的 IdP。我们为一些最受欢迎的提供商提供了具体指南：
- [How to configure SCIM with Microsoft Entra ID](./security-sso-entra-id-scim)
- [How to configure SCIM with Okta](./security-sso-okta-scim)

### 集线器上的存储区域
https://huggingface.co/docs/hub/storage-regions.md