<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 用户管理

> [!警告]
> 此功能是团队和企业计划的一部分。

启用了 SSO 的组织可以使用以下功能。有关每种模式的详细信息，请参阅 [Basic SSO](./security-sso-basic) 和 [Managed SSO](./enterprise-advanced-sso)。

	
	

## 会话超时

该值设置组织成员的会话持续时间。

此后，系统将提示成员重新向您的身份提供商进行身份验证，以访问组织的资源。

默认值为 7 天。

## 角色映射

启用后，角色映射允许您根据身份提供商提供的数据动态地将 [roles](./organizations-security#access-control-in-organizations) 分配给组织成员。

此部分允许您定义从 IdP 的用户配置文件数据到 Hugging Face 中分配的角色的映射。

- **IdP 角色属性路径**

  用户 IdP 配置文件数据中属性的 JSON 路径。
  它支持点表示法（例如`user.role`或`groups`）。
  对于 SAML，这可以是 URI（例如 `http://schemas.microsoft.com/ws/2008/06/identity/claims/role`）。

- **角色映射**

  从 IdP 属性值到 Hugging Face 组织中分配的角色的映射。

  可用角色有 `admin`、`write`、`contributor` 和 `read`。更多详情请参见[roles documentation](./organizations-security#access-control-in-organizations)。> [!警告]
> 您必须在配置中映射至少一个 `admin` 角色。

如果 IdP 响应中的属性包含与多个映射匹配的多个值（例如组列表），则为用户分配**最高特权的匹配角色**。角色层次结构（从最低权限到最高权限）为 `read` < ⟦T9⟧ < ⟦T10⟧ < ⟦T11⟧.

If there is no match, the role is determined as follows:

- If the user was invited to the organization with a specific role, that invitation role is used as a fallback.
- Otherwise, the user is assigned the default role for your organization. The default role can be customized in the ⟦T12⟧ section of the organization's settings.

> [!NOTE]
> 当邀请其角色将由 SSO 角色映射控制的用户时，仅当用户尚未在 SSO 提供商中分配角色时，在邀请模式中选择的角色才适用。

每次登录时都会执行角色同步。

## 资源组映射

启用后，资源组映射允许您根据身份提供商提供的数据动态地将成员分配到组织中的[resource groups](./enterprise-resource-groups)。

	
	

- **IdP 属性路径**

  用户 IdP 配置文件数据中属性的 JSON 路径。与角色映射类似，这支持 SAML 的点表示法或 URI。

- **资源组映射**

  从 IdP 属性值到 Hugging Face 组织中的资源组的映射。您可以为每个资源组映射分配特定角色（`admin`、`write`、`contributor`、`read`、`inherit`）。与角色映射不同，**资源组映射是附加的**。如果用户匹配多个映射（例如，他们属于 IdP 中映射到不同资源组的多个组），他们将被添加到**所有**匹配的资源组。

如果不匹配，则不会将用户分配到任何资源组。

> [!警告]
> 仍可以从 Hub UI 或 API 手动编辑映射所针对的资源组，但这些编辑会在用户下次 SSO 登录时被覆盖：角色更改将恢复到映射的角色，并且会重新添加手动删除的用户。要永久更改映射用户的访问权限，请更新您的 IdP 数据或映射配置。

## 匹配电子邮件域

> [!注意]
> 此功能仅与[Basic SSO](./security-sso-basic)相关。对于 [Managed SSO](./enterprise-advanced-sso)，用户帐户完全由组织管理，因此电子邮件域匹配不适用。

启用后，“匹配电子邮件域”仅允许组织成员在您的身份提供商提供的电子邮件与 Hugging Face 上的其中一封电子邮件匹配的情况下完成 SSO。

要添加电子邮件域，请填写“匹配电子邮件域”字段，单击键盘上的 Enter 键，然后保存。

## 外部合作者外部协作者是组织外部的用户，他们需要在不通过单点登录 (SSO) 的情况下访问其部分资源。这通常意味着它们不在您的身份提供商 (IdP) 中。它们可在付费计划中使用，并在组织的成员列表中显示为“外部协作者”徽章。它们确实计入您组织中的付费席位数量。

您可以通过以下方式添加它们：

- **邀请他们加入组织**：在组织设置的 **成员** 页面上，发送邀请时选中 **外部协作者**。
- **转换现有成员**：在**成员**页面上，打开成员的菜单并选择**转换为外部协作者**。使用 **转换为成员** 将其切换回来。如果您的组织使用 SSO，他们随后需要通过它登录才能保留访问权限。你也可以这样做[via the API](./programmatic-user-access-control#change-member-account-type-via-api)。
- **共享资源组或存储库**：如果启用 [external invitations](./security-resource-groups#external-invitations)，资源组管理员可以直接邀请外部人员加入资源组，包括从存储库设置页面上的 **与外部协作者共享**。这些用户以 `no_access` 组织角色加入，并且仅获得该资源组。外部协作者仍需通过组织的访问控制 ([Resource Groups](./enterprise-resource-groups))。

> [!注意]
> 使用 [Managed SSO](./enterprise-advanced-sso)，内部成员通过您的身份提供商 (SCIM) 进行配置，因此受邀加入组织的用户只能添加为外部协作者。

### 查询数据集
https://huggingface.co/docs/hub/datasets-duckdb-select.md