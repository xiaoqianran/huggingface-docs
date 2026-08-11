<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 用户管理

> [!警告]
> 此功能是团队和企业计划的一部分。

启用了 SSO 的组织可以使用以下功能。有关每种模式的详细信息，请参阅[Basic SSO](./security-sso-basic)和[Managed SSO](./enterprise-advanced-sso)。

	
	

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

  从 IdP 属性值到 Hugging Face 组织中的资源组的映射。您可以为每个资源组映射分配特定角色（`admin`、`write`、`contributor`、`read`）。与角色映射不同，**资源组映射是附加的**。如果用户匹配多个映射（例如，他们属于 IdP 中映射到不同资源组的多个组），他们将被添加到**所有**匹配的资源组。

如果不匹配，则不会将用户分配到任何资源组。

## 匹配电子邮件域

> [!注意]
> 此功能仅与[Basic SSO](./security-sso-basic)相关。对于 [Managed SSO](./enterprise-advanced-sso)，用户帐户完全由组织管理，因此电子邮件域匹配不适用。

启用后，“匹配电子邮件域”仅允许组织成员在您的身份提供商提供的电子邮件与 Hugging Face 上的其中一封电子邮件匹配的情况下完成 SSO。

要添加电子邮件域，请填写“匹配电子邮件域”字段，单击键盘上的 Enter 键，然后保存。

## 外部合作者

这使得组织内的某些用户无需完成单点登录 (SSO) 流程即可访问资源。当您与不属于组织身份提供商 (IdP) 但需要访问特定资源的外部方合作时，这会很有帮助。要将用户添加为“外部协作者”，请访问组织设置中的 `SSO/Users Management` 部分。添加后，这些用户将不需要执行 SSO 流程。

但是，它们仍然受到您组织的访问控制 ([Resource Groups](./enterprise-resource-groups))。
仔细管理他们的访问权限对于维护组织的数据安全至关重要。

### 空间配置参考
https://huggingface.co/docs/hub/spaces-config-reference.md