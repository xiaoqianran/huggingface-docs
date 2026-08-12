<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 组织中的访问控制

> [!提示]
> 您可以设置 [Single Sign-On (SSO)](./security-sso) 以便能够从组织的身份提供商映射访问控制规则。

> [!提示]
> 通过[Resource Groups](./security-resource-groups)可以实现高级且更细粒度的访问控制。
>
> 资源组功能是团队和企业计划的一部分。

组织成员可以具有五种不同的角色：`no_access`、`read`、`contributor`、`write` 或 `admin`：

- `no_access`：成员属于该组织，但无权访问其存储库或设置。与 [Resource Groups](./security-resource-groups) 一起使用仅授予对特定存储库的访问权限。

- `read`：对组织的存储库和元数据/设置（例如组织的个人资料、成员列表、API 令牌等）的只读访问权限。

- `contributor`：对用户创建的组织存储库子集的附加写入权限。即，用户可以创建存储库，然后仅修改这些存储库。这类似于 `write` 角色，但范围仅限于用户创建的存储库。

- `write`：对本组织所有存储库的写入权。用户可以在组织命名空间中创建、删除或重命名任何存储库。用户还可以从浏览器编辑器编辑和删除文件，并使用`git`推送内容。- `admin`：除了对存储库的写入权限之外，管理员成员还可以更新组织的配置文件、刷新组织的 API 令牌以及管理组织成员。

作为组织 `admin`，请转到组织设置的 **成员** 部分来管理用户的角色。要以编程方式更改角色或资源组分配，请参阅 [Programmatic User Access Control Management](./programmatic-user-access-control) 指南​​。

## 查看会员的电子邮件地址

> [!警告]
> 此功能是团队和企业计划的一部分。

您也许能够查看组织成员的电子邮件地址。电子邮件地址的可见性取决于组织的 SSO 配置或已验证的组织状态。

- 通过您组织的[verifying an email domain](./organizations-managing#organization-email-domain)，您可以查看具有匹配电子邮件域的成员的电子邮件地址。
- 如果您的组织配置了 SSO，您可以通过在 SSO 配置中设置 `Matching email domains` 查看每个组织成员的电子邮件地址

## 管理具有对我的组织的访问权限的访问令牌

参见[Tokens Management](./enterprise-tokens-management)

### 安排工作
https://huggingface.co/docs/hub/jobs-schedule.md