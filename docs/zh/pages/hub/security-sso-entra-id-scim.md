<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 如何使用 Microsoft Entra ID (Azure AD) 配置 SCIM

本指南介绍如何使用 SCIM 在 Microsoft Entra ID 和 Hugging Face 组织之间设置自动用户和组配置。

> [!警告]
> 此功能是 Enterprise 和 Enterprise Plus 计划的一部分。

## 步骤 0：确认您的组织已正确设置

> [!注意]
> 仅当您在 Hugging Face 上设置托管用户时才需要执行此步骤。

在继续之前，请确保您的组织已转换为 Hugging Face **托管组织**。托管用户的 SCIM 配置仅适用于托管组织 - 如果您的组织尚未转换，请先联系您的 Hugging Face 客户团队，然后再继续执行以下步骤。

## 步骤 1：从 Hugging Face 获取 SCIM 配置1. 登录作为组织所有者的 Hugging Face 帐户（对于托管用户，请使用托管用户转换后提供的所有者帐户）。
2. 导航到 Hugging Face 上您组织的设置页面。
3. 转到 **SSO** 选项卡，然后单击 **SCIM** 子选项卡。
4. 复制 **SCIM 租户 URL**。您将需要它来进行 Entra ID 配置。
5. 单击**生成访问令牌**。将生成一个新的 SCIM 令牌。立即复制此令牌并安全存储，因为您将无法再次看到它。

    
    

## 步骤 2：在 Microsoft Entra ID 中配置配置

1. 在 Microsoft Entra 管理中心中，导航到您的 Hugging Face 企业应用程序。
2. 在左侧菜单中，选择**配置**。
3. 单击**开始**。
4. 将**配置模式**从“手动”更改为**自动**。

## 步骤 3：输入管理员凭据

1. 在 **管理员凭据** 部分中，将 Hugging Face 中的 **SCIM 租户 URL** 粘贴到 **租户 URL** 字段中。
2. 将 Hugging Face 中的 **SCIM 令牌** 粘贴到 **Secret Token** 字段中。
3. 单击“**测试连接**”。您应该会看到一条成功通知。
4. 单击“**保存**”。## 步骤 4：配置属性映射

1. 在 **映射** 部分下，单击 **配置 Microsoft Entra ID 用户**。
2. 默认属性映射通常需要调整以实现稳健的配置。我们建议使用以下配置。您可以删除此处未列出的属性：

    | `customappsso` 属性 | Microsoft Entra ID 属性 |匹配优先级 |
    |---|---|---|
    | `userName` | `Replace([mailNickname], ".", "", "", "", "", "")` | |
    | `active` | `Switch([IsSoftDeleted], , "False", "True", "True", "False")` | |
    | `emails[type eq "work"].value` | `userPrincipalName` | |
    | `name.givenName` | `givenName` | |
    | `name.familyName` | `surname` | |
    | `externalId` | `objectId` | `1` |

3. 用户名需符合以下规则。

> [!警告]
> 
> 用户名中仅接受常规字符和`-`。
> 禁止使用`--`（双破折号）。
> `-` 无法开始或结束名称。
> 不接受纯数字名称。
> 最小长度为 2，最大长度为 42。
> 用户名在您的组织内必须是唯一的。
> 

4. 配置用户映射后，返回“配置”屏幕并单击“**配置 Microsoft Entra ID 组**”以查看组映射。组的默认设置通常就足够了。

## 步骤 5：开始配置1. 在主配置屏幕上，将**配置状态**设置为**打开**。
2. 在**设置**下，您可以将**范围**配置为“仅同步分配的用户和组”或“同步所有用户和组”。我们建议从“仅同步分配的用户和组”开始。
3. 保存您的更改。

初始同步最多可能需要 40 分钟才能开始。您可以在 **配置日志** 选项卡中监控进度。

### 分配用户和组进行配置

要控制向您的 Hugging Face 组织配置哪些用户和组，您需要将它们分配给 Microsoft Entra ID 中的 Hugging Face 企业应用程序。这是在应用程序的 **用户和组** 选项卡中完成的。

1. 导航到 Microsoft Entra 管理中心中的 Hugging Face 企业应用程序。
2. 转到 **用户和组** 选项卡。
3. 单击“**添加用户/组**”。
4. 选择您要配置的用户和组，然后单击“**分配**”。

    
    

如果您将**范围**设置为“仅同步分配的用户和组”，则只有您在此处分配的用户和组才会配置到 Hugging Face。> [!提示]
> Active Directory 计划注意事项
> 
> 通过免费、Office 365 和高级 P1/P2 计划，您可以将单个用户分配给应用程序进行配置。
> 对于高级 P1/P2 计划，您还可以分配组。这是大规模管理访问的推荐方法，因为您可以在 AD 中管理组成员身份，并且更改将自动反映在 Hugging Face 中。
> 

## 步骤 6：验证 Hugging Face 中的配置

同步完成后，导航回您的 Hugging Face 组织设置：
- 已配置的用户将显示在“**用户管理**”选项卡中。
- 配置的组将显示在 **SCIM** 选项卡中的 **SCIM 组** 下。然后可以将这些组分配给[Resource Groups](./security-resource-groups)以进行细粒度的访问控制。

## 步骤 7：将 SCIM 组链接到 Hugging Face 资源组

从 Entra ID 配置您的组后，您可以将它们链接到 Hugging Face 资源组以大规模管理权限。这允许 SCIM 组的所有成员自动接收资源集合的特定角色（例如读取或写入）。> [!注意]
> 在链接之前，请确保您要链接的资源组是**空**（没有现有成员）并且**未**启用自动加入。这两个条件都是必需的 - 否则链接将失败。

1. 在 Hugging Face 组织设置中，导航到 **SSO** -> **SCIM** 选项卡。您将在 **SCIM 组** 下看到您配置的组的列表。

    
    

2. 找到您要配置的组，然后单击其行中的 **链接资源组**。
3. 将出现一个对话框。单击“**链接资源组**”。
4. 从下拉菜单中，选择要链接的 **资源组** 以及要授予 SCIM 组成员的 **角色分配**。
5. 单击 **链接到 SCIM 组** 并保存映射。

链接后，资源组将变为 **SCIM 管理**：SCIM 组中已有的任何成员都会立即添加到资源组（回填），并且 Entra ID 中所有未来的成员资格更改都会自动反映。通过 Hub UI 或 API 对资源组进行手动成员资格编辑将被阻止。

### 代理
https://huggingface.co/docs/hub/agents-overview.md