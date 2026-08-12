<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 审核日志

> [!警告]
> 此功能是团队和企业计划的一部分。

审核日志使组织管理员能够轻松查看成员采取的操作，包括组织成员身份、存储库设置和计费更改。

  <img
    class="block dark:hidden m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/audit-logs.png"
    alt="screenshot of Hugging Face Audit Logs feature"
  />
  <img
    class="hidden dark:block m-0!"
    src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/dark-audit-logs.png"
    alt="screenshot of Hugging Face Audit Logs feature"
  />

## 访问审核日志

可通过您的组织设置访问审核日志。每个日志条目包括：

- 谁执行了该动作
- 采取了什么类型的行动
- 变更的描述
- 位置和匿名 IP 地址
- 行动的日期和时间

您还可以将完整的审核日志下载为 JSON 文件以进行进一步分析。

## 跟踪哪些事件？

每个操作都有一个 **事件名称**，采用 `scope.action` 格式（例如 `repo.create`、`collection.delete`）。这是每个日志条目和导出的 JSON 中的 `type` 字段 - 在搜索或过滤日志时使用它。

### 组织管理与安全

  > [!提示]
  > 下面的 **设置更改** 事件仅适用于 2026 年 6 月 16 日之后采取的操作。该日期之前的事件使用唯一的 `org.update_settings` 事件类型。任何过滤或解析 2026 年 6 月 16 日之前创建的组织的事件 `type` 字段的集成都应该处理这两者。- **核心组织变更** — 创建、删除、恢复和重命名。
  - **活动：** `org.create`、`org.delete`、`org.restore`、`org.rename`
- **设置更改** — 组织设置的更新会记录为精细的 `org.settings.*` 事件，因此您可以准确地查明更改的设置。
  - 常规设置 — 配置文件、存储区域、资源组和发布者门控。
  - **活动：** `org.settings.profile`、`org.settings.regions`、`org.settings.resource_groups`、`org.settings.publisher_gating`
  - 推理提供程序 — 提供程序配置、API 密钥添加/删除以及使用设置。
  - **活动：** `org.settings.inference_providers`、`org.settings.inference_providers.keys.add`、`org.settings.inference_providers.keys.remove`、`org.settings.inference_providers.usage`
  - SSO — SSO 配置更新，以及启用或禁用 SSO。
  - **活动：** `org.settings.sso`、`org.settings.sso.enable`、`org.settings.sso.disable`
  - 安全性 — 默认存储库可见性，以及启用或禁用 2FA 实施、自动加入和成员隐私。
  - **活动：** `org.settings.security.repo_visibility`、`org.settings.security.2fa.enable`、`org.settings.security.2fa.disable`、`org.settings.security.auto_join.enable`、`org.settings.security.auto_join.disable`、`org.settings.security.members_privacy.enable`、`org.settings.security.members_privacy.disable`
  - 网络 — 网络配置更新，以及启用或禁用 IP 访问限制和身份验证强制。
  - **活动：** `org.settings.network`、`org.settings.network.ip_restriction.enable`、`org.settings.network.ip_restriction.disable`、`org.settings.network.auth_enforcement.enable`、`org.settings.network.auth_enforcement.disable`- **安全管理**
  - 组织 API 令牌轮换。
  - **活动：** `org.rotate_token`
  - 令牌批准系统 — 启用或禁用策略、授权请求、批准、拒绝和撤销。
  - **活动：** `org.token_approval.enabled`、`org.token_approval.disabled`、`org.token_approval.authorization_request`、`org.token_approval.authorization_request.authorized`、`org.token_approval.authorization_request.revoked`、`org.token_approval.authorization_request.denied`
  - SSO — 通过 SSO 登录和加入。
  - **活动：** `org.sso_login`、`org.sso_join`
- **加入设置** — 基于域的访问和自动加入配置。
  - **活动：** `org.update_join_settings`

### 会员资格和访问控制

- **成员生命周期** — 添加和删除成员、角色更改以及成员离开组织。
  - **活动：** `org.add_user`、`org.remove_user`、`org.change_role`、`org.leave`
- **邀请** — 通过电子邮件发送邀请、邀请链接以及用户接受邀请。
  - **活动：** `org.invite_user`、`org.invite.accept`、`org.invite.email`
- **自动加入** — 通过经过验证的电子邮件域或“请求访问”加入。
  - **活动：** `org.join.from_domain`、`org.join.automatic`

### 内容和资源管理- **存储库管理** — 创建、删除、移动、禁用/重新启用、重复设置、DOI 删除、资源组分配和常规存储库设置（可见性、门控、讨论等）。还有LFS文件删除。
  - **活动：** `repo.create`、`repo.delete`、`repo.move`、`repo.disable`、`repo.removeDisable`、`repo.duplication`、`repo.delete_doi`、`repo.update_resource_group`、`repo.update_settings`、`repo.delete_lfs_file`
- **集合** — 创建和删除集合。
  - **活动：** `collection.create`、`collection.delete`
- **存储库安全性** — 秘密和变量（单独和批量添加/更新/删除）。
  - **事件（秘密）：** `repo.add_secret`、`repo.update_secret`、`repo.remove_secret`、`repo.add_secrets`、`repo.remove_secrets`
  - **事件（变量）：** `repo.add_variable`、`repo.update_variable`、`repo.remove_variable`、`repo.add_variables`、`repo.remove_variables`
- **空间配置** — 存储层更改、硬件（风格）更新和睡眠时间调整。
  - **活动：** `spaces.add_storage`、`spaces.remove_storage`、`spaces.update_hardware`、`spaces.update_sleep_time`

### 资源组

- **资源组管理** — 创建、删除和设置更改。
  - **活动：** `resource_group.create`、`resource_group.delete`、`resource_group.settings`
- **资源组成员** — 添加和删除用户以及角色更改。
  - **活动：** `resource_group.add_users`、`resource_group.remove_users`、`resource_group.change_role`

### 作业和预定作业- **工作** — 工作岗位的创造（例如在空间上）和取消。
  - **活动：** `jobs.create`、`jobs.cancel`
- **计划作业** — 创建、删除、恢复、暂停和触发运行。
  - **活动：** `scheduled_job.create`、`scheduled_job.delete`、`scheduled_job.resume`、`scheduled_job.suspend`、`scheduled_job.run`

### 计费和云集成

- **付款和客户** — 付款方式更新、附加和删除；客户帐户创建。
  - **活动：** `billing.update_payment_method`、`billing.create_customer`、`billing.remove_payment_method`
- **云市场** — AWS 和 GCP 市场链接/取消链接以及市场批准。
  - **活动：** `billing.aws_add`、`billing.aws_remove`、`billing.gcp_add`、`billing.gcp_remove`、`billing.marketplace_approve`
- **订阅** — 启动、续订、取消、重新激活和更新订阅（包括计划和合同详细信息）。
  - **活动：** `billing.start_subscription`、`billing.renew_subscription`、`billing.cancel_subscription`、`billing.un_cancel_subscription`、`billing.update_subscription`、`billing.update_subscription_plan`、`billing.update_subscription_contract_details`

## 事件参考

上面的列表涵盖了审核日志 UI 和导出中显示的每种事件类型。事件名称遵循 `scope.action` 模式；范围包括 `org`、`repo`、`collection`、`spaces`、`resource_group`、`jobs`、`scheduled_job` 和 `billing`。导出操作本身记录为 `org.audit_log.export`，但该事件不包含在默认审核日志视图中。

### 抱脸登录
https://huggingface.co/docs/hub/oauth.md