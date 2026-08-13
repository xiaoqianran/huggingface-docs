<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 管理组织

## 创建组织

访问 [New Organization](https://hf.co/organizations/new) 表单来创建组织。

## 管理会员

通过访问 **组织设置** 并单击 **成员** 选项卡，可以将新成员添加到组织中。在那里，您将能够生成邀请链接、单独添加成员或批量发送电子邮件邀请。如果启用了**允许从组织页面加入的请求**设置，您还可以在**成员**页面上批准或拒绝任何待处理的请求。

您还可以在此页面上撤销用户的成员资格或更改他们的角色。

### 通过 SSO 邀请成员如果您的组织启用了 [Basic SSO](./security-sso-basic)，则可以复制直接加入链接并与新成员共享。此 SSO 加入链接在 **SSO** 和 **成员** 设置选项卡中均可用。由于启用了 SSO 的组织无法使用经典邀请链接，因此 SSO 加入链接是邀请团队成员加入组织的主要方法。只需单击复制按钮即可将链接复制到剪贴板并与您想要邀请的成员共享。当收件人单击共享链接时，他们将能够通过 SSO 进行身份验证并直接加入您的组织。

使用[Managed SSO](./enterprise-advanced-sso)的组织通过[SCIM](./enterprise-scim)直接通过其身份提供商提供用户。

## 组织域名

在组织设置的**帐户**选项卡下，您可以设置**组织电子邮件域**。指定域将允许 Hugging Face Hub 上具有匹配电子邮件地址的任何用户加入您的组织。

## 离开组织

用户可以离开组织，访问其 [organization settings](https://huggingface.co/settings/organizations) 并单击他们想要离开的组织旁边的 **离开组织**。组织管理员始终可以如上所述删除用户。### 使用 CircleCI 工作流程管理空间
https://huggingface.co/docs/hub/spaces-circleci.md