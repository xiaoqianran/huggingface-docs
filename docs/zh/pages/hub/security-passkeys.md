<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 密钥

通过密钥，您可以使用指纹、面部、设备屏幕锁或硬件安全密钥（而不是密码）登录 Hugging Face Hub。它们基于 [WebAuthn](https://www.w3.org/TR/webauthn/) 标准，并且能够抵御网络钓鱼，因为密钥仅适用于为其创建的网站。

密钥可以通过两种方式使用：
- 作为**登录方法**，替换您的用户名和密码
- 作为[two-factor authentication (2FA)](./security-2fa)的**第二个因素**，在您使用密码登录后询问

## 添加密钥

1. 前往您的[Authentication settings](https://hf.co/settings/authentication)
2. 在 **密钥** 部分中，单击 **添加密钥**
3. 按照浏览器、设备或密码管理器的说明创建密钥

您可以注册多个密钥，例如每个设备一个或密码管理器中一个，硬件安全密钥上一个。通过密码管理器在您的设备之间同步的密钥标有 **已同步** 徽章。

    
    

> [!注意]
> [Managed accounts](./enterprise-advanced-sso)，通过其组织的身份提供商登录，无法添加密钥。

## 使用密码登录在 [Hub sign-in page](https://hf.co/login) 上，单击用户名或电子邮件字段。如果您有 Hugging Face 的密码，您的浏览器或密码管理器会建议您使用它。选择它并使用您的设备确认登录，无需输入密码。

使用密钥登录已经验证了您在设备上的身份，因此系统不会提示您输入额外的两步验证码。

## 使用密钥进行双因素身份验证

当[two-factor authentication](./security-2fa)打开时，您的任何密钥都可以用作使用密码登录后的第二步。要使用密钥将其打开：

1. 添加至少一个密钥
2. 前往您的[Authentication settings](https://hf.co/settings/authentication)
3. 在**双因素身份验证**部分中，单击**打开**

然后，集线器会显示恢复代码列表。下载它们并将它们保存在安全的地方。更多详情请参见[Recovery codes](./security-2fa#recovery-codes)。

使用密码登录后，系统会要求您使用其中一个密钥进行确认。如果您还设置了身份验证器应用程序，则可以选择使用它，或使用恢复代码。

关闭双因素身份验证不会删除您的密码，您仍然可以使用它们登录。

## 管理你的密钥在 [Authentication settings](https://hf.co/settings/authentication) 的 **密钥** 部分中，您可以查看每个密钥的添加时间和上次使用时间，以及：
- **重命名**密钥以更轻松地识别它
- **删除**密钥。您将无法再使用它登录。

> [!提示]
> 如果双因素身份验证已开启，并且您没有身份验证器应用程序，则需要先设置身份验证器应用程序或关闭双因素身份验证，然后才能删除最后一个密钥。

当您从集线器中删除密钥时，它可能仍存储在您的设备或密码管理器中。您也可以在那里删除它。

### 单点登录配置指南
https://huggingface.co/docs/hub/security-sso-configuration-guides.md