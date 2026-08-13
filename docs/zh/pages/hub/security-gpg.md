<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 GPG 签署提交

`git` 有一个身份验证层来控制谁可以将提交推送到存储库，但它不会验证实际的提交作者。

换句话说，您可以以 `Elon Musk <elon@tesla.com>` 身份提交更改，将它们推送到您首选的 `git` 主机（例如 github.com），并且您的提交将链接到 Elon 的 GitHub 个人资料。 （尝试一下！但是如果埃隆因为你冒充他而生你的气，请不要责怪我们。）

我们实施 GPG 签名的原因是：
- 提供更细粒度的安全性，特别是随着越来越多的企业用户依赖中心。
- 提供由加密安全源支持的机器学习基准。

请参阅 Ale Segala 的 [How (and why) to sign ⟦T7⟧ commits](https://withblue.ink/2020/05/17/how-and-why-to-sign-git-commits.html) 了解更多背景信息。

您可以使用 GNU Privacy Guard (GPG) 和密钥服务器证明提交是由您编写的。 GPG 是一种加密工具，用于验证消息来源的真实性。我们将在下面解释如何在 Hugging Face 上进行设置。

像往常一样，Pro Git 书是关于提交签名的很好的资源：[Pro Git: Signing your work](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)。

## 设置签名提交验证您需要在系统上安装[GPG](https://gnupg.org/)才能执行以下命令。
> 它默认包含在大多数 Linux 发行版中。
> 在 Windows 上，它包含在 Git Bash 中（Windows 附带 `git`）。

您可以使用 [GPG](https://gnupg.org/) 在本地签署您的提交。
然后配置您的配置文件以在集线器上将这些提交标记为**已验证**，
这样其他人就可以确信它们来自可靠的来源。

有关 git 和 GPG 如何交互的更深入解释，请访问 [git documentation on the subject](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work)

提交可以具有以下签名状态：

|状态 |说明|
| ----------------- | ------------------------------------------------------------------------ |
|已验证 |提交已签名并验证签名 |
|未经验证 |提交已签名但无法验证签名 |
|无签约状态 |提交未签名 |

要将提交标记为**已验证**，您需要上传用于在 Hugging Face 帐户上对其进行签名的公钥。使用 `gpg --list-secret-keys` 命令列出您同时拥有公钥和私钥的 GPG 密钥。
签署提交或标签需要私钥。

如果您没有 GPG 密钥对或者不想使用现有密钥来签署您的提交，请转到 **生成新的 GPG 密钥**。

否则，直接前往[Adding a GPG key to your account](#adding-a-gpg-key-to-your-account)。

## 生成新的 GPG 密钥

要生成 GPG 密钥，请运行以下命令：

```bash
gpg --gen-key
```

然后，GPG 将指导您完成创建 GPG 密钥对的过程。

确保您为此密钥指定了一个电子邮件地址，并且该电子邮件地址与您在 Hugging Face [account](https://huggingface.co/settings/account) 中指定的电子邮件地址相匹配。

## 将 GPG 密钥添加到您的帐户

1. 首先，在您的计算机上选择或生成 GPG 密钥。确保密钥的电子邮件地址与您的 Hugging Face [account](https://huggingface.co/settings/account) 中的电子邮件地址相符，并且您帐户的电子邮件已验证。

2. 导出所选密钥的公共部分：

```bash
gpg --armor --export <YOUR KEY ID>
```

3. 然后访问您的个人资料[settings page](https://huggingface.co/settings/keys)并单击“**添加 GPG 密钥**”。

将 `gpg --export` 命令的输出复制并粘贴到文本区域中，然后单击 **添加密钥**。

4. 恭喜！ 🎉 您刚刚向您的帐户添加了 GPG 密钥！

## 配置 git 使用 GPG 签署您的提交最后一步是配置 git 来签署您的提交：

```bash
git config user.signingkey <Your GPG Key ID>
git config user.email <Your email on hf.co>
```

然后将 `-S` 标志添加到您的 `git commit` 命令中以签署您的提交！

```bash
git commit -S -m "My first signed commit"
```

一旦推送到集线器上，您应该会看到带有“已验证”徽章的提交。

> [!提示]
> 要默认在计算机上的任何本地存储库中签署所有提交，您可以运行 git config --global commit.gpgsign true。

### 空间上的 JupyterLab
https://huggingface.co/docs/hub/spaces-sdks-docker-jupyter.md