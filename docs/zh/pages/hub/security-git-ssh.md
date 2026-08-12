<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 通过 SSH 进行 Git

您可以使用 SSH（安全外壳协议）访问和写入 Huggingface.co 上的存储库中的数据。当您通过 SSH 连接时，您可以使用本地计算机上的私钥文件进行身份验证。

某些操作（例如推送更改或克隆私有存储库）将要求您将 SSH 公钥上传到您在 Huggingface.co 上的帐户。

您可以使用预先存在的 SSH 密钥，或专门为 Huggingface.co 生成一个新密钥。

## 检查现有的 SSH 密钥

如果您有现有的 SSH 密钥，则可以使用该密钥对通过 SSH 的 Git 操作进行身份验证。

SSH 密钥通常位于 Mac 和 Linux 上的 `~/.ssh` 下，以及 Windows 上的 `C:\\Users\\<username>\\.ssh` 下。列出该目录下的文件并查找以下形式的文件：

- id_rsa.pub
- id_ecdsa.pub
- id_ed25519.pub

这些文件包含您的 SSH 公钥。

如果`~/.ssh`下没有此类文件，则必须[generate a new key](#generating-a-new-ssh-keypair)。否则，你可以[add your existing SSH public key(s) to your huggingface.co account](#add-a-ssh-key-to-your-account)。

## 生成新的 SSH 密钥对

如果您的计算机上没有任何 SSH 密钥，您可以使用 `ssh-keygen` 生成新的 SSH 密钥对（公钥 + 私钥）：

```
$ ssh-keygen -t ed25519 -C "your.email@example.co"
```我们建议您在出现提示时输入密码。密码是额外的安全层：每当您使用 SSH 密钥时都会提示输入密码。

生成新密钥后，使用 `ssh-add` 将其添加到 SSH 代理：

```
$ ssh-add ~/.ssh/id_ed25519
```

如果您选择的位置与默认位置不同来存储 SSH 密钥，则必须将 `~/.ssh/id_ed25519` 替换为您使用的文件位置。

## 将 SSH 密钥添加到您的帐户

要使用 SSH 访问私有存储库，或通过 SSH 推送更改，您需要将 SSH 公钥添加到您的 Huggingface.co 帐户中。您可以管理您的 SSH 密钥[in your user settings](https://huggingface.co/settings/keys)。

要将 SSH 密钥添加到您的帐户，请单击“添加 SSH 密钥”按钮。

然后，输入该密钥的名称（例如“个人计算机”），并将 **公共** SSH 密钥的内容复制并粘贴到下面的区域中。公钥位于您在前面的步骤中找到或生成的 `~/.ssh/id_XXXX.pub` 文件中。

单击“添加密钥”，瞧！您已将 SSH 密钥添加到您的 Huggingface.co 帐户。

## 测试您的 SSH 身份验证

将 SSH 密钥添加到 Huggingface.co 帐户后，您可以测试连接是否按预期工作。

在终端中，运行：
```
$ ssh -T git@hf.co
```如果您看到一条包含您的用户名的消息，那么恭喜！一切顺利，您已准备好通过 SSH 使用 git。

否则，如果消息显示类似以下内容，请确保您的 SSH 密钥确实由 SSH 代理使用。
```
Hi anonymous, welcome to Hugging Face.
```

## HuggingFace 的 SSH 密钥指纹

公钥指纹可用于验证与远程服务器的连接。

这些是 HuggingFace 的公钥指纹：

> SHA256:aBG5R7IomF4BSsx/h6tNAUVLhEkkaNGB8Sluyh/Q/qY (ECDSA)
> SHA256:skgQjK2+RuzvdmHr24IIAJ6uLWQs0TGtEUt3FtzqirQ（DSA - 已弃用）
> SHA256:dVjzGIdV7d6cwKieZiCorMa2gMvSKfGZAvHf4gMiMao (ED25519)
> SHA256:uqjYymysBGCXXiMVebB8L8RIuWbPSKGBxQQNhcT5a3Q (RSA)

您可以将以下 ssh 密钥条目添加到 ~/.ssh/known_hosts 文件中，以避免手动验证 HuggingFace 主机：

```
hf.co ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQDtPB+snz63eZvTrbMY2Qt39a6HYile89JOum55z3lhIqAqUHxLtXFd+q+ED8izQvyORFPSmFIaPw05rtXo37bm+ixL6wDmvWrHN74oUUWmtrv2MNCLHE5VDb3+Q6MJjjDVIoK5QZIuTStlq0cUbGGxQk7vFZZ2VXdTPqgPjw4hMV7MGp3RFY/+Wy8rIMRv+kRCIwSAOeuaLPT7FzL0zUMDwj/VRjlzC08+srTQHqfoh0RguZiXZQneZKmM75AFhoMbP5x4AW2bVoZam864DSGiEwL8R2jMiyXxL3OuicZteZqll0qfRlNopKnzoxS29eBbXTr++ILqYz1QFqaruUgqSi3MIC9sDYEqh2Q8UxP5+Hh97AnlgWDZC0IhojVmEPNAc7Y2d+ctQl4Bt91Ik4hVf9bU+tqMXgaTrTMXeTURSXRxJEm2zfKQVkqn3vS/zGVnkDS+2b2qlVtrgbGdU/we8Fux5uOAn/dq5GygW/DUlHFw412GtKYDFdWjt3nJCY8=
hf.co ssh-dss AAAAB3NzaC1kc3MAAACBAORXmoE8fn/UTweWy7tCYXZxigmODg71CIvs/haZQN6GYqg0scv8OFgeIQvBmIYMnKNJ7eoo5ZK+fk1yPv8aa9+8jfKXNJmMnObQVyObxFVzB51x8yvtHSSrL4J3z9EAGX9l9b+Fr2+VmVFZ7a90j2kYC+8WzQ9HaCYOlrALzz2VAAAAFQC0RGD5dE5Du2vKoyGsTaG/mO2E5QAAAIAHXRCMYdZij+BYGC9cYn5Oa6ZGW9rmGk98p1Xc4oW+O9E/kvu4pCimS9zZordLAwHHWwOUH6BBtPfdxZamYsBgO8KsXOWugqyXeFcFkEm3c1HK/ysllZ5kM36wI9CUWLedc2vj5JC+xb5CUzhVlGp+Xjn59rGSFiYzIGQC6pVkHgAAAIBve2DugKh3x8qq56sdOH4pVlEDe997ovEg3TUxPPIDMSCROSxSR85fa0aMpxqTndFMNPM81U/+ye4qQC/mr0dpFLBzGuum4u2dEpjQ7B2UyJL9qhs1Ubby5hJ8Z3bmHfOK9/hV8nhyN8gf5uGdrJw6yL0IXCOPr/VDWSUbFrsdeQ==
hf.co ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBL0wtM52yIjm8gRecBy2wRyEMqr8ulG0uewT/IQOGz5K0ZPTIy6GIGHsTi8UXBiEzEIznV3asIz2sS7SiQ311tU=
hf.co ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAINJjhgtT9FOQrsVSarIoPVI1jFMh3VSHdKfdqp/O776s
```

### 任务
https://huggingface.co/docs/hub/models-tasks.md