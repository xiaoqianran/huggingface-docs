<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 快速入门

[Hugging Face Hub](https://huggingface.co/)是分享机器学习的首选之地
模型、演示、数据集和指标。 `huggingface_hub`库帮助您与
无需离开您的开发环境即可使用 Hub。您可以创建和管理
轻松存储库，下载和上传文件，并获取有用的模型和数据集
来自中心的元数据。

## 安装

首先，安装 `huggingface_hub` 库：

```bash
pip install --upgrade huggingface_hub
```

有关更多详细信息，请查看 [installation](installation) 指南。

> [!提示]
> `huggingface_hub` 还附带 [⟦T14⟧ CLI](./guides/cli)，可让您直接从终端与集线器交互。
> 如果您使用 AI 代理（Claude Code、Codex、Cursor...），请安装技能以让您的代理使用 CLI：
> ```bash
> # for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
> hf skills add
> # includes the above + Claude Code
> hf skills add --claude
> ```
> 查看 [Hugging Face CLI for AI Agents](https://huggingface.co/docs/hub/agents-cli) 指南了解更多详细信息。

## 下载文件

Hub上的存储库是git版本控制的，用户可以下载单个文件
或整个存储库。您可以使用[hf_hub_download()](/docs/huggingface_hub/v1.30.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)功能下载文件。
此函数将下载文件并将其缓存到本地磁盘上。下次有需要的时候
该文件，它将从您的缓存中加载，因此您无需重新下载它。您将需要存储库 ID 和要下载的文件的文件名。对于
例如，下载 [Pegasus](https://huggingface.co/google/pegasus-xsum) 模型
配置文件：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="google/pegasus-xsum", filename="config.json")
```

要下载文件的特定版本，请使用 `revision` 参数指定
分支名称、标签或提交哈希。如果您选择使用提交哈希，则它必须是
全长哈希而不是较短的 7 字符提交哈希：

```py
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(
...     repo_id="google/pegasus-xsum",
...     filename="config.json",
...     revision="4d33b01d79672f27f001f6abade33f22d993b151"
... )
```

有关更多详细信息和选项，请参阅 [hf_hub_download()](/docs/huggingface_hub/v1.30.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 的 API 参考。

 

## 身份验证

在很多情况下，您必须通过 Hugging Face 帐户进行身份验证才能与
中心：下载私人存储库、上传文件、创建 PR，...
[Create an account](https://huggingface.co/join)（如果您还没有）。

### 登录命令

最简单的身份验证方法是使用 [login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login) 命令：

```bash
hf auth login
```如果您已经登录，该命令将立即返回。要强制重新登录，请使用`hf auth login --force`。如果您尚未登录，系统将提示您使用浏览器登录：打开打印的 URL，输入短代码，批准请求，然后将检索访问令牌并将其保存在您的 `HF_HOME` 目录中（默认为 `~/.cache/huggingface/token`）。令牌会在一段时间后过期，但只要您继续使用它就会自动刷新。与集线器交互的任何脚本或库在发送请求时都将使用此令牌。或者，您可以粘贴从 [Settings page](https://huggingface.co/settings/tokens) 生成的 [User Access Token](https://huggingface.co/docs/hub/security-tokens)。

> [!提示]
> 用户访问令牌可以具有 `read` 或 `write` 权限。如果您想创建或编辑存储库，请确保拥有 `write` 访问令牌。否则，最好生成 `read` 代币，以降低代币无意泄露的风险。

或者，您可以在笔记本或脚本中使用 [login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login) 以编程方式登录：

```py
>>> from huggingface_hub import login
>>> login()
```

您一次只能登录一个帐户。登录新帐户后，您将自动退出前一个帐户。要确定您当前的活动帐户，只需运行 `hf auth whoami` 命令即可。> [!警告]
> 登录后，对集线器的所有请求（甚至不一定需要身份验证的方法）都将默认使用您的访问令牌。如果您想禁用令牌的隐式使用，您应该将 `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` 设置为环境变量（请参阅 [reference](../package_reference/environment_variables#hfhubdisableimplicittoken)）。

### 在本地管理多个令牌

您只需使用每个令牌使用 [login()](/docs/huggingface_hub/v1.30.0/en/package_reference/authentication#huggingface_hub.login) 命令登录即可在计算机上保存多个令牌。如果需要在本地切换这些令牌，可以使用 `auth switch` 命令：

```bash
hf auth switch
```

此命令将提示您从保存的令牌列表中按名称选择令牌。一旦选择，所选令牌将成为_active_令牌，它将用于与集线器的所有交互。

您可以使用 `hf auth list` 列出计算机上所有可用的访问令牌。

### 环境变量

环境变量`HF_TOKEN`也可用于验证您自己的身份。这在您可以将 `HF_TOKEN` 设置为 [Space secret](https://huggingface.co/docs/hub/spaces-overview#managing-secrets) 的空间中特别有用。

> [!提示]
> **新：** Google Colaboratory 允许您为笔记本定义 [private keys](https://twitter.com/GoogleColab/status/1719798406195867814)。定义一个`HF_TOKEN`秘密来自动验证！通过环境变量或秘密进行的身份验证优先于存储在计算机上的令牌。

### 方法参数

最后，还可以通过将令牌传递给任何接受 `token` 作为参数的方法来进行身份验证。

```
from huggingface_hub import whoami

user = whoami(token=...)
```

通常不鼓励这样做，除非您不想永久存储令牌或需要同时处理多个令牌。

> [!警告]
> 将标记作为参数传递时请小心。最佳实践始终是从安全保管库加载令牌，而不是将其硬编码到代码库或笔记本中。如果您无意中共享代码，硬编码令牌会带来重大泄漏风险。

## 创建存储库

注册并登录后，使用 [create_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建一个存储库
功能：

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model")
```

如果您希望您的存储库是私有的，那么：

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.create_repo(repo_id="super-cool-model", private=True)
```

除您自己之外，任何人都看不到私有存储库。

> [!提示]
> 要创建存储库或将内容推送到中心，您必须提供用户访问权限
> 具有`write`权限的令牌。创建时可以选择权限
> 您的 [Settings page](https://huggingface.co/settings/tokens) 中的代币。

## 上传文件使用 [upload_file()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 函数将文件添加到新创建的存储库中。你
需要指定：

1. 上传文件的路径。
2. 文件在存储库中的路径。
3. 要添加文件的存储库 ID。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/home/lysandre/dummy-test/README.md",
...     path_in_repo="README.md",
...     repo_id="lysandre/test-model",
... )
```

要一次上传多个文件，请查看 [Upload](./guides/upload) 指南
这将向您介绍几种上传文件的方法（使用或不使用 git）。

## 后续步骤

`huggingface_hub` 库为用户提供了一种与 Hub 交互的简单方法
使用Python。要了解有关如何管理文件和存储库的更多信息
Hub，我们建议您阅读我们的[how-to guides](./guides/overview)：

- [Manage your repository](./guides/repository)。
- 来自 Hub 的[Download](./guides/download) 文件。
- [Upload](./guides/upload) 文件到集线器。
- [Search the Hub](./guides/search) 您想要的模型或数据集。
- [Run Inference](./guides/inference) 跨 Hugging Face Hub 上托管的模型的多种服务。

### 安装
https://huggingface.co/docs/huggingface_hub/v1.30.0/installation.md