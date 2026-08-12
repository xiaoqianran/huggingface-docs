<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Spaces 开发模式：Spaces 中的无缝开发

> [!警告]
> Spaces 开发模式是 PRO 或团队和企业计划的一部分。

## 空间开发模式

Spaces 开发模式是一项可简化应用程序调试并使 Spaces 迭代速度更快的功能。

每当您向 Space 存储库提交一些更改时，底层 Docker 映像就会重新构建，然后配置一个新的虚拟机来托管新容器。

开发模式允许您通过覆盖 Docker 镜像来更快地更新您的空间。

开发模式 Docker 映像将您的应用程序作为子进程启动，允许您在不停止 Space 容器本身的情况下重新启动它。它还会在后台启动 SSH 服务器（用于本地编辑器和终端）和 VS Code 服务器（用于产品内的 VS Code Web），以便您可以连接到空间。

连接到正在运行的空间的能力解锁了多个用例：

    - 您可以更改应用程序代码，而无需每次都重建空间
    - 您可以调试正在运行的应用程序并实时监控资源

总的来说，它跳过了 Docker 镜像重建阶段，使得 Spaces 的开发和实验变得更快。

＃＃ 界面在您的空间上启用开发模式后，您应该会看到如下所示的模式。

当您更改代码时，应用程序不会自动重新启动。为了让您的更改显示在空间中，您需要使用 `Refresh` 按钮来重新启动应用程序。

  如果您使用的是 Gradio SDK，或者您的应用程序基于 Python，请注意，不会自动安装要求。
  您需要从 VS Code 或 SSH 手动运行 `pip install`。

### 正在连接

开发模式将运行空间公开为标准 SSH 主机。 VS Code 和任何其他编辑器通过同一 SSH 端点进行连接。

#### SSH

开发模式运行后，连接：

```shell
ssh <space-subdomain>@ssh.hf.space
```

子域显示在开发模式控件模态中。您还可以通过编程方式检索它：

```python
from huggingface_hub import HfApi
print(HfApi().space_info("namespace/repo").subdomain)
```

Hugging Face CLI 可以解析子域并为您打开会话：

```shell
# SSH into the Space's Dev Mode container
hf spaces ssh username/my-space

# Print the SSH command instead of running it
hf spaces ssh username/my-space --dry-run

# Use a specific identity file
hf spaces ssh username/my-space -i ~/.ssh/id_ed25519
```

如果尚未运行，则通过 `--auto` 启用开发模式而不提示。有关完整选项列表，请参阅[⟦T9⟧ CLI reference](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-spaces-ssh)。

您需要将计算机的 SSH 公钥添加到 [your user account](https://huggingface.co/settings/keys) 才能使用 SSH 连接到空间。
查看 [Git over SSH](./security-git-ssh#add-a-ssh-key-to-your-account) 文档以获取更详细的说明。

#### VS 代码您还可以使用本地安装的 VS Code 连接到 Space 容器。为此，请安装 [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) 扩展并连接到同一台 `<space-subdomain>@ssh.hf.space` 主机。

### 持续变化

启用开发模式时所做的更改不会自动保留到空间存储库中。
默认情况下，当禁用开发模式或空间进入睡眠状态时，它们将被丢弃。

如果您希望在启用开发模式时保留所做的更改，则需要从 Space 容器内部使用`git`（使用 VS Code 或 SSH）。例如：

```shell
# Add changes and commit them
git add .
git commit -m "Persist changes from Dev Mode"

# Push the commit to persist them in the repo
git push
```

如果您在空间中有未提交或未推送的更改，该模式将显示警告：

## 启用开发模式

您可以从 Web 界面或通过 API 在您的空间上启用开发模式。

### 通过 API

您可以通过编程方式切换开发模式：

```
POST https://huggingface.co/api/spaces/{namespace}/{repo}/dev-mode
Content-Type: application/json
Authorization: Bearer {token}

{
  "enabled": true
}
```

### 通过网络界面

您还可以创建一个启用开发模式的空间：

## 限制

开发模式目前不适用于静态空间。 Docker Spaces 还有一些额外的要求。

### Docker 空间

Docker Spaces 支持开发模式。但是，您的空间需要遵守以下规则才能使开发模式正常工作。1. 必须安装以下软件包：

- `bash`（建立 SSH 连接所需）
- `curl`、`wget` 和 `procps`（VS Code 服务器进程所需）
- `git` 和 `git-lfs` 能够从开发模式环境提交和推送更改

2. 您的应用程序代码必须位于 `/app` 文件夹中，以便开发模式守护程序能够检测更改。

3. `/app` 文件夹必须由 uid `1000` 的用户拥有，才能允许您更改代码。

4. Dockerfile 必须包含`CMD` 启动指令。查看[Docker's documentation](https://docs.docker.com/reference/dockerfile/#cmd)有关`CMD`指令的更多详细信息。

当基础镜像基于 debian（例如 ubuntu）时，开发模式效果很好。

更奇特的 Linux 发行版（例如 alpine）未经测试，并且不保证开发模式适用于它们。

### 兼容 Dockerfile 的示例

这是与 Spaces Dev 模式兼容的 Dockerfile 示例。

它使用 `apt-get` 安装所需的软件包，以及为开发人员方便而安装的其他软件包（即：`top`、`vim` 和 `nano`）。
然后它从`/app`启动一个NodeJS应用程序。

```Dockerfile
FROM node:19-slim

RUN apt-get update && \
    apt-get install -y \
      bash \
      git git-lfs \
      wget curl procps \
      htop vim nano && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --link ./ /app
RUN  npm i 

RUN chown 1000 /app
USER 1000
CMD ["node", "index.js"]
```

该组织中有几个兼容开发模式的 Docker 空间的示例。
请随意在您的命名空间中复制它们！Python 应用程序示例（FastAPI HTTP 服务器）：https://huggingface.co/spaces/dev-mode-explorers/dev-mode-python

示例 Javascript 应用程序（Express.js HTTP 服务器）：https://huggingface.co/spaces/dev-mode-explorers/dev-mode-javascript

## 反馈

您可以直接在 HF Hub 上分享您对 Spaces 开发模式的反馈：https://huggingface.co/spaces/dev-mode-explorers/README/discussions

### 空间概述
https://huggingface.co/docs/hub/spaces-overview.md