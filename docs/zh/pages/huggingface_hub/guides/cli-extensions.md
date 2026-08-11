<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 创建 CLI 扩展

`hf` CLI 支持社区提供的扩展、自定义命令，无缝集成
进入 CLI。扩展作为公共 GitHub 存储库托管，可以使用单个命令安装。
安装后，它们就像内置命令一样显示为顶级 `hf` 命令。

该系统的灵感来自[GitHub CLI extensions](https://docs.github.com/en/github-cli/github-cli/creating-github-cli-extensions)。
在本指南中，您将学习如何创建自己的扩展、发布它并使其可被发现。

> [!提示]
> 有关安装和管理扩展的面向用户的文档，请参阅
> [CLI reference for ⟦T13⟧](../package_reference/cli#hf-extensions)。

## 概述

有两种类型的扩展：

1. **Shell脚本扩展**：放置在
   存储库的根。
2. **Python扩展**：带有`pyproject.toml`的标准Python包。安装在隔离的虚拟环境中
   环境，因此依赖项不会与用户系统发生冲突。

两种类型共享相同的约定：

- GitHub 存储库**必须**命名为 `hf-<name>`（例如，`hf-claude`、`hf-mem`）。
- 安装后，用户使用`hf <name>`（例如`hf claude`）运行扩展。
- 扩展列在“扩展命令”下的`hf --help` 中。当用户运行`hf extensions install [OWNER/]hf-<name>`时，系统首先查找二进制/脚本文件
在存储库根目录中名为 `hf-<name>`。如果找到，它将作为 shell 脚本扩展安装。否则，它
退回到将存储库安装为 Python 包。

## 创建 shell 脚本扩展

shell 脚本扩展是最简单的类型。您只需要一个带有可执行文件的 GitHub 存储库
根名为 `hf-<name>`。

### 最小示例

使用单个文件在 GitHub 上创建名为 `hf-hello` 的存储库：

**`hf-hello`**（位于存储库根目录）：

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "Hello from hf-hello extension!"
echo "Arguments: $@"
```

就是这样！用户现在可以安装并运行它：

```bash
>>> hf extensions install <your-username>/hf-hello
>>> hf hello
Hello from hf-hello extension!
Arguments:
```

### shell 脚本扩展的技巧

- 为了安全起见，始终从 shebang (`#!/usr/bin/env bash`) 和 `set -euo pipefail` 开始。
- 脚本接收用户传递的所有额外参数。例如，`hf hello --name world`
  将 `--name world` 传递给脚本。
- 如果用户已登录，您可以通过 `HF_TOKEN` 环境变量访问用户的 Hugging Face 令牌。
- 在存储库根添加`manifest.json`以提供描述（请参阅[Add a description](#add-a-description)）。
- 外部依赖项（例如，`fzf`、`jq` 等）**不会**随您的扩展自动安装。检查
  在脚本开始时查找所需的工具，如果缺少这些工具，则会优雅地失败并显示有用的错误消息。> [!提示]
> 有关真实世界的示例，请参阅 [hanouticelina/hf-claude](https://github.com/hanouticelina/hf-claude) —
> 一个 shell 脚本扩展，可使用 HF 推理提供程序启动 Claude Code。

## 创建Python扩展

Python 扩展是安装在隔离虚拟环境中的完整 Python 包。这是最好的
当您的扩展具有 Python 依赖项或更复杂的逻辑时的选择。

### 最小示例

在 GitHub 上创建一个名为 `hf-hello` 的存储库，结构如下：

```
hf-hello/
├── pyproject.toml
└── src/
    └── hf_hello/
        ├── __init__.py
        └── cli.py
```

**`pyproject.toml`**：

```toml
[build-system]
requires = ["setuptools>=64"]
build-backend = "setuptools.backends._legacy:_Backend"

[project]
name = "hf-hello"
version = "0.1.0"
description = "A hello-world hf CLI extension"
requires-python = ">=3.10"

[project.scripts]
hf-hello = "hf_hello.cli:main"
```

**`src/hf_hello/cli.py`**：

```python
import sys

def main():
    print("Hello from hf-hello extension!")
    print(f"Arguments: {sys.argv[1:]}")

if __name__ == "__main__":
    main()
```

关键部分是`[project.scripts]`入口点：它**必须**命名为`hf-<name>`（此处为`hf-hello`）。
这就是`hf` CLI 发现并执行您的扩展的方式。

用户安装并运行它的方式相同：

```bash
>>> hf extensions install <your-username>/hf-hello
>>> hf hello
Hello from hf-hello extension!
Arguments: []
```

### Python 扩展是如何安装的

当用户安装 Python 扩展时，会发生以下情况：

1. 在`~/.local/share/hf/extensions/hf-<name>/venv/`创建虚拟环境。
2. 您的软件包是通过 GitHub 存储库存档中的 `pip install` 安装的。
3. 系统验证 venv 中是否创建了`hf-<name>` 控制台脚本。

这意味着您的扩展的依赖项是完全隔离的 - 它们不会与用户的依赖项发生冲突
其他 Python 包。> [!提示]
> 有关真实世界的示例，请参阅 [alvarobartt/hf-mem](https://github.com/alvarobartt/hf-mem) —
> 一个 Python 扩展，用于估计 HF 模型的推理内存需求。

## 添加描述

描述可以帮助用户了解您的扩展程序的用途。出现在`hf extensions list`
以及`hf --help`。

系统按以下顺序查找描述：

1. **`manifest.json`** 位于存储库根目录：

```json
{
    "description": "A short description of what your extension does"
}
```

2. **`pyproject.toml`** `description`字段（用于Python扩展）：

```toml
[project]
description = "A short description of what your extension does"
```

3. **GitHub 存储库描述**（存储库页面上的“关于”字段）。

对于Python扩展，在`pyproject.toml`中设置`description`是最自然的方法。
对于 shell 脚本扩展，请使用 `manifest.json` 文件或设置 GitHub 存储库描述。

## 让你的扩展可被发现

为了帮助用户找到您的扩展，请将 **`hf-extension`** 主题添加到您的 GitHub 存储库：

1. 转到 GitHub 上的存储库。
2. 单击右侧边栏“关于”旁边的齿轮图标。
3. 在“主题”下添加`hf-extension`。

这是一个社区约定，可以轻松浏览网站上的所有可用扩展
[hf-extension topic page](https://github.com/topics/hf-extension)。

然后，用户可以使用 `hf extensions search` 直接从 CLI 发现您的扩展，其中列出了
所有带有 `hf-extension` 主题标记的 GitHub 存储库，按星号排序：

```bash
>>> hf extensions search
NAME   REPO                    STARS DESCRIPTION                         INSTALLED
------ ----------------------- ----- ----------------------------------- ---------
claude hanouticelina/hf-claude     2 Extension for `hf` CLI to launch... yes
agents hanouticelina/hf-agents       HF extension to run local coding...
````INSTALLED` 列显示本地已安装哪些扩展。从那里，用户可以
使用 `hf extensions install <repo>` 安装任何列出的扩展。

## 测试你的扩展

在开发过程中，您可以直接从 GitHub 存储库安装扩展：

```bash
# Install from your repo
>>> hf extensions install <your-username>/hf-<name>

# Run it
>>> hf <name>

# Reinstall after making changes (push to GitHub first)
>>> hf extensions install <your-username>/hf-<name> --force

# Update to the latest version (only if a newer commit is available)
>>> hf extensions update <name>

# List installed extensions
>>> hf extensions list

# Remove when done
>>> hf extensions remove <name>
```

> [!提示]
> 在测试更新时使用 `--force` 覆盖以前安装的版本。

## 更新已安装的扩展

安装扩展后，您可以将其更新到 GitHub 上发布的最新版本：

```bash
# Update a single extension (accepts `<name>`, `hf-<name>` or `OWNER/hf-<name>`)
>>> hf extensions update hf-claude

# Check every installed extension and update the outdated ones
>>> hf extensions update
```

`hf extensions update` 仅更新已安装的扩展。如果未安装扩展，则会引发错误。已更新的扩展将被跳过。

## 命名规则

扩展名必须遵循以下规则：

- GitHub 存储库必须命名为 `hf-<name>`。
- `<name>` 必须以字母或数字开头。
- `<name>`可以包含字母、数字、`.`、`_`和`-`。
- `<name>` 不能与内置 `hf` 命令冲突（例如，`download`、`upload`、`auth`）。

安装时，用户可以指定完整的`OWNER/hf-<name>`或仅指定`hf-<name>`（其中
默认为`huggingface`组织）。

## 现有扩展

以下是一些您可以用作参考的社区扩展：|扩展|类型 |描述 |
|------------|------|-------------|
| [hanouticelina/hf-claude](https://github.com/hanouticelina/hf-claude) |外壳脚本 |与 HF 推理提供商一起启动 Claude Code |
| [alvarobartt/hf-mem](https://github.com/alvarobartt/hf-mem) |蟒蛇 |估计 HF 模型的推理内存需求 |

### 操作指南
https://huggingface.co/docs/huggingface_hub/v1.27.0/guides/overview.md