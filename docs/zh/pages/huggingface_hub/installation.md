<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 安装

在开始之前，您需要通过安装适当的软件包来设置环境。

`huggingface_hub` 在 **Python 3.10+** 上测试。

## 使用 pip 安装

强烈建议将`huggingface_hub`安装在[virtual environment](https://docs.python.org/3/library/venv.html)中。
如果你不熟悉Python虚拟环境，请看一下这个[guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)。
虚拟环境可以更轻松地管理不同的项目，并避免依赖项之间的兼容性问题。

首先在项目目录中创建一个虚拟环境：

```bash
python -m venv .venv
```

激活虚拟环境。在 Linux 和 macOS 上：

```bash
source .venv/bin/activate
```

在 Windows 上激活虚拟环境：

```bash
.venv/Scripts/activate
```

现在您已准备好安装`huggingface_hub`[from the PyPi registry](https://pypi.org/project/huggingface-hub/)：

```bash
pip install --upgrade huggingface_hub
```

完成后，[check installation](#check-installation) 即可正常工作。

### 安装可选依赖项

`huggingface_hub`的一些依赖项是[optional](https://setuptools.pypa.io/en/latest/userguide/dependency_management.html#optional-dependencies)，因为它们不需要运行`huggingface_hub`的核心功能。但是，如果未安装可选依赖项，`huggingface_hub` 的某些功能可能无法使用。

您可以通过 `pip` 安装可选依赖项：
```bash
# Install dependencies for both torch-specific and MCP-specific features.
pip install 'huggingface_hub[mcp,torch]'
```以下是 `huggingface_hub` 中的可选依赖项列表：
- `fastai`、`torch`：运行特定于框架的功能的依赖项。
- `dev`：为库做出贡献的依赖项。包括 `testing` （用于运行测试）、`typing` （用于运行类型检查器）和 `quality` （用于运行 linter）。

### 从源安装

在某些情况下，直接从源代码安装`huggingface_hub`会很有趣。
这允许您使用前沿 `main` 版本而不是最新的稳定版本。
例如，`main` 版本对于了解最新进展非常有用
如果自上次正式版本以来已修复错误但尚未推出新版本。

然而，这意味着`main`版本可能并不总是稳定。我们努力保持
`main`版本可运行，大多数问题通常都已解决
几个小时或一天之内。如果您遇到问题，请打开一个问题，以便我们可以
更快修复它！

```bash
pip install git+https://github.com/huggingface/huggingface_hub
```

从源安装时，您还可以指定特定分支。如果您这样做，这很有用
想要测试尚未合并的新功能或新错误修复：

```bash
pip install git+https://github.com/huggingface/huggingface_hub@my-feature-branch
```

完成后，[check installation](#check-installation) 即可正常工作。

### 可编辑安装从源代码安装允许您设置[editable install](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs)。
如果您打算为 `huggingface_hub` 做出贡献，这是一个更高级的安装
并且需要测试代码中的更改。您需要克隆 `huggingface_hub` 的本地副本
在你的机器上。

```bash
# First, clone repo locally
git clone https://github.com/huggingface/huggingface_hub.git

# Then, install with -e flag
cd huggingface_hub
pip install -e .
```

这些命令将链接您将存储库克隆到的文件夹和您的 Python 库路径。
除了正常的库路径之外，Python 现在还会查看您克隆到的文件夹内部。
例如，如果您的 Python 包通常安装在 `./.venv/lib/python3.13/site-packages/` 中，
Python 还将搜索您克隆的文件夹`./huggingface_hub/`。

## 安装 Hugging Face CLI 

使用我们的单行安装程序来设置 `hf` CLI，而无需接触您的 Python 环境：

在 macOS 和 Linux 上：

```bash
curl -LsSf https://hf.co/cli/install.sh | bash
```

在 Windows 上：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

要升级现有安装，请运行 `hf update` — 它会检测 `hf` 的安装方式（独立安装程序、Homebrew 或 pip）并运行匹配的命令。

## 使用 conda 安装

如果你更熟悉它，你可以使用[conda-forge channel](https://anaconda.org/conda-forge/huggingface_hub)安装`huggingface_hub`：

```bash
conda install -c conda-forge huggingface_hub
```

完成后，[check installation](#check-installation) 即可正常工作。

## 检查安装

安装后，通过运行以下命令检查`huggingface_hub`是否正常工作：

```bash
python -c "from huggingface_hub import model_info; print(model_info('gpt2'))"
```此命令将从集线器获取有关 [gpt2](https://huggingface.co/gpt2) 型号的信息。
输出应如下所示：

```text
Model Name: gpt2
Tags: ['pytorch', 'tf', 'jax', 'tflite', 'rust', 'safetensors', 'gpt2', 'text-generation', 'en', 'doi:10.57967/hf/0039', 'transformers', 'exbert', 'license:mit', 'has_space']
Task: text-generation
```

## Windows 限制

我们的目标是让优秀的机器学习民主化，我们将 `huggingface_hub` 打造成
跨平台库，特别是可以在基于 Unix 和 Windows 上正常工作
系统。然而，在某些情况下，`huggingface_hub`有一些限制：
在 Windows 上运行。以下是已知问题的详尽列表。请告诉我们如果您
遇到任何未记录的问题，请打开[an issue on Github](https://github.com/huggingface/huggingface_hub/issues/new/choose)。

- `huggingface_hub`的缓存系统依靠符号链接来有效地缓存下载的文件
来自枢纽。在 Windows 上，您必须激活开发人员模式或以管理员身份运行脚本才能
启用符号链接。如果它们没有被激活，缓存系统仍然可以工作，但是处于非优化状态
方式。请阅读[the cache limitations](./guides/manage-cache#limitations)部分了解更多详情。
- 集线器上的文件路径可以包含特殊字符（例如`"path/to?/my/file"`）。 Windows 是
对[special characters](https://learn.microsoft.com/en-us/windows/win32/intl/character-sets-used-in-file-names)有更多限制
这使得无法在 Windows 上下载这些文件。希望这是一个罕见的案例。
如果您认为这是一个错误，请联系仓库所有者或联系我们来解决
一个解决方案。

## 后续步骤一旦`huggingface_hub`正确安装在你的机器上，你可能想要
[configure environment variables](package_reference/environment_variables) 或 [check one of our guides](guides/overview) 开始。

### 🤗 Hub 客户端库
https://huggingface.co/docs/huggingface_hub/v1.29.0/index.md