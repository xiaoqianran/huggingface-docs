<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 安装

您可以从 PyPI 或源代码安装 TRL：

## PyPI

使用 pip 或 [uv](https://docs.astral.sh/uv/) 安装库：

uv 是一个基于 Rust 的快速 Python 包和项目管理器。安装说明请参阅[Installation](https://docs.astral.sh/uv/getting-started/installation/)。

```bash
uv pip install trl
```

```bash
pip install trl
```

## 来源

您还可以从源安装最新版本。首先克隆存储库，然后使用 `pip` 运行安装：

```bash
git clone https://github.com/huggingface/trl.git
cd trl/
pip install -e .
```

如果您想要开发安装，可以将 pip install 替换为以下内容：

```bash
pip install -e ".[dev]"
```

### RapidFire 人工智能集成
https://huggingface.co/docs/trl/v1.9.2/rapidfire_integration.md