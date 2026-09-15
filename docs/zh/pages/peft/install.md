<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 安装

在开始之前，您需要设置环境、安装适当的软件包并配置 🤗 PEFT。 🤗 PEFT 在 **Python 3.9+** 上进行了测试。

🤗 PEFT 可在 PyPI 和 GitHub 上使用：

## PyPI

要从 PyPI 安装 🤗 PEFT：

```bash
pip install peft
```

## CUDA Windows Arm64

要在 Windows Arm64 上安装带有 PyTorch CUDA 的 PEFT（例如 NVIDIA RTX Spark 笔记本电脑），请从 NVIDIA PyPI 索引安装 PyTorch。这些设备需要 NVIDIA 的 Windows Arm64 版本的 PyTorch，这在默认 PyPI 索引或标准 PyTorch 轮索引上不可用。

运行以下命令检查您的系统是否检测到 NVIDIA GPU。

```bash
nvidia-smi
```

从 NVIDIA PyPI 索引安装 PyTorch，然后安装 PEFT。

```bash
pip install torch --index-url https://pypi.nvidia.com/nvtorch_oot_nightly
pip install peft
```

## 来源

每天都会添加尚未发布的新功能，这也意味着可能存在一些错误。要试用它们，请从 GitHub 存储库安装：

```bash
pip install git+https://github.com/huggingface/peft
```

如果您希望使用源代码并在运行代码时查看实时结果，则可以使用可编辑版本
可以从存储库的本地克隆版本安装：

```bash
git clone https://github.com/huggingface/peft
cd peft
pip install -e ".[test]"
```如果您打算为 PEFT 做出贡献，请遵循 [contributing guide](developer_guides/contributing#installation)，其中还涵盖了分叉、添加上游远程以及创建工作分支。

### PEFT
https://huggingface.co/docs/peft/v0.21.0/index.md