<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 安装

在开始之前，您需要设置环境、安装适当的软件包并配置 Accelerate。加速在 **Python 3.8+** 上进行了测试。

Accelerate 可在 pypi 和 conda 以及 GitHub 上使用。每个安装的详细信息如下：

## 点

要从 pypi 安装 Accelerate，请执行：

```bash
pip install accelerate
```

## 康达

Accelerate 也可以与 conda 一起安装：

```bash
conda install -c conda-forge accelerate
```

## 来源

每天都会添加尚未发布的新功能。要亲自尝试它们，请安装
来自 GitHub 存储库：

```bash
pip install git+https://github.com/huggingface/accelerate
```

如果您正在为库做出贡献或希望使用源代码并查看实时内容 
当您运行代码时，可以从本地克隆版本安装可编辑版本 
存储库：

```bash
git clone https://github.com/huggingface/accelerate
cd accelerate
pip install -e .
```

## 配置

安装后，您需要配置 Accelerate 以了解如何设置当前系统进行训练。 
为此，请运行以下命令并回答提示的问题：

```bash
accelerate config
```

要编写不包含 DeepSpeed 配置等选项或在 TPU 上运行的准系统配置，您可以快速运行：

```bash
python -c "from accelerate.utils import write_basic_config; write_basic_config(mixed_precision='fp16')"
```Accelerate 将自动利用可用 GPU 的最大数量并设置混合精度模式。

要检查您的配置是否正常，请运行：

```bash
accelerate env
```

下面显示了一个示例输出，它描述了一台机器上的两个 GPU，没有使用混合精度：

```bash
- `Accelerate` version: 1.2.0.dev0
- Platform: Linux-6.8.0-47-generic-x86_64-with-glibc2.35
- `accelerate` bash location: /home/zach/miniconda3/envs/accelerate/bin/accelerate
- Python version: 3.10.13
- Numpy version: 1.26.4
- PyTorch version (GPU?): 2.5.1+cu124 (True)
- PyTorch XPU available: False
- PyTorch NPU available: False
- PyTorch MLU available: False
- PyTorch MUSA available: False
- System RAM: 187.91 GB
- GPU type: NVIDIA GeForce RTX 4090
- `Accelerate` default config:
        - compute_environment: LOCAL_MACHINE
        - distributed_type: MULTI_GPU
        - mixed_precision: no
        - use_cpu: False
        - debug: False
        - num_processes: 2
        - machine_rank: 0
        - num_machines: 1
        - gpu_ids: all
        - rdzv_backend: static
        - same_network: True
        - main_training_function: main
        - enable_cpu_affinity: False
        - downcast_bf16: no
        - tpu_use_cluster: False
        - tpu_use_sudo: False
        - tpu_env: []
```

### 启动加速脚本
https://huggingface.co/docs/accelerate/v1.14.0/basic_tutorials/launch.md