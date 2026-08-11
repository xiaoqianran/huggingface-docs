<!-- huggingface-docs: machine-translated zh-CN from English source -->

# Liger 内核集成

[Liger Kernel](https://github.com/linkedin/Liger-Kernel) 是专为 LLM 训练设计的 Triton 内核集合。可有效提升多GPU训练吞吐量20%，内存占用降低60%。这样，我们就可以 **4x** 我们的上下文长度，如下面的基准测试所述。他们已经实现了 Hugging Face 兼容 `RMSNorm`、`RoPE`、`SwiGLU`、`CrossEntropy`、`FusedLinearCrossEntropy`，还有更多功能即将推出。内核可与 [FlashAttention](https://github.com/Dao-AILab/flash-attention)、[PyTorch FSDP](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html) 和 [Microsoft DeepSpeed](https://github.com/microsoft/DeepSpeed) 开箱即用。

通过减少内存，您可以关闭 `cpu_offloading` 或梯度检查点以进一步提高性能。

|加速 |内存减少|
| --- | --- |
| ![Speed up](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-tps.png) | ![Memory](https://raw.githubusercontent.com/linkedin/Liger-Kernel/main/docs/images/e2e-memory.png) |

## 支持的培训师

以下 TRL 训练器支持 Liger Kernel：
- **SFT**（监督微调）
- **DPO**（直接偏好优化）
- **GRPO**（组相关策略优化）
- **KTO**（卡尼曼-特沃斯基优化）
- **GKD**（广义知识蒸馏）

## 用法

1.首先安装Liger内核：

  ```bash
  pip install liger-kernel
  ```

2. 安装后，在训练器配置中设置`use_liger_kernel=True`。不需要其他改变！

```python
from trl import SFTConfig

training_args = SFTConfig(..., use_liger_kernel=True)
```

```python
from trl import DPOConfig

training_args = DPOConfig(..., use_liger_kernel=True)
```

```python
from trl import GRPOConfig

training_args = GRPOConfig(..., use_liger_kernel=True)
```

```python
from trl import KTOConfig

training_args = KTOConfig(..., use_liger_kernel=True)
```

```python
from trl.experimental.gkd import GKDConfig

training_args = GKDConfig(..., use_liger_kernel=True)
```

要了解有关 Liger-Kernel 的更多信息，请访问他们的[official repository](https://github.com/linkedin/Liger-Kernel/)。### 聊天模板实用程序
https://huggingface.co/docs/trl/v1.9.2/chat_template_utils.md