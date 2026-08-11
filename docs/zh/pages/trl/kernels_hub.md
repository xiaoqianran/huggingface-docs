<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 内核中心集成和使用

[⟦T6⟧](https://huggingface.co/blog/hello-hf-kernels#get-started-and-next-steps) 库允许直接从 Hub 加载优化的计算内核。  
您可以在[dedicated orgs](https://huggingface.co/kernels-community)中找到`kernels`，或者在Hub内搜索[⟦T8⟧ tag](https://huggingface.co/models?other=kernel)。  

内核是**优化的代码片段**，有助于模型开发、训练和推理。在这里，我们将重点关注它们**与 TRL 的集成**，但请查看上述资源以了解有关它们的更多信息。

## 安装

要将内核与 TRL 一起使用，您需要在 Python 环境中安装该库：

```bash
pip install kernels
```

## 在 TRL 中使用 Hub 的内核

内核可以直接替换注意力实现，无需手动编译 Flash Attention 等注意力后端，只需从 Hub 中拉取相应的注意力内核即可提高训练速度。

您可以在加载模型时指定内核：

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    attn_implementation="kernels-community/flash-attn2"  # other options: kernels-community/vllm-flash-attn3, kernels-community/paged-attention
)
```

或者运行 TRL 训练脚本时：

```bash
python sft.py ... --attn_implementation kernels-community/flash-attn2
```

或者使用 TRL CLI：

```bash
trl sft ... --attn_implementation kernels-community/flash-attn2
```

> [!提示]
> 现在，您可以通过 Hub 为您的硬件配置利用更快的注意力后端和预先优化的内核，从而加快开发和培训速度。

## 比较注意力实现我们使用 **TRL** 和 **SFT** 评估了 Transformer 中可用的各种注意力实现以及不同的内核后端。  
实验在具有 **CUDA 12.9** 的单个 **H100 GPU** 上运行，利用 **Qwen3-8B**，**批量大小为 8**，**梯度累积为 1**，以及 **bfloat16** 精度。  
请记住，此处显示的结果特定于此设置，并且可能会因不同的训练配置而异。

下图说明了不同注意实现和内核后端的**延迟**（每个训练步骤的时间）和**分配的内存峰值**。  
基于内核的实现与自定义安装的注意力相当，并且增加模型的`max_length`进一步增强了性能。所有实现的内存消耗都相似，没有显着差异。我们获得了相同的性能，但摩擦更少，如 [the following section](#flash-attention-vs-hub-kernels) 中所述。

  
  

## Flash Attention 与 Hub 内核

从源代码构建 Flash Attention 可能非常耗时，通常需要几分钟到几小时，具体取决于您的硬件、CUDA/PyTorch 配置以及预编译轮子是否可用。相比之下，**Hugging Face Kernels** 提供了更快、更可靠的工作流程。开发人员无需担心复杂的设置——一切都会自动处理。在我们的基准测试中，内核在大约 **2.5 秒**内即可使用，无需编译。这使您几乎可以立即开始训练，从而显着加速开发。只需指定所需的版本，`kernels` 就会处理剩下的事情。

## 将 FlashAttention 内核与 Liger 内核相结合

您可以将 **FlashAttention 内核** 与 **Liger 内核** 结合起来，以进一步提高 TRL 性能。

首先，安装Liger内核依赖项：

```bash
pip install liger-kernel
```

然后，将两者结合到您的代码中：

```python
from transformers import AutoModelForCausalLM
from trl import SFTConfig

model = AutoModelForCausalLM.from_pretrained(
    "your-model-name",
    attn_implementation="kernels-community/flash-attn2"  # choose the desired FlashAttention variant
)

training_args = SFTConfig(
    use_liger_kernel=True,
    # ... other TRL training args
)
```

了解有关 [Liger Kernel Integration](./liger_kernel_integration) 的更多信息。

### Trackio 集成
https://huggingface.co/docs/trl/v1.9.2/trackio_integration.md