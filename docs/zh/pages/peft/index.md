<!-- huggingface-docs: machine-translated zh-CN from English source -->

#PEFT

🤗 PEFT（参数高效微调）是一个库，用于有效地将大型预训练模型适应各种下游应用程序，而无需微调模型的所有参数，因为它的成本过高。 PEFT 方法仅微调少量（额外）模型参数 - 显着降低计算和存储成本 - 同时产生与完全微调模型相当的性能。这使得在消费类硬件上训练和存储大型语言模型 (LLM) 和其他大型模型变得更加容易。

PEFT 与 Transformers、Diffusers 和 Accelerate 库集成，提供更快、更简单的方法来加载、训练和使用大型模型进行推理。

  
    有许多方法可以“适应”现有模型，通常广泛地集成到模型中。 PEFT 可以被认为是任意模型适应方法（修改权重、包装层、操作 KV 缓存等）的框架，同时也可以作为许多微调方法的参考实现。<a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="quicktour"
      >快速游览
      如果您是 🤗 PEFT 的新手，请从这里开始，了解该库的主要功能以及如何使用 PEFT 方法训练模型。
    
    <a class="!no-underline border dark:border-gray-700 p-5 rounded-lg shadow hover:shadow-lg" href="./methods/overview"
      >方法概述
      了解不同类别的 PEFT 方法，以了解如何在您的模型中使用这些方法。

### 维洛拉
https://huggingface.co/docs/peft/v0.21.0/package_reference/lora_variant_velora.md

### 维罗拉

> [!注意]
> 这是 LoRA 的一个变体，因此，除非本页另有说明，LoRA 的所有功能都适用于此方法。

[VeLoRA](https://huggingface.co/papers/2405.17991) 是一种 LoRA 变体，它通过压缩前向传递中为 LoRA 保存的激活，然后在后向传递中重建它们以实现更新规则来减少训练内存。在 PEFT 中，VeLoRA 通过 [LoraConfig](/docs/peft/v0.21.0/en/package_reference/lora#peft.LoraConfig) 上的 `velora_config` 参数配置为 LoRA 变体。

```py
from peft import LoraConfig, VeloraConfig

config = LoraConfig(
    target_modules=["q_proj", "v_proj"],
    velora_config=VeloraConfig(
        num_groups=64,
        scale=0.2,
        init_type="batch_average",
    ),
)
```VeLoRA 应用于`target_modules`选择的每个LoRA层。 `num_groups` 控制输入激活深度在压缩前如何分割。如果激活深度不能被 `num_groups` 整除，VeLoRA 会在内部填充分组表示，并在重建后删除填充。 `scale` 在向后传递过程中重新调整重建的激活，`init_type` 选择如何初始化投影。

使用 `batch_average_once` 从第一个训练批次初始化投影，使用 `batch_average` 从每次训练前向传递更新投影，或使用 `random` 立即从随机归一化向量初始化投影。

以下是 [MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) 的一些结果。

|变体 |训练损失|最大内存 (GiB) |令牌/秒 |
|---|---:|---:|---:|
|洛拉 | 0.5427 | 0.5427 27.69 | 27.69 2366.2 | 2366.2
| LoRA + GC | 0.5426 | 0.5426 13.17 | 13.17 1671.8 | 1671.8
| LoRA+VeLoRA | 0.5427 | 0.5427 19.94 | 19.94 2057.6 | 2057.6

#### 注意事项

- VeLoRA 目前仅在标准 LoRA 线性层上受支持。

### PEFT 类型
https://huggingface.co/docs/peft/v0.21.0/package_reference/peft_types.md