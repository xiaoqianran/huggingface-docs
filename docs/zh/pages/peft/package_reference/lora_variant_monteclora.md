<!-- huggingface-docs: machine-translated zh-CN from English source -->

# MonteCLoRA（蒙特卡罗低阶适应）

> [!注意]
> 这是 LoRA 的一个变体，因此，除非本页另有说明，LoRA 的所有功能都适用于此方法。

MonteCLoRA 将标准 LoRA 适配器与一个小型变分模块包装在一起，该模块在训练期间在 LoRA `A` 矩阵之上绘制随机扰动的蒙特卡罗样本。具体来说，它学习变分参数（基于 Wishart 的协方差、每个样本的多元正态噪声项和样本的狄利克雷加权），并在每次前向传递时将所得的平均扰动添加到 `lora_A`。将 KL 散度 + 熵项添加到训练损失中，以将这些变分参数锚定到合理的先验。在推理时，采样器被禁用，MonteCLoRA 的行为与常规 LoRA 适配器完全相同，因此**没有额外的推理成本或需要合并的额外参数**。有关完整方法，请参阅 https://huggingface.co/papers/2411.04358。

在以下情况下您可能需要考虑 MonteCLoRA：- 您正在对小型或嘈杂的数据集进行微调，并且需要比普通 LoRA 更强的正则化。蒙特卡洛平均和 KL 项一起充当贝叶斯式正则化器。
- 您希望适配器具有更好的不确定性校准/鲁棒性，而无需在推理时支付额外成本（变分机制仅用于训练）。
- Vanilla LoRA 过度拟合，降低 `r` 或增加 `lora_dropout` 是不够的。

当您拥有一个大型、干净的数据集并且普通 LoRA 已经稳定地训练时，您可能“不需要”MonteCLoRA - 在这种情况下，额外的变分参数主要会增加训练开销，而没有太多好处。

要启用 MonteCLoRA，请将 `MontecloraConfig` 传递给 `LoraConfig`：

```py
from peft import LoraConfig, MontecloraConfig

monteclora_config = MontecloraConfig(
    num_samples=8,         # number of Monte Carlo samples per forward pass
    sample_scaler=1e-4,    # magnitude of the variational perturbation
    kl_loss_weight=1e-5,   # weight of the KL term added to the training loss
)
config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    monteclora_config=monteclora_config,
)
```

在训练期间，您必须将变分正则化损失添加到任务损失中。最简单的方法是在底层`LoraModel`上调用`LoraModel._get_monteclora_loss()`：

```py
task_loss = ...  # standard loss returned by your model
monteclora_loss = model._get_monteclora_loss()  # 0.0 if MonteCLoRA is not used
total_loss = task_loss + monteclora_loss
total_loss.backward()
```

如果您使用 HF `Trainer` 进行训练，您可以简单地混入 `peft.helpers.MontecloraTrainerMixin`，它会在 `compute_loss` 中为您完成此操作：

```py
from transformers import Trainer
from peft.helpers import MontecloraTrainerMixin

class MontecloraTrainer(MontecloraTrainerMixin, Trainer):
    pass
```

[⟦T14⟧](https://github.com/huggingface/peft/tree/main/examples/monteclora_finetuning) 提供了完整的工作示例。

### 普软件
https://huggingface.co/docs/peft/v0.20.0/package_reference/psoft.md