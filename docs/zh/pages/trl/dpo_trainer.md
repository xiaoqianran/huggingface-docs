<!-- huggingface-docs: machine-translated zh-CN from English source -->

#DPO 培训师

[⟦T264⟧](https://huggingface.co/models?other=dpo,trl)[⟦T265⟧](https://github.com/huggingface/smol-course/tree/main/2_preference_alignment)

## 概述

TRL 支持直接偏好优化 (DPO) Trainer 来训练语言模型，如 [Rafael Rafailov](https://huggingface.co/rmrafailov)、Archit Sharma、Eric Mitchell、[Stefano Ermon](https://huggingface.co/ermonste)、[Christopher D. Manning](https://huggingface.co/manning)、[Chelsea Finn](https://huggingface.co/cbfinn) 的论文 [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290) 中所述。

论文摘要如下：

> 虽然大规模无监督语言模型 (LM) 学习广泛的世界知识和一些推理技能，但由于其训练的完全无监督性质，实现对其行为的精确控制很困难。获得这种可操纵性的现有方法收集模型生成相对质量的人类标签，并微调无监督的 LM 以符合这些偏好，通常通过人类反馈（RLHF）进行强化学习。然而，RLHF 是一个复杂且通常不稳定的过程，首先拟合反映人类偏好的奖励模型，然后使用强化学习对大型无监督 LM 进行微调，以最大化估计的奖励，而不会偏离原始模型太远。在本文中，我们引入了 RLHF 奖励模型的新参数化，它能够以封闭形式提取相应的最优策略，使我们能够只需简单的分类损失即可解决标准 RLHF 问题。由此产生的算法，我们称之为直接偏好优化 (DPO)，稳定、高性能且计算量轻，无需在微调或执行重要的超参数调整期间从 LM 进行采样。我们的实验表明，DPO 可以微调 LM 以符合人类偏好，并且比现有方法更好。值得注意的是，使用 DPO 进行微调在控制各代情绪的能力方面超过了基于 PPO 的 RLHF，并且匹配或提高了摘要和单轮对话中的响应质量，同时大大简化了实施和训练。

这个训练后的方法是由[Kashif Rasul](https://huggingface.co/kashif)贡献的，后来由[Quentin Gallouédec](https://huggingface.co/qgallouedec)重构。

## 快速开始

此示例演示如何使用 TRL 的 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 训练语言模型。我们在 [UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback) 上训练 [Qwen 3 0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 模型。

```python
from trl import DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen3-0.6B",
    train_dataset=load_dataset("trl-lib/ultrafeedback_binarized", split="train"),
)
trainer.train()
```

## 预期的数据集类型和格式

DPO 需要[preference](dataset_formats#preference) 数据集。 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 与[standard](dataset_formats#standard) 和[conversational](dataset_formats#conversational) 数据集格式兼容。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。

```python
# Standard format
## Explicit prompt (recommended)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Conversational format
## Explicit prompt (recommended)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```如果您的数据集不是这些格式之一，您可以对其进行预处理以将其转换为预期格式。以下是 [Vezora/Code-Preference-Pairs](https://huggingface.co/datasets/Vezora/Code-Preference-Pairs) 数据集的示例：

```python
from datasets import load_dataset

dataset = load_dataset("Vezora/Code-Preference-Pairs")

def preprocess_function(example):
    return {
        "prompt": [{"role": "user", "content": example["input"]}],
        "chosen": [{"role": "assistant", "content": example["accepted"]}],
        "rejected": [{"role": "assistant", "content": example["rejected"]}],
    }

dataset = dataset.map(preprocess_function, remove_columns=["instruction", "input", "accepted", "ID"])
print(next(iter(dataset["train"])))
```

```json
{
    "prompt": [{"role": "user", "content": "Create a nested loop to print every combination of numbers [...]"}],
    "chosen": [{"role": "assistant", "content": "Here is an example of a nested loop in Python [...]"}],
    "rejected": [{"role": "assistant", "content": "Here is an example of a nested loop in Python [...]"}],
}
```

## 深入研究 DPO 方法

直接偏好优化 (DPO) 是一种旨在使语言模型与偏好数据保持一致的训练方法。该模型不是受监督的输入输出对，而是根据同一提示的成对完成进行训练，其中一个完成比另一个更受青睐。该目标直接优化模型，以相对于参考模型扩大首选和不首选完成的对数似然之间的差距，而不需要明确的奖励模型。在实践中，这通常是通过抑制非首选完成的可能性而不是增加首选完成的可能性来实现的。

本节详细介绍了 DPO 在实践中的工作原理，涵盖关键步骤：**预处理**和**损失计算**。

### 预处理和标记化在训练期间，每个示例都应包含提示以及首选 (`chosen`) 和不首选 (`rejected`) 完成。有关预期格式的更多详细信息，请参阅[Dataset formats](dataset_formats)。
[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 使用模型的分词器对每个输入进行分词。

### 计算损失

![dpo_figure](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/dpo_figure.png)

DPO中使用的损失定义如下：
$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\mathbb{E}_{(x,y^{+},y^{-})}\!\left[\log \sigma\!\left(\beta\Big(\log\frac{\pi_{\theta}(y^{+}\!\mid) x)}{\pi_{\mathrm{ref}}(y^{+}\!\mid x)}-\log \frac{\pi_{\theta}(y^{-}\!\mid x)}{\pi_{\mathrm{ref}}(y^{-}\!\mid x)}\Big)\right)\right]
$$
  
其中 \\( x \\) 是提示符，\\( y^+ \\) 是首选补全，而 \\( y^- \\) 是不首选补全。  \\( \pi_{\theta} \\) 是正在训练的策略模型， \\( \pi_{\mathrm{ref}} \\) 是参考模型， \\( \sigma \\) 是 sigmoid 函数， \\( \beta > 0 \\) 是控制偏好信号强度的超参数。

#### 损失类型

文献中已经提出了几种目标表述。最初，DPO 的目标定义如上所述。| `loss_type=` |描述 |
| --- | --- |
| `"sigmoid"`（默认）|给定偏好数据，我们可以根据 Bradley-Terry 模型拟合二元分类器，事实上，[DPO](https://huggingface.co/papers/2305.18290) 作者提出了通过 `logsigmoid` 归一化似然的 sigmoid 损失来拟合逻辑回归。 |
| `"hinge"` | [RSO](https://huggingface.co/papers/2309.06657) 作者建议在 [SLiC](https://huggingface.co/papers/2305.10425) 论文中的归一化似然上使用铰链损失。在这种情况下，`beta`是边距的倒数。 |
| `"ipo"` | [IPO](https://huggingface.co/papers/2310.12036) 作者认为 logit 变换可能会过度拟合，并提出恒等变换来直接优化偏好； TRL 将其公开为 `loss_type="ipo"`。 |
| `"exo_pair"` | [EXO](https://huggingface.co/papers/2402.00856) 作者提出了反向 KL 偏好优化。 `label_smoothing`必须严格大于`0.0`；推荐值为`1e-3`（有关简化的成对变体，请参见方程 16）。完整的方法使用 `K>2` SFT 补全，并随着 `K` 的增长而接近 PPO。 |
| `"nca_pair"` | [NCA](https://huggingface.co/papers/2402.05369) 作者表明，NCA 优化了每个响应的绝对可能性，而不是相对可能性。 || `"robust"` | [Robust DPO](https://huggingface.co/papers/2403.00409) 作者提出了噪声偏好下的无偏 DPO 损失。使用[DPOConfig](/docs/trl/v1.9.2/en/dpo_trainer#trl.DPOConfig)中的`label_smoothing`来建模标签翻转概率；有效值在`[0.0, 0.5)`范围内。 |
| `"bco_pair"` | [BCO](https://huggingface.co/papers/2404.04656) 作者训练一个二元分类器，其 logit 作为奖励，以便分类器将 {提示，选择的完成} 对映射到 1，将 {提示，拒绝的完成} 对映射到 0。对于未配对的数据，我们推荐专用的 [experimental.bco.BCOTrainer](/docs/trl/v1.9.2/en/bco_trainer#trl.experimental.bco.BCOTrainer)。 |
| `"sppo_hard"` | [SPPO](https://huggingface.co/papers/2405.00675)的作者声称，SPPO能够通过将所选奖励推至大至1/2、将拒绝奖励推至小至-1/2来迭代解决纳什均衡，并且可以缓解数据稀疏问题。该实现通过使用硬标签概率来近似该算法，将 1 分配给获胜者，将 0 分配给失败者。 |
| `"aot"` 或 `loss_type="aot_unpaired"` | [AOT](https://huggingface.co/papers/2406.05882) 作者提出通过最优传输进行分布偏好对齐。 `loss_type="aot"`用于配对数据； `loss_type="aot_unpaired"` 适用于未配对的数据。两者都通过排序分位数强制随机支配；更大的每 GPU 批量大小会有所帮助。 || `"apo_zero"` 或 `loss_type="apo_down"` | [APO](https://huggingface.co/papers/2408.06266) 方法引入了锚定目标。 `apo_zero` 提高获胜者的权重并降低失败者的权重（当模型表现低于获胜者时很有用）。 `apo_down` 降低了两者的权重，对失败者施加更大的压力（当模型已经优于获胜者时很有用）。 |
| `"discopop"` | [DiscoPOP](https://huggingface.co/papers/2406.08414) 论文使用 LLM 来发现更有效的离线偏好优化损失。在论文中，提出的 DiscoPOP 损失（一种对数比调制损失）在不同任务（IMDb 正文本生成、Reddit TLDR 摘要和 Alpaca Eval 2.0）上优于其他优化损失。 |
| `"sft"` | SFT（监督微调）损失是负对数似然损失，用于训练模型生成首选响应。 |
| `"sigmoid_norm"` | [SimPO](https://huggingface.co/papers/2405.14734) 作者通过非掩码标记的数量标准化来解决原始 sigmoid 损失中的长度偏差； TRL 将其公开为 `loss_type="sigmoid_norm"`。 |

## 记录的指标

在培训和评估时，我们记录以下指标：* `global_step`：迄今为止采取的优化器步骤总数。
* `epoch`：当前纪元数，基于数据集迭代。
* `num_tokens`：到目前为止已处理的令牌总数。
* `loss`：当前记录间隔内的平均 DPO 损失。
* `entropy`：模型预测的令牌分布在非屏蔽令牌上的平均熵。
* `mean_token_accuracy`：模型的 top-1 预测与所选完成中的标记匹配的非屏蔽标记的比例。
* `learning_rate`：当前学习率，如果使用调度器，则可能会动态变化。
* `grad_norm`：梯度的 L2 范数，在梯度裁剪之前计算。
* `logits/chosen`：模型为所选完成中的标记分配的平均 Logit 值。
* `logits/rejected`：模型为拒绝完成中的标记分配的平均 Logit 值。
* `logps/chosen`：模型分配给所选完成中的标记的平均对数概率。
* `logps/rejected`：模型分配给被拒绝完成中的标记的平均对数概率。* `rewards/chosen`：为所选完成计算的平均隐式奖励，计算公式为 \\( \beta \log \frac{\pi_{\theta}(y^{+}\!\mid x)}{\pi_{\mathrm{ref}}(y^{+}\!\mid x)} \\)。
* `rewards/rejected`：针对被拒绝的完成计算的平均隐式奖励，计算公式为 \\( \beta \log \frac{\pi_{\theta}(y^{-}\!\mid x)}{\pi_{\mathrm{ref}}(y^{-}\!\mid x)} \\)。
* `rewards/margins`：选择完成和拒绝完成之间的平均隐性奖励差额。
* `rewards/accuracies`：选择完成的隐含奖励高于拒绝完成的隐含奖励的示例比例。

## 定制

### 兼容性和约束

当前 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 实现中有意限制了一些参数组合：

* `use_weighting=True` 不支持 `loss_type="aot"` 或 `loss_type="aot_unpaired"`。
* 对于 `use_liger_kernel=True`：
  * 仅支持单个`loss_type`，
  * 不支持`compute_metrics`，
  * 不支持`precompute_ref_log_probs=True`。
* 使用不保留独立 `ref_model` 的 PEFT 模型进行训练时，不支持`sync_ref_model=True`。
* `sync_ref_model=True` 不能与`precompute_ref_log_probs=True` 组合使用。
* `precompute_ref_log_probs=True` 不支持 `IterableDataset`（训练或评估）。

### 多损失组合DPO 训练器支持将多个损失函数与不同的权重相结合，从而实现更复杂的优化策略。这对于实现 MPO（混合偏好优化）等算法特别有用。 MPO 是一种结合了多个优化目标的训练方法，如论文[Enhancing the Reasoning Ability of Multimodal Large Language Models via Mixed Preference Optimization](https://huggingface.co/papers/2411.10442) 中所述。

要组合多个损失，请以列表形式指定损失类型和相应的权重：

```python
# MPO: Combines DPO (sigmoid) for preference and BCO (bco_pair) for quality
training_args = DPOConfig(
    loss_type=["sigmoid", "bco_pair", "sft"],  # loss types to combine
    loss_weights=[0.8, 0.2, 1.0]  # corresponding weights, as used in the MPO paper
)
```

### 模型初始化

您可以直接将`from_pretrained()`方法的kwargs传递给[DPOConfig](/docs/trl/v1.9.2/en/dpo_trainer#trl.DPOConfig)。例如，如果您想以不同的精度加载模型，类似于

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.bfloat16)
```

您可以通过将 `model_init_kwargs={"dtype": torch.bfloat16}` 参数传递给 [DPOConfig](/docs/trl/v1.9.2/en/dpo_trainer#trl.DPOConfig) 来实现。

```python
from trl import DPOConfig

training_args = DPOConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

请注意，支持 `from_pretrained()` 的所有关键字参数。

### 使用 PEFT 训练适配器

我们支持与🤗 PEFT 库紧密集成，允许任何用户方便地训练适配器并在 Hub 上共享它们，而不是训练整个模型。

```python
from datasets import load_dataset
from trl import DPOTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    "Qwen/Qwen3-0.6B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)

trainer.train()
```

您还可以继续训练您的`PeftModel`。为此，首先在 [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 之外加载一个 `PeftModel` 并将其直接传递给训练器，而不传递 `peft_config` 参数。

```python
from datasets import load_dataset
from trl import DPOTrainer
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained("trl-lib/Qwen3-4B-LoRA", is_trainable=True)
dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

trainer = DPOTrainer(
    model=model,
    train_dataset=dataset,
)

trainer.train()
```> [!提示]
> 在训练适配器时，您通常使用比完全微调更高的学习率 (≈1e-5)，因为只学习新参数。
>
> ```python
> DPOConfig(learning_rate=1e-5, ...)
> ```

### 使用 Liger Kernel 进行训练

Liger Kernel 是用于 LLM 训练的 Triton 内核集合，可将多 GPU 吞吐量提高 20%，将内存使用量减少 60%（支持长达 4 倍的上下文），并与 FlashAttention、PyTorch FSDP 和 DeepSpeed 等工具无缝协作。欲了解更多信息，请参阅[Liger Kernel Integration](liger_kernel_integration)。

### DPO 快速实验

RapidFire AI 是一款开源实验引擎，位于 TRL 之上，让您可以同时启动多个 DPO 配置，甚至在单个 GPU 上也是如此。 RapidFire 无需按顺序尝试配置，而是让您**更早地查看所有学习曲线，停止表现不佳的运行，并在飞行中使用新设置克隆有前途的运行**，而无需重新启动。欲了解更多信息，请参阅[RapidFire AI Integration](rapidfire_integration)。

### 不懒惰地训练Unsloth 是一个用于微调和强化学习的开源框架，其训练 LLM（如 Llama、Mistral、Gemma、DeepSeek 等）的速度提高了 2 倍，VRAM 减少了 70%，同时为训练、评估和部署提供了简化的、与 Hugging Face 兼容的工作流程。欲了解更多信息，请参阅[Unsloth Integration](unsloth_integration)。

## 使用 DPO 调用工具

[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer)完全支持具有_工具调用_功能的微调模型。在这种情况下，每个数据集示例应包括：

* 对话消息（提示、选择和拒绝），包括任何工具调用（`tool_calls`）和工具响应（`tool`角色消息）
* `tools` 列中的可用工具列表，通常以 JSON 模式提供

有关预期数据集结构的详细信息，请参阅 [Dataset Format — Tool Calling](dataset_formats#tool-calling) 部分。

## 训练视觉语言模型

[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer)完全支持训练视觉语言模型（VLM）。要训​​练 VLM，请提供具有 `image` 列（每个样本单个图像）或 `images` 列（每个样本图像列表）的数据集。有关预期数据集结构的更多信息，请参阅 [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) 部分。
此类数据集的一个示例是 [RLAIF-V Dataset](https://huggingface.co/datasets/HuggingFaceH4/rlaif-v_formatted) 数据集。

```python
from trl import DPOConfig, DPOTrainer
from datasets import load_dataset

trainer = DPOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=DPOConfig(max_length=None),
    train_dataset=load_dataset("HuggingFaceH4/rlaif-v_formatted", split="train"),
)
trainer.train()
```> [!提示]
> 对于 VLM，截断可能会删除图像标记，从而导致训练期间出现错误。为了避免这种情况，请在[DPOConfig](/docs/trl/v1.9.2/en/dpo_trainer#trl.DPOConfig)中设置`max_length=None`。这使得模型能够处理完整的序列长度，而无需截断图像标记。
>
> ```python
> DPOConfig(max_length=None, ...)
> ```
>
> 仅当您已验证截断不会删除整个数据集的图像标记时，才使用`max_length`。

## DPOTrainer[[trl.DPOTrainer]]

- **型号**（`str`或[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)或`PeftModel`）--
  待训练的模型。可以是：

  - 一个字符串，是在 Huggingface.co 上的模型存储库中托管的预训练模型的 *模型 id*，或者
    包含使用保存的模型权重的*目录*的路径
    [save_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.save_pretrained)，例如`'./my_model_directory/'`。模型已加载
    使用`<ModelArchitecture>.from_pretrained`（其中`<ModelArchitecture>`源自模型
    配置）与`args.model_init_kwargs`中的关键字参数。如果 `dtype` 未指定
    `args.model_init_kwargs`，默认为`float32`。这不同于
    [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained)，其中（自 Transformers v5 起）推断 dtype
    从模型配置。
  - 一个[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)对象。仅支持因果语言模型。
  - 一个`PeftModel`对象。仅支持因果语言模型。
- **参考模型**（[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)，*可选*）--
  用于计算参考对数概率的参考模型。- 如果提供，则直接使用该模型作为参考策略。
  - 如果`None`，训练器会自动使用`model`对应的初始策略，即模型
    在 DPO 培训开始之前声明。
- **参数**（[DPOConfig](/docs/trl/v1.9.2/en/dpo_trainer#trl.DPOConfig)，*可选*）--
  该训练器的配置。如果`None`，则使用默认配置。
- **data_collator**（`DataCollator`，*可选*）--
  用于从已处理的 `train_dataset` 或 `eval_dataset` 的元素列表中形成批次的函数。
  如果模型是语言模型并且则默认为`DataCollatorForPreference`
  `DataCollatorForVisionPreference` 如果模型是视觉语言模型。定制
  整理器必须在填充之前截断序列；训练器不应用整理后截断。
- **train_dataset** (`Dataset` 或 `IterableDataset`) --
  用于训练的数据集。该训练器同时支持[language modeling](#language-modeling)类型和
  [prompt-completion](#prompt-completion)型。样本的格式可以是：

  - [Standard](dataset_formats#standard)：每个样本都包含纯文本。
  - [Conversational](dataset_formats#conversational)：每个样本都包含结构化消息（例如，角色
    和内容）。当`train_dataset`是`IterableDataset`（例如流数据集）时，`max_steps`必须是
  在训练参数中设置，因为无法推断其长度和训练步骤的总数
  需要限制训练循环并配置学习率调度程序。
- **eval_dataset**（`Dataset`、`IterableDataset`、`DatasetDict`、`IterableDatasetDict` 或 `dict[str, Dataset | IterableDataset]`）--
  用于评估的数据集。它必须满足与`train_dataset`相同的要求。
- **处理类**（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)或[ProcessorMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/processors#transformers.ProcessorMixin)，*可选*）--
  处理类用于处理数据。填充边必须设置为“左”。如果`None`，则
  处理类从带有[from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)的模型名称加载。一个
  必须设置填充令牌`tokenizer.pad_token`。如果处理类没有设置填充标记，
  `tokenizer.eos_token` 将用作默认值。
- **计算指标**（`Callable[[EvalPrediction], dict]`，*可选*）--
  将用于计算评估指标的函数。必须采取一个
  [EvalPrediction](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.EvalPrediction) 并返回度量值的字典字符串。经过时
  [SFTConfig](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTConfig) 且 `batch_eval_metrics` 设置为 `True`，您的 `compute_metrics` 函数必须采用布尔值
  `compute_result` 论证。这将在最后一个评估批次后触发，以表明该函数需要计算并返回全局汇总统计信息，而不是累加批次级别
  统计数据。
- **回调**（[TrainerCallback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/callback#transformers.TrainerCallback)列表，*可选*）--
  用于自定义训练循环的回调列表。将它们添加到详细的默认回调列表中
  在[here](https://huggingface.co/docs/transformers/main_classes/callback)。

  如果您想删除使用的默认回调之一，请使用 [remove_callback](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer.remove_callback)
  方法。
- **优化器**（`tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None]`，*可选*，默认为`(None, None)`）--
  包含要使用的优化器和调度器的元组。将默认为您的 `AdamW` 实例
  模型和由[get_linear_schedule_with_warmup](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup)给出的调度器，由`args`控制。
- **量化配置**（[BitsAndBytesConfig](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/quantization#transformers.BitsAndBytesConfig)，*可选*）--
  从模型标识符加载模型时使用的量化配置。与`peft_config`结合
  用于 QLoRA 培训。如果模型已实例化，则忽略。
- **peft_config**（`PeftConfig`，*可选*）--
  PEFT 配置用于包裹模型。如果`None`，则模型未包装。

直接偏好优化 (DPO) 方法的培训师。该算法最初在论文[Direct
Preference Optimization: Your Language Model is Secretly a Reward Model](https://huggingface.co/papers/2305.18290)中提出。
该类是 [Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer) 类的包装器，并继承其所有属性和方法。

示例：

```python
>>> from trl import DPOTrainer
>>> from datasets import load_dataset

>>> dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

>>> trainer = DPOTrainer(
...     model="Qwen/Qwen2.5-0.5B-Instruct",
...     train_dataset=dataset,
... )
>>> trainer.train()
```- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于`True`，加载*args.output_dir* 中由前一个实例保存的最后一个检查点
  `Trainer`。如果存在，训练将从此处加载的模型/优化器/调度器状态恢复。
- **试用**（`optuna.Trial`或`dict[str, Any]`，*可选*）--
  用于超参数搜索的试运行或超参数字典。
- **ignore_keys_for_eval** (`list[str]`，*可选*) --
  模型输出中的键列表（如果它是字典），在以下情况下应忽略这些键：
  收集训练期间评估的预测。`~trainer_utils.TrainOutput`包含全局步数、训练损失和指标的对象。

主要培训切入点。

将保存模型，以便您可以使用`from_pretrained()`重新加载它。

只会从主进程中保存。- **commit_message** (`str`，*可选*，默认为`"End of training"`) --
  推送时要提交的消息。
- **阻塞**（`bool`，*可选*，默认为`True`）--
  函数是否仅在 `git push` 完成时返回。
- **令牌**（`str`，*可选*，默认为`None`）--
  具有写入权限的令牌，可以覆盖 Trainer 的原始参数。
- **修订**（`str`，*可选*）--
  要提交的 git 修订版本。默认为“主”分支的头部。
- **kwargs**（`dict[str, Any]`，*可选*）--
  传递给 `~Trainer.create_model_card` 的其他关键字参数。如果是 `blocking=False`，则推送模型的存储库的 URL，或者跟踪模型的 `Future` 对象
如果`blocking=True`，则提交进度。

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的🤗 模型中心。

## DPOConfig[[trl.DPOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'random'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ":STR |无 = 无"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str |无 = 无"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | STR |无 = 无"}, {"name": "deepspeed", "val": ": dict | STR | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str |无 = 无"}, {"name": "warmup_ratio", "val": ": float |无 = 无"}, {"name": "logging_dir", "val": ": str | None = None"}, {"name": "local_rank", "val": ": int = -1"}, {"name": "model_init_kwargs", "val": ": dict[str, Typing.Any] | None = None"}, {"name": "local_rank", "val": ": int = -1"} STR |无 = 无"}, {"name": "trust_remote_code", "val": ": bool = False"}, {"name": "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "dataset_num_proc", "val": ": int |无 = 无"}, {"name": "max_length", "val": ": int |无 = 1024"}, {"name": "truncation_mode", "val": ": str = 'keep_start'"}, {"name": "padding_free", "val": ": bool = False"}, {"name": "pad_to_multiple_of", "val": ": int | None = None"}, {"name": "precompute_ref_log_probs", "val": ": bool = False"}, {"name": "precompute_ref_batch_size", "val": ": int | None =无"}, {"name": "loss_type", "val": ": list = "}, {"name": "loss_weights", "val": ": list[float] | None = None"}, {"name": "ld_alpha", "val": ": float | None = None"}, {"name": "f_divergence_type", "val": ": str = 'reverse_kl'"}, {"name": "f_alpha_divergence_coef", "val": ": float = 0.5"}, {"name": "label_smoothing", "val": ": float = 0.0"}, {"name": "beta", "val": ": float = 0.1"}, {"name": "use_weighting", "val": ": bool = False"}, {"name": "discopop_tau", "val": ": float = 0.05"}, {"name": "activation_offloading", "val": ": bool = False"}, {"name": "sync_ref_model", "val": ": bool = False"}, {"name": "ref_model_mixup_alpha", "val": ": float = 0.6"}, {"name": "ref_model_sync_steps", "val": ": int = 512"}, {"name": "pad_token", "val": ": str | None = None"}]}>
控制模型的参数- **model_init_kwargs**（`dict[str, Any]`，*可选*）--
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 的关键字参数，在 `model` 时使用
  [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 的参数以字符串形式提供。
- **trust_remote_code**（`bool`，*可选*，默认为`False`）--
  是否允许加载从 Hub 发送自定义 Python 代码的模型和标记器。转发至
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 和
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoProcessor.from_pretrained)。
- **router_aux_loss_coef** (`float`，*可选*，默认为`0.001`) --
  负载平衡辅助损耗系数。仅在训练混合专家时有效
  （教育部）模型；对于其他模型，它不执行任何操作。辅助损失被添加到训练损失中
  重量。设置为`0.0`以禁用它。
- **disable_dropout**（`bool`，*可选*，默认为`True`）--
  是否在模型和参考模型中禁用 dropout。

控制数据预处理的参数- **dataset_num_proc** (`int`，*可选*) --
  用于处理数据集的进程数。
- **max_length**（`int`或`None`，*可选*，默认为`1024`）--
  标记化序列的最大长度。长度超过`max_length`的序列将从左侧截断或
  正确取决于`truncation_mode`。如果`None`，则不应用截断。
- **截断模式**（`str`，*可选*，默认为`"keep_start"`）--
  当序列超过`max_length`时使用的截断模式。唯一支持的值是
  `"keep_start"`。 `"keep_end"` 值已弃用，并将在 v2.0.0 中删除。
- **padding_free**（`bool`，*可选*，默认为`False`）--
  是否通过将批次中的所有序列展平为单个序列来执行前向传递而不进行填充
  连续序列。这通过消除填充开销来减少内存使用。目前，这只是
  FlashAttention 2或3支持，可以有效处理扁平化的批处理结构。
- **pad_to_multiple_of** (`int`，*可选*) --
  如果设置，序列将被填充到该值的倍数。
- **precompute_ref_log_probs**（`bool`，*可选*，默认为`False`）--是否预先计算整个训练数据集的参考模型对数概率
  培训。这可以在训练期间节省内存，因为参考模型不需要保存在
  记忆。
- **precompute_ref_batch_size** (`int`，*可选*) --
  预先计算参考模型对数概率时使用的批量大小。可以将其设置为高于
  训练批量大小以加快预处理速度。如果`None`，则默认为`per_device_train_batch_size`
  培训和`per_device_eval_batch_size`进行评估。

控制训练的参数- **loss_type** (`list[str]`，*可选*，默认为`["sigmoid"]`) --
  使用的损失类型。可能的值为：`'sigmoid'`、`'hinge'`、`'ipo'`、`'exo_pair'`、`'nca_pair'`、
  `'robust'`、`'bco_pair'`、`'sppo_hard'`、`'aot'`、`'aot_unpaired'`、`'apo_zero'`、`'apo_down'`、
  `'discopop'`、`'sft'`、`'sigmoid_norm'`。如果提供了多种损失类型，它们将使用组合
  `loss_weights` 中指定的重量。
- **损失权重**（`list[float]`，*可选*）--
  多损失组合的损失权重列表。组合多种损失类型时使用。示例：`[0.8,
  0.2, 1.0]` for MPO. If not provided, defaults to equal weights (`1.0`) 对于所有损失类型。
- **ld_alpha**（`float`，*可选*）--
  LD-DPO 论文中的 α 参数，它控制详细标记对数概率的权重
  回应。如果`None`，则不对详细部分应用加权，并且损失相当于
  标准 DPO 损失。必须在 [0.0, 1.0] 范围内：`ld_alpha=1.0` 不应用加权，并且 `ld_alpha=0.0` 掩码
  超出共享长度的标记。
- **f_divergence_type** (`str`，*可选*，默认为`"reverse_kl"`) --
  政策和参考之间的 f-分歧正则化器（f-DPO 论文）。可能的值为：`reverse_kl`
  （默认）、`forward_kl`、`js_divergence`、`alpha_divergence`。
- **f_alpha_divergence_coef** (`float`，*可选*，默认为`0.5`) --α-散度 u^-α 正则化器的 α 系数，仅在 `f_divergence_type='alpha_divergence'` 时使用。
- **label_smoothing**（`float`，*可选*，默认为`0.0`）--
  Robust DPO 和 EXO 中使用的标签平滑参数。在 Robust DPO 中，它被解释为概率
  首选项标签被翻转并且必须位于 [0.0, 0.5); a typical value recommended by the Robust
  DPO paper is 0.1. In EXO, it corresponds to the ε label smoothing parameter, for which the paper recommends
  a typical value of 1e-3.
- **beta** (⟦T225⟧, *optional*, defaults to ⟦T226⟧) --
  Parameter controlling the deviation from the reference model. Higher β means less deviation from the
  reference model. For the IPO loss (⟦T227⟧), this value is the regularization parameter denoted
  by τ in the [paper](https://huggingface.co/papers/2310.12036) 中。
- **use_weighting**（`bool`，*可选*，默认为`False`）--
  是否将 WPO 风格的权重 (https://huggingface.co/papers/2406.11827) 应用于偏好对
  策略的长度归一化序列概率。
- **discopop_tau**（`float`，*可选*，默认为`0.05`）--
  DiscoPOP 论文中的 τ/温度参数，控制对数比调制损耗的形状
  使用`loss_type='discopop'`时。论文推荐默认值`discopop_tau=0.05`。
- **activation_offloading**（`bool`，*可选*，默认为`False`）--
  是否将激活卸载到 CPU。
- **sync_ref_model**（`bool`，*可选*，默认为`False`）--
  是否每`ref_model_sync_steps`步同步参考模型与活动模型，使用
  `ref_model_mixup_alpha` 参数。这种同步源于
  [TR-DPO](https://huggingface.co/papers/2404.09656)纸。 `sync_ref_model=True` 尚不兼容
  PEFT 或`precompute_ref_log_probs=True`。- **ref_model_mixup_alpha** （`float`，*可选*，默认为`0.6`）--
  TR-DPO论文中的α参数，控制当前策略和之前策略的混合
  更新时参考政策。参考策略根据等式更新：`π_ref = α *
  π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`。
- **ref_model_sync_steps**（`int`，*可选*，默认为`512`）--
  TR-DPO 论文中的 τ 参数，决定了当前策略与
  参考政策。要使用此参数，必须设置`sync_ref_model=True`。

已弃用的参数

- **pad_token** --

  

  参数 `pad_token` 已弃用，并将在 v2.0.0 版本中删除。套装`tokenizer.pad_token`
  直接将其作为`processing_class`传递给培训师。

  

[DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) 的配置类。

此类仅包含特定于 DPO 培训的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以把这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在
命令行。> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`1e-6`，而不是`5e-5`。

### 奖励功能
https://huggingface.co/docs/trl/v1.9.2/rewards.md