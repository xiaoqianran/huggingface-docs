<!-- huggingface-docs: machine-translated zh-CN from English source -->

#KTO训练师

[⟦T212⟧](https://huggingface.co/models?other=kto,trl)

## 概述

TRL 支持使用 Kahneman-Tversky Optimization (KTO) Trainer 来训练语言模型，如 [Kawin Ethayarajh](https://huggingface.co/kawine)、[Winnie Xu](https://huggingface.co/xwinxu)、[Niklas Muennighoff](https://huggingface.co/Muennighoff)、Dan Jurafsky、[Douwe Kiela](https://huggingface.co/douwekiela) 的论文 [KTO: Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306) 中所述。

论文摘要如下：

> 卡尼曼和特沃斯基的前景理论告诉我们，人类以一种有偏见但明确的方式感知随机变量；例如，人类是出了名的厌恶损失。我们表明，使法学硕士与人类反馈保持一致的目标隐含地包含了许多此类偏差——这些目标（例如，DPO）相对于交叉熵最小化的成功可以部分归因于它们是人类感知的损失函数（HALO）。然而，这些方法赋予人类的效用函数仍然与前景理论文献中的不同。使用人类效用的 Kahneman-Tversky 模型，我们提出了一个 HALO，它直接最大化各代人的效用，而不是像当前方法那样最大化偏好的对数似然。我们将这种方法称为 Kahneman-Tversky Optimization (KTO)，它在 1B 到 30B 的规模上匹配或超过了基于偏好的方法的性能。至关重要的是，KTO 不需要偏好s——仅表示对于给定输入而言输出是期望的还是不期望的二进制信号。这使得它在偏好数据稀缺且昂贵的现实世界中更容易使用。

官方代码可以在[ContextualAI/HALOs](https://github.com/ContextualAI/HALOs)找到。

这种后训练方法是由[Kashif Rasul](https://huggingface.co/kashif)、[Younes Belkada](https://huggingface.co/ybelkada)、[Lewis Tunstall](https://huggingface.co/lewtun)、Pablo Vicente 贡献的，后来由[Albert Villanova del Moral](https://huggingface.co/albertvillanova)重构。

## 快速开始

此示例演示如何使用 KTO 方法训练模型。我们使用[Qwen 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct)作为基础型号。我们使用来自[KTO Mix 14k](https://huggingface.co/datasets/trl-lib/kto-mix-14k)的偏好数据。您可以在此处查看数据集中的数据：

<iframe
  src="https://huggingface.co/datasets/trl-lib/kto-mix-14k/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

以下是训练模型的脚本：

```python
# train_kto.py
from datasets import load_dataset
from trl import KTOConfig, KTOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
train_dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

training_args = KTOConfig(output_dir="Qwen2-0.5B-KTO")
trainer = KTOTrainer(model=model, args=training_args, processing_class=tokenizer, train_dataset=train_dataset)
trainer.train()
```

使用以下命令执行脚本：

```bash
accelerate launch train_kto.py
```

训练分布在 8 个 H100 GPU 上，大约需要 30 分钟。您可以通过查看奖励图来验证训练进度。奖励幅度的增加趋势表明模型正在改进并随着时间的推移产生更好的响应。

![kto qwen2 reward margin](https://huggingface.co/datasets/trl-lib/documentation-images/resolve/main/kto-qwen2-reward-margin.png)

要查看[trained model](https://huggingface.co/trl-lib/Qwen2-0.5B-KTO)的性能，您可以使用[Transformers Chat CLI](https://huggingface.co/docs/transformers/quicktour#chat-with-text-generation-models)。

$ 变形金刚聊天 trl-lib/Qwen2-0.5B-KTO
<quentin_gallouedec>：
最好的编程语言是什么？<trl-lib/Qwen2-0.5B-KTO>：
最佳编程语言可能会根据个人喜好、行业特定要求、技术技能以及对特定用例或任务的熟悉程度而有所不同。以下是一些广泛使用的编程语言，它们被认为是流行和广泛使用的：

为项目选择编程语言时需要考虑以下一些其他因素：

 1 JavaScript：JavaScript 是 Web 的核心，可用于构建 Web 应用程序、API 和交互式前端应用程序，例如 React 和 Angular 等框架。它的语法结构类似于 C、C++ 和 F#，并且易于学习且易于使用，使其成为初学者和专业人士的热门选择。
 2 Java：Java 以其面向对象编程 (OOP) 以及对 Java 8 和 .NET 的支持而闻名，用于开发企业级软件应用程序、高性能游戏以及移动应用程序、游戏开发和桌面应用程序。3 C++：C++以其灵活性和可扩展性而闻名，提供全面的面向对象编程，是高性能计算和其他技术领域的流行选择。它是一个强大的平台，可用于大规模构建现实世界的应用程序和游戏。
 4 Python：Python 由 Guido van Rossum 于 1991 年开发，是一种高级解释型动态类型语言，以其简单性、可读性和多功能性而闻名。

## 预期的数据集类型和格式

KTO 需要 [unpaired preference](dataset_formats#unpaired-preference) 数据集。或者，您可以提供*配对*偏好数据集（也简称为*偏好数据集*）。在这种情况下，训练器将通过分离所选响应和拒绝响应，将 `label = True` 分配给所选完成，将 `label = False` 分配给拒绝响应，自动将其转换为不配对的格式。

[KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 与[standard](dataset_formats#standard) 和[conversational](dataset_formats#conversational) 数据集格式兼容。当提供对话数据集时，培训师将自动将聊天模板应用到数据集。

```python
# Standard format
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}

# Conversational format
unpaired_preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                               "completion": [{"role": "assistant", "content": "It is blue."}],
                               "label": True}
```理论上，数据集应至少包含一项选择的完成和一项拒绝的完成。然而，一些用户*仅*使用选定或仅拒绝的数据成功运行 KTO。如果仅使用拒绝的数据，建议采用保守的学习率。

## 深入研究 KTO 方法

Kahneman-Tversky Optimization (KTO) 是一种训练方法，旨在仅使用关于给定输入的输出是否“理想”或“不理想”的二进制信号来对齐语言模型，而不是使用成对的首选/不首选补全。它借鉴卡尼曼和特沃斯基的前景理论，定义了“人类意识损失”（HALO），它直接最大化几代人的效用，不对称地权衡理想和不理想的例子，以反映人类损失厌恶。

本节详细介绍了 KTO 在实践中的工作原理，涵盖关键步骤：**预处理**和**损失计算**。

### 预处理和标记化在训练期间，每个示例都应包含一个提示、一个`completion`和一个布尔值`label`，指示完成是否需要（`True`）或不需要（`False`）。有关预期格式的更多详细信息，请参阅[Dataset formats](dataset_formats)。
[KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 使用模型的分词器对每个输入进行分词。

### 计算损失

KTO 损失使用的 KL 散度项是通过将每个提示与从批次中其他地方提取的不匹配完成配对来估计的。因此，估计 KL 项的损失类型（除了 `apo_zero_unpaired` 之外的所有损失类型）都需要大于 1 的每设备训练批次大小和顺序采样策略，以便不匹配的对在批次中保持稳定。

KTO（[paper](https://huggingface.co/papers/2402.01306)的方程7）中使用的损失定义如下：

$$
\mathcal{L}_{\mathrm{KTO}}(\theta) = \mathbb{E}_{(x,y)}\!\left[ w(y)\Big(1 - v(x, y)\Big) \right]
$$

其中值 \\( v(x, y) \\) 是$$
v(x, y) = \begin{案例}
\sigma\!\left(\beta\big(\log\frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} - \mathrm{KL}\big)\right) & \text{如果 } y \text{ 是理想的} \\
\sigma\!\left(\beta\big(\mathrm{KL} - \log\frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}\big)\right) & \text{如果 } y \text{ 是不可取的}
\结束{案例}
$$

这里 \\( x \\) 是提示符， \\( y \\) 是补全， \\( \pi_{\theta} \\) 是正在训练的策略模型， \\( \pi_{\mathrm{ref}} \\) 是参考模型， \\( \sigma \\) 是 sigmoid 函数， \\( \beta > 0 \\) 控制与参考模型的偏差， \\( \mathrm{KL} \\) 是估计的 KL 散度项。对于期望的完成，权重 \\( w(y) \\) 是 `desirable_weight` ，对于不期望的完成，权重 `undesirable_weight` ，用于抵消期望和不期望的示例数量之间的不平衡。

#### 损失类型| `loss_type=` |描述 |
| --- | --- |
| `"kto"`（默认）| [KTO](https://huggingface.co/papers/2402.01306) 论文中的 KTO 损失。基于 Kahneman-Tversky 前景理论的人类感知损失 (HALO)，使用估计的 KL 散度项作为参考点，最大化所需完成的效用并最小化不需要的完成的效用。 |
| `"apo_zero_unpaired"` | [APO](https://huggingface.co/papers/2408.06266) 论文中 APO 零损失的未配对变体。它增加了理想完成的可能性并降低了不需要的完成的可能性，而无需估计 KL 散度项。当您认为理想的完成效果优于模型的默认输出时，请使用此损失。 |

## 记录的指标

在培训和评估时，我们记录以下指标：* `global_step`：迄今为止采取的优化器步骤总数。
* `epoch`：当前纪元数，基于数据集迭代。
* `num_tokens`：到目前为止已处理的令牌总数。
* `loss`：当前记录间隔内的平均 KTO 损失。
* `entropy`：模型预测的令牌分布在非屏蔽令牌上的平均熵。
* `kl`：策略和参考模型之间平均估计的KL散度，用作KTO损失中的参考点。
* `learning_rate`：当前学习率，如果使用调度器，则可能会动态变化。
* `grad_norm`：梯度的 L2 范数，在梯度裁剪之前计算。
* `logits/chosen`：模型为所选（所需）完成中的标记分配的平均 logit 值。
* `logits/rejected`：模型分配给被拒绝（不需要的）完成中的标记的平均 logit 值。
* `logps/chosen`：模型分配给所选（所需）完成中的标记的平均对数概率。
* `logps/rejected`：模型分配给被拒绝（不需要的）完成中的标记的平均对数概率。* `rewards/chosen`：为所选（期望的）完成计算的平均隐式奖励，计算方式为 \\( \beta \log \frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} \\)。
* `rewards/rejected`：针对被拒绝（不需要的）完成计算的平均隐式奖励，计算公式为 \\( \beta \log \frac{\pi_{\theta}(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)} \\)。
* `rewards/margins`：选择完成和拒绝完成之间的平均隐性奖励差额。

## 定制

### 兼容性和约束

当前 [KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 实现中有意限制了一些参数组合：

* 对于 `use_liger_kernel=True`：
  * 仅支持`loss_type="kto"`（不支持`"apo_zero_unpaired"`），
  * 不支持`compute_metrics`，
  * 不支持`precompute_ref_log_probs=True`，
  * 不支持 PEFT 型号。
* 使用不保留独立 `ref_model` 的 PEFT 模型进行训练时，不支持 `sync_ref_model=True`。
* `sync_ref_model=True` 不能与`precompute_ref_log_probs=True` 组合使用。
* `precompute_ref_log_probs=True` 不支持 `IterableDataset`（训练或评估）或视觉数据集。
* 估计 KL 散度项的损失类型（除 `"apo_zero_unpaired"` 之外的所有损失类型）都需要 `train_sampling_strategy="sequential"` 以及每设备训练批量大小大于 1。

### 模型初始化您可以直接将`from_pretrained()`方法的kwargs传递给[KTOConfig](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOConfig)。例如，如果您想以不同的精度加载模型，类似于

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B-Instruct", dtype=torch.bfloat16)
```

您可以通过将 `model_init_kwargs={"dtype": torch.bfloat16}` 参数传递给 [KTOConfig](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOConfig) 来实现。

```python
from trl import KTOConfig

training_args = KTOConfig(
    model_init_kwargs={"dtype": torch.bfloat16},
)
```

请注意，支持 `from_pretrained()` 的所有关键字参数。

### 使用 PEFT 训练适配器

我们支持与🤗 PEFT 库紧密集成，允许任何用户方便地训练适配器并在 Hub 上共享它们，而不是训练整个模型。

```python
from datasets import load_dataset
from trl import KTOTrainer
from peft import LoraConfig

dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

trainer = KTOTrainer(
    "Qwen/Qwen2-0.5B-Instruct",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)

trainer.train()
```

您还可以继续训练您的`PeftModel`。为此，首先在 [KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 之外加载一个 `PeftModel` 并将其直接传递给训练器，而不传递 `peft_config` 参数。

> [!提示]
> 在训练适配器时，您通常使用比完全微调更高的学习率，因为只学习新参数。

### 使用 Liger Kernel 进行训练

Liger Kernel 是用于 LLM 训练的 Triton 内核集合，可将多 GPU 吞吐量提高 20%，将内存使用量减少 60%（支持长达 4 倍的上下文），并与 FlashAttention、PyTorch FSDP 和 DeepSpeed 等工具无缝协作。有关更多信息，请参阅[Liger Kernel Integration](liger_kernel_integration)。

## 使用 KTO 调用工具[KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer)完全支持具有_工具调用_功能的微调模型。在这种情况下，每个数据集示例应包括：

* 对话消息（提示和完成），包括任何工具调用（`tool_calls`）和工具响应（`tool`角色消息）
* `tools` 列中的可用工具列表，通常以 JSON 模式提供

有关预期数据集结构的详细信息，请参阅 [Dataset Format — Tool Calling](dataset_formats#tool-calling) 部分。

## 训练视觉语言模型

[KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer)完全支持训练视觉语言模型（VLM）。要训​​练 VLM，请提供具有 `image` 列（每个样本单个图像）或 `images` 列（每个样本图像列表）的数据集。有关预期数据集结构的更多信息，请参阅 [Dataset Format — Vision Dataset](dataset_formats#vision-dataset) 部分。

```python
from trl import KTOConfig, KTOTrainer
from datasets import load_dataset

trainer = KTOTrainer(
    model="Qwen/Qwen2.5-VL-3B-Instruct",
    args=KTOConfig(max_length=None),
    train_dataset=load_dataset("trl-internal-testing/zen-image", "conversational_unpaired_preference", split="train"),
)
trainer.train()
```

> [!提示]
> 对于 VLM，截断可能会删除图像标记，从而导致训练期间出现错误。为了避免这种情况，请在[KTOConfig](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOConfig)中设置`max_length=None`。这使得模型能够处理完整的序列长度，而无需截断图像标记。
>
> ```python
> KTOConfig(max_length=None, ...)
> ```
>
> 仅当您已验证截断不会删除整个数据集的图像标记时，才使用`max_length`。

## 示例脚本

我们提供了一个示例脚本来使用 KTO 方法训练模型。该脚本可在 [⟦T63⟧](https://github.com/huggingface/trl/blob/main/trl/scripts/kto.py) 中找到要使用 [UltraFeedback dataset](https://huggingface.co/datasets/trl-lib/kto-mix-14k) 上的 [Qwen2 0.5B model](https://huggingface.co/Qwen/Qwen2-0.5B-Instruct) 测试 KTO 脚本，请运行以下命令：

```bash
accelerate launch trl/scripts/kto.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/kto-mix-14k \
    --num_train_epochs 1 \
    --output_dir Qwen2-0.5B-KTO
```

## 使用技巧

### 对于混合专家模型：启用辅助损失

如果负载在专家之间平均分配，MOE 的效率最高。  
为了确保我们在偏好调整期间以类似的方式训练 MOE，将负载均衡器的辅助损失添加到最终损失中是有益的。

通过在模型配置中设置`output_router_logits=True`（例如[MixtralConfig](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/mixtral#transformers.MixtralConfig)）来启用此选项。  
要缩放辅助损失对总损失的贡献程度，请在模型配置中使用超参数`router_aux_loss_coef=...`（默认值：`0.001`）。

### 批量大小建议

使用至少为 4 的每步批量大小，以及 16 到 128 之间的有效批量大小。即使您的有效批量大小很大，如果您的每步批量大小很差，那么 KTO 中的 KL 估计也会很差。

### 学习率建议`beta` 的每个选择都有一个在学习性能下降之前可以容忍的最大学习率。对于默认设置`beta = 0.1`，大多数模型的学习率通常不应超过`1e-6`。随着`beta`的减小，学习率也应该相应减小。一般来说，我们强烈建议将学习率保持在`5e-7`和`5e-6`之间。即使数据集很小，我们也建议不要使用超出此范围的学习率。相反，选择更多的纪元以获得更好的结果。

### 数据不平衡

[KTOConfig](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOConfig)的`desirable_weight`和`undesirable_weight`指的是对期望/正例和不良/负例的损失的权重。
默认情况下，它们均为 1。但是，如果其中一种或另一种较多，则应增加不太常见的类型的权重，使 (`desirable_weight` \\(\times\\) 正数数量) 与 (`undesirable_weight` \\(\times\\) 负数数量) 的比率在 1:1 到 4:3 范围内。

## KTOTrainer[[trl.KTOTrainer]]

- **型号**（`str`或[PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)或`PeftModel`）--
  待训练的模型。可以是：- 一个字符串，是在 Huggingface.co 上的模型存储库中托管的预训练模型的 *模型 id*，或者
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
    在 KTO 培训开始前声明。
- **参数**（[KTOConfig](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOConfig)，*可选*）--
  该训练器的配置。如果`None`，则使用默认配置。
- **data_collator**（`DataCollator`，*可选*）--
  用于从已处理的 `train_dataset` 或 `eval_dataset` 的元素列表中形成批次的函数。
  如果模型是语言模型，则默认为`DataCollatorForUnpairedPreference`
  如果模型是视觉语言，则为 `DataCollatorForVisionUnpairedPreference`
  模型。自定义整理器必须在填充之前截断序列；培训师不应用后整理
  截断。
- **train_dataset** (`Dataset` 或 `IterableDataset`) --
  用于训练的数据集。本训练器支持[unpaired preference](#unpaired-preference)型号。的
  样本的格式可以是：

  - [Standard](dataset_formats#standard)：每个样本都包含纯文本。
  - [Conversational](dataset_formats#conversational)：每个样本都包含结构化消息（例如，角色
    和内容）。当`train_dataset`是`IterableDataset`（例如流数据集）时，`max_steps`必须是
  在训练参数中设置，因为无法推断其长度和训练步骤的总数
  需要限制训练循环并配置学习率调度程序。
- **eval_dataset**（`Dataset`、`IterableDataset`、`DatasetDict`、`IterableDatasetDict` 或 `dict[str, Dataset | IterableDataset]`）--
  用于评估的数据集。必须满足与`train_dataset`相同的要求。
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

Kahneman-Tversky Optimization (KTO) 方法的训练器。该算法最初在论文[KTO:
Model Alignment as Prospect Theoretic Optimization](https://huggingface.co/papers/2402.01306)中提出。这个类是一个
包装[Trainer](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.Trainer)类并继承其所有属性和方法。

示例：

```python
>>> from trl import KTOTrainer
>>> from datasets import load_dataset

>>> dataset = load_dataset("trl-lib/kto-mix-14k", split="train")

>>> trainer = KTOTrainer(
...     model="Qwen/Qwen2.5-0.5B-Instruct",
...     train_dataset=dataset,
... )
>>> trainer.train()
```- **resume_from_checkpoint** （`str` 或 `bool`，*可选*）--
  如果是 `str`，则为由 `Trainer` 的先前实例保存的已保存检查点的本地路径。如果一个
  `bool` 且等于 `True`，加载 *args.output_dir* 中由前一个实例保存的最后一个检查点
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

将 `self.model` 和 `self.processing_class` 上传到存储库 `self.args.hub_model_id` 上的 🤗 模型中心。

## KTOConfig[[trl.KTOConfig]]"}, {"name": "batch_eval_metrics", "val": ": bool = False"}, {"name": "save_only_model", "val": ": bool = False"}, {"name": "save_strategy", "val": ": Transformers.trainer_utils.SaveStrategy | str = 'steps'"}, {"name": "save_steps", "val": ": float = 500"}, {"name": "save_on_each_node", "val": ": bool = False"}, {"name": "save_total_limit", "val": ": int |无 = 无"}, {"name": "enable_jit_checkpoint", "val": ": bool = False"}, {"name": "push_to_hub", "val": ": bool = False"}, {"name": "hub_token", "val": ": str |无 = 无"}, {"name": "hub_private_repo", "val": ": bool |无 = 无"}, {"name": "hub_model_id", "val": ": str |无=无”}，{“name”：“hub_strategy”，“val”：“：transformers.trainer_utils.HubStrategy | str = 'every_save'"}, {"name": "hub_always_push", "val": ": bool = False"}, {"name": "hub_revision", "val": ": str |无 = 无"}, {"name": "load_best_model_at_end", "val": ": bool = False"}, {"name": "metric_for_best_model", "val": ": str |无 = 无"}, {"name": "greater_is_better", "val": ": bool |无 = 无"}, {"name": "ignore_data_skip", "val": ": bool = False"}, {"name": "restore_callback_states_from_checkpoint", "val": ": bool = False"}, {"name": "full_决定论", "val": ": bool = False"}, {"name": "seed", "val": ": int = 42"}, {"name": "data_seed", "val": ": int |无 = 无"}, {"name": "use_cpu", "val": ": bool = False"}, {"name": "accelerator_config", "val": ": dict | STR |无=无”}，{“name”：“parallelism_config”，“val”：“：accelerate.parallelism_config.ParallelismConfig |无 = 无"}, {"name": "dataloader_drop_last", "val": ": bool = False"}, {"name": "dataloader_num_workers", "val": ": int = 0"}, {"name": "dataloader_pin_memory", "val": ": bool = True"}, {"name": "dataloader_persistent_workers", "val": ": bool = False"}, {"name": "dataloader_prefetch_factor", "val": ": int | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}, {"name": "label_names", "val": ": list[str] | None = None"}, {"name": "remove_unused_columns", "val": ": bool = True"}无 = 无"}, {"name": "train_sampling_strategy", "val": ": str = 'sequential'"}, {"name": "length_column_name", "val": ": str = 'length'"}, {"name": "ddp_find_unused_pa​​rameters", "val": ": bool |无 = 无"}, {"name": "ddp_bucket_cap_mb", "val": ": int |无 = 无"}, {"name": "ddp_broadcast_buffers", "val": ": bool |无 = 无"}, {"name": "ddp_static_graph", "val": ": bool |无 = 无"}, {"name": "ddp_backend", "val": ": str | None = None"}, {"name": "ddp_timeout", "val": ": int = 1800"}, {"name": "fsdp", "val": ": str | None = None"}, {"name": "fsdp_config", "val": ": dict[str, Typing.Any] | str | None = None"}, {"name": "deepspeed", "val": ": dict | str | None = None"}, {"name": "debug", "val": ": str | list[transformers.debug_utils.DebugOption] = ''"}, {"name": "skip_memory_metrics", "val": ": bool = True"}, {"name": "do_train", "val": ": bool = False"}, {"name": "do_eval", "val": ": bool = False"}, {"name": "do_predict", "val": ": bool = False"}, {"name": "resume_from_checkpoint", "val": ": str | None = None"}, {"name": "warmup_ratio", "val": ": float | None = None"}, {"name": “logging_dir”，“val”：“：str | None = None”}，{“name”：“local_rank”，“val”：“：int = -1”}，{“name”：“model_init_kwargs”，“val”：“：dict [str，typing.Any] | str | None = None”}，{“name”：“trust_remote_code”，“val”：“：bool = False”}，{“name”： "router_aux_loss_coef", "val": ": float = 0.001"}, {"name": "disable_dropout", "val": ": bool = True"}, {"name": "dataset_num_proc", "val": ": int | None = None"}, {"name": "max_length", "val": ": int | None = 1024"}, {“name”：“pad_to_multiple_of”，“val”：“：int | None = None”}，{"name": "precompute_ref_log_probs", "val": ": bool = False"}, {"name": "precompute_ref_batch_size", "val": ": int | None = None"}, {"name": "loss_type", "val": ": str = 'kto'"}, {"name": "beta", "val": ": float = 0.1"}, {"name": "desirable_weight", "val": ": float = 1.0"}, {"name": "undesirable_weight", "val": ": float = 1.0"}, {"name": "activation_offloading", "val": ": bool = False"}, {"name": "sync_ref_model", "val": ": bool = False"}, {"name": "ref_model_mixup_alpha", "val": ": float = 0.6"}, {"name": "ref_model_sync_steps", "val": ": int = 512"}]}>
控制模型的参数- **model_init_kwargs**（`dict[str, Any]`，*可选*）--
  [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 的关键字参数，在 `model` 时使用
  [KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 的参数以字符串形式提供。
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
  标记化序列的最大长度。长度超过`max_length`的序列将从右侧截断。
  如果`None`，则不应用截断。
- **pad_to_multiple_of** (`int`，*可选*) --
  如果设置，序列将被填充到该值的倍数。
- **precompute_ref_log_probs**（`bool`，*可选*，默认为`False`）--
  是否预先计算整个训练数据集的参考模型对数概率
  培训。这可以在训练期间节省内存，因为参考模型不需要保存在
  记忆。
- **precompute_ref_batch_size** (`int`，*可选*) --
  预先计算参考模型对数概率时使用的批量大小。可以将其设置为高于
  训练批量大小以加快预处理速度。如果`None`，则默认为`per_device_train_batch_size`
  培训和`per_device_eval_batch_size`进行评估。

控制训练的参数

- **loss_type** (`str`，*可选*，默认为`"kto"`) --
  使用的损失类型。可能的值为：- `"kto"`：[KTO](https://huggingface.co/papers/2402.01306)论文中的KTO损失。
  - `"apo_zero_unpaired"`：APO 的未配对变体-零损失
    [APO](https://huggingface.co/papers/2408.06266)纸。

- **beta**（`float`，*可选*，默认为`0.1`）--
  控制与参考模型的偏差的参数。较高的β意味着较小的偏差
  参考模型。
- **desirable_weight**（`float`，*可选*，默认为`1.0`）--
  期望的损失通过这个因素来衡量，以抵消期望和不期望的对的数量不等。
- **undesirable_weight**（`float`，*可选*，默认为`1.0`）--
  不期望的损失通过该因子来衡量，以抵消期望和不期望的对的数量不等。
- **activation_offloading**（`bool`，*可选*，默认为`False`）--
  是否将激活卸载到 CPU。
- **sync_ref_model**（`bool`，*可选*，默认为`False`）--
  是否每`ref_model_sync_steps`步同步参考模型与活动模型，使用
  `ref_model_mixup_alpha` 参数。这种同步源于
  [TR-DPO](https://huggingface.co/papers/2404.09656)纸。 `sync_ref_model=True` 尚不兼容
  PEFT 或`precompute_ref_log_probs=True`。
- **ref_model_mixup_alpha** （`float`，*可选*，默认为`0.6`）--TR-DPO论文中的α参数，控制当前策略和之前策略的混合
  更新时参考政策。参考策略根据等式更新：`π_ref = α *
  π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`。
- **ref_model_sync_steps**（`int`，*可选*，默认为`512`）--
  TR-DPO 论文中的 τ 参数，决定了当前策略与
  参考政策。要使用此参数，必须设置`sync_ref_model=True`。

[KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) 的配置类。

此类仅包含特定于 KTO 训练的参数。有关训练参数的完整列表，
请参阅[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)文档。请注意，此类中的默认值可能
与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)中的不同。

使用[HfArgumentParser](https://huggingface.co/docs/transformers/v5.14.1/en/internal/trainer_utils#transformers.HfArgumentParser)我们可以将这个类变成
[argparse](https://docs.python.org/3/library/argparse#module-argparse) 可以在上指定的参数
命令行。> [!注意]
> 这些参数的默认值与[TrainingArguments](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/trainer#transformers.TrainingArguments)不同：
> - `logging_steps`：默认为`10`，而不是`500`。
> - `gradient_checkpointing`：默认为`True`，而不是`False`。
> - `bf16`：如果未设置`fp16`，则默认为`True`，而不是`False`。
> - `learning_rate`：默认为`1e-6`，而不是`5e-5`。
> - `train_sampling_strategy`：默认为`"sequential"`，而不是`"random"`。损失类型
> 估计 KL 散度项（除了 `"apo_zero_unpaired"` 之外的所有项）需要顺序
> 采样，因为每个示例的 KL 补全都是针对其邻居预先计算的
> 固定订单批次；任何其他策略都会破坏这种配对。

### DPO 培训师
https://huggingface.co/docs/trl/v1.9.2/dpo_trainer.md