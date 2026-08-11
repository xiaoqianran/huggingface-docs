<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 多肌球

[Polytropon](https://hf.co/papers/2202.13914) 是一个多任务模型，其“库存”中有许多不同的 LoRA 适配器。该模型通过路由功能从库存中学习适配器的正确组合，从而为特定任务选择最佳的模块子集。 PEFT 还支持 Polytropon 的[Multi-Head Adapter Routing (MHR)](https://hf.co/papers/2211.03831)，它通过更精细地组合适配器头来构建和改进路由功能。适配器头被分成不相交的块，并且为每个块学习不同的路由功能，从而提供更多的表现力。

论文摘要是：*模块化设计鼓励神经模型解开和重组知识的不同方面，以更系统地推广到新任务。在这项工作中，我们假设每个任务都与（可能很小）库存中的潜在离散技能子集相关联。反过来，技能对应于参数高效（稀疏/低秩）模型参数化。通过共同学习这些和任务技能分配矩阵，每个任务的网络被实例化为主动技能参数的平均值。为了支持跨任务的技能的重要软划分，我们尝试了一系列归纳偏差，例如印度巴菲特过程先验和双速学习率。我们在两个主要设置上评估我们的潜在技能模型：1）多任务强化学习，用于遵循 BabyAI 平台 8 个级别的基础指令； 2）在 CrossFit（包含 160 个 NLP 任务的基准）上对预训练的文本到文本生成模型进行少量调整。我们发现，与完全使用的基线相比，网络的模块化设计显着提高了强化学习中的样本效率和监督学习中的小样本泛化能力。共享的、特定于任务的或有条件生成的参数，其中知识在任务之间纠缠在一起。此外，我们还展示了离散技能如何有助于可解释性，因为它们产生明确的任务层次结构。*

论文摘要是：

*用于跨任务泛化的参数高效微调（PEFT）包括在对测试任务进行几次调整之前对多任务训练集进行预训练适配器。 Polytropon [Ponti et al., 2023] (Poly) 联合学习适配器库存和路由函数，该函数在预训练和小样本适应期间为每个任务选择适配器的（可变大小）子集。在本文中，我们研究了适配器路由在其成功中所扮演的角色，并根据我们的发现设计了新的变体。首先，我们的直觉是更细粒度的路由提供了更多的表现力。因此，我们提出了 MHR（多头路由），它结合了适配器参数的子集，并且在可比较的参数预算下优于 Poly；通过仅微调路由功能而不是适配器（MHR-z），我们以极高的参数效率实现了具有竞争力的性能。其次，我们发现 Poly/MHR 性能是以下因素的结果更好的多任务优化，而不是像之前假设的那样促进适配器重组和局部适应的模块化归纳偏差。事实上，我们发现 MHR 在任务之间表现出比任何其他方法更高的梯度对齐。由于这意味着路由仅在多任务预训练期间至关重要，因此我们提出了 MHR-mu，它丢弃路由并在几次适应期间微调预训练适配器的平均值。这确立了 MHR-mu 作为单适配器微调的有效方法。*。

如果你想在没有训练的情况下尝试路由，你可以查看[Arrow LoRA variant](./lora#Arrow)。

# API

## PolyConfig[[peft.PolyConfig]]

#### pft.PolyConfig[[peft.PolyConfig]]

```python
peft.PolyConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, modules_to_save: Optional[list[str]] = None, init_weights: bool = True, poly_type: Literal['poly'] = 'poly', n_tasks: int = 1, n_skills: int = 4, n_splits: int = 1)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/poly/config.py#L25)

**参数：**

r (`int`) ：Poly 中每个 Lora 的注意力维度。

target_modules (`Union[List[str],str]`) ：要应用 Poly 的模块的名称。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。module_to_save (`List[str]`) ：除了 Poly 层之外，要设置为可训练并保存在最终检查点中的模块列表。

init_weights (bool) : 是否执行Poly权重的初始化。

poly_type (`Literal["poly"]`) ：要使用的 Poly 模块的变体。目前仅支持“poly”。

n_tasks (`int`) ：多任务场景中的任务数量。

n_skills (`int`) ：每个 Poly 层中的技能 (LoRA) 数量。

n_splits (`int`)：Poly 层的每个 LoRA 内的分割数。大于 1 的值表示使用多头路由 (MHR)。

这是存储[PolyModel](/docs/peft/v0.20.0/en/package_reference/poly#peft.PolyModel)配置的配置类。
- [Polytropon (Poly)](https://huggingface.co/papers/2202.13914)
- [Multi-Head Routing (MHR)](https://huggingface.co/papers/2211.03831)

## PolyModel[[peft.PolyModel]]

#### peft.PolyModel[[peft.PolyModel]]

```python
peft.PolyModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/poly/model.py#L28)

### P 调音
https://huggingface.co/docs/peft/v0.20.0/package_reference/p_tuning.md