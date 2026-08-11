<!-- huggingface-docs: machine-translated zh-CN from English source -->

# X-LoRA

LoRA 专家混合 ([X-LoRA](https://huggingface.co/papers/2402.07148)) 是一种 PEFT 方法，可基于高粒度（令牌、层、序列）缩放矩阵实现 LoRA 专家的稀疏或密集混合。这利用了冻结的 LoRA 适配器和冻结的基础模型，大大减少了需要微调的参数数量。

X-LoRA 的独特之处在于其多功能性：它可以应用于任何带有 LoRA 适配器的`transformers` 基本模型。这意味着，尽管混合了专家策略，但不必对模型代码进行任何更改。

下图演示了每个标记的不同提示的缩放比例如何变化。这突出显示了随着生成的进行和序列创建新的上下文不同适配器的激活。

![Token-by-token scalings](https://github.com/EricLBuehler/xlora/raw/master/res/token_by_token_scalings.gif)

对于每个步骤，X-LoRA 要求基础模型运行两次：首先，在没有任何 LoRA 适配器的情况下获取隐藏状态，其次，隐藏状态用于计算应用于 LoRA 适配器的缩放比例，并且模型第二次运行。第二次运行的输出是模型步骤的结果。最终，X-LoRA 允许模型通过双前向传递方案反映其知识，并动态地重新配置架构。

论文摘要是：

*我们报告了专家策略的混合，使用基于低秩适应（LoRA）的深层逐层标记级方法来创建微调的大型语言模型。从一组预先训练的 LoRA 适配器开始，我们的门控策略使用隐藏状态动态混合适应层，从而允许生成的 X-LoRA 模型利用不同的功能并创建以前从未使用过的深层分层组合来解决任务。该设计的灵感来自于普遍性和多样性的生物学原理，其中神经网络构建块可以在不同的层次表现形式中重复使用。因此，X-LoRA模型可以轻松地实现任何现有的大型语言模型（LLM），而无需修改底层结构。我们开发了定制的 X-LoRA 模型，该模型提供科学功能，包括正向/反向分析任务和增强的推理能力，重点关注生物材料分析、蛋白质力学和设计。这项工作的影响包括获得具有强大领域知识和跨知识领域集成能力的易于扩展和适应性强的模型。我们拥有生物学、数学、推理、仿生材料、力学和材料、化学、蛋白质生物物理学、力学和基于分子特性的量子力学方面的专家，开展一系列以物理学为重点的案例研究。我们研究知识回忆、蛋白质力学正向/反向任务、蛋白质设计、对抗性代理建模（包括本体知识图构建）以及分子设计。该模型不仅能够对蛋白质的纳米力学特性或量子力学分子特性进行定量预测，还能对结果进行推理并正确预测解释不同分子行为的可能机制。*。

请将 X-LoRA 引用为：
```bibtex
@article{10.1063/5.0203126,
    author = {Buehler, Eric L. and Buehler, Markus J.},
    title = "{X-LoRA: Mixture of low-rank adapter experts, a flexible framework for large language models with applications in protein mechanics and molecular design}",
    journal = {APL Machine Learning},
    volume = {2},
    number = {2},
    pages = {026119},
    year = {2024},
    month = {05},
    abstract = "{We report a mixture of expert strategy to create fine-tuned large language models using a deep layer-wise token-level approach based on low-rank adaptation (LoRA). Starting with a set of pre-trained LoRA adapters, our gating strategy uses the hidden states to dynamically mix adapted layers, allowing the resulting X-LoRA model to draw upon different capabilities and create never-before-used deep layer-wise combinations to solve tasks. The design is inspired by the biological principles of universality and diversity, where neural network building blocks are reused in different hierarchical manifestations. Hence, the X-LoRA model can be easily implemented for any existing large language model without a need for modifications of the underlying structure. We develop a tailored X-LoRA model that offers scientific capabilities, including forward/inverse analysis tasks and enhanced reasoning capability, focused on biomaterial analysis, protein mechanics, and design. The impact of this work includes access to readily expandable and adaptable models with strong domain knowledge and the capability to integrate across areas of knowledge. Featuring experts in biology, mathematics, reasoning, bio-inspired materials, mechanics and materials, chemistry, protein biophysics, mechanics, and quantum-mechanics based molecular properties, we conduct a series of physics-focused case studies. We examine knowledge recall, protein mechanics forward/inverse tasks, protein design, adversarial agentic modeling including ontological knowledge graph construction, and molecular design. The model is capable not only of making quantitative predictions of nanomechanical properties of proteins or quantum mechanical molecular properties but also reasoning over the results and correctly predicting likely mechanisms that explain distinct molecular behaviors.}",
    issn = {2770-9019},
    doi = {10.1063/5.0203126},
    url = {https://doi.org/10.1063/5.0203126},
    eprint = {https://pubs.aip.org/aip/aml/article-pdf/doi/10.1063/5.0203126/19964043/026119\_1\_5.0203126.pdf},
}
```

# API

## XLoraConfig[[peft.XLoraConfig]]

#### peft.XLoraConfig[[peft.XLoraConfig]]

```python
peft.XLoraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, hidden_size: int = None, adapters: dict[str, str] = None, enable_softmax: bool = True, enable_softmax_topk: bool = False, layerwise_scalings: bool = False, xlora_depth: int = 1, xlora_size: int = 2048, xlora_dropout_p: float = 0.2, use_trainable_adapters: bool = False, softmax_temperature: float = 1.0, top_k_lora: Optional[int] = None, scaling_pass_value: float = 0.0, global_scaling_weight: float = 1.0)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/config.py#L25)

**参数：**

hidden_size (`int`) ：基础模型的隐藏大小。适配器 (`dict`) ：适配器名称到 LoRA 适配器 ID 的映射，根据 PeftModel.load_adapter。 *它们将自动加载*，作为 LoRA 专家使用。使用 from_pretrained 时，将新的适配器字典作为关键字参数传递。

enable_softmax（`bool`，*可选*，默认为`True`）：启用X-LoRA分类器的softmax应用。

enable_softmax_topk（`bool`，*可选*，默认为`False`）：为top-k LoRA适配器启用softmax应用程序。与`enable_softmax`互斥，并且仅当`top_k_lora`存在时才必须设置。

softmax_温度（`float`，*可选*，默认为 1.0）：Softmax 温度，较低会产生更清晰的预测

layerwise_scalings（`bool`，*可选*，默认为`False`）：如果为 True，则为每个 LoRA 适配器（每层）生成缩放。如果这是 False，那么缩放将相同地广播到每一层。

top_k_lora (`int`，*可选*，默认为 None) ：稀疏选择 top_k LoRA 专家，而不是默认的密集方法。

xlora_depth (`int`，*可选*，默认为 1) ：X-LoRA 分类器的深度。

xlora_size (`int`，*可选*，默认为2048)：X-LoRA分类器的隐藏大小，如果`xlora_depth=1`则不相关。xlora_dropout_p（`float`，*可选*，默认为0.2）：X-LoRA分类器的丢弃概率，如果`xlora_depth=1`则不相关。

use_trainable_adapters（`bool`，*可选*，默认为 False）：使适配器可训练。

scaling_pass_value (`float`，*可选*，默认为0)：缩放通道值。

global_scaling_weight (`float`，*可选*，默认为 1) ：每个 LoRA 适配器的输出相乘的权重。

这是存储`XLoraModel`配置的配置类。当配置重新加载时，
`adapters` 字段的路径被忽略，有利于保存的适配器。因此，只有按键才重要
正在加载。

## XLoraModel[[peft.XLoraModel]]

#### peft.XLoraModel[[peft.XLoraModel]]

```python
peft.XLoraModel(model: nn.Module, config: Union[dict[str, XLoraConfig], XLoraConfig], adapter_name: str, torch_device: Optional[str] = None, ephemeral_gpu_offload: bool = False, autocast_adapter_dtype: bool = True, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L156)

**参数：**

model (`torch.nn.Module`) ：要适配的模型。

config ([XLoraConfig](/docs/peft/v0.20.0/en/package_reference/xlora#peft.XLoraConfig)) ：Lora 模型的配置。

adapter_name (`str`) ：适配器的名称，不影响 LoRA 适配器名称。

**返回：** `torch.nn.Module`

X-LoRA 模型。

根据预训练的 Transformer 模型创建 X-LoRA（LoRA 专家混合）模型。目前，这个X-LoRA
实现仅适用于具有变压器架构的模型。该方法在 https://huggingface.co/papers/2402.07148 中有详细描述。

示例：
```py
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
>>> from peft import XLoraConfig, get_peft_model, prepare_model_for_kbit_training

>>> model_config = AutoConfig.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
>>> config = XLoraConfig(
...     task_type="CAUSAL_LM",
...     hidden_size=model_config.hidden_size,
...     xlora_depth=4,
...     adapters={
...         "adapter_1": "./path/to/the/checkpoint/",
...         "adapter_2": "./path/to/the/checkpoint/",
...         "adapter_n": "./path/to/the/checkpoint/",
...     },
... )
>>> int8_config = BitsAndBytesConfig(load_in_8bit=True)
>>> model = AutoModelForCausalLM.from_pretrained(
...     "mistralai/Mistral-7B-Instruct-v0.1",
...     trust_remote_code=True,
...     attn_implementation="flash_attention_2",
...     device_map="cuda:0",
...     torch_dtype=torch.bfloat16,
...     quantization_config=int8_config,
... )
>>> model = prepare_model_for_kbit_training(model)
>>> xlora_model = get_peft_model(model, config)
```

####clear_scalings_log[[peft.XLoraModel.clear_scalings_log]]

```python
clear_scalings_log()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L511)

清除缩放日志。

####disable_scalings_logging[[peft.XLoraModel.disable_scalings_logging]]

```python
disable_scalings_logging()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L504)

禁用缩放日志记录，而不清除日志。

####enable_scalings_logging[[peft.XLoraModel.enable_scalings_logging]]

```python
enable_scalings_logging()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L497)

启用缩放日志记录。

#### get_bucketed_scalings_log[[peft.XLoraModel.get_bucketed_scalings_log]]

```python
get_bucketed_scalings_log()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L518)

返回分桶缩放，按 seq_len 分桶。每个值由位置（第一个）和
相关的张量。这些位置与相关的张量配对，并给出缩放中的位置
日志。

#### get_global_scaling_weight[[peft.XLoraModel.get_global_scaling_weight]]

```python
get_global_scaling_weight()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L474)

获取全局 LoRA 权重。

#### get_latest_scalings[[peft.XLoraModel.get_latest_scalings]]

```python
get_latest_scalings()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L481)

返回最新的缩放预测，如果没有预测缩放，则返回 None。张量的形状
（batch_size、seq_len、n_layers、n_classes）。

#### get_scalings_log[[peft.XLoraModel.get_scalings_log]]

```python
get_scalings_log()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L488)返回包含缩放日志的列表的浅表副本（仅复制列表本身而不复制张量）。
编辑列表不会更改基础日志。张量的形状为 (batch_size, seq_len, n_layers,
n_类）。 seq_len 的暗度可能随输入维度的不同而变化。

#### set_global_scaling_weight[[peft.XLoraModel.set_global_scaling_weight]]

```python
set_global_scaling_weight(weight: float)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L458)

设置全局 LoRA 权重，这是一个与每个 LoRA 适配器的输出相乘的标量。默认情况下为 1。
反映在配置中。

#### set_scaling_pass_value[[peft.XLoraModel.set_scaling_pass_value]]

```python
set_scaling_pass_value(value: float | None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L466)

设置缩放通道值，即在缩放通道期间设置缩放的值。如果值为 None，则
缩放传递值将为 1/n，其中 n 是适配器的数量。

#### set_topk_lora[[peft.XLoraModel.set_topk_lora]]

```python
set_topk_lora(value: Optional[int])
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/xlora/model.py#L450)

稀疏选择指定的 top_k LoRA 专家，而不是默认的密集方法。设置为 None 以使用密集。
这反映在配置中。

### PEANuT：使用权重感知神经调整器进行参数高效适应
https://huggingface.co/docs/peft/v0.20.0/package_reference/peanut.md