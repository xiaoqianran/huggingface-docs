<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 量化

量化用更少的位数表示数据，这使其成为减少内存使用和加速推理的有用技术，尤其是在涉及大型语言模型 (LLM) 时。量化模型的方法有多种，包括：

* 优化使用[AWQ](https://hf.co/papers/2306.00978)算法量化哪些模型权重
* 使用[GPTQ](https://hf.co/papers/2210.17323)算法独立量化权重矩阵的每一行
* 使用 [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 库量化至 8 位和 4 位精度
* 使用 [AQLM](https://huggingface.co/papers/2401.06118) 算法量化至低至 2 位精度

然而，模型量化后通常不会针对下游任务进行进一步训练，因为由于权重和激活的精度较低，训练可能不稳定。但由于 PEFT 方法仅添加“额外”可训练参数，因此您可以使用顶部的 PEFT 适配器来训练量化模型！即使对于在单个 GPU 上训练最大的模型，将量化与 PEFT 相结合也是一个很好的策略。例如，[QLoRA](https://hf.co/papers/2305.14314)是将模型量化为4位，然后用LoRA对其进行训练的方法。此方法允许您在单个 48GB GPU 上微调 65B 参数模型！在本指南中，您将了解如何将模型量化为 4 位并使用 LoRA 对其进行训练。

## 量化模型

[bitsandbytes](https://github.com/TimDettmers/bitsandbytes) 是一个集成了 Transformers 的量化库。通过这种集成，您可以将模型量化为 8 或 4 位，并通过配置 [BitsAndBytesConfig](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/quantization#transformers.BitsAndBytesConfig) 类启用许多其他选项。例如，您可以：

* 设置 `load_in_4bit=True` 在加载模型时将模型量化为 4 位
* 设置`bnb_4bit_quant_type="nf4"`为从正态分布初始化的权重使用特殊的4位数据类型
* 设置`bnb_4bit_use_double_quant=True`使用嵌套量化方案来量化已经量化的权重
* 设置`bnb_4bit_compute_dtype=torch.bfloat16`使用bfloat16以加快计算速度

```py
import torch
from transformers import BitsAndBytesConfig

config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

将 `config` 传递给 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 方法。

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=config)
```

接下来，您应该调用[prepare_model_for_kbit_training()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.prepare_model_for_kbit_training)函数对量化模型进行预处理以进行训练。

```py
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
```

现在量化模型已准备就绪，让我们设置一个配置。

## 洛拉配置

使用以下参数创建 [LoraConfig](/docs/peft/v0.20.0/en/package_reference/lora#peft.LoraConfig)（或选择您自己的参数）：

```py
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

然后使用 [get_peft_model()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.get_peft_model) 函数根据量化模型和配置创建 [PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)。

```py
from peft import get_peft_model

model = get_peft_model(model, config)
```

您已准备好使用您喜欢的任何训练方法进行训练！

### LoftQ初始化[LoftQ](https://hf.co/papers/2310.08659)初始化LoRA权重，使量化误差最小化，并且可以提高训练量化模型时的性能。要开始使用，请关注[these instructions](https://github.com/huggingface/peft/tree/main/examples/loftq_finetuning)。

一般来说，为了让 LoftQ 发挥最佳效果，建议使用 LoRA 定位尽可能多的层，因为那些未定位的层无法应用 LoftQ。这意味着通过`LoraConfig(..., target_modules="all-linear")`很可能会给出最好的结果。另外，当使用 4 位量化时，您应该在量化配置中使用 `nf4` 作为量化类型，即 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。

一般来说，为了让 LoftQ 发挥最佳效果，建议使用 LoRA 定位尽可能多的层，因为那些未定位的层无法应用 LoftQ。这意味着通过`LoraConfig(..., target_modules="all-linear")`很可能会给出最好的结果。另外，当使用 4 位量化时，您应该在量化配置中使用 `nf4` 作为量化类型，即 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")`。

目前有两种应用 LoftQ 的方式：1. 加载非量化基础模型，使用您想要的量化级别应用 LoftQ（例如，nf4 的`bits=4`）并
   保存生成的适配器。然后使用以下方法量化基本模型：
   `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` 并将 LoftQ 初始化的适配器加载到顶部
2.加载量化基础模型，使用LoRA（无loftq）初始化PEFT模型并使用`replace_lora_weights_loftq`
   通过从权重文件流式传输非量化基础模型的参考权重来应用 LoftQ 初始化
   （在下一节中进一步解释）

#### 使用 `replace_lora_weights_loftq` 进行即时 LoftQ 应用程序

应用 LoftQ 初始化的一种更简单但更有限的方法是使用便捷函数`replace_lora_weights_loftq`。这将量化的 PEFT 模型作为输入，并用 LoftQ 初始化的对应项替换 LoRA 权重。

```python
from peft import replace_lora_weights_loftq
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(load_in_4bit=True, ...)
base_model = AutoModelForCausalLM.from_pretrained(..., quantization_config=bnb_config)
# note: don't pass init_lora_weights="loftq" or loftq_config!
lora_config = LoraConfig(task_type="CAUSAL_LM")
peft_model = get_peft_model(base_model, lora_config)
replace_lora_weights_loftq(peft_model)
```

`replace_lora_weights_loftq` 还允许您传递 `callback` 参数，以便您更好地控制应该修改或不修改哪些层，这从经验上可以大大改善结果。要查看更详细的示例，请查看 [this notebook](https://github.com/huggingface/peft/blob/main/examples/loftq_finetuning/LoftQ_weight_replacement.ipynb)。`replace_lora_weights_loftq` 仅实现了 LoftQ 的一个迭代步骤。这意味着仅更新 LoRA 权重，而不是迭代更新 LoRA 权重和量化基础模型权重。这可能会导致性能降低，但优点是我们可以使用从基本模型派生的原始量化权重，而不必保留修改后的量化权重的额外副本。这种权衡是否值得取决于用例。

目前，`replace_lora_weights_loftq` 有以下额外限制：

- 模型文件必须存储为 `safetensors` 文件。
- 仅支持bitsandbytes 4 位量化。

### QLoRA 式培训

QLoRA 向 Transformer 架构中的所有线性层添加可训练权重。由于这些线性层的属性名称可能因架构而异，因此将 `target_modules` 设置为 `"all-linear"` 以将 LoRA 添加到所有线性层：

```py
config = LoraConfig(target_modules="all-linear", ...)
```

## GPTQ 量化

您可以在 [GPT-QModel](https://github.com/ModelCloud/GPTQModel) 和 Transformers [GPTQ](https://huggingface.co/docs/transformers/quantization/gptq) 文档中了解有关基于 GPTQ 的 `[2, 3, 4, 8]` 位量化的更多信息。 PEFT通过GPT-QModel支持GPTQ后训练。

```bash
# GPT-QModel install
pip install "gptqmodel>=7.0.0"
```

```py
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)

gptq_config = GPTQConfig(bits=4, group_size=128, dataset="wikitext2", tokenizer=tokenizer)

quantized_model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)

# save quantized model
quantized_model.save_pretrained("./opt-125m-gptq")
tokenizer.save_pretrained("./opt-125m-gptq")
```

量化后，您可以使用 PEFT API 对 GPTQ 模型进行后训练。

## AQLM 量化语言模型的加性量化（[AQLM](https://huggingface.co/papers/2401.06118)）是一种大型语言模型压缩方法。它将多个权重一起量化并利用它们之间的相互依赖性。 AQLM 将 8-16 个权重组表示为多个矢量代码的总和。这使得它能够将模型压缩至低至 2 位，并且精度损失相当低。

由于 AQLM 量化过程的计算成本较高，因此建议使用预量化模型。可用型号的部分列表可以在官方 aqlm [repository](https://github.com/Vahe1994/AQLM) 中找到。

这些模型支持 LoRA 适配器调整。要调整量化模型，您需要安装`aqlm`推理库：`pip install aqlm>=1.0.2`。微调的 LoRA 适配器应单独保存，因为不可能将它们与 AQLM 量化权重合并。

```py
quantized_model = AutoModelForCausalLM.from_pretrained(
    "BlackSamorez/Mixtral-8x7b-AQLM-2Bit-1x16-hf-test-dispatch",
    dtype="auto", device_map="auto", low_cpu_mem_usage=True,
)

peft_config = LoraConfig(...)

quantized_model = get_peft_model(quantized_model, peft_config)
```

您可以参考[Google Colab](https://colab.research.google.com/drive/12GTp1FCj5_0SnnNQH18h_2XFh9vS_guX?usp=sharing)示例来了解AQLM+LoRA微调的概述。

## EETQ 量化您还可以对 EETQ 量化模型执行 LoRA 微调。 [EETQ](https://github.com/NetEase-FuXi/EETQ)包提供了简单有效的方法来执行8位量化，据称比`LLM.int8()`算法更快。首先，确保您有一个与 EETQ 兼容的 Transformer 版本（例如，通过从最新的 pypi 或源代码安装它）。

```py
import torch
from transformers import EetqConfig

config = EetqConfig("int8")
```

将 `config` 传递给 [from_pretrained](https://huggingface.co/docs/transformers/v5.14.1/en/model_doc/auto#transformers.AutoModelForCausalLM.from_pretrained) 方法。

```py
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1", quantization_config=config)
```

创建一个 `LoraConfig` 并将其传递给 `get_peft_model`：

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, config)
```

## HQQ 量化

使用大型机器学习模型的半二次量化 ([HQQ](https://mobiusml.github.io/hqq_blog/)) 量化的模型支持 LoRA 适配器调整。要调整量化模型，您需要安装 `hqq` 库：`pip install hqq`。

```python
from hqq.engine.hf import HQQModelForCausalLM

device = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"

quantized_model = HQQModelForCausalLM.from_quantized(save_dir_or_hfhub, device=device)
peft_config = LoraConfig(...)
quantized_model = get_peft_model(quantized_model, peft_config)
```

或者使用与 HQQ 兼容的转换器版本（例如，通过从最新的 pypi 或源代码安装）。

```python
from transformers import HqqConfig, AutoModelForCausalLM

quant_config = HqqConfig(nbits=4, group_size=64)
quantized_model = AutoModelForCausalLM.from_pretrained(save_dir_or_hfhub, device_map=device_map, quantization_config=quant_config)
peft_config = LoraConfig(...)
quantized_model = get_peft_model(quantized_model, peft_config)
```

## torchao（PyTorch架构优化）

PEFT 支持使用 [torchao](https://github.com/pytorch/ao)（“ao”）进行 int8 量化的模型量化。

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, TorchAoConfig
from torchao.quantization import Int8WeightOnlyConfig

model_id = ...
quantization_config = TorchAoConfig(quant_type=Int8WeightOnlyConfig())
base_model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config)
peft_config = LoraConfig(...)
model = get_peft_model(base_model, peft_config)
```

### 注意事项：- 使用最新版本的 torchao (>= v0.4.0) 和 Transformers (> 4.42)。
- 目前仅支持线性图层。
- 目前不支持`quant_type = "int4_weight_only"`。
- `NF4` 尚未在 Transformer 中实现，因此也不支持。
- DoRA 目前仅适用于`quant_type = "int8_weight_only"`。
- 与 LoRA 一起使用时，明确支持 torchao。然而，当 torchao 量化一个层时，它的类不会改变，只会改变底层张量的类型。因此，除 LoRA 之外的 PEFT 方法通常也适用于 torchao，即使没有明确支持。但请注意，**合并仅适用于 LoRA 和 `quant_type = "int8_weight_only"`**。如果您使用不同的 PEFT 方法或数据类型，合并可能会导致错误，即使没有，结果仍然不正确。

## INC 量化英特尔神经压缩器（[INC](https://github.com/intel/neural-compressor)）支持各种设备的模型量化，
包括 Intel Gaudi 加速器（也称为 HPU 设备）。您可以对已经完成的模型进行 LoRA 微调
使用 INC 进行量化。要将 INC 与 PyTorch 模型一起使用，请使用以下命令安装库：`pip install neural-compressor[pt]`。
可以通过以下单步量化工作流程将 HPU 设备的模型量化为 FP8 精度：

```python
import torch
from neural_compressor.torch.quantization import FP8Config, convert, finalize_calibration, prepare
quant_configs = {
    ...
}
config = FP8Config(**quant_configs)
```

将配置传递给`prepare`方法，运行推理以收集校准统计数据，然后调用`finalize_calibration`
和 `convert` 将模型量化到 FP8 精度的方法：

```python
model = prepare(model, config)
# Run inference to collect calibration statistics
...
# Finalize calibration and convert the model to FP8 precision
finalize_calibration(model)
model = convert(model)
# Load PEFT LoRA adapter as usual
...
```

演示如何将 PEFT LoRA 适配器加载到 HPU 的 INC 量化 FLUX 文本到图像模型中的示例
提供设备[here](https://github.com/huggingface/peft/blob/main/examples/stable_diffusion/inc_flux_lora_hpu.py)。

### 注意事项：

- INC 量化模型目前不支持`merge()` 和`unmerge()` 方法。
- 目前，加载 PEFT 适配器时仅支持**线性** INC 量化层。

## 其他支持的 PEFT 方法

除了 LoRA 之外，以下 PEFT 方法也支持量化：

- **VeRA**（支持位和字节量化）
- **AdaLoRA**（支持位和字节和 GPTQ 量化）
- **(IA)³**（支持位和字节量化）

## 变压器引擎 (TE) LoRAPEFT 支持[NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/) 层（`te.pytorch.Linear`、`te.pytorch.LayerNormLinear` 和`te.pytorch.LayerNormMLP`）之上的 LoRA 适配器。 TE 层使用 FP8 和融合内核来加速 Transformer 训练，附加 LoRA 适配器可让您高效地参数微调 TE 加速模型。

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    task_type="TOKEN_CLS",
)
model = get_peft_model(te_model, config)
```

安装 Transformer Engine 后，PEFT 会自动将匹配层分派到 `TeLinear` 适配器 - 无需额外配置。

### 注意事项

- TE 层尚不支持`merge()` 和`unmerge()`。
- TE 层不支持 DoRA。

完整的 ESM2 令牌分类示例可在 `examples/lora_finetuning_transformer_engine/` 下找到。

## 后续步骤

如果您有兴趣了解有关量化的更多信息，以下内容可能会有所帮助：

* 了解有关 QLoRA 的更多详细信息，并在 [Making LLMs even more accessible with bitsandbytes, 4-bit quantization and QLoRA](https://huggingface.co/blog/4bit-transformers-bitsandbytes) 博客文章中查看一些有关其影响的基准。
* 在 Transformers [Quantization](https://hf.co/docs/transformers/main/quantization) 指南中了解有关不同量化方案的更多信息。

### 为 PEFT 做出贡献
https://huggingface.co/docs/peft/v0.20.0/developer_guides/contributing.md