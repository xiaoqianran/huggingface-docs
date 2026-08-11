<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型量化

## `bitsandbytes` 集成

Accelerate 为您的模型带来 `bitsandbytes` 量化。现在，您可以使用几行代码加载任何 8 位或 4 位的 pytorch 模型。

如果你想使用`bitsandbytes`的变形金刚模型，你应该遵循这个[documentation](https://huggingface.co/docs/transformers/main_classes/quantization)。 

要了解有关 `bitsandbytes` 量化工作原理的更多信息，请查看关于 [8-bit quantization](https://huggingface.co/blog/hf-bitsandbytes-integration) 和 [4-bit quantization](https://huggingface.co/blog/4bit-transformers-bitsandbytes) 的博客文章。

### 先决条件
您将需要安装以下要求：

- 安装`bitsandbytes`库
```bash
pip install bitsandbytes
```
对于非cuda设备，可以参考bitsandbytes安装指南[here](https://huggingface.co/docs/bitsandbytes/main/en/installation#multi-backend)。

- 从源安装最新的`accelerate`
```bash
pip install git+https://github.com/huggingface/accelerate.git
```
- 安装`minGPT`和`huggingface_hub`来运行示例
```bash
git clone https://github.com/karpathy/minGPT.git
pip install minGPT/
pip install huggingface_hub
```

### 它是如何工作的

首先，我们需要初始化我们的模型。为了节省内存，我们可以使用上下文管理器[init_empty_weights()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.init_empty_weights)初始化一个空模型。 

让我们从 minGPT 库中获取 GPT2 模型。
```py
from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    empty_model = GPT(model_config)
```

然后，我们需要获取模型权重的路径。该路径可以是 state_dict 文件（例如“pytorch_model.bin”）或包含分片检查点的文件夹。 

```py
from huggingface_hub import snapshot_download
weights_location = snapshot_download(repo_id="marcsun13/gpt2-xl-linear-sharded")
```

最后，您需要使用 [BnbQuantizationConfig](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.BnbQuantizationConfig) 设置量化配置。

以下是 8 位量化的示例：
```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_8bit=True, llm_int8_threshold = 6)
```

下面是 4 位量化的示例：
```py
from accelerate.utils import BnbQuantizationConfig
bnb_quantization_config = BnbQuantizationConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
```要使用所选配置量化空模型，您需要使用 [load_and_quantize_model()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.utils.load_and_quantize_model)。 

```py
from accelerate.utils import load_and_quantize_model
quantized_model = load_and_quantize_model(empty_model, weights_location=weights_location, bnb_quantization_config=bnb_quantization_config)
```

### 保存和加载8位模型

您可以使用 [save_model()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_model) 加速保存 8 位模型。 

```py
from accelerate import Accelerator
accelerate = Accelerator()
new_weights_location = "path/to/save_directory"
accelerate.save_model(quantized_model, new_weights_location)

quantized_model_from_saved = load_and_quantize_model(empty_model, weights_location=new_weights_location, bnb_quantization_config=bnb_quantization_config, device_map = "auto")
```

请注意，当前不支持 4 位模型序列化。

### 将模块卸载到 cpu 和磁盘 

如果 GPU 上没有足够的空间来在 GPU 上存储整个模型，您可以将某些模块卸载到 CPU/磁盘。
这在底层使用了大模型推理。查看此[documentation](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)了解更多详情。 

对于 8 位量化，所选模块将转换为 8 位精度。 

对于 4 位量化，所选模块将保留在用户传入 `BnbQuantizationConfig` 的`torch_dtype` 中。  当 4 位序列化成为可能时，我们将添加对这些卸载模块转换为 4 位的支持。 

 您只需要传递一个自定义的 `device_map` 即可卸载 cpu/磁盘上的模块。需要时，卸载模块将被调度到 GPU 上。这是一个例子：

```py
device_map = {
    "transformer.wte": 0,
    "transformer.wpe": 0,
    "transformer.drop": 0,
    "transformer.h": "cpu",
    "transformer.ln_f": "disk",
    "lm_head": "disk",
}
```
### 微调量化模型无法在这些模型上执行纯 8 位或 4 位训练。但是，您可以利用参数高效微调方法 (PEFT) 来训练这些模型，并在其之上训练例如适配器。请查看 [peft](https://github.com/huggingface/peft) 库了解更多详细信息。

目前，您无法在任何量化模型之上添加适配器。但是，借助 Transformers 模型适配器的官方支持，您可以对量化模型进行微调。如果您想微调 Transformers 模型，请按照此 [documentation](https://huggingface.co/docs/transformers/main_classes/quantization) 进行操作。查看此[demo](https://colab.research.google.com/drive/1VoYNfYDKcKRQRor98Zbf2-9VQTtGJ24k?usp=sharing)，了解如何微调 4 位 Transformers 模型。 

请注意，加载模型进行训练时不需要传递`device_map`。它会自动将您的模型加载到 GPU 上。请注意，`device_map=auto` 只能用于推理。

### 示例演示 - 在 Google Colab 上运行 GPT2 1.5b

查看 Google Colab [demo](https://colab.research.google.com/drive/1T1pOgewAWVpR9gKpaEWw4orOrzPFb3yM?usp=sharing) 在 GPT2 模型上运行量化模型。 GPT2-1.5B模型检查点位于使用6GB内存的FP32中。量化后，8位模块使用1.6GB，4位模块使用1.2GB。

### 将本地 SGD 与 Accelerate 结合使用
https://huggingface.co/docs/accelerate/v1.14.0/usage_guides/local_sgd.md