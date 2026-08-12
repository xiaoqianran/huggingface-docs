<!-- huggingface-docs: machine-translated zh-CN from English source -->

# DDUF

    
     

## 概述

DDUF（**D**DUF 的 **D**iffusion **U**nified **F**ormat）是一种扩散模型的单文件格式，旨在通过将所有模型组件打包到单个文件中来统一不同的模型分发方法和减重格式。它与语言无关，并且可以从远程位置进行解析，而无需下载整个文件。

该作品的灵感来自于[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)格式。

查看 [DDUF](https://huggingface.co/DDUF) 组织，开始使用 DDUF 中一些最流行的扩散模型。

> [!提示]
> 我们张开双臂欢迎贡献！
>
> 为了创建广泛采用的文件格式，我们需要社区的早期反馈。没有什么是一成不变的，我们重视每个人的意见。您的用例未涵盖吗？请通过DDUF组织[discussions](https://huggingface.co/spaces/DDUF/README/discussions/2)告知我们。

其主要特点如下。1. **单文件**包装。
2. 基于**ZIP 文件格式**，利用现有工具。
3. 无压缩，确保**`mmap`兼容性**，实现快速加载和保存。
4. **与语言无关**：工具可以用 Python、JavaScript、Rust、C++ 等实现。
5. **HTTP 友好**：可以使用 HTTP Range 请求远程获取元数据和文件结构。
6. **灵活**：每个模型组件都存储在自己的目录中，遵循当前的 Diffusers 结构。
7. **安全**：使用[Safetensors](https://huggingface.co/docs/diffusers/using-diffusers/other-formats#safetensors)作为减重格式，并禁止嵌套目录，防止ZIP炸弹。

## 技术规格从技术上讲，`.dduf` 文件**是** [⟦T9⟧ archive](https://en.wikipedia.org/wiki/ZIP_(file_format))。通过构建普遍支持的文件格式，我们确保已经存在强大的工具。然而，为了满足扩散模型的要求，需要强制执行一些约束：
- 数据必须以未压缩的方式存储（标志`0`），允许使用内存映射进行延迟加载。
- 数据必须使用ZIP64协议存储，可以保存4GB以上的文件。
- 存档只能包含 `.json`、`.safetensors`、`.model` 和 `.txt` 文件。
- `model_index.json` 文件必须存在于存档的根目录中。它必须包含带有有关模型及其组件的元数据的键值映射。
- 每个组件必须存储在自己的目录中（例如，`vae/`、`text_encoder/`）。嵌套文件必须使用 UNIX 样式路径分隔符 (`/`)。
- 每个目录必须对应`model_index.json`索引中的一个组件。
- 每个目录必须包含一个 json 配置文件（`config.json`、`tokenizer_config.json`、`preprocessor_config.json`、`scheduler_config.json` 之一）。
- 禁止子目录。

想要检查您的文件是否有效？使用此空间查看：https://huggingface.co/spaces/DDUF/dduf-check。

＃＃ 用法`huggingface_hub` 提供了在 Python 中处理 DDUF 文件的工具。它包括用于验证文件完整性的内置规则以及用于读取和导出 DDUF 文件的帮助程序。我们的目标是看到该工具在 Python 生态系统中得到采用，例如在 `diffusers` 集成中。可以为其他语言（JavaScript、Rust、C++ 等）开发类似的工具。

### 如何读取DDUF文件？

将路径传递给`read_dduf_file`以读取DDUF文件。仅读取元数据，这意味着这是一个轻量级调用，不会占用您的内存。在下面的示例中，我们认为您已经在本地下载了 [⟦T27⟧](https://huggingface.co/DDUF/FLUX.1-dev-DDUF/blob/main/FLUX.1-dev.dduf) 文件。

```python
>>> from huggingface_hub import read_dduf_file

# Read DDUF metadata
>>> dduf_entries = read_dduf_file("FLUX.1-dev.dduf")
```

`read_dduf_file` 返回一个映射，其中每个条目对应于 DDUF 存档中的一个文件。文件由 `DDUFEntry` 数据类表示，其中包含原始 DDUF 文件中的文件名、偏移量和条目长度。此信息对于读取其内容而不加载整个文件很有用。在实践中，您不必处理低级阅读，而是依赖助手。

例如，以下是如何加载`model_index.json`内容：
```python
>>> import json
>>> json.loads(dduf_entries["model_index.json"].read_text())
{'_class_name': 'FluxPipeline', '_diffusers_version': '0.32.0.dev0', '_name_or_path': 'black-forest-labs/FLUX.1-dev', ...
```对于二进制文件，您需要使用 `as_mmap` 访问原始字节。这将返回字节作为原始文件上的内存映射。内存映射允许您仅读取所需的字节，而无需将所有内容加载到内存中。例如，以下是加载安全张量权重的方法：

```python
>>> import safetensors.torch
>>> with dduf_entries["vae/diffusion_pytorch_model.safetensors"].as_mmap() as mm:
...     state_dict = safetensors.torch.load(mm) # `mm` is a bytes object
```

> [!提示]
> `as_mmap` 必须在上下文管理器中使用才能受益于内存映射属性。

### 如何写入DDUF文件？

将文件夹路径传递给`export_folder_as_dduf`以导出DDUF文件。

```python
# Export a folder as a DDUF file
>>> from huggingface_hub import export_folder_as_dduf
>>> export_folder_as_dduf("FLUX.1-dev.dduf", folder_path="path/to/FLUX.1-dev")
```

该工具扫描文件夹，添加相关条目并确保导出的文件有效。如果在此过程中出现任何问题，则会引发 `DDUFExportError`。

为了获得更大的灵活性，请使用 [`export_entries_as_dduf`] 显式指定要包含在最终 DDUF 文件中的文件列表：

```python
# Export specific files from the local disk.
>>> from huggingface_hub import export_entries_as_dduf
>>> export_entries_as_dduf(
...     dduf_path="stable-diffusion-v1-4-FP16.dduf",
...     entries=[ # List entries to add to the DDUF file (here, only FP16 weights)
...         ("model_index.json", "path/to/model_index.json"),
...         ("vae/config.json", "path/to/vae/config.json"),
...         ("vae/diffusion_pytorch_model.fp16.safetensors", "path/to/vae/diffusion_pytorch_model.fp16.safetensors"),
...         ("text_encoder/config.json", "path/to/text_encoder/config.json"),
...         ("text_encoder/model.fp16.safetensors", "path/to/text_encoder/model.fp16.safetensors"),
...         # ... add more entries here
...     ]
... )
```

如果您已经将模型保存在磁盘上，`export_entries_as_dduf` 就可以很好地工作。但是，如果您在内存中加载了一个模型并希望将其直接序列化到 DDUF 文件中，该怎么办？ `export_entries_as_dduf` 通过提供 Python `generator` 来告诉您如何迭代地序列化数据：

```python
(...)

# Export state_dicts one by one from a loaded pipeline
>>> def as_entries(pipe: DiffusionPipeline) -> Generator[Tuple[str, bytes], None, None]:
...     # Build a generator that yields the entries to add to the DDUF file.
...     # The first element of the tuple is the filename in the DDUF archive. The second element is the content of the file.
...     # Entries will be evaluated lazily when the DDUF file is created (only 1 entry is loaded in memory at a time)
...     yield "vae/config.json", pipe.vae.to_json_string().encode()
...     yield "vae/diffusion_pytorch_model.safetensors", safetensors.torch.save(pipe.vae.state_dict())
...     yield "text_encoder/config.json", pipe.text_encoder.config.to_json_string().encode()
...     yield "text_encoder/model.safetensors", safetensors.torch.save(pipe.text_encoder.state_dict())
...     # ... add more entries here

>>> export_entries_as_dduf(dduf_path="my-cool-diffusion-model.dduf", entries=as_entries(pipe))
```

### 使用 Diffusers 加载 DDUF 文件

Diffusers 具有 DDUF 文件的内置集成。以下是如何从集线器上存储的检查点加载管道的示例：```py
from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "DDUF/FLUX.1-dev-DDUF", dduf_file="FLUX.1-dev.dduf", torch_dtype=torch.bfloat16
).to("cuda")
image = pipe(
    "photo a cat holding a sign that says Diffusers", num_inference_steps=50, guidance_scale=3.5
).images[0]
image.save("cat.png")
```

## 常见问题解答

### 为什么要在 ZIP 之上构建？

ZIP 有几个优点：
- 普遍支持的文件格式
- 无需额外的阅读依赖
- 内置文件索引
- 广泛的语言支持

### 为什么不在存档开头使用带有目录的 TAR？

参见这个[comment](https://github.com/huggingface/huggingface_hub/pull/2692#issuecomment-2519863726)的解释。

### 为什么不压缩？

- 启用大文件的直接内存映射
- 确保一致且可预测的远程文件访问
- 防止文件读取期间的 CPU 开销
- 保持与安全张量的兼容性

### 我可以修改 DDUF 文件吗？

不。目前，DDUF 文件被设计为不可变的。要更新模型，请创建新的 DDUF 文件。
	
### 哪些框架/应用程序支持 DDUF？
	
- [Diffusers](https://github.com/huggingface/diffusers)
	
我们不断接触其他库和框架。如果您有兴趣为您的项目添加支持，请在 [DDUF org](https://huggingface.co/spaces/DDUF/README/discussions) 中打开讨论。

### 空间自定义域
https://huggingface.co/docs/hub/spaces-custom-domain.md