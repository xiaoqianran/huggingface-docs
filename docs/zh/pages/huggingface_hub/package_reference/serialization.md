<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 序列化

`huggingface_hub` 提供帮助程序以标准化方式保存和加载 ML 模型权重。这部分库仍在开发中，将在未来的版本中得到改进。目标是协调在 Hub 上保存和加载权重的方式，既消除库之间的代码重复，又建立一致的约定。

## DDUF 文件格式

DDUF 是一种专为扩散模型设计的文件格式。它允许将运行模型的所有信息保存在单个文件中。该作品的灵感来自于[GGUF](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)格式。 `huggingface_hub` 提供保存和加载 DDUF 文件的帮助程序，确保遵循文件格式。

> [!警告]
> 这是解析器的一个非常早期的版本。 API 和实现可以在不久的将来发展。
>
> 解析器目前只进行很少的验证。有关文件格式的更多详细信息，请查看 https://github.com/huggingface/huggingface.js/tree/main/packages/dduf。

### 如何写入DDUF文件？

以下是如何使用 [export_folder_as_dduf()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.export_folder_as_dduf) 导出包含扩散模型不同部分的文件夹：

```python
# Export a folder as a DDUF file
>>> from huggingface_hub import export_folder_as_dduf
>>> export_folder_as_dduf("FLUX.1-dev.dduf", folder_path="path/to/FLUX.1-dev")
```

为了获得更大的灵活性，您可以使用 [export_entries_as_dduf()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.export_entries_as_dduf) 并传递要包含在最终 DDUF 文件中的文件列表：

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
````entries`参数还支持传递路径或字节的可迭代对象。如果您有一个已加载的模型并希望将其直接序列化为 DDUF 文件，而不必先将每个组件序列化到磁盘，然后再将其序列化为 DDUF 文件，那么这可能非常有用。以下是如何将 `StableDiffusionPipeline` 序列化为 DDUF 的示例：

```python
# Export state_dicts one by one from a loaded pipeline 
>>> from diffusers import DiffusionPipeline
>>> from typing import Generator, Tuple
>>> import safetensors.torch
>>> from huggingface_hub import export_entries_as_dduf
>>> pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
... # ... do some work with the pipeline

>>> def as_entries(pipe: DiffusionPipeline) -> Generator[Tuple[str, bytes], None, None]:
...     # Build a generator that yields the entries to add to the DDUF file.
...     # The first element of the tuple is the filename in the DDUF archive (must use UNIX separator!). The second element is the content of the file.
...     # Entries will be evaluated lazily when the DDUF file is created (only 1 entry is loaded in memory at a time)
...     yield "vae/config.json", pipe.vae.to_json_string().encode()
...     yield "vae/diffusion_pytorch_model.safetensors", safetensors.torch.save(pipe.vae.state_dict())
...     yield "text_encoder/config.json", pipe.text_encoder.config.to_json_string().encode()
...     yield "text_encoder/model.safetensors", safetensors.torch.save(pipe.text_encoder.state_dict())
...     # ... add more entries here

>>> export_entries_as_dduf(dduf_path="stable-diffusion-v1-4.dduf", entries=as_entries(pipe))
```

**注意：** 在实践中，`diffusers`提供了直接序列化DDUF文件中的管道的方法。上面的代码片段仅作为示例。

### 如何读取DDUF文件？

```python
>>> import json
>>> import safetensors.torch
>>> from huggingface_hub import read_dduf_file

# Read DDUF metadata
>>> dduf_entries = read_dduf_file("FLUX.1-dev.dduf")

# Returns a mapping filename <> DDUFEntry
>>> dduf_entries["model_index.json"]
DDUFEntry(filename='model_index.json', offset=66, length=587)

# Load model index as JSON
>>> json.loads(dduf_entries["model_index.json"].read_text())
{'_class_name': 'FluxPipeline', '_diffusers_version': '0.32.0.dev0', '_name_or_path': 'black-forest-labs/FLUX.1-dev', 'scheduler': ['diffusers', 'FlowMatchEulerDiscreteScheduler'], 'text_encoder': ['transformers', 'CLIPTextModel'], 'text_encoder_2': ['transformers', 'T5EncoderModel'], 'tokenizer': ['transformers', 'CLIPTokenizer'], 'tokenizer_2': ['transformers', 'T5TokenizerFast'], 'transformer': ['diffusers', 'FluxTransformer2DModel'], 'vae': ['diffusers', 'AutoencoderKL']}

# Load VAE weights using safetensors
>>> with dduf_entries["vae/diffusion_pytorch_model.safetensors"].as_mmap() as mm:
...     state_dict = safetensors.torch.load(mm)
```

### 助手[[huggingface_hub.export_entries_as_dduf]]

#### Huggingface_hub.export_entries_as_dduf[[huggingface_hub.export_entries_as_dduf]]

```python
huggingface_hub.export_entries_as_dduf(dduf_path: str | os.PathLike, entries: Iterable)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_dduf.py#L160)

**参数：**

dduf_path（`str`或`os.PathLike`）：要写入的DDUF文件的路径。

Entrys (`Iterable[tuple[str, Union[str, Path, bytes]]]`) ：要写入 DDUF 文件的可迭代条目。每个条目都是一个包含文件名和内容的元组。文件名应该是 DDUF 存档中文件的路径。内容可以是字符串或表示本地磁盘上文件路径的pathlib.Path，也可以直接将内容表示为字节。

**加薪：** `-`

- - -- `DDUFExportError`：如果导出过程中出现任何问题（例如条目名称无效、缺少“model_index.json”等）。从可迭代的条目写入 DDUF 文件。

这是一个比 [export_folder_as_dduf()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.export_folder_as_dduf) 更低级别的帮助程序，在序列化数据时提供更大的灵活性。
特别是，在将数据导出到 DDUF 文件之前，您不需要将数据保存在磁盘上。

示例：
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

```python
# Export state_dicts one by one from a loaded pipeline
>>> from diffusers import DiffusionPipeline
>>> from typing import Generator, Tuple
>>> import safetensors.torch
>>> from huggingface_hub import export_entries_as_dduf
>>> pipe = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
... # ... do some work with the pipeline

>>> def as_entries(pipe: DiffusionPipeline) -> Generator[tuple[str, bytes], None, None]:
...     # Build a generator that yields the entries to add to the DDUF file.
...     # The first element of the tuple is the filename in the DDUF archive (must use UNIX separator!). The second element is the content of the file.
...     # Entries will be evaluated lazily when the DDUF file is created (only 1 entry is loaded in memory at a time)
...     yield "vae/config.json", pipe.vae.to_json_string().encode()
...     yield "vae/diffusion_pytorch_model.safetensors", safetensors.torch.save(pipe.vae.state_dict())
...     yield "text_encoder/config.json", pipe.text_encoder.config.to_json_string().encode()
...     yield "text_encoder/model.safetensors", safetensors.torch.save(pipe.text_encoder.state_dict())
...     # ... add more entries here

>>> export_entries_as_dduf(dduf_path="stable-diffusion-v1-4.dduf", entries=as_entries(pipe))
```

#### Huggingface_hub.export_folder_as_dduf[[huggingface_hub.export_folder_as_dduf]]

```python
huggingface_hub.export_folder_as_dduf(dduf_path: str | os.PathLike, folder_path: str | os.PathLike)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_dduf.py#L249)

**参数：**

dduf_path (`str` 或 `os.PathLike`) ：要写入的 DDUF 文件的路径。

folder_path（`str`或`os.PathLike`）：包含扩散模型的文件夹的路径。

将文件夹导出为 DDUF 文件。

A在引擎盖下使用[export_entries_as_dduf()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.export_entries_as_dduf)。

示例：
```python
>>> from huggingface_hub import export_folder_as_dduf
>>> export_folder_as_dduf(dduf_path="FLUX.1-dev.dduf", folder_path="path/to/FLUX.1-dev")
```

#### Huggingface_hub.read_dduf_file[[huggingface_hub.read_dduf_file]]

```python
huggingface_hub.read_dduf_file(dduf_path: os.PathLike | str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_dduf.py#L91)

**参数：**

dduf_path（`str`或`os.PathLike`）：要读取的DDUF文件的路径。

**退货：** `dict[str, DDUFEntry]`

按文件名索引的[DDUFEntry](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.DDUFEntry)字典。

**加薪：** `-`

- - -- `DDUFCorruptedFileError`：如果 DDUF 文件已损坏（即不遵循 DDUF 格式）。

读取 DDUF 文件并返回条目字典。

仅读取元数据，不会将数据加载到内存中。

示例：
```python
>>> import json
>>> import safetensors.torch
>>> from huggingface_hub import read_dduf_file

# Read DDUF metadata
>>> dduf_entries = read_dduf_file("FLUX.1-dev.dduf")

# Returns a mapping filename <> DDUFEntry
>>> dduf_entries["model_index.json"]
DDUFEntry(filename='model_index.json', offset=66, length=587)

# Load model index as JSON
>>> json.loads(dduf_entries["model_index.json"].read_text())
{'_class_name': 'FluxPipeline', '_diffusers_version': '0.32.0.dev0', '_name_or_path': 'black-forest-labs/FLUX.1-dev', ...

# Load VAE weights using safetensors
>>> with dduf_entries["vae/diffusion_pytorch_model.safetensors"].as_mmap() as mm:
...     state_dict = safetensors.torch.load(mm)
```

#### Huggingface_hub.DDUFEntry[[huggingface_hub.DDUFEntry]]

```python
huggingface_hub.DDUFEntry(filename: str, length: int, offset: int, dduf_path: Path)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_dduf.py#L36)

**参数：**filename (str) ：DDUF 存档中的文件名。

offset (int) ：文件在 DDUF 存档中的偏移量。

length (int) ：DDUF 存档中文件的长度。

dduf_path (str) ：DDUF 存档的路径（供内部使用）。

表示 DDUF 文件中的文件条目的对象。

请参阅[read_dduf_file()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.read_dduf_file)了解如何读取DDUF文件。

#### as_mmap[[huggingface_hub.DDUFEntry.as_mmap]]

```python
as_mmap()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_dduf.py#L58)

将文件作为内存映射文件打开。

对于直接从文件加载安全张量很有用。

示例：
```py
>>> import safetensors.torch
>>> with entry.as_mmap() as mm:
...     tensors = safetensors.torch.load(mm)
```

#### read_text[[huggingface_hub.DDUFEntry.read_text]]

```python
read_text(encoding: str = 'utf-8')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_dduf.py#L75)

以文本形式读取文件。

对于“.txt”和“.json”条目很有用。

示例：
```py
>>> import json
>>> index = json.loads(entry.read_text())
```

### 错误[[huggingface_hub.errors.DDUFError]]

#### Huggingface_hub.errors.DDUFError[[huggingface_hub.errors.DDUFError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L509)

与 DDUF 格式相关的错误的基本异常。

#### Huggingface_hub.errors.DDUFCorruptedFileError[[huggingface_hub.errors.DDUFCorruptedFileError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L513)

DDUF 文件损坏时抛出异常。

#### Huggingface_hub.errors.DDUFExportError[[huggingface_hub.errors.DDUFExportError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L517)

DDUF 导出期间错误的基本异常。#### Huggingface_hub.errors.DDUFInvalidEntryNameError[[huggingface_hub.errors.DDUFInvalidEntryNameError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/errors.py#L521)

当条目名称无效时抛出异常。

## 保存张量

`serialization`模块的主要助手将火炬`nn.Module`作为输入并将其保存到磁盘。它处理保存共享张量的逻辑（参见[safetensors explanation](https://huggingface.co/docs/safetensors/torch_shared_tensors)）以及将状态字典分割成分片的逻辑，在底层使用[split_torch_state_dict_into_shards()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.split_torch_state_dict_into_shards)。目前仅支持`torch`框架。

如果你想保存状态字典（例如层名称和相关张量之间的映射）而不是`nn.Module`，你可以使用[save_torch_state_dict()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.save_torch_state_dict)，它提供相同的功能。例如，如果您想在保存之前将自定义逻辑应用于状态字典，这非常有用。

### save_torch_model[[huggingface_hub.save_torch_model]]

#### Huggingface_hub.save_torch_model[[huggingface_hub.save_torch_model]]

```python
huggingface_hub.save_torch_model(model: torch.nn.Module, save_directory: str | pathlib.Path, filename_pattern: str | None = None, force_contiguous: bool = True, max_shard_size: int | str = '5GB', metadata: dict[str, str] | None = None, safe_serialization: bool = True, is_main_process: bool = True, shared_tensors_to_discard: list[str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L43)

**参数：**

model (`torch.nn.Module`) ：保存在磁盘上的模型。

save_directory（`str`或`Path`）：保存模型的目录。filename_pattern (`str`, *可选*) ：生成保存模型的文件名的模式。 Pattern 必须是可以使用 `filename_pattern.format(suffix=...)` 格式化的字符串，并且必须包含关键字 `suffix` 默认为 `"model{suffix}.safetensors"` 或 `pytorch_model{suffix}.bin`，具体取决于 `safe_serialization` 参数。

force_contigious (`boolean`, *可选*) ：强制将 state_dict 保存为连续张量。这对模型的正确性没有影响，但如果专门为此选择张量的布局，则可能会改变性能。默认为 `True`。

max_shard_size（`int`或`str`，*可选*）：每个分片的最大大小，以字节为单位。默认为 5GB。

元数据（`dict[str, str]`，*可选*）：与模型一起保存的额外信息。将为每个删除的张量添加一些元数据。这些信息不足以恢复整个共享​​结构，但可能有助于理解事物。

safe_serialization (`bool`, *可选*) ：是否保存为安全张量，这是默认行为。如果`False`，则分片将保存为pickle。出于安全原因，建议使用安全序列化。另存为 pickle 已被弃用，并将在未来版本中删除。is_main_process (`bool`, *可选*) ：调用此进程的进程是否为主进程。在像 TPU 这样的分布式训练中并且需要从所有进程调用此函数时非常有用。在这种情况下，仅在主进程上设置 `is_main_process=True` 以避免竞争条件。默认为 True。

shared_tensors_to_discard (`list[str]`, *可选*) ：保存共享张量时要删除的张量名称列表。如果未提供并且检测到共享张量，它将按字母顺序删除名字。

将给定的火炬模型保存到磁盘，处理分片和共享张量问题。

另请参阅[save_torch_state_dict()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.save_torch_state_dict)以更灵活地保存状态字典。

有关张量共享的更多信息，请查看[this guide](https://huggingface.co/docs/safetensors/torch_shared_tensors)。

模型状态字典被分成多个分片，以便每个分片都小于给定的大小。碎片是
与给定的 `filename_pattern` 一起保存在 `save_directory` 中。如果模型太大而无法放入单个分片中，
`save_directory`中保存有一个索引文件，以指示每个张量的保存位置。这个助手使用
[split_torch_state_dict_into_shards()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.split_torch_state_dict_into_shards) 在引擎盖下。如果`safe_serialization`是`True`，则分片保存为
安全张量（默认）。否则，碎片将保存为 pickle。在保存模型之前，会清除 `save_directory` 之前的任何分片文件。

> [!警告]
> 如果其中一个模型的张量大于 `max_shard_size`，它将最终出现在自己的分片中，该分片将具有
> 尺寸大于`max_shard_size`。

> [!警告]
> 如果您的模型是 `transformers.PreTrainedModel`，则应将 `model._tied_weights_keys` 作为 `shared_tensors_to_discard` 传递，以正确处理共享张量保存。这确保了在保存过程中丢弃正确的重复张量。

示例：

```py
>>> from huggingface_hub import save_torch_model
>>> model = ... # A PyTorch model

# Save state dict to "path/to/folder". The model will be split into shards of 5GB each and saved as safetensors.
>>> save_torch_model(model, "path/to/folder")

# Load model back
>>> from huggingface_hub import load_torch_model  # TODO
>>> load_torch_model(model, "path/to/folder")
>>>
```

### save_torch_state_dict[[huggingface_hub.save_torch_state_dict]]

#### Huggingface_hub.save_torch_state_dict[[huggingface_hub.save_torch_state_dict]]

```python
huggingface_hub.save_torch_state_dict(state_dict: dict, save_directory: str | pathlib.Path, filename_pattern: str | None = None, force_contiguous: bool = True, max_shard_size: int | str = '5GB', metadata: dict[str, str] | None = None, safe_serialization: bool = True, is_main_process: bool = True, shared_tensors_to_discard: list[str] | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L137)

**参数：**

state_dict (`dict[str, torch.Tensor]`) ：要保存的状态字典。

save_directory（`str`或`Path`）：保存模型的目录。

filename_pattern (`str`, *可选*) ：生成保存模型的文件名的模式。 Pattern 必须是可以使用 `filename_pattern.format(suffix=...)` 格式化的字符串，并且必须包含关键字 `suffix` 默认为 `"model{suffix}.safetensors"` 或 `pytorch_model{suffix}.bin`，具体取决于 `safe_serialization` 参数。force_contigious (`boolean`, *可选*) ：强制将 state_dict 保存为连续张量。这对模型的正确性没有影响，但如果专门为此选择张量的布局，则可能会改变性能。默认为`True`。

max_shard_size（`int`或`str`，*可选*）：每个分片的最大大小，以字节为单位。默认为 5GB。

元数据（`dict[str, str]`，*可选*）：与模型一起保存的额外信息。将为每个删除的张量添加一些元数据。这些信息不足以恢复整个共享​​结构，但可能有助于理解事物。

safe_serialization (`bool`, *可选*) ：是否保存为安全张量，这是默认行为。如果`False`，则碎片将保存为pickle。出于安全原因，建议使用安全序列化。另存为 pickle 已被弃用，并将在未来版本中删除。

is_main_process (`bool`, *可选*) ：调用此进程的进程是否为主进程。在像 TPU 这样的分布式训练中并且需要从所有进程调用此函数时非常有用。在这种情况下，仅在主进程上设置 `is_main_process=True` 以避免竞争条件。默认为 True。shared_tensors_to_discard (`list[str]`, *可选*) ：保存共享张量时要删除的张量名称列表。如果未提供并且检测到共享张量，它将按字母顺序删除名字。

将模型状态字典保存到磁盘，处理分片和共享张量问题。

另请参阅[save_torch_model()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.save_torch_model)直接保存 PyTorch 模型。

有关张量共享的更多信息，请查看[this guide](https://huggingface.co/docs/safetensors/torch_shared_tensors)。

模型状态字典被分成多个分片，以便每个分片都小于给定的大小。碎片是
与给定的 `filename_pattern` 一起保存在 `save_directory` 中。如果模型太大而无法放入单个分片中，
`save_directory`中保存有一个索引文件，以指示每个张量的保存位置。这个助手使用
[split_torch_state_dict_into_shards()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.split_torch_state_dict_into_shards) 在引擎盖下。如果`safe_serialization`是`True`，则分片保存为
安全张量（默认）。否则，碎片将保存为 pickle。

在保存模型之前，会清除 `save_directory` 之前的任何分片文件。

> [!警告]
> 如果其中一个模型的张量大于 `max_shard_size`，它将最终出现在自己的分片中，该分片将具有
> 尺寸大于`max_shard_size`。> [!警告]
> 如果您的模型是 `transformers.PreTrainedModel`，则应将 `model._tied_weights_keys` 作为 `shared_tensors_to_discard` 传递，以正确处理共享张量保存。这确保了在保存过程中丢弃正确的重复张量。

示例：

```py
>>> from huggingface_hub import save_torch_state_dict
>>> model = ... # A PyTorch model

# Save state dict to "path/to/folder". The model will be split into shards of 5GB each and saved as safetensors.
>>> state_dict = model_to_save.state_dict()
>>> save_torch_state_dict(state_dict, "path/to/folder")
```

`serialization` 模块还包含低级帮助程序，用于将状态字典拆分为多个分片，同时在此过程中创建适当的索引。这些助手可用于 `torch` 张量，并且旨在轻松扩展到任何其他 ML 框架。

### split_torch_state_dict_into_shards[[huggingface_hub.split_torch_state_dict_into_shards]]

#### Huggingface_hub.split_torch_state_dict_into_shards[[huggingface_hub.split_torch_state_dict_into_shards]]

```python
huggingface_hub.split_torch_state_dict_into_shards(state_dict: dict, filename_pattern: str = 'model{suffix}.safetensors', max_shard_size: int | str = '5GB')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L294)

**参数：**

state_dict (`dict[str, torch.Tensor]`) ：要保存的状态字典。

filename_pattern (`str`, *可选*) ：生成保存模型的文件名的模式。 Pattern 必须是可以使用 `filename_pattern.format(suffix=...)` 格式化的字符串，并且必须包含关键字 `suffix` 默认为 `"model{suffix}.safetensors"`。

max_shard_size（`int`或`str`，*可选*）：每个分片的最大大小，以字节为单位。默认为 5GB。

**返回：** `StateDictSplit`

一个 `StateDictSplit` 对象，包含分片和检索它们的索引。将模型状态字典拆分为分片，使每个分片小于给定大小。

分片是通过按照键的顺序迭代 `state_dict` 来确定的。没有优化
使每个分片尽可能接近通过的最大大小。例如，如果限制是 10GB 并且我们
有大小为 [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] 的张量，它们将被分片为 [6GB], [6+2GB], [6+2+2GB] 而不是
[6+2+2GB]、[6+2GB]、[6GB]。

> [!提示]
> 要将模型状态字典保存到磁盘，请参阅[save_torch_state_dict()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.save_torch_state_dict)。这个助手使用
> `split_torch_state_dict_into_shards` 在引擎盖下。

> [!警告]
> 如果其中一个模型的张量大于 `max_shard_size`，它将最终出现在自己的分片中，该分片将具有
> 尺寸大于`max_shard_size`。

示例：
```py
>>> import json
>>> import os
>>> from safetensors.torch import save_file as safe_save_file
>>> from huggingface_hub import split_torch_state_dict_into_shards

>>> def save_state_dict(state_dict: dict[str, torch.Tensor], save_directory: str):
...     state_dict_split = split_torch_state_dict_into_shards(state_dict)
...     for filename, tensors in state_dict_split.filename_to_tensors.items():
...         shard = {tensor: state_dict[tensor] for tensor in tensors}
...         safe_save_file(
...             shard,
...             os.path.join(save_directory, filename),
...             metadata={"format": "pt"},
...         )
...     if state_dict_split.is_sharded:
...         index = {
...             "metadata": state_dict_split.metadata,
...             "weight_map": state_dict_split.tensor_to_filename,
...         }
...         with open(os.path.join(save_directory, "model.safetensors.index.json"), "w") as f:
...             f.write(json.dumps(index, indent=2))
```

### split_state_dict_into_shards_factory[[huggingface_hub.split_state_dict_into_shards_factory]]

这是派生每个特定于框架的帮助器的底层工厂。在实践中，除非您需要将其调整到尚不支持的框架，否则您不需要直接使用该工厂。如果是这种情况，请通过 `huggingface_hub` 存储库上的 [opening a new issue](https://github.com/huggingface/huggingface_hub/issues/new) 告知我们。#### Huggingface_hub.split_state_dict_into_shards_factory[[huggingface_hub.split_state_dict_into_shards_factory]]

```python
huggingface_hub.split_state_dict_into_shards_factory(state_dict: dict, get_storage_size: Callable, filename_pattern: str, get_storage_id: Callable = <function <lambda> at 0x7f9108f630a0>, max_shard_size: int | str = '5GB')
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_base.py#L50)

**参数：**

state_dict (`dict[str, Tensor]`) ：要保存的状态字典。

get_storage_size (`Callable[[Tensor], int]`) ：一个函数，返回保存在磁盘上的张量的大小（以字节为单位）。

get_storage_id (`Callable[[Tensor], Optional[Any]]`, *可选*) ：向张量存储返回唯一标识符的函数。多个不同的张量可以共享相同的底层存储。在该张量的生命周期内，该标识符保证是唯一且恒定的。具有不重叠生命周期的两个张量存储可以具有相同的id。

filename_pattern (`str`, *可选*) ：生成保存模型的文件名的模式。 Pattern 必须是可以使用`filename_pattern.format(suffix=...)`格式化的字符串，并且必须包含关键字`suffix`

max_shard_size（`int`或`str`，*可选*）：每个分片的最大大小，以字节为单位。默认为 5GB。

**返回：** `StateDictSplit`

一个 `StateDictSplit` 对象，包含分片和检索它们的索引。

将模型状态字典拆分为分片，使每个分片小于给定大小。分片是通过按照键的顺序迭代 `state_dict` 来确定的。没有优化
使每个分片尽可能接近通过的最大大小。例如，如果限制是 10GB 并且我们
有大小为 [6GB, 6GB, 2GB, 6GB, 2GB, 2GB] 的张量，它们将被分片为 [6GB], [6+2GB], [6+2+2GB] 而不是
[6+2+2GB]、[6+2GB]、[6GB]。

> [!警告]
> 如果其中一个模型的张量大于 `max_shard_size`，它将最终出现在自己的分片中，该分片将具有
> 尺寸大于`max_shard_size`。

## 加载张量

加载助手支持 safetensors 或 pickle 格式的单文件和分片检查点。 [load_torch_model()](/docs/huggingface_hub/v1.30.0/en/package_reference/serialization#huggingface_hub.load_torch_model) 采用 `nn.Module` 和检查点路径（单个文件或目录）作为输入，并将权重加载到模型中。

### load_torch_model[[huggingface_hub.load_torch_model]]

#### Huggingface_hub.load_torch_model[[huggingface_hub.load_torch_model]]

```python
huggingface_hub.load_torch_model(model: torch.nn.Module, checkpoint_path: str | os.PathLike, strict: bool = False, safe: bool = True, weights_only: bool = False, map_location: typing.Union[str, ForwardRef('torch.device'), NoneType] = None, mmap: bool = False, filename_pattern: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L367)

**参数：**

model (`torch.nn.Module`) ：加载检查点的模型。

checkpoint_path（`str`或`os.PathLike`）：检查点文件或包含检查点的目录的路径。strict (`bool`，*可选*，默认为`False`)：是否严格强制模型状态字典中的键与检查点中的键匹配。

safe（`bool`，*可选*，默认为`True`）：如果`safe`为True，则将加载safetensors文件。如果 `safe` 为 False，该函数将首先尝试加载安全张量文件（如果可用），否则它将回退到加载 pickle 文件。 `filename_pattern` 参数优先于`safe` 参数。

weights_only（`bool`，*可选*，默认为`False`）：如果为 True，则仅加载模型权重，而不加载优化器状态和其他元数据。仅在 PyTorch >= 1.13 中受支持。

map_location（`str`或`torch.device`，*可选*）：指定如何重新映射存储位置的`torch.device`对象、字符串或字典。它指示应加载所有张量的位置。

mmap (`bool`, *可选*, 默认为`False`) : 是否使用内存映射文件加载。内存映射可以通过基于 zip 文件的检查点提高 PyTorch >= 2.1.0 中大型模型的加载性能。filename_pattern (`str`, *可选*) ：查找索引文件的模式。 Pattern 必须是可以使用 `filename_pattern.format(suffix=...)` 格式化的字符串，并且必须包含关键字 `suffix` 默认为 `"model{suffix}.safetensors"`。

**返回：** `NamedTuple`

具有 `missing_keys` 和 `unexpected_keys` 字段的命名元组。
- `missing_keys` 是包含缺失键的 str 列表，即模型中但不在检查点中的键。
- `unexpected_keys` 是包含意外键的 str 列表，即在检查点中但不在模型中的键。

**加薪：** ``FileNotFoundError`` or ``ImportError`` or ``ValueError``

- [⟦T174⟧](https://docs.python.org/3/library/exceptions.html#FileNotFoundError) -- 
  如果检查点文件或目录不存在。
- [⟦T175⟧](https://docs.python.org/3/library/exceptions.html#ImportError) -- 
  如果在尝试分别加载 .safetensors 文件或 PyTorch 检查点时未安装 safetensors 或 torch。
- [⟦T176⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果检查点路径无效或无法确定检查点格式。

将检查点加载到模型中，处理分片和非分片检查点。

示例：
```python
>>> from huggingface_hub import load_torch_model
>>> model = ... # A PyTorch model
>>> load_torch_model(model, "path/to/checkpoint")
```

### load_state_dict_from_file[[huggingface_hub.load_state_dict_from_file]]

#### Huggingface_hub.load_state_dict_from_file[[huggingface_hub.load_state_dict_from_file]]

```python
huggingface_hub.load_state_dict_from_file(checkpoint_file: str | os.PathLike, map_location: typing.Union[str, ForwardRef('torch.device'), NoneType] = None, weights_only: bool = False, mmap: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L573)

**参数：**checkpoint_file (`str` 或 `os.PathLike`) ：要加载的检查点文件的路径。可以是安全张量或泡菜（`.bin`）检查点。

map_location（`str`或`torch.device`，*可选*）：指定如何重新映射存储位置的`torch.device`对象、字符串或字典。它指示应加载所有张量的位置。

weights_only（`bool`，*可选*，默认为`False`）：如果为 True，则仅加载模型权重，而不加载优化器状态和其他元数据。仅支持 PyTorch >= 1.13 的 pickle (`.bin`) 检查点。加载 safetensors 文件时无效。

mmap (`bool`, *可选*, 默认为`False`) : 是否使用内存映射文件加载。内存映射可以通过基于 zip 文件的检查点提高 PyTorch >= 2.1.0 中大型模型的加载性能。加载 safetensors 文件时无效，因为 `safetensors` 库默认使用内存映射。

**退货：** `Union[dict[str, "torch.Tensor"], Any]`

已加载的检查点。
- 对于 safetensors 文件：始终返回将参数名称映射到张量的字典。
- 对于pickle文件：返回任何被pickle的Python对象（通常是一个状态字典，但也可以是
  整个模型、优化器状态或任何其他 Python 对象）。**加薪：** ``FileNotFoundError`` or ``ImportError`` or ``OSError`` or ``ValueError``

- [⟦T197⟧](https://docs.python.org/3/library/exceptions.html#FileNotFoundError) -- 
  如果检查点文件不存在。
- [⟦T198⟧](https://docs.python.org/3/library/exceptions.html#ImportError) -- 
  如果在尝试分别加载 .safetensors 文件或 PyTorch 检查点时未安装 safetensors 或 torch。
- [⟦T199⟧](https://docs.python.org/3/library/exceptions.html#OSError) -- 
  如果检查点文件格式无效或者 git-lfs 文件未正确下载。
- [⟦T200⟧](https://docs.python.org/3/library/exceptions.html#ValueError) -- 
  如果检查点文件路径为空或无效。

加载检查点文件，处理安全张量和 pickle 检查点格式。

示例：
```python
>>> from huggingface_hub import load_state_dict_from_file

# Load a PyTorch checkpoint
>>> state_dict = load_state_dict_from_file("path/to/model.bin", map_location="cpu")
>>> model.load_state_dict(state_dict)

# Load a safetensors checkpoint
>>> state_dict = load_state_dict_from_file("path/to/model.safetensors")
>>> model.load_state_dict(state_dict)
```

## 张量助手

### get_torch_storage_id[[huggingface_hub.get_torch_storage_id]]

#### Huggingface_hub.get_torch_storage_id[[huggingface_hub.get_torch_storage_id]]

```python
huggingface_hub.get_torch_storage_id(tensor: torch.Tensor)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L764)

将唯一标识符返回到张量存储。

多个不同的张量可以共享相同的底层存储。这个标识符是
保证该张量在其生命周期内的存储是唯一且恒定的。两个张量存储
非重叠的生命周期可能具有相同的 id。
对于元张量，我们返回 None，因为我们无法判断它们是否共享相同的存储。摘自https://github.com/huggingface/transformers/blob/1ecf5f7c982d761b4daaa96719d162c324187c64/src/transformers/pytorch_utils.py#L278。

### get_torch_storage_size[[huggingface_hub.get_torch_storage_size]]

#### Huggingface_hub.get_torch_storage_size[[huggingface_hub.get_torch_storage_size]]

```python
huggingface_hub.get_torch_storage_size(tensor: torch.Tensor)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.30.0/src/huggingface_hub/serialization/_torch.py#L781)

摘自https://github.com/huggingface/safetensors/blob/08db34094e9e59e2f9218f2df133b7b4aaff5a99/bindings/python/py_src/safetensors/torch.py#L31C1-L41C59

### Mixins 和序列化方法
https://huggingface.co/docs/huggingface_hub/v1.30.0/package_reference/mixins.md