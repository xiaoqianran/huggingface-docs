<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 将大模型加载到内存中

在 PyTorch 中加载预训练模型时，通常的工作流程如下所示：

```py
import torch

my_model = ModelClass(...)
state_dict = torch.load(checkpoint_file)
my_model.load_state_dict(state_dict)
```

简而言之，这些步骤是：
1. 使用随机初始化的权重创建模型
2. 从磁盘加载模型权重（在通常称为状态字典的字典中）
3. 将这些权重加载到模型中

虽然这对于常规大小的模型非常有效，但当我们处理大型模型时，此工作流程有一些明显的局限性：在步骤 1 中，我们将模型的完整版本加载到 RAM 中，并花费一些时间随机初始化权重（将在步骤 3 中丢弃）。在步骤 2 中，我们将模型的另一个完整版本加载到 RAM 中，并带有预先训练的权重。如果您要加载具有 60 亿个参数的模型，这意味着模型的每个副本都需要 24GB RAM，因此总共需要 48GB（其中一半用于在 FP16 中加载模型）。

这个 API 相当新，仍处于实验阶段。虽然我们努力提供稳定的 API，但公共 API 的一些小部分将来可能会发生变化。

## 该流程如何运作：快速概述

## 流程如何运作：使用代码

### 实例化一个空模型Accelerate 引入的第一个帮助处理大型模型的工具是上下文管理器 [init_empty_weights()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.init_empty_weights)，它可以帮助您在不使用任何 RAM 的情况下初始化模型，以便可以在任何大小的模型上完成步骤 1。它的工作原理如下：

```py
from accelerate import init_empty_weights

with init_empty_weights():
    my_model = ModelClass(...)
```

例如：

```py
with init_empty_weights():
    model = nn.Sequential(*[nn.Linear(10000, 10000) for _ in range(1000)])
```

使用略多于 100B 的参数初始化一个空模型。在幕后，这依赖于 PyTorch 1.9 中引入的元设备。在上下文管理器下的初始化过程中，每次创建参数时，它都会立即移动到该设备。

    您无法直接在 CPU 或其他设备上移动这样初始化的模型，因为它没有任何数据。使用该空模型的前向传递也很可能会失败，因为元设备上并非支持所有操作。

### 分片检查点

您的模型可能太大，甚至单个副本都无法放入 RAM。这并不意味着它无法加载：如果您有一个或多个 GPU，则可以使用更多内存来存储模型。在这种情况下，最好将您的检查点分成几个较小的文件，我们称之为检查点分片。只要您遵循以下格式，Accelerate 就会处理分片检查点：您的检查点应该位于一个文件夹中，其中有多个包含部分状态字典的文件，并且应该有一个 JSON 格式的索引，其中包含将参数名称映射到包含其权重的文件的字典。您可以使用 [save_model()](/docs/accelerate/v1.14.0/en/package_reference/accelerator#accelerate.Accelerator.save_model) 轻松地对模型进行分片。例如，我们可以有一个包含以下内容的文件夹：

```bash
first_state_dict.bin
index.json
second_state_dict.bin
```

其中index.json是以下文件：

```
{
  "linear1.weight": "first_state_dict.bin",
  "linear1.bias": "first_state_dict.bin",
  "linear2.weight": "second_state_dict.bin",
  "linear2.bias": "second_state_dict.bin"
}
```

`first_state_dict.bin`包含`"linear1.weight"`和`"linear1.bias"`的权重，`second_state_dict.bin`包含`"linear2.weight"`和`"linear2.bias"`的权重

### 加载重量

Accelerate 引入的第二个工具是函数 [load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)，它允许您在空模型中加载检查点。这支持完整的检查点（包含整个状态字典的单个文件）以及分片检查点。它还会自动在您可用的设备（GPU、CPU RAM）上分配这些权重，因此，如果您正在加载分片检查点，则最大 RAM 使用量将是最大分片的大小。

如果您想对 Transformers 模型使用大模型推理，请查看此[documentation](https://huggingface.co/docs/transformers/main/en/main_classes/model#large-model-loading)。

以下是我们如何使用它来加载 [GPT2-1.5B](https://huggingface.co/marcsun13/gpt2-xl-linear-sharded) 模型。让我们下载该模型的分片版本。

```bash
pip install huggingface_hub
```

```py
from huggingface_hub import snapshot_download
checkpoint = "marcsun13/gpt2-xl-linear-sharded"
weights_location = snapshot_download(repo_id=checkpoint)
```

为了初始化模型，我们将使用库 minGPT。 

```bash
git clone https://github.com/karpathy/minGPT.git
pip install minGPT/
```

```py
from accelerate import init_empty_weights
from mingpt.model import GPT

model_config = GPT.get_default_config()
model_config.model_type = 'gpt2-xl'
model_config.vocab_size = 50257
model_config.block_size = 1024

with init_empty_weights():
    model = GPT(model_config)
```

然后，加载我们刚刚下载的检查点：

```py
from accelerate import load_checkpoint_and_dispatch

model = load_checkpoint_and_dispatch(
    model, checkpoint=weights_location, device_map="auto", no_split_module_classes=['Block']
)
```

通过传递 `device_map="auto"`，我们告诉 Accelerate 根据可用资源自动确定模型每一层的放置位置：
- 首先，我们使用 GPU 上的最大可用空间
- 如果我们仍然需要空间，我们将剩余的权重存储在CPU上
- 如果没有足够的 RAM，我们将剩余的权重作为内存映射张量存储在硬盘上

#### `no_split_module_classes`

此参数将指示某些名称为 `"Block"` 的模块不应拆分到不同的设备上。您应该在这里设置所有块 
包括某种剩余连接。

#### `device_map`

您可以通过访问模型的 `hf_device_map` 属性来查看 Accelerate 选择的 `device_map`：

```py
model.hf_device_map
```

```python out
{'transformer.wte': 0,
 'transformer.wpe': 0,
 'transformer.drop': 0,
 'transformer.h.0': 0,
 ...
 'transformer.h.21': 0, 
 'transformer.h.22': 1, 
 'transformer.h.23': 1, 
 'transformer.h.24': 1,
 ...
 'transformer.h.47': 1, 
 'transformer.ln_f': 1, 
 'lm_head': 1}
 ```

完全可以为要使用的层创建自己的设备映射，指定要使用的 GPU 设备（数字）、`"cpu"` 或 `"disk"` 并将其传入：

```python
device_map = {
    "transformer.wte": "cpu",
    "transformer.wpe": 0,
    "transformer.drop": "cpu",
    "transformer.h.0": "disk"
}

model = load_checkpoint_and_dispatch(
    model, checkpoint=weights_location, device_map=device_map
)

```

### 运行模型现在我们已经做到了这一点，我们的模型位于多个设备上，也许还有硬盘驱动器上。但它仍然可以用作常规 PyTorch 模型：

```py
from mingpt.bpe import BPETokenizer
tokenizer = BPETokenizer()
inputs = tokenizer("Hello, my name is").to(0)

outputs = model.generate(x1, max_new_tokens=10, do_sample=False)[0]
tokenizer.decode(outputs.cpu().squeeze())
```

Accelerate 在幕后为模型添加了钩子，以便：
- 在每一层，输入都放在正确的设备上（因此即使您的模型分布在多个 GPU 上，它也能工作）
- 对于 CPU 上卸载的权重，它们会在前向传递之前放在 GPU 上，并在前向传递之后清理
- 对于硬盘驱动器上卸载的权重，它们会加载到 RAM 中，然后在前向传递之前放入 GPU 上，并在之后清理

这样，即使您的模型不适合其中一个 GPU 或 CPU RAM，也可以运行推理！

    这仅支持模型的推理，而不支持训练。大多数计算发生在 `torch.no_grad()` 上下文管理器后面，以避免中间激活花费一些 GPU 内存。

### 设计设备映射

您可以通过将 `device_map` 设置为受支持的选项之一（`"auto"`、`"balanced"`、`"balanced_low_0"`、`"sequential"`）来让 Accelerate 处理设备映射计算，或者如果您希望更好地控制每一层的位置，请自行创建一个选项。您可以在元设备上的模型上导出模型的所有大小（从而计算 `device_map`）。

当您没有足够的 GPU 内存来容纳整个模型时（即适应 GPU 上的所有内容，然后在 CPU 上卸载权重，如果没有足够的 RAM，甚至在磁盘上卸载权重），所有选项都会产生相同的结果。 

当可用 GPU 内存多于模型大小时，每个选项之间的差异如下：
- `"auto"` 和 `"balanced"` 在所有可用 GPU 上均匀分割模型，使您可以使用大于 1 的批量大小。
- `"balanced_low_0"` 在除第一个 GPU 之外的所有 GPU 上均匀分割模型，并且仅将不适合其他 GPU 的内容放在 GPU 0 上。当您需要使用 GPU 0 进行某些输出处理时，例如使用 Transformers 模型的 `generate` 函数时，此选项非常有用
- `"sequential"` 将适应 GPU 0 上的情况，然后转移到 GPU 1 上，依此类推（因此如果不需要，不会使用最后的 GPU）。

    目前，选项 `"auto"` 和 `"balanced"` 会产生相同的结果，但如果我们找到更有意义的策略，`"auto"` 的行为将来可能会发生变化，而 `"balanced"` 将保持稳定。首先请注意，您可以使用 `max_memory` 参数（在 [infer_auto_device_map()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.infer_auto_device_map) 以及使用它的所有函数中可用）来限制每个 GPU 上使用的内存。设置 `max_memory` 时，您应该传递一个包含 GPU 标识符（例如 `0`、`1` 等）的字典以及用于 CPU 卸载的最大 RAM 的 `"cpu"` 键。这些值可以是整数（以字节为单位），也可以是表示数字及其单位的字符串，例如 `"10GiB"` 或 `"10GB"`。

下面是一个示例，我们不想在两个 GPU 上分别使用超过 10GiB 的内存，并且不希望为模型权重使用超过 30GiB 的 CPU RAM：

```python
from accelerate import infer_auto_device_map

device_map = infer_auto_device_map(my_model, max_memory={0: "10GiB", 1: "10GiB", "cpu": "30GiB"})
```

    当 PyTorch 中发生第一次分配时，它会加载 CUDA 内核，该内核大约需要 1-2GB 内存，具体取决于 GPU。因此，可用内存总是小于 GPU 的实际大小。要查看实际使用了多少内存，请执行 `torch.ones(1).cuda()` 并查看内存使用情况。

    因此，当您使用`max_memory`创建内存映射时，请确保相应地调整可用内存以避免内存不足错误。此外，如果您对输出执行一些额外的操作而不将它们放回 CPU（例如在 Transformers 的`generate`方法内），并且如果您将输入放置在 GPU 上，则该 GPU 将比其他 GPU 消耗更多的内存（Accelerate 始终将输出放回输入设备）。因此，如果您想优化最大批量大小并且您有许多 GPU，请为第一个 GPU 提供较少的内存。例如，在 8x80 A100 设置上使用 BLOOM-176B，接近理想的贴图为：

```python
max_memory = {0: "30GIB", 1: "46GIB", 2: "46GIB", 3: "46GIB", 4: "46GIB", 5: "46GIB", 6: "46GIB", 7: "46GIB"}
```
如您所见，我们为其余 7 个 GPU 提供了比 GPU 0 多约 50% 的内存。

如果您选择自己完全设计`device_map`，它应该是一个字典，键是模型的模块名称，值是有效的设备标识符（例如GPU的整数）或`"cpu"`用于CPU卸载，`"disk"`用于磁盘卸载。键需要覆盖整个模型，然后您可以根据需要定义设备映射：例如，如果您的模型有两个块（假设`block1`和`block2`），每个块包含三个线性层（假设`linear1`、`linear2`和`linear3`），则有效的设备映射可以是：

```python
device_map = {"block1": 0, "block2": 1}
```

另一种有效的可能是：

```python
device_map = {"block1": 0, "block2.linear1": 0, "block2.linear2": 1, "block2.linear3": 1}
```另一方面，这个是无效的，因为它没有涵盖模型的每个参数：

```python
device_map = {"block1": 0, "block2.linear1": 1, "block2.linear2": 1}
```

    为了达到最高效率，请确保您的设备映射按顺序将参数放在 GPU 上（例如，不要将第一个权重之一放在 GPU 0 上，然后将权重放在 GPU 1 上，最后一个权重放回 GPU 0），以避免在 GPU 之间进行多次数据传输。

## 仅 CPU 卸载

如果你想在 CPU 上卸载你的模型，你可以使用[cpu_offload()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.cpu_offload)。因此，模型的所有参数都将被卸载，并且仅保留模型状态字典的一份副本。在前向传递期间，将从该状态字典中提取参数并将其放在执行设备上并根据需要传递，然后再次卸载。 

```python
cpu_offload(model, execution_device)
```

您也可以使用[cpu_offload_with_hook()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.cpu_offload_with_hook)。该函数将卸载 CPU 上的模型，并在执行时将其放回执行设备。与[cpu_offload()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.cpu_offload)的区别在于，转发后模型保留在执行设备上，只有在调用返回的`hook`的`offload`方法时才会再次卸载。此外，[cpu_offload_with_hook()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.cpu_offload_with_hook)性能更高，但节省的内存更少。它对于循环运行模型的管道很有用：

```python
model_1, hook_1 = cpu_offload_with_hook(model_1, execution_device)
model_2, hook_2 = cpu_offload_with_hook(model_2, execution_device, prev_module_hook=hook_1)
model_3, hook_3 = cpu_offload_with_hook(model_3, execution_device, prev_module_hook=hook_2)

hid_1 = model_1(input)
for i in range(50):
    # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
    hid_2 = model_2(hid_1)
# model2 is offloaded to the CPU just before this forward.
hid_3 = model_3(hid_3)

# For model3, you need to manually call the hook offload method.
hook_3.offload()
```

## 仅磁盘卸载要执行磁盘卸载，您可以使用[disk_offload()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.disk_offload)。因此，模型的所有参数都将作为给定文件夹中的内存映射数组卸载。在正向传递期间，将从该文件夹访问参数，并将其放在需要时传递的执行设备上，然后再次卸载。

```python
disk_offload(model, offload_dir, execution_device)
```

## 限制和进一步发展

我们了解 API 当前的限制：

- [infer_auto_device_map()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.infer_auto_device_map)（或[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)中的`device_map="auto"`）尝试最大化执行时可用的GPU和CPU RAM。虽然 PyTorch 非常擅长有效管理 GPU RAM（并在不需要时将其归还），但对于 Python 和 CPU RAM 来说并不完全如此。因此，自动计算的设备映射可能会占用 CPU 太多的资源。如果由于 RAM 不足而导致崩溃，请将一些模块移至磁盘设备。
- [infer_auto_device_map()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.infer_auto_device_map)（或[load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)中的`device_map="auto"`）按顺序对设备进行属性设置（以避免来回移动事物），因此如果您的第一层大于您拥有的GPU的大小，那么它最终会将所有内容放在CPU/磁盘上。- [load_checkpoint_and_dispatch()](/docs/accelerate/v1.14.0/en/package_reference/big_modeling#accelerate.load_checkpoint_and_dispatch)和[load_checkpoint_in_model()](/docs/accelerate/v1.14.0/en/package_reference/utilities#accelerate.load_checkpoint_in_model)目前不会对状态字典与模型的正确性进行任何检查（这将在未来版本中修复），因此如果尝试加载键不匹配或丢失的检查点，可能会出现一些奇怪的错误。
- 当模型被分割到多个 GPU 上时使用的模型并行性是幼稚且未优化的，这意味着在给定时间只有一个 GPU 工作，而另一个则闲置。
- 当权重卸载到 CPU/硬盘驱动器上时，不会进行预取（但是，我们将在未来版本中处理此问题），这意味着权重会在需要时而不是之前放在 GPU 上。
- 如果您运行的硬件在磁盘和 CPU 之间没有快速通信（例如 NVMe），则硬盘驱动器卸载可能会非常慢。

### 低精度训练方法
https://huggingface.co/docs/accelerate/v1.14.0/concept_guides/low_ precision_training.md