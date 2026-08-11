<!-- huggingface-docs: machine-translated zh-CN from English source -->

# LoRA 转换

允许将非 LoRA PEFT 模型转换为 LoRA 模型的功能。

## 说明

PEFT 支持数十种不同的参数高效微调技术。迄今为止最受欢迎的一种是 LoRA。这意味着许多其他软件包也支持 LoRA。例如，[Diffusers](https://huggingface.co/docs/diffusers/main/en/api/loaders/lora)允许加载LoRA适配器来改变扩散模型的功能。 [vLLM](https://docs.vllm.ai/en/stable/features/lora/) 允许使用 LoRA 适配器为模型提供服务。这很好，但不幸的是，很少支持所有其他非 LoRA PEFT 方法。因此，即使另一种 PEFT 方法更适合您的特定用例，您也可能无法使用它，因为下游软件包不提供支持。

在这里，我们提出了一个潜在的解决方案。 PEFT 提供两个功能，[save_as_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.save_as_lora) 和 [convert_to_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.convert_to_lora)，允许将 PEFT 适配器转换为 LoRA 适配器。目前并非所有 PEFT 方法都支持此功能，但如果支持，则意味着您可以从最适合您的 PEFT 方法开始，然后像 LoRA 适配器一样使用它。

＃＃ 例子转换后的适配器的 LoRA 等级可以通过将 int > 0 传递给 `rank` 参数来设置为固定等级，也可以通过将 0 到 1 之间的浮点数传递给 `rank` 参数来设置为适应每一层的动态等级。动态排名可能会更有效（使用更少的参数获得相同的性能）。

### 固定 LoRA 排名

[save_as_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.save_as_lora)的用法相对简单：

```python
from peft import get_peft_model, save_as_lora

# first load and train your non-LoRA PEFT model as normal
base_model = ...
non_lora_config = ...
model = get_peft_model(base_model, non_lora_config)
# check that this PEFT method can indeed be converted to LoRA
assert model.supports_lora_conversion()
...  # train the model

# the rank of the LoRA adapter that you want to convert to
target_rank = 64
# save as a LoRA checkpoint
save_as_lora(output_path, model, rank=target_rank)
```

这将在 `output_path` 创建一个 LoRA 检查点，您可以像任何其他 LoRA 适配器一样加载该检查点，或者在 Diffusers 或 vLLM 等下游包中使用。

如果您不想保存转换后的 LoRA 适配器，而是想立即使用转换后的权重（例如执行评估），则 [convert_to_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.convert_to_lora) 函数非常有用：

```python
from peft import convert_to_lora, get_peft_model, set_peft_model_state_dict

base_model = ...
non_lora_config = ...
model = get_peft_model(base_model, non_lora_config)
...  # train the model

# get the lora config and state dict of the converted lora model
lora_config, lora_state_dict = convert_to_lora(model, rank=target_rank)
# reload the base model, or use model.unload()
base_model = ...
# apply the lora config to the base model
lora_model = get_peft_model(base_model, lora_config)
# load the LoRA weights onto the base model
set_peft_model_state_dict(lora_model, state_dict)
```

### 动态 LoRA 排名在上面的示例中，我们使用固定的 LoRA 等级进行转换。然而，可以想象，某些层不需要高等级来准确转换，而其他层则需要更高的等级。为了适应这一点，PEFT 提供了传递 0 到 1 之间的浮点数作为 `rank` 参数的选项。假设您通过了`rank=0.5`。这意味着对于每一层，选择 LoRA 适配器的等级，使得 LoRA 适配器解释原始适配器引入的权重差异的 50%。用更技术性的术语来说，我们在底层对适配器的权重贡献执行[Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)，然后取最高的奇异值，在标准化后，将其求和为传递的值。

```python
# set a dynamic rank by passing a float
threshold = 0.7
# save as a LoRA checkpoint
save_as_lora(output_path, model, rank=threshold)
# get the lora config and state dict directly:
lora_config, lora_state_dict = convert_to_lora(model, rank=threshold)
# inspect the different ranks per layer:
print(lora_config.rank_pattern)
```

如果不同层的贡献变化很大，那么使用这种类型的动态 LoRA 等级可能会很有用。缺点是这可能意味着某些层将具有非常高的 LoRA 等级，这可能会导致内存峰值。请测试最适合您的用例的方法。

### 编译模型对于大型模型，转换可能需要一些时间；例如，每个 PEFT 模块都必须经过 SVD 计算。通过将 `compile_kwargs` 传递给 [save_as_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.save_as_lora) 或 [convert_to_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.convert_to_lora)，您可以将 [⟦T13⟧](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) 应用于转换函数，并可能加快该过程。 `compile_kwargs` 是传递给 `torch.compile` 的关键字参数的字典（空字典也可以）。下面是一个例子：

```python
compile_kwargs = {"dynamic": True, "mode": "max-autotune-no-cudagraphs", "fullgraph": True}
save_as_lora(output_path, model, rank=rank, compile_kwargs=compile_kwargs)
```

### LoRA 到 LoRA 转换

还可以将 LoRA 适配器转换为另一个 LoRA 适配器。你为什么要这么做？原因有一个，那就是如果你想降低LoRA适配器的级别。如果在训练后，您想要缩小 LoRA 适配器，请使用 [save_as_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.save_as_lora) 或 [convert_to_lora()](/docs/peft/v0.20.0/en/package_reference/lora_conversion#peft.convert_to_lora) 并通过较小的等级。这将为您提供一个新的 LoRA 适配器，其内存和存储占用空间更小。

## 指标

### 非 LoRA 到 LoRA 转换当然，将一个 PEFT 适配器转换为另一个适配器是一个有损过程。新适配器的性能很可能不如初始适配器。因此，强烈建议**评估转换后的 LoRA 适配器**。这样，您可以确保转换后的适配器对于您的用例来说性能足够好。一般规则是，LoRA 适配器的等级越高，它就越接近您的初始适配器。这意味着转换后的 LoRA 适配器可能需要比原始适配器更多的参数才能实现类似的性能。

举个例子，以下是在 [PEFT MetaMathQA benchmark](https://github.com/huggingface/peft/tree/main/method_comparison/MetaMathQA) 上得出的一些数字。为此，使用[LoHa](https://huggingface.co/docs/peft/package_reference/loha)适配器在MetaMathQA上微调`meta-llama/Llama-3.2-3B`并在GSM8K上进行评估。最初的 LoKr 适配器排名为 32，产生 18,350,080 个可训练参数，测试准确率为 41.85%。评估需要 12.25 GB 内存。检查点被转换为具有不同值的 `rank` 的 LoRA。由此产生的结果是：|等级 |可训练参数|测试准确度(%) |精度变化|保留内存（最大，GB）|内存增加|
|------|---------------------:|------------------:|----------------:|----------------------------------------:|----------------:|
| 8 |              2293760 | 2293760             37.60 | 37.60           -4.25 | -4.25                     12.41 | 12.41            0.16 | 0.16
| 16 | 16              4587520 | 4587520             38.89 | 38.89           -2.96 | -2.96                     12.15 | 12.15           -0.10 | -0.10
| 32 | 32              9175040 | 9175040             40.11 |           -1.74 | -1.74                     12.41 | 12.41            0.16 | 0.16
| 64 | 64             18350080 |             39.20 | 39.20           -2.65 | -2.65                     12.18 | 12.18           -0.07 | -0.07
|      |                      |                   |                 |                           |                 |
| 0.4 | 0.4              2428928 | 2428928             37.60 | 37.60           -4.25 | -4.25                     12.41 | 12.41            0.16 | 0.16
| 0.5 | 0.5              4761600 | 4761600             40.18 |           -1.67 | -1.67                     12.41 | 12.41            0.16 | 0.16
| 0.6 | 0.6              8857600 |             39.42 | 39.42           -2.43 | -2.43                     12.41 | 12.41            0.16 | 0.16| 0.7 | 0.7             16230400 |             39.04 | 39.04           -2.81 | -2.81                     12.15 | 12.15           -0.10 | -0.10

正如您所看到的，如果等级足够高，我们可以获得接近原始 LoHa 适配器的测试精度。选择正确的排名是模型性能和模型效率之间的权衡。要重现此实验，请按照 https://github.com/huggingface/peft/tree/main/scripts/evaluate-lora-conversion.py 中的脚本进行操作。

请注意，可训练参数的数量无法一对一转换为内存使用情况。即使具有相同数量的可训练参数，一些 PEFT 方法需要更多内存，一些方法需要更少内存。因此，即使转换后，LoRA 适配器比原始适配器具有更多参数，它在服务时仍然可以提高内存效率。

### LoRA 到 LoRA 转换与上面的实验类似，我们还可以评估LoRA到LoRA的转换（即LoRA压缩）。在这里，我们从一个 64 级的 LoRA 适配器开始，在与上述 RS-LoRA 相同的设置上进行训练。初始适配器有 18,350,080 个可训练参数，测试准确率为 52.92%，需要 12.58 GB 内存用于评估。下表显示了将此适配器转换为较小级别的 LoRA 适配器的结果：

|等级 |可训练参数|测试准确度(%) |精度变化|保留内存（最大，GB）|内存增加|
|------|---------------------:|------------------:|----------------:|----------------------------------------:|----------------:|
| 8 |              2293760 | 2293760             43.37 | 43.37           -9.55 | -9.55                     12.38 | 12.38           -0.20 | -0.20
| 16 | 16              4587520 | 4587520             48.90 | 48.90           -4.02 | -4.02                     12.38 | 12.38           -0.20 | -0.20
| 32 | 32              9175040 | 9175040             51.48 |           -1.44 | -1.44                     12.49 | 12.49           -0.09 | -0.09
| 48 | 48             13762560 |             52.01 | 52.01           -0.91 | -0.91                     12.38 | 12.38           -0.20 | -0.20|      |                      |                   |                 |                           |                 |
| 0.5 | 0.5              2150400 | 2150400             44.12 |           -8.80 |                     12.37 | 12.37           -0.21 | -0.21
| 0.6 | 0.6              3082240 | 3082240             47.54 | 47.54           -5.38 | -5.38                     12.37 | 12.37           -0.21 | -0.21
| 0.7 | 0.7              4448256 | 4448256             50.49 | 50.49           -2.43 | -2.43                     12.37 | 12.37           -0.21 | -0.21
| 0.8 | 0.8              6510592 | 6510592             50.11 |           -2.81 | -2.81                     12.37 | 12.37           -0.21 | -0.21
| 0.9 | 0.9             10022912 |             51.55 | 51.55           -1.37 | -1.37                     12.38 | 12.38           -0.20 | -0.20
| 0.95 | 0.95             12976128 |             52.62 | 52.62           -0.30 | -0.30                     12.39 | 12.39           -0.19 | -0.19

因此，例如对于排名 0.95，我们可以将准确度差距缩小到 0.3 个百分点，同时将参数数量减少 30%。另请注意，这些压缩的 LoRA 可能比直接在较低级别上训练它们更好——例如对于排名 32，直接训练的测试准确率为 48.22%，而排名 64 的转换结果为 51.48%。

## 注意事项LoRA 转换存在一些限制。如上所述，预计性能会下降，并且转换后的 LoRA 的参数效率很可能低于原始适配器。此外，LoRA 转换具有以下局限性：

- 目前，只能转换应用于线性层的适配器。
- 目前并非所有 PEFT 方法都支持 LoRA 转换。

如果有很多扩展 LoRA 转换的需求，请通过创建[GitHub discussion](https://github.com/huggingface/peft/discussions)告知我们，我们将使其支持更多层类型和 PEFT 方法。

## API

### 将非LoRA模型转换为LoRA模型，返回`LoraConfig`和`state_dict`[[peft.convert_to_lora]]

#### peft.convert_to_lora[[peft.convert_to_lora]]

```python
peft.convert_to_lora(model: Module, rank: float, adapter_name: str = 'default', progressbar: bool = False, compile_kwargs = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/conversion.py#L146)

**参数：**

model ：要转换的模型。应该是具有支持转换的 PEFT 层的模型。等级（`int`或`float`）：返回的LoRA适配器所需的等级。等级越高，LoRA 适配器就越准确地反映原始适配器。然而，它也需要更多的内存、计算和磁盘空间。因此，选择一个代表您的用例最佳权衡的值并验证最终适配器。如果传递的是浮点数，则它被解释为解释的方差/能量阈值：我们选择最小的等级 k，使得前 k 个奇异值至少占总奇异值平方的那部分。如果少数单个可以捕获该层的适应，这实际上会导致分配较低的排名。较低的浮动意味着排名较低，反之亦然。请注意，动态排名可能会导致每层的排名非常不平等，这意味着某些层可能需要不成比例的大量内存来进行激活。选择固定（int）等级可以更好地实现可预测的内存需求。

adapter_name (`str`, *可选*) ：要转换的适配器的名称。一次只能转换一个适配器。默认为`"default"`。Progressbar (`bool`) ：是否显示指示转换进度的进度条（对于大模型可能需要几分钟）。

compile_kwargs (`dict`, *可选*) ：如果提供，编译函数以将各个模块转换为 LoRA，并将给定的 kwargs 传递给`torch.compile`。这可能会加快大型模型的转换速度。

**退货：**

lora_config (`LoraConfig`)
转换后的LoRA适配器对应的`LoraConfig`。
state_dict (`dict[str, torch.Tensor]`)
`state_dict` 包含 LoRA 权重。

将具有 PE​​FT 层的非 LoRA 模型转换为 LoRA 检查点。

仅某些允许等效转换的特定 PEFT 方法支持此功能。本质上，这来了
PEFT 方法通过使用增量权重更新基本权重来工作。另外，现在只有线性层
都支持。LoRA 适配器将尝试尽可能接近初始适配器。等级越高越好
的近似值。预计近似值永远无法达到原始的全部性能
LoRA 适配器的参数效率将低于原始适配器（即
对于类似的性能，将需要更多的参数）。该转换在许多情况下仍然有用：

- 在 PEFT 中，LoRA 支持比大多数其他方法更多的功能，例如混合适配器批次。因此转换后的
  适配器可以与这些功能一起使用。
- 一些下游软件包支持 LoRA 适配器，但不支持其他 PEFT 方法，例如扩散器。转换允许
  将非 LoRA 适配器与这些软件包一起使用。

LoRA 缩放因子已经融入 LoRA 权重中，因此缩放将始终为 1（即等级和
alpha 被选择为相同）。

注意：该函数尚不支持分片模型。

提高
类型错误：
如果提供的模型没有任何可以转换为 LoRA 的层，则会引发 `TypeError`。
值错误：
如果选择了无效的排名（太高或太低）。### 将非 LoRA 模型转换为 LoRA 模型，将适配器检查点和配置保存在给定路径[[peft.save_as_lora]]

#### peft.save_as_lora[[peft.save_as_lora]]

```python
peft.save_as_lora(path: str | os.PathLike, model: Module, rank: float, adapter_name: str = 'default', progressbar: bool = False, compile_kwargs = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/lora/conversion.py#L363)

**参数：**

model ：要转换的模型。应该是具有支持转换的 PEFT 层的模型。

等级（`int`或`float`）：返回的LoRA适配器所需的等级。等级越高，LoRA 适配器就越准确地反映原始适配器。然而，它也需要更多的内存、计算和磁盘空间。因此，选择一个代表您的用例最佳权衡的值并验证最终适配器。如果传递的是浮点数，则它被解释为解释的方差/能量阈值：我们选择最小的等级 k，使得前 k 个奇异值至少占总奇异值平方的那部分。如果少数单个可以捕获该层的适应，这实际上会导致分配较低的排名。较低的浮动意味着排名较低，反之亦然。请注意，动态排名可能会导致每层的排名非常不平等，这意味着某些层可能需要不成比例的大量内存来激活离子。选择固定（int）等级可以更好地实现可预测的内存需求。

adapter_name (`str`, *可选*) ：要转换的适配器的名称。一次只能转换一个适配器。默认为 `"default"`。

Progressbar (`bool`) ：是否显示指示转换进度的进度条（对于大模型可能需要几分钟）。

compile_kwargs (`dict`, *可选*) ：如果提供，编译函数以将各个模块转换为 LoRA，并将给定的 kwargs 传递给`torch.compile`。这可能会加快大型模型的转换速度。

将具有 PE​​FT 层的非 LoRA 模型转换为 LoRA，然后保存检查点文件和 PEFT 配置。

仅某些允许等效转换的特定 PEFT 方法支持此功能。本质上，这来了
PEFT 方法通过使用增量权重更新基本权重来工作。另外，现在只有线性层
都支持。LoRA 适配器将尝试尽可能接近初始适配器。等级越高越好
的近似值。预计近似值永远无法达到原始的全部性能
LoRA 适配器的参数效率将低于原始适配器（即
对于类似的性能，将需要更多的参数）。该转换在许多情况下仍然有用：

- 在 PEFT 中，LoRA 支持比大多数其他方法更多的功能，例如混合适配器批次。因此转换后的
  适配器可以与这些功能一起使用。
- 一些下游软件包支持 LoRA 适配器，但不支持其他 PEFT 方法，例如扩散器。转换允许
  将非 LoRA 适配器与这些软件包一起使用。

LoRA 缩放因子已经融入 LoRA 权重中，因此缩放将始终为 1（即等级和
alpha 被选择为相同）。

您可以像这样加载转换后的 LoRA 权重：

```py
>>> lora_path = ...
>>> save_as_lora(lora_path, model, rank=...)
>>> base_model = AutoModel.from_pretrained(...)
>>> lora_model = PeftModel.from_pretrained(base_model, lora_path)
```

注意：该函数尚不支持分片模型。提高
类型错误：
如果提供的模型没有任何可以转换为 LoRA 的层，则会引发 `TypeError`。
值错误：
如果选择了无效的排名（太高或太低）。

### 前缀调整
https://huggingface.co/docs/peft/v0.20.0/package_reference/prefix_tuning.md