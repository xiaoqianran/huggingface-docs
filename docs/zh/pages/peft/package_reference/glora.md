<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 格洛拉

广义低秩适应 ([GLoRA](https://huggingface.co/papers/2306.07967)) 是一种泛化 LoRA 和相关方法的 PEFT 方法。 GLoRA 将更新分解为可配置路径（A、B、C、D、E），其中每个路径可以根据路径使用低秩、向量、常量或禁用参数化。

每个路径支持四种参数化模式之一。他们权衡**参数数量**与**表现力**（更新的丰富程度）：

- `"lora"`：低阶分解（如标准 LoRA）。使用`r * (out + in)`参数并可以表达等级-`r`校正。最具表现力、最多参数。
- `"vector"`：单个向量（例如形状`(out, 1)`），在矩阵中广播。使用`O(out)`参数；仅按通道缩放或移位。
- `"constant"`：所有元素共享的单个标量。使用 1 个参数；可训练选项中表现力最低的。
- `"none"`：没有可训练参数的零点；完全禁用该路径。

并非每个路径都接受所有模式（例如，`config_D_E` 不支持`"lora"`）。在更多路径上选择`"lora"`可以增加容量和可训练参数； `"vector"`、`"constant"` 或 `"none"` 均减少。GLoRA 对于您想要试验结构化更新模式并将多种适应机制组合在单层中的研究和高级应用程序特别有用。

在较高层面上，GLoRA 通过以下方式修改冻结的线性层：

$$
W_{\mathrm{eff}} = W_0 + W_0 \odot A + B
$$

$$
b_{\mathrm{eff}} = b_0 + b_0 \odot D + E + W_0 C
$$

其中每个路径都是独立参数化的。

## GloraConfig[[peft.GloraConfig]]

#### peft.GloraConfig[[peft.GloraConfig]]

```python
peft.GloraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, target_modules: typing.Union[str, list[str], NoneType] = None, bias: str = 'none', modules_to_save: typing.Optional[list[str]] = None, config_A_B: typing.Literal['lora', 'vector', 'constant', 'none'] = 'lora', config_C: typing.Literal['lora', 'vector', 'none'] = 'lora', config_D_E: typing.Literal['vector', 'constant', 'none'] = 'constant', init_weights: bool = True)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/glora/config.py#L23)

**参数：**

r (`int`) ：当配置设置为`lora`时使用的低秩分解的等级。

target_modules (`Optional[Union[List[str], str]]`) ：要应用 Glora 的模块的名称。

config_A_B (`str`) ：A 和 B 矩阵的参数化（权重乘法和加法校正）。有效值：`lora`、`vector`、`constant`、`none`。

config_C (`str`) ：C 矩阵的参数化（权重与偏差耦合：b += W0 @ C）。有效值：`lora`、`vector`、`none`。

config_D_E (`str`)：D 和 E 标量的参数化（偏差乘法和加法校正）。不支持 `lora`，因为 D 和 E 是偏置大小的向量，而不是矩阵。有效值：`vector`、`constant`、`none`。init_weights (`bool`) ：如果为 True（默认），则将 GLoRA 初始化为无操作（零）。如果为 False，则使用 kaiming 初始化，因此适配器不是无操作的。

这是存储[GloraModel](/docs/peft/v0.20.0/en/package_reference/glora#peft.GloraModel)配置的配置类。

Glora 将冻结线性层 W0 修改为：`W_eff = W0 + W0 * A + B` 和 `b_eff = b0 + b0 * D + E + W0 @ C`。

每个矩阵（A、B、C、D、E）都可以独立参数化。配置值控制有多少个参数
使用以及它们可以表达什么形状：

- `lora`：低阶分解`Xd @ Xu`，形状为`(out, r)`和`(r, in)`。使用`r * (out + in)`参数
  并且可以表达任何rank-r校正。就像标准 LoRA 一样。
- `vector`：形状为`(out, 1)`的单列向量，在整个矩阵中广播。使用`out`参数；
  仅针对每个输出通道进行缩放或移位。
- `constant`：所有元素共享的单个标量。使用 1 个参数；最受限制。
- `none`：零，没有可训练的参数。有效禁用此路径。### 关键配置选项
- `r`：当路径配置为`"lora"`时使用的等级（默认：`8`）。
- `target_modules`：要适应的模块名称列表或正则表达式（例如，`["q_proj", "v_proj"]`）。
- `config_A_B`：A 和 B 的路径类型（“lora”、“向量”、“常量”、“无”）。
- `config_C`：C 的路径类型（“lora”、“向量”、“无”）。
- `config_D_E`：D 和 E 的路径类型（“常量”、“向量”、“无”）。
- `bias`：偏差处理（`"none"`、`"all"` 或 `"glora_only"`）。
- `init_weights`：如果`True`（默认），GLoRA 被初始化为无操作。如果`False`，则使用kaiming初始化。

注意事项：
- `config_D_E`不支持`"lora"`。
- 对于支持的模型类型，可以省略`target_modules`（使用 PEFT 默认映射）。

## GloraModel[[peft.GloraModel]]

#### peft.GloraModel[[peft.GloraModel]]

```python
peft.GloraModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/glora/model.py#L34)

从预训练的 Transformer 模型创建通用低阶适配器 (GLoRA) 模型。

- 包装基本模型并将 GLoRA 适配器注入指定模块。
- 支持多个适配器、适配器切换、合并/取消合并以及混合批量推理。
- 使用`set_adapter`、`merge_and_unload`以及相关方法进行适配器管理。

## GloraLayer 和 GloraLinear[[peft.tuners.glora.GloraLayer]]

#### peft.tuners.glora.GloraLayer[[peft.tuners.glora.GloraLayer]]

```python
peft.tuners.glora.GloraLayer(base_layer: nn.Module, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/glora/layer.py#L103)#### peft.tuners.glora.GloraLinear[[peft.tuners.glora.GloraLinear]]

```python
peft.tuners.glora.GloraLinear(base_layer: nn.Module, adapter_name: str, config, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/glora/layer.py#L298)

GLoRA 适配器包裹着密集的 `~torch.nn.Linear` `base_layer`。

- `GloraLayer`是广义低秩适配的核心逻辑，支持多种适配器和灵活的路径配置。
- `GloraLinear` 是`nn.Linear` 的直接替代品，支持 GLoRA。
- GLoRA 目前支持普通的 `torch.nn.Linear` 基础层。

## 用法示例

```python
from transformers import AutoModelForCausalLM
from peft import GloraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained("your-model-id")
glora_config = GloraConfig(
    r=8,
    target_modules=["q_proj", "v_proj"],
    config_A_B="lora",
    config_C="vector",
    config_D_E="constant",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, glora_config)
model.print_trainable_parameters()

# Switch adapters, merge, etc.
model.set_adapter("default")
model.merge_and_unload()
```

## 注释
- GLoRA 是 LoRA 的超集：将所有路径设置为“lora”可恢复标准 LoRA。
- 您可以为A/B/C/D/E使用不同的路径类型来尝试新的适应策略。
- GLoRA 支持所有标准 PEFT 适配器管理功能（添加、删除、切换、合并等）。

## 另请参阅
- [Adapter conceptual guide](../conceptual_guides/adapter)
- [LoRA reference](./lora)
- [Paper: https://huggingface.co/papers/2306.07967](https://huggingface.co/papers/2306.07967)

### 骆驼适配器
https://huggingface.co/docs/peft/v0.20.0/package_reference/llama_adapter.md