<!-- huggingface-docs: machine-translated zh-CN from English source -->

# UniLoRA

[Uni-LoRA](https://huggingface.co/papers/2506.00799) 是一种 PEFT 方法，共享紧凑的可训练
跨低阶适配器权重的向量库。 UniLoRA 不是独立学习每个 LoRA 矩阵元素，而是
确定性地将条目投影到共享 `theta_d` 值中并学习适配器使用的共享参数
更新。

## 快速入门

```python
from peft import UniLoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")

config = UniLoraConfig(
    r=32,
    theta_d_length=256,
    proj_seed=42,
    target_modules=["q_proj", "v_proj"],
    unilora_dropout=0.0,
    init_weights=True,
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, config)
peft_model.print_trainable_parameters()
```

## 重要参数

`r` 控制低阶适配器尺寸。较大的值会增加适配器容量和内存使用量。

`theta_d_length` 控制共享 UniLoRA 矢量库的长度。这是共享的主要可训练存储
预计的适配器条目。

`proj_seed` 控制`theta_d` 中固定投影的确定性索引生成。重复使用相同的种子和
配置使生成的适配器索引可重现。

`target_modules` 选择哪些模块接收 UniLoRA 适配器。使用模块后缀，例如 `["q_proj", "v_proj"]`、a
正则表达式字符串，或 `"all-linear"`（当模型架构支持时）。

`unilora_dropout` 在训练期间在 UniLoRA 适配器层内应用 dropout。

`init_weights` 控制UniLoRA参数初始化。将其设置为`False`以保持随机`theta_d`
当您需要手动管理初始化时进行初始化。`save_indices` 控制 UniLoRA 检查点是否将生成的索引和尺度张量与
共享`theta_d`参数。保持此禁用状态会提供更小的检查点并从中重新生成索引
`proj_seed`；启用它使保存的适配器独立于未来索引生成的更改。

## 基准概述

<iframe
src="https://peft-internal-testing-peft-method-comparison-embed.hf.space/?highlight[type]=UNILORA"
frameborder="0"
width="850"
height="1000"
>

# API

## UniLoraConfig[[peft.UniLoraConfig]]

#### peft.UniLoraConfig[[peft.UniLoraConfig]]

```python
peft.UniLoraConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 4, proj_seed: int = 42, theta_d_length: int = 256, target_modules: typing.Union[str, list[str], NoneType] = None, unilora_dropout: float = 0.0, fan_in_fan_out: bool = False, bias: str = 'none', modules_to_save: typing.Optional[list[str]] = None, init_theta_d_bound: float = 0.02, init_weights: bool = True, save_indices: bool = False, layers_to_transform: typing.Union[list[int], int, NoneType] = None, layers_pattern: typing.Union[str, list[str], NoneType] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/unilora/config.py#L23)

**参数：**

r (`int`) ：低阶自适应的阶数。这控制了 UniLora 更新的表达能力。

proj_seed (`int`) ：用于生成固定索引分配的随机种子。这确保了运行之间的再现性。

theta_d_length (`int`) ：共享 UniLora 向量`theta_d`的长度。target_modules (`Union[list[str], str]`, *可选*) ：应用 UniLora 适配器的模块的名称或模式。 - 如果提供了字符串，则将其视为正则表达式。 - 如果提供了列表，模块将按确切名称或后缀进行匹配。 - 特殊值“全线性”将 UniLora 应用于除输出层之外的所有线性/Conv1D 层。如果未指定，则从模型架构中推断模块。如果架构不受支持，则会引发错误。

unilora_dropout (`float`) ：在 UniLora 层中应用的 Dropout 概率。

fan_in_fan_out (`bool`) ：替换层是否以（fan_in，fan_out）格式存储权重。对于使用 Conv1D 层的模型（例如 GPT-2），应将其设置为 True。

bias (`str`) ：指定哪些偏差项是可训练的： - 'none'：没有偏差参数更新 - 'all'：更新所有偏差参数 - 'unilora_only'：仅更新 UniLora 层内的偏差 注意：即使禁用适配器，启用偏差更新也会更改模型输出。module_to_save (`list[str]`, *可选*) ：应保持可训练并保存在最终检查点的附加模块（UniLora 层之外）。这通常用于特定于任务的头，例如分类器。

init_theta_d_bound (`float`) ：UniLora 向量库的初始化边界。向量从 [-init_theta_d_bound, init_theta_d_bound] 中均匀采样。避免用零初始化以防止梯度消失。建议使用较小的值（例如 0.02）以实现稳定的训练。

init_weights (`bool`) : 是否使用默认的UniLora初始化来初始化`theta_d`。如果设置为 `False`，`theta_d` 保持随机初始化。

save_indices (`bool`) : 是否将生成的 UniLora 索引和缩放缓冲区与 `theta_d` 一起保存。这会增加检查点大小，但使保存的适配器独立于索引生成例程的未来更改。

Layers_to_transform (`Union[list[int], int]`, *可选*) ：应用 UniLora 的转换器层的索引。如果指定，则仅修改这些层。该选项仅当`target_modules`是列表时有效。Layers_pattern（`Union[list[str], str]`，*可选*）：当模型不遵循标准图层命名约定时，与`layers_to_transform`一起使用的自定义图层名称模式。该选项仅当`target_modules`是列表时有效。

UniLora 适配器的配置类。

此类定义了在 PEFT 框架内初始化和应用 UniLora 层所需的所有超参数。
配置有意最小化，仅包含当前主动使用的参数
UniLora 实施。

参考：
Uni-LoRA：您只需要一个向量 https://arxiv.org/abs/2506.00799

## UniLoraModel[[peft.UniLoraModel]]

#### peft.UniLoraModel[[peft.UniLoraModel]]

```python
peft.UniLoraModel(model, config, adapter_name, low_cpu_mem_usage: bool = False, state_dict: dict[str, torch.Tensor] | None = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/unilora/model.py#L31)

围绕预训练模型创建 UniLora 适配器。

####generate_index[[peft.UniLoraModel.generate_index]]

```python
generate_index(lora_param_count: int, theta_d_length: int, proj_seed: int)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/unilora/model.py#L127)

将确定性 `theta_d` 索引分配给扁平化的 UniLora 参数空间。一个简单的 `np.random.choice(np.arange(theta_d_length), size=lora_param_count)` 对每个位置进行采样
独立，这可能会留下一些`theta_d`条目未用于较小的适配器。 UniLora 相反使用
平衡确定性分配：每个索引出现 `floor(D / d)` 或 `ceil(D / d)` 次，其中 `D`
是扁平化的 LoRA 参数计数，`d` 是 `theta_d_length`。这使每个索引的标准化保持稳定
同时仍然用 `proj_seed` 打乱作业。

### 块对角 LoRA，用于消除张量并行 LoRA 服务中的通信开销
https://huggingface.co/docs/peft/v0.20.0/package_reference/lora_variant_bdlora.md