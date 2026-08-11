<!-- huggingface-docs: machine-translated zh-CN from English source -->

# DEFT：文本到图像模型的分解高效微调

[DEFT](https://proceedings.neurips.cc/paper_files/paper/2025/hash/93a34a7138bdad95e874018d5f491cc6-Abstract-Conference.html)
（分解高效微调）是一种用于文本到图像模型的参数高效微调方法。它
将冻结权重矩阵 `W` 的更新分解为两个可训练组件：删除低秩的投影
来自 `W` 的子空间，以及将新内容注入该子空间的低阶更新。该配方旨在
平衡与目标分布对齐，从一些图像中学习新概念（个性化），并保留
预训练模型的指令跟踪能力和可编辑性。

具体来说，DEFT 结合了两个可训练的低秩组件：（1）到低秩补集的投影
由低秩矩阵跨越的子空间，以及（2）低秩更新。第一个低秩矩阵定义子空间，而
第二个可以在该子空间内实现灵活的参数自适应。

何时使用 DEFT：最适合使模型适应新数据或概念，同时**保留甚至改进模型
基本模型的指令跟踪能力**并保持**忘记其先前的能力到最低程度**。对于每个目标层，DEFT 学习投影方向`P`（形状`out_features x r`）和注入矩阵`R`（形状
`r x in_features`）。有效权重是残差投影

```
W' = (I - P_proj) @ W + Q_P @ R
```

投影仪`P_proj`是根据`decomposition_method`从`P`导出的：

- `"relu"`（默认）：`Q_P = P`、`P_proj = P @ relu(P).T` — 非正交投影。
- `"qr"`：`Q_P = qr(P)`、`P_proj = Q_P @ Q_P.T` — 正交投影。

`(I - P_proj) @ W` 项删除了预训练权重的子空间，而 `Q_P @ R` 则向其中注入新内容。
默认情况下 (`init_weights=True`) `R` 被初始化，以便更新是初始化时的精确标识
(`W' == W`)，因此训练从预训练的权重开始并学习注入。此次更新相当于
低阶加性 delta `Q_P @ (R - right.T @ W)`，计算时无需形成 `out x out`
投影矩阵，可以合并到基本权重中以进行无推理部署。设置 `para=True` 选择
[PaRa](https://proceedings.iclr.cc/paper_files/paper/2025/hash/f09e8dd9274cb7c2dd0dc65ffc6f427a-Abstract-Conference.html)
（参数等级降低）变体：仅删除更新`W' = (I - P_proj) @ W`，仅保留子空间删除
期限并滴注。仅训练投影`P`（无注入矩阵`R`），因此适配器不是一个
初始化时的身份。 PaRa 是为了个性化文本到图像的扩散模型而引入的，可在此处获取
作为 DEFT 的一个特例。

DEFT 目前针对 `torch.nn.Linear` 和 `Conv1D`（例如 gpt-2，通过 `fan_in_fan_out`）层实现。最初的实现和实验
论文（Dreambooth、Dreambench Plus、InsDet、VisualCloze、关于稳定扩散和统一模型）可在
[github.com/MAXNORM8650/DEFT](https://github.com/MAXNORM8650/DEFT)。

如果您在工作中使用 DEFT，请引用该论文：

```bibtex
@article{kumar2026deft,
  title={DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models},
  author={Kumar, Komal and Anwer, Rao and Shahbaz Khan, Fahad and Khan, Salman and Laptev, Ivan and Cholakkal, Hisham},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={102009--102035},
  year={2026}
}
```

如果您使用 PaRa 变体 (`para=True`)，还请引用：

```bibtex
@inproceedings{chen2025personalizing,
  title={Para: Personalizing text-to-image diffusion via parameter rank reduction},
  author={Chen, Shangyu and Pan, Zizheng and Cai, Jianfei and Phung, Dinh},
  booktitle={International Conference on Learning Representations},
  year={2025}
}
```

## DeftConfig[[peft.DeftConfig]]

#### peft.DeftConfig[[peft.DeftConfig]]

```python
peft.DeftConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, r: int = 8, target_modules: Optional[Union[list[str], str]] = None, exclude_modules: Optional[Union[list[str], str]] = None, decomposition_method: Literal['relu', 'qr'] = 'relu', init_scale: float = 1.0, alpha: Optional[int] = None, para: bool = False, fan_in_fan_out: bool = False, deft_dropout: float = 0.0, init_weights: bool = True, layers_to_transform: Optional[Union[list[int], int]] = None, layers_pattern: Optional[Union[list[str], str]] = None, bias: str = 'none', modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/deft/config.py#L25)

**参数：**

r (`int`) ：跨层 DEFT 投影/注入的等级。target_modules (`Optional[Union[List[str], str]]`) ：要应用适配器的模块的名称。如果指定，则仅替换具有指定名称的模块。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。如果将其指定为“全线性”，则选择所有线性模块，不包括输出层。如果未指定，将根据模型架构选择模块。如果架构未知，则会引发错误 - 在这种情况下，您应该手动指定目标模块。

except_modules (`Optional[Union[List[str], str]]`) ：不应用适配器的模块的名称。传递字符串时，将执行正则表达式匹配。传递字符串列表时，要么执行精确匹配，要么检查模块名称是否以任何传递的字符串结尾。

Decomposition_method (`str`) ：投影仪`P_proj`如何从`P`衍生而来。 `"relu"`（默认，非正交`P @ relu(P).T`）或`"qr"`（正交`Q_P @ Q_P.T`）。init_scale (`float`) ：应用于用于初始化注入矩阵`R`的标准差的缩放（仅在`init_weights=False`时使用）。较小的值使注入的更新在初始化时更接近于零。默认为`1.0`。

alpha (`Optional[int]`) ：注入项的缩放因子，按`alpha / r`缩放（类似于LoRA的alpha）。如果`None`，则不应用缩放（因子`1.0`）。子空间移除项不受影响。

para (`bool`) ：是否使用PaRa方法：纯子空间去除（`delta = -P_proj @ W`），无注入项。当`True`、`R`未创建时，`P`是唯一可训练的矩阵，并且适配器不能是初始化时的恒等式。默认为 `False`（完全 DEFT）。

fan_in_fan_out (`bool`) ：如果要替换的图层存储像 (fan_in, fan_out) 这样的权重，则将其设置为 `True`。例如，gpt-2 使用 `Conv1D` 来存储 (fan_in, fan_out) 等权重，因此应将其设置为 `True`。

deft_dropout (`float`) ：应用于层输入的丢弃概率。默认为 `0.0`。init_weights (`bool`) ：是否使用 DEFT 的默认（身份）初始化适配器权重，因此适配器在训练开始时是无操作的。除非您确切知道自己在做什么，否则请勿更改此设置。默认为 `True`。

Layers_to_transform (`Union[List[int], int]`) ：要变换的图层索引。如果传递了一个整数列表，它会将适配器应用于此列表中指定的层索引。如果传递单个整数，它将在该索引处的图层上应用变换。

Layers_pattern (`Optional[Union[List[str], str]]`) ：图层图案名称，仅当`layers_to_transform`与`None`不同时使用。这应该针对模型的 `nn.ModuleList`，通常称为 `'layers'` 或 `'h'`。

bias (`str`) ：DEFT 的偏置类型。可以是 `'none'`、`'all'` 或 `'deft_only'`。

module_to_save (`List[str]`) ：除了适配器层之外要设置为可训练并保存在最终检查点中的模块列表。

这是存储[DeftModel](/docs/peft/v0.20.0/en/package_reference/deft#peft.DeftModel)配置的配置类。DEFT（分解高效微调）通过残差投影更新执行知识注入。对于
冻结的基础权重`W`，低阶投影方向`P`（形状`out_features x r`）和注入矩阵
学习了`R`（形状`r x in_features`）。调整后的权重为`W' = (I - P_proj) @ W + Q_P @ R`，其中
投影仪`P_proj`根据`decomposition_method`衍生自`P`：

- `"relu"`（默认）：`Q_P = P`、`P_proj = P @ relu(P).T`（非正交投影）
- `"qr"`: `Q_P = qr(P)`, `P_proj = Q_P @ Q_P.T`（正交投影）

默认情况下 (`init_weights=True`) `R` 已初始化，因此更新是初始化时的精确身份（调整后的权重）
等于`W`），因此训练从预训练的权重开始并学习注入。

## DeftModel[[peft.DeftModel]]

#### peft.DeftModel[[peft.DeftModel]]

```python
peft.DeftModel(model, peft_config: Union[PeftConfig, dict[str, PeftConfig]], adapter_name: str, low_cpu_mem_usage: bool = False, state_dict: Optional[dict[str, torch.Tensor]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/deft/model.py#L27)

**参数：**

model (`torch.nn.Module`) ：适配器调谐器层将附加到的模型。

config ([DeftConfig](/docs/peft/v0.20.0/en/package_reference/deft#peft.DeftConfig)) ：DEFT模型的配置。

adapter_name (`str`) ：适配器的名称，默认为`"default"`。

low_cpu_mem_usage（`bool`，`optional`，默认为`False`）：在元设备上创建空适配器权重。对于加快加载过程很有用。

**返回：** `torch.nn.Module`

DEFT 模型。从预训练模型创建 DEFT（分解高效微调）模型。

DEFT 冻结基本权重，并根据目标模块学习低秩投影方向 `P` 和注入
矩阵`R`。有效权重变为`(I - P_proj) @ W + Q_P @ R`，将`W`的一个子空间替换为新的
注入的内容（有关可用的 `decomposition_method` 变体，请参阅 [DeftConfig](/docs/peft/v0.20.0/en/package_reference/deft#peft.DeftConfig)）。

**属性**：
- **model** (`~torch.nn.Module`) -- 要适配的模型。
- **peft_config** ([DeftConfig](/docs/peft/v0.20.0/en/package_reference/deft#peft.DeftConfig))：DEFT 模型的配置。

### 辅助方法
https://huggingface.co/docs/peft/v0.20.0/package_reference/helpers.md