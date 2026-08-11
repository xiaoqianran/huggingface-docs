<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 配置

`PeftConfigMixin` 是存储[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel) 适配器配置的基础配置类，[PromptLearningConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PromptLearningConfig) 是软提示方法（p 调优、前缀调优、提示调优）的基础配置类。这些基类包含用于从 Hub 保存和加载模型配置的方法，指定要使用的 PEFT 方法、要执行的任务类型以及模型配置（例如层数和注意力头数）。

## PeftConfigMixin[[peft.config.PeftConfigMixin]]

#### peft.config.PeftConfigMixin[[peft.config.PeftConfigMixin]]

```python
peft.config.PeftConfigMixin(task_type: Optional[TaskType] = None, peft_type: Optional[PeftType] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L77)

**参数：**

peft_type (Union[`~peft.utils.config.PeftType`, `str`]) ：要使用的 Peft 方法的类型。

这是 PEFT 适配器模型的基本配置类。它包含了所有通用的方法
PEFT 适配器型号。该类继承自[PushToHubMixin](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.utils.PushToHubMixin)，其中包含以下方法
将您的模型推送到 Hub。方法 `save_pretrained` 会将适配器模型的配置保存在
目录。方法 `from_pretrained` 将从目录加载适配器模型的配置。

#### check_kwargs[[peft.config.PefConfigMixin.check_kwargs]]

```python
check_kwargs(**kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L328)

在初始化配置实例之前检查 kwargs。子类可以重写此方法来添加特定的检查。

#### from_json_file[[peft.config.PeftConfigMixin.from_json_file]]

```python
from_json_file(path_json_file: str, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L266)

**参数：**

path_json_file (`str`) ：json 文件的路径。

从 json 文件加载配置文件。

#### from_peft_type[[peft.config.PefConfigMixin.from_peft_type]]

```python
from_peft_type(**kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L165)

**参数：**

kwargs（配置关键字参数）：传递给配置初始化的关键字参数。

此方法从一组 kwargs 加载适配器模型的配置。

适当的配置类型由 `peft_type` 参数确定。如果没有提供`peft_type`，
调用类类型被实例化。

#### from_pretrained[[peft.config.PefConfigMixin.from_pretrained]]

```python
from_pretrained(pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L230)

**参数：**

pretrained_model_name_or_path (`str`) ：保存配置的目录或 Hub 存储库 ID。

kwargs（附加关键字参数，*可选*）：传递给子类初始化的附加关键字参数。

此方法从目录加载适配器模型的配置。

#### save_pretrained[[peft.config.PefConfigMixin.save_pretrained]]

```python
save_pretrained(save_directory: str, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L132)**参数：**

save_directory (`str`) ：保存配置的目录。

kwargs（附加关键字参数，*可选*）：传递给 [push_to_hub](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.utils.PushToHubMixin.push_to_hub) 方法的附加关键字参数。

此方法将适配器模型的配置保存在目录中。

#### to_dict[[peft.config.PefConfigMixin.to_dict]]

```python
to_dict()
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L126)

以字典形式返回适配器模型的配置。

## PeftConfig[[peft.PeftConfig]]

#### peft.PeftConfig[[peft.PeftConfig]]

```python
peft.PeftConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L351)

**参数：**

peft_type (Union[`~peft.utils.config.PeftType`, `str`]) ：要使用的 Peft 方法的类型。

task_type (Union[`~peft.utils.config.TaskType`, `str`]) ：要执行的任务类型。

inference_mode (`bool`, 默认为 `False`) : 是否在推理模式下使用 Peft 模型。

这是存储[PeftModel](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel)配置的基本配置类。

## PromptLearningConfig[[peft.PromptLearningConfig]]

#### peft.PromptLearningConfig[[peft.PromptLearningConfig]]

```python
peft.PromptLearningConfig(task_type: Optional[Union[str, TaskType]] = None, peft_type: Optional[Union[str, PeftType]] = None, auto_mapping: Optional[dict] = None, peft_version: Optional[str] = None, base_model_name_or_path: Optional[str] = None, revision: Optional[str] = None, inference_mode: bool = False, num_virtual_tokens: int = None, token_dim: int = None, num_transformer_submodules: Optional[int] = None, num_attention_heads: Optional[int] = None, num_layers: Optional[int] = None, modules_to_save: Optional[list[str]] = None)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/config.py#L371)

**参数：**

num_virtual_tokens (`int`) ：要使用的虚拟代币数量。

token_dim (`int`) ：基础 Transformer 模型的隐藏嵌入维度。num_transformer_submodules (`int`) ：基本变压器模型中变压器子模块的数量。

num_attention_heads (`int`) ：基础 Transformer 模型中的注意力头数量。

num_layers (`int`) ：基本变压器模型中的层数。

这是存储 `PrefixTuning`、[PromptEncoder](/docs/peft/v0.20.0/en/package_reference/p_tuning#peft.PromptEncoder) 的配置的基本配置类，或者
`PromptTuning`。

### 稀疏高等级适配器
https://huggingface.co/docs/peft/v0.20.0/package_reference/shira.md