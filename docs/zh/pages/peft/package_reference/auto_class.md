<!-- huggingface-docs: machine-translated zh-CN from English source -->

# AutoPeft模型

`AutoPeftModel` 类通过自动从配置文件中推断来加载适合任务类型的 PEFT 模型。它们旨在通过一行代码快速轻松地加载 PEFT 模型，而无需担心您需要哪个确切的模型类或手动加载 [PeftConfig](/docs/peft/v0.20.0/en/package_reference/config#peft.PeftConfig)。

## AutoPeftModel[[peft.AutoPeftModel]]

#### peft.AutoPeftModel[[peft.AutoPeftModel]]

```python
peft.AutoPeftModel(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L197)

#### from_pretrained[[peft.AutoPeftModel.from_pretrained]]

```python
from_pretrained(pretrained_model_name_or_path, adapter_name: str = 'default', is_trainable: bool = False, config: Optional[PeftConfig] = None, revision: Optional[str] = None, import_allowlist: Optional[list[str]] = None, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L88)

**参数：**

import_allowlist（`list[str]`，*可选*，默认为`{get_default_import_allowlist()}`）：AutoPeftModel 将尝试实例化适配器配置中配置的基本模型。由于此操作可能需要导入其他包，因此此允许列表是防止导入恶意包的安全措施。如果包的导入名称不在默认值中，您可能需要在此处指定该名称。

围绕用户加载 PEFT 模型所需执行的所有预处理步骤的包装器。夸格人
被传递到 `PeftConfig`，它会自动过滤 Hub 方法的 kwargs 并
配置对象初始化。参数与[PeftModel.from_pretrained()](/docs/peft/v0.20.0/en/package_reference/peft_model#peft.PeftModel.from_pretrained)相当。下面记录了差异。

## AutoPeftModelForCausalLM[[peft.AutoPeftModelForCausalLM]]

#### peft.AutoPeftModelForCausalLM[[peft.AutoPeftModelForCausalLM]]

```python
peft.AutoPeftModelForCausalLM(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L202)

## AutoPeftModelForSeq2SeqLM[[peft.AutoPeftModelForSeq2SeqLM]]

#### peft.AutoPeftModelForSeq2SeqLM[[peft.AutoPeftModelForSeq2SeqLM]]

```python
peft.AutoPeftModelForSeq2SeqLM(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L207)

## AutoPeftModelForSequenceClassification[[peft.AutoPeftModelForSequenceClassification]]

#### peft.AutoPeftModelForSequenceClassification[[peft.AutoPeftModelForSequenceClassification]]

```python
peft.AutoPeftModelForSequenceClassification(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L212)

## AutoPeftModelForTokenClassification[[peft.AutoPeftModelForTokenClassification]]

#### peft.AutoPeftModelForTokenClassification[[peft.AutoPeftModelForTokenClassification]]

```python
peft.AutoPeftModelForTokenClassification(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L217)

## AutoPeftModelForQuestionAnswering[[peft.AutoPeftModelForQuestionAnswering]]

#### peft.AutoPeftModelForQuestionAnswering[[peft.AutoPeftModelForQuestionAnswering]]

```python
peft.AutoPeftModelForQuestionAnswering(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L222)

## AutoPeftModelForFeatureExtraction[[peft.AutoPeftModelForFeatureExtraction]]

#### peft.AutoPeftModelForFeatureExtraction[[peft.AutoPeftModelForFeatureExtraction]]

```python
peft.AutoPeftModelForFeatureExtraction(*args, **kwargs)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/auto.py#L227)### 通过 Householder Reflection Adaptation (HRA) 弥合低阶适应和正交适应之间的差距
https://huggingface.co/docs/peft/v0.20.0/package_reference/hra.md