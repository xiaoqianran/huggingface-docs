<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型合并[[peft.utils.merge_utils.prune]]

PEFT 通过 TIES 和 DARE 方法为 [merging LoRA adapters](../developer_guides/model_merging) 提供了多个内部实用程序。

#### peft.utils.merge_utils.prune[[peft.utils.merge_utils.prune]]

```python
peft.utils.merge_utils.prune(tensor: Tensor, density: float, method: typing.Literal['magnitude', 'random'], rescale: bool = False)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L75)

**参数：**

张量 (`torch.Tensor`) -- 要修剪的张量。

密度 (`float`) -- 要保留的值的分数。应该在[0,1]内。

method (`str`) -- 用于修剪的方法。应该是[“大小”，“随机”]之一。

rescale (`bool`) --是否重新缩放结果以保留原始张量的期望值。

**返回：** `torch.Tensor`

剪枝后的张量。

根据`method`修剪任务张量的值。

#### peft.utils.merge_utils.calculate_majority_sign_mask[[peft.utils.merge_utils.calculate_majority_sign_mask]]

```python
peft.utils.merge_utils.calculate_majority_sign_mask(tensor: Tensor, method: typing.Literal['total', 'frequency'] = 'total')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L103)

**参数：**

张量 (`torch.Tensor`) --从中获取掩模的张量。

method (`str`) -- 用于获取掩码的方法。应该是[“总数”，“频率”]之一。

**返回：** `torch.Tensor`

大多数人签署面具。

获取任务张量中多数符号的掩码。任务张量堆叠在维度 0 上。

#### peft.utils.merge_utils.disjoint_merge[[peft.utils.merge_utils.disjoint_merge]]

```python
peft.utils.merge_utils.disjoint_merge(task_tensors: Tensor, majority_sign_mask: Tensor)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L128)

**参数：**task_tensors (`torch.Tensor`) -- 要合并的任务张量。

Majority_sign_mask (`torch.Tensor`) --任务张量中多数符号的掩码。

**返回：** `torch.Tensor`

合并的张量。

使用不相交合并来合并任务张量。

#### peft.utils.merge_utils.task_arithmetic[[peft.utils.merge_utils.task_arithmetic]]

```python
peft.utils.merge_utils.task_arithmetic(task_tensors: list, weights: Tensor)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L144)

**参数：**

task_tensors(`List[torch.Tensor]`) --要合并的任务张量。

权重 (`torch.Tensor`) --任务张量的权重。

**返回：** `torch.Tensor`

合并的张量。

使用 `task arithmetic` 合并任务张量。

#### peft.utils.merge_utils.ties[[peft.utils.merge_utils.ties]]

```python
peft.utils.merge_utils.ties(task_tensors: list, weights: Tensor, density: float, majority_sign_method: typing.Literal['total', 'frequency'] = 'total')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L185)

**参数：**

task_tensors(`List[torch.Tensor]`) --要合并的任务张量。

权重 (`torch.Tensor`) --任务张量的权重。

密度 (`float`) -- 要保留的值的分数。应该在[0,1]内。

Majority_sign_method (`str`) ：用于获取多数符号掩码的方法。应该是[“总数”，“频率”]之一。

**返回：** `torch.Tensor`

合并的张量。

使用 `ties` 合并任务张量。

#### peft.utils.merge_utils.dare_线性[[peft.utils.merge_utils.dare_linear]]

```python
peft.utils.merge_utils.dare_linear(task_tensors: list, weights: Tensor, density: float)
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L217)

**参数：**

task_tensors(`List[torch.Tensor]`) --要合并的任务张量。

权重 (`torch.Tensor`) --任务张量的权重。密度 (`float`) -- 要保留的值的分数。应该在[0,1]内。

**返回：** `torch.Tensor`

合并的张量。

使用 `dare linear` 合并任务张量。

#### peft.utils.merge_utils.dare_ties[[peft.utils.merge_utils.dare_ties]]

```python
peft.utils.merge_utils.dare_ties(task_tensors: list, weights: Tensor, density: float, majority_sign_method: typing.Literal['total', 'frequency'] = 'total')
```

[Source](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/utils/merge_utils.py#L239)

**参数：**

task_tensors(`List[torch.Tensor]`) --要合并的任务张量。

权重 (`torch.Tensor`) --任务张量的权重。

密度 (`float`) -- 要保留的值的分数。应该在[0,1]内。

Majority_sign_method (`str`) ：用于获取多数符号掩码的方法。应该是[“总数”，“频率”]之一。

**返回：** `torch.Tensor`

合并的张量。

使用 `dare ties` 合并任务张量。

### 可训练代币
https://huggingface.co/docs/peft/v0.20.0/package_reference/trainable_tokens.md