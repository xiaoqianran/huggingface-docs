<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据实用程序

## is_conversational[[trl.is_conversational]]

- **示例** (`dict[str, Any]`) --
  数据集的单个数据条目。根据数据集类型，该示例可以具有不同的键。如果数据是会话格式，则为`bool``True`，否则为`False`。

检查示例是否采用对话格式。

示例：

```python
>>> example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
>>> is_conversational(example)
True

>>> example = {"prompt": "The sky is"}
>>> is_conversational(example)
False
```

## Maybe_convert_to_chatml[[trl.maybe_convert_to_chatml]]

- **示例** (`dict[str, list]`) --
  包含消息列表的单个数据条目。`dict[str, list]`示例重新格式化为 ChatML 样式。

将包含字段 `from` 和 `value` 的对话数据集转换为 ChatML 格式。

此函数修改对话数据以符合 OpenAI 的 ChatML 格式：
- 将消息字典中的键 `"from"` 替换为 `"role"`。
- 将消息字典中的键 `"value"` 替换为 `"content"`。
- 将 `"conversations"` 重命名为 `"messages"` 以与 ChatML 保持一致。

示例：
```python
>>> from trl import maybe_convert_to_chatml

>>> example = {
...     "conversations": [
...         {"from": "user", "value": "What color is the sky?"},
...         {"from": "assistant", "value": "It is blue."},
...     ]
... }
>>> maybe_convert_to_chatml(example)
{'messages': [{'role': 'user', 'content': 'What color is the sky?'},
              {'role': 'assistant', 'content': 'It is blue.'}]}
```

## extract_prompt[[trl.extract_prompt]]- **示例** (`dict[str, list]`) --
  表示首选项数据集中单个数据条目的字典。它必须包含密钥
  `"chosen"` 和 `"rejected"`，其中每个值都是会话值或标准值 (`str`)。`dict[str, list]`包含以下内容的字典：
- `"prompt"`：“选择”和“拒绝”补全之间的最长公共前缀。
- `"chosen"`：“选择”完成的剩余部分，已删除提示。
- `"rejected"`：“拒绝”完成的剩余部分，并删除提示。

从首选项数据示例中提取共享提示，其中提示隐含在所选和
拒绝完成。

该函数识别“选择的”和“选择的”之间对话轮次的最长公共序列（前缀）
“拒绝”完成并将其提取为提示。然后它会从相应的“选择”中删除此提示
并“拒绝”完成。

示例：

```python
>>> example = {
...     "chosen": [
...         {"role": "user", "content": "What color is the sky?"},
...         {"role": "assistant", "content": "It is blue."},
...     ],
...     "rejected": [
...         {"role": "user", "content": "What color is the sky?"},
...         {"role": "assistant", "content": "It is green."},
...     ],
... }
>>> extract_prompt(example)
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
```

或者，使用 `Dataset` 的 `map` 方法：

```python
>>> from trl import extract_prompt
>>> from datasets import Dataset

>>> dataset_dict = {
...     "chosen": [
...         [
...             {"role": "user", "content": "What color is the sky?"},
...             {"role": "assistant", "content": "It is blue."},
...         ],
...         [
...             {"role": "user", "content": "Where is the sun?"},
...             {"role": "assistant", "content": "In the sky."},
...         ],
...     ],
...     "rejected": [
...         [
...             {"role": "user", "content": "What color is the sky?"},
...             {"role": "assistant", "content": "It is green."},
...         ],
...         [
...             {"role": "user", "content": "Where is the sun?"},
...             {"role": "assistant", "content": "In the sea."},
...         ],
...     ],
... }
>>> dataset = Dataset.from_dict(dataset_dict)
>>> dataset = dataset.map(extract_prompt)
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
```

## 取消配对偏好数据集[[trl.取消配对偏好数据集]]- **数据集**（`Dataset`或`DatasetDict`或`IterableDataset`或`IterableDatasetDict`）--
  优先选择要取消配对的数据集。数据集必须包含列 `"chosen"`、`"rejected"`，并且可选
  `"prompt"`。
- **map_kwargs**（`dict`，*可选*）--
  取消配对首选项时要传递给数据集的映射方法的其他关键字参数。`Dataset`或`DatasetDict`或`IterableDataset`或`IterableDatasetDict`未配对的首选项数据集。

取消配对首选项数据集。

输出包含 `"prompt"`、`"completion"` 和 `"label"` 以及任何额外的列，这些列是重复的
每个选择和拒绝的行。

示例：

```python
>>> from datasets import Dataset

>>> dataset_dict = {
...     "prompt": ["The sky is", "The sun is"],
...     "chosen": [" blue.", "in the sky."],
...     "rejected": [" green.", " in the sea."],
... }
>>> dataset = Dataset.from_dict(dataset_dict)
>>> dataset = unpair_preference_dataset(dataset)
>>> dataset
Dataset({
    features: ['prompt', 'completion', 'label'],
    num_rows: 4
})

>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.', 'label': True}
```

### 快速入门
https://huggingface.co/docs/trl/v1.9.2/quickstart.md