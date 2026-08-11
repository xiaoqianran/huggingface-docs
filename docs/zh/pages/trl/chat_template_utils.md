<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 聊天模板实用程序

有关与 TRL 捆绑在一起的聊天模板的概述以及训练补丁背后的基本原理，请参阅[Chat Templates](chat_templates)。

## 克隆聊天模板[[trl.克隆聊天模板]]

- **型号** ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel)) --
  要更新的模型。
- **分词器** ([PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)) --
  要更新的标记器。
- **source_tokenizer_path** (`str`) --
  要从中克隆的预训练分词器的路径或标识符。
- **resize_to_multiple_of** （`int` 或 `None`，*可选*，默认为 `64`）--
  嵌入层的大小将调整为新的词汇大小。如果这不是`None`，则会四舍五入
  新的词汇量大小最接近该值的倍数。模型 ([PreTrainedModel](https://huggingface.co/docs/transformers/v5.14.1/en/main_classes/model#transformers.PreTrainedModel))更新了模型，调整了令牌嵌入的大小并配置了 EOS 令牌。
分词器（[PreTrainedTokenizerBase](https://huggingface.co/docs/transformers/v5.14.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase)）：
使用聊天模板和应用的特殊令牌更新了令牌生成器。
添加的令牌（`list[int]`）：
从源标记生成器添加到标记生成器的标记列表。

将聊天模板从源标记生成器克隆到目标标记生成器并相应地更新模型。这个功能：
- 将聊天模板从源标记生成器复制到目标标记生成器。
- 将任何新标记从源标记生成器添加到目标标记生成器。
- 在令牌生成器和模型中设置和同步 EOS 令牌。
- 调整模型的标记嵌入大小以匹配新的词汇量大小，可以选择将其四舍五入为
  指定值。在这种情况下，虚拟标记会添加到标记生成器中，以确保词汇表大小匹配
  嵌入尺寸。

示例：
```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from trl import clone_chat_template

>>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
>>> model, tokenizer, added_tokens = clone_chat_template(model, tokenizer, "Qwen/Qwen3-0.6B")
```

## is_chat_template_prefix_preserving[[trl.chat_template_utils.is_chat_template_prefix_preserving]]

- **处理类**（`PreTrainedTokenizerBase`或`ProcessorMixin`）--
  要检查的分词器或处理器实例。如果聊天模板保留前缀，则`bool``True`，否则为`False`。

检查聊天模板在应用时是否保留前缀。

无论后面的消息是什么，保留前缀的聊天模板都会以相同的方式呈现较早的消息。这个
属性是 `_get_tool_suffix_ids` 所必需的，它通过比较来提取工具响应格式化标记
附加和不附加工具消息的标记化。

## get_training_chat_template[[trl.get_training_chat_template]]- **处理类**（`PreTrainedTokenizerBase`或`ProcessorMixin`）--
  要检查的分词器或处理器实例。`str`或`None`与训练兼容的聊天模板，或`None`（如果不需要修补）。

如果需要，获取与培训兼容的聊天模板。

返回一个修补后的聊天模板，该模板保留前缀并包含 `{%% generation %%}` / `{%% end Generation
%%}` markers for assistant-only loss masking. Returns `None` 如果模板已满足这两个要求。
目前有 Cohere、Cohere 2、DeepSeek-V3、Gemma、Gemma 2、Gemma 3、GLM-4-MoE、GPT-OSS、Idefics3、LLaMA 3、Phi-3、
Phi-3.5、Qwen2-VL、Qwen2.5、Qwen2.5-VL、Qwen3（包括 Instruct-2507 变体）、Qwen3-VL、Qwen3.5 和 Qwen3.6
都支持。

示例：

```python
>>> from trl.chat_template_utils import get_training_chat_template
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
>>> messages1 = [
...     {"role": "user", "content": "What is 2 * 3?"},
...     {
...         "role": "assistant",
...         "content": "",
...         "tool_calls": [{"type": "function", "function": {"name": "multiply", "arguments": {"a": 2, "b": 3}}}],
...     },
... ]
>>> messages2 = messages1 + [
...     {"role": "tool", "name": "multiply", "content": "6"},
... ]
>>> tokenizer.apply_chat_template(messages1, tokenize=False)
'<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n'

>>> tokenizer.apply_chat_template(messages2, tokenize=False, add_generation_prompt=True)
'<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n6\n</tool_response><|im_end|>\n<|im_start|>assistant\n'

>>> #                                                        ^ think tags missing
>>> chat_template = get_training_chat_template(tokenizer)
>>> tokenizer.apply_chat_template(messages1, tokenize=False, chat_template=chat_template)
'<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n'

>>> tokenizer.apply_chat_template(
...     messages2, tokenize=False, add_generation_prompt=True, chat_template=chat_template
... )
'<|im_start|>user\nWhat is 2 * 3?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "multiply", "arguments": {"a": 2, "b": 3}}\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n6\n</tool_response><|im_end|>\n<|im_start|>assistant\n'
```

### 实验性的
https://huggingface.co/docs/trl/v1.9.2/experimental_overview.md