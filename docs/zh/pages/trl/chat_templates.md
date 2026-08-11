<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 聊天模板

[chat template](https://huggingface.co/docs/transformers/en/chat_templating) 是 Jinja2 片段，它将消息格式化为模型训练所用的字符串。例如：

```python
>>> from transformers import AutoTokenizer
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B-Instruct")
>>> tokenizer.chat_template
"{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
>>> tokenizer.apply_chat_template([{"role": "user", "content": "Hi!"}], tokenize=False)
'<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nHi!<|im_end|>\n'
```

在大多数情况下，您无需担心聊天模板：模型随标记生成器一起提供模板，TRL 会为您应用它。整个事情都是透明的。但某些 TRL 配方依赖于大多数已发布模板不包含的功能：

- **具有 `assistant_only_loss=True`** 的 SFT 需要在辅助输出周围有 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 标记，因此损失掩码只能针对辅助标记。
- **带有工具调用的 GRPO** 需要模板“保留前缀”：附加工具消息不得更改较早消息的呈现方式。

TRL 在 [⟦T7⟧](https://github.com/huggingface/trl/tree/main/trl/chat_templates) 下为常见系列（Qwen、Llama、DeepSeek-V3、GPT-OSS 等）提供修补模板，并自动将它们交换为支持的模型。对于任何其他模型，您需要自己修补其模板。本页的其余部分列出了捆绑的内容。

## 支持的模型系列TRL 存储原始模板的参考副本，以便它可以在初始化时识别受支持的模型，并在需要时交换训练模板。已识别以下系列：Cohere、Cohere2、DeepSeek-V3、Gemma、Gemma3、Gemma4、GLM-4-MoE、GPT-OSS、Idefics3、Llama 3 / 3.1 / 3.2、Llava-Next、Nemotron 3（Nano、Super、Ultra）、Phi-3、Phi-3.5、Qwen2-VL、Qwen2.5、 Qwen2.5-VL、Qwen3（包括 Instruct-2507 变体）、Qwen3-VL、Qwen3.5、Qwen3.6。

## 培训模板

修复了特定于训练的问题的修补模板。当启用工具 (GRPO) 或 `assistant_only_loss=True` (SFT) 时，在初始化时交换。

### `cohere_training.jinja`

修补了 Cohere 模板。与 `cohere.jinja` 的差异：

用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `cohere2_training.jinja`

修补了 Cohere2 模板。与 `cohere2.jinja` 的差异：

将角色分派 `&#123;% endif %&#125;` 之后的尾随 `<|END_OF_TURN_TOKEN|>` 移至每个角色分支，并用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装辅助分支 (`<|START_RESPONSE|>...<|END_RESPONSE|><|END_OF_TURN_TOKEN|>`)，以便 `return_assistant_tokens_mask=True` 为 SFT 仅辅助损失生成正确的掩码。

### `deepseekv3_training.jinja`

修补了 DeepSeek-V3 模板。与 `deepseekv3.jinja` 的差异：- 在`tool['function']['arguments']`上使用`| tojson`，以便`arguments`可以作为`dict`传递（每个[transformers docs](https://huggingface.co/docs/transformers/en/chat_extras#tool-calling-example)的记录格式）。原始模板使用原始字符串连接，这会在字典输入上崩溃。
- 使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 标记包装助手消息输出，以实现仅 SFT 助手丢失。

### `gemma_training.jinja`

修补了 Gemma 模板（由 Gemma 和 Gemma2 共享，它们提供相同的聊天模板）。与 `gemma.jinja` 的差异：

拆分统一的助手输出，以便 `<start_of_turn>model\n` 标头（提示提示，不是由模型生成）位于生成块之外，并使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 标记包装助手内容，以实现仅 SFT 助手损失。

### `gemma3_training.jinja`

修补了 Gemma 3 模板。与 `gemma_training.jinja` 相同的差异（将统一输出行拆分为特定于角色的分支，因此 `<start_of_turn>model\n` 提示提示位于生成块之外，并使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手内容），应用于支持系统消息和多模式内容块的 Gemma 3 基本模板。

### `glm4moe_training.jinja`

修补了 GLM-4-MoE 模板。与 `glm4moe.jinja` 的差异：

解析前需要同时存在 `<think>` 和 `</think>`，以避免模型仅生成一个标签时出现错误拆分：

```diff
- {%- if '</think>' in content %}
+ {%- if '<think>' in content and '</think>' in content %}
```使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 标记包裹助手消息输出（包括思维块和工具调用），以实现仅 SFT 助手损失。

### `qwen3_training.jinja`

修补了 Qwen3 模板。与 `qwen3.jinja` 的差异：

解析前需要同时存在 `<think>` 和 `</think>`，以避免模型仅生成一个标签时出现错误拆分：

```diff
- {%- if '</think>' in content %}
+ {%- if '<think>' in content and '</think>' in content %}
```

无论消息的位置如何，始终包含思维块。原始版本基于`loop.last`有条件地省略它，这会在附加工具消息时更改辅助渲染，从而破坏前缀保留：

```diff
- {%- if loop.index0 > ns.last_query_index %}
-     {%- if loop.last or (not loop.last and reasoning_content) %}
-         {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
-     {%- else %}
-         {{- '<|im_start|>' + message.role + '\n' + content }}
-     {%- endif %}
- {%- else %}
-     {{- '<|im_start|>' + message.role + '\n' + content }}
- {%- endif %}
+ {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
```

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `qwen3_vl_training.jinja`

修补了 Qwen3-VL 模板。与 `qwen3_vl.jinja` 的差异：

用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出（`content` 和 `tool_calls`），以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `gptoss_training.jinja`

修补了 GPT-OSS 模板。与 `gptoss.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `idefics3_training.jinja`

修补了 Idefics3 模板。与 `idefics3.jinja` 的差异：将助手消息拆分到其自己的分支中，以便 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 标记包装助手内容。这使得 `return_assistant_tokens_mask=True` 能够为仅 SFT 助手损失生成正确的掩模。

### `llama3_training.jinja`

修补了 Llama 3 模板。与 `llama3.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `llava_next_training.jinja`

修补了 Llava-Next 模板。与 `llava_next.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `nemotron_3_nano_training.jinja`

修补 Nemotron Nano 模板。与 `nemotron_3_nano.jinja` 的差异：原始版本已经保留了前缀，因此唯一的变化是用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手丢失生成正确的掩码。

### `nemotron_3_super_training.jinja`

修补了 Nemotron 超级模板。 Diff vs `nemotron_3_super.jinja`：与 `nemotron_3_nano_training.jinja` 相同 — 原始版本已经保留前缀，因此唯一的变化是用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手丢失生成正确的掩码。

### `nemotron_3_ultra_training.jinja`已修补 Nemotron Ultra 模板。 Diff vs `nemotron_3_ultra.jinja`：与 `nemotron_3_nano_training.jinja` 相同 — 原始版本已经保留前缀，因此唯一的变化是使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手丢失生成正确的掩码。

### `phi3_training.jinja`

修补了 Phi-3 模板。与 `phi3.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `phi3_5_training.jinja`

修补了 Phi-3.5 模板。与 `phi3_5.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `qwen2_5_training.jinja`

修补了 Qwen2.5 模板。与 `qwen2_5.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `qwen2_5_vl_training.jinja`

修补了 Qwen2.5-VL 模板（也用于 Qwen2-VL，它提供了字节相同的模板）。与 `qwen2_5_vl.jinja` 的差异：

将助手消息拆分到其自己的分支中，以便 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 标记包装助手内容。  这使得 `return_assistant_tokens_mask=True` 能够为 SFT 辅助损失生成正确的掩模。

### `qwen3_instruct_2507_training.jinja`修补了 Qwen3-Instruct-2507 模板（由 `Qwen3-4B-Instruct-2507` 等模型使用，该模型提供了一个更简单的 Qwen3 变体，没有 `reasoning_content` / `<think>` 解析、`multi_step_tool` 跟踪或 `enable_thinking` 标志）。与 `qwen3_instruct_2507.jinja` 的差异：

使用 `&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 包装助手消息输出，以便 `return_assistant_tokens_mask=True` 为仅 SFT 助手损失生成正确的掩码。

### `qwen3_5_think_training.jinja` / `qwen3_5_nothink_training.jinja`

修补了 Qwen3.5 模板，两种风格共享逻辑（它们仅在 `enable_thinking` 标志的默认值上有所不同 - `qwen3_5_think_training.jinja` 默认为启用思考，由 Qwen3.5-4B 及更大版本使用；`qwen3_5_nothink_training.jinja` 默认为禁用思考，由 Qwen3.5-2B 及更小版本使用）。 Diff 与 `qwen3_5_think.jinja` / `qwen3_5_nothink.jinja`：与 `qwen3_training.jinja` 相同的一组更改 — 要求在解析之前同时存在 `<think>` 和 `</think>`，删除 `loop.index0 > ns.last_query_index` 条件，以便始终发出思考块（前缀保留），并将辅助输出包装为`&#123;% generation %&#125;` / `&#123;% endgeneration %&#125;` 仅用于 SFT 助手损失的标记。

### `qwen3_6_training.jinja`修补了 Qwen3.6 模板。 Diff vs `qwen3_6.jinja`：与 `qwen3_training.jinja` 相同的一组更改 — 要求在解析之前同时存在 `<think>` 和 `</think>`，删除 `loop.index0 > ns.last_query_index` 条件，以便始终发出思考块（前缀保留），并用 `&#123;% generation %&#125;` / 包装助手输出`&#123;% endgeneration %&#125;` 仅用于 SFT 辅助损失的标记。

## 相关实用程序

请参阅 [Chat Template Utilities](chat_template_utils) 了解在这些模板上运行的辅助函数（[clone_chat_template()](/docs/trl/v1.9.2/en/chat_template_utils#trl.clone_chat_template)、`is_chat_template_prefix_preserving`、[get_training_chat_template()](/docs/trl/v1.9.2/en/chat_template_utils#trl.get_training_chat_template)）。

### PRM 培训师
https://huggingface.co/docs/trl/v1.9.2/prm_trainer.md