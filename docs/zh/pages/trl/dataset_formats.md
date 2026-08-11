<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据集格式和类型

本指南概述了 TRL 中每个训练器支持的数据集格式和类型。

## 数据集格式和类型概述

- 数据集的*格式*是指数据的结构方式，通常分为*标准*或*会话*。
- *类型*与数据集设计的特定任务相关联，例如*仅提示*或*偏好*。每种类型都以其列为特征，这些列根据任务而变化，如表中所示。类型\格式
    标准型
    会话型
  
  
    语言建模
    
      {“text”：“天空是蓝色的。”}
    
    
      {"messages": [{"role": "user", "content": "天空是什么颜色？"},
              {"role": "assistant", "content": "它是蓝色的。"}]}
    
  
  
    仅提示
    
      {“提示”：“天空是”}
    
    
      {"prompt": [{"role": "user", "content": "天空是什么颜色？"}]}
    
  
  
    及时完成
    
      {"prompt": "天空是",
 “完成”：“蓝色。”}
    
    
      {"prompt": [{"role": "user", "content": "天空是什么颜色？"}],
 "completion": [{"role": "assistant", "content": "它是蓝色的。"}]}
    
  
  
  
    偏好
    
      {"prompt": "天空是",
 "选择": "蓝色。",
 “拒绝”：“绿色。”}
      或者，使用隐式提示：
      {“chosen”：“天空是蓝色的。”，
 "rejected": "天空是绿色的。"}
    
    
      {"prompt": [{"role": "user", "content": "天空是什么颜色？"}],
 "chosen": [{"role": "助理", "content": "它是蓝色的。"}],
 "rejected": [{"role": "assistant", "content": "It is green."}]}
      或者，使用隐式提示：
      {"chosen": [{"role": "user", "content": "天空是什么颜色？"},{"role": "助理", "content": "它是蓝色的。"}],
 "rejected": [{"role": "user", "content": "天空是什么颜色？"},
                {"role": "assistant", "content": "它是绿色的。"}]}
    
  
    不配对偏好
    
      {"prompt": "天空是",
 "completion": "蓝色。",
 “标签”：正确}
    
    
      {"prompt": [{"role": "user", "content": "天空是什么颜色？"}],
 "completion": [{"role": "assistant", "content": "绿色。"}],
 “标签”：假}
    
  
  
    分级监管
    
      {"prompt": "9.8 和 9.11 哪个数字更大？",
 "completions": ["9.8 的小数部分是 0.8。",
                 “9.11的小数部分是0.11。”,
                 "0.11 大于 0.8。",
                 “因此，9.11 > 9.8。”]，
 “标签”：[真，真，假，假]}
    
    
  

### 格式

#### 标准

标准数据集格式通常由纯文本字符串组成。数据集中的列因任务而异。这是 TRL 培训师所期望的格式。以下是不同任务的标准数据集格式的示例：

```python
# Language modeling
language_modeling_example = {"text": "The sky is blue."}
# Preference
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Unpaired preference
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
```

#### 对话对话数据集用于涉及用户和助手之间对话或聊天交互的任务。与标准数据集格式不同，它们包含消息序列，其中每条消息都有一个`role`（例如`"user"`或`"assistant"`）和`content`（消息文本）。

```python
messages = [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing great. How can I help you today?"},
    {"role": "user", "content": "I'd like to show off how chat templating works!"},
]
```

就像标准数据集一样，会话数据集中的列因任务而异。以下是针对不同任务的会话数据集格式的示例：

```python
# Prompt-completion
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
# Preference
preference_example = {
    "prompt": [{"role": "user", "content": "What color is the sky?"}],
    "chosen": [{"role": "assistant", "content": "It is blue."}],
    "rejected": [{"role": "assistant", "content": "It is green."}],
}
```

#### 工具调用

一些聊天模板支持*工具调用*，这允许模型在生成过程中与外部函数（称为**工具**）进行交互。这扩展了模型的对话功能，使其能够在决定调用工具时输出 `"tool_calls"` 字段而不是标准 `"content"` 消息。

助手发起工具调用后，工具执行并返回其输出。然后，助理可以处理该输出并相应地继续对话。

这是工具调用交互的一个简单示例：

```python
messages = [
    {"role": "user", "content": "Turn on the living room lights."},
    {"role": "assistant", "tool_calls": [
        {"type": "function", "function": {
            "name": "control_light",
            "arguments": {"room": "living room", "state": "on"}
        }}]
    },
    {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
    {"role": "assistant", "content": "Done!"}
]
```在通过工具调用准备用于监督微调 (SFT) 的数据集时，数据集包含名为 `tools` 的附加列非常重要。此列包含模型可用工具的列表，聊天模板通常使用该工具来构建系统提示。

这些工具必须以编码的 JSON 模式格式指定。您可以使用 `get_json_schema` 实用程序从 Python 函数签名自动生成此架构：

```python
import json
from transformers.utils import get_json_schema

def control_light(room: str, state: str) -> str:
    """
    Controls the lights in a room.

    Args:
        room: The name of the room.
        state: The desired state of the light ("on" or "off").

    Returns:
        str: A message indicating the new state of the lights.
    """
    return f"The lights in {room} are now {state}."

# Generate JSON schema
json_schema = get_json_schema(control_light)
```

生成的模式如下所示：

```python
{"type": "function", "function": {"name": "control_light", "description": "Controls the lights in a room.", "parameters": {"type": "object", "properties": {"room": {"type": "string", "description": "The name of the room."}, "state": {"type": "string", "description": "The desired state of the light (\"on\" or \"off\")."}}, "required": ["room", "state"]}, "return": {"type": "string", "description": "str: A message indicating the new state of the lights."}}}
```

SFT 的完整数据集条目可能如下所示：

```python
{"messages": messages, "tools": [json_schema]}
```

要获得 `Dataset`，您需要使用 `Json()` 类型作为工具参数，因为它们是任意 JSON 对象，而不是具有固定字段和类型的字典：

```python
from datasets import Dataset

data = [
    {"messages": messages1, "tools": [json_schema1]},
    {"messages": messages2, "tools": [json_schema2]},
]
# auto-apply the Json() type
dataset = Dataset.from_list(data, on_mixed_types="use_json")

# or specify the features manually
from datasets import Features, Json, List, Value

features = Features(
    {
        "messages": List({"role": Value("string"), "content": Value("string"), "tool_calls": List(Json())}),
        "tools": List(Json()),
    }
)
dataset = Dataset.from_list(data, features=features)
```在旧版本的 `datasets` (<4.7.0) that don't have the ⟦T70⟧ type, you should store ⟦T71⟧ as a JSON ⟦T72⟧ (with ⟦T73⟧):

⟦T8⟧

For more detailed information on tool calling, refer to the ⟦T128⟧ and the blog post ⟦T129⟧.

### Harmony

The ⟦T130⟧ was introduced with the ⟦T131⟧. It extends the conversational format by adding richer structure for reasoning, function calls, and metadata about the model’s behavior. Key features include:

- **Developer role** – Provides high level instructions (similar to a system prompt) and lists available tools.
- **Channels** – Separate types of assistant output into distinct streams:

  - ⟦T75⟧ – for internal reasoning, from the key ⟦T76⟧
  - ⟦T77⟧ – for the user-facing answer, from the key ⟦T78⟧
  - ⟦T79⟧ – for tool calls or meta notes

- **Reasoning effort** – Signals how much thinking the model should show (e.g., ⟦T80⟧, ⟦T81⟧, ⟦T82⟧).
- **Model identity** – Explicitly defines the assistant’s persona.

⟦T9⟧

This produces:

⟦T10⟧

For full details on message structure, supported fields, and advanced usage, see the ⟦T132⟧.

### Types

#### Language modeling

A language modeling dataset consists of a column ⟦T83⟧ (or ⟦T84⟧ for conversational datasets) containing a full sequence of text.

⟦T11⟧

#### Prompt-only

In a prompt-only dataset, only the initial prompt (the question or partial sentence) is provided under the key ⟦T85⟧. The training typically involves generating completion based on this prompt, where the model learns to continue or complete the given input.

⟦T12⟧

For examples of prompt-only datasets, refer to the ⟦T133⟧.

> [!TIP]
> 虽然仅提示和语言建模类型相似，但它们在处理输入的方式上有所不同。在仅提示类型中，提示表示期望模型完成或继续的部分输入，而在语言建模类型中，输入被视为完整的句子或序列。 TRL 对这两种类型的处理方式不同。下面的示例显示了每种类型的 `apply_chat_template` 函数的输出差异：
>
> ```python
> from transformers import AutoTokenizer
> from trl import apply_chat_template
>
> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
>
> # Example for prompt-only type
> prompt_only_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}]}
> apply_chat_template(prompt_only_example, tokenizer)
> # Output: {'prompt': '<|user|>\nWhat color is the sky?<|end|>\n<|assistant|>\n'}
>
> # Example for language modeling type
> lm_example = {"messages": [{"role": "user", "content": "What color is the sky?"}]}
> apply_chat_template(lm_example, tokenizer)
> # Output: {'text': '<|user|>\nWhat color is the sky?<|end|>\n<|endoftext|>'}
> ```
>
> - 仅提示输出包括 `'<|assistant|>\n'`，表示助手的回合开始并期望模型生成完成。
> - 相反，语言建模输出将输入视为完整序列并以 `'<|endoftext|>'` 终止，表示文本结束并且不期望任何附加内容。

#### 快速完成

提示完成数据集包括`"prompt"`和`"completion"`。

```python
# Standard format
prompt_completion_example = {"prompt": "The sky is", "completion": " blue."}
# Conversational format
prompt_completion_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                             "completion": [{"role": "assistant", "content": "It is blue."}]}
```

有关提示完成数据集的示例，请参阅[Prompt-completion datasets collection](https://huggingface.co/collections/trl-lib/prompt-completion-datasets-677ea2bb20bbb6bdccada216)。

#### 偏好偏好数据集用于训练模型在同一提示的两个或多个可能完成之间进行选择的任务。该数据集包括一个`"prompt"`、一个`"chosen"`补全和一个`"rejected"`补全。该模型经过训练，可以选择 `"chosen"` 响应而不是 `"rejected"` 响应。
某些数据集可能不包含 `"prompt"` 列，在这种情况下，提示是隐式的并直接包含在 `"chosen"` 和 `"rejected"` 补全中。我们建议尽可能使用明确的提示。

```python
# Standard format
## Explicit prompt (recommended)
preference_example = {"prompt": "The sky is", "chosen": " blue.", "rejected": " green."}
# Implicit prompt
preference_example = {"chosen": "The sky is blue.", "rejected": "The sky is green."}

# Conversational format
## Explicit prompt (recommended)
preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                      "chosen": [{"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "assistant", "content": "It is green."}]}
## Implicit prompt
preference_example = {"chosen": [{"role": "user", "content": "What color is the sky?"},
                                 {"role": "assistant", "content": "It is blue."}],
                      "rejected": [{"role": "user", "content": "What color is the sky?"},
                                   {"role": "assistant", "content": "It is green."}]}
```

有关偏好数据集的示例，请参阅[Preference datasets collection](https://huggingface.co/collections/trl-lib/preference-datasets-677e99b581018fcad9abd82c)。

一些偏好数据集可以通过[the tag ⟦T99⟧ on Hugging Face Hub](https://huggingface.co/datasets?other=dpo)找到。您还可以探索 [librarian-bots' DPO Collections](https://huggingface.co/collections/librarian-bots/direct-preference-optimization-datasets-66964b12835f46289b6ef2fc) 来识别偏好数据集。

#### 未配对的偏好

未配对的偏好数据集与偏好数据集类似，但它不是为同一提示提供 `"chosen"` 和 `"rejected"` 补全，而是包含单个 `"completion"` 和 `"label"` 指示补全是否是首选。

```python
# Standard format
unpaired_preference_example = {"prompt": "The sky is", "completion": " blue.", "label": True}
# Conversational format
unpaired_preference_example = {"prompt": [{"role": "user", "content": "What color is the sky?"}],
                               "completion": [{"role": "assistant", "content": "It is blue."}],
                               "label": True}
```

有关未配对偏好数据集的示例，请参阅[Unpaired preference datasets collection](https://huggingface.co/collections/trl-lib/unpaired-preference-datasets-677ea22bf5f528c125b0bcdf)。

#### 分级监督逐步（或过程）监督数据集类似于 [unpaired preference](#unpaired-preference) 数据集，但包含多个完成步骤，每个步骤都有自己的标签。这种结构对于需要详细、逐步标记的任务非常有用，例如推理任务。通过单独评估每个步骤并提供有针对性的标签，这种方法有助于准确识别推理的正确位置和发生错误的位置，从而可以对推理过程的每个部分进行有针对性的反馈。

```python
stepwise_example = {
    "prompt": "Which number is larger, 9.8 or 9.11?",
    "completions": ["The fractional part of 9.8 is 0.8, while the fractional part of 9.11 is 0.11.", "Since 0.11 is greater than 0.8, the number 9.11 is larger than 9.8."],
    "labels": [True, False]
}
```

有关逐步监督数据集的示例，请参阅[Stepwise supervision datasets collection](https://huggingface.co/collections/trl-lib/stepwise-supervision-datasets-677ea27fd4c5941beed7a96e)。

## 使用哪种数据集类型？

选择正确的数据集类型取决于您正在执行的任务以及您正在使用的 TRL 训练器的具体要求。以下是每个 TRL 训练器支持的数据集类型的简要概述。|培训师|预期数据集类型 |
| --- | --- |
| [DPOTrainer](/docs/trl/v1.9.2/en/bema_for_reference_model#trl.DPOTrainer) | [Preference (explicit prompt recommended)](#preference) |
| [GRPOTrainer](/docs/trl/v1.9.2/en/gspo_token#trl.GRPOTrainer) | [Prompt-only](#prompt-only) |
| [KTOTrainer](/docs/trl/v1.9.2/en/kto_trainer#trl.KTOTrainer) | [Unpaired preference](#unpaired-preference) 或 [Preference (explicit prompt recommended)](#preference) |
| [RewardTrainer](/docs/trl/v1.9.2/en/reward_trainer#trl.RewardTrainer) | [Preference (implicit prompt recommended)](#preference) |
| [RLOOTrainer](/docs/trl/v1.9.2/en/rloo_trainer#trl.RLOOTrainer) | [Prompt-only](#prompt-only) |
| [SFTTrainer](/docs/trl/v1.9.2/en/sft_trainer#trl.SFTTrainer) | [Language modeling](#language-modeling) 或 [Prompt-completion](#prompt-completion) |
| [experimental.bco.BCOTrainer](/docs/trl/v1.9.2/en/bco_trainer#trl.experimental.bco.BCOTrainer) | [Unpaired preference](#unpaired-preference) 或 [Preference (explicit prompt recommended)](#preference) |
| [experimental.cpo.CPOTrainer](/docs/trl/v1.9.2/en/cpo_trainer#trl.experimental.cpo.CPOTrainer) | [Preference (explicit prompt recommended)](#preference) |
| [experimental.gkd.GKDTrainer](/docs/trl/v1.9.2/en/gkd_trainer#trl.experimental.gkd.GKDTrainer) | [Prompt-completion](#prompt-completion) |
| [experimental.nash_md.NashMDTrainer](/docs/trl/v1.9.2/en/nash_md_trainer#trl.experimental.nash_md.NashMDTrainer) | [Prompt-only](#prompt-only) |
| [experimental.online_dpo.OnlineDPOTrainer](/docs/trl/v1.9.2/en/online_dpo_trainer#trl.experimental.online_dpo.OnlineDPOTrainer) | [Prompt-only](#prompt-only) |
| [experimental.orpo.ORPOTrainer](/docs/trl/v1.9.2/en/orpo_trainer#trl.experimental.orpo.ORPOTrainer) | [Preference (explicit prompt recommended)](#preference) |
| [experimental.ppo.PPOTrainer](/docs/trl/v1.9.2/en/ppo_trainer#trl.experimental.ppo.PPOTrainer) |标记化语言建模 |
| [experimental.prm.PRMTrainer](/docs/trl/v1.9.2/en/prm_trainer#trl.experimental.prm.PRMTrainer) | [Stepwise supervision](#stepwise-supervision) |
| [experimental.xpo.XPOTrainer](/docs/trl/v1.9.2/en/xpo_trainer#trl.experimental.xpo.XPOTrainer) | [Prompt-only](#prompt-only) |

## 使用带有 TRL 的任何数据集：预处理和转换

许多数据集采用针对特定任务定制的格式，这些格式可能与 TRL 不直接兼容。要将此类数据集与 TRL 一起使用，您可能需要对其进行预处理并将其转换为所需的格式。

为了使这更容易，我们提供了一组涵盖常见数据集转换的[example scripts](https://github.com/huggingface/trl/tree/main/examples/datasets)。

### 示例：UltraFeedback 数据集

我们以[UltraFeedback dataset](https://huggingface.co/datasets/openbmb/UltraFeedback)为例。这是数据集的预览：

<iframe
  src="https://huggingface.co/datasets/openbmb/UltraFeedback/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
>

如上所示，数据集格式与预期结构不匹配。它不是对话格式，列名称不同，结果涉及不同的模型（例如，Bard、GPT-4）和方面（例如，“乐于助人”、“诚实”）。通过使用提供的转换脚本[⟦T104⟧](https://github.com/huggingface/trl/tree/main/examples/datasets/ultrafeedback.py)，您可以将此数据集转换为不配对的偏好类型，并将其推送到Hub：

```sh
python examples/datasets/ultrafeedback.py --push_to_hub --repo_id trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness
```

转换后，数据集将如下所示：

<iframe
  src="https://huggingface.co/datasets/trl-lib/ultrafeedback-gpt-3.5-turbo-helpfulness/embed/viewer/default/train?row=0"
  frameborder="0"
  width="100%"
  height="560px"
>

现在，您可以将此数据集与 TRL 一起使用！

通过调整提供的脚本或创建您自己的脚本，您可以将任何数据集转换为与 TRL 兼容的格式。

## 转换数据集类型的实用程序

本部分提供示例代码来帮助您在不同数据集类型之间进行转换。虽然可以在应用聊天模板（即标准格式）后执行一些转换，但我们建议在应用聊天模板之前执行转换，以确保其一致工作。

为简单起见，下面的一些示例不遵循此建议并使用标准格式。但是，转换可以直接应用于会话格式而无需修改。|从 \ 到 |语言建模|快速完成 |仅提示 |带有隐式提示的偏好 |偏好 |未配对的偏好 |分级监管 |
| --- | --- | --- | --- | --- | --- | --- | --- |
|语言建模|不适用 |不适用 |不适用 |不适用 |不适用 |不适用 |不适用 |
|快速完成 | [🔗](#from-prompt-completion-to-language-modeling-dataset) |不适用 | [🔗](#from-prompt-completion-to-prompt-only-dataset) |不适用 |不适用 |不适用 |不适用 |
|仅提示 |不适用 |不适用 |不适用 |不适用 |不适用 |不适用 |不适用 |
|带有隐式提示的偏好 | [🔗](#from-preference-with-implicit-prompt-to-language-modeling-dataset) | [🔗](#from-preference-with-implicit-prompt-to-prompt-completion-dataset) | [🔗](#from-preference-with-implicit-prompt-to-prompt-only-dataset) |不适用 | [🔗](#from-implicit-to-explicit-prompt-preference-dataset) | [🔗](#from-preference-with-implicit-prompt-to-unpaired-preference-dataset) |不适用 |
|偏好 | [🔗](#from-preference-to-language-modeling-dataset) | [🔗](#from-preference-to-prompt-completion-dataset) | [🔗](#from-preference-to-prompt-only-dataset) | [🔗](#from-explicit-to-implicit-prompt-preference-dataset) |不适用 | [🔗](#from-preference-to-unpaired-preference-dataset) |不适用 |
|未配对的偏好 | [🔗](#from-unpaired-preference-to-language-modeling-dataset) | [🔗](#from-unpaired-preference-to-prompt-completion-dataset) | [🔗](#from-unpaired-preference-to-prompt-only-dataset) |不适用 |不适用 |不适用 |不适用 |
|分级监管 | [🔗](#from-stepwise-supervision-to-language-modeling-dataset) | [🔗](#from-stepwise-supervision-to-prompt-completion-dataset) | [🔗](#from-stepwise-supervision-to-prompt-only-dataset) |不适用 |不适用 | [🔗](#from-stepwise-supervision-to-unpaired-preference-dataset) |不适用 |

### 从提示完成到语言建模数据集

要将提示完成数据集转换为语言建模数据集，请连接提示和完成。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

def concat_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.map(concat_prompt_completion, remove_columns=["prompt", "completion"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### 从提示完成到仅提示数据集

要将提示完成数据集转换为仅提示数据集，请删除完成。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "completion": [" blue.", " in the sky."],
})

dataset = dataset.remove_columns("completion")
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### 从带有隐式提示的偏好到语言建模数据集要将具有隐式提示数据集的首选项转换为语言建模数据集，请删除拒绝的数据集，并将列 `"chosen"` 重命名为 `"text"`。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "chosen": ["The sky is blue.", "The sun is in the sky."],
    "rejected": ["The sky is green.", "The sun is in the sea."],
})

dataset = dataset.rename_column("chosen", "text").remove_columns("rejected")
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### 从带有隐式提示的偏好到提示完成数据集

要将带有隐式提示的首选项数据集转换为提示完成数据集，请提取带有 [extract_prompt()](/docs/trl/v1.9.2/en/data_utils#trl.extract_prompt) 的提示，删除拒绝的，并将列 `"chosen"` 重命名为 `"completion"`。

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns("rejected").rename_column("chosen", "completion")
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}], 'completion': [{'role': 'assistant', 'content': 'It is blue.'}]}
```

### 从带有隐式提示的偏好到仅提示的数据集

要将带有隐式提示的偏好数据集转换为仅提示数据集，请使用 [extract_prompt()](/docs/trl/v1.9.2/en/data_utils#trl.extract_prompt) 提取提示，并删除拒绝的和选择的。

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})
dataset = dataset.map(extract_prompt).remove_columns(["chosen", "rejected"])
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}]}
```

### 从隐式到显式的提示偏好数据集

要将具有隐式提示的偏好数据集转换为具有显式提示的偏好数据集，请使用 [extract_prompt()](/docs/trl/v1.9.2/en/data_utils#trl.extract_prompt) 提取提示。

```python
from datasets import Dataset
from trl import extract_prompt

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'chosen': [{'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'assistant', 'content': 'It is green.'}]}
```

### 从带有隐式提示的偏好到不配对的偏好数据集

要将带有隐式提示的偏好数据集转换为未配对的偏好数据集，请使用 [extract_prompt()](/docs/trl/v1.9.2/en/data_utils#trl.extract_prompt) 提取提示，并使用 [unpair_preference_dataset()](/docs/trl/v1.9.2/en/data_utils#trl.unpair_preference_dataset) 取消数据集配对。

```python
from datasets import Dataset
from trl import extract_prompt, unpair_preference_dataset

dataset = Dataset.from_dict({
    "chosen": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is blue."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "user", "content": "What color is the sky?"}, {"role": "assistant", "content": "It is green."}],
        [{"role": "user", "content": "Where is the sun?"}, {"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = dataset.map(extract_prompt)
dataset = unpair_preference_dataset(dataset)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}
```> [!警告]
> 请记住，偏好数据集中的 `"chosen"` 和 `"rejected"` 完成情况可能是好是坏。
> 在申请[unpair_preference_dataset()](/docs/trl/v1.9.2/en/data_utils#trl.unpair_preference_dataset)之前，请确保所有`"chosen"`完成情况都可以标记为良好，所有`"rejected"`完成情况可以标记为不好。
> 这可以通过检查每个完成的绝对评级来确保，例如来自奖励模型。

### 从偏好到语言建模数据集

要将偏好数据集转换为语言建模数据集，请删除拒绝的数据集，将提示和选择的数据集连接到 `"text"` 列中。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

def concat_prompt_chosen(example):
    return {"text": example["prompt"] + example["chosen"]}

dataset = dataset.map(concat_prompt_chosen, remove_columns=["prompt", "chosen", "rejected"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### 从偏好到提示完成数据集

要将首选项数据集转换为提示完成数据集，请删除拒绝的数据集，并将列 `"chosen"` 重命名为 `"completion"`。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns("rejected").rename_column("chosen", "completion")
```

```python
>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}
```

### 从偏好到仅提示数据集

要将首选项数据集转换为仅提示数据集，请删除拒绝的数据集和选择的数据集。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is"],
    "chosen": [" blue.", " in the sky."],
    "rejected": [" green.", " in the sea."],
})

dataset = dataset.remove_columns(["chosen", "rejected"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### 从显式到隐式提示偏好数据集

要将具有显式提示的首选项数据集转换为具有隐式提示的首选项数据集，请将提示连接到已选择和已拒绝的提示，然后删除提示。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

def concat_prompt_to_completions(example):
    return {"chosen": example["prompt"] + example["chosen"], "rejected": example["prompt"] + example["rejected"]}

dataset = dataset.map(concat_prompt_to_completions, remove_columns="prompt")
```

```python
>>> dataset[0]
{'chosen': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is blue.'}],
 'rejected': [{'role': 'user', 'content': 'What color is the sky?'}, {'role': 'assistant', 'content': 'It is green.'}]}
```

### 从偏好到不配对的偏好数据集要将数据集转换为未配对的偏好数据集，请取消数据集与 [unpair_preference_dataset()](/docs/trl/v1.9.2/en/data_utils#trl.unpair_preference_dataset) 的配对。

```python
from datasets import Dataset
from trl import unpair_preference_dataset

dataset = Dataset.from_dict({
    "prompt": [
        [{"role": "user", "content": "What color is the sky?"}],
        [{"role": "user", "content": "Where is the sun?"}],
    ],
    "chosen": [
        [{"role": "assistant", "content": "It is blue."}],
        [{"role": "assistant", "content": "In the sky."}],
    ],
    "rejected": [
        [{"role": "assistant", "content": "It is green."}],
        [{"role": "assistant", "content": "In the sea."}],
    ],
})

dataset = unpair_preference_dataset(dataset)
```

```python
>>> dataset[0]
{'prompt': [{'role': 'user', 'content': 'What color is the sky?'}],
 'completion': [{'role': 'assistant', 'content': 'It is blue.'}],
 'label': True}
```

> [!警告]
> 请记住，偏好数据集中的 `"chosen"` 和 `"rejected"` 完成度可能好也可能坏。
> 在申请[unpair_preference_dataset()](/docs/trl/v1.9.2/en/data_utils#trl.unpair_preference_dataset)之前，请确保所有`"chosen"`完成情况都可以标记为良好，所有`"rejected"`完成情况可以标记为不好。
> 这可以通过检查每个完成的绝对评级来确保，例如来自奖励模型。

### 从不成对的偏好到语言建模数据集

要将不成对的偏好数据集转换为语言建模数据集，请将具有良好完成的提示连接到 `"text"` 列，并删除提示、完成和标签列。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

def concatenate_prompt_completion(example):
    return {"text": example["prompt"] + example["completion"]}

dataset = dataset.filter(lambda x: x["label"]).map(concatenate_prompt_completion).remove_columns(["prompt", "completion", "label"])
```

```python
>>> dataset[0]
{'text': 'The sky is blue.'}
```

### 从不配对偏好到提示完成数据集

要将未配对的首选项数据集转换为提示完成数据集，请筛选好的标签，然后删除标签列。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.filter(lambda x: x["label"]).remove_columns(["label"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is', 'completion': ' blue.'}
```

### 从不配对的偏好到仅提示的数据集

要将未配对的首选项数据集转换为仅提示数据集，请删除完成列和标签列。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["The sky is", "The sun is", "The sky is", "The sun is"],
    "completion": [" blue.", " in the sky.", " green.", " in the sea."],
    "label": [True, True, False, False],
})

dataset = dataset.remove_columns(["completion", "label"])
```

```python
>>> dataset[0]
{'prompt': 'The sky is'}
```

### 从逐步监督到语言建模数据集要将逐步监督数据集转换为语言建模数据集，请将具有良好完成度的提示连接到 `"text"` 列中。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def concatenate_prompt_completions(example):
    completion = "".join(example["completions"])
    return {"text": example["prompt"] + completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(concatenate_prompt_completions, remove_columns=["prompt", "completions", "labels"])
```

```python
>>> dataset[0]
{'text': 'Blue light scatters more in the atmosphere, so the sky is green.'}
```

### 从逐步监督到提示完成数据集

要将逐步监督数据集转换为提示完成数据集，请加入好的完成数据并删除标签。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def join_completions(example):
    completion = "".join(example["completions"])
    return {"completion": completion}

dataset = dataset.filter(lambda x: all(x["labels"])).map(join_completions, remove_columns=["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.'}
```

### 从逐步监督到仅提示数据集

要将逐步监督数据集转换为仅提示数据集，请删除补全和标签。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

dataset = dataset.remove_columns(["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light'}
```

### 从逐步监督到不配对偏好数据集

要将逐步监督数据集转换为不配对的偏好数据集，请加入补全并合并标签。

合并标签的方法取决于具体任务。在此示例中，我们使用逻辑 AND 运算。这意味着如果步骤标签指示各个步骤的正确性，则生成的标签将反映整个序列的正确性。

```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "prompt": ["Blue light", "Water"],
    "completions": [[" scatters more in the atmosphere,", " so the sky is green."],
                   [" forms a less dense structure in ice,", " which causes it to expand when it freezes."]],
    "labels": [[True, False], [True, True]],
})

def merge_completions_and_labels(example):
    return {"prompt": example["prompt"], "completion": "".join(example["completions"]), "label": all(example["labels"])}

dataset = dataset.map(merge_completions_and_labels, remove_columns=["completions", "labels"])
```

```python
>>> dataset[0]
{'prompt': 'Blue light', 'completion': ' scatters more in the atmosphere, so the sky is green.', 'label': False}
```

## 视觉数据集一些训练器还支持使用图像文本对微调视觉语言模型（VLM）。在这种情况下，建议使用会话格式，因为每个模型以不同的方式处理文本中的图像占位符。

会话视觉数据集与标准会话数据集在两个关键方面有所不同：

1. 数据集必须包含带有图像数据（作为 PIL 图像列表）的密钥 `images` 或带有单个 PIL 图像的 `image`。
2. 消息中的`"content"`字段必须是字典列表，其中每个字典指定数据类型：`"image"`或`"text"`。

示例：

```python
# Textual dataset:
"content": "What color is the sky?"

# Vision dataset:
"content": [
    {"type": "image"}, 
    {"type": "text", "text": "What color is the sky in the image?"}
]
```

对话视觉数据集的一个例子是[openbmb/RLAIF-V-Dataset](https://huggingface.co/datasets/openbmb/RLAIF-V-Dataset)。下面是数据集训练数据的嵌入视图，允许您直接探索它：

<iframe
  src="https://huggingface.co/datasets/trl-lib/rlaif-v/embed/viewer/default/train"
  frameborder="0"
  width="100%"
  height="560px"
>

> [!注意]
> 可以在数据集中混合纯文本和视觉语言数据，但需要 `transformers` 4.57.0 或更高版本。示例：
>
> ```python
> dataset = Dataset.from_dict({
>     "prompt": [
>         [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What color is the sky in the image?"}]}],
>         [{"role": "user", "content": [{"type": "text", "text": "What is the capital of France?"}]}],
>     ],
>     "completion": [
>         [{"role": "assistant", "content": [{"type": "text", "text": "It is blue."}]}],
>         [{"role": "assistant", "content": [{"type": "text", "text": "Paris."}]}],
>     ],
>     "images": [
>         [PIL.Image.open("path/to/sky_image1.png")],
>         [],
>     ],
> })
> ```

### GSPO 代币
https://huggingface.co/docs/trl/v1.9.2/gspo_token.md