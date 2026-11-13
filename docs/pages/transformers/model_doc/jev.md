# JEV

> [!WARNING]
> Decision models, the `decision` pipeline and `DecisionConfig` are experimental: their API may change at any time, without deprecation.

JEV is a decision head that scores single-token labels with a linear layer at the last token of the prompt, soft-capped like the logits of a language model head. It was introduced by [JEV](https://huggingface.co/autotrust/JEV-9B) and is also used by [GEV](https://huggingface.co/autotrust/GEV-26B-Decide).

[JevForDecision](/docs/transformers/v5.23.0/en/model_doc/jev#transformers.JevForDecision) is the model of the checkpoint and its head, like a `ForCausalLM` is the model and its `lm_head`. It
reads the files that the checkpoint ships its head with, next to its weights (`readout.safetensors` and
`decision_config.json` for vLLM decision models, `head.safetensors` and `judge_config.json` for JEV).

```python
from transformers import pipeline

decide = pipeline("decision", model="autotrust/GEV-26B-Decide")
decide(
    {
        "state": "I was charged twice for my order and want a refund.",
        "questions": {"refund": {"type": "noul", "instructions": "Is a refund requested?"}},
    }
)
```

See the [decision pipeline](../main_classes/pipelines#transformers.DecisionPipeline) for image assets.

Without the pipeline, decision models are used like `generate`: the decision config renders a prompt per question (or
one for all of them) with the chat template, `decide` scores the options of each question, and `Question.answer` turns
the scores into answers:

> [!IMPORTANT]
> `from_pretrained` does not load the LoRA of checkpoints that keep it in a subfolder (`decision_config.adapter`, e.g.
> `adapter/` for JEV). The pipeline loads it for you; without the pipeline, call `load_adapter` yourself as below, or
> the model answers with the base weights.

```python
import torch
from transformers import AutoTokenizer, JevForDecision

repo = "autotrust/JEV-9B"
model = JevForDecision.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")
tokenizer, config = AutoTokenizer.from_pretrained(repo), model.decision_config  # read from the checkpoint's files
if config.adapter:  # JEV keeps its LoRA in a subfolder: `from_pretrained` doesn't load it, the pipeline does
    model.load_adapter(repo, adapter_kwargs={"subfolder": config.adapter})

request = {
    "state": "I was charged twice for my order and I want my money back today.",
    "questions": {
        "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
        "urgency": {"type": "score", "criteria": ["Can wait", "This week", "Today", "Now"]},
    },
}
answers = {}
for inputs, questions in config.prepare_inputs(request, tokenizer):
    scores = model.decide(**inputs, decision_questions=questions)
    answers |= {q.name: q.answer(s.softmax(-1).tolist()) for q, s in zip(questions, scores)}
```

The same head reads the 255 answer codes of [pplx-decider](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b)
and [d3](https://huggingface.co/vllm-sr/d3), whose Qwen3.5 backbone attends to the whole prompt
(`is_causal=False` in the `text_config` of their `JevForDecision`). Their head (`readout.safetensors`) and settings
(`decision_config.json`) are read as they are, next to their weights:

```python
pipeline("decision", model="vllm-sr/d3")(request)  # or perplexity-ai/pplx-decider-v1.1-27b
```

## JevConfig[[transformers.JevConfig]]

#### transformers.JevConfig[[transformers.JevConfig]]

```python
transformers.JevConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 4096, num_verbalizers: int = 24, offsets: dict[str, int] | None = None, softcap: float | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/jev/configuration_jev.py#L101)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `4096`) : Dimension of the hidden representations.

num_verbalizers (`int`, *optional*, defaults to 24) : Number of single-token labels scored by the head, for all question types.

offsets (`dict[str, int]`, *optional*, defaults to `{"noul" : 0, "score": 2, "choice": 8}`): Row of the first label of each question type: the options of a question are scored by the rows from `offsets[question.type]` on, in order.

softcap (`float`, *optional*) : Soft cap of the scores, `softcap * tanh(scores / softcap)`, like the logits of a language model head.

text_config (`dict` or `PreTrainedConfig`, *optional*) : Config of the model the head reads, the `config.json` of its repo.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a JevModel. It is used to instantiate a Jev
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [autotrust/JEV-9B](https://huggingface.co/autotrust/JEV-9B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:
```python
>>> from transformers import JevConfig, JevForDecision

>>> model = JevForDecision(JevConfig(hidden_size=4096))
```

#### decision_config[[transformers.JevConfig.decision_config]]

```python
decision_config(head_configs: dict, text_config: dict)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/jev/configuration_jev.py#L134)

The decision config of a checkpoint that ships a JEV head next to its weights, from its config files.

## JevForDecision[[transformers.JevForDecision]]

#### transformers.JevForDecision[[transformers.JevForDecision]]

```python
transformers.JevForDecision(config: JevConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/jev/modeling_jev.py#L38)

**Parameters:**

config ([JevConfig](/docs/transformers/v5.23.0/en/model_doc/jev#transformers.JevConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

JEV (autotrust/JEV-9B): the model of its repo and a decision head, linear scores of single-token labels at the token
before the `<|mark|>` of each question, soft-capped like the logits of a language model head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.JevForDecision.forward]]

```python
forward(input_ids: LongTensor, marks: LongTensor, num_options: LongTensor, question_type_ids: LongTensor, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/jev/modeling_jev.py#L45)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

marks (`torch.LongTensor` of shape `(num_marks,)`) : Positions recorded by the `<|mark|>` of a decision template: the number of tokens of the prompt before each of them.

num_options (`torch.LongTensor` of shape `(num_questions,)`) : Number of options of each question of the prompt.

question_type_ids (`torch.LongTensor` of shape `(num_questions,)`) : Type of each question of the prompt, as its index in `Question.questions_types`.

**Returns:**

`torch.Tensor`

The [JevForDecision](/docs/transformers/v5.23.0/en/model_doc/jev#transformers.JevForDecision) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

### OWLv2
https://huggingface.co/docs/transformers/v5.23.0/model_doc/owlv2.md
