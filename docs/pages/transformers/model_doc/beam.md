# Beam

## Overview

[Beam](https://reflection.ai/blog/introducing-beam) is Reflection's first open-weight model, a sparse mixture-of-experts (MoE) language model with 501B total parameters and 23B active parameters per token, built for coding, reasoning, and agentic workloads. It was pretrained on 23.8T tokens and post-trained with large-scale asynchronous reinforcement learning. Beam is text-only, and it reasons before every answer. A reasoning effort setting in the chat template controls how much.

Beam has 52 decoder layers and combines the following features:

- **Hybrid sliding and full attention.** Every sixth layer and the last layer use full attention, and the others attend to the most recent `sliding_window` (512) tokens. `layer_types` sets the pattern.
- **RoPE on sliding layers, NoPE on full layers.** `rope_parameters` is keyed by layer type, and `None` marks a layer type without positional encoding. The full-attention layers scale queries instead (scalable softmax): the query at 1-based position `n` is multiplied by `(1 + n / 2048) ** 0.1`, which keeps attention from flattening as the context grows.
- **Gated attention.** Grouped-query attention with 48 query heads and 8 key/value heads, and RMSNorm on each query and key head. A sigmoid gate computed from the attention input, with a learned bias per head, scales the attention output elementwise before the output projection.
- **Sparse MoE.** The first 3 layers use a dense SwiGLU MLP, and the remaining 49 layers route each token to 8 of 256 experts, with no shared expert (`mlp_layer_types` sets the pattern). The router scores experts with a sigmoid, balances load with a learned per-expert bias instead of an auxiliary loss, and restricts each token to the best `topk_group` of `n_group` expert groups. The routing weights are normalized, and the MoE output is scaled by `routed_scaling_factor`.
- **Sandwich norms with depth-scaled residuals.** Each attention and MLP block is wrapped in an RMSNorm before and after. Residual additions are rescaled so that, after `k` contributions (counting the embeddings as the first), the residual stream is their sum divided by `sqrt(k)`.
- **Final logit softcapping.** Logits are capped with `30 * tanh(logits / 30)` (`final_logit_softcapping`).

The implementation computes RMSNorm scaling, the SwiGLU activation, RoPE, query scaling, routing, and softcapping in FP32 to match the numerics Beam was trained and served with. The router weights and the expert bias are kept in FP32 at load time, and the returned `logits` are always FP32.

You can find all the original Beam checkpoints under the [Reflection](https://huggingface.co/reflection) organization.

This model was contributed by [yonigozlan](https://huggingface.co/yonigozlan).

## Usage examples

Beam's chat template formats conversations in Beam's Harmony format, and its [response template](../chat_response_parsing) parses the generated text back into a message with `thinking`, `content`, and `tool_calls` fields. The examples below generate a reply with [Pipeline](/docs/transformers/v5.21.0/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModelForCausalLM](/docs/transformers/v5.21.0/en/model_doc/auto#transformers.AutoModelForCausalLM) class.

The BF16 checkpoint is about 1 TB, so the examples shard it across 8 devices, placing whole routed experts on each device with [expert parallelism](../expert_parallelism). Launch them with `torchrun --nproc-per-node 8 <script.py>`.

```python
from transformers import DistributedConfig, pipeline

pipe = pipeline(
    task="text-generation",
    model="reflection/Beam-1.0-501B-A23B",
    model_kwargs={"distributed_config": DistributedConfig(tp_size=8, ep_size=8)},
)

messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
output = pipe(messages, max_new_tokens=2048)
reply = output[0]["generated_text"][-1]
print(reply["thinking"])
print(reply["content"])
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, DistributedConfig

model_id = "reflection/Beam-1.0-501B-A23B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, distributed_config=DistributedConfig(tp_size=8, ep_size=8))

messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True, reasoning_effort="low"
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=2048)
generated = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :])
message = tokenizer.parse_response(generated, prefix=inputs["input_ids"][0])
print(message["thinking"])
print(message["content"])
```

Prefer this over `device_map="auto"`, which places whole layers on each device, runs them one device at a time, and can run out of memory while loading.

Generation samples with Beam's recommended `temperature=0.7` and `top_p=0.99`, which the checkpoint's [GenerationConfig](/docs/transformers/v5.21.0/en/main_classes/text_generation#transformers.GenerationConfig) sets as defaults. Pass `do_sample=False` to decode greedily.

### Reasoning effort

Pass `reasoning_effort` to [apply_chat_template()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) to control how much Beam reasons. It accepts an integer budget or one of the labels below. Beam was not trained with efforts below 8, so smaller values aren't respected.

| Label | `minimal` | `low` | `medium` | `high` | `xhigh` | `max` |
|---|---|---|---|---|---|---|
| Effort | 8 | 32 | 64 | 128 | 256 | 512 |

When `reasoning_effort` is unset, the template uses `max`. If `max_new_tokens` runs out during reasoning, the parsed message has `thinking` but no `content`.

### Multi-turn conversations

Append the parsed message to the history as is. Beam's output degrades without its earlier reasoning, which the chat template reads from `thinking` (`reasoning_content` and `reasoning` are also accepted). By default, an assistant turn without reasoning renders with an empty reasoning block. Pass `allow_missing_thinking=False` to raise an error instead when `add_generation_prompt=True`, which is recommended when serving Beam.

```python
messages.append(message)
messages.append({"role": "user", "content": "And why are sunsets red?"})
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
```

### Tool calling

Pass `tools` to both [apply_chat_template()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.apply_chat_template) and [parse_response()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.parse_response). With `tools`, parsed arguments are cast to the types in each tool's schema; without it, every argument is a string.

```python
def get_weather(city: str):
    """
    Get the current weather in a city.

    Args:
        city: The city name.
    """
    return "Sunny, 22 C"

messages = [{"role": "user", "content": "What's the weather like in Paris right now?"}]
inputs = tokenizer.apply_chat_template(
    messages, tools=[get_weather], add_generation_prompt=True, return_tensors="pt", return_dict=True
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=2048)
generated = tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :])
message = tokenizer.parse_response(generated, prefix=inputs["input_ids"][0], tools=[get_weather])
print(message["tool_calls"])
# [{'type': 'function', 'function': {'name': 'get_weather', 'arguments': {'city': 'Paris'}}}]

messages.append(message)
for call in message["tool_calls"]:
    result = get_weather(**call["function"]["arguments"])
    messages.append({"role": "tool", "name": call["function"]["name"], "content": result})
```

Then generate again from `messages` to get the final answer. Parsed tool calls have no `id`, and the format matches results to calls by position, so return tool results in the order of the calls. Generation stops after a final answer (`<|return|>`) or a tool call (`<|call|>`); `generation_config.json` lists both.

For streaming, [get_response_parser()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.get_response_parser) returns a parser that you `feed()` text chunks to as they are generated. See [Response parsing](../chat_response_parsing) for more details.

## Notes

- **Template arguments.** Besides `reasoning_effort`, the chat template accepts `model_identity` (replaces the first line of the system message), `builtin_tools`, and `allow_missing_thinking`. `builtin_tools` should only name `search` and `browse` tools, and only with the definitions Beam was trained with. Other tools only need `tools`. `system` and `developer` messages are only rendered at the start of the conversation.
- **Training.** The chat template marks the assistant output for `apply_chat_template(..., return_dict=True, return_assistant_tokens_mask=True)`, so TRL's `SFTConfig(assistant_only_loss=True)` trains on assistant tokens only. The mask covers each assistant turn's reasoning, answer, and tool calls.
- **No auxiliary loss.** The router balances expert load through its bias, so `BeamForCausalLM` computes no load-balancing loss. `output_router_logits=True` still returns the router logits.

## BeamConfig[[transformers.BeamConfig]]

#### transformers.BeamConfig[[transformers.BeamConfig]]

```python
transformers.BeamConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 151936, hidden_size: int = 5120, intermediate_size: int = 20480, num_hidden_layers: int = 52, num_attention_heads: int = 48, num_key_value_heads: int = 8, hidden_act: str = 'silu', max_position_embeddings: int = 8192, initializer_range: float = 0.02, rms_norm_eps: float = 1e-06, use_cache: bool = True, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, sliding_window: int | None = 512, attention_dropout: float | int = 0.0, moe_intermediate_size: int = 2560, num_experts_per_tok: int = 8, num_experts: int = 256, norm_topk_prob: bool = True, output_router_logits: bool = False, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None, head_dim: int = 128, layer_types: list[str] | None = None, mlp_layer_types: list[str] | None = None, n_group: int | None = 8, topk_group: int | None = 4, routed_scaling_factor: int | float = 2.5, final_logit_softcapping: int | float | None = 30.0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/models/beam/configuration_beam.py#L29)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `151936`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `5120`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `20480`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `52`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `48`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `8192`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

sliding_window (`int`, *optional*, defaults to `512`) : Sliding window attention window size. If `None`, no sliding window is applied.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

moe_intermediate_size (`int`, *optional*, defaults to `2560`) : Intermediate size of the routed expert MLPs.

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

num_experts (`int`, *optional*, defaults to `256`) : Number of routed experts in MoE layers. 

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*) : Token id used for end-of-stream in the vocabulary.

head_dim (`int`, *optional*, defaults to `128`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

mlp_layer_types (`list[str]`, *optional*) : Per-layer MLP type, `"dense"` for a SwiGLU MLP or `"sparse"` for the MoE block. Length must equal `num_hidden_layers`. Defaults to three dense layers followed by sparse ones.

n_group (`int`, *optional*, defaults to 8) : Number of groups for routed experts.

topk_group (`int`, *optional*, defaults to `4`) : Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).

routed_scaling_factor (`Union[int, float]`, *optional*, defaults to `2.5`) : Scaling factor or routed experts.

final_logit_softcapping (`Union[int, float]`, *optional*, defaults to `30.0`) : Soft-capping value applied to the final logits before computing the probability distribution. Logits are scaled by `tanh(logit / cap) * cap`.

This is the configuration class to store the configuration of a BeamModel. It is used to instantiate a Beam
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [reflection/Beam-1.0-501B-A23B](https://huggingface.co/reflection/Beam-1.0-501B-A23B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import BeamConfig

>>> # Initializing a Beam configuration
>>> configuration = BeamConfig()
```

## BeamModel[[transformers.BeamModel]]

#### transformers.BeamModel[[transformers.BeamModel]]

```python
transformers.BeamModel(config: BeamConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/models/beam/modeling_beam.py#L498)

**Parameters:**

config ([BeamConfig](/docs/transformers/v5.21.0/en/model_doc/beam#transformers.BeamConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Beam Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.BeamModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/models/beam/modeling_beam.py#L515)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BeamConfig](/docs/transformers/v5.21.0/en/model_doc/beam#transformers.BeamConfig)) and inputs.

The [BeamModel](/docs/transformers/v5.21.0/en/model_doc/beam#transformers.BeamModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.

## BeamForCausalLM[[transformers.BeamForCausalLM]]

#### transformers.BeamForCausalLM[[transformers.BeamForCausalLM]]

```python
transformers.BeamForCausalLM(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/models/beam/modeling_beam.py#L587)

**Parameters:**

config ([BeamForCausalLM](/docs/transformers/v5.21.0/en/model_doc/beam#transformers.BeamForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Beam Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.BeamForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.0/src/transformers/models/beam/modeling_beam.py#L602)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([BeamConfig](/docs/transformers/v5.21.0/en/model_doc/beam#transformers.BeamConfig)) and inputs.

The [BeamForCausalLM](/docs/transformers/v5.21.0/en/model_doc/beam#transformers.BeamForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **aux_loss** (`torch.FloatTensor`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`, and the model trains its router with a load-balancing loss) -- Load-balancing auxiliary loss for the sparse modules. Models that balance their experts with a router bias
  instead (DeepSeek-V3 and the architectures derived from it) return `None` here while still returning
  `router_logits`.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`) -- Tuple of `torch.FloatTensor` (one for each sparse layer) of shape `(batch_size * sequence_length, num_experts)`.

  Raw router logits computed by the MoE routers. They can be used to compute a load-balancing loss or to
  inspect how tokens are dispatched to experts.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
```

### TextNet
https://huggingface.co/docs/transformers/v5.21.0/model_doc/textnet.md
