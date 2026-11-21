# Inkling 2 Mini

[Inkling 2 Mini](https://huggingface.co/collections/thinkingmachines/inkling-2) is a multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that accepts text and image inputs and generates text outputs. It is based on [Inkling 2 Small](./inkling2) but has only 24 decoder layers instead of 48. The lower layer count is compensated for by a looping mechanism in which layers 6 to 17 run three times. Information is shared between loop passes through a learned message-passing mechanism that mixes the residuals of the previous pass into the attention and mixture-of-experts inputs of each loop layer. The decoder has a hidden size of 2304 and routes each token to 8 of 128 experts. The image encoder, tokenizer and processor are identical to those of Inkling 2 Small.

Inkling 2 Mini comes with an additional [research checkpoint](https://huggingface.co/thinkingmachines/TML-d24-varloop-research) that supports a variable number of loop passes. The architecture is identical to the main Inkling 2 Mini model. The number of loop passes can be adjusted to trade off speed against quality. Fewer passes are faster, while more passes improve quality.

Both checkpoints contain NVFP4 and MXFP8 quantized layers that Transformers dequantizes when loading the model.

**Checkpoints**

- [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
- [Inkling 2 Mini varloop research](https://huggingface.co/thinkingmachines/TML-d24-varloop-research)

## Resources

- [Thinking Machines Blog](https://thinkingmachines.ai/news/inkling-2-mini/)
- [Hugging Face Blog](https://huggingface.co/blog/inkling2mini)

```python
import torch

from transformers import pipeline

pipe = pipeline(
    "image-text-to-text", model="thinkingmachines/Inkling-2-Mini", dtype=torch.bfloat16, device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "What animal is in this image, and where is it?"},
        ],
    },
]

output = pipe(messages, max_new_tokens=1024, return_full_text=False, skip_special_tokens=False)
print(output[0]["generated_text"])
# <|message_model|><|content_thinking|>The user is asking about an animal in an image. Let me
# analyze what I see: a small wild feline with thick, fluffy fur dusted with snow, short legs, a
# broad face, and black stripes/spots.
# ...
# <|end_message|><|message_model|><|content_text|>The animal appears to be a Pallas’s cat (manul) —
# a small, bear-like wild cat with thick speckled/striped fur. It’s standing on snow inside what
# looks like a zoo or wildlife enclosure, with snow-covered birch/log branches and a chain-link
# fence in the background.<|end_message|><|content_model_end_sampling|>
```

```python
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Mini"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "What animal is in this image, and where is it?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
# <|message_model|><|content_thinking|>The user is asking what animal is in the image and where it
# is. Let me analyze the image.
# ...
# <|end_message|><|message_model|><|content_text|>The animal appears to be a Pallas’s cat (manul), a
# stocky small wild cat with thick, snow-dusted fur. It is standing on snow, likely inside a zoo or
# wildlife enclosure — you can see birch/aspen-like posts and chain-link fencing in the
# background.<|end_message|><|content_model_end_sampling|>
```

```python
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/TML-d24-varloop-research"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
    text_config={"loop_num_passes": 3},  # number of loop passes, must be at least 1
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
            },
            {"type": "text", "text": "What animal is in this image, and where is it?"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)
input_len = inputs["input_ids"].shape[-1]

outputs = model.generate(**inputs, max_new_tokens=1024)
print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
# <|message_model|><|content_thinking|>We need answer image question. Need identify animal and where
# is it.
# ...
# <|end_message|><|message_model|><|content_text|>The animal is a Pallas’s cat (a.k.a. Manul), a
# stocky, densely furred small wild cat. It’s in the snow outdoors in a cold setting—likely a winter
# enclosure/habitat—standing/walking on snow at the center of the
# image.<|end_message|><|content_model_end_sampling|>
```

## Notes

- Attention runs in eager mode only because every layer adds a relative position bias and a sink.
- The short convolutions use fast GPU kernels when available: Hub kernels when loading with `use_kernels=True`,
  otherwise the [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) package if installed
  (`pip install causal-conv1d`). Without them, they fall back to a slower PyTorch implementation.
- Every pass through the loop layers keeps its own keys, values and convolution states, so the cache has one layer per layer visit (48 for the released checkpoint).
- `output_hidden_states` and `output_attentions` return one entry per layer visit.

## Inkling2LoopConfig[[transformers.Inkling2LoopConfig]]

#### transformers.Inkling2LoopConfig[[transformers.Inkling2LoopConfig]]

```python
transformers.Inkling2LoopConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: transformers.models.inkling2_loop.configuration_inkling2_loop.Inkling2LoopTextConfig | dict | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 200054)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/configuration_inkling2_loop.py#L181)

**Parameters:**

text_config (`Union[~models.inkling2_loop.configuration_inkling2_loop.Inkling2LoopTextConfig, dict]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to 200054) : Token index of the image placeholders, each replaced by one image feature.

This is the configuration class to store the configuration of a Inkling2LoopModel. It is used to instantiate a Inkling2 Loop
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Inkling2LoopTextConfig[[transformers.Inkling2LoopTextConfig]]

#### transformers.Inkling2LoopTextConfig[[transformers.Inkling2LoopTextConfig]]

```python
transformers.Inkling2LoopTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 201024, unpadded_vocab_size: int = 200058, hidden_size: int = 2304, num_hidden_layers: int = 24, num_attention_heads: int = 24, num_key_value_heads: int = 8, head_dim: int = 128, swa_num_attention_heads: int = 24, swa_num_key_value_heads: int = 8, swa_head_dim: int = 128, sliding_window_size: int = 1024, d_rel: int = 16, rel_extent: int = 1024, log_scaling_n_floor: int = 8192, log_scaling_alpha: float = 0.1, layer_types: list[str] | None = None, max_position_embeddings: int = 1048576, rms_norm_eps: float = 1e-06, conv_kernel_size: int = 4, hidden_act: str = 'silu', moe_intermediate_size: int = 1152, n_routed_experts: int = 128, num_experts_per_tok: int = 8, route_scale: float = 8.0, attention_dropout: float = 0.0, initializer_range: float = 0.02, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | None = 200006, qk_l2_clip_threshold: float = 22.627416997969522, rel_l2_clip_threshold: float = 8.0, n_layers_per_attn_res_block: int = 6, mlp_gate_softcap: float = 4.0, mlp_up_softcap: float = 25.0, final_logit_softcapping: float = 30.0, scoring_func: str = 'sigmoid', use_cache: bool = True, loop_layer_index_start: int = 6, loop_layer_index_end: int = 18, loop_num_passes: int = 3, loop_gate_rank: int = 64)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/configuration_inkling2_loop.py#L30)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `201024`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

unpadded_vocab_size (`int`, *optional*, defaults to 200058) : Output size of the language modeling head. The checkpoint unembedding is padded to `vocab_size`, the padded rows are dropped on load and restored as zeros on save.

hidden_size (`int`, *optional*, defaults to `2304`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `24`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `24`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `128`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

swa_num_attention_heads (`int`, *optional*, defaults to 24) : Number of attention heads in the sliding-window layers.

swa_num_key_value_heads (`int`, *optional*, defaults to 8) : Number of key/value heads in the sliding-window layers.

swa_head_dim (`int`, *optional*, defaults to 128) : Dimension of query and key heads in the sliding-window layers.

sliding_window_size (`int`, *optional*, defaults to 1024) : Size of the sliding attention window used by layers whose `layer_types` entry is `"hybrid_sliding"`.

d_rel (`int`, *optional*, defaults to 16) : Per-head dimension of the relative states that are mixed into the relative position bias.

rel_extent (`int`, *optional*, defaults to 1024) : Backward distance, in tokens, over which the relative position bias is applied in the full-attention layers.

log_scaling_n_floor (`int`, *optional*, defaults to 8192) : Position from which queries and relative states are scaled up logarithmically in the full-attention layers.

log_scaling_alpha (`float`, *optional*, defaults to 0.1) : Strength of the logarithmic scaling controlled by `log_scaling_n_floor`.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

max_position_embeddings (`int`, *optional*, defaults to `1048576`) : The maximum sequence length that this model might ever be used with.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

conv_kernel_size (`int`, *optional*, defaults to `4`) : The size of the convolutional kernel.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

moe_intermediate_size (`int`, *optional*, defaults to `1152`) : Intermediate size of the routed expert MLPs.

n_routed_experts (`int`, *optional*, defaults to `128`) : Number of routed experts.

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

route_scale (`float`, *optional*, defaults to `8.0`) : Scaling factor or routed experts.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`int`, *optional*, defaults to `200006`) : Token id used for end-of-stream in the vocabulary.

qk_l2_clip_threshold (`float`, *optional*, defaults to 22.627416997969522) : Maximum L2 norm of each query and key head vector. Longer vectors are scaled down to this norm.

rel_l2_clip_threshold (`float`, *optional*, defaults to 8.0) : Maximum L2 norm of each relative-state head vector.

n_layers_per_attn_res_block (`int`, *optional*, defaults to 6) : Number of decoder layers summed into one block before the block becomes a source of the attention residuals.

mlp_gate_softcap (`float`, *optional*, defaults to 4.0) : Softcap applied to the linear part of the experts' gate projection.

mlp_up_softcap (`float`, *optional*, defaults to 25.0) : Softcap applied to the experts' up projection.

final_logit_softcapping (`float`, *optional*, defaults to 30.0) : Softcap applied to the output logits.

scoring_func (`str`, *optional*, defaults to `"sigmoid"`) : Activation turning the router logits into expert scores.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

loop_layer_index_start (`int`, *optional*, defaults to 6) : Index of the first looped decoder layer. Must be a multiple of `n_layers_per_attn_res_block`.

loop_layer_index_end (`int`, *optional*, defaults to 18) : Index after the last looped decoder layer. Layers `loop_layer_index_start` to `loop_layer_index_end - 1` are looped. Must be a multiple of `n_layers_per_attn_res_block`.

loop_num_passes (`int`, *optional*, defaults to 3) : Number of loops.

loop_gate_rank (`int`, *optional*, defaults to 64) : Rank of the message gate projections that mix the residual blocks of the previous pass into the current one.

This is the configuration class to store the configuration of a Inkling2LoopModel. It is used to instantiate a Inkling2 Loop
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Inkling2LoopPreTrainedModel[[transformers.Inkling2LoopPreTrainedModel]]

#### transformers.Inkling2LoopPreTrainedModel[[transformers.Inkling2LoopPreTrainedModel]]

```python
transformers.Inkling2LoopPreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1176)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2LoopPreTrainedModel.forward]]

```python
forward(*args, **kwargs)
```

A mock value for a dotted path (e.g. `torch.float32`): attribute access chains,
calls behave as pass-through decorators, `repr` is the dotted path, and using it
as a base class substitutes a plain-`type` base (PEP 560 `__mro_entries__`), so
real subclasses keep a normal metaclass and `inspect.signature` reads their real
`__init__` instead of a mock's.

## Inkling2LoopTextModel[[transformers.Inkling2LoopTextModel]]

#### transformers.Inkling2LoopTextModel[[transformers.Inkling2LoopTextModel]]

```python
transformers.Inkling2LoopTextModel(config: Inkling2LoopTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1272)

**Parameters:**

config ([Inkling2LoopTextConfig](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Inkling2 Loop Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2LoopTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1304)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.Tensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [Inkling2LoopTextModel](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **router_logits** (`tuple(torch.Tensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.

## Inkling2LoopForCausalLM[[transformers.Inkling2LoopForCausalLM]]

#### transformers.Inkling2LoopForCausalLM[[transformers.Inkling2LoopForCausalLM]]

```python
transformers.Inkling2LoopForCausalLM(config: Inkling2LoopTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1465)

**Parameters:**

config ([Inkling2LoopTextConfig](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Inkling2 Loop Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2LoopForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1483)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the causal language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.  The labels are **shifted internally** so that `labels[..., 1:]` are used to predict `logits[..., :-1, :]`. Pass unshifted labels matching `input_ids` (i.e. `labels = input_ids`). Do **not** shift the labels yourself before passing them.  If the labels are already shifted (for example with sequence parallelism), pass them as the `shift_labels` keyword argument and they are used as is. `labels` must still be passed for the loss to be computed.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `Inkling2LoopCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `Inkling2LoopCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [Inkling2LoopForCausalLM](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the language model backbone at the output of each layer plus the optional initial
  embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attention weights of the language model backbone, after the attention softmax.
- **image_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the image encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no images were processed.
- **video_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the video encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no videos were processed.
- **audio_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the audio encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no audio was processed.
- **router_logits** (`tuple[torch.Tensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, Inkling2LoopForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Mini")
>>> model = Inkling2LoopForCausalLM.from_pretrained(
...     "thinkingmachines/Inkling-2-Mini", dtype=torch.bfloat16, device_map="auto"
... )

>>> messages = [{"role": "user", "content": "Explain in one sentence why the sky is usually blue."}]
>>> inputs = tokenizer.apply_chat_template(
...     messages,
...     tokenize=True,
...     add_generation_prompt=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device)
>>> input_len = inputs["input_ids"].shape[-1]

>>> outputs = model.generate(**inputs, max_new_tokens=1024)
>>> print(tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False))
```

## Inkling2LoopModel[[transformers.Inkling2LoopModel]]

#### transformers.Inkling2LoopModel[[transformers.Inkling2LoopModel]]

```python
transformers.Inkling2LoopModel(config: Inkling2LoopConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1570)

**Parameters:**

config ([Inkling2LoopConfig](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Base Inkling2Loop model which consists of a vision backbone and a language model without language modeling head.,

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2LoopModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1618)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : Temporal, height and width of each image, counted in language model image tokens.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Inkling2LoopModelOutputWithPast` or `tuple(torch.Tensor)`

A `Inkling2LoopModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [Inkling2LoopModel](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the language model backbone.
  If `cache` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the language model backbone at the output of each layer plus the optional initial
  embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attention weights of the language model backbone, after the attention softmax.
- **image_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the image encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no images were processed.
- **video_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the video encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no videos were processed.
- **audio_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the audio encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no audio was processed.
- **router_logits** (`tuple[torch.Tensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import torch
>>> from transformers import AutoProcessor, Inkling2LoopModel

>>> processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling-2-Mini")
>>> model = Inkling2LoopModel.from_pretrained(
...     "thinkingmachines/Inkling-2-Mini", dtype=torch.bfloat16, device_map="auto"
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
...             },
...             {"type": "text", "text": "What animal is in this image, and where is it?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     add_generation_prompt=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device)

>>> outputs = model(**inputs)
>>> last_hidden_state = outputs.last_hidden_state
```

## Inkling2LoopForConditionalGeneration[[transformers.Inkling2LoopForConditionalGeneration]]

#### transformers.Inkling2LoopForConditionalGeneration[[transformers.Inkling2LoopForConditionalGeneration]]

```python
transformers.Inkling2LoopForConditionalGeneration(config: Inkling2LoopConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1720)

**Parameters:**

config ([Inkling2LoopConfig](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Base Inkling2Loop model which consists of a vision backbone and a language model without language modeling head.,

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2LoopForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/inkling2_loop/modeling_inkling2_loop.py#L1742)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : Temporal, height and width of each image, counted in language model image tokens.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Inkling2LoopCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `Inkling2LoopCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [Inkling2LoopForConditionalGeneration](/docs/transformers/v5.24.0/en/model_doc/inkling2_loop#transformers.Inkling2LoopForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).
  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `cache`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
  Hidden-states of the language model backbone at the output of each layer plus the optional initial
  embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`. Attention weights of the language model backbone, after the attention softmax.
- **image_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the image encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no images were processed.
- **video_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the video encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no videos were processed.
- **audio_outputs** (`BaseModelOutputWithPooling`, *optional*) -- Output of the audio encoder: `last_hidden_state`, `pooler_output`, `hidden_states` and `attentions`.
  `None` if no audio was processed.
- **router_logits** (`tuple[torch.Tensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import torch
>>> from transformers import AutoProcessor, Inkling2LoopForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling-2-Mini")
>>> model = Inkling2LoopForConditionalGeneration.from_pretrained(
...     "thinkingmachines/Inkling-2-Mini", dtype=torch.bfloat16, device_map="auto"
... )

>>> messages = [
...     {
...         "role": "user",
...         "content": [
...             {
...                 "type": "image",
...                 "image": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
...             },
...             {"type": "text", "text": "What animal is in this image, and where is it?"},
...         ],
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     add_generation_prompt=True,
...     return_dict=True,
...     return_tensors="pt",
... ).to(model.device)
>>> input_len = inputs["input_ids"].shape[-1]

>>> outputs = model.generate(**inputs, max_new_tokens=1024)
>>> print(processor.decode(outputs[0][input_len:], skip_special_tokens=False))
```

### Gemma 4 Assistant
https://huggingface.co/docs/transformers/v5.24.0/model_doc/gemma4_assistant.md
