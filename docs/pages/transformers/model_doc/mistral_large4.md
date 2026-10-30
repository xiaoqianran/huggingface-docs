# Mistral Large 4

## Overview

Mistral Large 4 is a natively multimodal mixture-of-experts model with 1 trillion parameters and 49 billion active
parameters per token (52 billion including the embeddings and the output layer). It takes text and images as input,
produces text, and can answer directly or reason first, with the reasoning effort set per request.

The text decoder builds on the DeepSeek-V3.2 recipe (Multi-head Latent Attention made sparse by a DSA indexer) and adds
a few changes of its own:

- **Attention**: 72 layers of Deepseek Sparse Attention (DSA) with a 512-dimensional KV latent, made sparse by an
  indexer selecting the top 2048 key positions for each query. Queries use a direct projection, there is no Q-LoRA.
  Some layers compute their own selection, the others reuse the one of a previous layer (`"shared"`) like in 
  [GLM-5](./glm_moe_dsa).
- **NoPE layers**: every 4th layer uses no rotary embedding, in the attention and in the indexer.
- **KV sharing**: from layer 39 and on, every 4th layer does not compute a KV latent, instead attending to the one of
  layer 35. They keep their own query, KV up-projection and output weights.
- **Scaled softmax**: queries are scaled per head by `s * log(position + 1) + b`, with learned `s` and `b`, in the
  attention and in the indexer.
- **LeRoPE**: YaRN extends the 512-token base RoPE by a factor of 2048 (1,048,576 positions), and LeRoPE adds a
  learned multiplier per rotary frequency.
- **MoE**: the first layer is dense, every other layer routes each token to 4 of 128 experts, plus one shared expert.
- **Logits**: final logits are soft-capped with `30 * tanh(logits / 30)`.

The vision encoder and the multimodal projector are the same as in [Mistral 3](./mistral3) (Pixtral encoder with a
patch-merge projector).

## Usage examples

The checkpoint is stored in FP8 (128x128 block-scaled weights with dynamic activation scales) and is loaded as is.
The full model needs about 1 TB of GPU memory, so `device_map="auto"` is used to spread it across several GPUs.

The chat template accepts a `reasoning_effort` argument, set to `"none"` (default) or `"high"`. With `"high"`, the model
writes its reasoning between `[THINK]` and `[/THINK]` before answering.

```python
from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration

model_id = "mistralai/Mistral-Large-4-1T-A52B"

processor = AutoProcessor.from_pretrained(model_id)
model = MistralLarge4ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            {"type": "text", "text": "Describe this image."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

```python
from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration

model_id = "mistralai/Mistral-Large-4-1T-A52B"

processor = AutoProcessor.from_pretrained(model_id)
model = MistralLarge4ForConditionalGeneration.from_pretrained(model_id, device_map="auto")

messages = [{"role": "user", "content": "How many prime numbers are there between 100 and 150?"}]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    reasoning_effort="high",
).to(model.device)

output = model.generate(**inputs, max_new_tokens=4096)
# Keep the special tokens to see the reasoning between [THINK] and [/THINK]
print(processor.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False))
```

## Notes

- The checkpoint also contains a draft head for speculative decoding (the `mtp_dspark.*` weights) using the DSpark
    speculative decoding method.
- The DSA indexer is applied as an attention mask on top of dense attention, with the `eager` and `sdpa` backends.
  It gives the same outputs as sparse attention but not its speed-up on long sequences.

## MistralLarge4Config[[transformers.MistralLarge4Config]]

#### transformers.MistralLarge4Config[[transformers.MistralLarge4Config]]

```python
transformers.MistralLarge4Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_index: int = 10, projector_hidden_act: str = 'gelu', vision_feature_layer: int | list[int] = -1, multimodal_projector_bias: bool = False, spatial_merge_size: int = 2, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/configuration_mistral_large4.py#L210)

**Parameters:**

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

image_token_index (`int`, *optional*, defaults to `10`) : The image token index used as a placeholder for input images.

projector_hidden_act (`str`, *optional*, defaults to `gelu`) : The activation function used by the multimodal projector.

vision_feature_layer (`Union[int, list[int]]`, *optional*, defaults to `-1`) : The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the corresponding indices will be concatenated to form the vision features.

multimodal_projector_bias (`bool`, *optional*, defaults to `False`) : Whether to use bias in the multimodal projector.

spatial_merge_size (`int`, *optional*, defaults to `2`) : The size of the spatial merge window used to reduce the number of visual tokens by merging neighboring patches.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a MistralLarge4Model. It is used to instantiate a Mistral Large4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [mistralai/Mistral-Small-3.1-24B-Instruct-2503](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MistralLarge4ForConditionalGeneration, MistralLarge4Config, PixtralVisionConfig, MistralConfig

>>> # Initializing a Pixtral-vision config
>>> vision_config = PixtralVisionConfig()

>>> # Initializing a Mistral config
>>> text_config = MistralConfig()

>>> # Initializing a MistralLarge4 configuration
>>> configuration = MistralLarge4Config(vision_config, text_config)

>>> # Initializing a model from the mistral_large4.1 configuration
>>> model = MistralLarge4ForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MistralLarge4TextConfig[[transformers.MistralLarge4TextConfig]]

#### transformers.MistralLarge4TextConfig[[transformers.MistralLarge4TextConfig]]

```python
transformers.MistralLarge4TextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 204800, hidden_size: int = 6144, intermediate_size: int = 24576, moe_intermediate_size: int = 6144, num_hidden_layers: int = 72, num_attention_heads: int = 64, num_key_value_heads: int = 64, n_shared_experts: int = 1, n_routed_experts: int = 128, output_router_logits: bool = False, routed_scaling_factor: float = 1.0, kv_lora_rank: int = 512, qk_rope_head_dim: int = 64, v_head_dim: int = 128, qk_nope_head_dim: int = 64, n_group: int = 1, topk_group: int = 1, num_experts_per_tok: int = 4, hidden_act: str = 'silu', max_position_embeddings: int = 1048576, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = None, bos_token_id: int | None = 1, eos_token_id: int | list[int] | None = 2, tie_word_embeddings: bool = False, rope_parameters: dict | None = None, mlp_layer_types: list[str] | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, index_topk: int = 2048, index_head_dim: int = 128, index_n_heads: int = 32, head_dim: int = 64, first_k_dense_replace: int = 1, layer_types: list[str] | None = None, indexer_types: list[str] | None = None, no_rope_layers: list[int] | None = None, no_rope_layer_interval: int = 4, kv_source_layers: list[int] | None = None, final_logit_softcapping: float = 30.0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/configuration_mistral_large4.py#L31)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `204800`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `6144`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `24576`) : Dimension of the MLP representations.

moe_intermediate_size (`int`, *optional*, defaults to `6144`) : Intermediate size of the routed expert MLPs.

num_hidden_layers (`int`, *optional*, defaults to `72`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `64`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `64`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

n_shared_experts (`int`, *optional*, defaults to `1`) : Number of shared experts.

n_routed_experts (`int`, *optional*, defaults to `128`) : Number of routed experts.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

routed_scaling_factor (`float`, *optional*, defaults to `1.0`) : Scaling factor or routed experts.

kv_lora_rank (`int`, *optional*, defaults to `512`) : Rank of the LoRA matrices for key and value projections.

qk_rope_head_dim (`int`, *optional*, defaults to `64`) : Dimension of the query/key heads that use rotary position embeddings.

v_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the value heads.

qk_nope_head_dim (`int`, *optional*, defaults to `64`) : Dimension of the query/key heads that don't use rotary position embeddings.

n_group (`int`, *optional*, defaults to 1) : Number of groups for routed experts.

topk_group (`int`, *optional*, defaults to `1`) : Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).

num_experts_per_tok (`int`, *optional*, defaults to `4`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `1048576`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `1`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `2`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

mlp_layer_types (`list`, *optional*) : MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 1 dense + rest sparse.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

index_topk (`int`, *optional*, defaults to 2048) : Number of top tokens selected by the indexer for sparse attention.

index_head_dim (`int`, *optional*, defaults to 128) : Head dimension for the indexer projections (DSA).

index_n_heads (`int`, *optional*, defaults to 32) : Number of heads for the indexer projections (DSA).

head_dim (`int`, *optional*, defaults to `64`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

first_k_dense_replace (`int`, *optional*, defaults to 1) : Number of leading layers that use a dense MLP; the rest use the MoE block.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

indexer_types (`list[str]`, *optional*) : Per-layer indexer mode (`"full"` runs the indexer, `"shared"` reuses the previous full layer's top-k). Defaults to the pattern derived from `index_topk_freq` / `index_skip_topk_offset` (or `index_topk_pattern`).

no_rope_layers (`list[int]`, *optional*) : List with at least the same length as the number of layers in the model. A `1` at an index position indicates that the corresponding layer will use RoPE, while a `0` indicates that it's a NoPE layer.

no_rope_layer_interval (`int`, *optional*, defaults to 4) : Every `no_rope_layer_interval`-th layer is a NoPE layer (no rotary embedding in the attention nor in the indexer). The other layers use RoPE.

kv_source_layers (`list[int]`, *optional*) : For each layer, the index of the layer whose latent KV cache it attends to. A layer whose entry is its own index computes and caches its own latent KV; a layer pointing at an earlier layer has no `kv_a_proj_with_mqa` and reuses that layer's cache (the source layer must be its own source). Defaults to no sharing.

final_logit_softcapping (`float`, *optional*, defaults to `30.0`) : Soft-capping value applied to the final logits before computing the probability distribution. Logits are scaled by `tanh(logit / cap) * cap`.

This is the configuration class to store the configuration of a MistralLarge4Model. It is used to instantiate a Mistral Large4
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import MistralLarge4TextConfig, MistralLarge4TextModel

>>> # Initializing a Mistral Large 4 text configuration
>>> configuration = MistralLarge4TextConfig()

>>> # Initializing a model from the configuration
>>> model = MistralLarge4TextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MistralLarge4PreTrainedModel[[transformers.MistralLarge4PreTrainedModel]]

#### transformers.MistralLarge4PreTrainedModel[[transformers.MistralLarge4PreTrainedModel]]

```python
transformers.MistralLarge4PreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L672)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

## MistralLarge4TextModel[[transformers.MistralLarge4TextModel]]

#### transformers.MistralLarge4TextModel[[transformers.MistralLarge4TextModel]]

```python
transformers.MistralLarge4TextModel(config: MistralLarge4TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L712)

**Parameters:**

config ([MistralLarge4TextConfig](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Mistral Large4 Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MistralLarge4TextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L728)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MistralLarge4Config](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4Config)) and inputs.

The [MistralLarge4TextModel](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

## MistralLarge4Model[[transformers.MistralLarge4Model]]

#### transformers.MistralLarge4Model[[transformers.MistralLarge4Model]]

```python
transformers.MistralLarge4Model(config: MistralLarge4Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L893)

**Parameters:**

config ([MistralLarge4Config](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MistralLarge4 model which consists of a vision backbone and a language model, without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MistralLarge4Model.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, vision_feature_layer: int | list[int] | None = None, use_cache: bool | None = None, image_sizes: typing.Optional[torch.Tensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L963)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [PixtralImageProcessor](/docs/transformers/v5.22.1/en/model_doc/pixtral#transformers.PixtralImageProcessor). See [PixtralImageProcessor.__call__()](/docs/transformers/v5.22.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([PixtralProcessor](/docs/transformers/v5.22.1/en/model_doc/pixtral#transformers.PixtralProcessor) uses [PixtralImageProcessor](/docs/transformers/v5.22.1/en/model_doc/pixtral#transformers.PixtralImageProcessor) for processing images).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

vision_feature_layer (`Union[int, list[int]]`, *optional*) : The index of the layer to select the vision feature. If multiple indices are provided, the vision feature of the corresponding indices will be concatenated to form the vision features.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

image_sizes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) : The sizes of the images in the batch, being (height, width) for each image.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MistralLarge4ModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MistralLarge4ModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MistralLarge4Config](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4Config)) and inputs.

The [MistralLarge4Model](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_logits=True` is passed) -- Tuple of `torch.FloatTensor` (one for each MoE layer) of shape `(batch_size * sequence_length, num_experts)`.

## MistralLarge4ForConditionalGeneration[[transformers.MistralLarge4ForConditionalGeneration]]

#### transformers.MistralLarge4ForConditionalGeneration[[transformers.MistralLarge4ForConditionalGeneration]]

```python
transformers.MistralLarge4ForConditionalGeneration(config: MistralLarge4Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L1062)

**Parameters:**

config ([MistralLarge4Config](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MISTRAL_LARGE4 model which consists of a vision backbone and a language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MistralLarge4ForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.FloatTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, image_sizes: typing.Optional[torch.Tensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/mistral_large4/modeling_mistral_large4.py#L1093)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [PixtralImageProcessor](/docs/transformers/v5.22.1/en/model_doc/pixtral#transformers.PixtralImageProcessor). See [PixtralImageProcessor.__call__()](/docs/transformers/v5.22.1/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([PixtralProcessor](/docs/transformers/v5.22.1/en/model_doc/pixtral#transformers.PixtralProcessor) uses [PixtralImageProcessor](/docs/transformers/v5.22.1/en/model_doc/pixtral#transformers.PixtralImageProcessor) for processing images).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

image_sizes (`torch.Tensor` of shape `(batch_size, 2)`, *optional*) : The sizes of the images in the batch, being (height, width) for each image.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MistralLarge4CausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MistralLarge4CausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MistralLarge4Config](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4Config)) and inputs.

The [MistralLarge4ForConditionalGeneration](/docs/transformers/v5.22.1/en/model_doc/mistral_large4#transformers.MistralLarge4ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.FloatTensor`, *optional*) -- A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> from PIL import Image
>>> from huggingface_hub.utils import httpx
>>> from io import BytesIO
>>> from transformers import AutoProcessor, MistralLarge4ForConditionalGeneration

>>> model = MistralLarge4ForConditionalGeneration.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")
>>> processor = AutoProcessor.from_pretrained("mistralai/Mistral-Small-3.1-24B-Instruct-2503")

>>> prompt = "<s>[INST][IMG]What is the image?[/INST]"
>>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
>>> with httpx.stream("GET", url) as response:
...     image = Image.open(BytesIO(response.read()))

>>> inputs = processor(images=image, text=prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=15)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"What is the image?The image depicts two cats lying on a pink blanket."
```

### SeamlessM4T
https://huggingface.co/docs/transformers/v5.22.1/model_doc/seamless_m4t.md
