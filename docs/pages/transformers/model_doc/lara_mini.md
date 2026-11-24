# Lara Mini

## Overview

Lara Mini is a multilingual base language model by [Translated](https://translated.com) with 7.2B total and 0.6B
active parameters per token. It is a hybrid of linear-time recurrent layers, sliding-window attention and a few global
attention layers, each followed by a fine-grained mixture of experts.

For more details, see the [Lara Mini Base model card](https://huggingface.co/translated/Lara-4-Mini-Base) and
the [technical report](https://huggingface.co/papers/2611.16956). The weights are released under the CC BY-NC-SA 4.0
license (non-commercial use) in a gated repository: request access on the model page, then log in with
`hf auth login`.

This model was contributed by [cvalgian](https://huggingface.co/cvalgian).

> [!TIP]
> Set `use_kernels=True` in [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) to replace supported layers with optimized kernels from the Hub. Refer to [Loading kernels](../kernel_doc/loading_kernels) to learn more.

## Usage tips

- The published checkpoints fuse the gated delta net (GDN) input projections into one `in_proj` and call their gated
  output norm `out_norm`. They are converted when the model is loaded and converted back when it is saved.
- The GDN layers use fast GPU kernels when available: Hub kernels when loading with `use_kernels=True`, otherwise the
  [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) and
  [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) packages if installed. Without them, they fall back to a
  slower PyTorch implementation.

## Example

The following example shows how to generate text using the `AutoModelForCausalLM` class.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "translated/Lara-4-Mini-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("La capitale d'Italia è", return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Fine-tuning with LoRA

Since the GDN input projection is split when loading, LoRA adapters target `in_proj_qkv`, `in_proj_z`, `in_proj_b` and
`in_proj_a`, alongside `out_proj`, the attention `q_proj`/`k_proj`/`v_proj`/`o_proj` and the shared expert
`gate_proj`/`up_proj`/`down_proj`. The routed experts are 3D parameters: target them with `target_parameters`. Pass
`output_router_logits=True` to add the router load-balancing loss used in pretraining.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"],
    target_parameters=["mlp.experts.gate_up_proj", "mlp.experts.down_proj"],
)
model = get_peft_model(model, config)
```

> [!WARNING]
> Adapters trained with the Hub remote code (`trust_remote_code=True`) target the fused `linear_attn.in_proj` instead.
> They load here only partially and without a warning: the `in_proj` LoRA weights are dropped. Convert them by copying
> `lora_A` to the four projections and splitting `lora_B` by rows into
> `[2 * key_dim + value_dim, value_dim, num_v_heads, num_v_heads]`. The word embeddings are tied, so don't merge an
> adapter on `embed_tokens` or `lm_head` without untying them first.

## LaraMiniConfig[[transformers.LaraMiniConfig]]

#### transformers.LaraMiniConfig[[transformers.LaraMiniConfig]]

```python
transformers.LaraMiniConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 151936, hidden_size: int = 1024, num_hidden_layers: int = 34, num_attention_heads: int = 16, num_key_value_heads: int = 4, hidden_act: str = 'silu', max_position_embeddings: int = 131072, initializer_range: float = 0.02, rms_norm_eps: float = 1e-06, use_cache: bool = True, tie_word_embeddings: bool = True, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, head_dim: int = 128, linear_conv_kernel_dim: int = 4, linear_key_head_dim: int = 128, linear_value_head_dim: int = 128, linear_num_key_heads: int = 8, linear_num_value_heads: int = 16, moe_intermediate_size: int = 512, shared_expert_intermediate_size: int = 512, num_experts_per_tok: int = 4, num_experts: int = 128, norm_topk_prob: bool = True, output_router_logits: bool = False, router_aux_loss_coef: float = 0.001, layer_types: list[str] | None = None, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = 151643, sliding_window: int = 512)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/lara_mini/configuration_lara_mini.py#L29)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `151936`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `34`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `4`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `131072`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

head_dim (`int`, *optional*, defaults to `128`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

linear_conv_kernel_dim (`int`, *optional*, defaults to 4) : Kernel size of the causal depthwise convolution in the gated delta net (`"linear_attention"`) layers.

linear_key_head_dim (`int`, *optional*, defaults to 128) : Dimension of each query and key head in the gated delta net layers.

linear_value_head_dim (`int`, *optional*, defaults to 128) : Dimension of each value head in the gated delta net layers.

linear_num_key_heads (`int`, *optional*, defaults to 8) : Number of query and key heads in the gated delta net layers.

linear_num_value_heads (`int`, *optional*, defaults to 16) : Number of value heads in the gated delta net layers. Each query and key head serves `linear_num_value_heads // linear_num_key_heads` consecutive value heads.

moe_intermediate_size (`int`, *optional*, defaults to `512`) : Intermediate size of the routed expert MLPs.

shared_expert_intermediate_size (`int`, *optional*, defaults to `512`) : Intermediate size of the shared expert MLPs.

num_experts_per_tok (`int`, *optional*, defaults to `4`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

num_experts (`int`, *optional*, defaults to `128`) : Number of routed experts in MoE layers. 

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

router_aux_loss_coef (`float`, *optional*, defaults to `0.001`) : Auxiliary load balancing loss coefficient. Used to penalize uneven expert routing in MoE models.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `151643`) : Token id used for end-of-stream in the vocabulary.

sliding_window (`int`, *optional*, defaults to `512`) : Sliding window attention window size. If `None`, no sliding window is applied.

This is the configuration class to store the configuration of a LaraMiniModel. It is used to instantiate a Lara Mini
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [translated/Lara-4-Mini-Base](https://huggingface.co/translated/Lara-4-Mini-Base)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import LaraMiniModel, LaraMiniConfig

>>> # Initializing a LaraMini style configuration
>>> configuration = LaraMiniConfig()

>>> # Initializing a model from the LaraMini style configuration
>>> model = LaraMiniModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## LaraMiniModel[[transformers.LaraMiniModel]]

#### transformers.LaraMiniModel[[transformers.LaraMiniModel]]

```python
transformers.LaraMiniModel(config: LaraMiniConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/lara_mini/modeling_lara_mini.py#L909)

**Parameters:**

config ([LaraMiniConfig](/docs/transformers/v5.24.1/en/model_doc/lara_mini#transformers.LaraMiniConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Lara Mini Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.LaraMiniModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/lara_mini/modeling_lara_mini.py#L922)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.Tensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [LaraMiniModel](/docs/transformers/v5.24.1/en/model_doc/lara_mini#transformers.LaraMiniModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

## LaraMiniForCausalLM[[transformers.LaraMiniForCausalLM]]

#### transformers.LaraMiniForCausalLM[[transformers.LaraMiniForCausalLM]]

```python
transformers.LaraMiniForCausalLM(config: LaraMiniConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/lara_mini/modeling_lara_mini.py#L1065)

**Parameters:**

config ([LaraMiniConfig](/docs/transformers/v5.24.1/en/model_doc/lara_mini#transformers.LaraMiniConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Lara Mini Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.LaraMiniForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, output_router_logits: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/lara_mini/modeling_lara_mini.py#L1083)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the causal language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.  The labels are **shifted internally** so that `labels[..., 1:]` are used to predict `logits[..., :-1, :]`. Pass unshifted labels matching `input_ids` (i.e. `labels = input_ids`). Do **not** shift the labels yourself before passing them.  If the labels are already shifted (for example with sequence parallelism), pass them as the `shift_labels` keyword argument and they are used as is. `labels` must still be passed for the loss to be computed.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

output_router_logits (`bool`, *optional*) : Whether or not to return the logits of all the routers. They are useful for computing the router loss, and should not be returned during inference.

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [LaraMiniForCausalLM](/docs/transformers/v5.24.1/en/model_doc/lara_mini#transformers.LaraMiniForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **aux_loss** (`torch.Tensor`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`, and the model trains its router with a load-balancing loss) -- Load-balancing auxiliary loss for the sparse modules. Models that balance their experts with a router bias
  instead (DeepSeek-V3 and the architectures derived from it) return `None` here while still returning
  `router_logits`.
- **router_logits** (`tuple(torch.Tensor)`, *optional*, returned when `output_router_logits=True` is passed or when `config.output_router_logits=True`) -- Tuple of `torch.Tensor` (one for each sparse layer) of shape `(batch_size * sequence_length, num_experts)`.

  Raw router logits computed by the MoE routers. They can be used to compute a load-balancing loss or to
  inspect how tokens are dispatched to experts.
- **cache** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `cache` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoTokenizer, LaraMiniForCausalLM

>>> model = LaraMiniForCausalLM.from_pretrained("translated/Lara-4-Mini-Base")
>>> tokenizer = AutoTokenizer.from_pretrained("translated/Lara-4-Mini-Base")

>>> prompt = "La capitale d'Italia è"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(**inputs, max_new_tokens=16)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
```

### GraniteMoe
https://huggingface.co/docs/transformers/v5.24.1/model_doc/granitemoe.md
