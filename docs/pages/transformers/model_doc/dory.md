# Dory

Dory is a looped (latent-recurrent) language model from NVIDIA. It uses a [normalized-GPT](https://huggingface.co/papers/2410.01131) architecture with interleaved sliding-window and full attention layers. By applying a shared block of layers repeatedly, Dory can scale its intelligence by spending more compute per token. This compute can be adjusted by changing the loop count during inference with the same checkpoint.

Find all official checkpoints in the [Dory collection](https://huggingface.co/collections/nvidia/nemotron-labs-dory).

## Quickstart

Generate text with [Pipeline](/docs/transformers/v5.24.1/en/main_classes/pipelines#transformers.Pipeline) or [AutoModelForCausalLM](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoModelForCausalLM). Both examples use the chat template, which enables thinking by default, and the recommended sampling settings.

```python
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16",
)
pipe([{"role": "user", "content": "Name the largest planet in our solar system."}], max_new_tokens=512)
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")

inputs = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Name the largest planet in our solar system."}],
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=512)
print(tokenizer.decode(outputs[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

## Notes

- Set `n_recurrent_loops` to balance compute and accuracy. The released checkpoints default to 2. Try 1 for lower compute per token or 4 for higher accuracy. More loops can shorten responses, so measure end-to-end latency.
- Set `recurrent_kv_cache_mode="last_loop"` to share KV history across loops instead of keeping separate histories with the default `per_loop`. Cache memory then stays independent of loop count. Every loop still computes its own Q, K and V, but only the last loop writes them to the cache. Earlier loops read the cached past and attend over it plus their own new keys, without writing. The diagram below shows one recurrent layer `L` with 3 loops.

  ```text
  prefill "a b c":   cache[L] empty
    loop0: attend [k0(abc)]
    loop1: attend [k1(abc)]
    loop2: attend [k2(abc)]  → cache[L] = k2(abc)    (write)

  decode "d":        cache[L] = k2(abc)              (read)
    loop0: attend [k2(abc) | k0(d)]
    loop1: attend [k2(abc) | k1(d)]
    loop2: attend [k2(abc) | k2(d)] → cache[L] = k2(abcd)
  ```

## DoryConfig[[transformers.DoryConfig]]

#### transformers.DoryConfig[[transformers.DoryConfig]]

```python
transformers.DoryConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 131072, hidden_size: int = 2560, intermediate_size: int = 7168, num_hidden_layers: int = 36, num_attention_heads: int = 16, num_key_value_heads: int = 4, head_dim: int = 256, hidden_act: str = 'silu', max_position_embeddings: int = 8192, initializer_range: float = 0.0177, use_cache: bool = True, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None, tie_word_embeddings: bool = False, rope_parameters: dict | None = None, attention_bias: bool = False, attention_dropout: float = 0.0, mlp_bias: bool = False, sliding_window: int = 1024, layer_types: list[str] | None = None, ngpt_alpha_init: float = 0.05, n_input_layers: int = 8, n_recurrent_layers: int | None = None, n_output_layers: int = 8, n_recurrent_loops: int = 2, recurrent_kv_cache_mode: typing.Literal['per_loop', 'last_loop'] = 'per_loop')
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/dory/configuration_dory.py#L26)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `131072`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `2560`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `7168`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `36`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `4`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `256`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `8192`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.0177`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`float`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

sliding_window (`int`, *optional*, defaults to `1024`) : Sliding window attention window size. If `None`, no sliding window is applied.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

ngpt_alpha_init (`float`, *optional*, defaults to 0.05) : Initial value of the nGPT residual interpolation rates `lambda_1`/`lambda_2`.

n_input_layers (`int`, *optional*, defaults to 8) : Number of decoder layers run once before the recurrent section.

n_recurrent_layers (`int`, *optional*) : Number of decoder layers in the recurrent section. Defaults to the remaining layers.

n_output_layers (`int`, *optional*, defaults to 8) : Number of decoder layers run once after the recurrent section.

n_recurrent_loops (`int`, *optional*, defaults to 2) : Number of passes through the recurrent section. More loops trade compute for accuracy.

recurrent_kv_cache_mode (`str`, *optional*, defaults to `"per_loop"`) : `"per_loop"` keeps one KV cache per recurrent loop. `"last_loop"` keeps a single KV cache per recurrent layer, written only by the last loop (earlier loops read it), so cache memory does not grow with the loop count.

This is the configuration class to store the configuration of a DoryModel. It is used to instantiate a Dory
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import DoryConfig, DoryForCausalLM

>>> configuration = DoryConfig()
>>> model = DoryForCausalLM(configuration)
>>> configuration = model.config
```

## DoryModel[[transformers.DoryModel]]

#### transformers.DoryModel[[transformers.DoryModel]]

```python
transformers.DoryModel(config: DoryConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/dory/modeling_dory.py#L451)

**Parameters:**

config ([DoryConfig](/docs/transformers/v5.24.1/en/model_doc/dory#transformers.DoryConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dory Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.DoryModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/dory/modeling_dory.py#L468)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DoryConfig](/docs/transformers/v5.24.1/en/model_doc/dory#transformers.DoryConfig)) and inputs.

The [DoryModel](/docs/transformers/v5.24.1/en/model_doc/dory#transformers.DoryModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `cache` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
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

## DoryForCausalLM[[transformers.DoryForCausalLM]]

#### transformers.DoryForCausalLM[[transformers.DoryForCausalLM]]

```python
transformers.DoryForCausalLM(config: DoryConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/dory/modeling_dory.py#L571)

**Parameters:**

config ([DoryConfig](/docs/transformers/v5.24.1/en/model_doc/dory#transformers.DoryConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Dory Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.DoryForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/dory/modeling_dory.py#L586)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the causal language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.  The labels are **shifted internally** so that `labels[..., 1:]` are used to predict `logits[..., :-1, :]`. Pass unshifted labels matching `input_ids` (i.e. `labels = input_ids`). Do **not** shift the labels yourself before passing them.  If the labels are already shifted (for example with sequence parallelism), pass them as the `shift_labels` keyword argument and they are used as is. `labels` must still be passed for the loss to be computed.

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.Tensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([DoryConfig](/docs/transformers/v5.24.1/en/model_doc/dory#transformers.DoryConfig)) and inputs.

The [DoryForCausalLM](/docs/transformers/v5.24.1/en/model_doc/dory#transformers.DoryForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
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
>>> from transformers import AutoTokenizer, DoryForCausalLM

>>> model = DoryForCausalLM.from_pretrained("meta-dory/Dory-2-7b-hf")
>>> tokenizer = AutoTokenizer.from_pretrained("meta-dory/Dory-2-7b-hf")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

### AFMoE
https://huggingface.co/docs/transformers/v5.24.1/model_doc/afmoe.md
