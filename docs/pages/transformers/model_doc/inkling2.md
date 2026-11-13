# Inkling 2 Small

[Inkling 2 Small](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8) is a multimodal model from [Thinking Machines Lab](https://huggingface.co/thinkingmachines) that accepts text and image inputs and generates text outputs. It is intended for use in English and other languages, and across multiple coding languages. It keeps the decoder of [Inkling](./inkling), with learned relative attention, short causal convolutions and mixture-of-experts layers, and adds block attention residuals, gated attention with a learned sink and dynamic key/value convolutions. The language backbone consists of 48 mixture-of-experts decoder layers, each routing tokens to 8 of 384 experts. The model has 594B total and 17B active parameters. Images are encoded by a vision transformer based on the Perception Encoder and projected into the language model embeddings, one embedding per 40x40 pixel patch. 

The model is released with two checkpoints:
- [`thinkingmachines/Inkling-2-Small`](https://huggingface.co/thinkingmachines/Inkling-2-Small)
- [`thinkingmachines/Inkling-2-Small-MXFP8`](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8)

Both checkpoints contain quantized layers. `Inkling-2-Small` has both NVFP4 and MXFP8 quantized decoder layers. `Inkling-2-Small-MXFP8` contains only MXFP8 quantized layers. Currently only the `Inkling-2-Small-MXFP8` checkpoint is supported.

## Resources

- [Thinking Machines Blog](https://thinkingmachines.ai/news/inkling-2-small/)
- [Hugging Face Blog](https://huggingface.co/blog/thinkingmachines-inkling)

```python
import torch

from transformers import pipeline

pipe = pipeline(
    "image-text-to-text", model="thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
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
# <|message_model|><|content_thinking|>The user is asking to identify the animal and location in the image. Need answer based on visible content. The animal looks like a Pallas's cat (Otocolobus manul), also called manul:
# ...
# <|end_message|><|message_model|><|content_text|>The animal is a **Pallas’s cat** (also called a **manul**). It’s walking through **snow** in front of birch logs, likely in a **zoo or enclosed habitat**. The exact geographic location can’t be determined from the image alone; in the wild they’re native to parts of **Central Asia**.<|end_message|><|content_model_end_sampling|>
```

```python
import torch

from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "thinkingmachines/Inkling-2-Small-MXFP8"
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
# <|message_model|><|content_thinking|>The user is asking to identify the animal and location in the image. Need answer based on visible content. The animal looks like a Pallas's cat (Otocolobus manul), also called manul:
# ...
# <|end_message|><|message_model|><|content_text|>The animal is a **Pallas’s cat** (also called a **manul**). It’s walking through **snow** in front of birch logs, likely in a **zoo or enclosed habitat**. The exact geographic location can’t be determined from the image alone; in the wild they’re native to parts of **Central Asia**.<|end_message|><|content_model_end_sampling|>
```

## Notes

- Attention runs in eager mode only, because every layer adds a relative position bias and a sink.

## Inkling2Config[[transformers.Inkling2Config]]

#### transformers.Inkling2Config[[transformers.Inkling2Config]]

```python
transformers.Inkling2Config(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: transformers.models.inkling2.configuration_inkling2.Inkling2TextConfig | dict | None = None, vision_config: transformers.models.inkling2.configuration_inkling2.Inkling2VisionConfig | dict | None = None, image_token_id: int = 200054)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/configuration_inkling2.py#L194)

**Parameters:**

text_config (`Union[~models.inkling2.configuration_inkling2.Inkling2TextConfig, dict]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[~models.inkling2.configuration_inkling2.Inkling2VisionConfig, dict]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to 200054) : Token index of the image placeholders, each replaced by one image feature.

This is the configuration class to store the configuration of a Inkling2Model. It is used to instantiate a Inkling2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [thinkingmachines/Inkling-2-Small-MXFP8](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Inkling2TextConfig[[transformers.Inkling2TextConfig]]

#### transformers.Inkling2TextConfig[[transformers.Inkling2TextConfig]]

```python
transformers.Inkling2TextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 201024, unpadded_vocab_size: int = 200058, hidden_size: int = 4608, num_hidden_layers: int = 48, num_attention_heads: int = 48, num_key_value_heads: int = 8, head_dim: int = 128, swa_num_attention_heads: int = 48, swa_num_key_value_heads: int = 8, swa_head_dim: int = 128, sliding_window_size: int = 1024, d_rel: int = 16, rel_extent: int = 1024, log_scaling_n_floor: int = 128000, log_scaling_alpha: float = 0.1, layer_types: list[str] | None = None, max_position_embeddings: int = 1048576, rms_norm_eps: float = 1e-06, conv_kernel_size: int = 4, hidden_act: str = 'silu', moe_intermediate_size: int = 2304, n_routed_experts: int = 384, num_experts_per_tok: int = 8, route_scale: float = 8.0, attention_dropout: float = 0.0, initializer_range: float = 0.02, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | None = 200006, qk_l2_clip_threshold: float = 22.627416997969522, rel_l2_clip_threshold: float = 8.0, n_layers_per_attn_res_block: int = 6, mlp_gate_softcap: float = 4.0, mlp_up_softcap: float = 25.0, final_logit_softcapping: float = 30.0, scoring_func: str = 'sigmoid', use_cache: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/configuration_inkling2.py#L31)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `201024`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

unpadded_vocab_size (`int`, *optional*, defaults to 200058) : Number of rows of the unembedding matrix that hold real tokens. Logits beyond it are dropped.

hidden_size (`int`, *optional*, defaults to `4608`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `48`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `48`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*, defaults to `128`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

swa_num_attention_heads (`int`, *optional*, defaults to 48) : Number of attention heads in the sliding-window layers.

swa_num_key_value_heads (`int`, *optional*, defaults to 8) : Number of key/value heads in the sliding-window layers.

swa_head_dim (`int`, *optional*, defaults to 128) : Dimension of query and key heads in the sliding-window layers.

sliding_window_size (`int`, *optional*, defaults to 1024) : Size of the sliding attention window used by layers whose `layer_types` entry is `"hybrid_sliding"`.

d_rel (`int`, *optional*, defaults to 16) : Per-head dimension of the relative states that are mixed into the relative position bias.

rel_extent (`int`, *optional*, defaults to 1024) : Backward distance, in tokens, over which the relative position bias is applied in the full-attention layers.

log_scaling_n_floor (`int`, *optional*, defaults to 128000) : Position from which queries and relative states are scaled up logarithmically in the full-attention layers.

log_scaling_alpha (`float`, *optional*, defaults to 0.1) : Strength of the logarithmic scaling controlled by `log_scaling_n_floor`.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

max_position_embeddings (`int`, *optional*, defaults to `1048576`) : The maximum sequence length that this model might ever be used with.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

conv_kernel_size (`int`, *optional*, defaults to `4`) : The size of the convolutional kernel.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

moe_intermediate_size (`int`, *optional*, defaults to `2304`) : Intermediate size of the routed expert MLPs.

n_routed_experts (`int`, *optional*, defaults to `384`) : Number of routed experts.

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

This is the configuration class to store the configuration of a Inkling2Model. It is used to instantiate a Inkling2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [thinkingmachines/Inkling-2-Small-MXFP8](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Inkling2VisionConfig[[transformers.Inkling2VisionConfig]]

#### transformers.Inkling2VisionConfig[[transformers.Inkling2VisionConfig]]

```python
transformers.Inkling2VisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1024, intermediate_size: int | None = None, num_hidden_layers: int = 23, num_attention_heads: int = 16, num_channels: int = 3, patch_size: int = 20, hidden_act: str = 'gelu', layer_norm_eps: float = 1e-05, attention_dropout: float | int = 0.0, mlp_ratio: float = 4.0, layer_scale_init_value: float = 0.1, rope_parameters: dict | None = None, spatial_merge_size: int = 2, posemb_grid_size: int = 32, projection_hidden_dim: int = 4096, projector_bias: bool = False, text_hidden_size: int = 4608, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/configuration_inkling2.py#L139)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `23`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

patch_size (`int`, *optional*, defaults to `20`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

mlp_ratio (`float`, *optional*, defaults to `4.0`) : Ratio of the MLP hidden dim to the embedding dim.

layer_scale_init_value (`float`, *optional*, defaults to `0.1`) : Scale to use in the self-attention layers. 0.1 for base, 1e-6 for large. Set 0 to disable layer scale.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

spatial_merge_size (`int`, *optional*, defaults to 2) : Side, in native patches, of the square of patch features concatenated into one language model token.

posemb_grid_size (`int`, *optional*, defaults to 32) : Side of the learned position embedding grid, interpolated to each image's native patch grid.

projection_hidden_dim (`int`, *optional*, defaults to 4096) : Hidden size of the MLP projecting merged patch features to the language model.

projector_bias (`bool`, *optional*, defaults to `False`) : Whether the projection MLP uses biases.

text_hidden_size (`int`, *optional*, defaults to 4608) : Hidden size of the language model the image features are projected to.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Inkling2Model. It is used to instantiate a Inkling2
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [thinkingmachines/Inkling-2-Small-MXFP8](https://huggingface.co/thinkingmachines/Inkling-2-Small-MXFP8)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Inkling2PreTrainedModel[[transformers.Inkling2PreTrainedModel]]

#### transformers.Inkling2PreTrainedModel[[transformers.Inkling2PreTrainedModel]]

```python
transformers.Inkling2PreTrainedModel(config: PreTrainedConfig, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1162)

**Parameters:**

config ([PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2PreTrainedModel.forward]]

```python
forward(*args, **kwargs)
```

A mock value for a dotted path (e.g. `torch.float32`): attribute access chains,
calls behave as pass-through decorators, `repr` is the dotted path, and using it
as a base class substitutes a plain-`type` base (PEP 560 `__mro_entries__`), so
real subclasses keep a normal metaclass and `inspect.signature` reads their real
`__init__` instead of a mock's.

## Inkling2TextModel[[transformers.Inkling2TextModel]]

#### transformers.Inkling2TextModel[[transformers.Inkling2TextModel]]

```python
transformers.Inkling2TextModel(config: Inkling2TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1228)

**Parameters:**

config ([Inkling2TextConfig](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Inkling2 Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2TextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1254)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.Tensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) and inputs.

The [Inkling2TextModel](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2TextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
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

## Inkling2ForCausalLM[[transformers.Inkling2ForCausalLM]]

#### transformers.Inkling2ForCausalLM[[transformers.Inkling2ForCausalLM]]

```python
transformers.Inkling2ForCausalLM(config: Inkling2TextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1320)

**Parameters:**

config ([Inkling2TextConfig](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2TextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Inkling2 Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2ForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1338)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the causal language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.  The labels are **shifted internally** so that `labels[..., 1:]` are used to predict `logits[..., :-1, :]`. Pass unshifted labels matching `input_ids` (i.e. `labels = input_ids`). Do **not** shift the labels yourself before passing them.  If the labels are already shifted (for example with sequence parallelism), pass them as the `shift_labels` keyword argument and they are used as is. `labels` must still be passed for the loss to be computed.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `Inkling2CausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `Inkling2CausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) and inputs.

The [Inkling2ForCausalLM](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.Tensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.Tensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.Tensor`, *optional*) -- A `torch.Tensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
- **router_logits** (`tuple[torch.Tensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, Inkling2ForCausalLM

>>> tokenizer = AutoTokenizer.from_pretrained("thinkingmachines/Inkling-2-Small-MXFP8")
>>> model = Inkling2ForCausalLM.from_pretrained(
...     "thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
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

## Inkling2VisionModel[[transformers.Inkling2VisionModel]]

#### transformers.Inkling2VisionModel[[transformers.Inkling2VisionModel]]

```python
transformers.Inkling2VisionModel(config: Inkling2VisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1407)

**Parameters:**

config ([Inkling2VisionConfig](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2VisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Inkling2 Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2VisionModel.forward]]

```python
forward(pixel_values: Tensor, image_grid_thw: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1427)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(num_image_tokens, patch_size * spatial_merge_size, patch_size * spatial_merge_size, num_channels)`) : Image patches in `[-1, 1]`, one per language model image token, in raster order within each image.

image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`) : Temporal, height and width of each image, counted in language model image tokens.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) and inputs.

The [Inkling2VisionModel](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2VisionModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.Tensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Inkling2Model[[transformers.Inkling2Model]]

#### transformers.Inkling2Model[[transformers.Inkling2Model]]

```python
transformers.Inkling2Model(config: Inkling2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1475)

**Parameters:**

config ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Base Inkling2 model which consists of a vision backbone and a language model without language modeling head.,

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2Model.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1523)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Inkling2ImageProcessor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ImageProcessor). See [Inkling2ImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Inkling2Processor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Processor) uses [Inkling2ImageProcessor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : Temporal, height and width of each image, counted in language model image tokens.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Inkling2ModelOutputWithPast` or `tuple(torch.Tensor)`

A `Inkling2ModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) and inputs.

The [Inkling2Model](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Model) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
  `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
  input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.Tensor`, *optional*) -- A `torch.Tensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
- **router_logits** (`tuple[torch.Tensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import torch
>>> from transformers import AutoProcessor, Inkling2Model

>>> processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling-2-Small-MXFP8")
>>> model = Inkling2Model.from_pretrained(
...     "thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
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

## Inkling2ForConditionalGeneration[[transformers.Inkling2ForConditionalGeneration]]

#### transformers.Inkling2ForConditionalGeneration[[transformers.Inkling2ForConditionalGeneration]]

```python
transformers.Inkling2ForConditionalGeneration(config: Inkling2Config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1623)

**Parameters:**

config ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Base Inkling2 model which consists of a vision backbone and a language model without language modeling head.,

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Inkling2ForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/modeling_inkling2.py#L1647)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Inkling2ImageProcessor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ImageProcessor). See [Inkling2ImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Inkling2Processor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Processor) uses [Inkling2ImageProcessor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ImageProcessor) for processing images).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : Temporal, height and width of each image, counted in language model image tokens.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Inkling2CausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `Inkling2CausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Inkling2Config](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Config)) and inputs.

The [Inkling2ForConditionalGeneration](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.text_config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple[torch.Tensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.Tensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **image_hidden_states** (`torch.Tensor`, *optional*) -- A `torch.Tensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
  image_hidden_states of the model produced by the vision encoder after projecting last hidden state.
- **router_logits** (`tuple[torch.Tensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

Example:

```python
>>> import torch
>>> from transformers import AutoProcessor, Inkling2ForConditionalGeneration

>>> processor = AutoProcessor.from_pretrained("thinkingmachines/Inkling-2-Small-MXFP8")
>>> model = Inkling2ForConditionalGeneration.from_pretrained(
...     "thinkingmachines/Inkling-2-Small-MXFP8", dtype=torch.bfloat16, device_map="auto"
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

## Inkling2ImageProcessor[[transformers.Inkling2ImageProcessor]]

#### transformers.Inkling2ImageProcessor[[transformers.Inkling2ImageProcessor]]

```python
transformers.Inkling2ImageProcessor(**kwargs: *<class 'transformers.processing_utils.ImagesKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/image_processing_inkling2.py#L34)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 40, 'width': 40}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models. Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors. --

Constructs an Inkling2 image processor.

disable_grouping (`bool`, *kwargs*, *optional*):
Whether to disable grouping of images by size to process them individually and not in batches.
If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on
empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157
image_seq_length (`int`, *kwargs*, *optional*):
The number of image tokens to be used for each image in the input.
Added for backward compatibility but this should be set as a processor attribute in future models.

## Inkling2Processor[[transformers.Inkling2Processor]]

#### transformers.Inkling2Processor[[transformers.Inkling2Processor]]

```python
transformers.Inkling2Processor(image_processor = None, tokenizer = None, chat_template = None, image_token = 'REDACTED')
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/inkling2/processing_inkling2.py#L26)

**Parameters:**

image_processor (`Inkling2ImageProcessor`) : The image processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

image_token (`str`, *optional*, defaults to `"<|unused_200054|>"`) : Placeholder token for each image soft-token slot, replaced by image features.

Constructs a Inkling2Processor which wraps a image processor and a tokenizer into a single processor.

[Inkling2Processor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2Processor) offers all the functionalities of [Inkling2ImageProcessor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ImageProcessor) and `tokenizer_class`. See the
[~Inkling2ImageProcessor](/docs/transformers/v5.23.0/en/model_doc/inkling2#transformers.Inkling2ImageProcessor) and `~tokenizer_class` for more information.

### SpeechT5
https://huggingface.co/docs/transformers/v5.23.0/model_doc/speecht5.md
