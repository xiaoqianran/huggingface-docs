# Dots 3 Note Preview

Dots 3 Note Preview is a mixture-of-experts causal language model with native text, image, video, and audio inputs. It uses
a shared vision encoder for images and videos and a Whisper-style audio encoder. Both encoders project their outputs
into the language model's hidden space before autoregressive text generation.

BF16 and FP8 inference are supported.

The checkpoint needs a standard `processor_config.json` with Qwen image/video components and Dots preprocessing settings.
Saved configs use the native Transformers schema; older serving frameworks may need a compatibility adapter.

## Usage

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "dots-studio/dots3-note-prev"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]
inputs = processor.apply_chat_template(
    conversation,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=128)
print(processor.batch_decode(output_ids, skip_special_tokens=True)[0])
```

Use the same processing and generation calls with any of these user messages:

```python
# Text
conversation = [{"role": "user", "content": [{"type": "text", "text": "Briefly introduce yourself."}]}]

# Audio: replace the path with a local recording.
conversation = [{"role": "user", "content": [
    {"type": "audio", "path": "speech.wav"},
    {"type": "text", "text": "Transcribe this recording."},
]}]

# Video frames. Pass audio separately to include the soundtrack.
conversation = [{"role": "user", "content": [
    {"type": "video", "path": "concert.mp4"},
    {"type": "text", "text": "Describe what you see."},
]}]
```

`Dots3NoteModel` combines the text, vision and audio encoders. `Dots3NoteForConditionalGeneration` adds the
language-model head and generation interface. `Dots3NoteForCausalLM` is the text-only variant.

## Dots3NoteConfig[[transformers.Dots3NoteConfig]]

#### transformers.Dots3NoteConfig[[transformers.Dots3NoteConfig]]

```python
transformers.Dots3NoteConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.models.dots3_note.configuration_dots3_note.Dots3NoteTextConfig | None = None, vision_config: dict | transformers.models.dots3_note.configuration_dots3_note.Dots3NoteVisionConfig | None = None, audio_config: dict | transformers.models.dots3_note.configuration_dots3_note.Dots3NoteAudioConfig | None = None, image_token_id: int = 151660, video_token_id: int = 151680, audio_token_id: int = 151720, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/configuration_dots3_note.py#L391)

**Parameters:**

text_config (`Union[dict, ~models.dots3_note.configuration_dots3_note.Dots3NoteTextConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~models.dots3_note.configuration_dots3_note.Dots3NoteVisionConfig]`, *optional*) : The config object or dictionary of the vision backbone.

audio_config (`Union[dict, ~models.dots3_note.configuration_dots3_note.Dots3NoteAudioConfig]`, *optional*) : The config object or dictionary of the audio backbone.

image_token_id (`int`, *optional*, defaults to `151660`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `151680`) : The video token index used as a placeholder for input videos.

audio_token_id (`int`, *optional*, defaults to `151720`) : The audio token index used as a placeholder for input audio.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Dots3NoteModel. It is used to instantiate a Dots3 Note
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Dots3NoteTextConfig[[transformers.Dots3NoteTextConfig]]

#### transformers.Dots3NoteTextConfig[[transformers.Dots3NoteTextConfig]]

```python
transformers.Dots3NoteTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 152064, hidden_size: int = 5120, intermediate_size: int = 13824, moe_intermediate_size: int = 1536, num_hidden_layers: int = 46, num_attention_heads: int = 128, num_key_value_heads: int = 128, n_shared_experts: int = 1, n_routed_experts: int = 256, output_router_logits: bool = False, routed_scaling_factor: float = 1.0, kv_lora_rank: int = 512, q_lora_rank: int | None = 1024, qk_rope_head_dim: int = 64, v_head_dim: int = 128, qk_nope_head_dim: int = 128, n_group: int = 1, topk_group: int = 1, num_experts_per_tok: int = 8, norm_topk_prob: bool = True, hidden_act: str = 'silu', max_position_embeddings: int = 524288, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, pad_token_id: int | None = 151659, bos_token_id: int | None = 151643, eos_token_id: int | list[int] | None = 151668, tie_word_embeddings: bool = False, rope_parameters: dict | None = None, mlp_layer_types: list[str] | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, index_topk: int = 2048, index_head_dim: int = 128, index_n_heads: int = 64, mlp_bias: bool = False, head_dim: int = 256, layer_types: list[str] | None = None, sliding_window: int | None = None, shared_experts_intermediate_size: int = 1536)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/configuration_dots3_note.py#L191)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `152064`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `5120`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `13824`) : Dimension of the MLP representations.

moe_intermediate_size (`int`, *optional*, defaults to `1536`) : Intermediate size of the routed expert MLPs.

num_hidden_layers (`int`, *optional*, defaults to `46`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `128`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `128`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

n_shared_experts (`int`, *optional*, defaults to `1`) : Number of shared experts.

n_routed_experts (`int`, *optional*, defaults to `256`) : Number of routed experts.

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

routed_scaling_factor (`float`, *optional*, defaults to `1.0`) : Scaling factor or routed experts.

kv_lora_rank (`int`, *optional*, defaults to `512`) : Rank of the LoRA matrices for key and value projections.

q_lora_rank (`int`, *optional*, defaults to `1024`) : Rank of the LoRA matrices for query projections.

qk_rope_head_dim (`int`, *optional*, defaults to `64`) : Dimension of the query/key heads that use rotary position embeddings.

v_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the value heads.

qk_nope_head_dim (`int`, *optional*, defaults to `128`) : Dimension of the query/key heads that don't use rotary position embeddings.

n_group (`int`, *optional*, defaults to 1) : Number of groups into which routed experts are partitioned during selection.

topk_group (`int`, *optional*, defaults to `1`) : Number of selected groups for each token (for each token, ensuring the selected experts is only within `topk_group` groups).

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

norm_topk_prob (`bool`, *optional*, defaults to `True`) : Whether to normalize the weights of the routed experts. 

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `524288`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `151659`) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*, defaults to `151643`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `151668`) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

mlp_layer_types (`list[str]`, *optional*) : Per-layer `"dense"` or `"sparse"` MLP types, derived from the released MoE schedule when omitted.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

index_topk (`int`, *optional*, defaults to 2048) : Number of key positions selected by dynamic sparse attention.

index_head_dim (`int`, *optional*, defaults to 128) : Per-head width of the dynamic sparse attention indexer.

index_n_heads (`int`, *optional*, defaults to 64) : Number of heads in the dynamic sparse attention indexer.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

head_dim (`int`, *optional*, defaults to `256`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

sliding_window (`int`, *optional*) : Sliding window attention window size. If `None`, no sliding window is applied.

shared_experts_intermediate_size (`int`, *optional*, defaults to 1536) : Intermediate width of each shared expert.

This is the configuration class to store the configuration of a Dots3NoteModel. It is used to instantiate a Dots3 Note
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

#### convert_rope_params_to_dict[[transformers.Dots3NoteTextConfig.convert_rope_params_to_dict]]

```python
convert_rope_params_to_dict(**kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/configuration_dots3_note.py#L343)

Convert legacy RoPE settings to separate DSA/SWA entries, using each layer type's rotary head fraction.

## Dots3NoteVisionConfig[[transformers.Dots3NoteVisionConfig]]

#### transformers.Dots3NoteVisionConfig[[transformers.Dots3NoteVisionConfig]]

```python
transformers.Dots3NoteVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1536, intermediate_size: int = 4224, num_hidden_layers: int = 42, num_attention_heads: int = 24, num_channels: int = 3, patch_size: int = 14, hidden_act: str = 'silu', attention_dropout: float | int = 0.0, initializer_range: float = 0.02, rope_parameters: dict | None = None, moe_intermediate_size: int = 2112, spatial_merge_size: int = 2, rms_norm_eps: float = 1e-05, attention_bias: bool = False, mlp_bias: bool = False, mlp_layer_types: list[str] | None = None, num_local_experts: int = 64, num_experts_per_tok: int | None = None, router_scaling_factor: float | None = None, adapter_in_dim: int = 1536, adapter_out_dim: int = 5120, adapter_merge_size: int = 2)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/configuration_dots3_note.py#L28)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1536`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `4224`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `42`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `24`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

patch_size (`int`, *optional*, defaults to `14`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

moe_intermediate_size (`int`, *optional*, defaults to `2112`) : Intermediate size of the routed expert MLPs.

spatial_merge_size (`int`, *optional*, defaults to `2`) : The size of the spatial merge window used to reduce the number of visual tokens by merging neighboring patches.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

mlp_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in up_proj, down_proj and gate_proj layers in the MLP layers.

mlp_layer_types (`list[str]`, *optional*) : Dense or sparse MLP for each layer. Expert counts are specified through `per_layer_config`.

num_local_experts (`int`, *optional*, defaults to `64`) : Number of local experts on each device. `num_experts` should be divisible by `num_local_experts`.

num_experts_per_tok (`int`, *optional*) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

router_scaling_factor (`float`, *optional*, defaults to 1.0) : Scale applied to the vision router scores.

adapter_in_dim (`int`, *optional*, defaults to 1536) : Input width of the vision adapter. It must match `hidden_size`.

adapter_out_dim (`int`, *optional*, defaults to 5120) : Output width of the vision adapter. It must match the text hidden size.

adapter_merge_size (`int`, *optional*, defaults to 2) : Spatial merge factor used by the vision adapter.

This is the configuration class to store the configuration of a Dots3NoteModel. It is used to instantiate a Dots3 Note
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Dots3NoteAudioConfig[[transformers.Dots3NoteAudioConfig]]

#### transformers.Dots3NoteAudioConfig[[transformers.Dots3NoteAudioConfig]]

```python
transformers.Dots3NoteAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1280, intermediate_size: int = 5120, num_hidden_layers: int = 32, num_attention_heads: int = 20, num_key_value_heads: int | None = None, hidden_act: str = 'silu', max_position_embeddings: int = 6000, initializer_range: float = 0.02, rope_parameters: dict | None = None, attention_dropout: float | int = 0.0, dropout: float = 0.0, sampling_rate: int = 16000, feature_size: int = 128, hop_length: int = 160, downsample_hidden_size: int = 480, head_dim: int | None = None, attention_bias: bool = True, adapter_input_size: int = 1280, adapter_out_dim: int = 5120)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/configuration_dots3_note.py#L116)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1280`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `5120`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `32`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `20`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `6000`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

dropout (`float`, *optional*, defaults to `0.0`) : The ratio for all dropout layers.

sampling_rate (`int`, *optional*, defaults to `16000`) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

feature_size (`int`, *optional*, defaults to 128) : Number of log-mel filter-bank channels.

hop_length (`int`, *optional*, defaults to 160) : Number of waveform samples between adjacent log-mel frames.

downsample_hidden_size (`int`, *optional*, defaults to 480) : Hidden width of the temporal downsampling projection.

head_dim (`int`, *optional*) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

attention_bias (`bool`, *optional*, defaults to `True`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

adapter_input_size (`int`, *optional*, defaults to 1280) : Input width of the audio-to-text adapter.

adapter_out_dim (`int`, *optional*, defaults to 5120) : Output width of the audio-to-text adapter. It must match the text hidden size.

This is the configuration class to store the configuration of a Dots3NoteModel. It is used to instantiate a Dots3 Note
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [dots-studio/dots3-note-prev](https://huggingface.co/dots-studio/dots3-note-prev)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.23.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Dots3NoteForCausalLM[[transformers.Dots3NoteForCausalLM]]

#### transformers.Dots3NoteForCausalLM[[transformers.Dots3NoteForCausalLM]]

```python
transformers.Dots3NoteForCausalLM(config: Dots3NoteTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L783)

**Parameters:**

config ([Dots3NoteTextConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Dots3 Note Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Dots3NoteForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L799)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the causal language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.  The labels are **shifted internally** so that `labels[..., 1:]` are used to predict `logits[..., :-1, :]`. Pass unshifted labels matching `input_ids` (i.e. `labels = input_ids`). Do **not** shift the labels yourself before passing them.  If the labels are already shifted (for example with sequence parallelism), pass them as the `shift_labels` keyword argument and they are used as is. `labels` must still be passed for the loss to be computed.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) and inputs.

The [Dots3NoteForCausalLM](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteForCausalLM) forward method, overrides the `__call__` special method.

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
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoModelForCausalLM, AutoTokenizer

>>> model_id = "dots-studio/dots3-note-prev"
>>> tokenizer = AutoTokenizer.from_pretrained(model_id)
>>> model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
>>> inputs = tokenizer("Briefly introduce yourself.", return_tensors="pt").to(model.device)
>>> output = model.generate(**inputs, max_new_tokens=32)
>>> text = tokenizer.batch_decode(output, skip_special_tokens=True)
```

## Dots3NoteModel[[transformers.Dots3NoteModel]]

#### transformers.Dots3NoteModel[[transformers.Dots3NoteModel]]

```python
transformers.Dots3NoteModel(config: Dots3NoteConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1562)

**Parameters:**

config ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dots3 Note Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Dots3NoteModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, video_grid_thw: typing.Optional[torch.Tensor] = None, input_features: typing.Optional[torch.Tensor] = None, chunk_sample_lengths: typing.Optional[torch.Tensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1641)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

image_grid_thw (`torch.Tensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See [Qwen3VLVideoProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

video_grid_thw (`torch.Tensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor). See `Dots3NoteFeatureExtractor.__call__()` for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor) for processing audios).

chunk_sample_lengths (`torch.Tensor`, *optional*) : Waveform sample count for each audio feature chunk.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.Tensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) and inputs.

The [Dots3NoteModel](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteModel) forward method, overrides the `__call__` special method.

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

## Dots3NoteForConditionalGeneration[[transformers.Dots3NoteForConditionalGeneration]]

#### transformers.Dots3NoteForConditionalGeneration[[transformers.Dots3NoteForConditionalGeneration]]

```python
transformers.Dots3NoteForConditionalGeneration(config: Dots3NoteConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1724)

**Parameters:**

config ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Dots3 Note Model for token generation conditioned on other modalities (e.g. image-text-to-text generation).

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Dots3NoteForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.Tensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, chunk_sample_lengths: typing.Optional[torch.Tensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1738)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor). See `Dots3NoteFeatureExtractor.__call__()` for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

pixel_values_videos (`torch.Tensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See [Qwen3VLVideoProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2VideoProcessor.__call__) for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

chunk_sample_lengths (`torch.Tensor`, *optional*) : Number of waveform samples in each audio chunk.

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.Tensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) and inputs.

The [Dots3NoteForConditionalGeneration](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteForConditionalGeneration) forward method, overrides the `__call__` special method.

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
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from PIL import Image
>>> from transformers import AutoProcessor, Dots3NoteForConditionalGeneration

>>> model = Dots3NoteForConditionalGeneration.from_pretrained("dots-studio/dots3-note-prev")
>>> processor = AutoProcessor.from_pretrained("dots-studio/dots3-note-prev")

>>> messages = [
...     {
...         "role": "user", "content": [
...             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
...             {"type": "text", "text": "Where is the cat standing?"},
...         ]
...     },
... ]

>>> inputs = processor.apply_chat_template(
...     messages,
...     tokenize=True,
...     return_dict=True,
...     return_tensors="pt",
...     add_generation_prompt=True
... )
>>> # Generate
>>> generate_ids = model.generate(**inputs)
>>> processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
```

## Dots3NoteTextModel[[transformers.Dots3NoteTextModel]]

#### transformers.Dots3NoteTextModel[[transformers.Dots3NoteTextModel]]

```python
transformers.Dots3NoteTextModel(config: Dots3NoteTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L694)

**Parameters:**

config ([Dots3NoteTextConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteTextConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dots3 Note Text Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Dots3NoteTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L712)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.23.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.23.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.23.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `MoeModelOutputWithPast` or `tuple(torch.Tensor)`

A `MoeModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) and inputs.

The [Dots3NoteTextModel](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteTextModel) forward method, overrides the `__call__` special method.

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

## Dots3NoteVisionModel[[transformers.Dots3NoteVisionModel]]

#### transformers.Dots3NoteVisionModel[[transformers.Dots3NoteVisionModel]]

```python
transformers.Dots3NoteVisionModel(config: Dots3NoteVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1508)

**Parameters:**

config ([Dots3NoteVisionConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteVisionConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dots3 Note Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Dots3NoteVisionModel.forward]]

```python
forward(pixel_values: Tensor, grid_thw: Tensor, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1534)

**Parameters:**

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See [Qwen2VLImageProcessor.__call__()](/docs/transformers/v5.23.0/en/model_doc/molmo2#transformers.Molmo2ImageProcessor.__call__) for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

grid_thw (`torch.Tensor`) : Temporal, height, and width patch-grid dimensions for each input.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) and inputs.

The [Dots3NoteVisionModel](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteVisionModel) forward method, overrides the `__call__` special method.

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

## Dots3NoteAudioModel[[transformers.Dots3NoteAudioModel]]

#### transformers.Dots3NoteAudioModel[[transformers.Dots3NoteAudioModel]]

```python
transformers.Dots3NoteAudioModel(config: Dots3NoteAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1129)

**Parameters:**

config ([Dots3NoteAudioConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Dots3 Note Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.23.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Dots3NoteAudioModel.forward]]

```python
forward(input_features: Tensor, chunk_sample_lengths: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/modeling_dots3_note.py#L1142)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`) : The tensors corresponding to the input audio features. Audio features can be obtained using [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor). See `Dots3NoteFeatureExtractor.__call__()` for details ([Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) uses [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor) for processing audios).

chunk_sample_lengths (`torch.LongTensor` of shape `(num_chunks,)`) : Number of waveform samples in each audio chunk.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.23.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Dots3NoteConfig](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteConfig)) and inputs.

The [Dots3NoteAudioModel](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteAudioModel) forward method, overrides the `__call__` special method.

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

## Dots3NoteProcessor[[transformers.Dots3NoteProcessor]]

#### transformers.Dots3NoteProcessor[[transformers.Dots3NoteProcessor]]

```python
transformers.Dots3NoteProcessor(image_processor = None, tokenizer = None, video_processor = None, feature_extractor = None, chat_template = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/processing_dots3_note.py#L25)

**Parameters:**

image_processor (`Qwen2VLImageProcessor`) : The image processor is a required input.

tokenizer (`Qwen2Tokenizer`) : The tokenizer is a required input.

video_processor (`Qwen3VLVideoProcessor`) : The video processor is a required input.

feature_extractor (`Dots3NoteFeatureExtractor`) : The feature extractor is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

Constructs a Dots3NoteProcessor which wraps a image processor, a tokenizer, a video processor, and a feature extractor into a single processor.

[Dots3NoteProcessor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteProcessor) offers all the functionalities of [Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), [Qwen2Tokenizer](/docs/transformers/v5.23.0/en/model_doc/qwen2#transformers.Qwen2Tokenizer), [Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor), and [Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor). See the
[~Qwen2VLImageProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor), [~Qwen2Tokenizer](/docs/transformers/v5.23.0/en/model_doc/qwen2#transformers.Qwen2Tokenizer), [~Qwen3VLVideoProcessor](/docs/transformers/v5.23.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor), and [~Dots3NoteFeatureExtractor](/docs/transformers/v5.23.0/en/model_doc/dots3_note#transformers.Dots3NoteFeatureExtractor) for more information.

#### __call__[[transformers.Dots3NoteProcessor.__call__]]

```python
__call__(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, text: str | list[str] | list[list[str]] | None = None, videos: typing.Union[list['PIL.Image.Image'], numpy.ndarray, ForwardRef('torch.Tensor'), list[numpy.ndarray], list['torch.Tensor'], list[list['PIL.Image.Image']], list[list[numpy.ndarray]], list[list['torch.Tensor']], transformers.video_utils.URL, list[transformers.video_utils.URL], list[list[transformers.video_utils.URL]], transformers.video_utils.Path, list[transformers.video_utils.Path], list[list[transformers.video_utils.Path]], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/processing_utils.py#L670)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

text (`Union[str, list[str], list[list[str]]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

videos (`Union[list[PIL.Image.Image], numpy.ndarray, torch.Tensor, list[numpy.ndarray], list[torch.Tensor], list[list[PIL.Image.Image]], list[list[numpy.ndarray]], list[list[torch.Tensor]], ~video_utils.URL, list[~video_utils.URL], list[list[~video_utils.URL]], ~video_utils.Path, list[~video_utils.Path], list[list[~video_utils.Path]]]`, *optional*) : Video to preprocess. Expects a single or batch of videos with pixel values ranging from 0 to 255. If passing in videos with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_mm_token_type_ids (`bool`, *kwargs*) : Whether to return multimodal token type ids indicating multimodal placeholder token positions.

return_text_replacement_offsets (`bool`, *kwargs*) : Whether to return character offsets for each multimodal placeholder and its replacement.

return_tensors (`str` or [TensorType](/docs/transformers/v5.23.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.23.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

## Dots3NoteFeatureExtractor[[transformers.Dots3NoteFeatureExtractor]]

#### transformers.Dots3NoteFeatureExtractor[[transformers.Dots3NoteFeatureExtractor]]

```python
transformers.Dots3NoteFeatureExtractor(feature_size: int = 128, sampling_rate: int = 16000, padding_value: float = 0.0, n_fft: int = 400, hop_length: int = 160, chunk_length: int = 60, chunk_seconds: int | None = None, return_attention_mask: bool = False, dither: float = 0.0, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.23.0/src/transformers/models/dots3_note/feature_extraction_dots3_note.py#L36)

Convert 16 kHz mono waveforms into Dots 3 Note Preview log-mel chunks.

### MobileNet V2
https://huggingface.co/docs/transformers/v5.23.0/model_doc/mobilenet_v2.md
