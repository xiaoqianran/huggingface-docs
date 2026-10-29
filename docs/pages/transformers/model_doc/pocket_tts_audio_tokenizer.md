# Pocket TTS audio tokenizer

## Overview

The audio tokenizer of [Pocket TTS](https://huggingface.co/kyutai/pocket-tts): a codec with [Mimi](./mimi)'s architecture (SEANet convolutions and a small causal transformer on each side) whose residual vector quantizer is replaced by a continuous 32-dimensional bottleneck, the VAE-GAN of [Continuous Audio Language Models](https://huggingface.co/papers/2509.06926). It turns 24 kHz audio into latents at 12.5 Hz and back. Its encoder and decoder can be loaded on their own ([PocketTTSAudioTokenizerEncoderModel](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerEncoderModel), [PocketTTSAudioTokenizerDecoderModel](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerDecoderModel)). It is the `audio_tokenizer` of [PocketTTSProcessor](/docs/transformers/v5.21.1/en/model_doc/pocket_tts#transformers.PocketTTSProcessor), shared by every Pocket TTS checkpoint: the processor encodes voice prompts with it and decodes the generated latents.

This model was contributed by [Eustache Le Bihan](https://huggingface.co/eustlb). The original code is [kyutai-labs/pocket-tts](https://github.com/kyutai-labs/pocket-tts).

## Usage

```python
from datasets import Audio, load_dataset
from transformers import AutoFeatureExtractor, AutoModel

model = AutoModel.from_pretrained("eustlb/pocket-tts-audio-tokenizer", device_map="auto")
feature_extractor = AutoFeatureExtractor.from_pretrained("eustlb/pocket-tts-audio-tokenizer")

dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
dataset = dataset.cast_column("audio", Audio(sampling_rate=24000))
inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=24000, return_tensors="pt").to(model.device)

latents = model.encode(**inputs).latents
audio_values = model.decode(latents).audio_values
```

## PocketTTSAudioTokenizerConfig[[transformers.PocketTTSAudioTokenizerConfig]]

#### transformers.PocketTTSAudioTokenizerConfig[[transformers.PocketTTSAudioTokenizerConfig]]

```python
transformers.PocketTTSAudioTokenizerConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, sampling_rate: int = 24000, audio_channels: int = 1, hidden_size: int = 512, num_filters: int = 64, num_residual_layers: int = 1, upsampling_ratios: list[int] | None = None, kernel_size: int = 7, last_kernel_size: int = 3, residual_kernel_size: int = 3, dilation_growth_rate: int = 2, use_causal_conv: bool = True, pad_mode: str = 'constant', compress: int = 2, trim_right_ratio: float = 1.0, use_conv_shortcut: bool = False, upsample_groups: int = 512, num_hidden_layers: int = 2, intermediate_size: int = 2048, num_attention_heads: int = 8, num_key_value_heads: int = 8, head_dim: int | None = None, hidden_act: str = 'gelu_pytorch_tanh', max_position_embeddings: int = 8000, initializer_range: float = 0.02, norm_eps: float = 1e-05, use_cache: bool = False, use_streaming: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, sliding_window: int = 250, attention_dropout: float | int = 0.0, layer_scale_initial_scale: float = 0.01, attention_bias: bool = False, latent_stride: int = 16, latent_dim: int = 32)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/configuration_pocket_tts_audio_tokenizer.py#L38)

**Parameters:**

sampling_rate (`int`, *optional*, defaults to `24000`) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

audio_channels (`int`, *optional*, defaults to 1) : Number of channels in the audio data; the released checkpoints, and this implementation, are mono.

hidden_size (`int`, *optional*, defaults to `512`) : Dimension of the hidden representations.

num_filters (`int`, *optional*, defaults to 64) : Number of convolution kernels of the first `PocketTTSAudioTokenizerConv1d` layer of the encoder.

num_residual_layers (`int`, *optional*, defaults to 1) : Number of residual layers.

upsampling_ratios (`list[int]`, *optional*) : Kernel size and stride ratios of the decoder's transposed convolutions; the encoder uses them in reverse order. If not specified, defaults to `[6, 5, 4]`.

kernel_size (`int`, *optional*, defaults to `7`) : The size of the convolutional kernel.

last_kernel_size (`int`, *optional*, defaults to 3) : Kernel size for the last convolution layer.

residual_kernel_size (`int`, *optional*, defaults to 3) : Kernel size for the residual layers.

dilation_growth_rate (`int`, *optional*, defaults to 2) : How much to increase the dilation with each layer.

use_causal_conv (`bool`, *optional*, defaults to `True`) : Whether to use fully causal convolution.

pad_mode (`str`, *optional*, defaults to `"constant"`) : Padding mode for the convolutions.

compress (`int`, *optional*, defaults to 2) : Reduced dimensionality in residual branches.

trim_right_ratio (`float`, *optional*, defaults to 1.0) : Ratio for trimming at the right of the transposed convolution under the `use_causal_conv = True` setup. If equal to 1.0, it means that all the trimming is done at the right.

use_conv_shortcut (`bool`, *optional*, defaults to `False`) : Whether to use a convolution as the skip connection of the SEANet residual blocks; the released checkpoints do not.

upsample_groups (`int`, *optional*, defaults to 512) : Number of groups of the transposed convolution that upsamples the latents to the SEANet rate.

num_hidden_layers (`int`, *optional*, defaults to `2`) : Number of hidden layers in the Transformer decoder.

intermediate_size (`int`, *optional*, defaults to `2048`) : Dimension of the MLP representations.

num_attention_heads (`int`, *optional*, defaults to `8`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

head_dim (`int`, *optional*) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `8000`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

use_cache (`bool`, *optional*, defaults to `False`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

use_streaming (`bool`, *optional*, defaults to `False`) : Whether the encoder streams by default: it then returns the padding cache and key-value cache to pass to the next call.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

sliding_window (`int`, *optional*, defaults to `250`) : Sliding window attention window size. If `None`, no sliding window is applied.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

layer_scale_initial_scale (`float`, *optional*, defaults to `0.01`) : Scale to use in the self-attention layers. 0.1 for base, 1e-6 for large. Set 0 to disable layer scale.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

latent_stride (`int`, *optional*, defaults to 16) : Number of SEANet frames per latent frame: the stride of the encoder's convolution that downsamples the SEANet output to the latents, and of the decoder's transposed one that upsamples them back. A latent frame covers `prod(upsampling_ratios) * latent_stride` audio samples (`frame_size`, 1920 for the released checkpoints).

latent_dim (`int`, *optional*, defaults to 32) : Dimension of the continuous latents. Mimi's residual vector quantizer is replaced by this bottleneck: the encoder's downsampling convolution outputs `latent_dim` channels, and the decoder projects them back to `hidden_size` before upsampling.

This is the configuration class to store the configuration of a PocketTTSAudioTokenizerModel. It is used to instantiate a Pocket Tts Audio Tokenizer
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [eustlb/pocket-tts-audio-tokenizer](https://huggingface.co/eustlb/pocket-tts-audio-tokenizer)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import PocketTTSAudioTokenizerConfig, PocketTTSAudioTokenizerModel

>>> # a full codec, and its decoder alone
>>> model = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig())
>>> decoder_only = PocketTTSAudioTokenizerModel(PocketTTSAudioTokenizerConfig(encoder_config=None))
```

## PocketTTSAudioTokenizerEncoderConfig[[transformers.PocketTTSAudioTokenizerEncoderConfig]]

#### transformers.PocketTTSAudioTokenizerEncoderConfig[[transformers.PocketTTSAudioTokenizerEncoderConfig]]

```python
transformers.PocketTTSAudioTokenizerEncoderConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, sampling_rate: int = 24000, audio_channels: int = 1, hidden_size: int = 512, num_filters: int = 64, num_residual_layers: int = 1, upsampling_ratios: list[int] | None = None, kernel_size: int = 7, last_kernel_size: int = 3, residual_kernel_size: int = 3, dilation_growth_rate: int = 2, use_causal_conv: bool = True, pad_mode: str = 'constant', compress: int = 2, trim_right_ratio: float = 1.0, use_conv_shortcut: bool = False, upsample_groups: int = 512, num_hidden_layers: int = 2, intermediate_size: int = 2048, num_attention_heads: int = 8, num_key_value_heads: int = 8, head_dim: int | None = None, hidden_act: str = 'gelu_pytorch_tanh', max_position_embeddings: int = 8000, initializer_range: float = 0.02, norm_eps: float = 1e-05, use_cache: bool = False, use_streaming: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, sliding_window: int = 250, attention_dropout: float | int = 0.0, layer_scale_initial_scale: float = 0.01, attention_bias: bool = False, latent_stride: int = 16, latent_dim: int = 32)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/configuration_pocket_tts_audio_tokenizer.py#L214)

Configuration of the encoder half of PocketTTSAudioTokenizer (SEANet encoder, encoder transformer and downsampling). It shares every
field with [PocketTTSAudioTokenizerConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerConfig) (see its documentation for the arguments), so it can be loaded directly from a full PocketTTSAudioTokenizer
checkpoint.

## PocketTTSAudioTokenizerDecoderConfig[[transformers.PocketTTSAudioTokenizerDecoderConfig]]

#### transformers.PocketTTSAudioTokenizerDecoderConfig[[transformers.PocketTTSAudioTokenizerDecoderConfig]]

```python
transformers.PocketTTSAudioTokenizerDecoderConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, sampling_rate: int = 24000, audio_channels: int = 1, hidden_size: int = 512, num_filters: int = 64, num_residual_layers: int = 1, upsampling_ratios: list[int] | None = None, kernel_size: int = 7, last_kernel_size: int = 3, residual_kernel_size: int = 3, dilation_growth_rate: int = 2, use_causal_conv: bool = True, pad_mode: str = 'constant', compress: int = 2, trim_right_ratio: float = 1.0, use_conv_shortcut: bool = False, upsample_groups: int = 512, num_hidden_layers: int = 2, intermediate_size: int = 2048, num_attention_heads: int = 8, num_key_value_heads: int = 8, head_dim: int | None = None, hidden_act: str = 'gelu_pytorch_tanh', max_position_embeddings: int = 8000, initializer_range: float = 0.02, norm_eps: float = 1e-05, use_cache: bool = False, use_streaming: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, sliding_window: int = 250, attention_dropout: float | int = 0.0, layer_scale_initial_scale: float = 0.01, attention_bias: bool = False, latent_stride: int = 16, latent_dim: int = 32)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/configuration_pocket_tts_audio_tokenizer.py#L242)

Configuration of the decoder half of PocketTTSAudioTokenizer (upsampling, decoder transformer and SEANet decoder). It shares every
field with [PocketTTSAudioTokenizerConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerConfig) (see its documentation for the arguments), so it can be loaded directly from a full PocketTTSAudioTokenizer
checkpoint.

## PocketTTSAudioTokenizerModel[[transformers.PocketTTSAudioTokenizerModel]]

#### transformers.PocketTTSAudioTokenizerModel[[transformers.PocketTTSAudioTokenizerModel]]

```python
transformers.PocketTTSAudioTokenizerModel(config: PocketTTSAudioTokenizerConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L1179)

**Parameters:**

config ([PocketTTSAudioTokenizerConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Pocket TTS audio tokenizer: a Mimi-architecture codec with a continuous 32-dimensional latent at 12.5 Hz (the
VAE-GAN of Continuous Audio Language Models), turning 24 kHz audio into latents and back.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### encode[[transformers.PocketTTSAudioTokenizerModel.encode]]

```python
encode(input_values: FloatTensor, padding_mask: typing.Optional[torch.Tensor] = None, encoder_past_key_values: transformers.cache_utils.Cache | None = None, padding_cache: transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer.PocketTTSAudioTokenizerConv1dPaddingCache | None = None, use_streaming: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L1189)

**Parameters:**

input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`) : Audio waveform, as returned by [EncodecFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/encodec#transformers.EncodecFeatureExtractor), which zero-pads it to a multiple of `config.frame_size` samples as the original does.

padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*) : Mask of the audio samples (`1`) as opposed to padding (`0`), from which `latents_mask` is computed.

encoder_past_key_values (`Cache`, *optional*) : Key-value cache of the encoder transformer, returned by the previous call when streaming.

padding_cache (`PocketTTSAudioTokenizerConv1dPaddingCache`, *optional*) : Padding cache of the encoder's causal convolutions, returned by the previous call when streaming.

use_streaming (`bool`, *optional*) : Whether to encode in streaming mode, returning the caches to pass to the next call.

Encodes audio into continuous latents of shape `(batch_size, num_frames, latent_dim)`: the latents of
[PocketTTSAudioTokenizerEncoderModel.forward()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerEncoderModel.forward), frames first.

#### decode[[transformers.PocketTTSAudioTokenizerModel.decode]]

```python
decode(latents: FloatTensor, decoder_past_key_values: transformers.cache_utils.Cache | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L1244)

**Parameters:**

latents (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`) : Continuous latents to decode, one frame per `config.frame_size` audio samples.

decoder_past_key_values (`Cache`, *optional*) : Key-value cache of the decoder transformer, returned by the previous call when decoding in chunks.

Decodes continuous latents into audio: the latents are projected to `hidden_size`, then decoded by
[PocketTTSAudioTokenizerDecoderModel.forward()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerDecoderModel.forward).

#### forward[[transformers.PocketTTSAudioTokenizerModel.forward]]

```python
forward(input_values: FloatTensor, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L1272)

**Parameters:**

input_values (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`) : Audio waveform, as returned by [EncodecFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/encodec#transformers.EncodecFeatureExtractor).

padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*) : Mask of the audio samples (`1`) as opposed to padding (`0`).

**Returns:** `PocketTTSAudioTokenizerOutput` or `tuple(torch.FloatTensor)`

A `PocketTTSAudioTokenizerOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PocketTTSAudioTokenizerConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerConfig)) and inputs.

The [PocketTTSAudioTokenizerModel](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **latents** (`torch.FloatTensor` of shape `(batch_size, num_frames, latent_dim)`) -- Continuous latents of the audio.
- **latents_mask** (`torch.LongTensor` of shape `(batch_size, num_frames)`, *optional*) -- Mask of the latent frames computed from audio samples (`1`) rather than padding (`0`).
- **audio_values** (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`) -- Audio waveform reconstructed from the latents.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) -- Hidden states of the encoder transformer at the output of each layer plus its input.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) -- Attention weights of each encoder transformer layer.

## PocketTTSAudioTokenizerEncoderModel[[transformers.PocketTTSAudioTokenizerEncoderModel]]

#### transformers.PocketTTSAudioTokenizerEncoderModel[[transformers.PocketTTSAudioTokenizerEncoderModel]]

```python
transformers.PocketTTSAudioTokenizerEncoderModel(config: PocketTTSAudioTokenizerEncoderConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L879)

**Parameters:**

config ([PocketTTSAudioTokenizerEncoderConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerEncoderConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The encoder half of PocketTTSAudioTokenizer: SEANet encoder, encoder transformer and downsampling, from audio to continuous latents.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PocketTTSAudioTokenizerEncoderModel.forward]]

```python
forward(input_values: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, padding_mask: typing.Optional[torch.Tensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, padding_cache: transformers.models.pocket_tts_audio_tokenizer.modeling_pocket_tts_audio_tokenizer.PocketTTSAudioTokenizerConv1dPaddingCache | None = None, use_streaming: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L940)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`) : Float values of the input audio waveform.

padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`, *optional*) : Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0 for *masked*.

past_key_values (`Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks) of the encoder transformer, used to speed up streaming encoding.

padding_cache (`PocketTTSAudioTokenizerConv1dPaddingCache`, *optional*) : Padding cache of the causal convolutions, used for streaming encoding.

use_streaming (`bool`, *optional*) : Whether to encode in streaming mode, returning the caches to pass to the next call. 

Encodes the input audio waveform into continuous latents of shape `(batch_size, latent_dim, frames)`. --

**Returns:** `PocketTTSAudioTokenizerEncoderModelOutput` or `tuple(torch.FloatTensor)`

A `PocketTTSAudioTokenizerEncoderModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PocketTTSAudioTokenizerConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerConfig)) and inputs.

The [PocketTTSAudioTokenizerEncoderModel](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **latents** (`torch.FloatTensor` of shape `(batch_size, latent_dim, frames)`) -- Continuous latents of the audio, one frame per `config.frame_size` audio samples.
- **past_key_values** (`Cache`, *optional*) -- Pre-computed hidden-states (key and values in the self-attention blocks) of the encoder transformer, returned
  when streaming.
- **padding_cache** (`PocketTTSAudioTokenizerConv1dPaddingCache`, *optional*) -- Padding cache for PocketTTSAudioTokenizerConv1d causal convolutions in order to support streaming via cache padding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) -- Hidden states of the encoder transformer at the output of each layer plus its input.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) -- Attention weights of each encoder transformer layer.

## PocketTTSAudioTokenizerDecoderModel[[transformers.PocketTTSAudioTokenizerDecoderModel]]

#### transformers.PocketTTSAudioTokenizerDecoderModel[[transformers.PocketTTSAudioTokenizerDecoderModel]]

```python
transformers.PocketTTSAudioTokenizerDecoderModel(config: PocketTTSAudioTokenizerDecoderConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L1109)

**Parameters:**

config ([PocketTTSAudioTokenizerDecoderConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerDecoderConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The decoder half of PocketTTSAudioTokenizer: upsampling, decoder transformer and SEANet decoder, from continuous latents to audio.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.PocketTTSAudioTokenizerDecoderModel.forward]]

```python
forward(latents: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, padding_mask: typing.Optional[torch.Tensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/pocket_tts_audio_tokenizer/modeling_pocket_tts_audio_tokenizer.py#L1136)

**Parameters:**

latents (`torch.FloatTensor` of shape `(batch_size, hidden_size, frames)`) : Continuous latents to decode, as computed by [PocketTTSAudioTokenizerModel.decode()](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerModel.decode).

padding_mask (`torch.Tensor` of shape `(batch_size, channels, sequence_length)`, *optional*) : Indicates which inputs are to be ignored due to padding, where elements are either 1 for *not masked* or 0 for *masked*.

past_key_values (`Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks) of the decoder transformer, used to speed up streaming decoding. 

Decodes the given frames into an output audio waveform. Note that the output might be a bit bigger than the --

input. In that case, any extra steps at the end can be trimmed. --

**Returns:** `PocketTTSAudioTokenizerDecoderOutput` or `tuple(torch.FloatTensor)`

A `PocketTTSAudioTokenizerDecoderOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([PocketTTSAudioTokenizerConfig](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerConfig)) and inputs.

The [PocketTTSAudioTokenizerDecoderModel](/docs/transformers/v5.21.1/en/model_doc/pocket_tts_audio_tokenizer#transformers.PocketTTSAudioTokenizerDecoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **audio_values** (`torch.FloatTensor` of shape `(batch_size, audio_channels, num_samples)`, *optional*) -- Decoded audio values, obtained using the decoder part of PocketTTSAudioTokenizer.
- **decoder_past_key_values** (`Cache`, *optional*) -- Pre-computed hidden-states (key and values in the self-attention blocks) that can be used to speed up sequential decoding of the decoder transformer.
  This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

  The model will output the same cache format that is fed as input.

  If `past_key_values` are used, the user can optionally input only the last `audio_values` or `audio_codes (those that don't
  have their past key value states given to this model).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True`) -- Hidden states of the decoder transformer at the output of each layer plus its input.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True`) -- Attention weights of each decoder transformer layer.

### MetaCLIP 2
https://huggingface.co/docs/transformers/v5.21.1/model_doc/metaclip_2.md
