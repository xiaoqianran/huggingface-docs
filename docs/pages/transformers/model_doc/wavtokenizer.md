# WavTokenizer

## Overview

[WavTokenizer](https://huggingface.co/papers/2408.16532) encodes 24 kHz mono audio into discrete tokens using a
single codebook, at 40 or 75 tokens per second depending on the checkpoint. It can also reconstruct audio
from those tokens.

This implementation supports inference with [WavTokenizerModel](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerModel) for encoding and decoding, or
[WavTokenizerEncoderModel](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerEncoderModel) for encoding only. The 40-token-per-second variant supplies the audio tokenizer
for [Apertus 1.5](./apertus1p5). Training the codec is not supported.

This model was contributed by the [Swiss AI Initiative](https://huggingface.co/swiss-ai).
The original implementation is available on [GitHub](https://github.com/jishengpeng/WavTokenizer).

## Available checkpoints

These checkpoints are ready to load with `from_pretrained`:

| Checkpoint | Domain | Tokens per second |
|---|---|---:|
| [Small, 40 tokens/s](https://huggingface.co/swiss-ai/wavtokenizer-small-speech-40token) | Speech | 40 |
| [Small, 75 tokens/s](https://huggingface.co/swiss-ai/wavtokenizer-small-speech-75token) | Speech | 75 |
| [Medium](https://huggingface.co/swiss-ai/wavtokenizer-medium-speech-75token) | Speech | 75 |
| [Medium v2](https://huggingface.co/swiss-ai/wavtokenizer-medium-speech-75token-v2) | Speech | 75 |
| [Medium](https://huggingface.co/swiss-ai/wavtokenizer-medium-music-audio-75token) | Music/audio | 75 |
| [Medium v2](https://huggingface.co/swiss-ai/wavtokenizer-medium-music-audio-75token-v2) | Music/audio | 75 |
| [Large unified](https://huggingface.co/swiss-ai/wavtokenizer-large-unify-40token) | Unified | 40 |
| [Large v2](https://huggingface.co/swiss-ai/wavtokenizer-large-speech-75token-v2) | Speech | 75 |

## Usage example

Load a speech clip, resample it to the checkpoint's sampling rate, and encode and reconstruct it:

```python
import torch
from transformers import AutoFeatureExtractor, WavTokenizerModel
from transformers.audio_utils import load_audio

model_id = "swiss-ai/wavtokenizer-large-unify-40token"
model = WavTokenizerModel.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/belinda.wav",
    sampling_rate=feature_extractor.sampling_rate,
)
inputs = feature_extractor(audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
with torch.no_grad():
    audio_codes = model.encode(**inputs).audio_codes
    reconstruction = model.decode(audio_codes).audio_values[..., : len(audio)]
```

## Usage notes

- **Audio inputs:** The feature extractor expects mono audio at its configured sampling rate; it does not
  resample or downmix. The example uses `load_audio` to handle both. Decoding can add trailing samples due to
  internal padding, so the example trims the reconstruction to the original length.
- **Batching:** Padding clips of different lengths can change their codes. Encode clips separately when
  you need the same codes as processing each clip on its own.
- **Precision:** Load and run the tokenizer in `float32` (the default). Lower precision can change code
  assignments.

## WavTokenizerConfig[[transformers.WavTokenizerConfig]]

#### transformers.WavTokenizerConfig[[transformers.WavTokenizerConfig]]

```python
transformers.WavTokenizerConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, sampling_rate: int = 24000, audio_channels: int = 1, num_filters: int = 32, upsampling_ratios: list[int] | tuple[int, ...] = (6, 5, 5, 4), num_residual_layers: int = 1, dilation_growth_rate: int = 2, compress: int = 2, use_conv_shortcut: bool = True, use_causal_conv: bool = False, pad_mode: str = 'reflect', norm_type: str = 'weight_norm', kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3, num_lstm_layers: int = 2, hidden_size: int = 512, codebook_size: int = 4096, codebook_dim: int = 512, decoder_hidden_size: int = 768, decoder_intermediate_size: int = 2304, decoder_num_layers: int = 12, adanorm_num_embeddings: int = 4, decoder_attention_num_groups: int = 32, norm_eps: float = 1e-06)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/configuration_wavtokenizer.py#L26)

**Parameters:**

sampling_rate (`int`, *optional*, defaults to `24000`) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

audio_channels (`int`, *optional*, defaults to 1) : Number of channels in the audio data. Only mono (1) is supported.

num_filters (`int`, *optional*, defaults to 32) : Number of convolution kernels of the first `WavTokenizerConv1d` down sampling layer.

upsampling_ratios (`list[int]`, *optional*, defaults to `[6, 5, 5, 4]`) : Kernel size and stride ratios. The encoder uses downsampling ratios instead of upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here that must match the decoder order. The product (600 by default) is the hop length: the number of audio samples represented by a single code.

num_residual_layers (`int`, *optional*, defaults to 1) : Number of residual layers in each encoder stage.

dilation_growth_rate (`int`, *optional*, defaults to 2) : How much to increase the dilation with each residual layer in the encoder.

compress (`int`, *optional*, defaults to 2) : Reduced dimensionality in residual branches.

use_conv_shortcut (`bool`, *optional*, defaults to `True`) : Whether to use a convolutional layer as the (skip) shortcut in the residual blocks. If False, an identity function will be used, giving a generic residual connection.

use_causal_conv (`bool`, *optional*, defaults to `False`) : Whether to use fully causal convolution.

pad_mode (`str`, *optional*, defaults to `"reflect"`) : Padding mode for the convolutions.

norm_type (`str`, *optional*, defaults to `"weight_norm"`) : Normalization method for the encoder convolutions. Should be in `["weight_norm", "time_group_norm"]`.

kernel_size (`int`, *optional*, defaults to 7) : Kernel size for the initial encoder convolution.

last_kernel_size (`int`, *optional*, defaults to 7) : Kernel size for the last encoder convolution.

residual_kernel_size (`int`, *optional*, defaults to 3) : Kernel size for the residual layers of the encoder.

num_lstm_layers (`int`, *optional*, defaults to 2) : Number of LSTM layers at the end of the encoder.

hidden_size (`int`, *optional*, defaults to 512) : Dimensionality of the encoder output (the latent space of the quantizer).

codebook_size (`int`, *optional*, defaults to 4096) : Number of discrete codes that make up the single VQ codebook.

codebook_dim (`int`, *optional*, defaults to 512) : Dimension of the codebook vectors. Must match `hidden_size` (no projections are used).

decoder_hidden_size (`int`, *optional*, defaults to 768) : Dimensionality of the Vocos-style decoder backbone.

decoder_intermediate_size (`int`, *optional*, defaults to 2304) : Dimensionality of the pointwise convolutions inside the decoder ConvNeXt blocks.

decoder_num_layers (`int`, *optional*, defaults to 12) : Number of ConvNeXt blocks in the decoder backbone.

adanorm_num_embeddings (`int`, *optional*, defaults to 4) : Number of condition embeddings of the decoder's adaptive layer norms. Conditioning is a training-time feature (one embedding per bandwidth); inference always uses embedding 0.

decoder_attention_num_groups (`int`, *optional*, defaults to 32) : Number of groups of the decoder GroupNorm layers.

norm_eps (`float`, *optional*, defaults to 1e-06) : Epsilon of the decoder normalization layers.

This is the configuration class to store the configuration of a WavTokenizerModel. It is used to instantiate a Wavtokenizer
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [swiss-ai/wavtokenizer-large-unify-40token](https://huggingface.co/swiss-ai/wavtokenizer-large-unify-40token)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import WavTokenizerConfig, WavTokenizerModel

>>> # Initializing configuration
>>> configuration = WavTokenizerConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = WavTokenizerModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## WavTokenizerEncoderConfig[[transformers.WavTokenizerEncoderConfig]]

#### transformers.WavTokenizerEncoderConfig[[transformers.WavTokenizerEncoderConfig]]

```python
transformers.WavTokenizerEncoderConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, sampling_rate: int = 24000, audio_channels: int = 1, num_filters: int = 32, upsampling_ratios: list[int] | tuple[int, ...] = (6, 5, 5, 4), num_residual_layers: int = 1, dilation_growth_rate: int = 2, compress: int = 2, use_conv_shortcut: bool = True, use_causal_conv: bool = False, pad_mode: str = 'reflect', norm_type: str = 'weight_norm', kernel_size: int = 7, last_kernel_size: int = 7, residual_kernel_size: int = 3, num_lstm_layers: int = 2, hidden_size: int = 512, codebook_size: int = 4096, codebook_dim: int = 512, decoder_hidden_size: int = 768, decoder_intermediate_size: int = 2304, decoder_num_layers: int = 12, adanorm_num_embeddings: int = 4, decoder_attention_num_groups: int = 32, norm_eps: float = 1e-06)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/configuration_wavtokenizer.py#L159)

Configuration of [WavTokenizerEncoderModel](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerEncoderModel), the encoder and quantizer of WavTokenizer. It shares every field
with [WavTokenizerConfig](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerConfig) (see its documentation for the arguments), so it can be loaded from a full
WavTokenizer checkpoint. Decoder-specific fields are unused by the encoder model.

## WavTokenizerFeatureExtractor[[transformers.WavTokenizerFeatureExtractor]]

#### transformers.WavTokenizerFeatureExtractor[[transformers.WavTokenizerFeatureExtractor]]

```python
transformers.WavTokenizerFeatureExtractor(feature_size: int = 1, sampling_rate: int = 24000, padding_value: float = 0.0, hop_length: int = 600, normalize_audio: bool = False, target_peak_dB_FS: float = -3.0, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/feature_extraction_wavtokenizer.py#L27)

**Parameters:**

feature_size (`int`, *optional*, defaults to 1) : The feature dimension of the extracted features. Only mono (1) is supported.

sampling_rate (`int`, *optional*, defaults to 24000) : The sampling rate at which the audio waveform should be digitalized, expressed in hertz (Hz).

padding_value (`float`, *optional*, defaults to 0.0) : The value that is used for padding.

hop_length (`int`, *optional*, defaults to 600) : Number of audio samples represented by a single audio code.

normalize_audio (`bool`, *optional*, defaults to `False`) : Whether to peak-normalize each clip to `target_peak_dB_FS` before padding. The original WavTokenizer does not normalize; Apertus 1.5 enables it.

target_peak_dB_FS (`float`, *optional*, defaults to -3.0) : Target peak level in dB FS used when `normalize_audio` is `True`.

Constructs a WavTokenizer feature extractor for mono audio at the configured sampling rate. Audio is not resampled
or downmixed.

#### __call__[[transformers.WavTokenizerFeatureExtractor.__call__]]

```python
__call__(audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], padding: bool | str | transformers.utils.generic.PaddingStrategy | None = None, truncation: bool | None = False, max_length: int | None = None, return_tensors: str | transformers.utils.generic.TensorType | None = None, sampling_rate: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/feature_extraction_wavtokenizer.py#L72)

**Parameters:**

audio (`np.ndarray`, `torch.Tensor`, `list[float]`, `list[np.ndarray]`, `list[torch.Tensor]`, `list[list[float]]`) : The sequence or batch of sequences to be processed. Each sequence must be mono audio of shape `(num_samples,)`, already sampled at this feature extractor's `sampling_rate`. Audio is not resampled or downmixed.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.24.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `True`) : Padding strategy: `True` or `'longest'` pads a batch to its longest sequence (single inputs stay unpadded), `'max_length'` pads to `max_length`, `False` or `'do_not_pad'` disables padding.

truncation (`bool`, *optional*, defaults to `False`) : Whether to truncate sequences longer than `max_length` to `max_length`.

max_length (`int`, *optional*) : Target length for `'max_length'` padding and for `truncation`.

return_tensors (`str` or [TensorType](/docs/transformers/v5.24.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, return tensors instead of a list of numpy arrays: `'pt'` for PyTorch `torch.Tensor`, `'np'` for NumPy `np.ndarray`.

sampling_rate (`int`, *optional*) : Sampling rate of the `audio` input; pass it so a mismatch with the configured `sampling_rate` raises an error instead of failing silently.

**Returns:** [BatchFeature](/docs/transformers/v5.24.0/en/main_classes/feature_extractor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v5.24.0/en/main_classes/feature_extractor#transformers.BatchFeature) with the following fields:

- **input_values** -- Audio waveforms to be fed to the model, as a list of arrays of shape
  `(1, num_samples)`, or a tensor of shape `(batch_size, 1, num_samples)` when `return_tensors` is set.
- **padding_mask** -- Mask indicating valid audio samples (1) versus padding (0), of shape
  `(batch_size, num_samples)`.

Featurize and prepare one or several audio sequence(s) for the model.

## WavTokenizerEncoderModel[[transformers.WavTokenizerEncoderModel]]

#### transformers.WavTokenizerEncoderModel[[transformers.WavTokenizerEncoderModel]]

```python
transformers.WavTokenizerEncoderModel(config: WavTokenizerEncoderConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L540)

**Parameters:**

config ([WavTokenizerEncoderConfig](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerEncoderConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Encoder and quantizer of WavTokenizer for converting audio waveforms to discrete codes.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### encode[[transformers.WavTokenizerEncoderModel.encode]]

```python
encode(input_values: Tensor, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L554)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`) : Input audio waveform. Arbitrary non-zero lengths are supported; the encoder pads internally and emits `ceil(sequence_length / hop_length)` codes.

padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*) : Mask describing padding in `input_values`; used to compute `audio_codes_mask`.

**Returns:** `WavTokenizerEncoderOutput` or `tuple(torch.Tensor)`

A `WavTokenizerEncoderOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

- **audio_codes** (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Discrete code indices computed using `model.encode`. These represent the compressed, quantized form of the
  input audio signal (one codebook, `ceil(num_samples / hop_length)` codes).
- **audio_codes_mask** (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.

#### forward[[transformers.WavTokenizerEncoderModel.forward]]

```python
forward(input_values: Tensor, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L589)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`) : Input audio waveform.

padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*) : Padding mask used to compute `audio_codes_mask`.

**Returns:** `WavTokenizerEncoderOutput` or `tuple(torch.Tensor)`

A `WavTokenizerEncoderOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([WavTokenizerConfig](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerConfig)) and inputs.

The [WavTokenizerEncoderModel](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerEncoderModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **audio_codes** (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Discrete code indices computed using `model.encode`. These represent the compressed, quantized form of the
  input audio signal (one codebook, `ceil(num_samples / hop_length)` codes).
- **audio_codes_mask** (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.

## WavTokenizerModel[[transformers.WavTokenizerModel]]

#### transformers.WavTokenizerModel[[transformers.WavTokenizerModel]]

```python
transformers.WavTokenizerModel(config: WavTokenizerConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L610)

**Parameters:**

config ([WavTokenizerConfig](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Inference-only, single-codebook WavTokenizer neural audio codec.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### decode[[transformers.WavTokenizerModel.decode]]

```python
decode(audio_codes: Tensor, bandwidth_id: int = 0, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L639)

**Parameters:**

audio_codes (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`) : Discrete code indices computed using `model.encode`. One or more codes are supported; each code decodes to `hop_length` audio samples.

bandwidth_id (`int`, *optional*, defaults to 0) : Condition embedding id of the decoder's adaptive layer norms. Always 0 at inference.

**Returns:** `WavTokenizerDecoderOutput` or `tuple(torch.Tensor)`

A `WavTokenizerDecoderOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

- **audio_values** (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`, *optional*) -- Decoded audio waveform values in the time domain (`codes_length * hop_length` samples).

#### encode[[transformers.WavTokenizerModel.encode]]

```python
encode(input_values: Tensor, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L622)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`) : Input audio waveform. Arbitrary non-zero lengths are supported; the encoder pads internally and emits `ceil(sequence_length / hop_length)` codes.

padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*) : Mask describing padding in `input_values`; used to compute `audio_codes_mask`.

**Returns:** `WavTokenizerEncoderOutput` or `tuple(torch.Tensor)`

A `WavTokenizerEncoderOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

- **audio_codes** (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Discrete code indices computed using `model.encode`. These represent the compressed, quantized form of the
  input audio signal (one codebook, `ceil(num_samples / hop_length)` codes).
- **audio_codes_mask** (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.

#### forward[[transformers.WavTokenizerModel.forward]]

```python
forward(input_values: Tensor, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.0/src/transformers/models/wavtokenizer/modeling_wavtokenizer.py#L660)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`) : Input audio waveform.

padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)` or `(batch_size, 1, sequence_length)`, *optional*) : Mask indicating valid audio samples (1) versus padding (0), used to compute `audio_codes_mask`.

**Returns:** `WavTokenizerOutput` or `tuple(torch.Tensor)`

A `WavTokenizerOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [WavTokenizerModel](/docs/transformers/v5.24.0/en/model_doc/wavtokenizer#transformers.WavTokenizerModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **audio_values** (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`, *optional*) -- Decoded audio waveform values in the time domain, obtained using the decoder part of WavTokenizer.
- **audio_codes** (`torch.LongTensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Discrete code indices computed using `model.encode`.
- **audio_codes_mask** (`torch.Tensor` of shape `(batch_size, 1, codes_length)`, *optional*) -- Downsampled `padding_mask` indicating valid audio codes in `audio_codes`.

### SqueezeBERT
https://huggingface.co/docs/transformers/v5.24.0/model_doc/squeezebert.md
