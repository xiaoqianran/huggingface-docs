# Nemotron 3 Diarization

## Overview

Nemotron 3 Diarization is an open-weight streaming speaker diarization model designed to determine "who spoke when" in real-world audio. It supports both streaming and offline inference, handles up to eight speakers, and orders speaker outputs by each speaker's first arrival in the input audio.

The model uses the Arrival-Order Speaker Cache (AOSC) [1](https://huggingface.co/papers/2507.18446) and FIFO queue introduced for Streaming Sortformer [1](https://huggingface.co/papers/2507.18446), [2](https://huggingface.co/papers/2409.06656). A single checkpoint supports configurable latency profiles, from an 80 ms input buffer to a 30.4 s offline-style buffer, and configurable output frame resolution in multiples of 10 ms. With chunked inference, the maximum audio duration is not limited.

## Usage

### Offline

```python
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/Nemotron-3-Diarization"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto")

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    sampling_rate=sampling_rate,
)
inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)

with torch.inference_mode():
    logits = model(**inputs).logits  # (1, num_frames, 8), one frame every 10 ms

segments = processor.extract_speaker_dict(logits, inputs.attention_mask)[0]
for segment in segments:
    print(f"speaker_{segment['Speaker']}: {segment['Start']:.2f}s - {segment['End']:.2f}s")
```

### Streaming

Audio arrives chunk by chunk, and each forward takes one chunk: the processor cuts it for its `streaming_mode` and
adds `num_lookahead_frames`, the number of trailing look-ahead frames the model attends to but does not score, since
they open the next chunk. The forward returns the `speaker_cache` to pass to the next call. The last chunk of a
session is extracted with `is_last_audio_chunk=True`: it has no look-ahead, so every remaining frame is scored.

| `streaming_mode`          | Latency¹ |
| ------------------------- | -------- |
| `"low_latency"` (default) | 1.04 s   |
| `"very_low_latency"`      | 0.64 s   |
| `"ultra_low_latency"`     | 0.32 s   |

¹ Audio to wait for before the model runs on a chunk: the chunk plus its look-ahead, excluding compute time.

```python
import torch
from transformers import AutoModelForAudioFrameClassification, AutoProcessor
from transformers.audio_utils import load_audio

model_id = "nvidia/Nemotron-3-Diarization"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto")
processor.set_streaming_mode("low_latency")  # the default, can also be "very_low_latency" and "ultra_low_latency"
print(f"Streaming latency: {processor.streaming_latency_ms} ms")

sampling_rate = processor.feature_extractor.sampling_rate
audio = load_audio(
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/diarization_example.mp3",
    sampling_rate=sampling_rate,
)

def inputs_generator():
    """Yields the processor outputs of each chunk."""
    yield processor(
        audio[: processor.num_samples_first_audio_chunk],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=True,
    )

    mel_frame_idx = processor.num_mel_frames_per_step
    start_idx = processor.audio_chunk_start(mel_frame_idx)
    while (end_idx := start_idx + processor.num_samples_per_audio_chunk) <= audio.shape[0]:
        yield processor(
            audio[start_idx:end_idx],
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
        )
        mel_frame_idx += processor.num_mel_frames_per_step
        start_idx = processor.audio_chunk_start(mel_frame_idx)

    # the audio ended: the frames left in the buffer are the last ones of the session
    yield processor(
        audio[start_idx:],
        sampling_rate=sampling_rate,
        is_streaming=True,
        is_first_audio_chunk=False,
        is_last_audio_chunk=True,
    )

speaker_cache, logits = None, []
with torch.inference_mode():
    for inputs in inputs_generator():
        inputs = inputs.to(model.device, dtype=model.dtype)
        # `inputs` carries `num_lookahead_frames` for every chunk but the last, `speaker_cache` links the chunks
        outputs = model(**inputs, speaker_cache=speaker_cache)
        logits.append(outputs.logits)  # the chunk's frames, without its look-ahead
        speaker_cache = outputs.speaker_cache

logits = torch.cat(logits, dim=1)  # (1, num_frames, 8), one frame every 10 ms
segments = processor.extract_speaker_dict(logits)[0]  # [{"Start": 0.0, "End": 15.43, "Speaker": 0}, ...]
```

### Making it go brrr

The encoder input of a streaming step is `[speaker cache | FIFO | chunk]`, whose length changes as the cache and the
FIFO fill and shrink: `torch.compile` would recompile about a hundred times per session. Padding every step to the
largest window of the mode fixes the shape. Positions restart at zero on every chunk, so right padding does not change
the valid frames:

```python
import torch.nn.functional as F

chunk_length, chunk_right_context = processor.streaming_modes[processor.streaming_mode]
max_window = (
    model.config.streaming_config.speaker_cache_length
    + model.config.streaming_config.fifo_length
    + chunk_length
    + chunk_right_context
)
encoder = model.model
compiled_forward = torch.compile(encoder.forward, mode="reduce-overhead", fullgraph=True, dynamic=False)

def padded_forward(inputs_embeds, attention_mask=None, position_ids=None, **kwargs):
    batch_size, num_frames, _ = inputs_embeds.shape
    if attention_mask is None:
        attention_mask = inputs_embeds.new_ones(batch_size, num_frames, dtype=torch.bool)
    padding = max_window - num_frames
    hidden_states = compiled_forward(
        inputs_embeds=F.pad(inputs_embeds, (0, 0, 0, padding)),
        attention_mask=F.pad(attention_mask.bool(), (0, padding), value=False),
        position_ids=torch.arange(max_window, device=inputs_embeds.device)[None, :],
        **kwargs,
    )
    return hidden_states[:, :num_frames].clone()  # CUDA graphs reuse the output buffer

encoder.forward = padded_forward

# warm up before the session: compiles, then records the CUDA graph, so the first real chunk runs at full speed
with torch.inference_mode():
    for _ in range(3):
        hidden_size = model.config.audio_config.hidden_size
        padded_forward(torch.zeros(1, max_window, hidden_size, device=model.device, dtype=model.dtype))
```

The streaming loop above then compiles once. The offline forward chunks the same way, so the same wrapper applies with
`config.fifo_length`, `config.chunk_length` and `config.chunk_right_context` in `max_window`.

| Speedup vs eager (A100, batch size 1) | float32 | bfloat16 |
| ------------------------------------- | ------- | -------- |
| streaming, per step                   | 1.2x    | 4.4x     |
| offline, 488 s recording              | 1.3x    | 2.8x     |

## Nemotron3DiarizationConfig[[transformers.Nemotron3DiarizationConfig]]

#### transformers.Nemotron3DiarizationConfig[[transformers.Nemotron3DiarizationConfig]]

```python
transformers.Nemotron3DiarizationConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, audio_config: transformers.models.nemotron3_diarization.configuration_nemotron3_diarization.Nemotron3DiarizationAudioConfig | dict | None = None, head_config: transformers.models.nemotron3_diarization.configuration_nemotron3_diarization.Nemotron3DiarizationHeadConfig | dict | None = None, streaming_config: transformers.models.nemotron3_diarization.configuration_nemotron3_diarization.Nemotron3DiarizationStreamingConfig | dict | None = None, chunk_length: int = 340, chunk_right_context: int = 40, fifo_length: int = 40, speaker_cache_update_period: int = 300, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L146)

**Parameters:**

audio_config (`Nemotron3DiarizationAudioConfig` or `dict`, *optional*) : Configuration of the transformer audio encoder. Defaults to `Nemotron3DiarizationAudioConfig()`.

head_config (`Nemotron3DiarizationHeadConfig` or `dict`, *optional*) : Configuration of the speaker head. Defaults to `Nemotron3DiarizationHeadConfig()`.

streaming_config (`Nemotron3DiarizationStreamingConfig` or `dict`, *optional*) : Speaker-cache policy, and the FIFO sizes of streaming mode. Defaults to `Nemotron3DiarizationStreamingConfig()`.

chunk_length (`int`, *optional*, defaults to 340) : Offline mode: number of encoder frames per chunk when a whole recording is diarized in one forward. In streaming mode the chunk is the input of each forward.

chunk_right_context (`int`, *optional*, defaults to 40) : Offline mode: number of look-ahead encoder frames each chunk takes from the following ones. In streaming mode the look-ahead is `num_lookahead_frames` of each forward.

fifo_length (`int`, *optional*, defaults to 40) : Offline mode: capacity of the FIFO queue of the most recent encoder frames. Streaming mode uses `streaming_config.fifo_length`.

speaker_cache_update_period (`int`, *optional*, defaults to 300) : Offline mode: number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows. Streaming mode uses `streaming_config.speaker_cache_update_period`.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Nemotron3DiarizationModel. It is used to instantiate a Nemotron3 Diarization
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationAudioConfig[[transformers.Nemotron3DiarizationAudioConfig]]

#### transformers.Nemotron3DiarizationAudioConfig[[transformers.Nemotron3DiarizationAudioConfig]]

```python
transformers.Nemotron3DiarizationAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 512, intermediate_size: int = 2048, num_hidden_layers: int = 31, num_attention_heads: int = 8, num_key_value_heads: int | None = None, hidden_act: str = 'gelu', max_position_embeddings: int = 5000, initializer_range: float = 0.02, rope_parameters: dict | None = None, attention_dropout: float | int = 0.0, num_mel_bins: int = 128, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L30)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `512`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `2048`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `31`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `8`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `5000`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

num_mel_bins (`int`, *optional*, defaults to `128`) : Number of mel features used per input frame. Should correspond to the value used in the `AutoFeatureExtractor` class.

subsampling_factor (`int`, *optional*, defaults to 8) : Number of consecutive spectrogram frames stacked into one encoder frame. The classifier upsamples its outputs by the same factor, so speaker activity is predicted at the spectrogram frame rate.

This is the configuration class to store the configuration of a Nemotron3DiarizationModel. It is used to instantiate a Nemotron3 Diarization
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationHeadConfig[[transformers.Nemotron3DiarizationHeadConfig]]

#### transformers.Nemotron3DiarizationHeadConfig[[transformers.Nemotron3DiarizationHeadConfig]]

```python
transformers.Nemotron3DiarizationHeadConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 192, num_speakers: int = 8, audio_hidden_size: int = 512, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L71)

**Parameters:**

hidden_size (`int`, *optional*, defaults to 192) : Hidden size of the speaker head: the encoder output is projected to it, and the upsampler and the classifier keep it.

num_speakers (`int`, *optional*, defaults to 8) : Maximum number of speakers, i.e. the number of per-frame activity outputs. Speakers are ordered by their first arrival in the audio.

audio_hidden_size (`int`, *optional*, defaults to 512) : Hidden size of the encoder output the head projects from. Must match `Nemotron3DiarizationAudioConfig.hidden_size`.

subsampling_factor (`int`, *optional*, defaults to 8) : Upsampling factor of the head, back to the spectrogram frame rate. Must match `Nemotron3DiarizationAudioConfig.subsampling_factor`.

This is the configuration class to store the configuration of a Nemotron3DiarizationModel. It is used to instantiate a Nemotron3 Diarization
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationStreamingConfig[[transformers.Nemotron3DiarizationStreamingConfig]]

#### transformers.Nemotron3DiarizationStreamingConfig[[transformers.Nemotron3DiarizationStreamingConfig]]

```python
transformers.Nemotron3DiarizationStreamingConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, fifo_length: int = 264, speaker_cache_update_period: int = 222, speaker_cache_length: int = 264, speaker_cache_silence_frames_per_speaker: int = 1, prediction_score_threshold: float = 0.25, latest_frames_score_boost: float = 0.05, strong_boost_rate: float = 0.75, weak_boost_rate: float = 1.5, min_positive_scores_rate: float = 0.5, num_speakers: int = 8, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L97)

**Parameters:**

fifo_length (`int`, *optional*, defaults to 264) : Capacity of the FIFO queue of the most recent encoder frames in streaming mode (offline mode uses `Nemotron3DiarizationConfig.fifo_length`).

speaker_cache_update_period (`int`, *optional*, defaults to 222) : Number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows, in streaming mode (offline mode uses `Nemotron3DiarizationConfig.speaker_cache_update_period`).

speaker_cache_length (`int`, *optional*, defaults to 264) : Capacity of the Arrival-Order Speaker Cache. Must be at least `(1 + speaker_cache_silence_frames_per_speaker) * num_speakers`.

speaker_cache_silence_frames_per_speaker (`int`, *optional*, defaults to 1) : Number of speaker-cache slots per speaker reserved for the learned silence embedding when the cache is compressed.

prediction_score_threshold (`float`, *optional*, defaults to 0.25) : Lower clamp of the speaker probabilities before taking their log in the speaker-cache frame scores.

latest_frames_score_boost (`float`, *optional*, defaults to 0.05) : Score bonus given to the frames newly added to the speaker cache when it is compressed.

strong_boost_rate (`float`, *optional*, defaults to 0.75) : Fraction of the per-speaker cache budget whose best frames get a strong score boost, so that every speaker keeps at least that many frames in the cache.

weak_boost_rate (`float`, *optional*, defaults to 1.5) : Fraction of the per-speaker cache budget whose best frames get a weak score boost, which prevents one speaker from dominating the cache.

min_positive_scores_rate (`float`, *optional*, defaults to 0.5) : Fraction of the per-speaker cache budget: a speaker with at least that many positively scored frames has its non-positive (overlapped speech) frames excluded from the cache.

num_speakers (`int`, *optional*, defaults to 8) : Number of speakers tracked by the speaker cache. Must match `Nemotron3DiarizationHeadConfig.num_speakers`.

subsampling_factor (`int`, *optional*, defaults to 8) : Number of speaker-probability frames per encoder frame. Must match `Nemotron3DiarizationAudioConfig.subsampling_factor`.

This is the configuration class to store the configuration of a Nemotron3DiarizationModel. It is used to instantiate a Nemotron3 Diarization
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationAudioModel[[transformers.Nemotron3DiarizationAudioModel]]

#### transformers.Nemotron3DiarizationAudioModel[[transformers.Nemotron3DiarizationAudioModel]]

```python
transformers.Nemotron3DiarizationAudioModel(config: Nemotron3DiarizationAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L546)

**Parameters:**

config ([Nemotron3DiarizationAudioConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Nemotron3 Diarization Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3DiarizationAudioModel.forward]]

```python
forward(input_features: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L560)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See `NemotronAsrStreamingFeatureExtractor.__call__()` for details ([Nemotron3DiarizationProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) uses [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:** [BaseModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3DiarizationConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) and inputs.

The [Nemotron3DiarizationAudioModel](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationAudioModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Nemotron3DiarizationModel[[transformers.Nemotron3DiarizationModel]]

#### transformers.Nemotron3DiarizationModel[[transformers.Nemotron3DiarizationModel]]

```python
transformers.Nemotron3DiarizationModel(config: Nemotron3DiarizationConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L636)

**Parameters:**

config ([Nemotron3DiarizationConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Nemotron3Diarization model without the speaker classifier: encodes one chunk (with its cached frames) and
upsamples the encoder frames back to the spectrogram frame rate. It holds no streaming state, see
[Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification) for the chunked forward.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3DiarizationModel.forward]]

```python
forward(input_features: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L644)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See `NemotronAsrStreamingFeatureExtractor.__call__()` for details ([Nemotron3DiarizationProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) uses [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:** [BaseModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3DiarizationConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) and inputs.

The [Nemotron3DiarizationModel](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## Nemotron3DiarizationProcessor[[transformers.Nemotron3DiarizationProcessor]]

#### transformers.Nemotron3DiarizationProcessor[[transformers.Nemotron3DiarizationProcessor]]

```python
transformers.Nemotron3DiarizationProcessor(feature_extractor, subsampling_factor = 8, streaming_modes = None, streaming_mode = 'low_latency')
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py#L40)

**Parameters:**

feature_extractor (`NemotronAsrStreamingFeatureExtractor`) : The feature extractor is a required input.

subsampling_factor (`int`, *optional*, defaults to 8) : Number of mel frames per encoder frame, mirroring `Nemotron3DiarizationAudioConfig.subsampling_factor`.

streaming_modes (`dict[str, tuple[int, int]]`, *optional*) : Streaming modes the checkpoint supports, name to `(chunk_length, chunk_right_context)` in encoder frames. The processor is the single source of truth for this set: `set_streaming_mode()` validates against it. Defaults to the model-card modes, `"low_latency"` (9, 4), `"very_low_latency"` (6, 2) and `"ultra_low_latency"` (3, 1).

streaming_mode (`str`, *optional*, defaults to `"low_latency"`) : Streaming mode of the sessions, one of `streaming_modes`; change it with `set_streaming_mode()`. Offline use ignores it.

Constructs a Nemotron3DiarizationProcessor which wraps a feature extractor into a single processor.

[Nemotron3DiarizationProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) offers all the functionalities of [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See the
[~NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for more information.

#### __call__[[transformers.Nemotron3DiarizationProcessor.__call__]]

```python
__call__(audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], sampling_rate: int | None = None, is_streaming: bool = False, is_first_audio_chunk: bool = True, is_last_audio_chunk: bool = False, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py#L71)

**Parameters:**

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

sampling_rate (`int`, *optional*) : The sampling rate of the input audio in Hz. Validated against the feature extractor's expected sampling rate (16000 Hz) when provided.

is_streaming (`bool`, *optional*, defaults to `False`) : Whether the audio is one chunk of a streaming session, `is_first_audio_chunk` and `is_last_audio_chunk` telling the first and the last chunks from the others. The chunk sizes are those of `streaming_mode`, changed with `set_streaming_mode()`. Every chunk but the last must hold exactly `num_samples_first_audio_chunk` audio samples for the first one and `num_samples_per_audio_chunk` for the later ones.

is_first_audio_chunk (`bool`, *optional*, defaults to `True`) : Whether this is the first chunk of a streaming session. The feature extractor centers the analysis windows (`center=True`) for the first chunk and for offline use, and does not (`center=False`) for the later chunks, so that the per-chunk spectrogram reproduces, frame for frame, a single full-utterance pass. Must be `True` when `is_streaming=False`.

is_last_audio_chunk (`bool`, *optional*, defaults to `False`) : Whether this chunk ends the streaming session. A chunk of a session ends with `chunk_right_context` look-ahead encoder frames that the model scores at the next step only, and that its next chunk opens with. The last chunk has no next step, so every one of its frames is scored, whatever their number, and its end is zero-padded as a full-utterance pass pads the end of the audio, so that it yields the last frames of the utterance. Must be `False` when `is_streaming=False`.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.19.0/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:** [BatchFeature](/docs/transformers/v5.19.0/en/main_classes/image_processor#transformers.BatchFeature)

the feature extractor outputs, `input_features` and `attention_mask`. In streaming mode
the trailing frames whose analysis window reaches past a chunk other than the last are dropped, so `input_features` holds
exactly the frames of the chunk and can be passed to the model as is, and every chunk but the last also
carries `num_lookahead_frames`, the number of its trailing look-ahead encoder frames, which puts the
model in streaming mode.

## Nemotron3DiarizationSpeakerCache[[transformers.Nemotron3DiarizationSpeakerCache]]

#### transformers.Nemotron3DiarizationSpeakerCache[[transformers.Nemotron3DiarizationSpeakerCache]]

```python
transformers.Nemotron3DiarizationSpeakerCache(config: Nemotron3DiarizationStreamingConfig, fifo_length: int | None = None, speaker_cache_update_period: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L53)

**Parameters:**

config (`Nemotron3DiarizationStreamingConfig`) : Speaker-cache policy, and the FIFO sizes of streaming mode.

fifo_length (`int`, *optional*) : Capacity of the FIFO queue of the most recent encoder frames. Defaults to `config.fifo_length`.

speaker_cache_update_period (`int`, *optional*) : Number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows. Defaults to `config.speaker_cache_update_period`.

Streaming state of [Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification): the Arrival-Order Speaker Cache and the
FIFO queue of the most recent encoder frames, that every chunk attends to.

#### update[[transformers.Nemotron3DiarizationSpeakerCache.update]]

```python
update(chunk_input_embeds: Tensor, chunk_logits: Tensor, silence_embeds: Tensor, num_chunk_frames: int, mask: typing.Optional[torch.Tensor] = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L138)

**Parameters:**

chunk_input_embeds (`torch.Tensor` of shape `(batch_size, num_input_frames, hidden_size)`) : Encoder input of the step: the cached frames returned by `get_embeds`, the chunk and its look-ahead.

chunk_logits (`torch.Tensor` of shape `(batch_size, num_input_frames * subsampling_factor, num_speakers)`) : Speaker logits of the step, used to score the frames when the speaker cache is compressed.

silence_embeds (`torch.Tensor` of shape `(hidden_size,)`) : Learned silence embedding filling the reserved silence slots of a compressed cache.

num_chunk_frames (`int`) : Number of chunk frames following the cached frames in `chunk_input_embeds`. Only those join the FIFO queue: the look-ahead frames after them are fed again at the next step.

mask (`torch.Tensor` of shape `(batch_size, num_input_frames)`, *optional*) : Valid frames of `chunk_input_embeds`, whose padding frames are given zero speaker probabilities.

Pushes a processed chunk to the FIFO queue, moving its oldest frames to the speaker cache when it overflows.

## Nemotron3DiarizationOutput[[transformers.Nemotron3DiarizationOutput]]

#### transformers.Nemotron3DiarizationOutput[[transformers.Nemotron3DiarizationOutput]]

```python
transformers.Nemotron3DiarizationOutput(logits: typing.Optional[torch.Tensor] = None, hidden_states: tuple[torch.Tensor, ...] | None = None, attentions: tuple[torch.Tensor, ...] | None = None, speaker_cache: transformers.models.nemotron3_diarization.modeling_nemotron3_diarization.Nemotron3DiarizationSpeakerCache | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L255)

**Parameters:**

logits (`torch.FloatTensor` of shape `(batch_size, num_frames, config.head_config.num_speakers)`) : Per-frame speaker activity logits at the spectrogram frame rate. `logits.sigmoid()` gives the probability that each speaker is active in each frame; speakers are ordered by their first arrival in the audio.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True`) : Encoder hidden states of every chunk, in chunk order: the encoder runs once per chunk, so the tuple holds `config.audio_config.num_hidden_layers + 1` tensors per chunk. Their sequence length is the chunk's, cache and look-ahead frames included.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True`) : Encoder attention weights of every chunk, in chunk order, `config.audio_config.num_hidden_layers` tensors per chunk.

speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*, returned in streaming mode) : Updated streaming state, to pass to the forward of the next audio chunk of the same streams.

Output of [Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification).

## Nemotron3DiarizationForAudioFrameClassification[[transformers.Nemotron3DiarizationForAudioFrameClassification]]

#### transformers.Nemotron3DiarizationForAudioFrameClassification[[transformers.Nemotron3DiarizationForAudioFrameClassification]]

```python
transformers.Nemotron3DiarizationForAudioFrameClassification(config: Nemotron3DiarizationConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L678)

**Parameters:**

config ([Nemotron3DiarizationConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Streaming Sortformer speaker diarization model: predicts, for every spectrogram frame, the activity of up to
`config.head_config.num_speakers` speakers ordered by first arrival. Audio is processed chunk by chunk, each
chunk attending to a few look-ahead frames and to the Arrival-Order Speaker Cache and FIFO queue carried in a
[Nemotron3DiarizationSpeakerCache](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationSpeakerCache). A whole recording is chunked by the forward itself (offline mode); a stream
is fed one chunk per forward (streaming mode).

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3DiarizationForAudioFrameClassification.forward]]

```python
forward(input_features: Tensor, attention_mask: typing.Optional[torch.Tensor] = None, speaker_cache: transformers.models.nemotron3_diarization.modeling_nemotron3_diarization.Nemotron3DiarizationSpeakerCache | None = None, num_lookahead_frames: int | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L686)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`) : The tensors corresponding to the input audio features. Audio features can be obtained using [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See `NemotronAsrStreamingFeatureExtractor.__call__()` for details ([Nemotron3DiarizationProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) uses [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*) : Streaming state returned by the forward of the previous chunk of the same audio streams.

num_lookahead_frames (`int`, *optional*) : Streaming mode: number of trailing encoder frames of the input that are look-ahead only. They are attended to, but their logits are not returned and they do not join the FIFO queue, as they open the next chunk. [Nemotron3DiarizationProcessor](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) sets it for every chunk but the last one of a session.  The two arguments select the mode. Streaming mode, one chunk per forward: `num_lookahead_frames` given (a first chunk creates the `speaker_cache`, later chunks receive it), or `speaker_cache` given alone (the last chunk of the session, no look-ahead). The input minus its look-ahead is one chunk, whatever its length, pushed as a whole to the FIFO queue sized by `config.streaming_config`. Offline mode, neither given: the input is a whole recording, split by the forward into chunks of `config.chunk_length` encoder frames that take up to `config.chunk_right_context` look-ahead frames from the following ones, with a FIFO queue sized by `config.fifo_length`; no cache is returned.

**Returns:** [Nemotron3DiarizationOutput](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationOutput) or `tuple(torch.FloatTensor)`

A [Nemotron3DiarizationOutput](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3DiarizationConfig](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) and inputs.

The [Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.19.0/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **logits** (`torch.FloatTensor` of shape `(batch_size, num_frames, config.head_config.num_speakers)`) -- Per-frame speaker activity logits at the spectrogram frame rate. `logits.sigmoid()` gives the probability
  that each speaker is active in each frame; speakers are ordered by their first arrival in the audio.
- **hidden_states** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True`) -- Encoder hidden states of every chunk, in chunk order: the encoder runs once per chunk, so the tuple holds
  `config.audio_config.num_hidden_layers + 1` tensors per chunk. Their sequence length is the chunk's, cache
  and look-ahead frames included.
- **attentions** (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True`) -- Encoder attention weights of every chunk, in chunk order, `config.audio_config.num_hidden_layers` tensors
  per chunk.
- **speaker_cache** (`Nemotron3DiarizationSpeakerCache`, *optional*, returned in streaming mode) -- Updated streaming state, to pass to the forward of the next audio chunk of the same streams.

Example:

```python
>>> from transformers import AutoModelForAudioFrameClassification, AutoProcessor
>>> from transformers.audio_utils import load_audio

>>> model_id = "nvidia/Nemotron-3-Diarization"
>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = AutoModelForAudioFrameClassification.from_pretrained(model_id, device_map="auto")

>>> sampling_rate = processor.feature_extractor.sampling_rate
>>> audio = load_audio(
...     "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/en-Alice_woman.wav",
...     sampling_rate=sampling_rate,
... )
>>> inputs = processor(audio, sampling_rate=sampling_rate).to(model.device, dtype=model.dtype)
>>> probabilities = model(**inputs).logits.sigmoid()  # (1, num_frames, 8), one frame every 10 ms
```

### Quickstart
https://huggingface.co/docs/transformers/v5.19.0/model_doc/qwen3_5_moe.md

## Quickstart

```py
import torch
from transformers import pipeline

pipe = pipeline(
    task="text-generation",
    model="Qwen/Qwen3.5-35B-A3B",
    device_map="auto",
)
print(pipe("The capital of France is", max_new_tokens=20)[0]["generated_text"])
```

```py
import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B")
model = Qwen3_5MoeForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-35B-A3B",
    device_map="auto",
)

inputs = tokenizer("Explain mixture-of-experts in one paragraph.", return_tensors="pt").to(model.device)
generated_ids = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
```

## Usage tips and notes

- When training or fine-tuning, set `output_router_logits=True` so the forward returns router logits and the load-balancing auxiliary loss is added to the total loss (scaled by `router_aux_loss_coef`, default `0.001`). Without it, experts can collapse to a few popular slots.
- `Qwen3_5MoeCausalLMOutputWithPast` includes a `router_logits` field. Downstream code that destructures model outputs by position needs to account for it or switch to keyword access.
- For Qwen3.5-35B-A3B, the text config uses `hidden_size=2048` across 40 layers, 256 experts with 8 routed + 1 shared per token, and `moe_intermediate_size=512` — very different shapes from the dense Qwen3.5 checkpoints, so weights are not interchangeable.
- Native context is 262,144 tokens. To reach the advertised ~1M context, enable YaRN rope scaling via the config's `rope_scaling` field — plain loading gives you the native window only.
- As with Qwen3.5, linear-attention layers depend on optional `causal_conv1d` (from [Dao-AILab](https://github.com/Dao-AILab/causal-conv1d)). Without it, the model silently falls back to slower and more memory hungry PyTorch ops.
- On NVIDIA GB10 (compute capability 12.1 / SM121) `causal_conv1d` and `fla` have no SM121 build, so the Gated DeltaNet path always uses the slow PyTorch reference. Passing `use_kernels=True` (`pip install -U kernels`) to [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) swaps it for the same compute-capability-gated Hub kernel as the dense variant ([`Atlas-Inference/gdn`](https://huggingface.co/kernels/Atlas-Inference/gdn), shared because `Qwen3_5MoeGatedDeltaNet` has the same core as `Qwen3_5GatedDeltaNet`); every other GPU keeps the existing path. The kernel is numerically faithful to the fallback (identical greedy output) and speeds up prefill. Measured on `Qwen/Qwen3.6-35B-A3B` (bf16, GB10/SM121, 1024-token prompt, greedy decode of 256 tokens):

  | `use_kernels` | TTFT (prefill) | Decode |
  | --- | --- | --- |
  | `False` (PyTorch fallback) | 0.73 s | 16.3 tok/s |
  | `True` ([`Atlas-Inference/gdn`](https://huggingface.co/kernels/Atlas-Inference/gdn)) | 0.53 s (1.38x faster) | 16.7 tok/s |

  Decode is roughly flat because the single-token DeltaNet recurrence is memory-bandwidth-bound; the win is on the chunked-prefill core and grows with prompt length. Loading the mapped kernel currently requires `trust_remote_code=True` until `Atlas-Inference` is added to the trusted-kernels allowlist.

## Qwen3_5MoeConfig[[transformers.Qwen3_5MoeConfig]]

#### transformers.Qwen3_5MoeConfig[[transformers.Qwen3_5MoeConfig]]

```python
transformers.Qwen3_5MoeConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 248056, video_token_id: int = 248057, vision_start_token_id: int = 248053, vision_end_token_id: int = 248054, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L166)

**Parameters:**

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

image_token_id (`int`, *optional*, defaults to `248056`) : The image token index used as a placeholder for input images.

video_token_id (`int`, *optional*, defaults to `248057`) : The video token index used as a placeholder for input videos.

vision_start_token_id (`int`, *optional*, defaults to `248053`) : Token ID that marks the start of a visual segment in the multimodal input sequence.

vision_end_token_id (`int`, *optional*, defaults to `248054`) : Token ID that marks the end of a visual segment in the multimodal input sequence.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Qwen3_5MoeModel. It is used to instantiate a Qwen3 5 Moe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeConfig

>>> # Initializing a Qwen3.5-MoE style configuration
>>> configuration = Qwen3_5MoeConfig()

>>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
>>> model = Qwen3_5MoeForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen3_5MoeTextConfig[[transformers.Qwen3_5MoeTextConfig]]

#### transformers.Qwen3_5MoeTextConfig[[transformers.Qwen3_5MoeTextConfig]]

```python
transformers.Qwen3_5MoeTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 248320, hidden_size: int = 2048, num_hidden_layers: int = 40, num_attention_heads: int = 16, num_key_value_heads: int = 2, hidden_act: str = 'silu', max_position_embeddings: int = 32768, initializer_range: float = 0.02, rms_norm_eps: float = 1e-06, use_cache: bool = True, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, head_dim: int = 256, linear_conv_kernel_dim: int = 4, linear_key_head_dim: int = 128, linear_value_head_dim: int = 128, linear_num_key_heads: int = 16, linear_num_value_heads: int = 32, moe_intermediate_size: int = 512, shared_expert_intermediate_size: int = 512, num_experts_per_tok: int = 8, num_experts: int = 256, output_router_logits: bool = False, router_aux_loss_coef: float = 0.001, layer_types: list[str] | None = None, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L29)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `248320`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `2048`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `40`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `2`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `32768`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_bias (`bool`, *optional*, defaults to `False`) : Whether to use a bias in the query, key, value and output projection layers during self-attention.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

head_dim (`int`, *optional*, defaults to `256`) : The attention head dimension. If None, it will default to hidden_size // num_attention_heads

linear_conv_kernel_dim (`int`, *optional*, defaults to 4) : Kernel size of the convolution used in linear attention layers.

linear_key_head_dim (`int`, *optional*, defaults to 128) : Dimension of each key head in linear attention.

linear_value_head_dim (`int`, *optional*, defaults to 128) : Dimension of each value head in linear attention.

linear_num_key_heads (`int`, *optional*, defaults to 16) : Number of key heads used in linear attention layers.

linear_num_value_heads (`int`, *optional*, defaults to 32) : Number of value heads used in linear attention layers.

moe_intermediate_size (`int`, *optional*, defaults to `512`) : Intermediate size of the routed expert MLPs.

shared_expert_intermediate_size (`int`, *optional*, defaults to `512`) : Intermediate size of the shared expert MLPs.

num_experts_per_tok (`int`, *optional*, defaults to `8`) : Number of experts to route each token to. This is the top-k value for the token-choice routing.

num_experts (`int`, *optional*, defaults to `256`) : Number of routed experts in MoE layers. 

output_router_logits (`bool`, *optional*, defaults to `False`) : Whether or not the router logits should be returned by the model. Enabling this will also allow the model to output the auxiliary loss, including load balancing loss and router z-loss.

router_aux_loss_coef (`float`, *optional*, defaults to `0.001`) : Auxiliary load balancing loss coefficient. Used to penalize uneven expert routing in MoE models.

layer_types (`list[str]`, *optional*) : A list that explicitly maps each layer index with its layer type. If not provided, it will be automatically generated based on config values.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*) : Token id used for end-of-stream in the vocabulary.

This is the configuration class to store the configuration of a Qwen3_5MoeModel. It is used to instantiate a Qwen3 5 Moe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

```python
>>> from transformers import Qwen3_5MoeTextModel, Qwen3_5MoeTextConfig

>>> # Initializing a Qwen3.5-MoE style configuration
>>> configuration =  Qwen3_5MoeTextConfig()

>>> # Initializing a model from the Qwen3.5-35B-A3B style configuration
>>> model = Qwen3_5MoeTextModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Qwen3_5MoeVisionConfig[[transformers.Qwen3_5MoeVisionConfig]]

#### transformers.Qwen3_5MoeVisionConfig[[transformers.Qwen3_5MoeVisionConfig]]

```python
transformers.Qwen3_5MoeVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, depth: int = 27, hidden_size: int = 1152, hidden_act: str = 'gelu_pytorch_tanh', intermediate_size: int = 4304, num_heads: int = 16, in_channels: int = 3, patch_size: int | list[int] | tuple[int, int] = 16, spatial_merge_size: int = 2, temporal_patch_size: int | list[int] | tuple[int, int] = 2, out_hidden_size: int = 3584, num_position_embeddings: int = 2304, initializer_range: float = 0.02, rope_parameters: dict | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L136)

**Parameters:**

depth (`int`, *optional*, defaults to `27`) : Number of Transformer layers in the vision encoder.

hidden_size (`int`, *optional*, defaults to `1152`) : Dimension of the hidden representations.

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

intermediate_size (`int`, *optional*, defaults to `4304`) : Dimension of the MLP representations.

num_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

in_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `16`) : The size (resolution) of each patch.

spatial_merge_size (`int`, *optional*, defaults to `2`) : The size of the spatial merge window used to reduce the number of visual tokens by merging neighboring patches.

temporal_patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `2`) : Temporal patch size used in the 3D patch embedding for video inputs.

out_hidden_size (`int`, *optional*, defaults to 3584) : The output hidden size of the vision model.

num_position_embeddings (`int`, *optional*, defaults to 2304) : The maximum sequence length that this model might ever be used with

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

This is the configuration class to store the configuration of a Qwen3_5MoeModel. It is used to instantiate a Qwen3 5 Moe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Qwen3_5MoeVisionModel[[transformers.Qwen3_5MoeVisionModel]]

#### transformers.Qwen3_5MoeVisionModel[[transformers.Qwen3_5MoeVisionModel]]

```python
transformers.Qwen3_5MoeVisionModel(config, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1209)

#### forward[[transformers.Qwen3_5MoeVisionModel.forward]]

```python
forward(hidden_states: Tensor, grid_thw: Tensor, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1246)

**Parameters:**

hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`) : The final hidden states of the model.

grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`) : The temporal, height and width of feature shape of each image in LLM.

**Returns:** `torch.Tensor`

hidden_states.

## Qwen3_5MoeTextModel[[transformers.Qwen3_5MoeTextModel]]

#### transformers.Qwen3_5MoeTextModel[[transformers.Qwen3_5MoeTextModel]]

```python
transformers.Qwen3_5MoeTextModel(config: Qwen3_5MoeTextConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1322)

#### forward[[transformers.Qwen3_5MoeTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1337)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeTextModel](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

## Qwen3_5MoeModel[[transformers.Qwen3_5MoeModel]]

#### transformers.Qwen3_5MoeModel[[transformers.Qwen3_5MoeModel]]

```python
transformers.Qwen3_5MoeModel(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1411)

**Parameters:**

config ([Qwen3_5MoeModel](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen3 5 Moe Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Qwen3_5MoeModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_token_type_ids: typing.Optional[torch.IntTensor] = None, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1689)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See `Qwen2VLImageProcessor.__call__()` for details ([Qwen3VLProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See `Qwen3VLVideoProcessor.__call__()` for details ([Qwen3VLProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.19.0/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Qwen3_5MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `Qwen3_5MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeModel](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
- **rope_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) -- The rope index difference between sequence length and multimodal rope.
  The attribute is deprecated and will be removed in v5.20, use `model.base_model.rope_deltas` instead.
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.

## Qwen3_5MoeForCausalLM[[transformers.Qwen3_5MoeForCausalLM]]

#### transformers.Qwen3_5MoeForCausalLM[[transformers.Qwen3_5MoeForCausalLM]]

```python
transformers.Qwen3_5MoeForCausalLM(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1844)

**Parameters:**

config ([Qwen3_5MoeForCausalLM](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Qwen3 5 Moe Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Qwen3_5MoeForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, output_router_logits: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1864)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

output_router_logits (`bool`, *optional*) : Whether or not to return the logits of all the routers. They are useful for computing the router loss, and should not be returned during inference.

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeForCausalLM](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeForCausalLM) forward method, overrides the `__call__` special method.

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
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
>>> from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM

>>> model = Qwen3_5MoeForCausalLM.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")
>>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Next-80B-A3B-Instruct")

>>> prompt = "Hey, are you conscious? Can you talk to me?"
>>> inputs = tokenizer(prompt, return_tensors="pt")

>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
```

## Qwen3_5MoeForConditionalGeneration[[transformers.Qwen3_5MoeForConditionalGeneration]]

#### transformers.Qwen3_5MoeForConditionalGeneration[[transformers.Qwen3_5MoeForConditionalGeneration]]

```python
transformers.Qwen3_5MoeForConditionalGeneration(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1944)

#### forward[[transformers.Qwen3_5MoeForConditionalGeneration.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_token_type_ids: typing.Optional[torch.IntTensor] = None, output_router_logits: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, mm_encoder_outputs: dict[str, transformers.modeling_outputs.BaseModelOutputWithPooling] | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1985)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See `Qwen2VLImageProcessor.__call__()` for details ([Qwen3VLProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See `Qwen3VLVideoProcessor.__call__()` for details ([Qwen3VLProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.19.0/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.19.0/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

output_router_logits (`bool`, *optional*) : Whether or not to return the logits of all the routers. They are useful for computing the router loss, and should not be returned during inference.

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

mm_encoder_outputs (`dict[str, ~modeling_outputs.BaseModelOutputWithPooling]`, *optional*) : Dict where keys are supported modalities and values are encoded outputs for that modality. Each encoded output is a tuple that consists of (`pooler_output`, *optional*: `last_hidden_states`, *optional*: `hidden_states`, *optional*: `attentions`) `pooler_output` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of multimmodal features of the encoder merged into text embeddings.

**Returns:** `Qwen3_5MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `Qwen3_5MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeForConditionalGeneration](/docs/transformers/v5.19.0/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
  `past_key_values` input) to speed up sequential decoding.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **rope_deltas** (`torch.LongTensor` of shape `(batch_size, )`, *optional*) -- The rope index difference between sequence length and multimodal rope.
  The attribute is deprecated and will be removed in v5.20, use `model.base_model.rope_deltas` instead.
- **router_logits** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_router_logits=True` is passed or when `config.add_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Router logits of the model, useful to compute the auxiliary loss for Mixture of Experts models.
- **aux_loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided) -- aux_loss for the sparse modules.

Example:
```python
>>> from transformers import AutoProcessor, Qwen3_5MoeForConditionalGeneration

>>> model = Qwen3_5MoeForConditionalGeneration.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct", dtype="auto", device_map="auto")
>>> processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-35B-A3B-Instruct")

>>> messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://huggingface.co/datasets/hf-internal-testing/transformers-synthetic-assets/resolve/main/images/qwen_vl_demo.jpeg",
            },
            {"type": "text", "text": "Describe this image in short."},
        ],
    }
]

>>> # Preparation for inference
>>> inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
>>> inputs = inputs.to(model.device)

>>> # Generate
>>> generated_ids = model.generate(**inputs, max_new_tokens=128)
>>> generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
>>> processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
"A woman in a plaid shirt sits on a sandy beach at sunset, smiling as she gives a high-five to a yellow Labrador Retriever wearing a harness. The ocean waves roll in the background."
```

### Higgs Audio V2
https://huggingface.co/docs/transformers/v5.19.0/model_doc/higgs_audio_v2.md
