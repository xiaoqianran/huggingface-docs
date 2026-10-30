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
`config.fifo_length`, `config.chunk_left_context`, `config.chunk_length` and `config.chunk_right_context` in
`max_window`.

| Speedup vs eager (A100, batch size 1) | float32 | bfloat16 |
| ------------------------------------- | ------- | -------- |
| streaming, per step                   | 1.2x    | 4.4x     |
| offline, 488 s recording              | 1.3x    | 2.8x     |

## Nemotron3DiarizationConfig[[transformers.Nemotron3DiarizationConfig]]

#### transformers.Nemotron3DiarizationConfig[[transformers.Nemotron3DiarizationConfig]]

```python
transformers.Nemotron3DiarizationConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, audio_config: transformers.models.nemotron3_diarization.configuration_nemotron3_diarization.Nemotron3DiarizationAudioConfig | dict | None = None, head_config: transformers.models.nemotron3_diarization.configuration_nemotron3_diarization.Nemotron3DiarizationHeadConfig | dict | None = None, streaming_config: transformers.models.nemotron3_diarization.configuration_nemotron3_diarization.Nemotron3DiarizationStreamingConfig | dict | None = None, chunk_length: int = 340, chunk_right_context: int = 40, chunk_left_context: int = 0, fifo_length: int = 40, speaker_cache_update_period: int = 300, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L146)

**Parameters:**

audio_config (`Nemotron3DiarizationAudioConfig` or `dict`, *optional*) : Configuration of the transformer audio encoder. Defaults to `Nemotron3DiarizationAudioConfig()`.

head_config (`Nemotron3DiarizationHeadConfig` or `dict`, *optional*) : Configuration of the speaker head. Defaults to `Nemotron3DiarizationHeadConfig()`.

streaming_config (`Nemotron3DiarizationStreamingConfig` or `dict`, *optional*) : Speaker-cache policy, and the FIFO sizes of streaming mode. Defaults to `Nemotron3DiarizationStreamingConfig()`.

chunk_length (`int`, *optional*, defaults to 340) : Offline mode: number of encoder frames per chunk when a whole recording is diarized in one forward. In streaming mode the chunk is the input of each forward.

chunk_right_context (`int`, *optional*, defaults to 40) : Offline mode: number of look-ahead encoder frames each chunk takes from the following ones. In streaming mode the look-ahead is `num_lookahead_frames` of each forward.

chunk_left_context (`int`, *optional*, defaults to 0) : Offline mode: number of look-back encoder frames each chunk takes again from the preceding ones, fed between the cached frames and the chunk. Like the look-ahead, they are attended to but neither scored nor pushed to the FIFO queue. In streaming mode the look-back is `num_lookback_frames` of each forward.

fifo_length (`int`, *optional*, defaults to 40) : Offline mode: capacity of the FIFO queue of the most recent encoder frames. Streaming mode uses `streaming_config.fifo_length`.

speaker_cache_update_period (`int`, *optional*, defaults to 300) : Offline mode: number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows. Streaming mode uses `streaming_config.speaker_cache_update_period`.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

This is the configuration class to store the configuration of a Nemotron3DiarizationModel. It is used to instantiate a Nemotron3 Diarization
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationAudioConfig[[transformers.Nemotron3DiarizationAudioConfig]]

#### transformers.Nemotron3DiarizationAudioConfig[[transformers.Nemotron3DiarizationAudioConfig]]

```python
transformers.Nemotron3DiarizationAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 512, intermediate_size: int = 2048, num_hidden_layers: int = 31, num_attention_heads: int = 8, num_key_value_heads: int | None = None, hidden_act: str = 'gelu', max_position_embeddings: int = 5000, initializer_range: float = 0.02, rope_parameters: dict | None = None, attention_dropout: float | int = 0.0, num_mel_bins: int = 128, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L30)

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

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationHeadConfig[[transformers.Nemotron3DiarizationHeadConfig]]

#### transformers.Nemotron3DiarizationHeadConfig[[transformers.Nemotron3DiarizationHeadConfig]]

```python
transformers.Nemotron3DiarizationHeadConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 192, num_speakers: int = 8, audio_hidden_size: int = 512, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L71)

**Parameters:**

hidden_size (`int`, *optional*, defaults to 192) : Hidden size of the speaker head: the encoder output is projected to it, and the upsampler and the classifier keep it.

num_speakers (`int`, *optional*, defaults to 8) : Maximum number of speakers, i.e. the number of per-frame activity outputs. Speakers are ordered by their first arrival in the audio.

audio_hidden_size (`int`, *optional*, defaults to 512) : Hidden size of the encoder output the head projects from. Must match `Nemotron3DiarizationAudioConfig.hidden_size`.

subsampling_factor (`int`, *optional*, defaults to 8) : Upsampling factor of the head, back to the spectrogram frame rate. Must match `Nemotron3DiarizationAudioConfig.subsampling_factor`.

This is the configuration class to store the configuration of a Nemotron3DiarizationModel. It is used to instantiate a Nemotron3 Diarization
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/Nemotron-3-Diarization](https://huggingface.co/nvidia/Nemotron-3-Diarization)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationStreamingConfig[[transformers.Nemotron3DiarizationStreamingConfig]]

#### transformers.Nemotron3DiarizationStreamingConfig[[transformers.Nemotron3DiarizationStreamingConfig]]

```python
transformers.Nemotron3DiarizationStreamingConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, fifo_length: int = 264, speaker_cache_update_period: int = 222, speaker_cache_length: int = 264, speaker_cache_silence_frames_per_speaker: int = 1, prediction_score_threshold: float = 0.25, latest_frames_score_boost: float = 0.05, strong_boost_rate: float = 0.75, weak_boost_rate: float = 1.5, min_positive_scores_rate: float = 0.5, num_speakers: int = 8, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/configuration_nemotron3_diarization.py#L97)

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

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Nemotron3DiarizationAudioModel[[transformers.Nemotron3DiarizationAudioModel]]

#### transformers.Nemotron3DiarizationAudioModel[[transformers.Nemotron3DiarizationAudioModel]]

```python
transformers.Nemotron3DiarizationAudioModel(config: Nemotron3DiarizationAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L562)

**Parameters:**

config ([Nemotron3DiarizationAudioConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Nemotron3 Diarization Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3DiarizationAudioModel.forward]]

```python
forward(input_features: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L576)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See `NemotronAsrStreamingFeatureExtractor.__call__()` for details ([Nemotron3DiarizationProcessor](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) uses [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:** [BaseModelOutput](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3DiarizationConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) and inputs.

The [Nemotron3DiarizationAudioModel](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationAudioModel) forward method, overrides the `__call__` special method.

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

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L652)

**Parameters:**

config ([Nemotron3DiarizationConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Nemotron3Diarization model without the speaker classifier: encodes one chunk (with its cached frames) and
upsamples the encoder frames back to the spectrogram frame rate. It holds no streaming state, see
[Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification) for the chunked forward.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3DiarizationModel.forward]]

```python
forward(input_features: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L660)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See `NemotronAsrStreamingFeatureExtractor.__call__()` for details ([Nemotron3DiarizationProcessor](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) uses [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:** [BaseModelOutput](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3DiarizationConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) and inputs.

The [Nemotron3DiarizationModel](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationModel) forward method, overrides the `__call__` special method.

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

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py#L40)

**Parameters:**

feature_extractor (`NemotronAsrStreamingFeatureExtractor`) : The feature extractor is a required input.

subsampling_factor (`int`, *optional*, defaults to 8) : Number of mel frames per encoder frame, mirroring `Nemotron3DiarizationAudioConfig.subsampling_factor`.

streaming_modes (`dict[str, tuple[int, int]]`, *optional*) : Streaming modes the checkpoint supports, name to `(chunk_length, chunk_right_context)` in encoder frames. The processor is the single source of truth for this set: `set_streaming_mode()` validates against it. Defaults to the model-card modes, `"low_latency"` (9, 4), `"very_low_latency"` (6, 2) and `"ultra_low_latency"` (3, 1).

streaming_mode (`str`, *optional*, defaults to `"low_latency"`) : Streaming mode of the sessions, one of `streaming_modes`; change it with `set_streaming_mode()`. Offline use ignores it.

Constructs a Nemotron3DiarizationProcessor which wraps a feature extractor into a single processor.

[Nemotron3DiarizationProcessor](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) offers all the functionalities of [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See the
[~NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for more information.

#### __call__[[transformers.Nemotron3DiarizationProcessor.__call__]]

```python
__call__(audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], sampling_rate: int | None = None, is_streaming: bool = False, is_first_audio_chunk: bool = True, is_last_audio_chunk: bool = False, **kwargs: *<class 'transformers.models.nemotron3_diarization.processing_nemotron3_diarization.Nemotron3DiarizationProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/processing_nemotron3_diarization.py#L71)

**Parameters:**

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

sampling_rate (`int`, *optional*) : The sampling rate of the input audio in Hz. Validated against the feature extractor's expected sampling rate (16000 Hz) when provided.

is_streaming (`bool`, *optional*, defaults to `False`) : Whether the audio is one chunk of a streaming session, `is_first_audio_chunk` and `is_last_audio_chunk` telling the first and the last chunks from the others. The chunk sizes are those of `streaming_mode`, changed with `set_streaming_mode()`. Every chunk but the last must hold exactly `num_samples_first_audio_chunk` audio samples for the first one and `num_samples_per_audio_chunk` for the later ones.

is_first_audio_chunk (`bool`, *optional*, defaults to `True`) : Whether this is the first chunk of a streaming session. The feature extractor centers the analysis windows (`center=True`) for the first chunk and for offline use, and does not (`center=False`) for the later chunks, so that the per-chunk spectrogram reproduces, frame for frame, a single full-utterance pass. Must be `True` when `is_streaming=False`.

is_last_audio_chunk (`bool`, *optional*, defaults to `False`) : Whether this chunk ends the streaming session. A chunk of a session ends with `chunk_right_context` look-ahead encoder frames that the model scores at the next step only, and that its next chunk opens with. The last chunk has no next step, so every one of its frames is scored, whatever their number, and its end is zero-padded as a full-utterance pass pads the end of the audio, so that it yields the last frames of the utterance. Must be `False` when `is_streaming=False`.

return_tensors (`str` or [TensorType](/docs/transformers/v5.22.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.22.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:** [BatchFeature](/docs/transformers/v5.22.1/en/main_classes/feature_extractor#transformers.BatchFeature)

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

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L53)

**Parameters:**

config (`Nemotron3DiarizationStreamingConfig`) : Speaker-cache policy, and the FIFO sizes of streaming mode.

fifo_length (`int`, *optional*) : Capacity of the FIFO queue of the most recent encoder frames. Defaults to `config.fifo_length`.

speaker_cache_update_period (`int`, *optional*) : Number of encoder frames moved from the FIFO queue to the speaker cache when the queue overflows. Defaults to `config.speaker_cache_update_period`.

Streaming state of [Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification): the Arrival-Order Speaker Cache and the
FIFO queue of the most recent encoder frames, that every chunk attends to.

#### update[[transformers.Nemotron3DiarizationSpeakerCache.update]]

```python
update(chunk_input_embeds: Tensor, chunk_logits: Tensor, silence_embeds: Tensor, num_chunk_frames: int, mask: typing.Optional[torch.Tensor] = None, num_lookback_frames: int = 0)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L138)

**Parameters:**

chunk_input_embeds (`torch.Tensor` of shape `(batch_size, num_input_frames, hidden_size)`) : Encoder input of the step: the cached frames returned by `get_embeds`, the look-back frames, the chunk and its look-ahead.

chunk_logits (`torch.Tensor` of shape `(batch_size, num_input_frames * subsampling_factor, num_speakers)`) : Speaker logits of the step, used to score the frames when the speaker cache is compressed.

silence_embeds (`torch.Tensor` of shape `(hidden_size,)`) : Learned silence embedding filling the reserved silence slots of a compressed cache.

num_chunk_frames (`int`) : Number of chunk frames following the cached and look-back frames in `chunk_input_embeds`. Only those join the FIFO queue: the look-ahead frames after them are fed again at the next step.

mask (`torch.Tensor` of shape `(batch_size, num_input_frames)`, *optional*) : Valid frames of `chunk_input_embeds`, whose padding frames are given zero speaker probabilities.

num_lookback_frames (`int`, *optional*, defaults to 0) : Number of look-back frames between the cached frames and the chunk in `chunk_input_embeds`: frames of the previous chunk fed again, already pushed with it.

Pushes a processed chunk to the FIFO queue, moving its oldest frames to the speaker cache when it overflows.

## Nemotron3DiarizationOutput[[transformers.Nemotron3DiarizationOutput]]

#### transformers.Nemotron3DiarizationOutput[[transformers.Nemotron3DiarizationOutput]]

```python
transformers.Nemotron3DiarizationOutput(logits: typing.Optional[torch.Tensor] = None, hidden_states: tuple[torch.Tensor, ...] | None = None, attentions: tuple[torch.Tensor, ...] | None = None, speaker_cache: transformers.models.nemotron3_diarization.modeling_nemotron3_diarization.Nemotron3DiarizationSpeakerCache | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L267)

**Parameters:**

logits (`torch.FloatTensor` of shape `(batch_size, num_frames, config.head_config.num_speakers)`) : Per-frame speaker activity logits at the spectrogram frame rate. `logits.sigmoid()` gives the probability that each speaker is active in each frame; speakers are ordered by their first arrival in the audio.

hidden_states (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_hidden_states=True`) : Encoder hidden states of every chunk, in chunk order: the encoder runs once per chunk, so the tuple holds `config.audio_config.num_hidden_layers + 1` tensors per chunk. Their sequence length is the chunk's, cache and look-ahead frames included.

attentions (`tuple[torch.FloatTensor, ...]`, *optional*, returned when `output_attentions=True`) : Encoder attention weights of every chunk, in chunk order, `config.audio_config.num_hidden_layers` tensors per chunk.

speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*, returned in streaming mode) : Updated streaming state, to pass to the forward of the next audio chunk of the same streams.

Output of [Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification).

## Nemotron3DiarizationForAudioFrameClassification[[transformers.Nemotron3DiarizationForAudioFrameClassification]]

#### transformers.Nemotron3DiarizationForAudioFrameClassification[[transformers.Nemotron3DiarizationForAudioFrameClassification]]

```python
transformers.Nemotron3DiarizationForAudioFrameClassification(config: Nemotron3DiarizationConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L694)

**Parameters:**

config ([Nemotron3DiarizationConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Streaming Sortformer speaker diarization model: predicts, for every spectrogram frame, the activity of up to
`config.head_config.num_speakers` speakers ordered by first arrival. Audio is processed chunk by chunk, each
chunk attending to a few look-ahead frames and to the Arrival-Order Speaker Cache and FIFO queue carried in a
[Nemotron3DiarizationSpeakerCache](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationSpeakerCache). A whole recording is chunked by the forward itself (offline mode); a stream
is fed one chunk per forward (streaming mode).

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3DiarizationForAudioFrameClassification.forward]]

```python
forward(input_features: Tensor, attention_mask: typing.Optional[torch.Tensor] = None, speaker_cache: transformers.models.nemotron3_diarization.modeling_nemotron3_diarization.Nemotron3DiarizationSpeakerCache | None = None, num_lookahead_frames: int | None = None, num_lookback_frames: int | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_diarization/modeling_nemotron3_diarization.py#L702)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`) : The tensors corresponding to the input audio features. Audio features can be obtained using [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor). See `NemotronAsrStreamingFeatureExtractor.__call__()` for details ([Nemotron3DiarizationProcessor](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) uses [NemotronAsrStreamingFeatureExtractor](/docs/transformers/v5.22.1/en/model_doc/nemotron_asr_streaming#transformers.NemotronAsrStreamingFeatureExtractor) for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

speaker_cache (`Nemotron3DiarizationSpeakerCache`, *optional*) : Streaming state returned by the forward of the previous chunk of the same audio streams.

num_lookahead_frames (`int`, *optional*) : Streaming mode: number of trailing encoder frames of the input that are look-ahead only. They are attended to, but their logits are not returned and they do not join the FIFO queue, as they open the next chunk. [Nemotron3DiarizationProcessor](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationProcessor) sets it for every chunk but the last one of a session.  These arguments and `num_lookback_frames` select the mode. Streaming mode, one chunk per forward: `num_lookahead_frames` or `num_lookback_frames` given (a first chunk creates the `speaker_cache`, later chunks receive it), or `speaker_cache` given alone (the last chunk of the session, no look-ahead). The input minus its look-back and look-ahead frames is one chunk, whatever its length, pushed as a whole to the FIFO queue sized by `config.streaming_config`. Offline mode, neither given: the input is a whole recording, split by the forward into chunks of `config.chunk_length` encoder frames that take up to `config.chunk_left_context` look-back frames from the preceding ones and up to `config.chunk_right_context` look-ahead frames from the following ones, with a FIFO queue sized by `config.fifo_length`; no cache is returned.

num_lookback_frames (`int`, *optional*) : Streaming mode: number of leading encoder frames of the input that are look-back only, the end of the previous chunk fed again ahead of this one, on top of the frames of `speaker_cache`. They are attended to, but their logits are not returned and they do not join the FIFO queue again. Defaults to 0.

**Returns:** [Nemotron3DiarizationOutput](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationOutput) or `tuple(torch.FloatTensor)`

A [Nemotron3DiarizationOutput](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3DiarizationConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationConfig)) and inputs.

The [Nemotron3DiarizationForAudioFrameClassification](/docs/transformers/v5.22.1/en/model_doc/nemotron3_diarization#transformers.Nemotron3DiarizationForAudioFrameClassification) forward method, overrides the `__call__` special method.

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

### MiniCPM3
https://huggingface.co/docs/transformers/v5.22.1/model_doc/minicpm3.md
