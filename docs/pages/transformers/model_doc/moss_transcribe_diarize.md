# MOSS-Transcribe-Diarize

## Overview

**MOSS-Transcribe-Diarize 0.9B** is an end-to-end audio understanding model for long-form multi-speaker transcription,
speaker diarization, and timestamps. It combines a Whisper-style encoder, a 4× frame merge step, a multi-modal
projector, and a Qwen3 language model.

The model checkpoint is available at:
[itazap/MOSS-Transcribe-Diarize-HF](https://huggingface.co/itazap/MOSS-Transcribe-Diarize-HF)

Key capabilities include:

* **Joint transcription and diarization** in a single pass, with segments formatted as
  `[start][S01]text[end]`.
* **Long-form audio** via Whisper-window chunking upstream in the processor; the model reassembles the chunks per
  sample.
* **Promptable transcription** through `apply_transcription_request` or the chat template.
* **Multilingual support** across 50+ languages.

This model was contributed by [Ita Zaporozhets](https://huggingface.co/itazap) from the Hugging Face team. See the
[model card](https://huggingface.co/itazap/MOSS-Transcribe-Diarize-HF) and the
[OpenMOSS repository](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) for more details.

## Usage

> [!NOTE]
> Most examples below use `apply_transcription_request`, a convenience function that builds the chat template for
> you. See [Advanced usage with the chat template](#advanced-usage-with-the-chat-template) for the equivalent
> `apply_chat_template` call.

### Basic transcription and diarization

Each speaker turn is decoded as a `[start][S0N]text[end]` segment. `processor.decode` can return the raw decoded
string, the parsed segments as a list of dicts (`return_format="parsed"`), or just the concatenated transcription
(`return_format="transcription_only"`).

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/podcast_sample.wav"
inputs = processor.apply_transcription_request(audio_url)
inputs = inputs.to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, do_sample=False, max_new_tokens=512)
generated_ids = outputs[:, inputs.input_ids.shape[1] :]

decoded = processor.decode(generated_ids, skip_special_tokens=True)
print("RAW OUTPUT:", decoded)

parsed = processor.decode(generated_ids, return_format="parsed")[0]
print("\nPARSED (list of dicts):")
for segment in parsed:
    print(segment)

transcription = processor.decode(generated_ids, return_format="transcription_only")
print("\nTRANSCRIPTION ONLY:", transcription[0])

"""
RAW OUTPUT: ["[0.48][S01] Hey everyone, welcome back to the show.[2.43][2.46][S01] Today, we're talking about the best way to learn a new skill.[5.22][5.43][S01] I'm here with Jordan to dig into it.[7.14][8.40][S02] Thanks for having me.[9.42][9.63][S02] My take is that consistency beats intensity every single time.[12.87][13.14][S01] That makes sense.[13.92][13.92][S01] Even 20 minutes a day adds up over a few months.[16.17][16.32][S02] Exactly, and writing down small wins keeps you motivated along the way.[19.62]"]

PARSED (list of dicts):
{'Start': 0.48, 'End': 2.43, 'Speaker': 1, 'Content': 'Hey everyone, welcome back to the show.'}
{'Start': 2.46, 'End': 5.22, 'Speaker': 1, 'Content': "Today, we're talking about the best way to learn a new skill."}
{'Start': 5.43, 'End': 7.14, 'Speaker': 1, 'Content': "I'm here with Jordan to dig into it."}
{'Start': 8.4, 'End': 9.42, 'Speaker': 2, 'Content': 'Thanks for having me.'}
{'Start': 9.63, 'End': 12.87, 'Speaker': 2, 'Content': 'My take is that consistency beats intensity every single time.'}
{'Start': 13.14, 'End': 13.92, 'Speaker': 1, 'Content': 'That makes sense.'}
{'Start': 13.92, 'End': 16.17, 'Speaker': 1, 'Content': 'Even 20 minutes a day adds up over a few months.'}
{'Start': 16.32, 'End': 19.62, 'Speaker': 2, 'Content': 'Exactly, and writing down small wins keeps you motivated along the way.'}

TRANSCRIPTION ONLY: Hey everyone, welcome back to the show. Today, we're talking about the best way to learn a new skill. I'm here with Jordan to dig into it. Thanks for having me. My take is that consistency beats intensity every single time. That makes sense. Even 20 minutes a day adds up over a few months. Exactly, and writing down small wins keeps you motivated along the way.
"""
```

### Custom prompt and hotwords

`apply_transcription_request` accepts `keywords` to bias transcription toward specific terms (e.g. proper nouns or
domain-specific vocabulary), rendered as a `热词列表：`("hotword list") in the chat template. A custom `prompt` can be
passed alongside it for additional context.

Below we transcribe an audio clip that mentions "Yorùbá", comparing with and without passing it as a keyword.

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/languages_sample.wav"

# Without keywords
inputs = processor.apply_transcription_request(audio_url)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
transcription = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(f"WITHOUT keywords: {transcription[0]}")

# With keywords
inputs = processor.apply_transcription_request(audio_url, keywords=["Mandarin", "Yorùbá"])
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
transcription = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(f"WITH keywords   : {transcription[0]}")

"""
WITHOUT keywords: Let's talk about the most spoken languages in the world. Mandarin Chinese has the most native speakers, followed by Spanish, English, and Hindi. In West Africa, one of the most widely spoken languages is Yoruba, with over 50 million speakers.
WITH keywords   : Let's talk about the most spoken languages in the world. Mandarin Chinese has the most native speakers, followed by Spanish, English, and Hindi. In West Africa, one of the most widely spoken languages is Yorùbá, with over 50 million speakers.
"""
```

### Advanced usage with the chat template

`apply_transcription_request` is equivalent to a chat template with a user turn that contains only audio:

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav"
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
        ],
    },
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
).to(model.device, dtype=model.dtype)

outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
decoded = processor.decode(
    outputs[:, inputs.input_ids.shape[1] :],
    skip_special_tokens=True,
)
print(decoded)
```

### Batch inference

Pass a list of audio, and optionally a matching list of prompts and/or keywords, to transcribe a batch in one call.
Set an entry to `None` to skip it for that sample.

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
processor = AutoProcessor.from_pretrained(model_id)
# `from_pretrained` is used here for convenience, the model can also be built from a config directly.
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

audio = [
    "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav",
    "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/languages_sample.wav",
]
keywords = [None, ["Mandarin", "Yorùbá"]]

inputs = processor.apply_transcription_request(audio, keywords=keywords)
inputs = inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=128)
transcription = processor.decode(outputs[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(transcription)
```

### Training

MOSS-Transcribe-Diarize can be trained with the loss returned by the model when `labels` are provided. Build a
conversation with the audio in the user turn and the target transcription as the assistant turn, then set
`output_labels=True` when tokenizing so the processor builds the labels for you (audio and padding tokens are masked
out with `-100`).

```python
from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
model.train()

conversation = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "url": "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "[0.00][S01] Mister quilter is apostle of the middle classes, and we are glad to welcome his gospel.[3.00]",
                },
            ],
        },
    ],
]
inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    output_labels=True,
).to(model.device, dtype=model.dtype)

loss = model(**inputs).loss
print("Loss:", loss.item())
loss.backward()
```

### Torch compile

The model can be compiled for faster inference/training. `device_map="auto"` wraps the forward pass with `accelerate`
hooks that cause repeated recompilation, so load the model directly on a single device instead.

On a B200, we observed a speed-up of ~2.1x for a batch size of 4.

```python
import time

import torch

from transformers import AutoProcessor, AutoModelForCausalLM

model_id = "itazap/MOSS-Transcribe-Diarize-HF"

num_warmup = 5
num_runs = 20

# Load processor + model
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

# Prepare static inputs
audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav"
inputs = processor.apply_transcription_request([audio_url] * 4).to(model.device, torch.bfloat16)

# Benchmark without compile
print("Warming up without compile...")
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)

torch.accelerator.synchronize()

print("\nBenchmarking without torch.compile...")
torch.accelerator.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.accelerator.synchronize()
no_compile_time = (time.time() - start) / num_runs
print(f"Average time without compile: {no_compile_time:.4f}s")

# Benchmark with compile
print("\nCompiling model...")
model = torch.compile(model)

print("Warming up with compile (includes graph capture)...")
with torch.no_grad():
    for _ in range(num_warmup):
        _ = model(**inputs)

torch.accelerator.synchronize()

print("\nBenchmarking with torch.compile...")
torch.accelerator.synchronize()
start = time.time()
with torch.no_grad():
    for _ in range(num_runs):
        _ = model(**inputs)
torch.accelerator.synchronize()
compile_time = (time.time() - start) / num_runs
print(f"Average time with compile: {compile_time:.4f}s")

speedup = no_compile_time / compile_time
print(f"\nSpeedup: {speedup:.2f}x")
```

For autoregressive transcription, `torch.compile` accelerates the per-token forward passes inside `generate` by
providing a `CompileConfig` object instead of compiling `model.forward` directly.

```python
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, CompileConfig

model_id = "itazap/MOSS-Transcribe-Diarize-HF"
num_warmup = 3

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id).to("cuda")

audio_url = "https://huggingface.co/datasets/itazap/audio_samples/resolve/main/intro_sample.wav"
inputs = processor.apply_transcription_request([audio_url] * 4).to(model.device, torch.bfloat16)

compile_config = CompileConfig()

# Warmup
for _ in range(num_warmup):
    _ = model.generate(**inputs, max_new_tokens=128, cache_implementation="static", compile_config=compile_config)

# Apply model
generated_ids = model.generate(
    **inputs, max_new_tokens=128, cache_implementation="static", compile_config=compile_config
)
transcription = processor.decode(generated_ids[:, inputs.input_ids.shape[1] :], return_format="transcription_only")
print(transcription)
```

## MossTranscribeDiarizeConfig[[transformers.MossTranscribeDiarizeConfig]]

#### transformers.MossTranscribeDiarizeConfig[[transformers.MossTranscribeDiarizeConfig]]

```python
transformers.MossTranscribeDiarizeConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, audio_token_id: int = 151671, projector_hidden_act: str = 'silu', projector_bias: bool = True, audio_merge_size: int = 4, audio_chunk_size: int = 480000, tie_word_embeddings: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/configuration_moss_transcribe_diarize.py#L73)

**Parameters:**

audio_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio backbone.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

audio_token_id (`int`, *optional*, defaults to `151671`) : The audio token index used as a placeholder for input audio.

projector_hidden_act (`str`, *optional*, defaults to `silu`) : The activation function used by the multimodal projector.

projector_bias (`bool`, *optional*, defaults to `True`) : Whether to use bias in the multimodal projector.

audio_merge_size (`int`, *optional*, defaults to 4) : Number of consecutive Whisper encoder frames concatenated before the multi-modal projector.

audio_chunk_size (`int`, *optional*, defaults to 480000) : Whisper encoder window size in raw audio samples, used with `padding_mask` to recover `audio_chunk_mapping`.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a MossTranscribeDiarizeModel. It is used to instantiate a Moss Transcribe Diarize
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [hf-audio/MOSS-Transcribe-Diarize-HF](https://huggingface.co/hf-audio/MOSS-Transcribe-Diarize-HF)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MossTranscribeDiarizeForConditionalGeneration, MossTranscribeDiarizeConfig

>>> # Initializing a MossTranscribeDiarize configuration
>>> configuration = MossTranscribeDiarizeConfig()

>>> # Initializing a model from the configuration
>>> model = MossTranscribeDiarizeForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MossTranscribeDiarizeAudioConfig[[transformers.MossTranscribeDiarizeAudioConfig]]

#### transformers.MossTranscribeDiarizeAudioConfig[[transformers.MossTranscribeDiarizeAudioConfig]]

```python
transformers.MossTranscribeDiarizeAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, num_mel_bins: int = 80, num_hidden_layers: int = 24, num_attention_heads: int = 16, intermediate_size: int = 4096, activation_function: str = 'gelu', hidden_size: int = 1024, dropout: float | int = 0.0, attention_dropout: float | int = 0.0, activation_dropout: float | int = 0.0, initializer_range: float = 0.02, max_source_positions: int = 1500)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/configuration_moss_transcribe_diarize.py#L30)

**Parameters:**

num_mel_bins (`int`, *optional*, defaults to `80`) : Number of mel features used per input frame. Should correspond to the value used in the `AutoFeatureExtractor` class.

num_hidden_layers (`int`, *optional*, defaults to `24`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

intermediate_size (`int`, *optional*, defaults to `4096`) : Dimension of the MLP representations.

activation_function (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The ratio for all dropout layers.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

activation_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for activations inside the fully connected layer.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

max_source_positions (`int`, *optional*, defaults to 1500) : The maximum sequence length of log-mel filter-bank features that this model might ever be used with.

This is the configuration class to store the configuration of a MossTranscribeDiarizeModel. It is used to instantiate a Moss Transcribe Diarize
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [hf-audio/MOSS-Transcribe-Diarize-HF](https://huggingface.co/hf-audio/MOSS-Transcribe-Diarize-HF)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.21.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import MossTranscribeDiarizeAudioConfig, MossTranscribeDiarizeAudioModel

>>> # Initializing a MossTranscribeDiarizeAudioConfig
>>> configuration = MossTranscribeDiarizeAudioConfig()

>>> # Initializing a MossTranscribeDiarizeAudioModel (with random weights)
>>> model = MossTranscribeDiarizeAudioModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## MossTranscribeDiarizeAudioModel[[transformers.MossTranscribeDiarizeAudioModel]]

#### transformers.MossTranscribeDiarizeAudioModel[[transformers.MossTranscribeDiarizeAudioModel]]

```python
transformers.MossTranscribeDiarizeAudioModel(config: MossTranscribeDiarizeAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L256)

**Parameters:**

config ([MossTranscribeDiarizeAudioConfig](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MOSS-Transcribe-Diarize audio model, which is a Whisper encoder.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MossTranscribeDiarizeAudioModel.forward]]

```python
forward(input_features, attention_mask = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L295)

**Parameters:**

input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, sequence_length)`) : Log-mel features extracted from the raw speech waveform, one row per 30s Whisper window. See [__call__()](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor.__call__).

attention_mask (`torch.Tensor`, *optional*) : MossTranscribeDiarize does not support masking of the `input_features`, this argument is preserved for compatibility, but it is not used.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MossTranscribeDiarizeConfig](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeConfig)) and inputs.

The [MossTranscribeDiarizeAudioModel](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeAudioModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## MossTranscribeDiarizeFeatureExtractor[[transformers.MossTranscribeDiarizeFeatureExtractor]]

#### transformers.MossTranscribeDiarizeFeatureExtractor[[transformers.MossTranscribeDiarizeFeatureExtractor]]

```python
transformers.MossTranscribeDiarizeFeatureExtractor(feature_size = 80, sampling_rate = 16000, hop_length = 160, chunk_length = 30, n_fft = 400, padding_value = 0.0, dither = 0.0, return_attention_mask = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/feature_extraction_moss_transcribe_diarize.py#L35)

Constructs a MOSS-Transcribe-Diarize feature extractor.

This is a [WhisperFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/whisper#transformers.WhisperFeatureExtractor) that additionally splits audio longer than `chunk_length` seconds
into consecutive Whisper-window chunks before extracting log-mel features, and reports each sample's
original raw-audio length via `padding_mask` (as opposed to `input_features_mask`, which marks valid
frames per chunk).

#### __call__[[transformers.MossTranscribeDiarizeFeatureExtractor.__call__]]

```python
__call__(raw_speech: numpy.ndarray | list[float] | list[numpy.ndarray] | list[list[float]], truncation: bool = True, pad_to_multiple_of: int | None = None, return_tensors: str | transformers.utils.generic.TensorType | None = None, sampling_rate: int | None = None, device: str | None = 'cpu', **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/feature_extraction_moss_transcribe_diarize.py#L170)

**Parameters:**

raw_speech (`np.ndarray`, `list[float]`, `list[np.ndarray]`, `list[list[float]]`) : The sequence or batch of sequences to be processed. Mono-channel audio only.

truncation (`bool`, *optional*, defaults to `True`) : Activates truncation to cut each `n_samples`-long chunk to `n_samples`.

pad_to_multiple_of (`int`, *optional*) : If set, pads each chunk's raw audio to a multiple of this value.

device (`str`, *optional*, defaults to `"cpu"`) : Device used to compute the log-mel spectrogram.

**Returns:** [BatchFeature](/docs/transformers/v5.21.1/en/main_classes/feature_extractor#transformers.BatchFeature)

with `input_features` (log-mel features, one row per chunk), `input_features_mask`
(valid-frame mask per chunk), and `padding_mask` (valid raw-sample mask, one row per input sample).

Splits each audio sample into consecutive `n_samples`-long chunks, extracts log-mel features per
chunk, and reports each sample's original raw-audio length via `padding_mask`.

## MossTranscribeDiarizeProcessor[[transformers.MossTranscribeDiarizeProcessor]]

#### transformers.MossTranscribeDiarizeProcessor[[transformers.MossTranscribeDiarizeProcessor]]

```python
transformers.MossTranscribeDiarizeProcessor(feature_extractor = None, tokenizer = None, chat_template: str | None = None, audio_token: str = '<|audio_pad|>', audio_bos_token: str = '<|audio_start|>', audio_eos_token: str = '<|audio_end|>', audio_tokens_per_second: float = 12.5, audio_merge_size: int = 4, time_marker_every_seconds: int = 2)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/processing_moss_transcribe_diarize.py#L43)

**Parameters:**

feature_extractor (`MossTranscribeDiarizeFeatureExtractor`) : The feature extractor is a required input.

tokenizer (`Qwen2Tokenizer`) : The tokenizer is a required input.

chat_template (`str`, *optional*) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

audio_token (`str`, *optional*, defaults to `"<|audio_pad|>"`) : Special token used to represent audio inputs in the chat template.

audio_bos_token (`str`, *optional*, defaults to `"<|audio_start|>"`) : Special token marking the start of the audio span in the chat template.

audio_eos_token (`str`, *optional*, defaults to `"<|audio_end|>"`) : Special token marking the end of the audio span in the chat template.

audio_tokens_per_second (`float`, *optional*, defaults to 12.5) : Expected number of audio placeholder tokens per second of input audio.

audio_merge_size (`int`, *optional*, defaults to 4) : Whisper frame merge factor used when counting audio tokens.

time_marker_every_seconds (`int`, *optional*, defaults to 2) : Insert numeric time-marker tokens into the audio span every N seconds. Set to `0` to disable time markers and emit a plain audio placeholder span instead.

Constructs a MossTranscribeDiarizeProcessor which wraps a feature extractor and a tokenizer into a single processor.

[MossTranscribeDiarizeProcessor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeProcessor) offers all the functionalities of [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor) and [Qwen2Tokenizer](/docs/transformers/v5.21.1/en/model_doc/qwen2#transformers.Qwen2Tokenizer). See the
[~MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor) and [~Qwen2Tokenizer](/docs/transformers/v5.21.1/en/model_doc/qwen2#transformers.Qwen2Tokenizer) for more information.

#### __call__[[transformers.MossTranscribeDiarizeProcessor.__call__]]

```python
__call__(text: str | list[str], audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, output_labels: bool | None = False, **kwargs: *<class 'transformers.models.moss_transcribe_diarize.processing_moss_transcribe_diarize.MossTranscribeDiarizeProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/processing_moss_transcribe_diarize.py#L104)

**Parameters:**

text (`Union[str, list[str]]`) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

output_labels (`bool`, *optional*, default=False) : Whether to return labels for training.

return_tensors (`str` or [TensorType](/docs/transformers/v5.21.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:** [BatchFeature](/docs/transformers/v5.21.1/en/main_classes/feature_extractor#transformers.BatchFeature)

A dictionary with tokenized text (`input_ids`, `attention_mask`) and
audio features (`input_values`, `padding_mask`).

#### apply_chat_template[[transformers.MossTranscribeDiarizeProcessor.apply_chat_template]]

```python
apply_chat_template(conversation: list[dict[str, str]] | list[list[dict[str, str]]], chat_template: str | None = None, tools: list[dict] | None = None, documents: list[dict[str, str]] | None = None, add_generation_prompt: bool = False, continue_final_message: bool | str = False, return_assistant_tokens_mask: bool = False, tokenize: bool = False, return_tensors: str | transformers.utils.generic.TensorType | None = None, return_dict: bool = False, load_audio_from_video: bool = False, processor_kwargs: dict | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/processing_utils.py#L1994)

**Parameters:**

conversation (`Union[list[Dict, [str, str]], list[list[dict[str, str]]]]`) : The conversation to format.

chat_template (`Optional[str]`, *optional*) : The Jinja template to use for formatting the conversation. If not provided, the tokenizer's chat template is used.

Similar to the `apply_chat_template` method on tokenizers, this method applies a Jinja template to input
conversations to turn them into a single tokenizable string.

The input is expected to be in the following format, where each message content is a list consisting of text and
optionally image or video inputs. One can also provide an image, video, URL or local path which will be used to form
`pixel_values` when `return_dict=True`. If not provided, one will get only the formatted text, optionally tokenized text.

conversation = [
{
"role": "user",
"content": [
{"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
{"type": "text", "text": "Please describe this image in detail."},
],
},
]

#### apply_transcription_request[[transformers.MossTranscribeDiarizeProcessor.apply_transcription_request]]

```python
apply_transcription_request(audio: typing.Union[str, list[str], numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], prompt: str | list[str] | None = None, keywords: str | list[str] | list[list[str]] | None = None, **kwargs: *<class 'transformers.models.moss_transcribe_diarize.processing_moss_transcribe_diarize.MossTranscribeDiarizeProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/processing_moss_transcribe_diarize.py#L213)

**Parameters:**

audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) : Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.

prompt (`str` or `list[str]`, *optional*) : Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`, the chat template supplies the default timestamped diarization prompt.

keywords (`str`, `list[str]`, or `list[list[str]]`, *optional*) : Hotwords/domain terms (e.g. proper nouns) to bias transcription toward the correct spelling. A string or flat list of strings is shared across the batch; a nested list supplies separate keywords per audio sample.

- ****kwargs** : Additional keyword arguments forwarded to [apply_chat_template()](/docs/transformers/v5.21.1/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template).

Prepare inputs for transcription / diarization without manually writing the chat template.

## MossTranscribeDiarizeModel[[transformers.MossTranscribeDiarizeModel]]

#### transformers.MossTranscribeDiarizeModel[[transformers.MossTranscribeDiarizeModel]]

```python
transformers.MossTranscribeDiarizeModel(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L382)

**Parameters:**

config ([MossTranscribeDiarizeModel](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MOSS-Transcribe-Diarize model: Whisper encoder, 4x time merge, multi-modal projector, and Qwen3 language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MossTranscribeDiarizeModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, input_features_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L459)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor). See [MossTranscribeDiarizeFeatureExtractor.__call__()](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor.__call__) for details ([MossTranscribeDiarizeProcessor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeProcessor) uses [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor) for processing audios).

input_features_mask (`torch.Tensor` of shape `(num_chunks, feature_sequence_length)`, *optional*) : Mask marking valid (non-padded) feature indices, one row per chunked log-mel feature row in `input_features`. Used to compute each chunk's valid encoder-output length.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Forwarded as-is to the language model.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Precomputed embeddings, in place of passing `input_ids`. When `input_features` is also given, `input_ids` is still required so the audio placeholder positions in `inputs_embeds` can be located and filled with the audio embeddings.

padding_mask (`torch.Tensor` of shape `(batch_size, max_audio_length)`, *optional*) : Mask marking each audio sample's valid raw-audio length. Used with `config.audio_chunk_size` to recover `audio_chunk_mapping`.

**Returns:** `MossTranscribeDiarizeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `MossTranscribeDiarizeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MossTranscribeDiarizeConfig](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeConfig)) and inputs.

The [MossTranscribeDiarizeModel](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- Projected audio hidden states.

#### get_audio_features[[transformers.MossTranscribeDiarizeModel.get_audio_features]]

```python
get_audio_features(input_features: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, input_features_mask: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, padding_mask: <module 'torch.Tensor' (<doc_builder.mock_imports.MockFinder object at 0x7fef3bad2690>)>, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L394)

**Parameters:**

input_features (`doc_builder.mock_imports.torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`) : The tensors corresponding to the input audio features. Audio features can be obtained using [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor). See [MossTranscribeDiarizeFeatureExtractor.__call__()](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor.__call__) for details ([MossTranscribeDiarizeProcessor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeProcessor) uses [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor) for processing audios).

input_features_mask (`torch.Tensor` of shape `(num_chunks, feature_sequence_length)`) : Mask marking valid (non-padded) feature indices, one row per chunked log-mel feature row in `input_features`. Used to compute each chunk's valid encoder-output length.

padding_mask (`torch.Tensor` of shape `(batch_size, max_audio_length)`) : Mask marking each audio sample's valid raw-audio length, one row per sample. Used with `config.audio_chunk_size` to recover `audio_chunk_mapping`.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.21.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MossTranscribeDiarizeConfig](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeConfig)) and inputs.

Extract MOSS audio embeddings from log-mel features, reassembling multi-chunk audio per sample.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(batch_size, hidden_size)`) -- Last layer hidden-state of the first token of the sequence (classification token) after further processing
  through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
  the classification token after processing through a linear layer and a tanh activation function. The linear
  layer weights are trained from the next sentence prediction (classification) objective during pretraining.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

## MossTranscribeDiarizeForConditionalGeneration[[transformers.MossTranscribeDiarizeForConditionalGeneration]]

#### transformers.MossTranscribeDiarizeForConditionalGeneration[[transformers.MossTranscribeDiarizeForConditionalGeneration]]

```python
transformers.MossTranscribeDiarizeForConditionalGeneration(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L522)

**Parameters:**

config ([MossTranscribeDiarizeForConditionalGeneration](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeForConditionalGeneration)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The MOSS-Transcribe-Diarize model for conditional generation with transcription and speaker diarization.

This model inherits from [PreTrainedModel](/docs/transformers/v5.21.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.MossTranscribeDiarizeForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, input_features_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L535)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.21.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.21.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor). See [MossTranscribeDiarizeFeatureExtractor.__call__()](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor.__call__) for details ([MossTranscribeDiarizeProcessor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeProcessor) uses [MossTranscribeDiarizeFeatureExtractor](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeFeatureExtractor) for processing audios).

input_features_mask (`torch.Tensor` of shape `(num_chunks, feature_sequence_length)`, *optional*) : Mask marking valid (non-padded) feature indices, one row per chunked log-mel feature row in `input_features`. Used to compute each chunk's valid encoder-output length.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Forwarded as-is to the language model.

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Precomputed embeddings, in place of passing `input_ids`. When `input_features` is also given, `input_ids` is still required so the audio placeholder positions in `inputs_embeds` can be located and filled with the audio embeddings.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss.

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

padding_mask (`torch.Tensor` of shape `(batch_size, max_audio_length)`, *optional*) : Mask marking each audio sample's valid raw-audio length. Used with `config.audio_chunk_size` to recover `audio_chunk_mapping`.

**Returns:** `MossTranscribeDiarizeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MossTranscribeDiarizeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([MossTranscribeDiarizeConfig](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeConfig)) and inputs.

The [MossTranscribeDiarizeForConditionalGeneration](/docs/transformers/v5.21.1/en/model_doc/moss_transcribe_diarize#transformers.MossTranscribeDiarizeForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.21.1/en/internal/generation_utils#transformers.Cache) instance.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- Hidden states of the audio encoder after projection.

Example:

```python
>>> from transformers import MossTranscribeDiarizeForConditionalGeneration, AutoProcessor

>>> model_id = "hf-audio/MOSS-Transcribe-Diarize-HF"
>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = MossTranscribeDiarizeForConditionalGeneration.from_pretrained(model_id, device_map="auto")
>>> inputs = processor.apply_transcription_request("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3")
>>> inputs = inputs.to(model.device, dtype=model.dtype)
>>> outputs = model.generate(**inputs, do_sample=False, max_new_tokens=500)
>>> processor.batch_decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
```

#### get_audio_features[[transformers.MossTranscribeDiarizeForConditionalGeneration.get_audio_features]]

```python
get_audio_features(input_features, input_features_mask, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.21.1/src/transformers/models/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py#L532)

### ViTMatte
https://huggingface.co/docs/transformers/v5.21.1/model_doc/vitmatte.md
