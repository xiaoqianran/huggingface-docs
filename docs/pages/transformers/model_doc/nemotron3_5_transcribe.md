# Nemotron 3.5 Transcribe

## Overview

Nemotron 3.5 Transcribe is an audio language model for multilingual transcription, speaker-attributed transcription,
speech translation and audio question answering. A speech recognition encoder and the
[Nemotron 3 Diarization](./nemotron3_diarization) model read the same spectrogram; their outputs are projected into a
[Nemotron-H](./nemotron_h) language model. A word timestamp head on the encoder outputs gives word-level times.

## Usage

### Transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(samples + "en-Alice_woman.wav").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(processor.decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=True))
```

### Batched transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

audio = [samples + "en-Alice_woman.wav", samples + "diarization_example.mp3"]
speaker_attributed_prompt = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write "
    "numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)
prompt = [None, speaker_attributed_prompt]

inputs = processor.apply_transcription_request(audio, prompt=prompt).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
texts = processor.decode(outputs[:, inputs.input_ids.shape[1] :], skip_special_tokens=False)
print([text.replace(processor.tokenizer.pad_token, "") for text in texts])
```

### Speaker-attributed transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

prompt = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write "
    "numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "audio", "path": samples + "diarization_example.mp3"},
        ],
    }
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    enable_thinking=False,
).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
text = processor.decode(outputs[0, inputs.input_ids.shape[1] :], skip_special_tokens=False)
print(text)
```

[extract_speaker_dict()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeProcessor.extract_speaker_dict) splits the output into one dictionary per speaker turn:

```python
for turn in processor.extract_speaker_dict(text):
    print(turn)
```

```text
{'Speaker': 0, 'Content': "To celebrate the release of Nematron three Dierization, we invited ... from NVIDIA."}
{'Speaker': 1, 'Content': "Hi everybody, thanks for having me. I'm Jane, a pocket voice from QTie."}
{'Speaker': 2, 'Content': "Nice to meet you Chris and Jane. I'm Serena, the Queen three Voice."}
...
```

### Word timestamps

Requires `torchaudio` and `scipy`.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B"
samples = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

prompt = (
    "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write "
    "numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription."
)

inputs = processor.apply_transcription_request(samples + "diarization_example.mp3", prompt=prompt).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024, return_word_timestamp_logits=True)
texts = processor.decode(outputs.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=False)
timestamps = processor.align_word_timestamps(
    texts,
    outputs.word_timestamp_logits,
    inputs.input_features_mask,
    outputs.speaker_logits,
)
print(timestamps[0]["chunks"][:3])
```

To align a transcript that was not generated:

```python
input_features_mask = inputs.input_features_mask
audio_outputs = model.get_audio_features(inputs.input_features, input_features_mask)
word_timestamp_logits = model.get_word_timestamp_logits(audio_outputs.asr_last_hidden_state, input_features_mask)
timestamps = processor.align_word_timestamps(
    ["<spk:0> To celebrate the ..."],
    word_timestamp_logits,
    input_features_mask,
    audio_outputs.speaker_logits,
)
```

## Nemotron3_5TranscribeConfig[[transformers.Nemotron3_5TranscribeConfig]]

#### transformers.Nemotron3_5TranscribeConfig[[transformers.Nemotron3_5TranscribeConfig]]

```python
transformers.Nemotron3_5TranscribeConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, audio_token_id: int = 131072, word_timestamp_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/configuration_nemotron3_5_transcribe.py#L223)

**Parameters:**

audio_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio backbone.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

audio_token_id (`int`, *optional*, defaults to 131072) : The audio token index used as a placeholder for the audio embeddings. The original checkpoint appends it after the language model's vocabulary, with no embedding row: it is embedded as token 0, then replaced by the audio.

word_timestamp_config (`Union[dict, Nemotron3_5TranscribeWordTimestampConfig]`, *optional*) : Configuration of the word timestamp encoder and head, which predict alignment tokenizer pieces from the speech recognition encoder states.

This is the configuration class to store the configuration of a Nemotron3 5 TranscribeModel. It is used to instantiate a Nemotron3 5 Transcribe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Nemotron3_5TranscribeForConditionalGeneration, Nemotron3_5TranscribeConfig

>>> # Initializing a Nemotron3_5Transcribe configuration
>>> configuration = Nemotron3_5TranscribeConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = Nemotron3_5TranscribeForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Nemotron3_5TranscribeAsrAudioConfig[[transformers.Nemotron3_5TranscribeAsrAudioConfig]]

#### transformers.Nemotron3_5TranscribeAsrAudioConfig[[transformers.Nemotron3_5TranscribeAsrAudioConfig]]

```python
transformers.Nemotron3_5TranscribeAsrAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1280, intermediate_size: int = 5120, num_hidden_layers: int = 32, num_attention_heads: int = 10, num_key_value_heads: int | None = None, hidden_act: str = 'gelu', max_position_embeddings: int = 5000, initializer_range: float = 0.02, rope_parameters: dict | None = None, attention_dropout: float | int = 0.0, num_mel_bins: int = 128, subsampling_factor: int = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/configuration_nemotron3_5_transcribe.py#L31)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1280`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `5120`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `32`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `10`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `5000`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

num_mel_bins (`int`, *optional*, defaults to `128`) : Number of mel features used per input frame. Should correspond to the value used in the `AutoFeatureExtractor` class.

subsampling_factor (`int`, *optional*, defaults to 8) : Number of consecutive spectrogram frames stacked into one encoder frame.

This is the configuration class to store the configuration of a Nemotron3 5 TranscribeModel. It is used to instantiate a Nemotron3 5 Transcribe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Nemotron3_5TranscribeAsrAudioModel, Nemotron3_5TranscribeAsrAudioConfig

>>> # Initializing a Nemotron3_5TranscribeAsrAudioModel configuration
>>> configuration = Nemotron3_5TranscribeAsrAudioConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = Nemotron3_5TranscribeAsrAudioModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Nemotron3_5TranscribeAudioConfig[[transformers.Nemotron3_5TranscribeAudioConfig]]

#### transformers.Nemotron3_5TranscribeAudioConfig[[transformers.Nemotron3_5TranscribeAudioConfig]]

```python
transformers.Nemotron3_5TranscribeAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, asr_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, diarization_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, speaker_activity_threshold: float = 0.5)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/configuration_nemotron3_5_transcribe.py#L152)

**Parameters:**

asr_config (`Union[dict, Nemotron3_5TranscribeAsrAudioConfig]`, *optional*) : Configuration of the speech recognition encoder.

diarization_config (`Union[dict, Nemotron3DiarizationConfig]`, *optional*) : Configuration of the speaker diarization model, whose speaker activities are added to the speech recognition encoder outputs.

speaker_activity_threshold (`float`, *optional*, defaults to 0.5) : A speaker is active in an encoder frame when its probability is above this threshold.

This is the configuration class to store the configuration of a Nemotron3 5 TranscribeModel. It is used to instantiate a Nemotron3 5 Transcribe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Nemotron3_5TranscribeAudioModel, Nemotron3_5TranscribeAudioConfig

>>> # Initializing a Nemotron3_5TranscribeAudioModel configuration
>>> configuration = Nemotron3_5TranscribeAudioConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = Nemotron3_5TranscribeAudioModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## Nemotron3_5TranscribeWordTimestampConfig[[transformers.Nemotron3_5TranscribeWordTimestampConfig]]

#### transformers.Nemotron3_5TranscribeWordTimestampConfig[[transformers.Nemotron3_5TranscribeWordTimestampConfig]]

```python
transformers.Nemotron3_5TranscribeWordTimestampConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1280, intermediate_size: int = 640, num_hidden_layers: int = 2, num_attention_heads: int = 16, num_key_value_heads: int | None = None, hidden_act: str = 'gelu', max_position_embeddings: int = 5000, initializer_range: float = 0.02, rope_parameters: dict | None = None, attention_dropout: float | int = 0.0, vocab_size: int = 23863, blank_token_id: int = 23862, residual_scale: float = 0.1, upsampling_factor: int = 2)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/configuration_nemotron3_5_transcribe.py#L88)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1280`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `640`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `2`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `5000`) : The maximum sequence length that this model might ever be used with.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rope_parameters (`dict`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

vocab_size (`int`, *optional*, defaults to 23863) : Number of classes of the word timestamp head: the pieces of its alignment tokenizer, then the blank.

blank_token_id (`int`, *optional*, defaults to 23862) : Index of the blank class, after the pieces of the alignment tokenizer.

residual_scale (`float`, *optional*, defaults to 0.1) : Scale of the transformer output added to the speech recognition encoder states before the classifier.

upsampling_factor (`int`, *optional*, defaults to 2) : Number of head frames per encoder frame, made by a transposed convolution before the transformer: 2 gives 40 ms frames from the 80 ms encoder frames.

This is the configuration class to store the configuration of a Nemotron3 5 TranscribeModel. It is used to instantiate a Nemotron3 5 Transcribe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.22.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Nemotron3_5TranscribeConfig, Nemotron3_5TranscribeWordTimestampConfig

>>> # Initializing a Nemotron3_5TranscribeWordTimestamp configuration
>>> configuration = Nemotron3_5TranscribeWordTimestampConfig()

>>> # Using it as the word timestamp head of the model
>>> model_configuration = Nemotron3_5TranscribeConfig(word_timestamp_config=configuration)
```

## Nemotron3_5TranscribeProcessor[[transformers.Nemotron3_5TranscribeProcessor]]

#### transformers.Nemotron3_5TranscribeProcessor[[transformers.Nemotron3_5TranscribeProcessor]]

```python
transformers.Nemotron3_5TranscribeProcessor(feature_extractor, tokenizer, alignment_tokenizer, chat_template = None, audio_token = '<|audio|>', default_transcription_prompt = 'Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only.', subsampling_factor = 8)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py#L235)

**Parameters:**

feature_extractor (`feature_extractor_class`) : The feature extractor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

alignment_tokenizer (`PreTrainedTokenizerFast`) : The vocabulary of the word timestamp head: 23,862 SentencePiece BPE pieces, the head's class 23,862 being the CTC blank. Only used by [align_word_timestamps()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeProcessor.align_word_timestamps), saved in the `alignment_tokenizer/` subfolder. Prepend a space to reproduce the original SentencePiece ids of any text, including a text that starts with a tag: `alignment_tokenizer.encode(" " + text, add_special_tokens=False)`.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

audio_token (`str`, *optional*, defaults to `"<|audio|>"`) : Special token used to represent audio inputs in the chat template.

default_transcription_prompt (`str`, *optional*) : Default prompt used by [apply_transcription_request()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeProcessor.apply_transcription_request).

subsampling_factor (`int`, *optional*, defaults to 8) : Number of spectrogram frames per audio token, mirroring `Nemotron3_5TranscribeAsrAudioConfig.subsampling_factor`. The spectrogram is padded to a multiple of it.

Constructs a Nemotron3_5TranscribeProcessor which wraps a feature extractor, a tokenizer, and a alignment tokenizer into a single processor.

[Nemotron3_5TranscribeProcessor](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeProcessor) offers all the functionalities of `feature_extractor_class`, `tokenizer_class`, and `{alignment_tokenizer_class}`. See the
`~feature_extractor_class`, `~tokenizer_class`, and `~{alignment_tokenizer_class}` for more information.

#### __call__[[transformers.Nemotron3_5TranscribeProcessor.__call__]]

```python
__call__(text: str | list[str], audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, output_labels: bool | None = False, **kwargs: *<class 'transformers.models.nemotron3_5_transcribe.processing_nemotron3_5_transcribe.Nemotron3_5TranscribeProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py#L268)

**Parameters:**

text (`Union[str, list[str]]`) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

output_labels (`bool`, *optional*, default=False) : Whether to return labels for training.

return_tensors (`str` or [TensorType](/docs/transformers/v5.22.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.22.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:** [BatchFeature](/docs/transformers/v5.22.1/en/main_classes/feature_extractor#transformers.BatchFeature)

A dictionary with tokenized text (`input_ids`, `attention_mask`) and
audio features (`input_features`, `input_features_mask`).

#### apply_transcription_request[[transformers.Nemotron3_5TranscribeProcessor.apply_transcription_request]]

```python
apply_transcription_request(audio: typing.Union[str, list[str], numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], prompt: str | list[str] | None = None, **kwargs: *<class 'transformers.models.nemotron3_5_transcribe.processing_nemotron3_5_transcribe.Nemotron3_5TranscribeProcessorKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py#L340)

**Parameters:**

audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) : Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.

prompt (`str` or `list[str]`, *optional*) : Custom prompt(s) to include in the user turn. A list must be the same length as the batch. When `None`, each sample uses `default_transcription_prompt`.

- ****kwargs** : Additional keyword arguments forwarded to [apply_chat_template()](/docs/transformers/v5.22.1/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template) (for example `text_kwargs`, `audio_kwargs`, ...). Thinking is disabled unless `enable_thinking=True` is passed (the chat template enables it by default).

**Returns:** [BatchFeature](/docs/transformers/v5.22.1/en/main_classes/feature_extractor#transformers.BatchFeature)

Processor outputs ready to be passed to
`Nemotron3_5TranscribeForConditionalGeneration.generate()`.

Prepare inputs for automatic speech recognition without manually writing the default transcription prompt.

#### apply_chat_template[[transformers.Nemotron3_5TranscribeProcessor.apply_chat_template]]

```python
apply_chat_template(conversation: list[dict[str, str]] | list[list[dict[str, str]]], chat_template: str | None = None, tools: list[dict] | None = None, documents: list[dict[str, str]] | None = None, add_generation_prompt: bool = False, continue_final_message: bool | str = False, return_assistant_tokens_mask: bool = False, tokenize: bool = False, return_tensors: str | transformers.utils.generic.TensorType | None = None, return_dict: bool = False, load_audio_from_video: bool = False, processor_kwargs: dict | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/processing_utils.py#L1994)

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

#### extract_speaker_dict[[transformers.Nemotron3_5TranscribeProcessor.extract_speaker_dict]]

```python
extract_speaker_dict(text: str | list[str])
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py#L414)

**Parameters:**

text (`str` or `list[str]`) : Single text or batch of texts, decoded with `skip_special_tokens=False`: the `<spk:N>` speaker tags are special tokens. Everything from the end-of-text token on is dropped.

**Returns:** `list[dict]` for a single text, `list[list[dict]]` for a batch

one dict per speaker turn, with the
`"Speaker"` index `N` of its `<spk:N>` tag and its `"Content"`. A text without tags is a single turn with
`"Speaker": None`, as is the text before the first tag, if any.

Splits a speaker-attributed transcript into speaker turns, in the format of
`VibeVoiceAsrProcessor.extract_speaker_dict()` without the timestamps, which this model does not predict.

Example:

```python
>>> processor.extract_speaker_dict("<spk:0> Hi, how are you? <spk:1> Fine, thanks.<|im_end|>")
[{'Speaker': 0, 'Content': 'Hi, how are you?'}, {'Speaker': 1, 'Content': 'Fine, thanks.'}]
```

#### align_word_timestamps[[transformers.Nemotron3_5TranscribeProcessor.align_word_timestamps]]

```python
align_word_timestamps(transcriptions: str | list[str], word_timestamp_logits: torch.Tensor, input_features_mask: torch.Tensor, speaker_logits: torch.Tensor | None = None, language: str | list[str | None] | None = None, speaker_activity_threshold: float = 0.5, max_word_duration: float = 1.0, max_word_frames: int = 12)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py#L455)

**Parameters:**

transcriptions (`str` or `list[str]`) : Generated text of each recording, decoded with `skip_special_tokens=False` to keep the `<spk:N>` tags.

word_timestamp_logits (`torch.Tensor` of shape `(batch_size, num_frames * upsampling_factor, vocab_size)`) : Logits of the word timestamp head (`word_timestamp_logits` of `generate(..., return_word_timestamp_logits=True)` or `model.get_word_timestamp_logits(...)`), the last class being the CTC blank. Their number of frames per encoder frame, the head's `upsampling_factor`, is read from their shape.

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask of the valid spectrogram frames of each recording, as returned by the processor: their number sets the duration of the recording, hence the time step of the head's frames.

speaker_logits (`torch.Tensor` of shape `(batch_size, num_frames * subsampling_factor, num_speakers)`, *optional*) : Logits of the diarization model (`speaker_logits` of `generate(..., return_word_timestamp_logits=True)` or of `get_audio_features`). Without them, each transcript is aligned as a single speaker and no diarization is returned.

language (`str` or `list[str | None]`, *optional*) : Language of the transcripts, one for all or one per transcript, detected from the script when `None`: `"zh"` and `"ja"` are aligned character by character, `"th"` is not supported.

speaker_activity_threshold (`float`, *optional*, defaults to 0.5) : Probability above which a diarization speaker is active.

max_word_duration (`float`, *optional*, defaults to 1.0) : Maximum duration of an aligned word, in seconds.

max_word_frames (`int`, *optional*, defaults to 12) : Maximum shift and duration of a word snapped to a diarization segment, in encoder frames of at most 80 ms.

**Returns:** `dict` or `list[dict]`

for the recording (or for each recording of a list of transcripts), as the word timestamps of the speech recognition pipeline, `"chunks"`
(`{"text", "timestamp", "speaker"}` per word, `timestamp` = `(start, end)` in seconds, sorted by time;
`speaker` is the `N` of the word's tag, `None` before the first tag and 0 in a transcript without tags),
`"diarization"` (`{"speaker", "timestamp"}` activity segments of each diarization speaker),
`"speaker_tag_to_diarization_speaker"` (`None` for an unmatched tag) and `"failed_speakers"` (the tags
whose words could not be aligned, missing from `"chunks"`).

Times each word of generated transcripts and attributes it to a speaker: the words of each `<spk:N>` tag are
force-aligned (`torchaudio.functional.forced_align`) on the frames of the word timestamp head where the
diarization speaker matched to the tag is active, then snapped to that speaker's diarization segments. Port of
NeMo's `MultiSpeakerSOTWordTimestampAligner`, with an exact search where it approximates long recordings.

Example:

```python
>>> inputs = processor.apply_transcription_request(audio, prompt=speaker_attributed_prompt).to(model.device)
>>> outputs = model.generate(**inputs, return_word_timestamp_logits=True, max_new_tokens=500)
>>> transcriptions = processor.decode(
...     outputs.sequences[:, inputs.input_ids.shape[1] :], skip_special_tokens=False
... )
>>> timestamps = processor.align_word_timestamps(
...     transcriptions, outputs.word_timestamp_logits, inputs.input_features_mask, outputs.speaker_logits
... )
>>> timestamps[0]["chunks"][0]  # {'text': ..., 'timestamp': (..., ...), 'speaker': 0}
```

## Nemotron3_5TranscribeAsrAudioModel[[transformers.Nemotron3_5TranscribeAsrAudioModel]]

#### transformers.Nemotron3_5TranscribeAsrAudioModel[[transformers.Nemotron3_5TranscribeAsrAudioModel]]

```python
transformers.Nemotron3_5TranscribeAsrAudioModel(config: Nemotron3_5TranscribeAsrAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L379)

**Parameters:**

config ([Nemotron3_5TranscribeAsrAudioConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeAsrAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Nemotron 3.5 Transcribe speech recognition encoder: the Nemotron 3 Diarization audio encoder with the queries
and keys of each attention head normalized, and rotary position embeddings on part of each head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3_5TranscribeAsrAudioModel.forward]]

```python
forward(input_features: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L400)

**Parameters:**

input_features (`torch.Tensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using `feature_extractor_class`. See `feature_extractor_class.__call__` for details (`processor_class` uses `feature_extractor_class` for processing audios).

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

**Returns:** [BaseModelOutput](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or `tuple(torch.FloatTensor)`

A [BaseModelOutput](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) and inputs.

The [Nemotron3_5TranscribeAsrAudioModel](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeAsrAudioModel) forward method, overrides the `__call__` special method.

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

## Nemotron3_5TranscribeAudioModel[[transformers.Nemotron3_5TranscribeAudioModel]]

#### transformers.Nemotron3_5TranscribeAudioModel[[transformers.Nemotron3_5TranscribeAudioModel]]

```python
transformers.Nemotron3_5TranscribeAudioModel(config: Nemotron3_5TranscribeAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L547)

**Parameters:**

config ([Nemotron3_5TranscribeAudioConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Nemotron 3.5 Transcribe audio model: a speech recognition encoder and a speaker diarization model run in
parallel on the same spectrogram, and the speaker activities are added to the speech recognition encoder outputs.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3_5TranscribeAudioModel.forward]]

```python
forward(input_features: FloatTensor, input_features_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L561)

**Parameters:**

input_features (`torch.FloatTensor` of shape `(batch_size, feature_sequence_length, num_mel_bins)`) : Log mel spectrogram, padded to a multiple of `config.subsampling_factor` frames. As in the original, audio longer than `config.diarization_config.chunk_length` encoder frames is encoded in chunks with context.

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`, *optional*) : Mask of the valid spectrogram frames. Padding must be on the right.

**Returns:** `Nemotron3_5TranscribeAudioModelOutput` or `tuple(torch.FloatTensor)`

A `Nemotron3_5TranscribeAudioModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) and inputs.

The [Nemotron3_5TranscribeAudioModel](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeAudioModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.
- **pooler_output** (`torch.FloatTensor` of shape `(num_audio_tokens, text_config.hidden_size)`, *optional*) -- Projected audio embeddings of the valid encoder frames of every recording, in batch order: the embeddings that
  replace the audio tokens of the prompt. Returned by `get_audio_features`.
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.
- **asr_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, num_frames, audio_config.hidden_size)`) -- Output of the speech recognition encoder, before its normalization and the speaker embeddings: the input of
  [get_word_timestamp_logits()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeForConditionalGeneration.get_word_timestamp_logits).
- **speaker_logits** (`torch.FloatTensor` of shape `(batch_size, feature_sequence_length, audio_config.num_speakers)`) -- Logits of the diarization model, one frame per spectrogram frame.

## Nemotron3_5TranscribeWordTimestampModel[[transformers.Nemotron3_5TranscribeWordTimestampModel]]

#### transformers.Nemotron3_5TranscribeWordTimestampModel[[transformers.Nemotron3_5TranscribeWordTimestampModel]]

```python
transformers.Nemotron3_5TranscribeWordTimestampModel(config: Nemotron3_5TranscribeWordTimestampConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L444)

**Parameters:**

config ([Nemotron3_5TranscribeWordTimestampConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeWordTimestampConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The transformer of the Nemotron 3.5 Transcribe word timestamp head.
`Nemotron3_5TranscribeForConditionalGeneration.word_timestamp_head` classifies its output.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3_5TranscribeWordTimestampModel.forward]]

```python
forward(inputs_embeds: FloatTensor, attention_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L470)

**Parameters:**

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_frames, hidden_size)`) : Speech recognition encoder states, before the audio model's `asr_norm` and speaker embeddings.

attention_mask (`torch.Tensor` of shape `(batch_size, num_frames)`, *optional*) : Mask of the valid encoder frames.

The [Nemotron3_5TranscribeWordTimestampModel](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeWordTimestampModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

Returns `num_frames * upsampling_factor` frames.

## Nemotron3_5TranscribeModel[[transformers.Nemotron3_5TranscribeModel]]

#### transformers.Nemotron3_5TranscribeModel[[transformers.Nemotron3_5TranscribeModel]]

```python
transformers.Nemotron3_5TranscribeModel(config: Nemotron3_5TranscribeConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L641)

**Parameters:**

config ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Nemotron 3.5 Transcribe model (speech recognition encoder and speaker diarization model in parallel, linear
projector and Nemotron-H language model), without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3_5TranscribeModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, input_features_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L708)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using `feature_extractor_class`. See `feature_extractor_class.__call__` for details (`processor_class` uses `feature_extractor_class` for processing audios).

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask to avoid performing attention on padding feature indices.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** `Nemotron3_5TranscribeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `Nemotron3_5TranscribeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) and inputs.

The [Nemotron3_5TranscribeModel](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
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
- **audio_hidden_states** (`torch.FloatTensor`, *optional*) -- Projected audio hidden states.

#### get_audio_features[[transformers.Nemotron3_5TranscribeModel.get_audio_features]]

```python
get_audio_features(input_features: FloatTensor, input_features_mask: Tensor, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L654)

**Parameters:**

input_features (`torch.FloatTensor`) : Float values of mel features extracted from the raw speech waveform.

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask to avoid performing attention on padded feature indices.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.22.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) and inputs.

This method is used to get the audio embeddings from input features (a log mel spectrogram), meaning inferring the audio encoder and the multi-modal projector.

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

#### get_word_timestamp_hidden_states[[transformers.Nemotron3_5TranscribeModel.get_word_timestamp_hidden_states]]

```python
get_word_timestamp_hidden_states(asr_last_hidden_state: FloatTensor, input_features_mask: Tensor)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L767)

**Parameters:**

asr_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, audio_config.hidden_size)`) : Output of the speech recognition encoder before its normalization and the speaker embeddings, as returned by [get_audio_features()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeModel.get_audio_features) for `input_features_mask`.

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask of the valid spectrogram frames. Padding must be on the right.

**Returns:**

`torch.FloatTensor` of shape `(batch_size, num_frames * word_timestamp_config.upsampling_factor,
audio_config.hidden_size)`: `upsampling_factor` frames per encoder frame, in the dtype of the word timestamp
encoder.

Runs the word timestamp encoder on the speech recognition encoder states, in the chunks of the encoder: the
states the word timestamp head classifies.

## Nemotron3_5TranscribeForConditionalGeneration[[transformers.Nemotron3_5TranscribeForConditionalGeneration]]

#### transformers.Nemotron3_5TranscribeForConditionalGeneration[[transformers.Nemotron3_5TranscribeForConditionalGeneration]]

```python
transformers.Nemotron3_5TranscribeForConditionalGeneration(config: Nemotron3_5TranscribeConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L808)

**Parameters:**

config ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Nemotron 3.5 Transcribe model (speech recognition encoder and speaker diarization model in parallel, linear
projector and Nemotron-H language model) with a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.22.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Nemotron3_5TranscribeForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, input_features_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L828)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.22.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.22.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using `feature_extractor_class`. See `feature_extractor_class.__call__` for details (`processor_class` uses `feature_extractor_class` for processing audios).

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask to avoid performing attention on padding feature indices.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `Nemotron3_5TranscribeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `Nemotron3_5TranscribeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Nemotron3_5TranscribeConfig](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeConfig)) and inputs.

The [Nemotron3_5TranscribeForConditionalGeneration](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.22.1/en/internal/generation_utils#transformers.Cache) instance.
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
>>> from transformers import Nemotron3_5TranscribeForConditionalGeneration, AutoProcessor

>>> model_id = "nvidia/audio-flamingo-3-hf"
>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = Nemotron3_5TranscribeForConditionalGeneration.from_pretrained(model_id, device_map="auto")
```

#### get_audio_features[[transformers.Nemotron3_5TranscribeForConditionalGeneration.get_audio_features]]

```python
get_audio_features(input_features, input_features_mask, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L820)

**Parameters:**

input_features (`` of shape `(batch_size, sequence_length, feature_dim)`) : The tensors corresponding to the input audio features. Audio features can be obtained using `feature_extractor_class`. See `feature_extractor_class.__call__` for details (`processor_class` uses `feature_extractor_class` for processing audios).

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask of the valid spectrogram frames. Padding must be on the right.

Example:

```python
```

#### get_word_timestamp_logits[[transformers.Nemotron3_5TranscribeForConditionalGeneration.get_word_timestamp_logits]]

```python
get_word_timestamp_logits(asr_last_hidden_state: FloatTensor, input_features_mask: Tensor)
```

[Source](https://github.com/huggingface/transformers/blob/v5.22.1/src/transformers/models/nemotron3_5_transcribe/modeling_nemotron3_5_transcribe.py#L890)

**Parameters:**

asr_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_frames, audio_config.hidden_size)`) : Output of the speech recognition encoder before its normalization and the speaker embeddings, as returned by [get_audio_features()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeForConditionalGeneration.get_audio_features) for `input_features_mask`.

input_features_mask (`torch.Tensor` of shape `(batch_size, feature_sequence_length)`) : Mask of the valid spectrogram frames. Padding must be on the right.

**Returns:**

`torch.FloatTensor` of shape `(batch_size, num_frames * word_timestamp_config.upsampling_factor,
word_timestamp_config.vocab_size)`: raw logits over the pieces of the alignment tokenizer and the blank,
`upsampling_factor` frames per encoder frame (the processor applies the log-softmax).

Logits of the word timestamp head on the speech recognition encoder states, which
[align_word_timestamps()](/docs/transformers/v5.22.1/en/model_doc/nemotron3_5_transcribe#transformers.Nemotron3_5TranscribeProcessor.align_word_timestamps) turns into word times.

Example:

```python
```

### MobileViTV2
https://huggingface.co/docs/transformers/v5.22.1/model_doc/mobilevitv2.md
