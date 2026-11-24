# OmniASR

## Overview

OmniASR (Omnilingual ASR) is a suite of multilingual speech-recognition models from Meta FAIR, introduced in
[Omnilingual ASR: Open-Source Multilingual Speech Recognition for 1600+ Languages](https://huggingface.co/papers/2511.09690), that covers more than
1,600 languages — including hundreds never previously supported by any ASR system. Each model pairs a Wav2Vec2-style
audio encoder (convolutional layers that downsample the 16 kHz waveform ~320× to a 50 Hz frame rate,
followed by a pre-norm Transformer encoder) with one of two heads:

- **CTC variant** (`OmniASRForCTC`, see [OmniASR CTC](./omniasr_ctc)): a single linear projection to the vocabulary, decoded non-autoregressively with greedy CTC.
- **ALM (audio language model) variant** (`OmniASRForConditionalGeneration`, this page): the audio embeddings are linearly projected into a Llama decoder which autoregressively generates the transcription. This variant additionally supports optional **language conditioning**: passing a language code such as `"eng_Latn"` makes the processor write the matching language token into the decoder context, which generally improves transcription quality.

Checkpoints can be found in [this collection](https://huggingface.co/collections/hf-audio/omnilingual-asr-transformers-compatible): 4x ALM and 4x CTC checkpoints of various sizes (300M, 1B, 3B, 7B). Original weights are released by Meta, and can be found on their [GitHub repo](https://github.com/facebookresearch/omnilingual-asr#model-architectures).

This model was contributed by [Eric Bezzam](https://huggingface.co/bezzam).

## Usage

> [!NOTE]
> The simplest way to transcribe audio is with `apply_transcription_request`, which handles the chat template formatting for you, namely it is a convenience wrapper for `apply_chat_template` (see [Chat template](#chat-template) below).

### Simple transcription

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "hf-audio/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

inputs = processor.apply_transcription_request(
    audio="https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
)
inputs.to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

### Language hint

Without a language, the model runs in its language-agnostic mode. Passing a language code (e.g. `"eng_Latn"`,
`"fra_Latn"`, `"cmn_Hans"`) writes the matching language token into the decoder prompt, which generally improves
transcription quality. The list of supported languages and naming convention follows the original [here](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py#L9).
For the languages that have one, the more common ISO 639-1 codes (e.g. `"en"`, `"fr"`, `"zh"`) are also accepted.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "hf-audio/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

audio = "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mandarin_voxcpm_zh.wav"

# Without language hint (language-agnostic)
inputs = processor.apply_transcription_request(audio=audio).to(model.device, dtype=model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(f"Language-agnostic: {processor.decode(generated_ids, skip_special_tokens=True)[0]}")

# With language hint
inputs = processor.apply_transcription_request(audio=audio, language="zh").to(model.device, dtype=model.dtype)
output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(f"Language hint:     {processor.decode(generated_ids, skip_special_tokens=True)[0]}")
```

### Batch inference

Batch inference is possible by passing a list of audios and, if provided, a single language for the whole batch or a
list of languages (one per audio, `None` for the language-agnostic mode).

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "hf-audio/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

audio = [
    "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
    "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mandarin_voxcpm_zh.wav",
]

inputs = processor.apply_transcription_request(audio, language=["en", None])
inputs.to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
for i, text in enumerate(processor.decode(generated_ids, skip_special_tokens=True)):
    print(f"Audio {i + 1}: {text}")
```

### Chat template

The model also accepts chat template inputs. The `apply_transcription_request` usage
[above](#simple-transcription) is a convenience wrapper for `apply_chat_template`. The language hint is given as a
`{"type": "language", ...}` item of the user turn; leaving it out selects the language-agnostic mode.

```python
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "hf-audio/omniasr-llm-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")

conversations = [
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav",
                },
                {"type": "language", "language": "en"},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio",
                    "path": "https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/mandarin_voxcpm_zh.wav",
                },
                {"type": "language", "language": "zh"},
            ],
        },
    ],
]

inputs = processor.apply_chat_template(
    conversations, tokenize=True, add_generation_prompt=True, return_dict=True
)
inputs.to(model.device, dtype=model.dtype)

output_ids = model.generate(**inputs, max_new_tokens=256)
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
for i, text in enumerate(processor.decode(generated_ids, skip_special_tokens=True)):
    print(f"Audio {i + 1}: {text}")
```

### Training

OmniASR can be trained with the loss outputted by the model. Put the target transcript in the assistant turn of the
[chat template](#chat-template) and pass `output_labels=True`. Audio and padding positions are masked automatically.

Note that OmniASR transcribes in lowercase, so the (uppercase) LibriSpeech transcripts are lowercased below.

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor

model_id = "hf-audio/omniasr-llm-300m-v2"
NUM_SAMPLES = 2

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
conversation = [
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": sample["audio"]["array"]},
                # any supported language code, or no `language` item to train the language-agnostic mode
                {"type": "language", "language": "en"},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["text"].lower()}]},
    ]
    for sample in ds.select(range(NUM_SAMPLES))
]

inputs = processor.apply_chat_template(
    conversation,
    tokenize=True,
    return_dict=True,
    # only train on the assistant outputs: mask the labels outside the assistant turns
    return_assistant_tokens_mask=True,
    return_tensors="pt",
    processor_kwargs={"output_labels": True},
).to(model.device, dtype=model.dtype)
inputs["labels"][inputs.pop("assistant_masks") == 0] = -100

loss = model(**inputs).loss
print("Loss:", loss.item())
loss.backward()
```

### Torch compile

For autoregressive transcription, `torch.compile` accelerates the per-token forward passes inside `generate` by
providing a `CompileConfig` object, which requires a static cache.

Note that the `StaticCache` is created once and reused across `generate` calls to avoid recompiles.

On an A100, we observed a speed-up of ~2.3 for a batch size of 4 ([script](https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-example_compile_llm-py)).

```python
import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor, CompileConfig, StaticCache

model_id = "hf-audio/omniasr-llm-300m-v2"
NUM_SAMPLES = 4
num_warmup = 3
max_new_tokens = 256

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForMultimodalLM.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

audio_url = "https://huggingface.co/datasets/bezzam/audio_samples/resolve/main/librispeech_mr_quilter.wav"
inputs = processor.apply_transcription_request(
    audio=[audio_url] * NUM_SAMPLES,
    language="en",
).to(model.device, dtype=model.dtype)

# Created once and reused across `generate` calls, to avoid recompiling at every call
static_cache = StaticCache(
    config=model.config,
    max_cache_len=inputs["input_ids"].shape[1] + max_new_tokens,
)
compile_config = CompileConfig()

# Warmup
with torch.inference_mode():
    for _ in range(num_warmup):
        static_cache.reset()
        _ = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            cache=static_cache, compile_config=compile_config,
        )
torch.cuda.synchronize()

# Apply model
with torch.inference_mode():
    static_cache.reset()
    output_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        cache=static_cache, compile_config=compile_config,
    )
generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
print(processor.decode(generated_ids, skip_special_tokens=True)[0])
```

## OmniASRAudioConfig[[transformers.OmniASRAudioConfig]]

#### transformers.OmniASRAudioConfig[[transformers.OmniASRAudioConfig]]

```python
transformers.OmniASRAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, hidden_size: int = 1024, conv_dim: list[int] | tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512), conv_kernel: list[int] | tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2), conv_stride: list[int] | tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2), conv_bias: bool = True, num_attention_heads: int = 16, num_hidden_layers: int = 24, num_conv_pos_embeddings: int = 128, num_conv_pos_embedding_groups: int = 16, intermediate_size: int = 4096, attention_dropout: float | int = 0.0, hidden_dropout: float | int = 0.1, layerdrop: float | int = 0.1, activation_dropout: float | int = 0.1, initializer_range: float = 0.02, layer_norm_eps: float = 1e-05, hidden_act: str = 'gelu')
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/configuration_omniasr.py#L24)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

conv_dim (`tuple[int]` or `list[int]`, *optional*, defaults to `(512, 512, 512, 512, 512, 512, 512)`) : A tuple of integers defining the number of input and output channels of each 1D convolutional layer in the feature encoder. The length of *conv_dim* defines the number of 1D convolutional layers.

conv_kernel (`tuple[int]` or `list[int]`, *optional*, defaults to `(10, 3, 3, 3, 3, 2, 2)`) : A tuple of integers defining the kernel size of each 1D convolutional layer in the feature encoder. The length of *conv_kernel* defines the number of convolutional layers and has to match the length of *conv_dim*.

conv_stride (`tuple[int]` or `list[int]`, *optional*, defaults to `(5, 2, 2, 2, 2, 2, 2)`) : A tuple of integers defining the stride of each 1D convolutional layer in the feature encoder. The length of *conv_stride* defines the number of convolutional layers and has to match the length of *conv_dim*.

conv_bias (`bool`, *optional*, defaults to `True`) : Whether the 1D convolutional layers have a bias.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_hidden_layers (`int`, *optional*, defaults to `24`) : Number of hidden layers in the Transformer decoder.

num_conv_pos_embeddings (`int`, *optional*, defaults to 128) : Number of convolutional positional embeddings. Defines the kernel size of the 1D convolutional positional embeddings layer.

num_conv_pos_embedding_groups (`int`, *optional*, defaults to 16) : Number of groups of the 1D convolutional positional embeddings layer.

intermediate_size (`int`, *optional*, defaults to `4096`) : Dimension of the MLP representations.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

hidden_dropout (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

layerdrop (`Union[float, int]`, *optional*, defaults to `0.1`) : The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

activation_dropout (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout ratio for activations inside the fully connected layer.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

This is the configuration class to store the configuration of a OmniASRModel. It is used to instantiate a Omniasr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [hf-audio/omniasr-ctc-300m-v2](https://huggingface.co/hf-audio/omniasr-ctc-300m-v2)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import OmniASRAudioConfig, OmniASRAudioModel

>>> # Initializing an OmniASR encoder configuration
>>> configuration = OmniASRAudioConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = OmniASRAudioModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## OmniASRConfig[[transformers.OmniASRConfig]]

#### transformers.OmniASRConfig[[transformers.OmniASRConfig]]

```python
transformers.OmniASRConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, audio_token_id: int = 10289, initializer_range: float = 0.02, bos_token_id: int | None = 0, pad_token_id: int | None = 1, eos_token_id: int | None = 2)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/configuration_omniasr.py#L95)

**Parameters:**

audio_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio backbone.

text_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the text backbone.

audio_token_id (`int`, *optional*, defaults to `10289`) : The audio token index used as a placeholder for input audio.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

bos_token_id (`int`, *optional*, defaults to `0`) : Token id used for beginning-of-stream in the vocabulary.

pad_token_id (`int`, *optional*, defaults to `1`) : Token id used for padding in the vocabulary.

eos_token_id (`int`, *optional*, defaults to `2`) : Token id used for end-of-stream in the vocabulary.

This is the configuration class to store the configuration of a OmniASRModel. It is used to instantiate a Omniasr
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [hf-audio/omniasr-llm-300m-v2](https://huggingface.co/hf-audio/omniasr-llm-300m-v2)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import OmniASRForConditionalGeneration, OmniASRConfig

>>> # Initializing an OmniASR-LLM configuration
>>> configuration = OmniASRConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = OmniASRForConditionalGeneration(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## OmniASRFeatureExtractor[[transformers.OmniASRFeatureExtractor]]

#### transformers.OmniASRFeatureExtractor[[transformers.OmniASRFeatureExtractor]]

```python
transformers.OmniASRFeatureExtractor(feature_size = 1, sampling_rate = 16000, padding_value = 0.0, return_attention_mask = True, do_normalize = True, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/feature_extraction_omniasr.py#L33)

**Parameters:**

feature_size (`int`, *optional*, defaults to 1) : The feature dimension of the extracted features.

sampling_rate (`int`, *optional*, defaults to 16000) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

padding_value (`float`, *optional*, defaults to 0.0) : The value that is used to fill the padding values.

return_attention_mask (`bool`, *optional*, defaults to `True`) : Whether or not `__call__()` should return `attention_mask`. OmniASR needs the mask to know which frames are padding, so batched inputs are only encoded correctly when it is returned.

do_normalize (`bool`, *optional*, defaults to `True`) : Whether or not to normalize each audio sample to zero mean and unit variance. The statistics are computed per sample, over its non-padded samples, to match the [original implementation](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/datasets/utils/audio.py#L23).

Constructs an OmniASR feature extractor.

## OmniASRProcessor[[transformers.OmniASRProcessor]]

#### transformers.OmniASRProcessor[[transformers.OmniASRProcessor]]

```python
transformers.OmniASRProcessor(feature_extractor, tokenizer, chat_template = None, audio_token = 'REDACTED', conv_kernel = None, conv_stride = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/processing_omniasr.py#L64)

**Parameters:**

feature_extractor (`OmniASRFeatureExtractor`) : The feature extractor is a required input.

tokenizer (`TokenizersBackend`) : The tokenizer is a required input.

chat_template (`str`) : A Jinja template to convert lists of messages in a chat into a tokenizable string.

audio_token (`str`, *optional*, defaults to `"<extra_id_1>"`) : The placeholder token that stands for one speech encoder frame in the prompt, i.e. the token of `OmniASRConfig.audio_token_id()`.

conv_kernel (`list[int]`, *optional*) : Kernel size of each convolution of the speech encoder's feature encoder, i.e. `OmniASRAudioConfig.conv_kernel()`. Needed to count the frames an audio input is subsampled to, and therefore how many audio placeholders its prompt holds. Defaults to `[10, 3, 3, 3, 3, 2, 2]`.

conv_stride (`list[int]`, *optional*) : Stride of each convolution of the speech encoder's feature encoder, i.e. `OmniASRAudioConfig.conv_stride()`. Defaults to `[5, 2, 2, 2, 2, 2, 2]`.

Constructs a OmniASRProcessor which wraps a feature extractor and a tokenizer into a single processor.

[OmniASRProcessor](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRProcessor) offers all the functionalities of [OmniASRFeatureExtractor](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRFeatureExtractor) and [TokenizersBackend](/docs/transformers/v5.24.1/en/main_classes/tokenizer#transformers.TokenizersBackend). See the
[~OmniASRFeatureExtractor](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRFeatureExtractor) and [~TokenizersBackend](/docs/transformers/v5.24.1/en/main_classes/tokenizer#transformers.TokenizersBackend) for more information.

#### apply_chat_template[[transformers.OmniASRProcessor.apply_chat_template]]

```python
apply_chat_template(conversation, *args, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/processing_omniasr.py#L210)

Same as [apply_chat_template()](/docs/transformers/v5.24.1/en/main_classes/processors#transformers.ProcessorMixin.apply_chat_template), but resolves the language codes of the conversation first.
The chat template writes the language token as is, so this override is needed to accept ISO 639-1 codes (e.g.
`"en"` for `"eng_Latn"`), and to raise on unsupported codes instead of silently writing an unknown token.

#### apply_transcription_request[[transformers.OmniASRProcessor.apply_transcription_request]]

```python
apply_transcription_request(audio: typing.Union[str, list[str], numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor']], language: str | list[str] | None = None, tokenize: bool = True, **kwargs: *<class 'transformers.processing_utils.ProcessingKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/processing_omniasr.py#L155)

**Parameters:**

audio (`str`, `list[str]`, `np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`) : Audio to transcribe. Strings are interpreted as local paths or URLs and will be loaded automatically by the chat template loader; NumPy arrays and PyTorch tensors are forwarded directly.

language (`str` or `list[str]`, *optional*) : Language code(s) (e.g. `"eng_Latn"` or `["eng_Latn", "fra_Latn"]`), either one for the whole batch or one per audio. ISO 639-1 codes (e.g. `"en"`) are also accepted for the languages that have one. `None` or `"auto"` selects the model's language-agnostic mode; naming the language explicitly gives better transcription quality.

tokenize (`bool`, *optional*, defaults to `True`) : Whether to tokenize the prompts and process the audio. If `False`, the audio is not loaded and the prompts are returned as strings with unexpanded audio placeholders. Passing them with the audio to the processor, `processor(text=prompts, audio=audio)`, gives the same outputs as `tokenize=True`.

- ****kwargs** : Additional keyword arguments forwarded to [apply_chat_template()](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRProcessor.apply_chat_template). Processor kwargs (for example `text_kwargs`, `audio_kwargs`, ...) should be passed as `processor_kwargs={...}`.

**Returns:** [BatchFeature](/docs/transformers/v5.24.1/en/main_classes/feature_extractor#transformers.BatchFeature) or `list[str]`

One prompt per audio when `tokenize=False`, otherwise processor
outputs ready to be passed to [OmniASRForConditionalGeneration.generate()](/docs/transformers/v5.24.1/en/main_classes/text_generation#transformers.GenerationMixin.generate).

Prepare inputs for speech recognition without manually writing the chat template.

## OmniASRAudioModel[[transformers.OmniASRAudioModel]]

#### transformers.OmniASRAudioModel[[transformers.OmniASRAudioModel]]

```python
transformers.OmniASRAudioModel(config: OmniASRAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/modeling_omniasr.py#L347)

**Parameters:**

config ([OmniASRAudioConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRAudioConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The OmniASR speech encoder, which is a Wav2Vec2-style encoder.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.OmniASRAudioModel.forward]]

```python
forward(input_values: typing.Optional[torch.Tensor], padding_mask: typing.Optional[torch.Tensor] = None, output_attention_mask: bool = True, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/modeling_omniasr.py#L368)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See `processor_class.__call__` for details.

padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`) : Padding mask used to pad `input_values`.

output_attention_mask (`bool`, *optional*, defaults to `True`) : Whether to return the subsampled attention mask. Only effective when `padding_mask` is provided.

**Returns:** `OmniASREncoderModelOutput` or `tuple(torch.Tensor)`

A `OmniASREncoderModelOutput` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [OmniASRAudioModel](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRAudioModel) forward method, overrides the `__call__` special method.

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
- **attention_mask** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) -- Mask to avoid performing attention on padding token indices after sequence compression. Returned because the
  sequence length may differ from the input sequence length. Mask values selected in `[0, 1]`:

  - 1 for tokens that are **not masked**,
  - 0 for tokens that are **masked**.

## OmniASRModel[[transformers.OmniASRModel]]

#### transformers.OmniASRModel[[transformers.OmniASRModel]]

```python
transformers.OmniASRModel(config: OmniASRConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/modeling_omniasr.py#L439)

**Parameters:**

config ([OmniASRConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model,
without a language modeling head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.OmniASRModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, input_values: typing.Optional[torch.Tensor] = None, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/modeling_omniasr.py#L500)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See `OmniASRProcessor.__call__()` for details.

padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*) : Padding mask to remove padded parts of audio.

**Returns:** `OmniASRBaseModelOutputWithPast` or `tuple(torch.Tensor)`

A `OmniASRBaseModelOutputWithPast` or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OmniASRConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRConfig)) and inputs.

The [OmniASRModel](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRModel) forward method, overrides the `__call__` special method.

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
- **audio_features** (`torch.Tensor` of shape `(num_audio_frames, hidden_size)`, *optional*) -- Encoder output frames, before the multi-modal projector, for the non-padded audio frames.

## OmniASRForConditionalGeneration[[transformers.OmniASRForConditionalGeneration]]

#### transformers.OmniASRForConditionalGeneration[[transformers.OmniASRForConditionalGeneration]]

```python
transformers.OmniASRForConditionalGeneration(config: OmniASRConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/modeling_omniasr.py#L540)

**Parameters:**

config ([OmniASRConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OmniASR model, which consists of a Wav2Vec2 encoder, a multi-modal projector and a LLama language model.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.OmniASRForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, input_values: typing.Optional[torch.Tensor] = None, padding_mask: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, cache: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.Tensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr/modeling_omniasr.py#L554)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.24.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

input_values (`torch.Tensor` of shape `(batch_size, 1, num_samples)`, *optional*) : Float values of the raw audio waveform, scattered over the audio placeholders of `input_ids`. See [OmniASRModel.forward()](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRModel.forward).

padding_mask (`torch.Tensor` of shape `(batch_size, num_samples)`, *optional*) : Mask to avoid running the speech encoder over padding samples of `input_values`, with 1 for samples that are **not** padding. Distinct from `attention_mask`, which covers the decoder sequence.

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

cache (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `cache` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `cache` are passed, [DynamicCache](/docs/transformers/v5.24.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `cache` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the causal language modeling loss. They are shifted internally, so they must be aligned with the decoder sequence -- i.e. as long as `input_ids` / `inputs_embeds`, with `-100` on the positions that should not contribute to the loss (the audio context, and padding).

use_cache (`bool`, *optional*) : If set to `True`, `cache` key value states are returned and can be used to speed up decoding (see `cache`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.Tensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OmniASRConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRConfig)) and inputs.

The [OmniASRForConditionalGeneration](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRForConditionalGeneration) forward method, overrides the `__call__` special method.

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
>>> from transformers import AutoProcessor, OmniASRForConditionalGeneration
>>> from datasets import load_dataset, Audio

>>> model_id = "hf-audio/omniasr-llm-300m-v2"
>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = OmniASRForConditionalGeneration.from_pretrained(model_id)

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

>>> inputs = processor.apply_transcription_request(ds[0]["audio"]["array"], language="eng_Latn")
>>> generated_ids = model.generate(**inputs, max_new_tokens=256)
>>> transcription = processor.decode(generated_ids, skip_special_tokens=True)
```

### PP-OCRv6_medium_det
https://huggingface.co/docs/transformers/v5.24.1/model_doc/pp_ocrv6_medium_det.md
