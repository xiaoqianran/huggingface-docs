# OmniASR CTC

## Overview

Check the [OmniASR](./omniasr) page for more information about this model family and checkpoints.

## Usage

The CTC variant transcribes in lowercase without punctuation, and does not take a language hint.

### Simple transcription

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "hf-audio/omniasr-ctc-300m-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

inputs = processor(ds[0]["audio"]["array"], sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
print(processor.decode(outputs, skip_special_tokens=True)[0])
```

### Batch inference

Batch inference is possible by passing a list of audios, which the processor pads to the longest one.

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "hf-audio/omniasr-ctc-300m-v2"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]

inputs = processor(speech_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)
outputs = model.generate(**inputs)
for i, text in enumerate(processor.decode(outputs, skip_special_tokens=True)):
    print(f"Audio {i + 1}: {text}")
```

### Training

The model can be trained with the loss it outputs. Passing `text` to the processor prepares the CTC `labels`, with
padding positions masked out. Note that OmniASR transcribes in lowercase, so the (uppercase) LibriSpeech transcripts
are lowercased below.

```python
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor

model_id = "hf-audio/omniasr-ctc-300m-v2"
NUM_SAMPLES = 5

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, device_map="auto")
model.train()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]
text_samples = [text.lower() for text in ds["text"][:NUM_SAMPLES]]

# passing `text` to the processor will prepare inputs' `labels` key
inputs = processor(audio=speech_samples, text=text_samples, sampling_rate=processor.feature_extractor.sampling_rate)
inputs.to(model.device, dtype=model.dtype)

outputs = model(**inputs)
print("Loss:", outputs.loss.item())
outputs.loss.backward()
```

### Torch compile

The CTC variant runs a single forward pass, which is compiled by passing a `CompileConfig` to `generate`. Padding all
inputs to the same length (here 30 seconds) keeps the input shapes static, so that new inputs don't trigger a
recompilation.

On an A100, we observed a speed-up of ~1.3 for a batch size of 4 ([script](https://gist.github.com/ebezzam/26af2bd40fa207af322de39701179650#file-example_compile_ctc-py)).

```python
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, AutoProcessor, CompileConfig

model_id = "hf-audio/omniasr-ctc-300m-v2"
NUM_SAMPLES = 4
num_warmup = 3

processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForCTC.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
speech_samples = [el["array"] for el in ds["audio"][:NUM_SAMPLES]]

inputs = processor(
    speech_samples,
    sampling_rate=processor.feature_extractor.sampling_rate,
    padding="max_length",
    max_length=30 * processor.feature_extractor.sampling_rate,
).to(model.device, dtype=model.dtype)

compile_config = CompileConfig(fullgraph=True, mode="reduce-overhead")

# Warmup
for _ in range(num_warmup):
    _ = model.generate(**inputs, compile_config=compile_config)
torch.cuda.synchronize()

# Apply model
outputs = model.generate(**inputs, compile_config=compile_config)
print(processor.decode(outputs, skip_special_tokens=True)[0])
```

## OmniASRCTCConfig[[transformers.OmniASRCTCConfig]]

#### transformers.OmniASRCTCConfig[[transformers.OmniASRCTCConfig]]

```python
transformers.OmniASRCTCConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 10288, ctc_loss_reduction: str = 'mean', ctc_zero_infinity: bool = False, initializer_range: float = 0.02, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, blank_token_id: int = 0, pad_token_id: int | None = 1)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr_ctc/configuration_omniasr_ctc.py#L24)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `10288`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

ctc_loss_reduction (`str`, *optional*, defaults to `"mean"`) : Specifies the reduction to apply to the output of `torch.nn.CTCLoss`. Only relevant when training an instance of [OmniASRForCTC](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRForCTC).

ctc_zero_infinity (`bool`, *optional*, defaults to `False`) : Whether to zero infinite losses and the associated gradients of `torch.nn.CTCLoss`. Infinite losses mainly occur when the inputs are too short to be aligned to the targets. Only relevant when training an instance of [OmniASRForCTC](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRForCTC).

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

audio_config (`Union[dict, PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio encoder. Defaults to [OmniASRAudioConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRAudioConfig).

blank_token_id (`int`, *optional*, defaults to 0) : The token ID of the blank symbol used by the CTC loss.

pad_token_id (`int`, *optional*, defaults to `1`) : Token id used for padding in the vocabulary.

This is the configuration class to store the configuration of a OmniASRCTCModel. It is used to instantiate a Omniasr Ctc
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [hf-audio/omniasr-ctc-300m-v2](https://huggingface.co/hf-audio/omniasr-ctc-300m-v2)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.24.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import OmniASRForCTC, OmniASRCTCConfig

>>> # Initializing an OmniASR-CTC configuration
>>> configuration = OmniASRCTCConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = OmniASRForCTC(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## OmniASRCTCProcessor[[transformers.OmniASRCTCProcessor]]

#### transformers.OmniASRCTCProcessor[[transformers.OmniASRCTCProcessor]]

```python
transformers.OmniASRCTCProcessor(feature_extractor, tokenizer)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr_ctc/processing_omniasr_ctc.py#L23)

**Parameters:**

feature_extractor (`OmniASRFeatureExtractor`) : The feature extractor is a required input.

tokenizer (`ParakeetTokenizer`) : The tokenizer is a required input.

Constructs a OmniASRCTCProcessor which wraps a feature extractor and a tokenizer into a single processor.

[OmniASRCTCProcessor](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRCTCProcessor) offers all the functionalities of [OmniASRFeatureExtractor](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRFeatureExtractor) and [ParakeetTokenizer](/docs/transformers/v5.24.1/en/model_doc/parakeet#transformers.ParakeetTokenizer). See the
[~OmniASRFeatureExtractor](/docs/transformers/v5.24.1/en/model_doc/omniasr#transformers.OmniASRFeatureExtractor) and [~ParakeetTokenizer](/docs/transformers/v5.24.1/en/model_doc/parakeet#transformers.ParakeetTokenizer) for more information.

## OmniASRCTCModel[[transformers.OmniASRCTCModel]]

#### transformers.OmniASRCTCModel[[transformers.OmniASRCTCModel]]

```python
transformers.OmniASRCTCModel(config: OmniASRCTCConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr_ctc/modeling_omniasr_ctc.py#L99)

**Parameters:**

config ([OmniASRCTCConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRCTCConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The OmniASR CTC model, i.e. the speech encoder without the CTC head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.OmniASRCTCModel.forward]]

```python
forward(input_values: Tensor, padding_mask: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr_ctc/modeling_omniasr_ctc.py#L105)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See `OmniASRCTCProcessor.__call__()` for details.

padding_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Padding mask used to pad `input_values`.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.Tensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([OmniASRCTCConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRCTCConfig)) and inputs.

The [OmniASRCTCModel](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRCTCModel) forward method, overrides the `__call__` special method.

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

Example:

```python
>>> from transformers import AutoProcessor, OmniASRCTCModel
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("hf-audio/omniasr-ctc-300m-v2")
>>> model = OmniASRCTCModel.from_pretrained("hf-audio/omniasr-ctc-300m-v2")

>>> # audio file is decoded on the fly
>>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
>>> with torch.no_grad():
...     logits = model(**inputs).logits
>>> predicted_ids = torch.argmax(logits, dim=-1)

>>> # transcribe speech
>>> transcription = processor.batch_decode(predicted_ids)
>>> transcription[0]
...

>>> inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

>>> # compute loss
>>> loss = model(**inputs).loss
>>> round(loss.item(), 2)
...
```

## OmniASRForCTC[[transformers.OmniASRForCTC]]

#### transformers.OmniASRForCTC[[transformers.OmniASRForCTC]]

```python
transformers.OmniASRForCTC(config: OmniASRCTCConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr_ctc/modeling_omniasr_ctc.py#L125)

**Parameters:**

config ([OmniASRCTCConfig](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRCTCConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

OmniASR Encoder with a Connectionist Temporal Classification (CTC) head.

This model inherits from [PreTrainedModel](/docs/transformers/v5.24.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.OmniASRForCTC.forward]]

```python
forward(input_values: Tensor, padding_mask: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.Tensor] = None, **kwargs: *<class 'transformers.utils.generic.TransformersKwargs'>)
```

[Source](https://github.com/huggingface/transformers/blob/v5.24.1/src/transformers/models/omniasr_ctc/modeling_omniasr_ctc.py#L135)

**Parameters:**

input_values (`torch.Tensor` of shape `(batch_size, sequence_length)`) : Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file into an array of type `list[float]`, a `numpy.ndarray` or a `torch.Tensor`, *e.g.* via the torchcodec library (`pip install torchcodec`) or the soundfile library (`pip install soundfile`). To prepare the array into `input_values`, the [AutoProcessor](/docs/transformers/v5.24.1/en/model_doc/auto#transformers.AutoProcessor) should be used for padding and conversion into a tensor of type `torch.Tensor`. See `processor_class.__call__` for details.

padding_mask (`torch.Tensor` of shape `(batch_size, 1, sequence_length)`) : Padding mask used to pad `input_values`.

labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

**Returns:** [CausalLMOutput](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or `tuple(torch.Tensor)`

A [CausalLMOutput](/docs/transformers/v5.24.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutput) or a tuple of
`torch.Tensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [OmniASRForCTC](/docs/transformers/v5.24.1/en/model_doc/omniasr_ctc#transformers.OmniASRForCTC) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.Tensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.Tensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.Tensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.Tensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.Tensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.Tensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoProcessor, OmniASRForCTC
>>> from datasets import load_dataset, Audio

>>> model_id = "hf-audio/omniasr-ctc-300m-v2"
>>> processor = AutoProcessor.from_pretrained(model_id)
>>> model = OmniASRForCTC.from_pretrained(model_id)

>>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
>>> ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))

>>> inputs = processor(ds[0]["audio"]["array"], text=ds[0]["text"])
>>> outputs = model(**inputs)

>>> print(outputs.loss)
```

### Nemotron 3 Diarization
https://huggingface.co/docs/transformers/v5.24.1/model_doc/nemotron3_diarization.md
