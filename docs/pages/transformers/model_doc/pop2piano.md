# Pop2Piano

## Overview

The Pop2Piano model was proposed in [Pop2Piano : Pop Audio-based Piano Cover Generation](https://huggingface.co/papers/2211.00895) by Jongho Choi and Kyogu Lee.

Piano covers of pop music are widely enjoyed, but generating them from music is not a trivial task. It requires great
expertise with playing piano as well as knowing different characteristics and melodies of a song. With Pop2Piano you
can directly generate a cover from a song's audio waveform. It is the first model to directly generate a piano cover
from pop audio without melody and chord extraction modules.

Pop2Piano is an encoder-decoder Transformer model based on [T5](https://huggingface.co/papers/1910.10683). The input audio
is transformed to its waveform and passed to the encoder, which transforms it to a latent representation. The decoder
uses these latent representations to generate token ids in an autoregressive way. Each token id corresponds to one of four
different token types: time, velocity, note and 'special'. The token ids are then decoded to their equivalent MIDI file.

The abstract from the paper is the following:

*Piano covers of pop music are enjoyed by many people. However, the
task of automatically generating piano covers of pop music is still
understudied. This is partly due to the lack of synchronized
{Pop, Piano Cover} data pairs, which made it challenging to apply
the latest data-intensive deep learning-based methods. To leverage
the power of the data-driven approach, we make a large amount of
paired and synchronized {Pop, Piano Cover} data using an automated
pipeline. In this paper, we present Pop2Piano, a Transformer network
that generates piano covers given waveforms of pop music. To the best
of our knowledge, this is the first model to generate a piano cover
directly from pop audio without using melody and chord extraction
modules. We show that Pop2Piano, trained with our dataset, is capable
of producing plausible piano covers.*

This model was contributed by [Susnato Dhar](https://huggingface.co/susnato).
The original code can be found [here](https://github.com/sweetcocoa/pop2piano).

## Usage tips

* To use Pop2Piano, you will need to install the 🤗 Transformers library, as well as the following third party modules:  

```bash
pip install pretty-midi==0.2.9 essentia==2.1b6.dev1034 librosa scipy
```

Please note that you may need to restart your runtime after installation.

* Pop2Piano is an Encoder-Decoder based model like T5.
* Pop2Piano can be used to generate midi-audio files for a given audio sequence.
* Choosing different composers in `Pop2PianoForConditionalGeneration.generate()` can lead to variety of different results.
* Setting the sampling rate to 44.1 kHz when loading the audio file can give good performance.
* Though Pop2Piano was mainly trained on Korean Pop music, it also does pretty well on other Western Pop or Hip Hop songs.

## Examples

- Example using HuggingFace Dataset:

```python
from datasets import load_dataset

from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano", device_map="auto")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")
ds = load_dataset("sweetcocoa/pop2piano_ci", split="test")

inputs = processor(
    audio=ds["audio"][0]["array"], sampling_rate=ds["audio"][0]["sampling_rate"], return_tensors="pt"
)
model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"][0]
tokenizer_output.write("./Outputs/midi_output.mid")
```

- Example using your own audio file:

```python
import librosa

from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

audio, sr = librosa.load("<your_audio_file_here>", sr=44100)  # feel free to change the sr to a suitable value.
model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano", device_map="auto")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

inputs = processor(audio=audio, sampling_rate=sr, return_tensors="pt").to(model.device)
model_output = model.generate(input_features=inputs["input_features"], composer="composer1")
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"][0]
tokenizer_output.write("./Outputs/midi_output.mid")
```

- Example of processing multiple audio files in batch:

```python
import librosa

from transformers import Pop2PianoForConditionalGeneration, Pop2PianoProcessor

# feel free to change the sr to a suitable value.
audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)
audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano", device_map="auto")
processor = Pop2PianoProcessor.from_pretrained("sweetcocoa/pop2piano")

inputs = processor(audio=[audio1, audio2], sampling_rate=[sr1, sr2], return_attention_mask=True, return_tensors="pt").to(model.device)
# Since we now generating in batch(2 audios) we must pass the attention_mask
model_output = model.generate(
    input_features=inputs["input_features"],
    attention_mask=inputs["attention_mask"],
    composer="composer1",
)
tokenizer_output = processor.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"]

# Since we now have 2 generated MIDI files
tokenizer_output[0].write("./Outputs/midi_output1.mid")
tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

- Example of processing multiple audio files in batch (Using `Pop2PianoFeatureExtractor` and `Pop2PianoTokenizer`):

```python
import librosa

from transformers import Pop2PianoFeatureExtractor, Pop2PianoForConditionalGeneration, Pop2PianoTokenizer

# feel free to change the sr to a suitable value.
audio1, sr1 = librosa.load("<your_first_audio_file_here>", sr=44100)
audio2, sr2 = librosa.load("<your_second_audio_file_here>", sr=44100)
model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano", device_map="auto")
feature_extractor = Pop2PianoFeatureExtractor.from_pretrained("sweetcocoa/pop2piano")
tokenizer = Pop2PianoTokenizer.from_pretrained("sweetcocoa/pop2piano")

inputs = feature_extractor(
    audio=[audio1, audio2],
    sampling_rate=[sr1, sr2],
    return_attention_mask=True,
    return_tensors="pt",
)
# Since we now generating in batch(2 audios) we must pass the attention_mask
model_output = model.generate(
    input_features=inputs["input_features"],
    attention_mask=inputs["attention_mask"],
    composer="composer1",
)
tokenizer_output = tokenizer.batch_decode(
    token_ids=model_output, feature_extractor_output=inputs
)["pretty_midi_objects"]

# Since we now have 2 generated MIDI files
tokenizer_output[0].write("./Outputs/midi_output1.mid")
tokenizer_output[1].write("./Outputs/midi_output2.mid")
```

## Pop2PianoConfig[[transformers.Pop2PianoConfig]]

#### transformers.Pop2PianoConfig[[transformers.Pop2PianoConfig]]

```python
transformers.Pop2PianoConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, is_encoder_decoder: bool = True, vocab_size: int = 2400, composer_vocab_size: int = 21, d_model: int = 512, d_kv: int = 64, d_ff: int = 2048, num_layers: int = 6, num_decoder_layers: int | None = None, num_heads: int = 8, relative_attention_num_buckets: int = 32, relative_attention_max_distance: int = 128, dropout_rate: float | int = 0.1, layer_norm_epsilon: float = 1e-06, initializer_factor: float = 1.0, feed_forward_proj: str = 'gated-gelu', use_cache: bool = True, pad_token_id: int | None = 0, eos_token_id: int | list[int] | None = 1, dense_act_fn: str = 'relu', is_decoder: bool = False, tie_word_embeddings: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/configuration_pop2piano.py#L24)

**Parameters:**

is_encoder_decoder (`bool`, *optional*, defaults to `True`) : Whether the model is used as an encoder/decoder or not.

vocab_size (`int`, *optional*, defaults to `2400`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

composer_vocab_size (`int`, *optional*, defaults to 21) : Denotes the number of composers.

d_model (`int`, *optional*, defaults to `512`) : Size of the encoder layers and the pooler layer.

d_kv (`int`, *optional*, defaults to `64`) : Size of the key, query, value projections per attention head. The `inner_dim` of the projection layer will be defined as `num_heads * d_kv`.

d_ff (`int`, *optional*, defaults to `2048`) : Dimension of the MLP representations.

num_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

num_decoder_layers (`int`, *optional*) : Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.

num_heads (`int`, *optional*, defaults to `8`) : Number of attention heads for each attention layer in the Transformer decoder.

relative_attention_num_buckets (`int`, *optional*, defaults to 32) : The number of buckets to use for each attention layer.

relative_attention_max_distance (`int`, *optional*, defaults to 128) : The maximum distance of the longer sequences for the bucket separation.

dropout_rate (`Union[float, int]`, *optional*, defaults to `0.1`) : The ratio for all dropout layers.

layer_norm_epsilon (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

initializer_factor (`float`, *optional*, defaults to `1.0`) : A factor for initializing all weight matrices (should be kept to 1, used internally for initialization testing).

feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`) : Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*, defaults to `0`) : Token id used for padding in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*, defaults to `1`) : Token id used for end-of-stream in the vocabulary.

dense_act_fn (`string`, *optional*, defaults to `"relu"`) : Type of Activation Function to be used in `Pop2PianoDenseActDense` and in `Pop2PianoDenseGatedActDense`.

is_decoder (`bool`, *optional*, defaults to `False`) : Whether the model is used as a decoder or not. If `False`, the model is used as an encoder.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a Pop2PianoModel. It is used to instantiate a Pop2Piano
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [sweetcocoa/pop2piano](https://huggingface.co/sweetcocoa/pop2piano)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Pop2PianoFeatureExtractor[[transformers.Pop2PianoFeatureExtractor]]

#### transformers.Pop2PianoFeatureExtractor[[transformers.Pop2PianoFeatureExtractor]]

```python
transformers.Pop2PianoFeatureExtractor(sampling_rate: int = 22050, padding_value: int = 0, window_size: int = 4096, hop_length: int = 1024, min_frequency: float = 10.0, feature_size: int = 512, num_bars: int = 2, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/feature_extraction_pop2piano.py#L49)

**Parameters:**

sampling_rate (`int`, *optional*, defaults to 22050) : Target Sampling rate of audio signal. It's the sampling rate that we forward to the model.

padding_value (`int`, *optional*, defaults to 0) : Padding value used to pad the audio. Should correspond to silences.

window_size (`int`, *optional*, defaults to 4096) : Length of the window in samples to which the Fourier transform is applied.

hop_length (`int`, *optional*, defaults to 1024) : Step size between each window of the waveform, in samples.

min_frequency (`float`, *optional*, defaults to 10.0) : Lowest frequency that will be used in the log-mel spectrogram.

feature_size (`int`, *optional*, defaults to 512) : The feature dimension of the extracted features.

num_bars (`int`, *optional*, defaults to 2) : Determines interval between each sequence.

Constructs a Pop2Piano feature extractor.

This feature extractor inherits from [SequenceFeatureExtractor](/docs/transformers/v5.19.0/en/main_classes/feature_extractor#transformers.SequenceFeatureExtractor) which contains
most of the main methods. Users should refer to this superclass for more information regarding those methods.

This class extracts rhythm and preprocesses the audio before it is passed to the model. First the audio is passed
to `RhythmExtractor2013` algorithm which extracts the beat_times, beat positions and estimates their confidence as
well as tempo in bpm, then beat_times is interpolated and to get beatsteps. Later we calculate
extrapolated_beatsteps from it to be used in tokenizer. On the other hand audio is resampled to self.sampling_rate
and preprocessed and then log mel spectogram is computed from that to be used in our transformer model.

#### __call__[[transformers.Pop2PianoFeatureExtractor.__call__]]

```python
__call__(audio: numpy.ndarray | list[float] | list[numpy.ndarray] | list[list[float]], sampling_rate: int | list[int], steps_per_beat: int = 2, resample: bool | None = True, return_attention_mask: bool | None = False, return_tensors: str | transformers.utils.generic.TensorType | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/feature_extraction_pop2piano.py#L341)

**Parameters:**

audio (`np.ndarray`, `List`) : The audio or batch of audio to be processed. Each audio can be a numpy array, a list of float values, a list of numpy arrays or a list of list of float values.

sampling_rate (`int`) : The sampling rate at which the `audio` input was sampled. It is strongly recommended to pass `sampling_rate` at the forward call to prevent silent errors.

steps_per_beat (`int`, *optional*, defaults to 2) : This is used in interpolating `beat_times`.

resample (`bool`, *optional*, defaults to `True`) : Determines whether to resample the audio to `sampling_rate` or not before processing. Must be True during inference.

return_attention_mask (`bool` *optional*, defaults to `False`) : Denotes if attention_mask for input_features, beatsteps and extrapolated_beatstep will be given as output or not. Automatically set to True for batched inputs.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are: - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return Numpy `np.ndarray` objects. If nothing is specified, it will return list of `np.ndarray` arrays.

Main method to featurize and prepare for the model.

## Pop2PianoForConditionalGeneration[[transformers.Pop2PianoForConditionalGeneration]]

#### transformers.Pop2PianoForConditionalGeneration[[transformers.Pop2PianoForConditionalGeneration]]

```python
transformers.Pop2PianoForConditionalGeneration(config: Pop2PianoConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/modeling_pop2piano.py#L763)

**Parameters:**

config ([Pop2PianoConfig](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Pop2Piano Model with a `language modeling` head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Pop2PianoForConditionalGeneration.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.FloatTensor] = None, decoder_input_ids: typing.Optional[torch.LongTensor] = None, decoder_attention_mask: typing.Optional[torch.BoolTensor] = None, encoder_outputs: tuple[tuple[torch.Tensor]] | None = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, input_features: typing.Optional[torch.FloatTensor] = None, decoder_inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/modeling_pop2piano.py#L851)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`) : Indices of input sequence tokens in the vocabulary. Pop2Piano is a model with relative position embeddings so you should be able to pad the inputs on both the right and the left. Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for detail. [What are input IDs?](../glossary#input-ids) To know more on how to prepare `input_ids` for pretraining take a look a [Pop2Piano Training](./Pop2Piano#training).

attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

decoder_input_ids (`torch.LongTensor` of shape `(batch_size, target_sequence_length)`, *optional*) : Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details. [What are decoder input IDs?](../glossary#decoder-input-ids) Pop2Piano uses the `pad_token_id` as the starting token for `decoder_input_ids` generation. If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see `past_key_values`). To know more on how to prepare

decoder_attention_mask (`torch.BoolTensor` of shape `(batch_size, target_sequence_length)`, *optional*) : Default behavior: generate a tensor that ignores pad tokens in `decoder_input_ids`. Causal mask will also be used by default.

encoder_outputs (`tuple[tuple[torch.Tensor]]`, *optional*) : Tuple consists of (`last_hidden_state`, *optional*: `hidden_states`, *optional*: `attentions`) `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) is a sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, feature_dim)`, *optional*) : The tensors corresponding to the input audio features. Audio features can be obtained using [Pop2PianoFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoFeatureExtractor). See [Pop2PianoFeatureExtractor.__call__()](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoFeatureExtractor.__call__) for details ([Pop2PianoProcessor](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoProcessor) uses [Pop2PianoFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoFeatureExtractor) for processing audios).

decoder_inputs_embeds (`torch.FloatTensor` of shape `(batch_size, target_sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `decoder_input_ids` you can choose to directly pass an embedded representation. If `past_key_values` is used, optionally only the last `decoder_inputs_embeds` have to be input (see `past_key_values`). This is useful if you want more control over how to convert `decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.  If `decoder_input_ids` and `decoder_inputs_embeds` are both unset, `decoder_inputs_embeds` takes the value of `inputs_embeds`.

labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*) : Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ..., config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [Seq2SeqLMOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or `tuple(torch.FloatTensor)`

A [Seq2SeqLMOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.Seq2SeqLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Pop2PianoConfig](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoConfig)) and inputs.

The [Pop2PianoForConditionalGeneration](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`EncoderDecoderCache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [EncoderDecoderCache](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.EncoderDecoderCache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

  Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
  blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.
- **decoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
- **decoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.
- **cross_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
  weighted average in the cross-attention heads.
- **encoder_last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) -- Sequence of hidden-states at the output of the last layer of the encoder of the model.
- **encoder_hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
- **encoder_attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
  self-attention heads.

Example:

```python
>>> from transformers import AutoProcessor, Pop2PianoForConditionalGeneration
>>> from datasets import load_dataset
>>> import torch

>>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
>>> dataset = dataset.sort("id")
>>> sampling_rate = dataset.features["audio"].sampling_rate

>>> processor = AutoProcessor.from_pretrained("sweetcocoa/pop2piano")
>>> model = Pop2PianoForConditionalGeneration.from_pretrained("sweetcocoa/pop2piano")

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

#### generate[[transformers.Pop2PianoForConditionalGeneration.generate]]

```python
generate(input_features, attention_mask = None, composer = 'composer1', generation_config = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/modeling_pop2piano.py#L953)

**Parameters:**

input_features (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : This is the featurized version of audio generated by `Pop2PianoFeatureExtractor`.

attention_mask : For batched generation `input_features` are padded to have the same shape across all examples. `attention_mask` helps to determine which areas were padded and which were not. - 1 for tokens that are **not padded**, - 0 for tokens that are **padded**.

composer (`str`, *optional*, defaults to `"composer1"`) : This value is passed to `Pop2PianoConcatEmbeddingToMel` to generate different embeddings for each `"composer"`. Please make sure that the composer value is present in `composer_to_feature_token` in `generation_config`. For an example please see https://huggingface.co/sweetcocoa/pop2piano/blob/main/generation_config.json .

generation_config (`~generation.GenerationConfig`, *optional*) : The generation configuration to be used as base parametrization for the generation call. `**kwargs` passed to generate matching the attributes of `generation_config` will override them. If `generation_config` is not provided, the default will be used, which had the following loading priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model configuration. Please note that unspecified parameters will inherit [GenerationConfig](/docs/transformers/v5.19.0/en/main_classes/text_generation#transformers.GenerationConfig)'s default values, whose documentation should be checked to parameterize generation.

kwargs : Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

**Returns:** [ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) or `torch.LongTensor`

A [ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) (if `return_dict_in_generate=True`
or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.
Since Pop2Piano is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
[ModelOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.utils.ModelOutput) types are:
- [GenerateEncoderDecoderOutput](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.generation.GenerateEncoderDecoderOutput),
- [GenerateBeamEncoderDecoderOutput](/docs/transformers/v5.19.0/en/internal/generation_utils#transformers.generation.GenerateBeamEncoderDecoderOutput)

Generates token ids for midi outputs.

Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
model's default generation configuration. You can override any `generation_config` by passing the corresponding
parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`. For an overview of generation
strategies and code examples, check out the [following guide](./generation_strategies).

## Pop2PianoTokenizer[[transformers.Pop2PianoTokenizer]]

#### transformers.Pop2PianoTokenizer[[transformers.Pop2PianoTokenizer]]

```python
transformers.Pop2PianoTokenizer(vocab, default_velocity = 77, num_bars = 2, unk_token = '-1', eos_token = '1', pad_token = '0', bos_token = '2', **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/tokenization_pop2piano.py#L62)

**Parameters:**

vocab (`str`) : Path to the vocab file which contains the vocabulary.

default_velocity (`int`, *optional*, defaults to 77) : Determines the default velocity to be used while creating midi Notes.

num_bars (`int`, *optional*, defaults to 2) : Determines cutoff_time_idx in for each token.

unk_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to `"-1"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

eos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 1) : The end of sequence token.

pad_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 0) : A special token used to make arrays of tokens the same size for batching purpose. Will then be ignored by attention mechanisms or loss computation.

bos_token (`str` or `tokenizers.AddedToken`, *optional*, defaults to 2) : The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.

Constructs a Pop2Piano tokenizer. This tokenizer does not require training.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v5.19.0/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### __call__[[transformers.Pop2PianoTokenizer.__call__]]

```python
__call__(notes: numpy.ndarray | list[pretty_midi.Note] | list[list[pretty_midi.Note]], padding: bool | str | transformers.utils.generic.PaddingStrategy = False, truncation: bool | str | transformers.tokenization_utils_base.TruncationStrategy | None = None, max_length: int | None = None, pad_to_multiple_of: int | None = None, return_attention_mask: bool | None = None, return_tensors: str | transformers.utils.generic.TensorType | None = None, verbose: bool = True, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/tokenization_pop2piano.py#L473)

**Parameters:**

notes (`numpy.ndarray` of shape `[batch_size, max_sequence_length, 4]` or `list` of `pretty_midi.Note` objects) : This represents the midi notes.  If `notes` is a `numpy.ndarray`: - Each sequence must have 4 values, they are `onset idx`, `offset idx`, `pitch` and `velocity`. If `notes` is a `list` containing `pretty_midi.Note` objects: - Each sequence must have 4 attributes, they are `start`, `end`, `pitch` and `velocity`.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence if provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided. - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters. If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

return_attention_mask (`bool`, *optional*) : Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer's default, defined by the `return_outputs` attribute.  [What are attention masks?](../glossary#attention-mask)

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return Numpy `np.ndarray` objects.

verbose (`bool`, *optional*, defaults to `True`) : Whether or not to print more information and warnings.

**Returns:**

`BatchEncoding` containing the token_ids.

This is the `__call__` method for `Pop2PianoTokenizer`. It converts the midi notes to the transformer generated
token ids.

## Pop2PianoProcessor[[transformers.Pop2PianoProcessor]]

#### transformers.Pop2PianoProcessor[[transformers.Pop2PianoProcessor]]

```python
transformers.Pop2PianoProcessor(feature_extractor, tokenizer)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/processing_pop2piano.py#L29)

**Parameters:**

feature_extractor (`Pop2PianoFeatureExtractor`) : The feature extractor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

Constructs a Pop2PianoProcessor which wraps a feature extractor and a tokenizer into a single processor.

[Pop2PianoProcessor](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoProcessor) offers all the functionalities of [Pop2PianoFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoFeatureExtractor) and `tokenizer_class`. See the
[~Pop2PianoFeatureExtractor](/docs/transformers/v5.19.0/en/model_doc/pop2piano#transformers.Pop2PianoFeatureExtractor) and `~tokenizer_class` for more information.

#### __call__[[transformers.Pop2PianoProcessor.__call__]]

```python
__call__(audio: numpy.ndarray | list[float] | list[numpy.ndarray] = None, sampling_rate: int | list[int] | None = None, steps_per_beat: int = 2, resample: bool | None = True, notes: list | transformers.utils.generic.TensorType = None, padding: bool | str | transformers.utils.generic.PaddingStrategy = False, truncation: bool | str | transformers.tokenization_utils_base.TruncationStrategy = None, max_length: int | None = None, pad_to_multiple_of: int | None = None, verbose: bool = True, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/pop2piano/processing_pop2piano.py#L33)

**Parameters:**

audio (`Union[numpy.ndarray, list[float], list[numpy.ndarray]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

sampling_rate (`int` or `list[int]`, *optional*) : The sampling rate of the input audio in Hz. This should match the sampling rate used by the feature extractor. If not provided, the default sampling rate from the processor configuration will be used.

steps_per_beat (`int`, *optional*, defaults to `2`) : The number of time steps per musical beat. This parameter controls the temporal resolution of the musical representation. A higher value provides finer temporal granularity but increases the sequence length. Used when processing audio to extract musical features.

resample (`bool`, *optional*, defaults to `True`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

notes (`list` or `TensorType`, *optional*) : Pre-extracted musical notes in MIDI format. When provided, the processor skips audio feature extraction and directly processes the notes through the tokenizer. Each note should be represented as a list or tensor containing pitch, velocity, and timing information.

padding (bool, str or [PaddingStrategy](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (bool, str or [TruncationStrategy](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided. - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable using Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

verbose (`bool`, *optional*, defaults to `True`) : Whether or not to print more information and warnings.

return_tensors (`str` or [TensorType](/docs/transformers/v5.19.0/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

**Returns:**

`Union[~feature_extraction_utils.BatchFeature, ~tokenization_utils_base.BatchEncoding]`

### EoMT
https://huggingface.co/docs/transformers/v5.19.0/model_doc/eomt.md
