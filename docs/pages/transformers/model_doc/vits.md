# VITS

[VITS (Variational Inference with adversarial learning for end-to-end Text-to-Speech)](https://huggingface.co/papers/2106.06103) is an end-to-end speech synthesis model, simplifying the traditional two-stage text-to-speech (TTS) systems. It's unique because it directly synthesizes speech from text using variational inference, adversarial learning, and normalizing flows to produce natural and expressive speech with diverse rhythms and intonations.

You can find all the original VITS checkpoints under the [AI at Meta](https://huggingface.co/facebook?search_models=mms-tts) organization.

> [!TIP]
> Click on the VITS models in the right sidebar for more examples of how to apply VITS.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v5.15.1/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoModel) class.

```python
from scipy.io.wavfile import write

from transformers import pipeline, set_seed

set_seed(555)

pipe = pipeline(
    task="text-to-speech",
    model="facebook/mms-tts-eng",
    device=0
)

speech = pipe("Hello, my dog is cute")

# Extract audio data and sampling rate
audio_data = speech["audio"]
sampling_rate = speech["sampling_rate"]

# Save as WAV file
write("hello.wav", sampling_rate, audio_data.squeeze())
```

```python
import scipy
import torch
from IPython.display import Audio

from transformers import AutoTokenizer, VitsModel, set_seed

tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
model = VitsModel.from_pretrained("facebook/mms-tts-eng", device_map="auto")
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(model.device)

set_seed(555)

with torch.no_grad():
    outputs = model(**inputs)

waveform = outputs.waveform[0]
scipy.io.wavfile.write("hello.wav", rate=model.config.sampling_rate, data=waveform)

# display in Colab notebook
Audio(waveform, rate=model.config.sampling_rate)
```

## Notes

- Set a seed for reproducibility because VITS synthesizes speech non-deterministically.
- For languages with non-Roman alphabets (Korean, Arabic, etc.), install the [uroman](https://github.com/isi-nlp/uroman) package to preprocess the text inputs to the Roman alphabet. You can check if the tokenizer requires uroman as shown below.

   ```py
   # pip install -U uroman
   from transformers import VitsTokenizer

   tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
   print(tokenizer.is_uroman)
   ```

   If your language requires uroman, the tokenizer automatically applies it to the text inputs. Python >= 3.10 doesn't require any additional preprocessing steps. For Python < 3.10, follow the steps below.

   ```bash
   git clone https://github.com/isi-nlp/uroman.git
   cd uroman
   export UROMAN=$(pwd)
   ```

   Create a function to preprocess the inputs. You can either use the bash variable `UROMAN` or pass the directory path directly to the function.

   ```py
   import torch
   from transformers import VitsTokenizer, VitsModel, set_seed
   import os
   import subprocess

   tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-kor")
   model = VitsModel.from_pretrained("facebook/mms-tts-kor", device_map="auto")

   def uromanize(input_string, uroman_path):
       """Convert non-Roman strings to Roman using the `uroman` perl package."""
       script_path = os.path.join(uroman_path, "bin", "uroman.pl")

       command = ["perl", script_path]

       process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
       # Execute the perl command
       stdout, stderr = process.communicate(input=input_string.encode())

       if process.returncode != 0:
           raise ValueError(f"Error {process.returncode}: {stderr.decode()}")

       # Return the output as a string and skip the new-line character at the end
       return stdout.decode()[:-1]

   text = "이봐 무슨 일이야"
   uromanized_text = uromanize(text, uroman_path=os.environ["UROMAN"])

   inputs = tokenizer(text=uromanized_text, return_tensors="pt").to(model.device)

   set_seed(555)  # make deterministic
   with torch.no_grad():
      outputs = model(inputs["input_ids"])

   waveform = outputs.waveform[0]
   ```

## VitsConfig[[transformers.VitsConfig]]

#### transformers.VitsConfig[[transformers.VitsConfig]]

```python
transformers.VitsConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, vocab_size: int = 38, hidden_size: int = 192, num_hidden_layers: int = 6, num_attention_heads: int = 2, window_size: int = 4, use_bias: bool = True, ffn_dim: int = 768, layerdrop: float | int = 0.1, ffn_kernel_size: int = 3, flow_size: int = 192, spectrogram_bins: int = 513, hidden_act: str = 'relu', hidden_dropout: float | int = 0.1, attention_dropout: float | int = 0.1, activation_dropout: float | int = 0.1, initializer_range: float = 0.02, layer_norm_eps: float = 1e-05, use_stochastic_duration_prediction: bool = True, num_speakers: int = 1, speaker_embedding_size: int = 0, upsample_initial_channel: int = 512, upsample_rates: list[int] | tuple[int, ...] = (8, 8, 2, 2), upsample_kernel_sizes: list[int] | tuple[int, ...] = (16, 16, 4, 4), resblock_kernel_sizes: list[int] | tuple[int, ...] = (3, 7, 11), resblock_dilation_sizes: list | tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5)), leaky_relu_slope: float = 0.1, depth_separable_channels: int = 2, depth_separable_num_layers: int = 3, duration_predictor_flow_bins: int = 10, duration_predictor_tail_bound: float = 5.0, duration_predictor_kernel_size: int = 3, duration_predictor_dropout: float | int = 0.5, duration_predictor_num_flows: int = 4, duration_predictor_filter_channels: int = 256, prior_encoder_num_flows: int = 4, prior_encoder_num_wavenet_layers: int = 4, posterior_encoder_num_wavenet_layers: int = 16, wavenet_kernel_size: int = 5, wavenet_dilation_rate: int = 1, wavenet_dropout: float | int = 0.0, speaking_rate: float | int = 1.0, noise_scale: float = 0.667, noise_scale_duration: float = 0.8, sampling_rate: int = 16000, pad_token_id: int | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/configuration_vits.py#L24)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `38`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `192`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `6`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `2`) : Number of attention heads for each attention layer in the Transformer decoder.

window_size (`int`, *optional*, defaults to 4) : Window size for the relative positional embeddings in the attention layers of the Transformer encoder.

use_bias (`bool`, *optional*, defaults to `True`) : Whether to use bias in the key, query, value projection layers in the Transformer encoder.

ffn_dim (`int`, *optional*, defaults to `768`) : Dimension of the MLP representations.

layerdrop (`Union[float, int]`, *optional*, defaults to `0.1`) : The LayerDrop probability. See the [LayerDrop paper](see https://huggingface.co/papers/1909.11556) for more details.

ffn_kernel_size (`int`, *optional*, defaults to 3) : Kernel size of the 1D convolution layers used by the feed-forward network in the Transformer encoder.

flow_size (`int`, *optional*, defaults to 192) : Dimensionality of the flow layers.

spectrogram_bins (`int`, *optional*, defaults to 513) : Number of frequency bins in the target spectrogram.

hidden_act (`str`, *optional*, defaults to `relu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

hidden_dropout (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout ratio for the attention probabilities.

activation_dropout (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout ratio for activations inside the fully connected layer.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the layer normalization layers.

use_stochastic_duration_prediction (`bool`, *optional*, defaults to `True`) : Whether to use the stochastic duration prediction module or the regular duration predictor.

num_speakers (`int`, *optional*, defaults to 1) : Number of speakers if this is a multi-speaker model.

speaker_embedding_size (`int`, *optional*, defaults to 0) : Number of channels used by the speaker embeddings. Is zero for single-speaker models.

upsample_initial_channel (`int`, *optional*, defaults to 512) : The number of input channels into the HiFi-GAN upsampling network.

upsample_rates (`tuple[int]` or `list[int]`, *optional*, defaults to `[8, 8, 2, 2]`) : A tuple of integers defining the stride of each 1D convolutional layer in the HiFi-GAN upsampling network. The length of `upsample_rates` defines the number of convolutional layers and has to match the length of `upsample_kernel_sizes`.

upsample_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[16, 16, 4, 4]`) : A tuple of integers defining the kernel size of each 1D convolutional layer in the HiFi-GAN upsampling network. The length of `upsample_kernel_sizes` defines the number of convolutional layers and has to match the length of `upsample_rates`.

resblock_kernel_sizes (`tuple[int]` or `list[int]`, *optional*, defaults to `[3, 7, 11]`) : A tuple of integers defining the kernel sizes of the 1D convolutional layers in the HiFi-GAN multi-receptive field fusion (MRF) module.

resblock_dilation_sizes (`tuple[tuple[int]]` or `list[list[int]]`, *optional*, defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`) : A nested tuple of integers defining the dilation rates of the dilated 1D convolutional layers in the HiFi-GAN multi-receptive field fusion (MRF) module.

leaky_relu_slope (`float`, *optional*, defaults to 0.1) : The angle of the negative slope used by the leaky ReLU activation.

depth_separable_channels (`int`, *optional*, defaults to 2) : Number of channels to use in each depth-separable block.

depth_separable_num_layers (`int`, *optional*, defaults to 3) : Number of convolutional layers to use in each depth-separable block.

duration_predictor_flow_bins (`int`, *optional*, defaults to 10) : Number of channels to map using the unonstrained rational spline in the duration predictor model.

duration_predictor_tail_bound (`float`, *optional*, defaults to 5.0) : Value of the tail bin boundary when computing the unconstrained rational spline in the duration predictor model.

duration_predictor_kernel_size (`int`, *optional*, defaults to 3) : Kernel size of the 1D convolution layers used in the duration predictor model.

duration_predictor_dropout (`float`, *optional*, defaults to 0.5) : The dropout ratio for the duration predictor model.

duration_predictor_num_flows (`int`, *optional*, defaults to 4) : Number of flow stages used by the duration predictor model.

duration_predictor_filter_channels (`int`, *optional*, defaults to 256) : Number of channels for the convolution layers used in the duration predictor model.

prior_encoder_num_flows (`int`, *optional*, defaults to 4) : Number of flow stages used by the prior encoder flow model.

prior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 4) : Number of WaveNet layers used by the prior encoder flow model.

posterior_encoder_num_wavenet_layers (`int`, *optional*, defaults to 16) : Number of WaveNet layers used by the posterior encoder model.

wavenet_kernel_size (`int`, *optional*, defaults to 5) : Kernel size of the 1D convolution layers used in the WaveNet model.

wavenet_dilation_rate (`int`, *optional*, defaults to 1) : Dilation rates of the dilated 1D convolutional layers used in the WaveNet model.

wavenet_dropout (`float`, *optional*, defaults to 0.0) : The dropout ratio for the WaveNet layers.

speaking_rate (`float`, *optional*, defaults to 1.0) : Speaking rate. Larger values give faster synthesised speech.

noise_scale (`float`, *optional*, defaults to 0.667) : How random the speech prediction is. Larger values create more variation in the predicted speech.

noise_scale_duration (`float`, *optional*, defaults to 0.8) : How random the duration prediction is. Larger values create more variation in the predicted durations.

sampling_rate (`int`, *optional*, defaults to `16000`) : The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

This is the configuration class to store the configuration of a VitsModel. It is used to instantiate a Vits
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [facebook/mms-tts-eng](https://huggingface.co/facebook/mms-tts-eng)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import VitsModel, VitsConfig

>>> # Initializing a "facebook/mms-tts-eng" style configuration
>>> configuration = VitsConfig()

>>> # Initializing a model (with random weights) from the "facebook/mms-tts-eng" style configuration
>>> model = VitsModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## VitsTokenizer[[transformers.VitsTokenizer]]

#### transformers.VitsTokenizer[[transformers.VitsTokenizer]]

```python
transformers.VitsTokenizer(vocab_file, pad_token = '<pad>', unk_token = '<unk>', language = None, add_blank = True, normalize = True, phonemize = True, is_uroman = False, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/tokenization_vits.py#L46)

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

language (`str`, *optional*) : Language identifier.

add_blank (`bool`, *optional*, defaults to `True`) : Whether to insert token id 0 in between the other tokens.

normalize (`bool`, *optional*, defaults to `True`) : Whether to normalize the input text by removing all casing and punctuation.

phonemize (`bool`, *optional*, defaults to `True`) : Whether to convert the input text into phonemes.

is_uroman (`bool`, *optional*, defaults to `False`) : Whether the `uroman` Romanizer needs to be applied to the input text prior to tokenizing.

Construct a VITS tokenizer. Also supports MMS-TTS.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### __call__[[transformers.VitsTokenizer.__call__]]

```python
__call__(text: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None, text_pair: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None, text_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None, text_pair_target: TextInput | PreTokenizedInput | list[TextInput] | list[PreTokenizedInput] | None = None, add_special_tokens: bool = True, padding: bool | str | PaddingStrategy = False, truncation: bool | str | TruncationStrategy | None = None, max_length: int | None = None, stride: int = 0, is_split_into_words: bool = False, pad_to_multiple_of: int | None = None, padding_side: str | None = None, return_tensors: str | TensorType | None = None, return_token_type_ids: bool | None = None, return_attention_mask: bool | None = None, return_overflowing_tokens: bool = False, return_special_tokens_mask: bool = False, return_offsets_mapping: bool = False, return_length: bool = False, verbose: bool = True, tokenizer_kwargs: dict[str, Any] | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/tokenization_utils_base.py#L2417)

**Parameters:**

text (`str`, `list[str]`, `list[list[str]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

text_pair (`str`, `list[str]`, `list[list[str]]`, *optional*) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

text_target (`str`, `list[str]`, `list[list[str]]`, *optional*) : The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

text_pair_target (`str`, `list[str]`, `list[list[str]]`, *optional*) : The sequence or batch of sequences to be encoded as target texts. Each sequence can be a string or a list of strings (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).

tokenizer_kwargs (`dict[str, Any]`, *optional*) : Additional kwargs to pass to the tokenizer. These will be merged with the explicit parameters and other kwargs, with explicit parameters taking precedence. 

add_special_tokens (`bool`, *optional*, defaults to `True`) : Whether or not to add special tokens when encoding the sequences. This will use the underlying `PretrainedTokenizerBase.build_inputs_with_special_tokens` function, which defines which tokens are automatically added to the input ids. This is useful if you want to add `bos` or `eos` tokens automatically.

padding (`bool`, `str` or [PaddingStrategy](/docs/transformers/v5.15.1/en/internal/file_utils#transformers.utils.PaddingStrategy), *optional*, defaults to `False`) : Activates and controls padding. Accepts the following values:  - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence is provided). - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different lengths).

truncation (`bool`, `str` or [TruncationStrategy](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.tokenization_utils_base.TruncationStrategy), *optional*, defaults to `False`) : Activates and controls truncation. Accepts the following values:  - `True` or `'longest_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will truncate token by token, removing a token from the longest sequence in the pair if a pair of sequences (or a batch of pairs) is provided. - `'only_first'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the first sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `'only_second'`: Truncate to a maximum length specified with the argument `max_length` or to the maximum acceptable input length for the model if that argument is not provided. This will only truncate the second sequence of a pair if a pair of sequences (or a batch of pairs) is provided. - `False` or `'do_not_truncate'` (default): No truncation (i.e., can output batch with sequence lengths greater than the model maximum admissible input size).

max_length (`int`, *optional*) : Controls the maximum length to use by one of the truncation/padding parameters.  If left unset or set to `None`, this will use the predefined model maximum length if a maximum length is required by one of the truncation/padding parameters. If the model has no specific maximum input length (like XLNet) truncation/padding to a maximum length will be deactivated.

stride (`int`, *optional*, defaults to 0) : If set to a number along with `max_length`, the overflowing tokens returned when `return_overflowing_tokens=True` will contain some tokens from the end of the truncated sequence returned to provide some overlap between truncated and overflowing sequences. The value of this argument defines the number of overlapping tokens.

is_split_into_words (`bool`, *optional*, defaults to `False`) : Whether or not the input is already pre-tokenized (e.g., split into words). If set to `True`, the tokenizer assumes the input is already split into words (for instance, by splitting it on whitespace) which it will tokenize. This is useful for NER or token classification.

pad_to_multiple_of (`int`, *optional*) : If set will pad the sequence to a multiple of the provided value. Requires `padding` to be activated. This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability `>= 7.5` (Volta).

padding_side (`str`, *optional*) : The side on which the model should have padding applied. Should be selected between ['right', 'left']. Default value is picked from the class attribute of the same name.

return_tensors (`str` or [TensorType](/docs/transformers/v5.15.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors instead of list of python integers. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return Numpy `np.ndarray` objects. 

return_token_type_ids (`bool`, *optional*) : Whether to return token type IDs. If left to the default, will return the token type IDs according to the specific tokenizer's default, defined by the `return_outputs` attribute.  [What are token type IDs?](../glossary#token-type-ids)

return_attention_mask (`bool`, *optional*) : Whether to return the attention mask. If left to the default, will return the attention mask according to the specific tokenizer's default, defined by the `return_outputs` attribute.  [What are attention masks?](../glossary#attention-mask)

return_overflowing_tokens (`bool`, *optional*, defaults to `False`) : Whether or not to return overflowing token sequences. If a pair of sequences of input ids (or a batch of pairs) is provided with `truncation_strategy = longest_first` or `True`, an error is raised instead of returning overflowing tokens.

return_special_tokens_mask (`bool`, *optional*, defaults to `False`) : Whether or not to return special tokens mask information.

return_offsets_mapping (`bool`, *optional*, defaults to `False`) : Whether or not to return `(char_start, char_end)` for each token.  This is only available on fast tokenizers inheriting from [PreTrainedTokenizerFast](/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.TokenizersBackend), if using Python's tokenizer, this method will raise `NotImplementedError`.

return_length  (`bool`, *optional*, defaults to `False`) : Whether or not to return the lengths of the encoded inputs.

verbose (`bool`, *optional*, defaults to `True`) : Whether or not to print more information and warnings.

- ****kwargs** : passed to the `self.tokenize()` method

**Returns:** [BatchEncoding](/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.BatchEncoding)

A [BatchEncoding](/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.BatchEncoding) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.

  [What are input IDs?](../glossary#input-ids)

- **token_type_ids** -- List of token type ids to be fed to a model (when `return_token_type_ids=True` or
  if *"token_type_ids"* is in `self.model_input_names`).

  [What are token type IDs?](../glossary#token-type-ids)

- **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
  `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names`).

  [What are attention masks?](../glossary#attention-mask)

- **overflowing_tokens** -- List of overflowing tokens sequences (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **num_truncated_tokens** -- Number of tokens truncated (when a `max_length` is specified and
  `return_overflowing_tokens=True`).
- **special_tokens_mask** -- List of 0s and 1s, with 1 specifying added special tokens and 0 specifying
  regular sequence tokens (when `add_special_tokens=True` and `return_special_tokens_mask=True`).
- **length** -- The length of the inputs (when `return_length=True`)

Main method to tokenize and prepare for the model one or several sequence(s) or one or several pair(s) of
sequences.

#### save_vocabulary[[transformers.VitsTokenizer.save_vocabulary]]

```python
save_vocabulary(save_directory: str, filename_prefix: str | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/tokenization_vits.py#L233)

## VitsModel[[transformers.VitsModel]]

#### transformers.VitsModel[[transformers.VitsModel]]

```python
transformers.VitsModel(config: VitsConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/modeling_vits.py#L1231)

**Parameters:**

config ([VitsConfig](/docs/transformers/v5.15.1/en/model_doc/vits#transformers.VitsConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The complete VITS model, for text-to-speech synthesis.

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.VitsModel.forward]]

```python
forward(input_ids: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, speaker_id: int | None = None, output_attentions: bool | None = None, output_hidden_states: bool | None = None, return_dict: bool | None = None, labels: typing.Optional[torch.FloatTensor] = None, speaking_rate: float | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/vits/modeling_vits.py#L1258)

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

speaker_id (`int`, *optional*) : Which speaker embedding to use. Only used for multispeaker models.

output_attentions (`bool`, *optional*) : Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned tensors for more detail.

output_hidden_states (`bool`, *optional*) : Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more detail.

return_dict (`bool`, *optional*) : Whether or not to return a [ModelOutput](/docs/transformers/v5.15.1/en/main_classes/output#transformers.utils.ModelOutput) instead of a plain tuple.

labels (`torch.FloatTensor` of shape `(batch_size, config.spectrogram_bins, sequence_length)`, *optional*) : Float values of target spectrogram. Timesteps set to `-100.0` are ignored (masked) for the loss computation.

speaking_rate (`float`, *optional*) : Speaking rate.

**Returns:** `VitsModelOutput` or `tuple(torch.FloatTensor)`

A `VitsModelOutput` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([VitsConfig](/docs/transformers/v5.15.1/en/model_doc/vits#transformers.VitsConfig)) and inputs.

The [VitsModel](/docs/transformers/v5.15.1/en/model_doc/vits#transformers.VitsModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **waveform** (`torch.FloatTensor` of shape `(batch_size, sequence_length)`) -- The final audio waveform predicted by the model.
- **sequence_lengths** (`torch.FloatTensor` of shape `(batch_size,)`) -- The length in samples of each element in the `waveform` batch.
- **spectrogram** (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`) -- The log-mel spectrogram predicted at the output of the flow model. This spectrogram is passed to the Hi-Fi
  GAN decoder model to obtain the final audio waveform.
- **hidden_states** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple[torch.FloatTensor]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import VitsTokenizer, VitsModel, set_seed
>>> import torch

>>> tokenizer = VitsTokenizer.from_pretrained("facebook/mms-tts-eng")
>>> model = VitsModel.from_pretrained("facebook/mms-tts-eng")

>>> inputs = tokenizer(text="Hello - my dog is cute", return_tensors="pt")

>>> set_seed(555)  # make deterministic

>>> with torch.no_grad():
...     outputs = model(inputs["input_ids"])
>>> outputs.waveform.shape
torch.Size([1, 45824])
```

### Quickstart
https://huggingface.co/docs/transformers/v5.15.1/model_doc/qwen3_5_moe.md

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
- On NVIDIA GB10 (compute capability 12.1 / SM121) `causal_conv1d` and `fla` have no SM121 build, so the Gated DeltaNet path always uses the slow PyTorch reference. Passing `use_kernels=True` (`pip install -U kernels`) to [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) swaps it for the same compute-capability-gated Hub kernel as the dense variant ([`Atlas-Inference/gdn`](https://huggingface.co/kernels/Atlas-Inference/gdn), shared because `Qwen3_5MoeGatedDeltaNet` has the same core as `Qwen3_5GatedDeltaNet`); every other GPU keeps the existing path. The kernel is numerically faithful to the fallback (identical greedy output) and speeds up prefill. Measured on `Qwen/Qwen3.6-35B-A3B` (bf16, GB10/SM121, 1024-token prompt, greedy decode of 256 tokens):

  | `use_kernels` | TTFT (prefill) | Decode |
  | --- | --- | --- |
  | `False` (PyTorch fallback) | 0.73 s | 16.3 tok/s |
  | `True` ([`Atlas-Inference/gdn`](https://huggingface.co/kernels/Atlas-Inference/gdn)) | 0.53 s (1.38x faster) | 16.7 tok/s |

  Decode is roughly flat because the single-token DeltaNet recurrence is memory-bandwidth-bound; the win is on the chunked-prefill core and grows with prompt length. Loading the mapped kernel currently requires `trust_remote_code=True` until `Atlas-Inference` is added to the trusted-kernels allowlist.

## Qwen3_5MoeConfig[[transformers.Qwen3_5MoeConfig]]

#### transformers.Qwen3_5MoeConfig[[transformers.Qwen3_5MoeConfig]]

```python
transformers.Qwen3_5MoeConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, text_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, image_token_id: int = 248056, video_token_id: int = 248057, vision_start_token_id: int = 248053, vision_end_token_id: int = 248054, tie_word_embeddings: bool = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L166)

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

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

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
transformers.Qwen3_5MoeTextConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, vocab_size: int = 248320, hidden_size: int = 2048, num_hidden_layers: int = 40, num_attention_heads: int = 16, num_key_value_heads: int = 2, hidden_act: str = 'silu', max_position_embeddings: int = 32768, initializer_range: float = 0.02, rms_norm_eps: float = 1e-06, use_cache: bool = True, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, attention_bias: bool = False, attention_dropout: float | int = 0.0, head_dim: int = 256, linear_conv_kernel_dim: int = 4, linear_key_head_dim: int = 128, linear_value_head_dim: int = 128, linear_num_key_heads: int = 16, linear_num_value_heads: int = 32, moe_intermediate_size: int = 512, shared_expert_intermediate_size: int = 512, num_experts_per_tok: int = 8, num_experts: int = 256, output_router_logits: bool = False, router_aux_loss_coef: float = 0.001, layer_types: list[str] | None = None, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L29)

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

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

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
transformers.Qwen3_5MoeVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, depth: int = 27, hidden_size: int = 1152, hidden_act: str = 'gelu_pytorch_tanh', intermediate_size: int = 4304, num_heads: int = 16, in_channels: int = 3, patch_size: int | list[int] | tuple[int, int] = 16, spatial_merge_size: int = 2, temporal_patch_size: int | list[int] | tuple[int, int] = 2, out_hidden_size: int = 3584, num_position_embeddings: int = 2304, initializer_range: float = 0.02)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py#L139)

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

This is the configuration class to store the configuration of a Qwen3_5MoeModel. It is used to instantiate a Qwen3 5 Moe
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

## Qwen3_5MoeVisionModel[[transformers.Qwen3_5MoeVisionModel]]

#### transformers.Qwen3_5MoeVisionModel[[transformers.Qwen3_5MoeVisionModel]]

```python
transformers.Qwen3_5MoeVisionModel(config, *inputs, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1092)

#### forward[[transformers.Qwen3_5MoeVisionModel.forward]]

```python
forward(hidden_states: Tensor, grid_thw: Tensor, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1155)

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

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1234)

#### forward[[transformers.Qwen3_5MoeTextModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1249)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeTextModel](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeTextModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1323)

**Parameters:**

config ([Qwen3_5MoeModel](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Qwen3 5 Moe Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Qwen3_5MoeModel.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_token_type_ids: typing.Optional[torch.IntTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1605)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using `image_processor_class`. See `image_processor_class.__call__` for details (`processor_class` uses `image_processor_class` for processing images).

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using `video_processor_class`. See `video_processor_class.__call__` for details (`processor_class` uses `video_processor_class` for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.15.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details.

**Returns:** `Qwen3_5MoeModelOutputWithPast` or `tuple(torch.FloatTensor)`

A `Qwen3_5MoeModelOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration (`None`) and inputs.

The [Qwen3_5MoeModel](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeModel) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **last_hidden_state** (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`) -- Sequence of hidden-states at the output of the last layer of the model.

  If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
  hidden_size)` is output.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1758)

**Parameters:**

config ([Qwen3_5MoeForCausalLM](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Qwen3 5 Moe Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Qwen3_5MoeForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, output_router_logits: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1778)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

output_router_logits (`bool`, *optional*) : Whether or not to return the logits of all the routers. They are useful for computing the router loss, and should not be returned during inference.

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeForCausalLM](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeForCausalLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **aux_loss** (`torch.FloatTensor`, *optional*, returned when `labels` is provided) -- aux_loss for the sparse modules.
- **router_logits** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_router_probs=True` and `config.add_router_probs=True` is passed or when `config.output_router_probs=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_experts)`.

  Raw router logits (post-softmax) that are computed by MoE routers, these terms are used to compute the auxiliary
  loss for Mixture of Experts models.
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1863)

#### forward[[transformers.Qwen3_5MoeForConditionalGeneration.forward]]

```python
forward(input_ids: LongTensor = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, pixel_values: typing.Optional[torch.Tensor] = None, pixel_values_videos: typing.Optional[torch.FloatTensor] = None, image_grid_thw: typing.Optional[torch.LongTensor] = None, video_grid_thw: typing.Optional[torch.LongTensor] = None, mm_token_type_ids: typing.Optional[torch.IntTensor] = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py#L1904)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

pixel_values (`torch.Tensor` of shape `(batch_size, num_channels, image_size, image_size)`, *optional*) : The tensors corresponding to the input images. Pixel values can be obtained using [Qwen2VLImageProcessor](/docs/transformers/v5.15.1/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor). See `Qwen2VLImageProcessor.__call__()` for details ([Qwen3VLProcessor](/docs/transformers/v5.15.1/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) uses [Qwen2VLImageProcessor](/docs/transformers/v5.15.1/en/model_doc/qwen2_vl#transformers.Qwen2VLImageProcessor) for processing images).

pixel_values_videos (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, frame_size, frame_size)`, *optional*) : The tensors corresponding to the input video. Pixel values for videos can be obtained using [Qwen3VLVideoProcessor](/docs/transformers/v5.15.1/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor). See `Qwen3VLVideoProcessor.__call__()` for details ([Qwen3VLProcessor](/docs/transformers/v5.15.1/en/model_doc/qwen3_vl#transformers.Qwen3VLProcessor) uses [Qwen3VLVideoProcessor](/docs/transformers/v5.15.1/en/model_doc/qwen3_vl#transformers.Qwen3VLVideoProcessor) for processing videos).

image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*) : The temporal, height and width of feature shape of each image in LLM.

video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*) : The temporal, height and width of feature shape of each video in LLM.

mm_token_type_ids (`torch.IntTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens matching each modality. For example text (0), image (1), video (2). Multimodal token type ids can be obtained using [AutoProcessor](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoProcessor). See [ProcessorMixin.__call__()](/docs/transformers/v5.15.1/en/main_classes/processors#transformers.ProcessorMixin.__call__) for details. 

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** `Qwen3_5MoeCausalLMOutputWithPast` or `tuple(torch.FloatTensor)`

A `Qwen3_5MoeCausalLMOutputWithPast` or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Qwen3_5MoeConfig](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeConfig)) and inputs.

The [Qwen3_5MoeForConditionalGeneration](/docs/transformers/v5.15.1/en/model_doc/qwen3_5_moe#transformers.Qwen3_5MoeForConditionalGeneration) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Language modeling loss (for next-token prediction).
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **past_key_values** (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`) -- It is a [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance. For more details, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache).

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
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
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
https://huggingface.co/docs/transformers/v5.15.1/model_doc/higgs_audio_v2.md
