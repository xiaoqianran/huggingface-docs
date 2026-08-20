# CTRL

## Overview

CTRL model was proposed in [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://huggingface.co/papers/1909.05858) by Nitish Shirish Keskar*, Bryan McCann*, Lav R. Varshney, Caiming Xiong and
Richard Socher. It's a causal (unidirectional) transformer pre-trained using language modeling on a very large corpus
of ~140 GB of text data with the first token reserved as a control code (such as Links, Books, Wikipedia etc.).

The abstract from the paper is the following:

*Large-scale language models show promising text generation capabilities, but users cannot easily control particular
aspects of the generated text. We release CTRL, a 1.63 billion-parameter conditional transformer language model,
trained to condition on control codes that govern style, content, and task-specific behavior. Control codes were
derived from structure that naturally co-occurs with raw text, preserving the advantages of unsupervised learning while
providing more explicit control over text generation. These codes also allow CTRL to predict which parts of the
training data are most likely given a sequence. This provides a potential method for analyzing large amounts of data
via model-based source attribution.*

This model was contributed by [keskarnitishr](https://huggingface.co/keskarnitishr). The original code can be found
[here](https://github.com/salesforce/ctrl).

## Usage tips

- CTRL makes use of control codes to generate text: it requires generations to be started by certain words, sentences
  or links to generate coherent text. Refer to the [original implementation](https://github.com/salesforce/ctrl) for
  more information.
- CTRL is a model with absolute position embeddings so it's usually advised to pad the inputs on the right rather than
  the left.
- CTRL was trained with a causal language modeling (CLM) objective and is therefore powerful at predicting the next
  token in a sequence. Leveraging this feature allows CTRL to generate syntactically coherent text as it can be
  observed in the *run_generation.py* example script.
- The PyTorch models can take the `past_key_values` as input, which is the previously computed key/value attention pairs.
  Using the `past_key_values` value prevents the model from re-computing
  pre-computed values in the context of text generation. See the [forward](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLModel.forward)
  method for more information on the usage of this argument.

## Resources

- [Text classification task guide](../tasks/sequence_classification)
- [Causal language modeling task guide](../tasks/language_modeling)

## CTRLConfig[[transformers.CTRLConfig]]

#### transformers.CTRLConfig[[transformers.CTRLConfig]]

```python
transformers.CTRLConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, vocab_size: int = 246534, n_positions: int = 256, n_embd: int = 1280, dff: int = 8192, n_layer: int = 48, n_head: int = 16, resid_pdrop: float | int = 0.1, embd_pdrop: float | int = 0.1, layer_norm_epsilon: float = 1e-06, initializer_range: float = 0.02, use_cache: bool = True, pad_token_id: int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None, tie_word_embeddings: bool = True)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/configuration_ctrl.py#L24)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `246534`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

n_positions (`int`, *optional*, defaults to `256`) : The maximum sequence length that this model might ever be used with.

n_embd (`int`, *optional*, defaults to `1280`) : Dimensionality of the embeddings and hidden states.

dff (`int`, *optional*, defaults to 8192) : Dimensionality of the inner dimension of the feed forward networks (FFN).

n_layer (`int`, *optional*, defaults to `48`) : Number of hidden layers in the Transformer decoder.

n_head (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

resid_pdrop (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

embd_pdrop (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout ratio for the embeddings.

layer_norm_epsilon (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

pad_token_id (`int`, *optional*) : Token id used for padding in the vocabulary.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

This is the configuration class to store the configuration of a CTRLModel. It is used to instantiate a Ctrl
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [Salesforce/ctrl](https://huggingface.co/Salesforce/ctrl)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import CTRLConfig, CTRLModel

>>> # Initializing a CTRL configuration
>>> configuration = CTRLConfig()

>>> # Initializing a model (with random weights) from the configuration
>>> model = CTRLModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## CTRLTokenizer[[transformers.CTRLTokenizer]]

#### transformers.CTRLTokenizer[[transformers.CTRLTokenizer]]

```python
transformers.CTRLTokenizer(vocab_file, merges_file, unk_token = '<unk>', **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/tokenization_ctrl.py#L107)

**Parameters:**

vocab_file (`str`) : Path to the vocabulary file.

merges_file (`str`) : Path to the merges file.

unk_token (`str`, *optional*, defaults to `"<unk>"`) : The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this token instead.

Construct a CTRL tokenizer. Based on Byte-Pair-Encoding.

This tokenizer inherits from [PreTrainedTokenizer](/docs/transformers/v5.15.1/en/main_classes/tokenizer#transformers.PythonBackend) which contains most of the main methods. Users should refer to
this superclass for more information regarding those methods.

#### save_vocabulary[[transformers.CTRLTokenizer.save_vocabulary]]

```python
save_vocabulary(save_directory: str, filename_prefix: str | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/tokenization_python.py#L1362)

**Parameters:**

save_directory (`str`) : The directory in which to save the vocabulary.

filename_prefix (`str`, *optional*) : An optional prefix to add to the named of the saved files.

**Returns:** `tuple[str, ...]`

Paths to the files saved, or empty tuple if no files saved.

Default implementation for common vocabulary saving patterns.
Saves self.encoder/self.vocab as JSON, optionally with self.bpe_ranks as merges.
Returns empty tuple if no vocabulary exists.

Override this method if your tokenizer needs custom saving logic (e.g., SentencePiece models,
multiple vocabulary files, or special file formats).

## CTRLModel[[transformers.CTRLModel]]

#### transformers.CTRLModel[[transformers.CTRLModel]]

```python
transformers.CTRLModel(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/modeling_ctrl.py#L216)

**Parameters:**

config ([CTRLModel](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Ctrl Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.CTRLModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, attention_mask: typing.Optional[torch.FloatTensor] = None, token_type_ids: typing.Optional[torch.LongTensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, use_cache: bool | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/modeling_ctrl.py#L242)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [BaseModelOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CTRLConfig](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLConfig)) and inputs.

The [CTRLModel](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLModel) forward method, overrides the `__call__` special method.

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

Example:

```python
>>> from transformers import AutoTokenizer, CTRLModel
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLModel.from_pretrained("Salesforce/ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> outputs = model(**inputs)

>>> last_hidden_states = outputs.last_hidden_state
>>> list(last_hidden_states.shape)
[1, 5, 1280]
```

## CTRLLMHeadModel[[transformers.CTRLLMHeadModel]]

#### transformers.CTRLLMHeadModel[[transformers.CTRLLMHeadModel]]

```python
transformers.CTRLLMHeadModel(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/modeling_ctrl.py#L346)

**Parameters:**

config ([CTRLLMHeadModel](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLLMHeadModel)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a language modeling head on top (linear layer with weights tied to the input
embeddings).

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.CTRLLMHeadModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, attention_mask: typing.Optional[torch.FloatTensor] = None, token_type_ids: typing.Optional[torch.LongTensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/modeling_ctrl.py#L357)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CTRLConfig](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLConfig)) and inputs.

The [CTRLLMHeadModel](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLLMHeadModel) forward method, overrides the `__call__` special method.

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

Example:

```python
>>> import torch
>>> from transformers import AutoTokenizer, CTRLLMHeadModel

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLLMHeadModel.from_pretrained("Salesforce/ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Wikipedia The llama is", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> sequence_ids = model.generate(inputs["input_ids"])
>>> sequences = tokenizer.batch_decode(sequence_ids)
>>> sequences
['Wikipedia The llama is a member of the family Bovidae. It is native to the Andes of Peru,']

>>> outputs = model(**inputs, labels=inputs["input_ids"])
>>> round(outputs.loss.item(), 2)
9.21

>>> list(outputs.logits.shape)
[1, 5, 246534]
```

## CTRLForSequenceClassification[[transformers.CTRLForSequenceClassification]]

#### transformers.CTRLForSequenceClassification[[transformers.CTRLForSequenceClassification]]

```python
transformers.CTRLForSequenceClassification(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/modeling_ctrl.py#L466)

**Parameters:**

config ([CTRLForSequenceClassification](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLForSequenceClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The CTRL Model transformer with a sequence classification head on top (linear layer).
[CTRLForSequenceClassification](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) uses the last token in order to do the classification, as other causal models
(e.g. GPT-2) do. Since it does classification on the last token, it requires to know the position of the last
token. If a `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in
each row. If no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot
guess the padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last
value in each row of the batch).

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.CTRLForSequenceClassification.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, attention_mask: typing.Optional[torch.FloatTensor] = None, token_type_ids: typing.Optional[torch.LongTensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/ctrl/modeling_ctrl.py#L476)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*) : Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

**Returns:** [SequenceClassifierOutput](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [SequenceClassifierOutput](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([CTRLConfig](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLConfig)) and inputs.

The [CTRLForSequenceClassification](/docs/transformers/v5.15.1/en/model_doc/ctrl#transformers.CTRLForSequenceClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification (or regression if config.num_labels==1) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`) -- Classification (or regression if config.num_labels==1) scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example of single-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, CTRLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl")

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```

```python
>>> import torch

>>> torch.manual_seed(42)
>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> labels = torch.tensor(1)
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
0.93
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, CTRLForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("Salesforce/ctrl")
>>> model = CTRLForSequenceClassification.from_pretrained(
...     "Salesforce/ctrl", problem_type="multi_label_classification"
... )

>>> # CTRL was trained with control codes as the first token
>>> inputs = tokenizer("Opinion My dog is cute", return_tensors="pt")
>>> assert inputs["input_ids"][0, 0].item() in tokenizer.control_codes.values()

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
'LABEL_0'
```

```python
>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = CTRLForSequenceClassification.from_pretrained("Salesforce/ctrl", num_labels=num_labels)

>>> num_labels = len(model.config.id2label)
>>> labels = torch.nn.functional.one_hot(torch.tensor([predicted_class_id]), num_classes=num_labels).to(
...     torch.float
... )
>>> loss = model(**inputs, labels=labels).loss
>>> loss.backward()
```

### Phi4 Multimodal
https://huggingface.co/docs/transformers/v5.15.1/model_doc/phi4_multimodal.md

## Phi4 Multimodal

[Phi4 Multimodal](https://huggingface.co/papers/2503.01743) is a multimodal model capable of text, image, and speech and audio inputs or any combination of these. It features a mixture of LoRA adapters for handling different inputs, and each input is routed to the appropriate encoder.

You can find all the original Phi4 Multimodal checkpoints under the [Phi4](https://huggingface.co/collections/microsoft/phi-4-677e9380e514feb5577a40e4) collection.

> [!TIP]
> This model was contributed by [cyrilvallez](https://huggingface.co/cyrilvallez).
>
> Click on the Phi-4 Multimodal in the right sidebar for more examples of how to apply Phi-4 Multimodal to different tasks.

The example below demonstrates how to generate text based on an image with [Pipeline](/docs/transformers/v5.15.1/en/main_classes/pipelines#transformers.Pipeline) or the [AutoModel](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoModel) class.

```python
from transformers import pipeline

generator = pipeline("text-generation", model="microsoft/Phi-4-multimodal-instruct", device=0)

prompt = "Explain the concept of multimodal AI in simple terms."

result = generator(prompt, max_length=50)
print(result[0]['generated_text'])
```

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "microsoft/Phi-4-multimodal-instruct"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

model.load_adapter(model_path, adapter_name="vision", device_map="auto", adapter_kwargs={"subfolder": 'vision-lora'})

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://www.ilankelman.org/stopsigns/australia.jpg"},
            {"type": "text", "text": "What is shown in this image?"},
        ],
    },
]

model.set_adapter("vision")
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'Response\n{response}')
```

## Notes

The example below demonstrates inference with an audio and text input.

```python
import torch

from transformers import AutoModelForCausalLM, AutoProcessor

model_path = "microsoft/Phi-4-multimodal-instruct"

processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")

model.load_adapter(model_path, adapter_name="speech", device_map="auto", adapter_kwargs={"subfolder": 'speech-lora'})
model.set_adapter("speech")
audio_url = "https://upload.wikimedia.org/wikipedia/commons/b/b0/Barbara_Sahakian_BBC_Radio4_The_Life_Scientific_29_May_2012_b01j5j24.flac"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "audio", "url": audio_url},
            {"type": "text", "text": "Transcribe the audio to text, and then translate the audio to French. Use <sep> as a separator between the origina transcript and the translation."},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

generate_ids = model.generate(
    **inputs,
    max_new_tokens=1000,
    do_sample=False,
)
generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
response = processor.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(f'Response\n{response}')
```

## Phi4MultimodalFeatureExtractor[[transformers.Phi4MultimodalFeatureExtractor]]

#### transformers.Phi4MultimodalFeatureExtractor[[transformers.Phi4MultimodalFeatureExtractor]]

```python
transformers.Phi4MultimodalFeatureExtractor(feature_size: int = 80, sampling_rate: int = 16000, hop_length: int = 160, n_fft: int = 512, win_length: int = 400, preemphasis: float = 0.97, padding_value: float = 0.0, audio_compression_rate: int = 8, audio_downsample_rate: int = 1, audio_feat_stride: int = 1, mel_min_frequency: float = 0, mel_max_frequency: float = 7690, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/feature_extraction_phi4_multimodal.py#L34)

## Phi4MultimodalImageProcessor[[transformers.Phi4MultimodalImageProcessor]]

#### transformers.Phi4MultimodalImageProcessor[[transformers.Phi4MultimodalImageProcessor]]

```python
transformers.Phi4MultimodalImageProcessor(**kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/image_processing_phi4_multimodal.py#L44)

**Parameters:**

do_convert_rgb (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*, defaults to `{'height' : 448, 'width': 448}`): Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*, defaults to `Resampling.BICUBIC`) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*, defaults to `0.00392156862745098`) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*, defaults to `True`) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*, defaults to `[0.5, 0.5, 0.5]`) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

patch_size (`int`, *kwargs*, *optional*) : The size of the patch.

dynamic_hd (`int`, *kwargs*, *optional*) : The maximum number of crops per image.

Constructs a Phi4MultimodalImageProcessor image processor.

#### preprocess[[transformers.Phi4MultimodalImageProcessor.preprocess]]

```python
preprocess(images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor']], **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/image_processing_phi4_multimodal.py#L60)

**Parameters:**

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

do_convert_rgb (`bool`, *kwargs*, *optional*) : Whether to convert the image to RGB.

do_resize (`bool`, *kwargs*, *optional*) : Whether to resize the image.

size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Describes the maximum input dimensions to the model.

default_to_square (`bool`, *kwargs*, *optional*) : Whether to default to a square image when resizing, if size is an int.

crop_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : Size of the output image after applying `center_crop`.

resample (`Annotated[Union[int, PILImageResampling, NoneType], None]`, *kwargs*) : Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only has an effect if `do_resize` is set to `True`.

do_rescale (`bool`, *kwargs*, *optional*) : Whether to rescale the image.

rescale_factor (`float`, *kwargs*, *optional*) : Rescale factor to rescale the image by if `do_rescale` is set to `True`.

do_normalize (`bool`, *kwargs*, *optional*) : Whether to normalize the image.

image_mean (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.

image_std (`Union[float, list[float], tuple[float, ...]]`, *kwargs*, *optional*) : Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to `True`.

do_pad (`bool`, *kwargs*, *optional*) : Whether to pad the image. Padding is done either to the largest size in the batch or to a fixed square size per image. The exact padding strategy depends on the model.

pad_size (`Annotated[int | list[int] | tuple[int, ...] | dict[str, int] | None, None]`, *kwargs*) : The size in `{"height": int, "width" int}` to pad the images to. Must be larger than any image size provided for preprocessing. If `pad_size` is not provided, images will be padded to the largest height and width in the batch. Applied only when `do_pad=True.`

do_center_crop (`bool`, *kwargs*, *optional*) : Whether to center crop the image.

data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : Only `ChannelDimension.FIRST` is supported. Added for compatibility with slow processors.

input_data_format (`Union[str, ~image_utils.ChannelDimension]`, *kwargs*, *optional*) : The channel dimension format for the input image. If unset, the channel dimension format is inferred from the input image. Can be one of: - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format. - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format. - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

device (`Annotated[Union[str, torch.device, NoneType], None]`, *kwargs*) : The device to process the videos on. If unset, the device is inferred from the input videos.

return_tensors (`Annotated[str | ~utils.generic.TensorType | None, None]`, *kwargs*) : Returns stacked tensors if set to `'pt'`, otherwise returns a list of tensors.

disable_grouping (`bool`, *kwargs*, *optional*) : Whether to disable grouping of images by size to process them individually and not in batches. If None, will be set to True if the images are on CPU, and False otherwise. This choice is based on empirical observations, as detailed here: https://github.com/huggingface/transformers/pull/38157

image_seq_length (`int`, *kwargs*, *optional*) : The number of image tokens to be used for each image in the input. Added for backward compatibility but this should be set as a processor attribute in future models.

patch_size (`int`, *kwargs*, *optional*) : The size of the patch.

dynamic_hd (`int`, *kwargs*, *optional*) : The maximum number of crops per image.

**Returns:** `~image_processing_base.BatchFeature`

- **data** (`dict`) -- Dictionary of lists/arrays/tensors returned by the __call__ method ('pixel_values', etc.).
- **tensor_type** (`Union[None, str, TensorType]`, *optional*) -- You can give a tensor_type here to convert the lists of integers in PyTorch/Numpy Tensors at
  initialization.

## Phi4MultimodalProcessor[[transformers.Phi4MultimodalProcessor]]

#### transformers.Phi4MultimodalProcessor[[transformers.Phi4MultimodalProcessor]]

```python
transformers.Phi4MultimodalProcessor(image_processor, audio_processor, tokenizer, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/processing_phi4_multimodal.py#L41)

**Parameters:**

image_processor (`Phi4MultimodalImageProcessor`) : The image processor is a required input.

audio_processor (`{audio_processor_class}`) : The audio processor is a required input.

tokenizer (`tokenizer_class`) : The tokenizer is a required input.

Constructs a Phi4MultimodalProcessor which wraps a image processor, a audio processor, and a tokenizer into a single processor.

[Phi4MultimodalProcessor](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalProcessor) offers all the functionalities of [Phi4MultimodalImageProcessor](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalImageProcessor), `{audio_processor_class}`, and `tokenizer_class`. See the
[~Phi4MultimodalImageProcessor](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalImageProcessor), `~{audio_processor_class}`, and `~tokenizer_class` for more information.

#### __call__[[transformers.Phi4MultimodalProcessor.__call__]]

```python
__call__(text: str | list[str], images: typing.Union[ForwardRef('PIL.Image.Image'), numpy.ndarray, ForwardRef('torch.Tensor'), list['PIL.Image.Image'], list[numpy.ndarray], list['torch.Tensor'], NoneType] = None, audio: typing.Union[numpy.ndarray, ForwardRef('torch.Tensor'), collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence['torch.Tensor'], NoneType] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/processing_phi4_multimodal.py#L55)

**Parameters:**

text (`Union[str, list[str]]`) : The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings (pretokenized string). If you pass a pretokenized input, set `is_split_into_words=True` to avoid ambiguity with batched inputs.

images (`Union[PIL.Image.Image, numpy.ndarray, torch.Tensor, list[PIL.Image.Image], list[numpy.ndarray], list[torch.Tensor]]`, *optional*) : Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If passing in images with pixel values between 0 and 1, set `do_rescale=False`.

audio (`Union[numpy.ndarray, torch.Tensor, collections.abc.Sequence[numpy.ndarray], collections.abc.Sequence[torch.Tensor]]`, *optional*) : The audio or batch of audios to be prepared. Each audio can be a NumPy array or PyTorch tensor. In case of a NumPy array/PyTorch tensor, each audio should be of shape (C, T), where C is a number of channels, and T is the sample length of the audio.

return_tensors (`str` or [TensorType](/docs/transformers/v5.15.1/en/internal/file_utils#transformers.TensorType), *optional*) : If set, will return tensors of a particular framework. Acceptable values are:  - `'pt'`: Return PyTorch `torch.Tensor` objects. - `'np'`: Return NumPy `np.ndarray` objects.

- ****kwargs** ([ProcessingKwargs](/docs/transformers/v5.15.1/en/main_classes/processors#transformers.ProcessingKwargs), *optional*) : Additional processing options for each modality (text, images, videos, audio). Model-specific parameters are listed above; see the TypedDict class for the complete list of supported arguments.

**Returns:** [BatchFeature](/docs/transformers/v5.15.1/en/main_classes/image_processor#transformers.BatchFeature)

A [BatchFeature](/docs/transformers/v5.15.1/en/main_classes/image_processor#transformers.BatchFeature) with the following fields:

- **input_ids** -- List of token ids to be fed to a model.
- **attention_mask** -- List of indices specifying which tokens should be attended to by the model.
- **input_image_embeds** -- Pixel values to be fed to a model.
- **image_sizes** -- List of tuples specifying the size of each image in `input_image_embeds`.
- **image_attention_mask** -- List of attention masks for each image in `input_image_embeds`.
- **input_audio_embeds** -- Audio embeddings to be fed to a model.
- **audio_embed_sizes** -- List of integers specifying the size of each audio in `input_audio_embeds`.

## Phi4MultimodalAudioConfig[[transformers.Phi4MultimodalAudioConfig]]

#### transformers.Phi4MultimodalAudioConfig[[transformers.Phi4MultimodalAudioConfig]]

```python
transformers.Phi4MultimodalAudioConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, hidden_size: int = 1024, intermediate_size: int = 1536, num_blocks: int = 24, num_attention_heads: int = 16, activation: str = 'swish', chunk_size: int = -1, left_chunk: int = 18, dropout_rate: float | int = 0.0, ext_pw_out_channel: int = 1024, depthwise_separable_out_channel: int = 1024, depthwise_multiplier: int = 1, kernel_size: int = 3, conv_activation: str = 'swish', input_size: int = 80, conv_glu_type: str = 'swish', time_reduction: int = 8, bias_max_distance: int = 1000, bias_symmetric: bool = False, nemo_activation: str = 'relu', nemo_conv_channels: int = 1024, downsample_rate: int = 1, initializer_range: float = 0.02, audio_token_id: int = 200011, feature_layer: int = -2)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L68)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1024`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `1536`) : Dimension of the MLP representations.

num_blocks (`int`, *optional*, defaults to 24) : Number of hidden layers in the Transformer encoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

activation (`str`, *optional*, defaults to `"swish"`) : The non-linear activation function in the MLPs.

chunk_size (`int`, *optional*, defaults to -1) : The chunk size to create the masks.

left_chunk (`int`, *optional*, defaults to 18) : The left chunk to create the masks.

dropout_rate (`float`, *optional*, defaults to 0.0) : The dropout ratio.

ext_pw_out_channel (`int`, *optional*, defaults to 1024) : Number of out channels in the point-wise conv modules.

depthwise_separable_out_channel (`int`, *optional*, defaults to 1024) : Number of out channels in the depth-wise separable conv modules.

depthwise_multiplier (`int`, *optional*, defaults to 1) : Input size multiplier for the depth-wise separable conv modules.

kernel_size (`int`, *optional*, defaults to 3) : Kernel size for the depth-wise separable conv modules.

conv_activation (`str`, *optional*, defaults to `"swish"`) : The non-linear activation function in the conv modules.

input_size (`int`, *optional*, defaults to 80) : Input size for the audio model.

conv_glu_type (`str`, *optional*, defaults to `"swish"`) : The non-linear activation function in the point-wise conv modules.

time_reduction (`int`, *optional*, defaults to 8) : Time reduction (subsampling factor).

bias_max_distance (`int`, *optional*, defaults to 1000) : Max distance for the relative attention bias module.

bias_symmetric (`bool`, *optional*, defaults to `False`) : Whether the relative attention bias should be symmetric or not.

nemo_activation (`str`, *optional*, defaults to `"relu"`) : The non-linear activation function in the nemo conv modules.

nemo_conv_channels (`int`, *optional*, defaults to 1024) : Number of channels in the nemo conv modules.

downsample_rate (`int`, *optional*, defaults to 1) : Downsample rate for the audio feature extractor.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

audio_token_id (`int`, *optional*, defaults to `200011`) : The audio token index used as a placeholder for input audio.

feature_layer (`int`, *optional*, defaults to -2) : The index of the layer of the encoder from which to extract audio features.

This is the configuration class to store the configuration of a Phi4MultimodalModel. It is used to instantiate a Phi4 Multimodal
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Phi4MultimodalAudioConfig

>>> # Initializing a Phi4MultimodalAudioConfig with microsoft/Phi-4-multimodal-instruct style configuration
>>> configuration = Phi4MultimodalAudioConfig()
```

## Phi4MultimodalVisionConfig[[transformers.Phi4MultimodalVisionConfig]]

#### transformers.Phi4MultimodalVisionConfig[[transformers.Phi4MultimodalVisionConfig]]

```python
transformers.Phi4MultimodalVisionConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, hidden_size: int = 1152, intermediate_size: int = 4304, num_hidden_layers: int = 27, num_attention_heads: int = 16, num_channels: int = 3, image_size: int | list[int] | tuple[int, int] = 448, patch_size: int | list[int] | tuple[int, int] = 14, hidden_act: str = 'gelu_pytorch_tanh', layer_norm_eps: float = 1e-06, attention_dropout: float | int = 0.0, crop_size: int = 448, image_token_id: int = 200010, feature_layer: int = -2)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L32)

**Parameters:**

hidden_size (`int`, *optional*, defaults to `1152`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `4304`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `27`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `16`) : Number of attention heads for each attention layer in the Transformer decoder.

num_channels (`int`, *optional*, defaults to `3`) : The number of input channels.

image_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `448`) : The size (resolution) of each image.

patch_size (`Union[int, list[int], tuple[int, int]]`, *optional*, defaults to `14`) : The size (resolution) of each patch.

hidden_act (`str`, *optional*, defaults to `gelu_pytorch_tanh`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

layer_norm_eps (`float`, *optional*, defaults to `1e-06`) : The epsilon used by the layer normalization layers.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

crop_size (`int`, *optional*, defaults to 448) : Crop size for the input images.

image_token_id (`int`, *optional*, defaults to `200010`) : The image token index used as a placeholder for input images.

feature_layer (`int`, *optional*, defaults to -2) : The index of the layer of the encoder from which to extract image features.

This is the configuration class to store the configuration of a Phi4MultimodalModel. It is used to instantiate a Phi4 Multimodal
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Phi4MultimodalVisionConfig

>>> # Initializing a Phi4MultimodalVisionConfig with microsoft/Phi-4-multimodal-instruct style configuration
>>> configuration = Phi4MultimodalVisionConfig()
```

## Phi4MultimodalConfig[[transformers.Phi4MultimodalConfig]]

#### transformers.Phi4MultimodalConfig[[transformers.Phi4MultimodalConfig]]

```python
transformers.Phi4MultimodalConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: typing.Union[str, ForwardRef('torch.dtype'), NoneType] = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: typing.Optional[typing.Literal['regression', 'single_label_classification', 'multi_label_classification']] = None, vocab_size: int = 200064, hidden_size: int = 3072, intermediate_size: int = 8192, num_hidden_layers: int = 32, num_attention_heads: int = 32, num_key_value_heads: int | None = 8, resid_pdrop: float | int = 0.0, embd_pdrop: float | int = 0.0, attention_dropout: float | int = 0.0, hidden_act: str = 'silu', max_position_embeddings: int = 131072, original_max_position_embeddings: int | None = 4096, initializer_range: float = 0.02, rms_norm_eps: float = 1e-05, use_cache: bool = True, tie_word_embeddings: bool = False, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None, bos_token_id: int | None = 199999, eos_token_id: int | list[int] | None = None, pad_token_id: int | None = 199999, sliding_window: int | None = None, vision_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None, audio_config: dict | transformers.configuration_utils.PreTrainedConfig | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L160)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `200064`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `3072`) : Dimension of the hidden representations.

intermediate_size (`int`, *optional*, defaults to `8192`) : Dimension of the MLP representations.

num_hidden_layers (`int`, *optional*, defaults to `32`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `32`) : Number of attention heads for each attention layer in the Transformer decoder.

num_key_value_heads (`int`, *optional*, defaults to `8`) : This is the number of key_value heads that should be used to implement Grouped Query Attention. If `num_key_value_heads=num_attention_heads`, the model will use Multi Head Attention (MHA), if `num_key_value_heads=1` the model will use Multi Query Attention (MQA) otherwise GQA is used. When converting a multi-head checkpoint to a GQA checkpoint, each group key and value head should be constructed by meanpooling all the original heads within that group. For more details, check out [this paper](https://huggingface.co/papers/2305.13245). If it is not specified, will default to `num_attention_heads`.

resid_pdrop (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

embd_pdrop (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the embeddings.

attention_dropout (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

hidden_act (`str`, *optional*, defaults to `silu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

max_position_embeddings (`int`, *optional*, defaults to `131072`) : The maximum sequence length that this model might ever be used with.

original_max_position_embeddings (`int`, *optional*, defaults to 4096) : The maximum sequence length that this model was trained with. This is used to determine the size of the original RoPE embeddings when using long scaling.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (`float`, *optional*, defaults to `1e-05`) : The epsilon used by the rms normalization layers.

use_cache (`bool`, *optional*, defaults to `True`) : Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if `config.is_decoder=True` or when the model is a decoder-only generative model.

tie_word_embeddings (`bool`, *optional*, defaults to `False`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

bos_token_id (`int`, *optional*, defaults to `199999`) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*) : Token id used for end-of-stream in the vocabulary.

pad_token_id (`int`, *optional*, defaults to `199999`) : Token id used for padding in the vocabulary.

sliding_window (`int`, *optional*) : Sliding window attention window size. If `None`, no sliding window is applied.

vision_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the vision backbone.

audio_config (`Union[dict, ~configuration_utils.PreTrainedConfig]`, *optional*) : The config object or dictionary of the audio backbone.

This is the configuration class to store the configuration of a Phi4MultimodalModel. It is used to instantiate a Phi4 Multimodal
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.15.1/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Example:

```python
>>> from transformers import Phi4MultimodalModel, Phi4MultimodalConfig

>>> # Initializing a Phi4Multimodal style configuration
>>> configuration = Phi4MultimodalConfig.from_pretrained("microsoft/Phi-4-multimodal-instruct")

>>> # Initializing a model from the configuration
>>> model = Phi4MultimodalModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

#### validate_rope[[transformers.Phi4MultimodalConfig.validate_rope]]

```python
validate_rope()
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/configuration_phi4_multimodal.py#L255)

Validate the `rope_parameters` configuration.

## Phi4MultimodalAudioModel[[transformers.Phi4MultimodalAudioModel]]

#### transformers.Phi4MultimodalAudioModel[[transformers.Phi4MultimodalAudioModel]]

```python
transformers.Phi4MultimodalAudioModel(config: Phi4MultimodalAudioConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L917)

#### forward_embeddings[[transformers.Phi4MultimodalAudioModel.forward_embeddings]]

```python
forward_embeddings(hidden_states, masks)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L953)

Forwarding the inputs through the top embedding layers

## Phi4MultimodalVisionModel[[transformers.Phi4MultimodalVisionModel]]

#### transformers.Phi4MultimodalVisionModel[[transformers.Phi4MultimodalVisionModel]]

```python
transformers.Phi4MultimodalVisionModel(config: Phi4MultimodalVisionConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L391)

## Phi4MultimodalModel[[transformers.Phi4MultimodalModel]]

#### transformers.Phi4MultimodalModel[[transformers.Phi4MultimodalModel]]

```python
transformers.Phi4MultimodalModel(config: Phi4MultimodalConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1479)

**Parameters:**

config ([Phi4MultimodalConfig](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The bare Phi4 Multimodal Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Phi4MultimodalModel.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, image_pixel_values: typing.Optional[torch.FloatTensor] = None, image_sizes: typing.Optional[torch.LongTensor] = None, image_attention_mask = None, audio_input_features: typing.Optional[torch.FloatTensor] = None, audio_embed_sizes = None, audio_attention_mask = None, use_cache: bool | None = None, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1501)

image_pixel_values (`torch.FloatTensor`, *optional*):
If the input contains images, these correspond to the pixel values after transformations (as returned by
the Processor)
image_sizes (`torch.LongTensor`, *optional*):
If the input contains images, these correspond to size of each image.
image_attention_mask (`torch.LongTensor`, *optional*):
Attention mask for the images.
audio_input_features (`torch.FloatTensor`, *optional*):
If the input contains audio samples, these correspond to the values after transformation (as returned by
the Processor).
audio_embed_sizes (`torch.Tensor`, *optional*):
Size of the audio inputs.
audio_attention_mask (`torch.Tensor, *optional*):
Attention mask for the audio inputs.

## Phi4MultimodalForCausalLM[[transformers.Phi4MultimodalForCausalLM]]

#### transformers.Phi4MultimodalForCausalLM[[transformers.Phi4MultimodalForCausalLM]]

```python
transformers.Phi4MultimodalForCausalLM(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1589)

**Parameters:**

config ([Phi4MultimodalForCausalLM](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalForCausalLM)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Phi4 Multimodal Model for causal language modeling.

This model inherits from [PreTrainedModel](/docs/transformers/v5.15.1/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.Phi4MultimodalForCausalLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, past_key_values: transformers.cache_utils.Cache | None = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, image_pixel_values: typing.Optional[torch.FloatTensor] = None, image_sizes: typing.Optional[torch.LongTensor] = None, image_attention_mask = None, audio_input_features: typing.Optional[torch.FloatTensor] = None, audio_embed_sizes = None, audio_attention_mask = None, labels: typing.Optional[torch.LongTensor] = None, use_cache: bool | None = None, logits_to_keep: typing.Union[int, torch.Tensor] = 0, **kwargs)
```

[Source](https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/models/phi4_multimodal/modeling_phi4_multimodal.py#L1604)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.15.1/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.15.1/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

past_key_values (`~cache_utils.Cache`, *optional*) : Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values` returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.  Only [Cache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.Cache) instance is allowed as input, see our [kv cache guide](https://huggingface.co/docs/transformers/en/kv_cache). If no `past_key_values` are passed, [DynamicCache](/docs/transformers/v5.15.1/en/internal/generation_utils#transformers.DynamicCache) will be initialized by default.  The model will output the same cache format that is fed as input.  If `past_key_values` are used, the user is expected to input only unprocessed `input_ids` (those that don't have their past key value states given to this model) of shape `(batch_size, unprocessed_length)` instead of all `input_ids` of shape `(batch_size, sequence_length)`.

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

image_pixel_values (`torch.FloatTensor`, *optional*) : If the input contains images, these correspond to the pixel values after transformations (as returned by the Processor)

image_sizes (`torch.LongTensor`, *optional*) : If the input contains images, these correspond to size of each image.

image_attention_mask (`torch.LongTensor`, *optional*) : Attention mask for the images.

audio_input_features (`torch.FloatTensor`, *optional*) : If the input contains audio samples, these correspond to the values after transformation (as returned by the Processor).

audio_embed_sizes (`torch.Tensor`, *optional*) : Size of the audio inputs.

audio_attention_mask (`torch.Tensor, *optional*) : Attention mask for the audio inputs.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should either be in `[0, ..., config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

use_cache (`bool`, *optional*) : If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see `past_key_values`).

logits_to_keep (`Union[int, torch.Tensor]`, *optional*, defaults to `0`) : If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that token can save memory, which becomes pretty significant for long sequences or large vocabulary size. If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension. This is useful when using packed tensor format (single dimension for batch and sequence length).

**Returns:** [CausalLMOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or `tuple(torch.FloatTensor)`

A [CausalLMOutputWithPast](/docs/transformers/v5.15.1/en/main_classes/output#transformers.modeling_outputs.CausalLMOutputWithPast) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([Phi4MultimodalConfig](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalConfig)) and inputs.

The [Phi4MultimodalForCausalLM](/docs/transformers/v5.15.1/en/model_doc/phi4_multimodal#transformers.Phi4MultimodalForCausalLM) forward method, overrides the `__call__` special method.

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

Example:
```python
>>> from transformers import AutoTokenizer, Phi4MultimodalForCausalLM
>>> model = Phi4MultimodalForCausalLM.from_pretrained("TBA")
>>> tokenizer = AutoTokenizer.from_pretrained("TBA")
>>> prompt = "This is an example script ."
>>> inputs = tokenizer(prompt, return_tensors="pt")
>>> # Generate
>>> generate_ids = model.generate(inputs.input_ids, max_length=30)
>>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
'This is an example script .\n Certainly! Below is a sample script that demonstrates a simple task, such as calculating the sum'
```

### GOT-OCR2
https://huggingface.co/docs/transformers/v5.15.1/model_doc/got_ocr2.md
