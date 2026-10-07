# GTE

## Overview

GTE was proposed in [mGTE: Generalized Long-Context Text Representation and Reranking Models for Multilingual Text Retrieval](https://huggingface.co/papers/2407.19669) by Xin Zhang, Yanzhao Zhang, Dingkun Long, Wen Xie, Ziqi Dai, Jialong Tang, Huan Lin, Baosong Yang, Pengjun Xie, Fei Huang, Meishan Zhang, Wenjie Li and Min Zhang.

The abstract from the paper is the following:

*We present systematic efforts in building long-context multilingual text representation model (TRM) and reranker from scratch for text retrieval. We first introduce a text encoder (base size) enhanced with RoPE and unpadding, pre-trained in a native 8192-token context (longer than 512 of previous multilingual encoders). Then we construct a hybrid TRM and a cross-encoder reranker by contrastive learning. Evaluations show that our text encoder outperforms the same-sized previous state-of-the-art XLM-R. Meanwhile, our TRM and reranker match the performance of large-sized state-of-the-art BGE-M3 models and achieve better results on long-context retrieval benchmarks. Further analysis demonstrate that our proposed models exhibit higher efficiency during both training and inference. We believe their efficiency and effectiveness could benefit various researches and industrial applications.*

GTE is a BERT-style bidirectional encoder that replaces absolute position embeddings with RoPE, uses a gated MLP, and applies layer normalization after each residual connection. The same architecture backs Alibaba's `gte-*-v1.5`, `gte-multilingual-*` and `gte-en-mlm-*` checkpoints as well as Snowflake's `snowflake-arctic-embed-m-v2.0`.

This model was contributed by [Harshal Janjani](https://huggingface.co/harshaljanjani).
The original code can be found [here](https://huggingface.co/Alibaba-NLP/new-impl).

> [!TIP]
> Click on the GTE models in the right sidebar for more examples of how to apply GTE to different language tasks.

The example below demonstrates how to extract features (embeddings) with [Pipeline](/docs/transformers/v5.19.0/en/main_classes/pipelines#transformers.Pipeline) and [AutoModel](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoModel).

```python
from transformers import pipeline

pipeline = pipeline(
    task="feature-extraction",
    model="alibaba-nlp-community/gte-multilingual-base",
)
pipeline("Plants create oxygen through a process known as photosynthesis.")
```

```python
import torch

from transformers import AutoModel, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("alibaba-nlp-community/gte-multilingual-base")
model = AutoModel.from_pretrained("alibaba-nlp-community/gte-multilingual-base", device_map="auto")
inputs = tokenizer("Plants create oxygen through a process known as photosynthesis.", return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]

print(f"Embeddings shape: {embeddings.shape}")
```

## Notes

- GTE uses RoPE, so for correct positional encoding either use right padding (the default), or use left padding and prepare `position_ids` accordingly.
- `type_vocab_size` differs across checkpoints. `alibaba-nlp-community/gte-base-en-v1.5` sets it to `0`, in which case no token type embedding is created and `token_type_ids` are ignored.
- The `gte-*-v1.5` and `gte-multilingual-*` checkpoints apply static NTK scaling on top of RoPE. It is expressed as a `linear` [RopeParameters](/docs/transformers/v5.19.0/en/internal/rope_utils#transformers.RopeParameters) entry whose `rope_theta` is the base scaled by the NTK factor.

## GteConfig[[transformers.GteConfig]]

#### transformers.GteConfig[[transformers.GteConfig]]

```python
transformers.GteConfig(transformers_version: str | None = None, architectures: list[str] | None = None, output_hidden_states: bool | None = False, return_dict: bool | None = True, dtype: str | torch.dtype | None = None, chunk_size_feed_forward: int = 0, is_encoder_decoder: bool = False, id2label: dict[int, str] | dict[str, str] | None = None, label2id: dict[str, int] | dict[str, str] | None = None, problem_type: Literal['regression', 'single_label_classification', 'multi_label_classification'] | None = None, vocab_size: int = 250048, hidden_size: int = 768, num_hidden_layers: int = 12, num_attention_heads: int = 12, intermediate_size: int = 3072, hidden_act: str = 'gelu', hidden_dropout_prob: float | int = 0.1, attention_probs_dropout_prob: float | int = 0.0, max_position_embeddings: int = 8192, type_vocab_size: int = 1, initializer_range: float = 0.02, layer_norm_eps: float = 1e-12, pad_token_id: int | None = 1, classifier_dropout: float | int | None = None, bos_token_id: int | None = None, eos_token_id: int | list[int] | None = None, tie_word_embeddings: bool = True, rope_parameters: transformers.modeling_rope_utils.RopeParameters | dict | None = None)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/configuration_gte.py#L30)

**Parameters:**

vocab_size (`int`, *optional*, defaults to `250048`) : Vocabulary size of the model. Defines the number of different tokens that can be represented by the `input_ids`.

hidden_size (`int`, *optional*, defaults to `768`) : Dimension of the hidden representations.

num_hidden_layers (`int`, *optional*, defaults to `12`) : Number of hidden layers in the Transformer decoder.

num_attention_heads (`int`, *optional*, defaults to `12`) : Number of attention heads for each attention layer in the Transformer decoder.

intermediate_size (`int`, *optional*, defaults to `3072`) : Dimension of the MLP representations.

hidden_act (`str`, *optional*, defaults to `gelu`) : The non-linear activation function (function or string) in the decoder. For example, `"gelu"`, `"relu"`, `"silu"`, etc.

hidden_dropout_prob (`Union[float, int]`, *optional*, defaults to `0.1`) : The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.

attention_probs_dropout_prob (`Union[float, int]`, *optional*, defaults to `0.0`) : The dropout ratio for the attention probabilities.

max_position_embeddings (`int`, *optional*, defaults to `8192`) : The maximum sequence length that this model might ever be used with.

type_vocab_size (`int`, *optional*, defaults to `1`) : The vocabulary size of the `token_type_ids`.

initializer_range (`float`, *optional*, defaults to `0.02`) : The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

layer_norm_eps (`float`, *optional*, defaults to `1e-12`) : The epsilon used by the layer normalization layers.

pad_token_id (`int`, *optional*, defaults to `1`) : Token id used for padding in the vocabulary.

classifier_dropout (`Union[float, int]`, *optional*) : The dropout ratio for classifier.

bos_token_id (`int`, *optional*) : Token id used for beginning-of-stream in the vocabulary.

eos_token_id (`Union[int, list[int]]`, *optional*) : Token id used for end-of-stream in the vocabulary.

tie_word_embeddings (`bool`, *optional*, defaults to `True`) : Whether to tie weight embeddings according to model's `tied_weights_keys` mapping.

rope_parameters (`Union[~modeling_rope_utils.RopeParameters, dict]`, *optional*) : Dictionary containing the configuration parameters for the RoPE embeddings. The dictionary should contain a value for `rope_theta` and optionally parameters used for scaling in case you want to use RoPE with longer `max_position_embeddings`.

This is the configuration class to store the configuration of a GteModel. It is used to instantiate a Gte
model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
defaults will yield a similar configuration to that of the [alibaba-nlp-community/gte-multilingual-base](https://huggingface.co/alibaba-nlp-community/gte-multilingual-base)

Configuration objects inherit from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) and can be used to control the model outputs. Read the
documentation from [PreTrainedConfig](/docs/transformers/v5.19.0/en/main_classes/configuration#transformers.PreTrainedConfig) for more information.

Examples:

```python
>>> from transformers import GteConfig, GteModel

>>> # Initializing a GTE alibaba-nlp-community/gte-multilingual-base style configuration
>>> configuration = GteConfig()

>>> # Initializing a model (with random weights) from the alibaba-nlp-community/gte-multilingual-base style configuration
>>> model = GteModel(configuration)

>>> # Accessing the model configuration
>>> configuration = model.config
```

## GteModel[[transformers.GteModel]]

#### transformers.GteModel[[transformers.GteModel]]

```python
transformers.GteModel(config: GteConfig, add_pooling_layer = False)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L337)

**Parameters:**

config ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

add_pooling_layer (`bool`, *optional*, defaults to `False`) : Whether to add a pooling layer.

The bare Gte Model outputting raw hidden-states without any specific head on top.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GteModel.forward]]

```python
forward(input_ids: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, token_type_ids: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L364)

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

**Returns:** [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or `tuple(torch.FloatTensor)`

A [BaseModelOutputWithPooling](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) and inputs.

The [GteModel](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteModel) forward method, overrides the `__call__` special method.

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

## GteForMaskedLM[[transformers.GteForMaskedLM]]

#### transformers.GteForMaskedLM[[transformers.GteForMaskedLM]]

```python
transformers.GteForMaskedLM(config: GteConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L444)

**Parameters:**

config ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Gte Model with a `language modeling` head on top."

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GteForMaskedLM.forward]]

```python
forward(input_ids: typing.Optional[torch.LongTensor] = None, attention_mask: typing.Optional[torch.FloatTensor] = None, token_type_ids: typing.Optional[torch.LongTensor] = None, position_ids: typing.Optional[torch.LongTensor] = None, inputs_embeds: typing.Optional[torch.FloatTensor] = None, labels: typing.Optional[torch.LongTensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L462)

**Parameters:**

input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

**Returns:** [MaskedLMOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or `tuple(torch.FloatTensor)`

A [MaskedLMOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.MaskedLMOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) and inputs.

The [GteForMaskedLM](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteForMaskedLM) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Masked language modeling (MLM) loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`) -- Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoTokenizer, GteForMaskedLM
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("alibaba-nlp-community/gte-multilingual-base")
>>> model = GteForMaskedLM.from_pretrained("alibaba-nlp-community/gte-multilingual-base")

>>> inputs = tokenizer("The capital of France is <mask>.", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> # retrieve index of <mask>
>>> mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

>>> predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)
>>> tokenizer.decode(predicted_token_id)
...

>>> labels = tokenizer("The capital of France is Paris.", return_tensors="pt")["input_ids"]
>>> # mask labels of non-<mask> tokens
>>> labels = torch.where(inputs.input_ids == tokenizer.mask_token_id, labels, -100)

>>> outputs = model(**inputs, labels=labels)
>>> round(outputs.loss.item(), 2)
...
```

## GteForSequenceClassification[[transformers.GteForSequenceClassification]]

#### transformers.GteForSequenceClassification[[transformers.GteForSequenceClassification]]

```python
transformers.GteForSequenceClassification(config: GteConfig)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L514)

**Parameters:**

config ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

Gte Model transformer with a sequence classification/regression head on top (a linear layer on top of the pooled
output) e.g. for GLUE tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GteForSequenceClassification.forward]]

```python
forward(input_ids: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, token_type_ids: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.Tensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L529)

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*) : Labels for computing the sequence classification/regression loss. Indices should be in `[0, ..., config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If `config.num_labels > 1` a classification loss is computed (Cross-Entropy).

**Returns:** [SequenceClassifierOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or `tuple(torch.FloatTensor)`

A [SequenceClassifierOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.SequenceClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) and inputs.

The [GteForSequenceClassification](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteForSequenceClassification) forward method, overrides the `__call__` special method.

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
>>> from transformers import AutoTokenizer, GteForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("alibaba-nlp-community/gte-multilingual-base")
>>> model = GteForSequenceClassification.from_pretrained("alibaba-nlp-community/gte-multilingual-base")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_id = logits.argmax().item()
>>> model.config.id2label[predicted_class_id]
...

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = GteForSequenceClassification.from_pretrained("alibaba-nlp-community/gte-multilingual-base", num_labels=num_labels)

>>> labels = torch.tensor([1])
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

Example of multi-label classification:

```python
>>> import torch
>>> from transformers import AutoTokenizer, GteForSequenceClassification

>>> tokenizer = AutoTokenizer.from_pretrained("alibaba-nlp-community/gte-multilingual-base")
>>> model = GteForSequenceClassification.from_pretrained("alibaba-nlp-community/gte-multilingual-base", problem_type="multi_label_classification")

>>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

>>> # To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
>>> num_labels = len(model.config.id2label)
>>> model = GteForSequenceClassification.from_pretrained(
...     "alibaba-nlp-community/gte-multilingual-base", num_labels=num_labels, problem_type="multi_label_classification"
... )

>>> labels = torch.sum(
...     torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
... ).to(torch.float)
>>> loss = model(**inputs, labels=labels).loss
```

## GteForTokenClassification[[transformers.GteForTokenClassification]]

#### transformers.GteForTokenClassification[[transformers.GteForTokenClassification]]

```python
transformers.GteForTokenClassification(config)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L588)

**Parameters:**

config ([GteForTokenClassification](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteForTokenClassification)) : Model configuration class with all the parameters of the model. Initializing with a config file does not load the weights associated with the model, only the configuration. Check out the [from_pretrained()](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel.from_pretrained) method to load the model weights.

The Gte transformer with a token classification head on top (a linear layer on top of the hidden-states
output) e.g. for Named-Entity-Recognition (NER) tasks.

This model inherits from [PreTrainedModel](/docs/transformers/v5.19.0/en/main_classes/model#transformers.PreTrainedModel). Check the superclass documentation for the generic methods the
library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
etc.)

This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
and behavior.

#### forward[[transformers.GteForTokenClassification.forward]]

```python
forward(input_ids: typing.Optional[torch.Tensor] = None, attention_mask: typing.Optional[torch.Tensor] = None, token_type_ids: typing.Optional[torch.Tensor] = None, position_ids: typing.Optional[torch.Tensor] = None, inputs_embeds: typing.Optional[torch.Tensor] = None, labels: typing.Optional[torch.Tensor] = None, **kwargs: Unpack)
```

[Source](https://github.com/huggingface/transformers/blob/v5.19.0/src/transformers/models/gte/modeling_gte.py#L603)

**Parameters:**

input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of input sequence tokens in the vocabulary. Padding will be ignored by default.  Indices can be obtained using [AutoTokenizer](/docs/transformers/v5.19.0/en/model_doc/auto#transformers.AutoTokenizer). See [PreTrainedTokenizer.encode()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.encode) and [PreTrainedTokenizer.__call__()](/docs/transformers/v5.19.0/en/internal/tokenization_utils#transformers.PreTrainedTokenizerBase.__call__) for details.  [What are input IDs?](../glossary#input-ids)

attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:  - 1 for tokens that are **not masked**, - 0 for tokens that are **masked**.  [What are attention masks?](../glossary#attention-mask)

token_type_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:  - 0 corresponds to a *sentence A* token, - 1 corresponds to a *sentence B* token.  [What are token type IDs?](../glossary#token-type-ids)

position_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*) : Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.  [What are position IDs?](../glossary#position-ids)

inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*) : Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more control over how to convert `input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*) : Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

**Returns:** [TokenClassifierOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or `tuple(torch.FloatTensor)`

A [TokenClassifierOutput](/docs/transformers/v5.19.0/en/main_classes/output#transformers.modeling_outputs.TokenClassifierOutput) or a tuple of
`torch.FloatTensor` (if `return_dict=False` is passed or when `config.return_dict=False`) comprising various
elements depending on the configuration ([GteConfig](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteConfig)) and inputs.

The [GteForTokenClassification](/docs/transformers/v5.19.0/en/model_doc/gte#transformers.GteForTokenClassification) forward method, overrides the `__call__` special method.

Although the recipe for forward pass needs to be defined within this function, one should call the `Module`
instance afterwards instead of this since the former takes care of running the pre and post processing steps while
the latter silently ignores them.

- **loss** (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided) -- Classification loss.
- **logits** (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.num_labels)`) -- Classification scores (before SoftMax).
- **hidden_states** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`) -- Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
  one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

  Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
- **attentions** (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`) -- Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
  sequence_length)`.

  Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
  heads.

Example:

```python
>>> from transformers import AutoTokenizer, GteForTokenClassification
>>> import torch

>>> tokenizer = AutoTokenizer.from_pretrained("alibaba-nlp-community/gte-multilingual-base")
>>> model = GteForTokenClassification.from_pretrained("alibaba-nlp-community/gte-multilingual-base")

>>> inputs = tokenizer(
...     "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
... )

>>> with torch.no_grad():
...     logits = model(**inputs).logits

>>> predicted_token_class_ids = logits.argmax(-1)

>>> # Note that tokens are classified rather then input words which means that
>>> # there might be more predicted token classes than words.
>>> # Multiple token classes might account for the same word
>>> predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
>>> predicted_tokens_classes
...

>>> labels = predicted_token_class_ids
>>> loss = model(**inputs, labels=labels).loss
>>> round(loss.item(), 2)
...
```

### GLM-4-0414
https://huggingface.co/docs/transformers/v5.19.0/model_doc/glm4.md
