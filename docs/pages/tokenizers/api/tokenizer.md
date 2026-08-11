# Tokenizer

## Tokenizer[[tokenizers.Tokenizer]]

#### tokenizers.Tokenizer[[tokenizers.Tokenizer]]

A `Tokenizer` works as a pipeline. It processes some raw text as input
and outputs an [Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding).

The pipeline is structured as follows:

1. The [Normalizer](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.Normalizer) normalizes the raw input text.
2. The [PreTokenizer](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) splits the normalized text
   into word-level tokens.
3. The [Model](/docs/tokenizers/v0.23.1/en/api/models#tokenizers.models.Model) tokenizes each word into subword tokens
   and maps them to IDs.
4. The `PostProcessor` applies any final
   transformations (e.g., adding special tokens like `[CLS]` and `[SEP]`).

Example:

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.models import BPE
>>> from tokenizers.normalizers import Lowercase
>>> from tokenizers.pre_tokenizers import Whitespace
>>> tokenizer = Tokenizer(BPE(unk_token="REDACTED"))
>>> tokenizer.normalizer = Lowercase()
>>> tokenizer.pre_tokenizer = Whitespace()
>>> # Load a pre-built tokenizer from HuggingFace Hub
>>> tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
```

decodertokenizers.Tokenizer.decoder[]
The *optional* `Decoder` in use by the Tokenizer

**Parameters:**

model ([Model](/docs/tokenizers/v0.23.1/en/api/models#tokenizers.models.Model)) : The core algorithm that this `Tokenizer` should be using.
#### model[[tokenizers.Tokenizer.model]]

The [Model](/docs/tokenizers/v0.23.1/en/api/models#tokenizers.models.Model) in use by the Tokenizer
#### normalizer[[tokenizers.Tokenizer.normalizer]]

The *optional* [Normalizer](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.Normalizer) in use by the Tokenizer
#### padding[[tokenizers.Tokenizer.padding]]

Get the current padding parameters

*Cannot be set, use* `enable_padding()` *instead*

**Returns:**

`(`dict`, *optional*)`

A dict with the current padding parameters if padding is enabled
#### post_processor[[tokenizers.Tokenizer.post_processor]]

The *optional* `PostProcessor` in use by the Tokenizer
#### pre_tokenizer[[tokenizers.Tokenizer.pre_tokenizer]]

The *optional* [PreTokenizer](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) in use by the Tokenizer
#### truncation[[tokenizers.Tokenizer.truncation]]

Get the currently set truncation parameters

*Cannot set, use* `enable_truncation()` *instead*

**Returns:**

`(`dict`, *optional*)`

A dict with the current truncation parameters if truncation is enabled
#### add_special_tokens[[tokenizers.Tokenizer.add_special_tokens]]

Add the given special tokens to the Tokenizer.

If these tokens are already part of the vocabulary, it just let the Tokenizer know about
them. If they don't exist, the Tokenizer creates them, giving them a new id.

These special tokens will never be processed by the model (ie won't be split into
multiple tokens), and they can be removed from the output when decoding.

**Parameters:**

tokens (A `List` of [AddedToken](/docs/tokenizers/v0.23.1/en/api/added-tokens#tokenizers.AddedToken) or `str`) : The list of special tokens we want to add to the vocabulary. Each token can either be a string or an instance of [AddedToken](/docs/tokenizers/v0.23.1/en/api/added-tokens#tokenizers.AddedToken) for more customization.

**Returns:**

``int``

The number of tokens that were created in the vocabulary
#### add_tokens[[tokenizers.Tokenizer.add_tokens]]

Add the given tokens to the vocabulary

The given tokens are added only if they don't already exist in the vocabulary.
Each token then gets a new attributed id.

**Parameters:**

tokens (A `List` of [AddedToken](/docs/tokenizers/v0.23.1/en/api/added-tokens#tokenizers.AddedToken) or `str`) : The list of tokens we want to add to the vocabulary. Each token can be either a string or an instance of [AddedToken](/docs/tokenizers/v0.23.1/en/api/added-tokens#tokenizers.AddedToken) for more customization.

**Returns:**

``int``

The number of tokens that were created in the vocabulary
#### async_decode_batch[[tokenizers.Tokenizer.async_decode_batch]]

Decode a batch of ids back to their corresponding string

**Parameters:**

sequences (`List` of `List[int]`) : The batch of sequences we want to decode 

skip_special_tokens (`bool`, defaults to `True`) : Whether the special tokens should be removed from the decoded strings

**Returns:**

``List[str]``

A list of decoded strings
#### async_encode[[tokenizers.Tokenizer.async_encode]]

Asynchronously encode the given input with character offsets.

This is an async version of encode that can be awaited in async Python code.

Example:

Here are some examples of the inputs that are accepted:

```python
await async_encode("A single sequence")
```

**Parameters:**

sequence (`~tokenizers.InputSequence`) : The main input sequence we want to encode. This sequence can be either raw text or pre-tokenized, according to the `is_pretokenized` argument:  - If `is_pretokenized=False`: `TextInputSequence` - If `is_pretokenized=True`: `PreTokenizedInputSequence()` 

pair (`~tokenizers.InputSequence`, *optional*) : An optional input sequence. The expected format is the same that for `sequence`. 

is_pretokenized (`bool`, defaults to `False`) : Whether the input is already pre-tokenized 

add_special_tokens (`bool`, defaults to `True`) : Whether to add the special tokens

**Returns:**

`[Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding)`

The encoded result
#### async_encode_batch[[tokenizers.Tokenizer.async_encode_batch]]

Asynchronously encode the given batch of inputs with character offsets.

This is an async version of encode_batch that can be awaited in async Python code.

Example:

Here are some examples of the inputs that are accepted:

```python
await async_encode_batch([
"A single sequence",
("A tuple with a sequence", "And its pair"),
[ "A", "pre", "tokenized", "sequence" ],
([ "A", "pre", "tokenized", "sequence" ], "And its pair")
])
```

**Parameters:**

input (A `List`/``Tuple` of `~tokenizers.EncodeInput`) : A list of single sequences or pair sequences to encode. Each sequence can be either raw text or pre-tokenized, according to the `is_pretokenized` argument:  - If `is_pretokenized=False`: `TextEncodeInput()` - If `is_pretokenized=True`: `PreTokenizedEncodeInput()` 

is_pretokenized (`bool`, defaults to `False`) : Whether the input is already pre-tokenized 

add_special_tokens (`bool`, defaults to `True`) : Whether to add the special tokens

**Returns:**

`A `List` of [`~tokenizers.Encoding``]`

The encoded batch
#### async_encode_batch_fast[[tokenizers.Tokenizer.async_encode_batch_fast]]

Asynchronously encode the given batch of inputs without tracking character offsets.

This is an async version of encode_batch_fast that can be awaited in async Python code.

Example:

Here are some examples of the inputs that are accepted:

```python
await async_encode_batch_fast([
"A single sequence",
("A tuple with a sequence", "And its pair"),
[ "A", "pre", "tokenized", "sequence" ],
([ "A", "pre", "tokenized", "sequence" ], "And its pair")
])
```

**Parameters:**

input (A `List`/``Tuple` of `~tokenizers.EncodeInput`) : A list of single sequences or pair sequences to encode. Each sequence can be either raw text or pre-tokenized, according to the `is_pretokenized` argument:  - If `is_pretokenized=False`: `TextEncodeInput()` - If `is_pretokenized=True`: `PreTokenizedEncodeInput()` 

is_pretokenized (`bool`, defaults to `False`) : Whether the input is already pre-tokenized 

add_special_tokens (`bool`, defaults to `True`) : Whether to add the special tokens

**Returns:**

`A `List` of [`~tokenizers.Encoding``]`

The encoded batch
#### decode[[tokenizers.Tokenizer.decode]]

Decode the given list of ids back to a string

This is used to decode anything coming back from a Language Model

**Parameters:**

ids (A `List/Tuple` of `int`) : The list of ids that we want to decode 

skip_special_tokens (`bool`, defaults to `True`) : Whether the special tokens should be removed from the decoded string

**Returns:**

``str``

The decoded string
#### decode_batch[[tokenizers.Tokenizer.decode_batch]]

Decode a batch of ids back to their corresponding string

**Parameters:**

sequences (`List` of `List[int]`) : The batch of sequences we want to decode 

skip_special_tokens (`bool`, defaults to `True`) : Whether the special tokens should be removed from the decoded strings

**Returns:**

``List[str]``

A list of decoded strings
#### enable_padding[[tokenizers.Tokenizer.enable_padding]]

Enable the padding

**Parameters:**

direction (`str`, *optional*, defaults to `right`) : The direction in which to pad. Can be either `right` or `left` 

pad_to_multiple_of (`int`, *optional*) : If specified, the padding length should always snap to the next multiple of the given value. For example if we were going to pad witha length of 250 but `pad_to_multiple_of=8` then we will pad to 256. 

pad_id (`int`, defaults to 0) : The id to be used when padding 

pad_type_id (`int`, defaults to 0) : The type id to be used when padding 

pad_token (`str`, defaults to `[PAD]`) : The pad token to be used when padding 

length (`int`, *optional*) : If specified, the length at which to pad. If not specified we pad using the size of the longest sequence in a batch.
#### enable_truncation[[tokenizers.Tokenizer.enable_truncation]]

Enable truncation

**Parameters:**

max_length (`int`) : The max length at which to truncate 

stride (`int`, *optional*) : The length of the previous first sequence to be included in the overflowing sequence 

strategy (`str`, *optional*, defaults to `longest_first`) : The strategy used to truncation. Can be one of `longest_first`, `only_first` or `only_second`. 

direction (`str`, defaults to `right`) : Truncate direction
#### encode[[tokenizers.Tokenizer.encode]]

Encode the given sequence and pair. This method can process raw text sequences
as well as already pre-tokenized sequences.

Example:

Here are some examples of the inputs that are accepted:

```python
encode("A single sequence")*
encode("A sequence", "And its pair")*
encode([ "A", "pre", "tokenized", "sequence" ], is_pretokenized=True)`
encode(
[ "A", "pre", "tokenized", "sequence" ], [ "And", "its", "pair" ],
is_pretokenized=True
)
```

**Parameters:**

sequence (`~tokenizers.InputSequence`) : The main input sequence we want to encode. This sequence can be either raw text or pre-tokenized, according to the `is_pretokenized` argument:  - If `is_pretokenized=False`: `TextInputSequence` - If `is_pretokenized=True`: `PreTokenizedInputSequence()` 

pair (`~tokenizers.InputSequence`, *optional*) : An optional input sequence. The expected format is the same that for `sequence`. 

is_pretokenized (`bool`, defaults to `False`) : Whether the input is already pre-tokenized 

add_special_tokens (`bool`, defaults to `True`) : Whether to add the special tokens

**Returns:**

`[Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding)`

The encoded result
#### encode_batch[[tokenizers.Tokenizer.encode_batch]]

Encode the given batch of inputs. This method accept both raw text sequences
as well as already pre-tokenized sequences. The reason we use *PySequence* is
because it allows type checking with zero-cost (according to PyO3) as we don't
have to convert to check.

Example:

Here are some examples of the inputs that are accepted:

```python
encode_batch([
"A single sequence",
("A tuple with a sequence", "And its pair"),
[ "A", "pre", "tokenized", "sequence" ],
([ "A", "pre", "tokenized", "sequence" ], "And its pair")
])
```

**Parameters:**

input (A `List`/``Tuple` of `~tokenizers.EncodeInput`) : A list of single sequences or pair sequences to encode. Each sequence can be either raw text or pre-tokenized, according to the `is_pretokenized` argument:  - If `is_pretokenized=False`: `TextEncodeInput()` - If `is_pretokenized=True`: `PreTokenizedEncodeInput()` 

is_pretokenized (`bool`, defaults to `False`) : Whether the input is already pre-tokenized 

add_special_tokens (`bool`, defaults to `True`) : Whether to add the special tokens

**Returns:**

`A `List` of [`~tokenizers.Encoding``]`

The encoded batch
#### encode_batch_fast[[tokenizers.Tokenizer.encode_batch_fast]]

Encode the given batch of inputs. This method is faster than *encode_batch*
because it doesn't keep track of offsets, they will be all zeros.

Example:

Here are some examples of the inputs that are accepted:

```python
encode_batch_fast([
"A single sequence",
("A tuple with a sequence", "And its pair"),
[ "A", "pre", "tokenized", "sequence" ],
([ "A", "pre", "tokenized", "sequence" ], "And its pair")
])
```

**Parameters:**

input (A `List`/``Tuple` of `~tokenizers.EncodeInput`) : A list of single sequences or pair sequences to encode. Each sequence can be either raw text or pre-tokenized, according to the `is_pretokenized` argument:  - If `is_pretokenized=False`: `TextEncodeInput()` - If `is_pretokenized=True`: `PreTokenizedEncodeInput()` 

is_pretokenized (`bool`, defaults to `False`) : Whether the input is already pre-tokenized 

add_special_tokens (`bool`, defaults to `True`) : Whether to add the special tokens

**Returns:**

`A `List` of [`~tokenizers.Encoding``]`

The encoded batch
#### from_buffer[[tokenizers.Tokenizer.from_buffer]]

Instantiate a new [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer) from the given buffer.

**Parameters:**

buffer (`bytes`) : A buffer containing a previously serialized [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)

**Returns:**

`[Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)`

The new tokenizer
#### from_file[[tokenizers.Tokenizer.from_file]]

Instantiate a new [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer) from the file at the given path.

**Parameters:**

path (`str`) : A path to a local JSON file representing a previously serialized [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)

**Returns:**

`[Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)`

The new tokenizer
#### from_pretrained[[tokenizers.Tokenizer.from_pretrained]]

Instantiate a new [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer) from an existing file on the
Hugging Face Hub.

**Parameters:**

identifier (`str`) : The identifier of a Model on the Hugging Face Hub, that contains a tokenizer.json file

revision (`str`, defaults to *main*) : A branch or commit id

token (`str`, *optional*, defaults to *None*) : An optional auth token used to access private repositories on the Hugging Face Hub

**Returns:**

`[Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)`

The new tokenizer
#### from_str[[tokenizers.Tokenizer.from_str]]

Instantiate a new [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer) from the given JSON string.

**Parameters:**

json (`str`) : A valid JSON string representing a previously serialized [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)

**Returns:**

`[Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)`

The new tokenizer
#### get_added_tokens_decoder[[tokenizers.Tokenizer.get_added_tokens_decoder]]

Get the underlying vocabulary

**Returns:**

``Dict[int, AddedToken]``

The vocabulary
#### get_vocab[[tokenizers.Tokenizer.get_vocab]]

Get the underlying vocabulary

**Parameters:**

with_added_tokens (`bool`, defaults to `True`) : Whether to include the added tokens

**Returns:**

``Dict[str, int]``

The vocabulary
#### get_vocab_size[[tokenizers.Tokenizer.get_vocab_size]]

Get the size of the underlying vocabulary

**Parameters:**

with_added_tokens (`bool`, defaults to `True`) : Whether to include the added tokens

**Returns:**

``int``

The size of the vocabulary
#### id_to_token[[tokenizers.Tokenizer.id_to_token]]

Convert the given id to its corresponding token if it exists

**Parameters:**

id (`int`) : The id to convert

**Returns:**

``Optional[str]``

An optional token, `None` if out of vocabulary
#### no_padding[[tokenizers.Tokenizer.no_padding]]

Disable padding
#### no_truncation[[tokenizers.Tokenizer.no_truncation]]

Disable truncation
#### num_special_tokens_to_add[[tokenizers.Tokenizer.num_special_tokens_to_add]]

Return the number of special tokens that would be added for single/pair sentences.
:param is_pair: Boolean indicating if the input would be a single sentence or a pair
:return:
#### post_process[[tokenizers.Tokenizer.post_process]]

Apply all the post-processing steps to the given encodings.

The various steps are:

1. Truncate according to the set truncation params (provided with
   `enable_truncation()`)
2. Apply the `PostProcessor`
3. Pad according to the set padding params (provided with
   `enable_padding()`)

**Parameters:**

encoding ([Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding)) : The [Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding) corresponding to the main sequence. 

pair ([Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding), *optional*) : An optional [Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding) corresponding to the pair sequence. 

add_special_tokens (`bool`) : Whether to add the special tokens

**Returns:**

`[Encoding](/docs/tokenizers/v0.23.1/en/api/encoding#tokenizers.Encoding)`

The final post-processed encoding
#### save[[tokenizers.Tokenizer.save]]

Save the [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer) to the file at the given path.

**Parameters:**

path (`str`) : A path to a file in which to save the serialized tokenizer. 

pretty (`bool`, defaults to `True`) : Whether the JSON file should be pretty formatted.
#### to_str[[tokenizers.Tokenizer.to_str]]

Gets a serialized string representing this [Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer).

**Parameters:**

pretty (`bool`, defaults to `False`) : Whether the JSON string should be pretty formatted.

**Returns:**

``str``

A string representing the serialized Tokenizer
#### token_to_id[[tokenizers.Tokenizer.token_to_id]]

Convert the given token to its corresponding id if it exists

**Parameters:**

token (`str`) : The token to convert

**Returns:**

``Optional[int]``

An optional id, `None` if out of vocabulary
#### train[[tokenizers.Tokenizer.train]]

Train the Tokenizer using the given files.

Reads the files line by line, while keeping all the whitespace, even new lines.
If you want to train from data store in-memory, you can check
`train_from_iterator()`

**Parameters:**

files (`List[str]`) : A list of path to the files that we should use for training 

trainer (`~tokenizers.trainers.Trainer`, *optional*) : An optional trainer that should be used to train our Model
#### train_from_iterator[[tokenizers.Tokenizer.train_from_iterator]]

Train the Tokenizer using the provided iterator.

You can provide anything that is a Python Iterator

- A list of sequences `List[str]`
- A generator that yields `str` or `List[str]`
- A Numpy array of strings
- ...

**Parameters:**

iterator (`Iterator`) : Any iterator over strings or list of strings 

trainer (`~tokenizers.trainers.Trainer`, *optional*) : An optional trainer that should be used to train our Model 

length (`int`, *optional*) : The total number of sequences in the iterator. This is used to provide meaningful progress tracking

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Input Sequences
https://huggingface.co/docs/tokenizers/v0.23.1/api/input-sequences.md
