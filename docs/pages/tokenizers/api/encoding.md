# Encoding

## Encoding[[tokenizers.Encoding]]

#### tokenizers.Encoding[[tokenizers.Encoding]]

```python
tokenizers.Encoding()
```

The [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding) represents the output of a [Tokenizer](/docs/tokenizers/v0.23.2/en/api/tokenizer#tokenizers.Tokenizer).

It holds all the information about the tokenized input, including the token IDs,
token strings, attention masks, offsets, and more. This is the main data structure
returned by `encode()` and
`encode_batch()`.

Example:

```python
>>> from tokenizers import Tokenizer
>>> tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
>>> encoding = tokenizer.encode("Hello, world!")
>>> encoding.ids
[101, 7592, 1010, 2088, 999, 102]
>>> encoding.tokens
['[CLS]', 'hello', ',', 'world', '!', '[SEP]']
>>> encoding.offsets
[(0, 0), (0, 5), (5, 6), (7, 12), (12, 13), (0, 0)]
```

#### attention_mask[[tokenizers.Encoding.attention_mask]]

**Returns:** `List[int]`

The attention mask

The attention mask

This indicates to the LM which tokens should be attended to, and which should not.
This is especially important when batching sequences, where we need to applying
padding.

#### ids[[tokenizers.Encoding.ids]]

**Returns:** `List[int]`

The list of IDs

The generated IDs

The IDs are the main input to a Language Model. They are the token indices,
the numerical representations that a LM understands.

#### n_sequences[[tokenizers.Encoding.n_sequences]]

**Returns:** `int`

The number of sequences in this [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding)

The number of sequences represented

#### offsets[[tokenizers.Encoding.offsets]]

**Returns:** A `List` of `Tuple[int, int]`

The list of offsets

The offsets associated to each token

These offsets let's you slice the input string, and thus retrieve the original
part that led to producing the corresponding token.

#### overflowing[[tokenizers.Encoding.overflowing]]

A `List` of overflowing [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding)

When using truncation, the [Tokenizer](/docs/tokenizers/v0.23.2/en/api/tokenizer#tokenizers.Tokenizer) takes care of splitting
the output into as many pieces as required to match the specified maximum length.
This field lets you retrieve all the subsequent pieces.

When you use pairs of sequences, the overflowing pieces will contain enough
variations to cover all the possible combinations, while respecting the provided
maximum length.

#### sequence_ids[[tokenizers.Encoding.sequence_ids]]

**Returns:** A `List` of `Optional[int]`

A list of optional sequence index.

The generated sequence indices.

They represent the index of the input sequence associated to each token.
The sequence id can be None if the token is not related to any input sequence,
like for example with special tokens.

#### special_tokens_mask[[tokenizers.Encoding.special_tokens_mask]]

**Returns:** `List[int]`

The special tokens mask

The special token mask

This indicates which tokens are special tokens, and which are not.

#### tokens[[tokenizers.Encoding.tokens]]

**Returns:** `List[str]`

The list of tokens

The generated tokens

They are the string representation of the IDs.

#### type_ids[[tokenizers.Encoding.type_ids]]

**Returns:** `List[int]`

The list of type ids

The generated type IDs

Generally used for tasks like sequence classification or question answering,
these tokens let the LM know which input sequence corresponds to each tokens.

#### word_ids[[tokenizers.Encoding.word_ids]]

**Returns:** A `List` of `Optional[int]`

A list of optional word index.

The generated word indices.

They represent the index of the word associated to each token.
When the input is pre-tokenized, they correspond to the ID of the given input label,
otherwise they correspond to the words indices as defined by the
[PreTokenizer](/docs/tokenizers/v0.23.2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) that was used.

For special tokens and such (any token that was generated from something that was
not part of the input), the output is `None`

#### words[[tokenizers.Encoding.words]]

**Returns:** A `List` of `Optional[int]`

A list of optional word index.

The generated word indices.

This is deprecated and will be removed in a future version.
Please use `~tokenizers.Encoding.word_ids` instead.

They represent the index of the word associated to each token.
When the input is pre-tokenized, they correspond to the ID of the given input label,
otherwise they correspond to the words indices as defined by the
[PreTokenizer](/docs/tokenizers/v0.23.2/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) that was used.

For special tokens and such (any token that was generated from something that was
not part of the input), the output is `None`

#### char_to_token[[tokenizers.Encoding.char_to_token]]

```python
char_to_token(char_pos, sequence_index = 0)
```

**Parameters:**

char_pos (`int`) : The position of a char in the input string

sequence_index (`int`, defaults to `0`) : The index of the sequence that contains the target char

**Returns:** `int`

The index of the token that contains this char in the encoded sequence

Get the token that contains the char at the given position in the input sequence.

#### char_to_word[[tokenizers.Encoding.char_to_word]]

```python
char_to_word(char_pos, sequence_index = 0)
```

**Parameters:**

char_pos (`int`) : The position of a char in the input string

sequence_index (`int`, defaults to `0`) : The index of the sequence that contains the target char

**Returns:** `int`

The index of the word that contains this char in the input sequence

Get the word that contains the char at the given position in the input sequence.

#### merge[[tokenizers.Encoding.merge]]

```python
merge(encodings, growing_offsets = True)
```

**Parameters:**

encodings (A `List` of [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding)) : The list of encodings that should be merged in one 

growing_offsets (`bool`, defaults to `True`) : Whether the offsets should accumulate while merging

**Returns:** [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding)

The resulting Encoding

Merge the list of encodings into one final [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding)

#### pad[[tokenizers.Encoding.pad]]

```python
pad(length, direction = 'right', pad_id = 0, pad_type_id = 0, pad_token = '[PAD]')
```

**Parameters:**

length (`int`) : The desired length 

direction : (`str`, defaults to `right`): The expected padding direction. Can be either `right` or `left` 

pad_id (`int`, defaults to `0`) : The ID corresponding to the padding token 

pad_type_id (`int`, defaults to `0`) : The type ID corresponding to the padding token 

pad_token (`str`, defaults to *[PAD]*) : The pad token to use

Pad the [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding) at the given length

#### set_sequence_id[[tokenizers.Encoding.set_sequence_id]]

```python
set_sequence_id(sequence_id)
```

Set the given sequence index

Set the given sequence index for the whole range of tokens contained in this
[Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding).

#### token_to_chars[[tokenizers.Encoding.token_to_chars]]

```python
token_to_chars(token_index)
```

**Parameters:**

token_index (`int`) : The index of a token in the encoded sequence.

**Returns:** `Tuple[int, int]`

The token offsets `(first, last + 1)`

Get the offsets of the token at the given index.

The returned offsets are related to the input sequence that contains the
token.  In order to determine in which input sequence it belongs, you
must call `~tokenizers.Encoding.token_to_sequence()`.

#### token_to_sequence[[tokenizers.Encoding.token_to_sequence]]

```python
token_to_sequence(token_index)
```

**Parameters:**

token_index (`int`) : The index of a token in the encoded sequence.

**Returns:** `int`

The sequence id of the given token

Get the index of the sequence represented by the given token.

In the general use case, this method returns `0` for a single sequence or
the first sequence of a pair, and `1` for the second sequence of a pair

#### token_to_word[[tokenizers.Encoding.token_to_word]]

```python
token_to_word(token_index)
```

**Parameters:**

token_index (`int`) : The index of a token in the encoded sequence.

**Returns:** `int`

The index of the word in the relevant input sequence.

Get the index of the word that contains the token in one of the input sequences.

The returned word index is related to the input sequence that contains
the token.  In order to determine in which input sequence it belongs, you
must call `~tokenizers.Encoding.token_to_sequence()`.

#### truncate[[tokenizers.Encoding.truncate]]

```python
truncate(max_length, stride = 0, direction = 'right')
```

**Parameters:**

max_length (`int`) : The desired length 

stride (`int`, defaults to `0`) : The length of previous content to be included in each overflowing piece 

direction (`str`, defaults to `right`) : Truncate direction

Truncate the [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding) at the given length

If this [Encoding](/docs/tokenizers/v0.23.2/en/api/encoding#tokenizers.Encoding) represents multiple sequences, when truncating
this information is lost. It will be considered as representing a single sequence.

#### word_to_chars[[tokenizers.Encoding.word_to_chars]]

```python
word_to_chars(word_index, sequence_index = 0)
```

**Parameters:**

word_index (`int`) : The index of a word in one of the input sequences.

sequence_index (`int`, defaults to `0`) : The index of the sequence that contains the target word

**Returns:** `Tuple[int, int]`

The range of characters (span) `(first, last + 1)`

Get the offsets of the word at the given index in one of the input sequences.

#### word_to_tokens[[tokenizers.Encoding.word_to_tokens]]

```python
word_to_tokens(word_index, sequence_index = 0)
```

**Parameters:**

word_index (`int`) : The index of a word in one of the input sequences.

sequence_index (`int`, defaults to `0`) : The index of the sequence that contains the target word

**Returns:** `Tuple[int, int]`

The range of tokens: `(first, last + 1)`

Get the encoded tokens corresponding to the word at the given index
in one of the input sequences.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Models
https://huggingface.co/docs/tokenizers/v0.23.2/api/models.md
