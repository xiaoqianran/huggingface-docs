# Pre-tokenizers

## BertPreTokenizer[[tokenizers.pre_tokenizers.BertPreTokenizer]]

#### tokenizers.pre_tokenizers.BertPreTokenizer[[tokenizers.pre_tokenizers.BertPreTokenizer]]

BertPreTokenizer

This pre-tokenizer splits tokens on whitespace and punctuation. Each occurrence of
a punctuation character will be treated as a separate token. This is the pre-tokenizer
used by the original BERT model.

Example:

```python
>>> from tokenizers.pre_tokenizers import BertPreTokenizer
>>> pre_tokenizer = BertPreTokenizer()
>>> pre_tokenizer.pre_tokenize_str("Hello, I'm a single sentence!")
[('Hello', (0, 5)), (',', (5, 6)), ('I', (7, 8)), ("'", (8, 9)), ('m', (9, 10)), ('a', (11, 12)), ('single', (13, 19)), ('sentence', (20, 28)), ('!', (28, 29))]
```

## ByteLevel[[tokenizers.pre_tokenizers.ByteLevel]]

#### tokenizers.pre_tokenizers.ByteLevel[[tokenizers.pre_tokenizers.ByteLevel]]

ByteLevel PreTokenizer

This pre-tokenizer takes care of replacing all bytes of the given string
with a corresponding representation, as well as splitting into words.

Example:

```python
>>> from tokenizers.pre_tokenizers import ByteLevel
>>> pre_tokenizer = ByteLevel()
>>> pre_tokenizer.pre_tokenize_str("Hello my friend, how is it going?")
[('ĠHello', (0, 5)), ('Ġmy', (5, 8)), ('Ġfriend,', (8, 15)), ('Ġhow', (15, 19)), ('Ġis', (19, 22)), ('Ġit', (22, 25)), ('Ġgoing?', (25, 32))]
```

alphabettokenizers.pre_tokenizers.ByteLevel.alphabet[]`List[str]`A list of characters that compose the alphabet
Returns the alphabet used by this PreTokenizer.

Since the ByteLevel works as its name suggests, at the byte level, it
encodes each byte value to a unique visible character. This means that there is a
total of 256 different characters composing this alphabet.

**Parameters:**

add_prefix_space (`bool`, *optional*, defaults to `True`) : Whether to add a space to the first word if there isn't already one. This lets us treat *hello* exactly like *say hello*.

use_regex (`bool`, *optional*, defaults to `True`) : Set this to `False` to prevent this *pre_tokenizer* from using the GPT2 specific regexp for spliting on whitespace.

**Returns:**

``List[str]``

A list of characters that compose the alphabet

## CharDelimiterSplit[[tokenizers.pre_tokenizers.CharDelimiterSplit]]

#### tokenizers.pre_tokenizers.CharDelimiterSplit[[tokenizers.pre_tokenizers.CharDelimiterSplit]]

This pre-tokenizer simply splits on the provided char. Works like `str.split`
with a single-character delimiter.

Example:

```python
>>> from tokenizers.pre_tokenizers import CharDelimiterSplit
>>> pre_tokenizer = CharDelimiterSplit("x")
>>> pre_tokenizer.pre_tokenize_str("helloxthere")
[('hello', (0, 5)), ('there', (6, 11))]
```

**Parameters:**

delimiter (`str`) : The single character that will be used to split the input. The delimiter is removed from the output.

## Digits[[tokenizers.pre_tokenizers.Digits]]

#### tokenizers.pre_tokenizers.Digits[[tokenizers.pre_tokenizers.Digits]]

This pre-tokenizer simply splits using the digits in separate tokens

If set to True, digits will each be separated as follows:

```python
"Call 123 please" -> "Call ", "1", "2", "3", " please"
```

If set to False, digits will grouped as follows:

```python
"Call 123 please" -> "Call ", "123", " please"
```

**Parameters:**

individual_digits (`bool`, *optional*, defaults to `False`) --

## Metaspace[[tokenizers.pre_tokenizers.Metaspace]]

#### tokenizers.pre_tokenizers.Metaspace[[tokenizers.pre_tokenizers.Metaspace]]

Metaspace pre-tokenizer

This pre-tokenizer replaces any whitespace by the provided replacement character.
It then tries to split on these spaces.

Example:

```python
>>> from tokenizers.pre_tokenizers import Metaspace
>>> pre_tokenizer = Metaspace()
>>> pre_tokenizer.pre_tokenize_str("Hello my friend")
[('▁Hello', (0, 5)), ('▁my', (6, 8)), ('▁friend', (9, 15))]
```

**Parameters:**

replacement (`str`, *optional*, defaults to `▁`) : The replacement character. Must be exactly one character. By default we use the *▁* (U+2581) meta symbol (Same as in SentencePiece). 

prepend_scheme (`str`, *optional*, defaults to `"always"`) : Whether to add a space to the first word if there isn't already one. This lets us treat *hello* exactly like *say hello*. Choices: "always", "never", "first". First means the space is only added on the first token (relevant when special tokens are used or other pre_tokenizer are used).

## PreTokenizer[[tokenizers.pre_tokenizers.PreTokenizer]]

#### tokenizers.pre_tokenizers.PreTokenizer[[tokenizers.pre_tokenizers.PreTokenizer]]

Base class for all pre-tokenizers

This class is not supposed to be instantiated directly. Instead, any implementation of a
PreTokenizer will return an instance of this class when instantiated.

pre_tokenizetokenizers.pre_tokenizers.PreTokenizer.pre_tokenize[{"name": "pretok", "val": ""}]- **pretok** (`~tokenizers.PreTokenizedString) --
  The pre-tokenized string on which to apply this
  :class:`~tokenizers.pre_tokenizers.PreTokenizer`0
Pre-tokenize a `~tokenizers.PyPreTokenizedString` in-place

This method allows to modify a `PreTokenizedString` to
keep track of the pre-tokenization, and leverage the capabilities of the
`PreTokenizedString`. If you just want to see the result of
the pre-tokenization of a raw string, you can use
`pre_tokenize_str()`

**Parameters:**

pretok (`~tokenizers.PreTokenizedString) : The pre-tokenized string on which to apply this :class:`~tokenizers.pre_tokenizers.PreTokenizer`
#### pre_tokenize_str[[tokenizers.pre_tokenizers.PreTokenizer.pre_tokenize_str]]

Pre tokenize the given string

This method provides a way to visualize the effect of a
[PreTokenizer](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) but it does not keep track of the
alignment, nor does it provide all the capabilities of the
`PreTokenizedString`. If you need some of these, you can use
`pre_tokenize()`

**Parameters:**

sequence (`str`) : A string to pre-tokeize

**Returns:**

``List[Tuple[str, Offsets]]``

A list of tuple with the pre-tokenized parts and their offsets

## Punctuation[[tokenizers.pre_tokenizers.Punctuation]]

#### tokenizers.pre_tokenizers.Punctuation[[tokenizers.pre_tokenizers.Punctuation]]

This pre-tokenizer simply splits on punctuation as individual characters.

Example:

```python
>>> from tokenizers.pre_tokenizers import Punctuation
>>> pre_tokenizer = Punctuation()
>>> pre_tokenizer.pre_tokenize_str("Hello, how are you?")
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (15, 18)), ('?', (18, 19))]
```

**Parameters:**

behavior (`SplitDelimiterBehavior`) : The behavior to use when splitting. Choices: "removed", "isolated" (default), "merged_with_previous", "merged_with_next", "contiguous"

## Sequence[[tokenizers.pre_tokenizers.Sequence]]

#### tokenizers.pre_tokenizers.Sequence[[tokenizers.pre_tokenizers.Sequence]]

This pre-tokenizer composes other pre-tokenizers and applies them in sequence.
Each pre-tokenizer in the list is applied to the output of the previous one,
allowing complex tokenization strategies to be built by chaining simpler components.

Example:

```python
>>> from tokenizers.pre_tokenizers import Punctuation, Whitespace, Sequence
>>> pre_tokenizer = Sequence([Whitespace(), Punctuation()])
>>> pre_tokenizer.pre_tokenize_str("Hello, world!")
[('Hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ('!', (12, 13))]
```

**Parameters:**

pretokenizers (`List[PreTokenizer]`) : A list of [PreTokenizer](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.PreTokenizer) to be applied in sequence.

## Split[[tokenizers.pre_tokenizers.Split]]

#### tokenizers.pre_tokenizers.Split[[tokenizers.pre_tokenizers.Split]]

Split PreTokenizer

This versatile pre-tokenizer splits using the provided pattern and
according to the provided behavior. The pattern can be inverted by
making use of the invert flag.

Example:

```python
>>> from tokenizers import Regex
>>> from tokenizers.pre_tokenizers import Split
>>> # Split on commas, removing them
>>> pre_tokenizer = Split(",", behavior="removed")
>>> pre_tokenizer.pre_tokenize_str("one,two,three")
[('one', (0, 3)), ('two', (4, 7)), ('three', (8, 13))]
>>> # Split using a regex, keeping the delimiter isolated
>>> Split(Regex(r"\s+"), behavior="isolated").pre_tokenize_str("hello   world")
[('hello', (0, 5)), ('   ', (5, 8)), ('world', (8, 13))]
```

**Parameters:**

pattern (`str` or `Regex`) : A pattern used to split the string. Usually a string or a regex built with *tokenizers.Regex*. If you want to use a regex pattern, it has to be wrapped around a *tokenizers.Regex*, otherwise we consider is as a string pattern. For example *pattern="|"* means you want to split on *|* (imagine a csv file for example), while *pattern=tokenizers.Regex("1|2")* means you split on either '1' or '2'.

behavior (`SplitDelimiterBehavior`) : The behavior to use when splitting. Choices: "removed", "isolated", "merged_with_previous", "merged_with_next", "contiguous" 

invert (`bool`, *optional*, defaults to `False`) : Whether to invert the pattern.

## UnicodeScripts[[tokenizers.pre_tokenizers.UnicodeScripts]]

#### tokenizers.pre_tokenizers.UnicodeScripts[[tokenizers.pre_tokenizers.UnicodeScripts]]

This pre-tokenizer splits on characters that belong to different language families.
It roughly follows the SentencePiece script boundaries, with Hiragana and Katakana
fused into the Han script category. This mimics the SentencePiece Unigram
implementation and is useful for multilingual models that need to handle CJK text.

Example:

```python
>>> from tokenizers.pre_tokenizers import UnicodeScripts
>>> pre_tokenizer = UnicodeScripts()
>>> pre_tokenizer.pre_tokenize_str("どこ Where")
[('どこ', (0, 2)), ('Where', (3, 8))]
```

## Whitespace[[tokenizers.pre_tokenizers.Whitespace]]

#### tokenizers.pre_tokenizers.Whitespace[[tokenizers.pre_tokenizers.Whitespace]]

This pre-tokenizer splits on word boundaries according to the `\w+|[^\w\s]+`
regex pattern. It splits on word characters or characters that aren't words or
whitespaces (punctuation such as hyphens, apostrophes, commas, etc.).

Example:

```python
>>> from tokenizers.pre_tokenizers import Whitespace
>>> pre_tokenizer = Whitespace()
>>> pre_tokenizer.pre_tokenize_str("Hello, world! Let's tokenize.")
[('Hello', (0, 5)), (',', (5, 6)), ('world', (7, 12)), ('!', (12, 13)), ('Let', (14, 17)), ("'", (17, 18)), ('s', (18, 19)), ('tokenize', (20, 28)), ('.', (28, 29))]
```

## WhitespaceSplit[[tokenizers.pre_tokenizers.WhitespaceSplit]]

#### tokenizers.pre_tokenizers.WhitespaceSplit[[tokenizers.pre_tokenizers.WhitespaceSplit]]

This pre-tokenizer simply splits on whitespace. Works like `str.split` with no
arguments — it splits on any whitespace and discards the whitespace tokens. Unlike
[Whitespace](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.Whitespace), it does not split on punctuation.

Example:

```python
>>> from tokenizers.pre_tokenizers import WhitespaceSplit
>>> pre_tokenizer = WhitespaceSplit()
>>> pre_tokenizer.pre_tokenize_str("Hello, world! How are you?")
[('Hello,', (0, 6)), ('world!', (7, 13)), ('How', (14, 17)), ('are', (18, 21)), ('you?', (22, 26))]
```

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Normalizers
https://huggingface.co/docs/tokenizers/v0.23.1/api/normalizers.md
