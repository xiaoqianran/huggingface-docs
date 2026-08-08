# Normalizers

## ByteLevel[[tokenizers.normalizers.ByteLevel]]

#### tokenizers.normalizers.ByteLevel[[tokenizers.normalizers.ByteLevel]]

Bytelevel Normalizer

Converts all bytes in the input to their Unicode representation using the GPT-2
byte-to-unicode mapping. Every byte value (0–255) is mapped to a unique visible
character so that any arbitrary binary input can be tokenized without needing a
special unknown token.

This normalizer is used together with the
[ByteLevel](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel) pre-tokenizer and
[ByteLevel](/docs/tokenizers/v0.23.1/en/api/decoders#tokenizers.decoders.ByteLevel) decoder.

Example:

```python
>>> from tokenizers.normalizers import ByteLevel
>>> normalizer = ByteLevel()
>>> normalizer.normalize_str("hello\nworld")
'helloĊworld'
```

## Lowercase[[tokenizers.normalizers.Lowercase]]

#### tokenizers.normalizers.Lowercase[[tokenizers.normalizers.Lowercase]]

Lowercase Normalizer

Converts all text to lowercase using Unicode-aware lowercasing. This is equivalent
to calling `str.lower` on the input.

Example:

```python
>>> from tokenizers.normalizers import Lowercase
>>> normalizer = Lowercase()
>>> normalizer.normalize_str("Hello World")
'hello world'
```

## NFC[[tokenizers.normalizers.NFC]]

#### tokenizers.normalizers.NFC[[tokenizers.normalizers.NFC]]

NFC Unicode Normalizer

Applies Unicode NFC (Canonical Decomposition, followed by Canonical Composition)
normalization. First decomposes characters, then recomposes them using canonical
composition rules. This produces the canonical composed form.

Example:

```python
>>> from tokenizers.normalizers import NFC
>>> normalizer = NFC()
>>> normalizer.normalize_str("e\u0301")  # 'e' + combining accent
'é'
```

## NFD[[tokenizers.normalizers.NFD]]

#### tokenizers.normalizers.NFD[[tokenizers.normalizers.NFD]]

NFD Unicode Normalizer

Applies Unicode NFD (Canonical Decomposition) normalization. Decomposes characters into
their canonical components. For example, accented characters like `é` (U+00E9) are
decomposed into `e` (U+0065) + combining accent (U+0301).

This is often used as a first step before stripping accents with
[StripAccents](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.StripAccents).

Example:

```python
>>> from tokenizers.normalizers import NFD
>>> normalizer = NFD()
>>> normalizer.normalize_str("Héllo")
'He\u0301llo'
```

## NFKC[[tokenizers.normalizers.NFKC]]

#### tokenizers.normalizers.NFKC[[tokenizers.normalizers.NFKC]]

NFKC Unicode Normalizer

Applies Unicode NFKC (Compatibility Decomposition, followed by Canonical Composition)
normalization. Like NFC but also maps compatibility characters to their canonical
equivalents. This is the normalization used by Python's `str.casefold` and
by many NLP pipelines.

Example:

```python
>>> from tokenizers.normalizers import NFKC
>>> normalizer = NFKC()
>>> normalizer.normalize_str("ﬁne caf\u00e9")
'fine café'
```

## NFKD[[tokenizers.normalizers.NFKD]]

#### tokenizers.normalizers.NFKD[[tokenizers.normalizers.NFKD]]

NFKD Unicode Normalizer

Applies Unicode NFKD (Compatibility Decomposition) normalization. Like NFD but also
decomposes compatibility characters. For example, the ligature `ﬁ` (U+FB01) is
decomposed into `f` + `i`.

Example:

```python
>>> from tokenizers.normalizers import NFKD
>>> normalizer = NFKD()
>>> normalizer.normalize_str("ﬁne")
'fine'
```

## Nmt[[tokenizers.normalizers.Nmt]]

#### tokenizers.normalizers.Nmt[[tokenizers.normalizers.Nmt]]

Nmt normalizer

Normalizer used in the Google NMT pipeline. It handles various text cleaning tasks
including removing control characters, normalizing whitespace, and replacing certain
Unicode characters. This is equivalent to the normalization done in the original
SentencePiece NMT preprocessing.

Example:

```python
>>> from tokenizers.normalizers import Nmt
>>> normalizer = Nmt()
>>> normalizer.normalize_str("Hello\x00World")
'Hello World'
```

## Normalizer[[tokenizers.normalizers.Normalizer]]

#### tokenizers.normalizers.Normalizer[[tokenizers.normalizers.Normalizer]]

Base class for all normalizers

This class is not supposed to be instantiated directly. Instead, any implementation of a
Normalizer will return an instance of this class when instantiated.

normalizetokenizers.normalizers.Normalizer.normalize[{"name": "normalized", "val": ""}]- **normalized** (`NormalizedString`) --
  The normalized string on which to apply this
  [Normalizer](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.Normalizer)0
Normalize a `NormalizedString` in-place

This method allows to modify a `NormalizedString` to
keep track of the alignment information. If you just want to see the result
of the normalization on a raw string, you can use
`normalize_str()`

**Parameters:**

normalized (`NormalizedString`) : The normalized string on which to apply this [Normalizer](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.Normalizer)
#### normalize_str[[tokenizers.normalizers.Normalizer.normalize_str]]

Normalize the given string

This method provides a way to visualize the effect of a
[Normalizer](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.Normalizer) but it does not keep track of the alignment
information. If you need to get/convert offsets, you can use
`normalize()`

**Parameters:**

sequence (`str`) : A string to normalize

**Returns:**

``str``

A string after normalization

## Precompiled[[tokenizers.normalizers.Precompiled]]

#### tokenizers.normalizers.Precompiled[[tokenizers.normalizers.Precompiled]]

Precompiled normalizer

A normalizer that uses a precompiled character map built from a SentencePiece model.
This normalizer is automatically extracted from SentencePiece `.model` files and
should not be constructed manually — it is used internally for full compatibility
with SentencePiece-based tokenizers.

**Parameters:**

precompiled_charsmap (`bytes`) : The raw bytes of the precompiled character map, as found inside a SentencePiece `.model` file.

## Replace[[tokenizers.normalizers.Replace]]

#### tokenizers.normalizers.Replace[[tokenizers.normalizers.Replace]]

Replace normalizer

Replaces occurrences of a pattern in the input string with the given content.
The pattern can be either a plain string or a regular expression wrapped in
`Regex`.

Example:

```python
>>> from tokenizers import Regex
>>> from tokenizers.normalizers import Replace
>>> # Replace a literal string
>>> Replace(".", " ").normalize_str("hello.world")
'hello world'
>>> # Replace using a regex
>>> Replace(Regex(r"\s+"), " ").normalize_str("hello   world")
'hello world'
```

**Parameters:**

pattern (`str` or `Regex`) : The pattern to search for. Use a plain string for literal replacement, or wrap a regex pattern in `Regex` for regex replacement. 

content (`str`) : The string to replace each match with.

## Sequence[[tokenizers.normalizers.Sequence]]

#### tokenizers.normalizers.Sequence[[tokenizers.normalizers.Sequence]]

Allows concatenating multiple other Normalizer as a Sequence.
All the normalizers run in sequence in the given order

Example:

```python
>>> from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
>>> normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
>>> normalizer.normalize_str("Héllo Wörld")
'hello world'
```

**Parameters:**

normalizers (`List[Normalizer]`) : A list of Normalizer to be run as a sequence

## Strip[[tokenizers.normalizers.Strip]]

#### tokenizers.normalizers.Strip[[tokenizers.normalizers.Strip]]

Strip normalizer

Removes leading and/or trailing whitespace from the input string.

Example:

```python
>>> from tokenizers.normalizers import Strip
>>> normalizer = Strip()
>>> normalizer.normalize_str("  hello world  ")
'hello world'
>>> Strip(right=False).normalize_str("  hello  ")
'hello  '
```

**Parameters:**

left (`bool`, defaults to `True`) : Whether to strip leading (left) whitespace. 

right (`bool`, defaults to `True`) : Whether to strip trailing (right) whitespace.

## StripAccents[[tokenizers.normalizers.StripAccents]]

#### tokenizers.normalizers.StripAccents[[tokenizers.normalizers.StripAccents]]

StripAccents normalizer

Strips all accent marks (combining diacritical characters) from the input. This
normalizer should typically be used after applying [NFD](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.NFD)
or [NFKD](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.NFKD) decomposition, which separates base
characters from their combining accents.

Example:

```python
>>> from tokenizers.normalizers import NFD, StripAccents, Sequence
>>> normalizer = Sequence([NFD(), StripAccents()])
>>> normalizer.normalize_str("café")
'cafe'
```

## BertNormalizer[[tokenizers.normalizers.BertNormalizer]]

#### tokenizers.normalizers.BertNormalizer[[tokenizers.normalizers.BertNormalizer]]

BertNormalizer

Takes care of normalizing raw text before giving it to a Bert model.
This includes cleaning the text, handling accents, chinese chars and lowercasing

Example:

```python
>>> from tokenizers.normalizers import BertNormalizer
>>> normalizer = BertNormalizer(lowercase=True)
>>> normalizer.normalize_str("Héllo WORLD")
'hello world'
```

**Parameters:**

clean_text (`bool`, *optional*, defaults to `True`) : Whether to clean the text, by removing any control characters and replacing all whitespaces by the classic one. 

handle_chinese_chars (`bool`, *optional*, defaults to `True`) : Whether to handle chinese chars by putting spaces around them. 

strip_accents (`bool`, *optional*) : Whether to strip all accents. If this option is not specified (ie == None), then it will be determined by the value for *lowercase* (as in the original Bert). 

lowercase (`bool`, *optional*, defaults to `True`) : Whether to lowercase.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.
