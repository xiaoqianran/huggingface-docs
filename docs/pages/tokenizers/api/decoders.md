# Decoders

## DecodeStream[[tokenizers.decoders.DecodeStream]]
#### tokenizers.decoders.DecodeStream[[tokenizers.decoders.DecodeStream]]

Provides incremental decoding of token IDs as they are generated, yielding
decoded text chunks as soon as they are available.

Unlike batch decoding, streaming decode is designed for use with autoregressive
generation — tokens arrive one at a time and the decoder needs to handle
multi-byte sequences (e.g., UTF-8 characters split across token boundaries) and
byte-fallback tokens gracefully.

The decoder internally buffers tokens until it can produce a valid UTF-8 string
chunk, then yields that chunk and advances its internal state. This means
individual calls to `step()` may return
`None` when the current token completes a partial sequence that cannot yet
be decoded.

Example:

```python
>>> from tokenizers import Tokenizer
>>> from tokenizers.decoders import DecodeStream
>>> tokenizer = Tokenizer.from_pretrained("gpt2")
>>> stream = DecodeStream(skip_special_tokens=True)
>>> # Simulate streaming token-by-token generation
>>> token_ids = tokenizer.encode("Hello, streaming world!").ids
>>> for token_id in token_ids:
...     chunk = stream.step(tokenizer, token_id)
...     if chunk is not None:
...         print(chunk, end="", flush=True)
```

steptokenizers.decoders.DecodeStream.step[{"name": "tokenizer", "val": ""}, {"name": "id", "val": ""}]- **tokenizer** ([Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)) --
  The tokenizer whose decoder pipeline will be used.

- **id** (`int` or `List[int]`) --
  The next token ID, or a list of token IDs to append to the stream.0`Optional[str]`The next decoded text chunk if enough tokens have
accumulated, or `None` if more tokens are still needed.
Add the next token ID (or list of IDs) to the stream and return the next
decoded text chunk if one is available.

Because some characters span multiple tokens (e.g. multi-byte UTF-8
sequences or byte-fallback tokens), this method may return `None`
when the provided token does not yet complete a decodable unit. Callers
should simply continue feeding tokens until a non-`None` value is
returned.

**Parameters:**

skip_special_tokens (`bool`, defaults to `False`) : Whether to skip special tokens (e.g. `[CLS]`, `[SEP]`, `&amp;lt;s>`) when decoding.

**Returns:**

``Optional[str]``

The next decoded text chunk if enough tokens have
accumulated, or `None` if more tokens are still needed.

## BPEDecoder[[tokenizers.decoders.BPEDecoder]]

#### tokenizers.decoders.BPEDecoder[[tokenizers.decoders.BPEDecoder]]

BPEDecoder Decoder

Example:

```python
>>> from tokenizers.decoders import BPEDecoder
>>> decoder = BPEDecoder()
>>> decoder.decode(["Hello&amp;lt;/w>", "world&amp;lt;/w>"])
'Hello world'
```

**Parameters:**

suffix (`str`, *optional*, defaults to `&amp;lt;/w>`) : The suffix that was used to characterize an end-of-word. This suffix will be replaced by whitespaces during the decoding

## ByteFallback[[tokenizers.decoders.ByteFallback]]

#### tokenizers.decoders.ByteFallback[[tokenizers.decoders.ByteFallback]]

ByteFallback Decoder

ByteFallback is a decoder that handles tokens representing raw bytes in the
`&amp;lt;0xNN>` format (e.g., `&amp;lt;0x61>` for the byte `0x61` = `'a'`). It converts
such tokens to their corresponding bytes and attempts to decode the resulting byte
sequence as UTF-8. This is used in LLaMA/SentencePiece models that use byte fallback
for unknown characters. Inconvertible byte tokens are replaced with the Unicode
replacement character (U+FFFD).

Example:

```python
>>> from tokenizers.decoders import ByteFallback, Fuse, Sequence
>>> decoder = Sequence([ByteFallback(), Fuse()])
>>> decoder.decode(["&amp;lt;0x48>", "&amp;lt;0x65>", "&amp;lt;0x6C>", "&amp;lt;0x6C>", "&amp;lt;0x6F>"])
'Hello'
```

## ByteLevel[[tokenizers.decoders.ByteLevel]]

#### tokenizers.decoders.ByteLevel[[tokenizers.decoders.ByteLevel]]

ByteLevel Decoder

This decoder is to be used in tandem with the
[ByteLevel](/docs/tokenizers/v0.23.1/en/api/pre-tokenizers#tokenizers.pre_tokenizers.ByteLevel) pre-tokenizer. It reverses the
byte-to-unicode mapping applied during pre-tokenization, converting the special
Unicode characters back into the original bytes to reconstruct the original string.

Example:

```python
>>> from tokenizers.decoders import ByteLevel
>>> decoder = ByteLevel()
>>> decoder.decode(["ĠHello", "Ġworld"])
' Hello world'
```

## CTC[[tokenizers.decoders.CTC]]

#### tokenizers.decoders.CTC[[tokenizers.decoders.CTC]]

CTC Decoder

Example:

```python
>>> from tokenizers.decoders import CTC
>>> decoder = CTC()
>>> decoder.decode(["h", "e", "e", "&amp;lt;pad>", "l", "l", "o", "|", "w", "o", "r", "l", "d"])
'hello world'
```

**Parameters:**

pad_token (`str`, *optional*, defaults to `&amp;lt;pad>`) : The pad token used by CTC to delimit a new token.

word_delimiter_token (`str`, *optional*, defaults to `|`) : The word delimiter token. It will be replaced by a &amp;lt;space>

cleanup (`bool`, *optional*, defaults to `True`) : Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation, and some abbreviated english forms.

## Fuse[[tokenizers.decoders.Fuse]]

#### tokenizers.decoders.Fuse[[tokenizers.decoders.Fuse]]

Fuse Decoder

Fuse simply concatenates every token into a single string without any separator.
This is typically the last step in a decoder chain when other decoders need to
operate on individual tokens before they are joined together.

Example:

```python
>>> from tokenizers.decoders import Fuse
>>> decoder = Fuse()
>>> decoder.decode(["Hello", ",", " ", "world", "!"])
'Hello, world!'
```

## Metaspace[[tokenizers.decoders.Metaspace]]

#### tokenizers.decoders.Metaspace[[tokenizers.decoders.Metaspace]]

Metaspace Decoder

Example:

```python
>>> from tokenizers.decoders import Metaspace
>>> decoder = Metaspace()
>>> decoder.decode(["▁Hello", "▁my", "▁friend"])
'Hello my friend'
```

**Parameters:**

replacement (`str`, *optional*, defaults to `▁`) : The replacement character. Must be exactly one character. By default we use the *▁* (U+2581) meta symbol (Same as in SentencePiece). 

prepend_scheme (`str`, *optional*, defaults to `"always"`) : Whether to add a space to the first word if there isn't already one. This lets us treat *hello* exactly like *say hello*. Choices: "always", "never", "first". First means the space is only added on the first token (relevant when special tokens are used or other pre_tokenizer are used).

## Replace[[tokenizers.decoders.Replace]]

#### tokenizers.decoders.Replace[[tokenizers.decoders.Replace]]

Replace Decoder

This decoder is to be used in tandem with the
[Replace](/docs/tokenizers/v0.23.1/en/api/normalizers#tokenizers.normalizers.Replace) normalizer or a similar replace operation.
It reverses a string replacement by substituting the replacement content back
with the original pattern.

Example:

```python
>>> from tokenizers.decoders import Replace
>>> decoder = Replace("▁", " ")
>>> decoder.decode(["▁Hello", "▁world"])
' Hello world'
```

**Parameters:**

pattern (`str` or `Regex`) : The pattern that was used as the replacement target during encoding. 

content (`str`) : The string to replace each match of the pattern with during decoding.

## Sequence[[tokenizers.decoders.Sequence]]

#### tokenizers.decoders.Sequence[[tokenizers.decoders.Sequence]]

Sequence Decoder

Chains multiple decoders together, applying them in order. Each decoder in the
sequence processes the output of the previous one, allowing complex decoding
pipelines to be built from simpler components.

Example:

```python
>>> from tokenizers.decoders import ByteFallback, Fuse, Metaspace, Sequence
>>> decoder = Sequence([ByteFallback(), Fuse(), Metaspace()])
>>> decoder.decode(["▁Hello", "▁world"])
'Hello world'
```

**Parameters:**

decoders (`List[Decoder]`) : The list of decoders to chain together.

## Strip[[tokenizers.decoders.Strip]]

#### tokenizers.decoders.Strip[[tokenizers.decoders.Strip]]

Strip Decoder

Strips a given number of occurrences of a character from the left and/or right
side of each token. This is useful for removing padding characters or special
prefix/suffix markers added during tokenization.

Example:

```python
>>> from tokenizers.decoders import Strip
>>> decoder = Strip(content="▁", left=1)
>>> decoder.decode(["▁Hello", "▁world"])
'Hello world'
```

**Parameters:**

content (`str`, defaults to `" "`) : The character to strip from each token. 

left (`int`, defaults to `0`) : The number of occurrences of `content` to remove from the left side of each token. 

right (`int`, defaults to `0`) : The number of occurrences of `content` to remove from the right side of each token.

## WordPiece[[tokenizers.decoders.WordPiece]]

#### tokenizers.decoders.WordPiece[[tokenizers.decoders.WordPiece]]

WordPiece Decoder

Example:

```python
>>> from tokenizers.decoders import WordPiece
>>> decoder = WordPiece()
>>> decoder.decode(["Hello", ",", "##world", "!"])
'Hello, world!'
```

**Parameters:**

prefix (`str`, *optional*, defaults to `##`) : The prefix to use for subwords that are not a beginning-of-word 

cleanup (`bool`, *optional*, defaults to `True`) : Whether to cleanup some tokenization artifacts. Mainly spaces before punctuation, and some abbreviated english forms.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Pre-tokenizers
https://huggingface.co/docs/tokenizers/v0.23.1/api/pre-tokenizers.md
