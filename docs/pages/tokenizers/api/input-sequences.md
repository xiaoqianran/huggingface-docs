# Input Sequences

These types represent all the different kinds of sequence that can be used as input of a Tokenizer.
Globally, any sequence can be either a string or a list of strings, according to the operating
mode of the tokenizer: `raw text` vs `pre-tokenized`.

## TextInputSequence[[tokenizers.TextInputSequence]]

tokenizers.TextInputSequence

A `str` that represents an input sequence

## PreTokenizedInputSequence[[tokenizers.PreTokenizedInputSequence]]

tokenizers.PreTokenizedInputSequence

A pre-tokenized input sequence. Can be one of:
- A `List` of `str`
- A `Tuple` of `str`

alias of `Union[List[str], Tuple[str]]`.

## InputSequence[[tokenizers.InputSequence]]

tokenizers.InputSequence

Represents all the possible types of input sequences for encoding. Can be:
- When `is_pretokenized=False`: [TextInputSequence](#tokenizers.TextInputSequence)
- When `is_pretokenized=True`: [PreTokenizedInputSequence](#tokenizers.PreTokenizedInputSequence)

alias of `Union[str, List[str], Tuple[str]]`.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Decoders
https://huggingface.co/docs/tokenizers/v0.23.1/api/decoders.md
