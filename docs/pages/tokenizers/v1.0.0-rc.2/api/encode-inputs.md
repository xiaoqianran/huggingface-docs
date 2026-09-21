# Encode Inputs

Not supported by the rc0 bindings yet. rc0 loads a `tokenizer.json` and encodes and
decodes with it — building a tokenizer from its components, editing one, saving one
and training are not exposed. They are coming soon, along with the other bindings.
See [`REQUIRED_FOR_V1.md`](https://github.com/huggingface/tokenizers/blob/main/REQUIRED_FOR_V1.md) for the full list.

These types represent all the different kinds of input that a `Tokenizer` accepts
when using `encode_batch()`.

## TextEncodeInput[[[[tokenizers.TextEncodeInput]]]]

tokenizers.TextEncodeInput

Represents a textual input for encoding. Can be either:
- A single sequence: [TextInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.TextInputSequence)
- A pair of sequences:
  - A Tuple of [TextInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.TextInputSequence)
  - Or a List of [TextInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.TextInputSequence) of size 2

alias of `Union[str, Tuple[str, str], List[str]]`.

## PreTokenizedEncodeInput[[[[tokenizers.PreTokenizedEncodeInput]]]]

tokenizers.PreTokenizedEncodeInput

Represents a pre-tokenized input for encoding. Can be either:
- A single sequence: [PreTokenizedInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.PreTokenizedInputSequence)
- A pair of sequences:
  - A Tuple of [PreTokenizedInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.PreTokenizedInputSequence)
  - Or a List of [PreTokenizedInputSequence](/docs/tokenizers/api/input-sequences#tokenizers.PreTokenizedInputSequence) of size 2

alias of `Union[List[str], Tuple[str], Tuple[Union[List[str], Tuple[str]], Union[List[str], Tuple[str]]], List[Union[List[str], Tuple[str]]]]`.

## EncodeInput[[[[tokenizers.EncodeInput]]]]

tokenizers.EncodeInput

Represents all the possible types of input for encoding. Can be:
- When `is_pretokenized=False`: [TextEncodeInput](#tokenizers.TextEncodeInput)
- When `is_pretokenized=True`: [PreTokenizedEncodeInput](#tokenizers.PreTokenizedEncodeInput)

alias of `Union[str, Tuple[str, str], List[str], Tuple[str], Tuple[Union[List[str], Tuple[str]], Union[List[str], Tuple[str]]], List[Union[List[str], Tuple[str]]]]`.

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

### Normalizers
https://huggingface.co/docs/tokenizers/v1.0.0-rc.2/api/normalizers.md
