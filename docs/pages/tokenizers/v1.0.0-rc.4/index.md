# Tokenizers

## Docs

- [Tokenizers](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/index.md)
- [Node](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/node.md)
- [The tokenization pipeline](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/pipeline.md)
- [Training from memory](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/training_from_memory.md)
- [Installation](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/installation.md)
- [Components](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/components.md)
- [Quicktour](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/quicktour.md)
- [Pre-tokenizers](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/pre-tokenizers.md)
- [Tokenizer](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/tokenizer.md)
- [Visualizer](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/visualizer.md)
- [Trainers](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/trainers.md)
- [Models](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/models.md)
- [Encode Inputs](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/encode-inputs.md)
- [Decoders](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/decoders.md)
- [Encoding](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/encoding.md)
- [Normalizers](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/normalizers.md)
- [Added Tokens](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/added-tokens.md)
- [Post-processors](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/post-processors.md)
- [Input Sequences](https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/api/input-sequences.md)

### Tokenizers
https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/index.md

# Tokenizers

Not supported by the rc0 bindings yet. rc0 loads a `tokenizer.json` and encodes and
decodes with it — building a tokenizer from its components, editing one, saving one
and training are not exposed. They are coming soon, along with the other bindings.
See [`REQUIRED_FOR_V1.md`](https://github.com/huggingface/tokenizers/blob/main/REQUIRED_FOR_V1.md) for the full list.

Fast State-of-the-art tokenizers, optimized for both research and
production

[🤗 Tokenizers](https://github.com/huggingface/tokenizers) provides an
implementation of today's most used tokenizers, with a focus on
performance and versatility. These tokenizers are also used in [🤗 Transformers](https://github.com/huggingface/transformers).

# Main features:

- Train new vocabularies and tokenize, using today's most used tokenizers.
- Extremely fast (both training and tokenization), thanks to the Rust implementation. Takes less than 20 seconds to tokenize a GB of text on a server's CPU.
- Easy to use, but also extremely versatile.
- Designed for both research and production.
- Full alignment tracking. Even with destructive normalization, it's always possible to get the part of the original sentence that corresponds to any token.
- Does all the pre-processing: Truncation, Padding, add the special tokens your model needs.

### Node
https://huggingface.co/docs/tokenizers/v1.0.0-rc.4/node.md
