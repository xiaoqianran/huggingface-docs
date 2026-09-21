# Post-processors

**TODO -- coming soon.** Reference documentation on this page is not generated yet: the
rc0 bindings do not expose all of these APIs. It comes back with them, along with the
other bindings. See [`REQUIRED_FOR_V1.md`](https://github.com/huggingface/tokenizers/blob/main/REQUIRED_FOR_V1.md) for the full list.

Every post-processor is a template. `BertProcessing`, `RobertaProcessing`, `ByteLevel` and a
`Sequence` of post-processors were spellings of one, so a `tokenizer.json` using any of them is
rewritten into a `TemplateProcessing` when it is converted, and there is no separate class for
them.

## PostProcessor[[tokenizers.processors.PostProcessor]]

tokenizers.processors.PostProcessor

## TemplateProcessing[[tokenizers.processors.TemplateProcessing]]

tokenizers.processors.TemplateProcessing

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

### Trainers
https://huggingface.co/docs/tokenizers/v1.0.0-rc.2/api/trainers.md
