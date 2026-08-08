# Visualizer

## Annotation[[tokenizers.tools.Annotation]]

#### tokenizers.tools.Annotation[[tokenizers.tools.Annotation]]

[Source](https://github.com/huggingface/tokenizers/blob/v0.23.1/bindings/python/py_src/tokenizers/tools/visualizer.py#L16)

## EncodingVisualizer[[tokenizers.tools.EncodingVisualizer]]

#### tokenizers.tools.EncodingVisualizer[[tokenizers.tools.EncodingVisualizer]]

[Source](https://github.com/huggingface/tokenizers/blob/v0.23.1/bindings/python/py_src/tokenizers/tools/visualizer.py#L67)

Build an EncodingVisualizer

__call__tokenizers.tools.EncodingVisualizer.__call__https://github.com/huggingface/tokenizers/blob/v0.23.1/bindings/python/py_src/tokenizers/tools/visualizer.py#L110[{"name": "text", "val": ": str"}, {"name": "annotations", "val": ": typing.Optional[typing.List[typing.Any]] = None"}, {"name": "default_to_notebook", "val": ": typing.Optional[bool] = None"}]- **text** (`str`) --
  The text to tokenize

- **annotations** (`List[Annotation]`, *optional*) --
  An optional list of annotations of the text. The can either be an annotation class
  or anything else if you instantiated the visualizer with a converter function

- **default_to_notebook** (`bool`, *optional*, defaults to *False*) --
  If True, will render the html in a notebook. Otherwise returns an html string.0The HTML string if default_to_notebook is False, otherwise (default) returns None and
renders the HTML in the notebook

Build a visualization of the given text

**Parameters:**

tokenizer ([Tokenizer](/docs/tokenizers/v0.23.1/en/api/tokenizer#tokenizers.Tokenizer)) : A tokenizer instance 

default_to_notebook (`bool`) : Whether to render html output in a notebook by default 

annotation_converter (`Callable`, *optional*) : An optional (lambda) function that takes an annotation in any format and returns an Annotation object

**Returns:**

The HTML string if default_to_notebook is False, otherwise (default) returns None and
renders the HTML in the notebook

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Added Tokens
https://huggingface.co/docs/tokenizers/v0.23.1/api/added-tokens.md
