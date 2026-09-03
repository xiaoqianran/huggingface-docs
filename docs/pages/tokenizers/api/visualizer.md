# Visualizer

## Annotation[[tokenizers.tools.Annotation]]

#### tokenizers.tools.Annotation[[tokenizers.tools.Annotation]]

```python
tokenizers.tools.Annotation(start: int, end: int, label: str)
```

[Source](https://github.com/huggingface/tokenizers/blob/v0.23.2/bindings/python/py_src/tokenizers/tools/visualizer.py#L16)

## EncodingVisualizer[[tokenizers.tools.EncodingVisualizer]]

#### tokenizers.tools.EncodingVisualizer[[tokenizers.tools.EncodingVisualizer]]

```python
tokenizers.tools.EncodingVisualizer(tokenizer: Tokenizer, default_to_notebook: bool = True, annotation_converter: typing.Optional[typing.Callable[[typing.Any], tokenizers.tools.visualizer.Annotation]] = None)
```

[Source](https://github.com/huggingface/tokenizers/blob/v0.23.2/bindings/python/py_src/tokenizers/tools/visualizer.py#L67)

**Parameters:**

tokenizer ([Tokenizer](/docs/tokenizers/v0.23.2/en/api/tokenizer#tokenizers.Tokenizer)) : A tokenizer instance 

default_to_notebook (`bool`) : Whether to render html output in a notebook by default 

annotation_converter (`Callable`, *optional*) : An optional (lambda) function that takes an annotation in any format and returns an Annotation object

Build an EncodingVisualizer

#### __call__[[tokenizers.tools.EncodingVisualizer.__call__]]

```python
__call__(text: str, annotations: typing.Optional[typing.List[typing.Any]] = None, default_to_notebook: typing.Optional[bool] = None)
```

[Source](https://github.com/huggingface/tokenizers/blob/v0.23.2/bindings/python/py_src/tokenizers/tools/visualizer.py#L110)

**Parameters:**

text (`str`) : The text to tokenize 

annotations (`List[Annotation]`, *optional*) : An optional list of annotations of the text. The can either be an annotation class or anything else if you instantiated the visualizer with a converter function 

default_to_notebook (`bool`, *optional*, defaults to *False*) : If True, will render the html in a notebook. Otherwise returns an html string.

**Returns:**

The HTML string if default_to_notebook is False, otherwise (default) returns None and
renders the HTML in the notebook

Build a visualization of the given text

The Rust API Reference is available directly on the [Docs.rs](https://docs.rs/tokenizers/latest/tokenizers/) website.

The node API has not been documented yet.

### Added Tokens
https://huggingface.co/docs/tokenizers/v0.23.2/api/added-tokens.md
