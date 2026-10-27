# Pipeline states

## PipelineState[[diffusers.modular_pipelines.PipelineState]]

#### diffusers.modular_pipelines.PipelineState[[diffusers.modular_pipelines.PipelineState]]

```python
diffusers.modular_pipelines.PipelineState(values: dict = <factory>, kwargs_mapping: dict = <factory>)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L226)

`PipelineState` stores the state of a pipeline. It is used to pass data between pipeline blocks.

#### get[[diffusers.modular_pipelines.PipelineState.get]]

```python
get(keys: str | list[str], default: Any = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L251)

**Parameters:**

keys (str | list[str]) : Key or list of keys for the values

default (Any) : The default value to return if not found

**Returns:** Any | dict[str, Any]

Single value if keys is str, dictionary of values if keys is list

Get one or multiple values from the pipeline state.

#### get_by_kwargs[[diffusers.modular_pipelines.PipelineState.get_by_kwargs]]

```python
get_by_kwargs(kwargs_type: str)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L266)

**Parameters:**

kwargs_type (str) : The kwargs_type to filter by

**Returns:** dict[str, Any]

Dictionary of values with matching kwargs_type

Get all values with matching kwargs_type.

#### set[[diffusers.modular_pipelines.PipelineState.set]]

```python
set(key: str, value: Any, kwargs_type: str = None)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L234)

**Parameters:**

key (str) : The key for the value

value (Any) : The value to store

kwargs_type (str) : The kwargs_type with which the value is associated

Add a value to the pipeline state.

#### to_dict[[diffusers.modular_pipelines.PipelineState.to_dict]]

```python
to_dict()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L279)

Convert PipelineState to a dictionary.

## BlockState[[diffusers.modular_pipelines.BlockState]]

#### diffusers.modular_pipelines.BlockState[[diffusers.modular_pipelines.BlockState]]

```python
diffusers.modular_pipelines.BlockState(**kwargs)
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L316)

Container for block state data with attribute access and formatted representation.

#### as_dict[[diffusers.modular_pipelines.BlockState.as_dict]]

```python
as_dict()
```

[Source](https://github.com/huggingface/diffusers/blob/v0.42.1/src/diffusers/modular_pipelines/modular_pipeline.py#L333)

**Returns:** dict[str, Any]

Dictionary containing all attributes of the BlockState

Convert BlockState to a dictionary.

### DDPMScheduler
https://huggingface.co/docs/diffusers/v0.42.1/api/schedulers/ddpm.md
