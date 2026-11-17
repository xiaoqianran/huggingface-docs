# Logging[[accelerate.logging.get_logger]]

Refer to the [Troubleshooting guide](../basic_tutorials/troubleshooting#logging) or to the example below to learn 
how to use Accelerate's logger. 

#### accelerate.logging.get_logger[[accelerate.logging.get_logger]]

```python
accelerate.logging.get_logger(name: str, log_level: str | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L135)

**Parameters:**

name (`str`) : The name for the logger, such as `__file__`

log_level (`str`, *optional*) : The log level to use. If not passed, defaults to the `ACCELERATE_LOG_LEVEL` environment variable. If neither is set, the logger's existing level is preserved, inheriting from its parent if no level has been set.

Returns a `logging.Logger` for `name` that can handle multiprocessing.

If a log should be called on all processes, pass `main_process_only=False` If a log should be called on all
processes and in order, also pass `in_order=True`

Example:

```python
>>> from accelerate.logging import get_logger
>>> from accelerate import Accelerator

>>> logger = get_logger(__name__)

>>> accelerator = Accelerator()
>>> logger.info("My log", main_process_only=False)
>>> logger.debug("My log", main_process_only=True)

>>> logger = get_logger(__name__, log_level="DEBUG")
>>> logger.info("My log")
>>> logger.debug("My second log")

>>> array = ["a", "b", "c", "d"]
>>> letter_at_rank = array[accelerator.process_index]
>>> logger.info(letter_at_rank, in_order=True)
```

## Library verbosity[[accelerate.logging.get_verbosity]]

Use [logging.set_verbosity()](/docs/accelerate/v1.16.0/en/package_reference/logging#accelerate.logging.set_verbosity) to set the default level for Accelerate's internal loggers:

```python
from accelerate.logging import set_verbosity

set_verbosity("warning")
```

Per-logger levels set through `log_level` or `ACCELERATE_LOG_LEVEL` take precedence.

#### accelerate.logging.get_verbosity[[accelerate.logging.get_verbosity]]

```python
accelerate.logging.get_verbosity()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L23)

Return the effective logging level of the Accelerate library logger as an integer.

#### accelerate.logging.set_verbosity[[accelerate.logging.set_verbosity]]

```python
accelerate.logging.set_verbosity(verbosity: int | str)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L28)

**Parameters:**

verbosity (`int` or `str`) : A Python logging level, such as `logging.WARNING`, or a case-insensitive level name, such as `"warning"`.

Set the logging level of the Accelerate library logger.

This affects existing and future loggers in the `accelerate` namespace that inherit their level. Loggers with a
level set explicitly, including through `get_logger`'s `log_level` argument or `ACCELERATE_LOG_LEVEL`, retain that
level. The Python root logger and loggers outside the `accelerate` namespace are not modified.

#### accelerate.logging.set_verbosity_debug[[accelerate.logging.set_verbosity_debug]]

```python
accelerate.logging.set_verbosity_debug()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L45)

Set the Accelerate library's verbosity to `DEBUG`.

#### accelerate.logging.set_verbosity_info[[accelerate.logging.set_verbosity_info]]

```python
accelerate.logging.set_verbosity_info()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L50)

Set the Accelerate library's verbosity to `INFO`.

#### accelerate.logging.set_verbosity_warning[[accelerate.logging.set_verbosity_warning]]

```python
accelerate.logging.set_verbosity_warning()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L55)

Set the Accelerate library's verbosity to `WARNING`.

#### accelerate.logging.set_verbosity_error[[accelerate.logging.set_verbosity_error]]

```python
accelerate.logging.set_verbosity_error()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L60)

Set the Accelerate library's verbosity to `ERROR`.

### Experiment Trackers
https://huggingface.co/docs/accelerate/v1.16.0/package_reference/tracking.md
