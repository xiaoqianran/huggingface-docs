<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 日志记录[[accelerate.logging.get_logger]]

参考[Troubleshooting guide](../basic_tutorials/troubleshooting#logging)或者下面的例子来学习 
如何使用 Accelerate 的记录器。 

####加速.logging.get_logger[[accelerate.logging.get_logger]]

```python
accelerate.logging.get_logger(name: str, log_level: str | None = None)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L135)

**参数：**

name (`str`) ：记录器的名称，例如`__file__`

log_level (`str`, *可选*) ：要使用的日志级别。如果不传递，则默认为 `ACCELERATE_LOG_LEVEL` 环境变量。如果两者均未设置，则保留记录器的现有级别，如果未设置级别，则从其父级继承。

返回可以处理多处理的 `name` 的 `logging.Logger`。

如果应该在所有进程上调用日志，则传递 `main_process_only=False` 如果应该在所有进程上调用日志
处理并按顺序，也通过`in_order=True`

示例：

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

## 库详细程度[[accelerate.logging.get_verbosity]]

使用 [logging.set_verbosity()](/docs/accelerate/v1.16.0/en/package_reference/logging#accelerate.logging.set_verbosity) 设置 Accelerate 内部记录器的默认级别：

```python
from accelerate.logging import set_verbosity

set_verbosity("warning")
```

通过 `log_level` 或 `ACCELERATE_LOG_LEVEL` 设置的每个记录器级别优先。

####加速.logging.get_verbosity[[accelerate.logging.get_verbosity]]

```python
accelerate.logging.get_verbosity()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L23)

以整数形式返回 Accelerate 库记录器的有效日志记录级别。

####加速.logging.set_verbosity[[accelerate.logging.set_verbosity]]

```python
accelerate.logging.set_verbosity(verbosity: int | str)
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L28)

**参数：**详细程度（`int` 或 `str`）：Python 日志记录级别，例如 `logging.WARNING`，或不区分大小写的级别名称，例如 `"warning"`。

设置 Accelerate 库记录器的日志记录级别。

这会影响 `accelerate` 命名空间中继承其级别的现有和未来记录器。记录仪带有
显式设置级别，包括通过 `get_logger` 的 `log_level` 参数或 `ACCELERATE_LOG_LEVEL`，保留该级别
水平。 Python 根记录器和 `accelerate` 命名空间之外的记录器不会被修改。

####加速.logging.set_verbosity_debug[[accelerate.logging.set_verbosity_debug]]

```python
accelerate.logging.set_verbosity_debug()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L45)

将 Accelerate 库的详细程度设置为 `DEBUG`。

####加速.logging.set_verbosity_info[[accelerate.logging.set_verbosity_info]]

```python
accelerate.logging.set_verbosity_info()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L50)

将 Accelerate 库的详细程度设置为 `INFO`。

####加速.logging.set_verbosity_warning[[accelerate.logging.set_verbosity_warning]]

```python
accelerate.logging.set_verbosity_warning()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L55)

将 Accelerate 库的详细程度设置为 `WARNING`。

####加速.logging.set_verbosity_error[[accelerate.logging.set_verbosity_error]]

```python
accelerate.logging.set_verbosity_error()
```

[Source](https://github.com/huggingface/accelerate/blob/v1.16.0/src/accelerate/logging.py#L60)

将 Accelerate 库的详细程度设置为 `ERROR`。

### 实验追踪器
https://huggingface.co/docs/accelerate/v1.16.0/package_reference/tracking.md