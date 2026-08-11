<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 日志记录[[accelerate.logging.get_logger]]

参考[Troubleshooting guide](../usage_guides/troubleshooting#logging)或下面的例子来了解 
如何使用 Accelerate 的记录器。 

####加速.logging.get_logger[[accelerate.logging.get_logger]]

[Source](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/logging.py#L93)

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

**参数：**

name (`str`) : 记录器的名称，例如`__file__`

log_level (`str`, *可选*) ：要使用的日志级别。如果不传递，则默认为 `LOG_LEVEL` 环境变量，如果不传递则默认为 `INFO`

### 完全分片数据并行实用程序
https://huggingface.co/docs/accelerate/v1.14.0/package_reference/fsdp.md