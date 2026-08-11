<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 严格的数据类

`huggingface_hub` 包提供了一个实用程序来创建**严格的数据类**。这些是 Python 标准 `dataclass` 的增强版本，具有附加验证功能。严格的数据类确保字段在初始化和分配期间都得到验证，这使得它们非常适合数据完整性至关重要的场景。

## 概述

严格的数据类是使用 `@strict` 装饰器创建的。它们通过以下方式扩展常规数据类的功能：

- 根据类型提示验证字段类型
- 支持自定义验证器进行额外检查
- 可以选择在构造函数中允许任意关键字参数
- 在初始化和分配期间验证字段

## 好处

- **数据完整性**：确保字段始终包含有效数据
- **易于使用**：与 Python 的 `dataclass` 模块无缝集成
- **灵活性**：支持复杂验证逻辑的自定义验证器
- **轻量级**：不需要额外的依赖项，例如 Pydantic、attrs 或类似的库

## 用法

### 基本示例

```python
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict, as_validated_field

# Custom validator to ensure a value is positive
@as_validated_field
def positive_int(value: int):
    if not value > 0:
        raise ValueError(f"Value must be positive, got {value}")

@strict
@dataclass
class Config:
    model_type: str
    hidden_size: int = positive_int(default=16)
    vocab_size: int = 32  # Default value

    # Methods named `validate_xxx` are treated as class-wise validators
    def validate_big_enough_vocab(self):
        if self.vocab_size < self.hidden_size:
            raise ValueError(f"vocab_size ({self.vocab_size}) must be greater than hidden_size ({self.hidden_size})")
```

字段在初始化期间进行验证：

```python
config = Config(model_type="bert", hidden_size=24)   # Valid
config = Config(model_type="bert", hidden_size=-1)   # Raises StrictDataclassFieldValidationError
```初始化期间也会验证字段之间的一致性（按类验证）：

```python
# `vocab_size` too small compared to `hidden_size`
config = Config(model_type="bert", hidden_size=32, vocab_size=16)   # Raises StrictDataclassClassValidationError
```

字段也会在分配期间进行验证：

```python
config.hidden_size = 512  # Valid
config.hidden_size = -1   # Raises StrictDataclassFieldValidationError
```

要在分配后重新运行类范围的验证，您必须显式调用 `.validate`：

```python
config.validate()  # Runs all class validators
```

### 自定义验证器

您可以使用 `validated_field` 将多个自定义验证器附加到字段。验证器是一个可调用的函数，它接受单个参数，并在该值无效时引发异常。

```python
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict, validated_field

def multiple_of_64(value: int):
    if value % 64 != 0:
        raise ValueError(f"Value must be a multiple of 64, got {value}")

@strict
@dataclass
class Config:
    hidden_size: int = validated_field(validator=[positive_int, multiple_of_64])
```

在此示例中，两个验证器都应用于 `hidden_size` 字段。

### 附加关键字参数

默认情况下，严格数据类仅接受类中定义的字段。您可以通过在 `@strict` 装饰器中设置 `accept_kwargs=True` 来允许其他关键字参数。

```python
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict(accept_kwargs=True)
@dataclass
class ConfigWithKwargs:
    model_type: str
    vocab_size: int = 16

config = ConfigWithKwargs(model_type="bert", vocab_size=30000, extra_field="extra_value")
print(config)  # ConfigWithKwargs(model_type='bert', vocab_size=30000, *extra_field='extra_value')
```

其他关键字参数出现在数据类的字符串表示形式中，但以 `*` 为前缀，以突出显示它们未经验证。

### 与类型提示集成

严格的数据类尊重类型提示并自动验证它们。例如：

```python
from typing import List
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict
@dataclass
class Config:
    layers: List[int]

config = Config(layers=[64, 128])  # Valid
config = Config(layers="not_a_list")  # Raises StrictDataclassFieldValidationError
```

支持的类型包括：
- 任意
- 联盟
- 可选
- 字面意思
- 列表
- 字典
- 元组
- 设置

以及这些类型的任意组合。如果您需要更复杂的类型验证，可以通过自定义验证器来完成。### 类验证器

名为 `validate_xxx` 的方法被视为类验证器。这些方法必须仅采用 `self` 作为参数。类验证器在初始化期间运行一次，就在`__post_init__`之后。您可以根据需要定义任意多个 - 它们将按照出现的顺序依次执行。

请注意，初始化后更新字段时，类验证器不会自动重新运行。要手动重新验证对象，您需要调用`obj.validate()`。

```py
from dataclasses import dataclass
from huggingface_hub.dataclasses import strict

@strict
@dataclass
class Config:
    foo: str
    foo_length: int
    upper_case: bool = False

    def validate_foo_length(self):
        if len(self.foo) != self.foo_length:
            raise ValueError(f"foo must be {self.foo_length} characters long, got {len(self.foo)}")

    def validate_foo_casing(self):
        if self.upper_case and self.foo.upper() != self.foo:
            raise ValueError(f"foo must be uppercase, got {self.foo}")

config = Config(foo="bar", foo_length=3) # ok

config.upper_case = True
config.validate() # Raises StrictDataclassClassValidationError

Config(foo="abcd", foo_length=3) # Raises StrictDataclassFieldValidationError
Config(foo="Bar", foo_length=3, upper_case=True) # Raises StrictDataclassFieldValidationError
```

> [!警告]
> 方法`.validate()`是严格数据类上的保留名称。
> 为了防止意外行为，如果您的类已经定义了一个 `StrictDataclassDefinitionError` 错误，则会引发该错误。

## API 参考

### `@strict`[[huggingface_hub.dataclasses.strict]]

`@strict` 装饰器通过严格的验证增强了数据类。

#### Huggingface_hub.dataclasses.strict[[huggingface_hub.dataclasses.strict]]

```python
huggingface_hub.dataclasses.strict(accept_kwargs: bool = False)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/dataclasses.py#L56)

**参数：**

cls ：要转换为严格数据类的类。

Accept_kwargs (`bool`, *可选*) ：如果为 True，则允许 `__init__` 中的任意关键字参数。默认为 False。

**退货：**

增强的数据类，对字段分配进行严格验证。

装饰器向数据类添加严格的验证。该装饰器必须在`@dataclass`之上使用，以确保IDE和静态类型工具
将该类识别为数据类。

可以带或不带参数使用：
- `@strict`
- `@strict(accept_kwargs=True)`

示例：
```py
>>> from dataclasses import dataclass
>>> from huggingface_hub.dataclasses import as_validated_field, strict, validated_field

>>> @as_validated_field
>>> def positive_int(value: int):
...     if not value >= 0:
...         raise ValueError(f"Value must be positive, got {value}")

>>> @strict(accept_kwargs=True)
... @dataclass
... class User:
...     name: str
...     age: int = positive_int(default=10)

# Initialize
>>> User(name="John")
User(name='John', age=10)

# Extra kwargs are accepted
>>> User(name="John", age=30, lastname="Doe")
User(name='John', age=30, *lastname='Doe')

# Invalid type => raises
>>> User(name="John", age="30")
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
    TypeError: Field 'age' expected int, got str (value: '30')

# Invalid value => raises
>>> User(name="John", age=-1)
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
    ValueError: Value must be positive, got -1
```

### `validate_typed_dict`[[huggingface_hub.dataclasses.validate_typed_dict]]

验证字典是否符合 `TypedDict` 类中定义的类型的方法。

这相当于数据类验证，但适用于 `TypedDict`s。由于类型化字典永远不会被实例化（仅由静态类型检查器使用），因此必须手动调用验证步骤。

#### Huggingface_hub.dataclasses.validate_typed_dict[[huggingface_hub.dataclasses.validate_typed_dict]]

```python
huggingface_hub.dataclasses.validate_typed_dict(schema: type, data: dict)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/dataclasses.py#L286)

**参数：**

schema (`type[TypedDictType]`) ：定义预期结构和类型的 TypedDict 类。

data (`dict`) ：要验证的字典。

**加薪：** ``StrictDataclassFieldValidationError``

- ``StrictDataclassFieldValidationError`` -- 
  如果字典中的任何字段不符合预期类型。

验证字典是否符合 TypedDict 类中定义的类型。

在底层，类型化的字典被转换为严格的数据类，并使用 `@strict` 装饰器进行验证。

示例：
```py
>>> from typing import Annotated, TypedDict
>>> from huggingface_hub.dataclasses import validate_typed_dict

>>> def positive_int(value: int):
...     if not value >= 0:
...         raise ValueError(f"Value must be positive, got {value}")

>>> class User(TypedDict):
...     name: str
...     age: Annotated[int, positive_int]

>>> # Valid data
>>> validate_typed_dict(User, {"name": "John", "age": 30})

>>> # Invalid type for age
>>> validate_typed_dict(User, {"name": "John", "age": "30"})
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
    TypeError: Field 'age' expected int, got str (value: '30')

>>> # Invalid value for age
>>> validate_typed_dict(User, {"name": "John", "age": -1})
huggingface_hub.errors.StrictDataclassFieldValidationError: Validation error for field 'age':
    ValueError: Value must be positive, got -1
```

### `as_validated_field`[[huggingface_hub.dataclasses.as_validated_field]]装饰器创建一个`validated_field`。建议用于具有单个验证器的字段以避免样板代码。

#### Huggingface_hub.dataclasses.as_validated_field[[huggingface_hub.dataclasses.as_validated_field]]

```python
huggingface_hub.dataclasses.as_validated_field(validator: Callable)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/dataclasses.py#L426)

**参数：**

validator (`Callable`) ：一种将值作为输入并在该值无效时引发 ValueError/TypeError 的方法。

将验证器函数装饰为`validated_field`（即具有自定义验证器的数据类字段）。

### `validated_field`[[huggingface_hub.dataclasses.validated_field]]

创建具有自定义验证的数据类字段。

#### Huggingface_hub.dataclasses.validated_field[[huggingface_hub.dataclasses.validated_field]]

```python
huggingface_hub.dataclasses.validated_field(validator: list[collections.abc.Callable[[typing.Any], None]] | collections.abc.Callable[[typing.Any], None], default: typing.Any = <dataclasses._MISSING_TYPE object at 0x7f8e7d51d090>, default_factory: typing.Any = <dataclasses._MISSING_TYPE object at 0x7f8e7d51d090>, init: bool = True, repr: bool = True, hash: bool | None = None, compare: bool = True, metadata: dict | None = None, **kwargs: typing.Any)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/dataclasses.py#L383)

**参数：**

验证器（`Callable` 或 `list[Callable]`）：一种将值作为输入并在该值无效时引发 ValueError/TypeError 的方法。可以是应用多个检查的验证器列表。

- ****kwargs** ：传递给`dataclasses.field()`的附加参数。

**退货：**

元数据中附加了验证器的字段

使用自定义验证器创建数据类字段。

对一个字段应用多项检查很有用。如果只应用一条规则，请查看 `as_validated_field` 装饰器。

### 错误[[huggingface_hub.errors.StrictDataclassError]]#### Huggingface_hub.errors.StrictDataclassError[[huggingface_hub.errors.StrictDataclassError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/errors.py#L528)

严格数据类的基本异常。

#### Huggingface_hub.errors.StrictDataclassDefinitionError[[huggingface_hub.errors.StrictDataclassDefinitionError]]

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/errors.py#L532)

当严格数据类定义不正确时抛出异常。

#### Huggingface_hub.errors.StrictDataclassFieldValidationError[[huggingface_hub.errors.StrictDataclassFieldValidationError]]

```python
huggingface_hub.errors.StrictDataclassFieldValidationError(field: str, cause: Exception)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/errors.py#L536)

当严格数据类对给定字段的验证失败时抛出异常。

## 为什么不使用`pydantic`？ （或`attrs`？或`marshmallow_dataclass`？）- 请参阅 https://github.com/huggingface/transformers/issues/36329 中有关添加 Pydantic 作为依赖项的讨论。这将是一个沉重的补充，需要仔细的逻辑来支持 v1 和 v2。
- 我们不需要 Pydantic 的大部分功能，尤其是那些与自动转换、jsonschema、序列化、别名等相关的功能。
- 我们不需要从字典实例化类的能力。
- 我们不想改变数据。在`@strict`中，“验证”的意思是“检查值是否有效”。在 Pydantic 中，“验证”意味着“转换一个值，可能会改变它，然后检查它是否有效”。
- 我们不需要极快的验证。 `@strict` 不适用于性能至关重要的重负载。常见用例涉及验证模型配置（执行一次，与运行模型相比可以忽略不计）。这使我们能够保持最少的代码。

### TensorBoard 记录器
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/tensorboard.md