# Table Classes

Each `Dataset` object is backed by a PyArrow Table.
A Table can be loaded from either the disk (memory mapped) or in memory.
Several Table types are available, and they all inherit from [table.Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table).

## Table[[datasets.table.Table]]

#### datasets.table.Table[[datasets.table.Table]]

```python
datasets.table.Table(table: Table)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L210)

Wraps a pyarrow Table by using composition.
This is the base class for `InMemoryTable`, `MemoryMappedTable` and `ConcatenationTable`.

It implements all the basic attributes/methods of the pyarrow Table class except
the Table transforms: `slice, filter, flatten, combine_chunks, cast, add_column,
append_column, remove_column, set_column, rename_columns` and `drop`.

The implementation of these methods differs for the subclasses.

#### validate[[datasets.table.Table.validate]]

```python
validate(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L235)

**Parameters:**

full (`bool`, defaults to `False`) : If `True`, run expensive checks, otherwise cheap checks only.

**Raises:** ``pa.lib.ArrowInvalid``

- ``pa.lib.ArrowInvalid`` -- if validation fails

Perform validation checks.  An exception is raised if validation fails.

By default only cheap validation checks are run.  Pass `full=True`
for thorough validation checks (potentially `O(n)`).

#### equals[[datasets.table.Table.equals]]

```python
equals(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L251)

**Parameters:**

other ([Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)) : Table to compare against.

check_metadata `bool`, defaults to `False`) : Whether schema metadata equality should be checked as well.

**Returns:** `bool`

Check if contents of two tables are equal.

#### to_batches[[datasets.table.Table.to_batches]]

```python
to_batches(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L268)

**Parameters:**

max_chunksize (`int`, defaults to `None`) : Maximum size for `RecordBatch` chunks. Individual chunks may be smaller depending on the chunk layout of individual columns.

**Returns:**

`List[pyarrow.RecordBatch]`

Convert Table to list of (contiguous) `RecordBatch` objects.

#### to_pydict[[datasets.table.Table.to_pydict]]

```python
to_pydict(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L282)

**Returns:** `dict`

Convert the Table to a `dict` or `OrderedDict`.

#### to_pandas[[datasets.table.Table.to_pandas]]

```python
to_pandas(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L300)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : Arrow MemoryPool to use for allocations. Uses the default memory pool is not passed.

strings_to_categorical (`bool`, defaults to `False`) : Encode string (UTF8) and binary types to `pandas.Categorical`.

categories (`list`, defaults to `empty`) : List of fields that should be returned as `pandas.Categorical`. Only applies to table-like data structures.

zero_copy_only (`bool`, defaults to `False`) : Raise an `ArrowException` if this function call would require copying the underlying data.

integer_object_nulls (`bool`, defaults to `False`) : Cast integers with nulls to objects.

date_as_object (`bool`, defaults to `True`) : Cast dates to objects. If `False`, convert to `datetime64[ns]` dtype.

timestamp_as_object (`bool`, defaults to `False`) : Cast non-nanosecond timestamps (`np.datetime64`) to objects. This is useful if you have timestamps that don't fit in the normal date range of nanosecond timestamps (1678 CE-2262 CE). If `False`, all timestamps are converted to `datetime64[ns]` dtype.

use_threads (`bool`, defaults to `True`) : Whether to parallelize the conversion using multiple threads.

deduplicate_objects (`bool`, defaults to `False`) : Do not create multiple copies Python objects when created, to save on memory use. Conversion will be slower.

ignore_metadata (`bool`, defaults to `False`) : If `True`, do not use the 'pandas' metadata to reconstruct the DataFrame index, if present.

safe (`bool`, defaults to `True`) : For certain data types, a cast is needed in order to store the data in a pandas DataFrame or Series (e.g. timestamps are always stored as nanoseconds in pandas). This option controls whether it is a safe cast or not.

split_blocks (`bool`, defaults to `False`) : If `True`, generate one internal "block" for each column when creating a pandas.DataFrame from a `RecordBatch` or `Table`. While this can temporarily reduce memory note that various pandas operations can trigger "consolidation" which may balloon memory use.

self_destruct (`bool`, defaults to `False`) : EXPERIMENTAL: If `True`, attempt to deallocate the originating Arrow memory while converting the Arrow object to pandas. If you use the object after calling `to_pandas` with this option it will crash your program.

types_mapper (`function`, defaults to `None`) : A function mapping a pyarrow DataType to a pandas `ExtensionDtype`. This can be used to override the default pandas type for conversion of built-in pyarrow types or in absence of `pandas_metadata` in the Table schema. The function receives a pyarrow DataType and is expected to return a pandas `ExtensionDtype` or `None` if the default conversion should be used for that type. If you have a dictionary mapping, you can pass `dict.get` as function.

**Returns:** `pandas.Series` or `pandas.DataFrame`

`pandas.Series` or `pandas.DataFrame` depending on type of object

Convert to a pandas-compatible NumPy array or DataFrame, as appropriate.

#### to_string[[datasets.table.Table.to_string]]

```python
to_string(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L362)

#### field[[datasets.table.Table.field]]

```python
field(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L381)

**Parameters:**

i (`Union[int, str]`) : The index or name of the field to retrieve.

**Returns:**

`pyarrow.Field`

Select a schema field by its column name or numeric index.

#### column[[datasets.table.Table.column]]

```python
column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L394)

**Parameters:**

i (`Union[int, str]`) : The index or name of the column to retrieve.

**Returns:**

`pyarrow.ChunkedArray`

Select a column by its column name, or numeric index.

#### itercolumns[[datasets.table.Table.itercolumns]]

```python
itercolumns(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L407)

**Yields:**

`pyarrow.ChunkedArray`

Iterator over all columns in their numerical order.

#### schema[[datasets.table.Table.schema]]

```python
schema()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L416)

**Returns:**

`pyarrow.Schema`

Schema of the table and its columns.

#### columns[[datasets.table.Table.columns]]

```python
columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L426)

**Returns:**

`List[pa.ChunkedArray]`

List of all columns in numerical order.

#### num_columns[[datasets.table.Table.num_columns]]

```python
num_columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L436)

**Returns:**

int

Number of columns in this table.

#### num_rows[[datasets.table.Table.num_rows]]

```python
num_rows()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L446)

**Returns:**

int

Number of rows in this table.

Due to the definition of a table, all columns have the same number of
rows.

#### shape[[datasets.table.Table.shape]]

```python
shape()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L459)

**Returns:** `(int, int)`

Number of rows and number of columns.

Dimensions of the table: (#rows, #columns).

#### nbytes[[datasets.table.Table.nbytes]]

```python
nbytes()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L469)

Total number of bytes consumed by the elements of the table.

## InMemoryTable[[datasets.table.InMemoryTable]]

#### datasets.table.InMemoryTable[[datasets.table.InMemoryTable]]

```python
datasets.table.InMemoryTable(table: Table)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L695)

The table is said in-memory when it is loaded into the user's RAM.

Pickling it does copy all the data using memory.
Its implementation is simple and uses the underlying pyarrow Table methods directly.

This is different from the `MemoryMapped` table, for which pickling doesn't copy all the
data in memory. For a `MemoryMapped`, unpickling instead reloads the table from the disk.

`InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
data bigger than memory or when you want the memory footprint of your application to
stay low.

#### validate[[datasets.table.InMemoryTable.validate]]

```python
validate(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L235)

**Parameters:**

full (`bool`, defaults to `False`) : If `True`, run expensive checks, otherwise cheap checks only.

**Raises:** ``pa.lib.ArrowInvalid``

- ``pa.lib.ArrowInvalid`` -- if validation fails

Perform validation checks.  An exception is raised if validation fails.

By default only cheap validation checks are run.  Pass `full=True`
for thorough validation checks (potentially `O(n)`).

#### equals[[datasets.table.InMemoryTable.equals]]

```python
equals(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L251)

**Parameters:**

other ([Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)) : Table to compare against.

check_metadata `bool`, defaults to `False`) : Whether schema metadata equality should be checked as well.

**Returns:** `bool`

Check if contents of two tables are equal.

#### to_batches[[datasets.table.InMemoryTable.to_batches]]

```python
to_batches(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L268)

**Parameters:**

max_chunksize (`int`, defaults to `None`) : Maximum size for `RecordBatch` chunks. Individual chunks may be smaller depending on the chunk layout of individual columns.

**Returns:**

`List[pyarrow.RecordBatch]`

Convert Table to list of (contiguous) `RecordBatch` objects.

#### to_pydict[[datasets.table.InMemoryTable.to_pydict]]

```python
to_pydict(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L282)

**Returns:** `dict`

Convert the Table to a `dict` or `OrderedDict`.

#### to_pandas[[datasets.table.InMemoryTable.to_pandas]]

```python
to_pandas(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L300)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : Arrow MemoryPool to use for allocations. Uses the default memory pool is not passed.

strings_to_categorical (`bool`, defaults to `False`) : Encode string (UTF8) and binary types to `pandas.Categorical`.

categories (`list`, defaults to `empty`) : List of fields that should be returned as `pandas.Categorical`. Only applies to table-like data structures.

zero_copy_only (`bool`, defaults to `False`) : Raise an `ArrowException` if this function call would require copying the underlying data.

integer_object_nulls (`bool`, defaults to `False`) : Cast integers with nulls to objects.

date_as_object (`bool`, defaults to `True`) : Cast dates to objects. If `False`, convert to `datetime64[ns]` dtype.

timestamp_as_object (`bool`, defaults to `False`) : Cast non-nanosecond timestamps (`np.datetime64`) to objects. This is useful if you have timestamps that don't fit in the normal date range of nanosecond timestamps (1678 CE-2262 CE). If `False`, all timestamps are converted to `datetime64[ns]` dtype.

use_threads (`bool`, defaults to `True`) : Whether to parallelize the conversion using multiple threads.

deduplicate_objects (`bool`, defaults to `False`) : Do not create multiple copies Python objects when created, to save on memory use. Conversion will be slower.

ignore_metadata (`bool`, defaults to `False`) : If `True`, do not use the 'pandas' metadata to reconstruct the DataFrame index, if present.

safe (`bool`, defaults to `True`) : For certain data types, a cast is needed in order to store the data in a pandas DataFrame or Series (e.g. timestamps are always stored as nanoseconds in pandas). This option controls whether it is a safe cast or not.

split_blocks (`bool`, defaults to `False`) : If `True`, generate one internal "block" for each column when creating a pandas.DataFrame from a `RecordBatch` or `Table`. While this can temporarily reduce memory note that various pandas operations can trigger "consolidation" which may balloon memory use.

self_destruct (`bool`, defaults to `False`) : EXPERIMENTAL: If `True`, attempt to deallocate the originating Arrow memory while converting the Arrow object to pandas. If you use the object after calling `to_pandas` with this option it will crash your program.

types_mapper (`function`, defaults to `None`) : A function mapping a pyarrow DataType to a pandas `ExtensionDtype`. This can be used to override the default pandas type for conversion of built-in pyarrow types or in absence of `pandas_metadata` in the Table schema. The function receives a pyarrow DataType and is expected to return a pandas `ExtensionDtype` or `None` if the default conversion should be used for that type. If you have a dictionary mapping, you can pass `dict.get` as function.

**Returns:** `pandas.Series` or `pandas.DataFrame`

`pandas.Series` or `pandas.DataFrame` depending on type of object

Convert to a pandas-compatible NumPy array or DataFrame, as appropriate.

#### to_string[[datasets.table.InMemoryTable.to_string]]

```python
to_string(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L362)

#### field[[datasets.table.InMemoryTable.field]]

```python
field(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L381)

**Parameters:**

i (`Union[int, str]`) : The index or name of the field to retrieve.

**Returns:**

`pyarrow.Field`

Select a schema field by its column name or numeric index.

#### column[[datasets.table.InMemoryTable.column]]

```python
column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L394)

**Parameters:**

i (`Union[int, str]`) : The index or name of the column to retrieve.

**Returns:**

`pyarrow.ChunkedArray`

Select a column by its column name, or numeric index.

#### itercolumns[[datasets.table.InMemoryTable.itercolumns]]

```python
itercolumns(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L407)

**Yields:**

`pyarrow.ChunkedArray`

Iterator over all columns in their numerical order.

#### schema[[datasets.table.InMemoryTable.schema]]

```python
schema()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L416)

**Returns:**

`pyarrow.Schema`

Schema of the table and its columns.

#### columns[[datasets.table.InMemoryTable.columns]]

```python
columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L426)

**Returns:**

`List[pa.ChunkedArray]`

List of all columns in numerical order.

#### num_columns[[datasets.table.InMemoryTable.num_columns]]

```python
num_columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L436)

**Returns:**

int

Number of columns in this table.

#### num_rows[[datasets.table.InMemoryTable.num_rows]]

```python
num_rows()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L446)

**Returns:**

int

Number of rows in this table.

Due to the definition of a table, all columns have the same number of
rows.

#### shape[[datasets.table.InMemoryTable.shape]]

```python
shape()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L459)

**Returns:** `(int, int)`

Number of rows and number of columns.

Dimensions of the table: (#rows, #columns).

#### nbytes[[datasets.table.InMemoryTable.nbytes]]

```python
nbytes()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L469)

Total number of bytes consumed by the elements of the table.

#### column_names[[datasets.table.InMemoryTable.column_names]]

```python
column_names()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L476)

Names of the table's columns.

#### slice[[datasets.table.InMemoryTable.slice]]

```python
slice(offset = 0, length = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L859)

**Parameters:**

offset (`int`, defaults to `0`) : Offset from start of table to slice.

length (`int`, defaults to `None`) : Length of slice (default is until end of table starting from offset).

**Returns:**

`datasets.table.Table`

Compute zero-copy slice of this Table.

#### filter[[datasets.table.InMemoryTable.filter]]

```python
filter(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L876)

Select records from a Table. See `pyarrow.compute.filter` for full usage.

#### flatten[[datasets.table.InMemoryTable.flatten]]

```python
flatten(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L882)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : For memory allocations, if required, otherwise use default pool.

**Returns:**

`datasets.table.Table`

Flatten this Table.  Each column with a struct type is flattened
into one column per struct field.  Other columns are left unchanged.

#### combine_chunks[[datasets.table.InMemoryTable.combine_chunks]]

```python
combine_chunks(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L896)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : For memory allocations, if required, otherwise use default pool.

**Returns:**

`datasets.table.Table`

Make a new table by combining the chunks this table has.

All the underlying chunks in the `ChunkedArray` of each column are
concatenated into zero or one chunk.

#### cast[[datasets.table.InMemoryTable.cast]]

```python
cast(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L912)

**Parameters:**

target_schema (`Schema`) : Schema to cast to, the names and order of fields must match.

safe (`bool`, defaults to `True`) : Check for overflows or other unsafe conversions.

**Returns:**

`datasets.table.Table`

Cast table values to another schema.

#### replace_schema_metadata[[datasets.table.InMemoryTable.replace_schema_metadata]]

```python
replace_schema_metadata(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L927)

**Parameters:**

metadata (`dict`, defaults to `None`) --

**Returns:** `datasets.table.Table`

shallow_copy

EXPERIMENTAL: Create shallow copy of table by replacing schema
key-value metadata with the indicated new metadata (which may be `None`,
which deletes any existing metadata).

#### add_column[[datasets.table.InMemoryTable.add_column]]

```python
add_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L941)

**Parameters:**

i (`int`) : Index to place the column at.

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column added.

Add column to Table at position.

A new table is returned with the column added, the original table
object is left unchanged.

#### append_column[[datasets.table.InMemoryTable.append_column]]

```python
append_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L962)

**Parameters:**

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column added.

Append column at end of columns.

#### remove_column[[datasets.table.InMemoryTable.remove_column]]

```python
remove_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L979)

**Parameters:**

i (`int`) : Index of column to remove.

**Returns:** `datasets.table.Table`

New table without the column.

Create new Table with the indicated column removed.

#### set_column[[datasets.table.InMemoryTable.set_column]]

```python
set_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L993)

**Parameters:**

i (`int`) : Index to place the column at.

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column set.

Replace column in Table at position.

#### rename_columns[[datasets.table.InMemoryTable.rename_columns]]

```python
rename_columns(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1012)

Create new table with columns renamed to provided names.

#### select[[datasets.table.InMemoryTable.select]]

```python
select(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1035)

**Parameters:**

columns (`Union[List[str], List[int]]`) : The column names or integer indices to select.

**Returns:** [datasets.table.Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)

New table with the specified columns, and metadata preserved.

Select columns of the table.

Returns a new table with the specified columns, and metadata preserved.

#### drop[[datasets.table.InMemoryTable.drop]]

```python
drop(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1018)

**Parameters:**

columns (`List[str]`) : List of field names referencing existing columns.

**Returns:** `datasets.table.Table`

New table without the columns.

**Raises:** ``KeyError``

- ``KeyError`` -- : if any of the passed columns name are not existing.

Drop one or more columns and return a new table.

#### from_file[[datasets.table.InMemoryTable.from_file]]

```python
from_file(filename: str)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L719)

#### from_buffer[[datasets.table.InMemoryTable.from_buffer]]

```python
from_buffer(buffer: Buffer)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L724)

#### from_pandas[[datasets.table.InMemoryTable.from_pandas]]

```python
from_pandas(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L729)

**Parameters:**

df (`pandas.DataFrame`) --

schema (`pyarrow.Schema`, *optional*) : The expected schema of the Arrow Table. This can be used to indicate the type of columns if we cannot infer it automatically. If passed, the output will have exactly this schema. Columns specified in the schema that are not found in the DataFrame columns or its index will raise an error. Additional columns or index levels in the DataFrame which are not specified in the schema will be ignored.

preserve_index (`bool`, *optional*) : Whether to store the index as an additional column in the resulting `Table`. The default of None will store the index as a column, except for RangeIndex which is stored as metadata only. Use `preserve_index=True` to force it to be stored as a column.

nthreads (`int`, defaults to `None` (may use up to system CPU count threads)) : If greater than 1, convert columns to Arrow in parallel using indicated number of threads.

columns (`List[str]`, *optional*) : List of column to be converted. If `None`, use all columns.

safe (`bool`, defaults to `True`) : Check for overflows or other unsafe conversions,

**Returns:** `datasets.table.Table`

Convert pandas.DataFrame to an Arrow Table.

The column types in the resulting Arrow Table are inferred from the
dtypes of the pandas.Series in the DataFrame. In the case of non-object
Series, the NumPy dtype is translated to its Arrow equivalent. In the
case of `object`, we need to guess the datatype by looking at the
Python objects in this Series.

Be aware that Series of the `object` dtype don't carry enough
information to always lead to a meaningful Arrow type. In the case that
we cannot infer a type, e.g. because the DataFrame is of length 0 or
the Series only contains `None/nan` objects, the type is set to
null. This behavior can be avoided by constructing an explicit schema
and passing it to this function.

Examples:
```python
>>> import pandas as pd
>>> import pyarrow as pa
>>> df = pd.DataFrame({
    ...     'int': [1, 2],
    ...     'str': ['a', 'b']
    ... })
>>> pa.Table.from_pandas(df)
<pyarrow.lib.Table object at 0x7f05d1fb1b40>
```

#### from_arrays[[datasets.table.InMemoryTable.from_arrays]]

```python
from_arrays(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L787)

**Parameters:**

arrays (`List[Union[pyarrow.Array, pyarrow.ChunkedArray]]`) : Equal-length arrays that should form the table.

names (`List[str]`, *optional*) : Names for the table columns. If not passed, schema must be passed.

schema (`Schema`, defaults to `None`) : Schema for the created table. If not passed, names must be passed.

metadata (`Union[dict, Mapping]`, defaults to `None`) : Optional metadata for the schema (if inferred).

**Returns:**

`datasets.table.Table`

Construct a Table from Arrow arrays.

#### from_pydict[[datasets.table.InMemoryTable.from_pydict]]

```python
from_pydict(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L807)

**Parameters:**

mapping (`Union[dict, Mapping]`) : A mapping of strings to Arrays or Python lists.

schema (`Schema`, defaults to `None`) : If not passed, will be inferred from the Mapping values

metadata (`Union[dict, Mapping]`, defaults to `None`) : Optional metadata for the schema (if inferred).

**Returns:**

`datasets.table.Table`

Construct a Table from Arrow arrays or columns.

#### from_batches[[datasets.table.InMemoryTable.from_batches]]

```python
from_batches(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L843)

**Parameters:**

batches (`Union[Sequence[pyarrow.RecordBatch], Iterator[pyarrow.RecordBatch]]`) : Sequence of `RecordBatch` to be converted, all schemas must be equal.

schema (`Schema`, defaults to `None`) : If not passed, will be inferred from the first `RecordBatch`.

**Returns:** `datasets.table.Table`

Construct a Table from a sequence or iterator of Arrow `RecordBatches`.

## MemoryMappedTable[[datasets.table.MemoryMappedTable]]

#### datasets.table.MemoryMappedTable[[datasets.table.MemoryMappedTable]]

```python
datasets.table.MemoryMappedTable(table: Table, path: str, replays: typing.Optional[list[tuple[str, tuple, dict]]] = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1055)

The table is said memory mapped when it doesn't use the user's RAM but loads the data
from the disk instead.

Pickling it doesn't copy the data into memory.
Instead, only the path to the memory mapped arrow file is pickled, as well as the list
of transforms to "replay" when reloading the table from the disk.

Its implementation requires to store an history of all the transforms that were applied
to the underlying pyarrow Table, so that they can be "replayed" when reloading the Table
from the disk.

This is different from the `InMemoryTable` table, for which pickling does copy all the
data in memory.

`InMemoryTable` must be used when data fit in memory, while `MemoryMapped` are reserved for
data bigger than memory or when you want the memory footprint of your application to
stay low.

#### validate[[datasets.table.MemoryMappedTable.validate]]

```python
validate(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L235)

**Parameters:**

full (`bool`, defaults to `False`) : If `True`, run expensive checks, otherwise cheap checks only.

**Raises:** ``pa.lib.ArrowInvalid``

- ``pa.lib.ArrowInvalid`` -- if validation fails

Perform validation checks.  An exception is raised if validation fails.

By default only cheap validation checks are run.  Pass `full=True`
for thorough validation checks (potentially `O(n)`).

#### equals[[datasets.table.MemoryMappedTable.equals]]

```python
equals(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L251)

**Parameters:**

other ([Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)) : Table to compare against.

check_metadata `bool`, defaults to `False`) : Whether schema metadata equality should be checked as well.

**Returns:** `bool`

Check if contents of two tables are equal.

#### to_batches[[datasets.table.MemoryMappedTable.to_batches]]

```python
to_batches(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L268)

**Parameters:**

max_chunksize (`int`, defaults to `None`) : Maximum size for `RecordBatch` chunks. Individual chunks may be smaller depending on the chunk layout of individual columns.

**Returns:**

`List[pyarrow.RecordBatch]`

Convert Table to list of (contiguous) `RecordBatch` objects.

#### to_pydict[[datasets.table.MemoryMappedTable.to_pydict]]

```python
to_pydict(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L282)

**Returns:** `dict`

Convert the Table to a `dict` or `OrderedDict`.

#### to_pandas[[datasets.table.MemoryMappedTable.to_pandas]]

```python
to_pandas(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L300)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : Arrow MemoryPool to use for allocations. Uses the default memory pool is not passed.

strings_to_categorical (`bool`, defaults to `False`) : Encode string (UTF8) and binary types to `pandas.Categorical`.

categories (`list`, defaults to `empty`) : List of fields that should be returned as `pandas.Categorical`. Only applies to table-like data structures.

zero_copy_only (`bool`, defaults to `False`) : Raise an `ArrowException` if this function call would require copying the underlying data.

integer_object_nulls (`bool`, defaults to `False`) : Cast integers with nulls to objects.

date_as_object (`bool`, defaults to `True`) : Cast dates to objects. If `False`, convert to `datetime64[ns]` dtype.

timestamp_as_object (`bool`, defaults to `False`) : Cast non-nanosecond timestamps (`np.datetime64`) to objects. This is useful if you have timestamps that don't fit in the normal date range of nanosecond timestamps (1678 CE-2262 CE). If `False`, all timestamps are converted to `datetime64[ns]` dtype.

use_threads (`bool`, defaults to `True`) : Whether to parallelize the conversion using multiple threads.

deduplicate_objects (`bool`, defaults to `False`) : Do not create multiple copies Python objects when created, to save on memory use. Conversion will be slower.

ignore_metadata (`bool`, defaults to `False`) : If `True`, do not use the 'pandas' metadata to reconstruct the DataFrame index, if present.

safe (`bool`, defaults to `True`) : For certain data types, a cast is needed in order to store the data in a pandas DataFrame or Series (e.g. timestamps are always stored as nanoseconds in pandas). This option controls whether it is a safe cast or not.

split_blocks (`bool`, defaults to `False`) : If `True`, generate one internal "block" for each column when creating a pandas.DataFrame from a `RecordBatch` or `Table`. While this can temporarily reduce memory note that various pandas operations can trigger "consolidation" which may balloon memory use.

self_destruct (`bool`, defaults to `False`) : EXPERIMENTAL: If `True`, attempt to deallocate the originating Arrow memory while converting the Arrow object to pandas. If you use the object after calling `to_pandas` with this option it will crash your program.

types_mapper (`function`, defaults to `None`) : A function mapping a pyarrow DataType to a pandas `ExtensionDtype`. This can be used to override the default pandas type for conversion of built-in pyarrow types or in absence of `pandas_metadata` in the Table schema. The function receives a pyarrow DataType and is expected to return a pandas `ExtensionDtype` or `None` if the default conversion should be used for that type. If you have a dictionary mapping, you can pass `dict.get` as function.

**Returns:** `pandas.Series` or `pandas.DataFrame`

`pandas.Series` or `pandas.DataFrame` depending on type of object

Convert to a pandas-compatible NumPy array or DataFrame, as appropriate.

#### to_string[[datasets.table.MemoryMappedTable.to_string]]

```python
to_string(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L362)

#### field[[datasets.table.MemoryMappedTable.field]]

```python
field(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L381)

**Parameters:**

i (`Union[int, str]`) : The index or name of the field to retrieve.

**Returns:**

`pyarrow.Field`

Select a schema field by its column name or numeric index.

#### column[[datasets.table.MemoryMappedTable.column]]

```python
column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L394)

**Parameters:**

i (`Union[int, str]`) : The index or name of the column to retrieve.

**Returns:**

`pyarrow.ChunkedArray`

Select a column by its column name, or numeric index.

#### itercolumns[[datasets.table.MemoryMappedTable.itercolumns]]

```python
itercolumns(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L407)

**Yields:**

`pyarrow.ChunkedArray`

Iterator over all columns in their numerical order.

#### schema[[datasets.table.MemoryMappedTable.schema]]

```python
schema()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L416)

**Returns:**

`pyarrow.Schema`

Schema of the table and its columns.

#### columns[[datasets.table.MemoryMappedTable.columns]]

```python
columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L426)

**Returns:**

`List[pa.ChunkedArray]`

List of all columns in numerical order.

#### num_columns[[datasets.table.MemoryMappedTable.num_columns]]

```python
num_columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L436)

**Returns:**

int

Number of columns in this table.

#### num_rows[[datasets.table.MemoryMappedTable.num_rows]]

```python
num_rows()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L446)

**Returns:**

int

Number of rows in this table.

Due to the definition of a table, all columns have the same number of
rows.

#### shape[[datasets.table.MemoryMappedTable.shape]]

```python
shape()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L459)

**Returns:** `(int, int)`

Number of rows and number of columns.

Dimensions of the table: (#rows, #columns).

#### nbytes[[datasets.table.MemoryMappedTable.nbytes]]

```python
nbytes()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L469)

Total number of bytes consumed by the elements of the table.

#### column_names[[datasets.table.MemoryMappedTable.column_names]]

```python
column_names()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L476)

Names of the table's columns.

#### slice[[datasets.table.MemoryMappedTable.slice]]

```python
slice(offset = 0, length = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1114)

**Parameters:**

offset (`int`, defaults to `0`) : Offset from start of table to slice.

length (`int`, defaults to `None`) : Length of slice (default is until end of table starting from offset).

**Returns:**

`datasets.table.Table`

Compute zero-copy slice of this Table.

#### filter[[datasets.table.MemoryMappedTable.filter]]

```python
filter(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1133)

Select records from a Table. See `pyarrow.compute.filter` for full usage.

#### flatten[[datasets.table.MemoryMappedTable.flatten]]

```python
flatten(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1141)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : For memory allocations, if required, otherwise use default pool.

**Returns:**

`datasets.table.Table`

Flatten this Table.  Each column with a struct type is flattened
into one column per struct field.  Other columns are left unchanged.

#### combine_chunks[[datasets.table.MemoryMappedTable.combine_chunks]]

```python
combine_chunks(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1157)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : For memory allocations, if required, otherwise use default pool.

**Returns:**

`datasets.table.Table`

Make a new table by combining the chunks this table has.

All the underlying chunks in the ChunkedArray of each column are
concatenated into zero or one chunk.

#### cast[[datasets.table.MemoryMappedTable.cast]]

```python
cast(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1175)

**Parameters:**

target_schema (`Schema`) : Schema to cast to, the names and order of fields must match.

safe (`bool`, defaults to `True`) : Check for overflows or other unsafe conversions.

**Returns:**

`datasets.table.Table`

Cast table values to another schema

#### replace_schema_metadata[[datasets.table.MemoryMappedTable.replace_schema_metadata]]

```python
replace_schema_metadata(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1192)

**Parameters:**

metadata (`dict`, defaults to `None`) --

**Returns:** `datasets.table.Table`

shallow_copy

EXPERIMENTAL: Create shallow copy of table by replacing schema
key-value metadata with the indicated new metadata (which may be None,
which deletes any existing metadata.

#### add_column[[datasets.table.MemoryMappedTable.add_column]]

```python
add_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1208)

**Parameters:**

i (`int`) : Index to place the column at.

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column added.

Add column to Table at position.

A new table is returned with the column added, the original table
object is left unchanged.

#### append_column[[datasets.table.MemoryMappedTable.append_column]]

```python
append_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1231)

**Parameters:**

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column added.

Append column at end of columns.

#### remove_column[[datasets.table.MemoryMappedTable.remove_column]]

```python
remove_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1250)

**Parameters:**

i (`int`) : Index of column to remove.

**Returns:** `datasets.table.Table`

New table without the column.

Create new Table with the indicated column removed.

#### set_column[[datasets.table.MemoryMappedTable.set_column]]

```python
set_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1266)

**Parameters:**

i (`int`) : Index to place the column at.

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column set.

Replace column in Table at position.

#### rename_columns[[datasets.table.MemoryMappedTable.rename_columns]]

```python
rename_columns(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1287)

Create new table with columns renamed to provided names.

#### select[[datasets.table.MemoryMappedTable.select]]

```python
select(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1314)

**Parameters:**

columns (`Union[List[str], List[int]]`) : The column names or integer indices to select.

**Returns:** [datasets.table.Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)

New table with the specified columns, and metadata preserved.

Select columns of the table.

Returns a new table with the specified columns, and metadata preserved.

#### drop[[datasets.table.MemoryMappedTable.drop]]

```python
drop(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1295)

**Parameters:**

columns (`List[str]`) : List of field names referencing existing columns.

**Returns:** `datasets.table.Table`

New table without the columns.

**Raises:** ``KeyError``

- ``KeyError`` -- : if any of the passed columns name are not existing.

Drop one or more columns and return a new table.

#### from_file[[datasets.table.MemoryMappedTable.from_file]]

```python
from_file(filename: str, replays = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1081)

## ConcatenationTable[[datasets.table.ConcatenationTable]]

#### datasets.table.ConcatenationTable[[datasets.table.ConcatenationTable]]

```python
datasets.table.ConcatenationTable(table: Table, blocks: list)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1339)

The table comes from the concatenation of several tables called blocks.
It enables concatenation on both axis 0 (append rows) and axis 1 (append columns).

The underlying tables are called "blocks" and can be either `InMemoryTable`
or `MemoryMappedTable` objects.
This allows to combine tables that come from memory or that are memory mapped.
When a `ConcatenationTable` is pickled, then each block is pickled:
- the `InMemoryTable` objects are pickled by copying all the data in memory.
- the MemoryMappedTable objects are pickled without copying the data into memory.
Instead, only the path to the memory mapped arrow file is pickled, as well as the list
of transforms to "replays" when reloading the table from the disk.

Its implementation requires to store each block separately.
The `blocks` attributes stores a list of list of blocks.
The first axis concatenates the tables along the axis 0 (it appends rows),
while the second axis concatenates tables along the axis 1 (it appends columns).

If some columns are missing when concatenating on axis 0, they are filled with null values.
This is done using `pyarrow.concat_tables(tables, promote=True)`.

You can access the fully combined table by accessing the `ConcatenationTable.table` attribute,
and the blocks by accessing the `ConcatenationTable.blocks` attribute.

#### validate[[datasets.table.ConcatenationTable.validate]]

```python
validate(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L235)

**Parameters:**

full (`bool`, defaults to `False`) : If `True`, run expensive checks, otherwise cheap checks only.

**Raises:** ``pa.lib.ArrowInvalid``

- ``pa.lib.ArrowInvalid`` -- if validation fails

Perform validation checks.  An exception is raised if validation fails.

By default only cheap validation checks are run.  Pass `full=True`
for thorough validation checks (potentially `O(n)`).

#### equals[[datasets.table.ConcatenationTable.equals]]

```python
equals(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L251)

**Parameters:**

other ([Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)) : Table to compare against.

check_metadata `bool`, defaults to `False`) : Whether schema metadata equality should be checked as well.

**Returns:** `bool`

Check if contents of two tables are equal.

#### to_batches[[datasets.table.ConcatenationTable.to_batches]]

```python
to_batches(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L268)

**Parameters:**

max_chunksize (`int`, defaults to `None`) : Maximum size for `RecordBatch` chunks. Individual chunks may be smaller depending on the chunk layout of individual columns.

**Returns:**

`List[pyarrow.RecordBatch]`

Convert Table to list of (contiguous) `RecordBatch` objects.

#### to_pydict[[datasets.table.ConcatenationTable.to_pydict]]

```python
to_pydict(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L282)

**Returns:** `dict`

Convert the Table to a `dict` or `OrderedDict`.

#### to_pandas[[datasets.table.ConcatenationTable.to_pandas]]

```python
to_pandas(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L300)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : Arrow MemoryPool to use for allocations. Uses the default memory pool is not passed.

strings_to_categorical (`bool`, defaults to `False`) : Encode string (UTF8) and binary types to `pandas.Categorical`.

categories (`list`, defaults to `empty`) : List of fields that should be returned as `pandas.Categorical`. Only applies to table-like data structures.

zero_copy_only (`bool`, defaults to `False`) : Raise an `ArrowException` if this function call would require copying the underlying data.

integer_object_nulls (`bool`, defaults to `False`) : Cast integers with nulls to objects.

date_as_object (`bool`, defaults to `True`) : Cast dates to objects. If `False`, convert to `datetime64[ns]` dtype.

timestamp_as_object (`bool`, defaults to `False`) : Cast non-nanosecond timestamps (`np.datetime64`) to objects. This is useful if you have timestamps that don't fit in the normal date range of nanosecond timestamps (1678 CE-2262 CE). If `False`, all timestamps are converted to `datetime64[ns]` dtype.

use_threads (`bool`, defaults to `True`) : Whether to parallelize the conversion using multiple threads.

deduplicate_objects (`bool`, defaults to `False`) : Do not create multiple copies Python objects when created, to save on memory use. Conversion will be slower.

ignore_metadata (`bool`, defaults to `False`) : If `True`, do not use the 'pandas' metadata to reconstruct the DataFrame index, if present.

safe (`bool`, defaults to `True`) : For certain data types, a cast is needed in order to store the data in a pandas DataFrame or Series (e.g. timestamps are always stored as nanoseconds in pandas). This option controls whether it is a safe cast or not.

split_blocks (`bool`, defaults to `False`) : If `True`, generate one internal "block" for each column when creating a pandas.DataFrame from a `RecordBatch` or `Table`. While this can temporarily reduce memory note that various pandas operations can trigger "consolidation" which may balloon memory use.

self_destruct (`bool`, defaults to `False`) : EXPERIMENTAL: If `True`, attempt to deallocate the originating Arrow memory while converting the Arrow object to pandas. If you use the object after calling `to_pandas` with this option it will crash your program.

types_mapper (`function`, defaults to `None`) : A function mapping a pyarrow DataType to a pandas `ExtensionDtype`. This can be used to override the default pandas type for conversion of built-in pyarrow types or in absence of `pandas_metadata` in the Table schema. The function receives a pyarrow DataType and is expected to return a pandas `ExtensionDtype` or `None` if the default conversion should be used for that type. If you have a dictionary mapping, you can pass `dict.get` as function.

**Returns:** `pandas.Series` or `pandas.DataFrame`

`pandas.Series` or `pandas.DataFrame` depending on type of object

Convert to a pandas-compatible NumPy array or DataFrame, as appropriate.

#### to_string[[datasets.table.ConcatenationTable.to_string]]

```python
to_string(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L362)

#### field[[datasets.table.ConcatenationTable.field]]

```python
field(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L381)

**Parameters:**

i (`Union[int, str]`) : The index or name of the field to retrieve.

**Returns:**

`pyarrow.Field`

Select a schema field by its column name or numeric index.

#### column[[datasets.table.ConcatenationTable.column]]

```python
column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L394)

**Parameters:**

i (`Union[int, str]`) : The index or name of the column to retrieve.

**Returns:**

`pyarrow.ChunkedArray`

Select a column by its column name, or numeric index.

#### itercolumns[[datasets.table.ConcatenationTable.itercolumns]]

```python
itercolumns(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L407)

**Yields:**

`pyarrow.ChunkedArray`

Iterator over all columns in their numerical order.

#### schema[[datasets.table.ConcatenationTable.schema]]

```python
schema()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L416)

**Returns:**

`pyarrow.Schema`

Schema of the table and its columns.

#### columns[[datasets.table.ConcatenationTable.columns]]

```python
columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L426)

**Returns:**

`List[pa.ChunkedArray]`

List of all columns in numerical order.

#### num_columns[[datasets.table.ConcatenationTable.num_columns]]

```python
num_columns()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L436)

**Returns:**

int

Number of columns in this table.

#### num_rows[[datasets.table.ConcatenationTable.num_rows]]

```python
num_rows()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L446)

**Returns:**

int

Number of rows in this table.

Due to the definition of a table, all columns have the same number of
rows.

#### shape[[datasets.table.ConcatenationTable.shape]]

```python
shape()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L459)

**Returns:** `(int, int)`

Number of rows and number of columns.

Dimensions of the table: (#rows, #columns).

#### nbytes[[datasets.table.ConcatenationTable.nbytes]]

```python
nbytes()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L469)

Total number of bytes consumed by the elements of the table.

#### column_names[[datasets.table.ConcatenationTable.column_names]]

```python
column_names()
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L476)

Names of the table's columns.

#### slice[[datasets.table.ConcatenationTable.slice]]

```python
slice(offset = 0, length = None)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1548)

**Parameters:**

offset (`int`, defaults to `0`) : Offset from start of table to slice.

length (`int`, defaults to `None`) : Length of slice (default is until end of table starting from offset).

**Returns:**

`datasets.table.Table`

Compute zero-copy slice of this Table.

#### filter[[datasets.table.ConcatenationTable.filter]]

```python
filter(mask, *args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1579)

Select records from a Table. See `pyarrow.compute.filter` for full usage.

#### flatten[[datasets.table.ConcatenationTable.flatten]]

```python
flatten(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1590)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : For memory allocations, if required, otherwise use default pool.

**Returns:**

`datasets.table.Table`

Flatten this Table.  Each column with a struct type is flattened
into one column per struct field.  Other columns are left unchanged.

#### combine_chunks[[datasets.table.ConcatenationTable.combine_chunks]]

```python
combine_chunks(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1608)

**Parameters:**

memory_pool (`MemoryPool`, defaults to `None`) : For memory allocations, if required, otherwise use default pool.

**Returns:**

`datasets.table.Table`

Make a new table by combining the chunks this table has.

All the underlying chunks in the `ChunkedArray` of each column are
concatenated into zero or one chunk.

#### cast[[datasets.table.ConcatenationTable.cast]]

```python
cast(target_schema, *args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1628)

**Parameters:**

target_schema (`Schema`) : Schema to cast to, the names and order of fields must match.

safe (`bool`, defaults to `True`) : Check for overflows or other unsafe conversions.

**Returns:**

`datasets.table.Table`

Cast table values to another schema.

#### replace_schema_metadata[[datasets.table.ConcatenationTable.replace_schema_metadata]]

```python
replace_schema_metadata(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1659)

**Parameters:**

metadata (`dict`, defaults to `None`) --

**Returns:** `datasets.table.Table`

shallow_copy

EXPERIMENTAL: Create shallow copy of table by replacing schema
key-value metadata with the indicated new metadata (which may be `None`,
which deletes any existing metadata).

#### add_column[[datasets.table.ConcatenationTable.add_column]]

```python
add_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1677)

**Parameters:**

i (`int`) : Index to place the column at.

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column added.

Add column to Table at position.

A new table is returned with the column added, the original table
object is left unchanged.

#### append_column[[datasets.table.ConcatenationTable.append_column]]

```python
append_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1698)

**Parameters:**

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column added.

Append column at end of columns.

#### remove_column[[datasets.table.ConcatenationTable.remove_column]]

```python
remove_column(i, *args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1715)

**Parameters:**

i (`int`) : Index of column to remove.

**Returns:** `datasets.table.Table`

New table without the column.

Create new Table with the indicated column removed.

#### set_column[[datasets.table.ConcatenationTable.set_column]]

```python
set_column(*args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1739)

**Parameters:**

i (`int`) : Index to place the column at.

field_ (`Union[str, pyarrow.Field]`) : If a string is passed then the type is deduced from the column data.

column (`Union[pyarrow.Array, List[pyarrow.Array]]`) : Column data.

**Returns:** `datasets.table.Table`

New table with the passed column set.

Replace column in Table at position.

#### rename_columns[[datasets.table.ConcatenationTable.rename_columns]]

```python
rename_columns(names, *args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1758)

Create new table with columns renamed to provided names.

#### select[[datasets.table.ConcatenationTable.select]]

```python
select(columns, *args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1792)

**Parameters:**

columns (`Union[List[str], List[int]]`) : The column names or integer indices to select.

**Returns:** [datasets.table.Table](/docs/datasets/v5.2.0/en/package_reference/table_classes#datasets.table.Table)

New table with the specified columns, and metadata preserved.

Select columns of the table.

Returns a new table with the specified columns, and metadata preserved.

#### drop[[datasets.table.ConcatenationTable.drop]]

```python
drop(columns, *args, **kwargs)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1771)

**Parameters:**

columns (`List[str]`) : List of field names referencing existing columns.

**Returns:** `datasets.table.Table`

New table without the columns.

**Raises:** ``KeyError``

- ``KeyError`` -- : if any of the passed columns name are not existing.

Drop one or more columns and return a new table.

#### from_blocks[[datasets.table.ConcatenationTable.from_blocks]]

```python
from_blocks(blocks: ~TableBlockContainer)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1444)

#### from_tables[[datasets.table.ConcatenationTable.from_tables]]

```python
from_tables(tables: list, axis: int = 0)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1458)

**Parameters:**

tables (list of `Table` or list of `pyarrow.Table`) : List of tables.

axis (`{0, 1}`, defaults to `0`, meaning over rows) : Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns (horizontally).  

Create `ConcatenationTable` from list of tables.

## Utils[[datasets.table.concat_tables]]

#### datasets.table.concat_tables[[datasets.table.concat_tables]]

```python
datasets.table.concat_tables(tables: list, axis: int = 0)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1812)

**Parameters:**

tables (list of `Table`) : List of tables to be concatenated.

axis (`{0, 1}`, defaults to `0`, meaning over rows) : Axis to concatenate over, where `0` means over rows (vertically) and `1` means over columns (horizontally).  

**Returns:** `datasets.table.Table`

If the number of input tables is > 1, then the returned table is a `datasets.table.ConcatenationTable`.
Otherwise if there's only one table, it is returned as is.

Concatenate tables.

#### datasets.table.list_table_cache_files[[datasets.table.list_table_cache_files]]

```python
datasets.table.list_table_cache_files(table: Table)
```

[Source](https://github.com/huggingface/datasets/blob/5.2.0/src/datasets/table.py#L1835)

**Returns:** `List[str]`

A list of paths to the cache files loaded by the table.

Get the cache files that are loaded by the table.
Cache file are used when parts of the table come from the disk via memory mapping.

### Main classes
https://huggingface.co/docs/datasets/v5.2.0/package_reference/main_classes.md
