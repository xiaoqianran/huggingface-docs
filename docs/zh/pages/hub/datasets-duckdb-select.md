<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 查询数据集

查询数据集是数据分析的基本步骤。在这里，我们将指导您使用各种方法查询数据集。

有[several ways](https://duckdb.org/docs/data/parquet/overview.html)来选择您的数据。

使用 `FROM` 语法：
```bash
FROM 'hf://datasets/jamescalam/world-cities-geo/train.jsonl' SELECT city, country, region LIMIT 3;

┌────────────────┬─────────────┬───────────────┐
│      city      │   country   │    region     │
│    varchar     │   varchar   │    varchar    │
├────────────────┼─────────────┼───────────────┤
│ Kabul          │ Afghanistan │ Southern Asia │
│ Kandahar       │ Afghanistan │ Southern Asia │
│ Mazar-e Sharif │ Afghanistan │ Southern Asia │
└────────────────┴─────────────┴───────────────┘

```

使用 `SELECT` 和 `FROM` 语法：

```bash
SELECT city, country, region FROM 'hf://datasets/jamescalam/world-cities-geo/train.jsonl' USING SAMPLE 3;

┌──────────┬─────────┬────────────────┐
│   city   │ country │     region     │
│ varchar  │ varchar │    varchar     │
├──────────┼─────────┼────────────────┤
│ Wenzhou  │ China   │ Eastern Asia   │
│ Valdez   │ Ecuador │ South America  │
│ Aplahoue │ Benin   │ Western Africa │
└──────────┴─────────┴────────────────┘

```

计算与 glob 模式匹配的所有 JSONL 文件的数量：

```bash
SELECT COUNT(*) FROM 'hf://datasets/jamescalam/world-cities-geo/*.jsonl';

┌──────────────┐
│ count_star() │
│    int64     │
├──────────────┤
│         9083 │
└──────────────┘

```

您还可以使用 `read_parquet` 函数（或其别名 `parquet_scan`）查询 Parquet 文件。此函数与其他 [parameters](https://duckdb.org/docs/data/parquet/overview.html#parameters) 一起，提供了处理 Parquet 文件的灵活性，特别是在它们没有 `.parquet` 扩展名的情况下。让我们使用同一数据集中自动转换的 Parquet 文件来探索这些函数。

使用[read_parquet](https://duckdb.org/docs/guides/file_formats/query_parquet.html)功能选择：

```bash
SELECT * FROM read_parquet('hf://datasets/jamescalam/world-cities-geo@~parquet/default/**/*.parquet') LIMIT 3;

┌────────────────┬─────────────┬───────────────┬───────────┬────────────┬────────────┬────────────────────┬───────────────────┬────────────────────┐
│      city      │   country   │    region     │ continent │  latitude  │ longitude  │         x          │         y         │         z          │
│    varchar     │   varchar   │    varchar    │  varchar  │   double   │   double   │       double       │      double       │       double       │
├────────────────┼─────────────┼───────────────┼───────────┼────────────┼────────────┼────────────────────┼───────────────────┼────────────────────┤
│ Kabul          │ Afghanistan │ Southern Asia │ Asia      │ 34.5166667 │ 69.1833344 │  1865.546409629258 │ 4906.785732164055 │ 3610.1012966606136 │
│ Kandahar       │ Afghanistan │ Southern Asia │ Asia      │      31.61 │ 65.6999969 │  2232.782351694877 │ 4945.064042683584 │  3339.261233224765 │
│ Mazar-e Sharif │ Afghanistan │ Southern Asia │ Asia      │ 36.7069444 │ 67.1122208 │ 1986.5057687360124 │  4705.51748048584 │  3808.088900172991 │
└────────────────┴─────────────┴───────────────┴───────────┴────────────┴────────────┴────────────────────┴───────────────────┴────────────────────┘

```

读取与 glob 模式匹配的所有文件，并包含指定每行来自哪个文件的文件名列：

```bash
SELECT city, country, filename FROM read_parquet('hf://datasets/jamescalam/world-cities-geo@~parquet/default/**/*.parquet', filename = true) LIMIT 3;

┌────────────────┬─────────────┬───────────────────────────────────────────────────────────────────────────────┐
│      city      │   country   │                                   filename                                    │
│    varchar     │   varchar   │                                    varchar                                    │
├────────────────┼─────────────┼───────────────────────────────────────────────────────────────────────────────┤
│ Kabul          │ Afghanistan │ hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet │
│ Kandahar       │ Afghanistan │ hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet │
│ Mazar-e Sharif │ Afghanistan │ hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet │
└────────────────┴─────────────┴───────────────────────────────────────────────────────────────────────────────┘

```

## 获取元数据和模式

[parquet_metadata](https://duckdb.org/docs/data/parquet/metadata.html) 函数可用于查询 Parquet 文件中包含的元数据。

```bash
SELECT * FROM parquet_metadata('hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet');

┌───────────────────────────────────────────────────────────────────────────────┬──────────────┬────────────────────┬─────────────┐
│                                   file_name                                   │ row_group_id │ row_group_num_rows │ compression │
│                                    varchar                                    │    int64     │       int64        │   varchar   │
├───────────────────────────────────────────────────────────────────────────────┼──────────────┼────────────────────┼─────────────┤
│ hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet │            0 │               1000 │ SNAPPY      │
│ hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet │            0 │               1000 │ SNAPPY      │
│ hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet │            0 │               1000 │ SNAPPY      │
└───────────────────────────────────────────────────────────────────────────────┴──────────────┴────────────────────┴─────────────┘

```

获取列名称和列类型：

```bash
DESCRIBE SELECT * FROM 'hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet';

┌─────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│ column_name │ column_type │  null   │   key   │ default │  extra  │
│   varchar   │   varchar   │ varchar │ varchar │ varchar │ varchar │
├─────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ city        │ VARCHAR     │ YES     │         │         │         │
│ country     │ VARCHAR     │ YES     │         │         │         │
│ region      │ VARCHAR     │ YES     │         │         │         │
│ continent   │ VARCHAR     │ YES     │         │         │         │
│ latitude    │ DOUBLE      │ YES     │         │         │         │
│ longitude   │ DOUBLE      │ YES     │         │         │         │
│ x           │ DOUBLE      │ YES     │         │         │         │
│ y           │ DOUBLE      │ YES     │         │         │         │
│ z           │ DOUBLE      │ YES     │         │         │         │
└─────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘

```

获取内部架构（不包括文件名）：

```bash
SELECT * EXCLUDE (file_name) FROM parquet_schema('hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet');

┌───────────┬────────────┬─────────────┬─────────────────┬──────────────┬────────────────┬───────┬───────────┬──────────┬──────────────┐
│   name    │    type    │ type_length │ repetition_type │ num_children │ converted_type │ scale │ precision │ field_id │ logical_type │
│  varchar  │  varchar   │   varchar   │     varchar     │    int64     │    varchar     │ int64 │   int64   │  int64   │   varchar    │
├───────────┼────────────┼─────────────┼─────────────────┼──────────────┼────────────────┼───────┼───────────┼──────────┼──────────────┤
│ schema    │            │             │ REQUIRED        │            9 │                │       │           │          │              │
│ city      │ BYTE_ARRAY │             │ OPTIONAL        │              │ UTF8           │       │           │          │ StringType() │
│ country   │ BYTE_ARRAY │             │ OPTIONAL        │              │ UTF8           │       │           │          │ StringType() │
│ region    │ BYTE_ARRAY │             │ OPTIONAL        │              │ UTF8           │       │           │          │ StringType() │
│ continent │ BYTE_ARRAY │             │ OPTIONAL        │              │ UTF8           │       │           │          │ StringType() │
│ latitude  │ DOUBLE     │             │ OPTIONAL        │              │                │       │           │          │              │
│ longitude │ DOUBLE     │             │ OPTIONAL        │              │                │       │           │          │              │
│ x         │ DOUBLE     │             │ OPTIONAL        │              │                │       │           │          │              │
│ y         │ DOUBLE     │             │ OPTIONAL        │              │                │       │           │          │              │
│ z         │ DOUBLE     │             │ OPTIONAL        │              │                │       │           │          │              │
├───────────┴────────────┴─────────────┴─────────────────┴──────────────┴────────────────┴───────┴───────────┴──────────┴──────────────┤

```

## 获取统计数据`SUMMARIZE` 命令可用于通过查询获取各种聚合（min、max、approx_unique、avg、std、q25、q50、q75、count）。它返回这些统计信息以及列名、列类型和 NULL 值的百分比。

```bash
SUMMARIZE SELECT latitude, longitude FROM 'hf://datasets/jamescalam/world-cities-geo@~parquet/default/train/0000.parquet';

┌─────────────┬─────────────┬──────────────┬─────────────┬───────────────┬────────────────────┬────────────────────┬────────────────────┬────────────────────┬────────────────────┬───────┬─────────────────┐
│ column_name │ column_type │     min      │     max     │ approx_unique │        avg         │        std         │        q25         │        q50         │        q75         │ count │ null_percentage │
│   varchar   │   varchar   │   varchar    │   varchar   │     int64     │      varchar       │      varchar       │      varchar       │      varchar       │      varchar       │ int64 │  decimal(9,2)   │
├─────────────┼─────────────┼──────────────┼─────────────┼───────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼────────────────────┼───────┼─────────────────┤
│ latitude    │ DOUBLE      │ -54.8        │ 67.8557214  │          7324 │ 22.5004568364307   │ 26.770454684690925 │ 6.089858461951687  │ 29.321258648324747 │ 44.90191158328915  │  9083 │            0.00 │
│ longitude   │ DOUBLE      │ -175.2166595 │ 179.3833313 │          7802 │ 14.699333721953098 │ 63.93672742608224  │ -6.877990418604821 │ 19.12963979385393  │ 43.873513093419966 │  9083 │            0.00 │
└─────────────┴─────────────┴──────────────┴─────────────┴───────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┴────────────────────┴───────┴─────────────────┘

```

### 在拥抱脸部时使用句子转换器
https://huggingface.co/docs/hub/sentence-transformers.md