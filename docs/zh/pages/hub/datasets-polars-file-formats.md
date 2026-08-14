<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 文件格式

从 Hugging Face 读取时，Polars 支持以下文件格式：

- [Parquet](https://docs.pola.rs/api/python/stable/reference/api/polars.read_parquet.html)
- [CSV](https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html)
- [JSON Lines](https://docs.pola.rs/api/python/stable/reference/api/polars.read_ndjson.html)

下面的示例仅显示默认设置。使用上面的链接查看 API 参考指南中的所有可用参数。

# 镶木地板

Parquet 是首选文件格式，因为它在文件中存储具有类型信息的架构。这可以避免解析时出现任何歧义并加快阅读速度。要在 Polars 中读取 Parquet 文件，请使用 `read_parquet` 函数：

```python
pl.read_parquet("hf://datasets/roneneldan/TinyStories/data/train-00000-of-00004-2d5a1467fff1081b.parquet")
```

# CSV

`read_csv`函数可用于读取CSV文件：

```python
pl.read_csv("hf://datasets/lhoestq/demo1/data/train.csv")
```

# JSON

Polars 支持使用 `read_ndjson` 函数读取新行分隔的 JSON（也称为 [json lines](https://jsonlines.org/)）：

```python
pl.read_ndjson("hf://datasets/proj-persona/PersonaHub/persona.jsonl")
```

### 在拥抱脸部使用 PaddleNLP
https://huggingface.co/docs/hub/paddlenlp.md