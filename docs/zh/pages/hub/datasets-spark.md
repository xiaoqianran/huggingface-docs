<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 火花

Spark 能够在分布式环境中实现实时、大规模的数据处理。

您可以使用 `pyspark_huggingface` 通过“huggingface”数据源访问 PySpark 中的 Hugging Face 数据集存储库。

在 Hugging Face Spaces 上尝试 [Spark Notebooks](https://huggingface.co/spaces/Dataset-Tools/Spark-Notebooks) 以获得预装 PySpark 和 `pyspark_huggingface` 的笔记本。

## 设置

### 安装

为了能够读取和写入拥抱人脸数据集，您需要安装 `pyspark_huggingface` 库：

```
pip install pyspark_huggingface
```

这还将安装所需的依赖项，例如用于身份验证的`huggingface_hub`，以及用于读取和写入数据集的`pyarrow`。

### 身份验证

您需要向 Hugging Face 进行身份验证才能读取私有/门控数据集存储库或写入数据集存储库。

您可以使用 CLI，例如：

```
hf auth login
```

还可以为您的 Hugging Face 令牌提供 `HF_TOKEN` 环境变量或将 `token` 选项传递给阅读器。
有关身份验证的更多详细信息，请查看[this guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)。

### 启用“huggingface”数据源

PySpark 4 附带了一个新的数据源 API，允许使用来自自定义源的数据集。
如果安装了 `pyspark_huggingface`，PySpark 会自动导入它并启用“huggingface”数据源。该库还向后移植了 PySpark 3.5、3.4 和 3.3 的“huggingface”数据源的数据源 API。
然而在这种情况下，应该显式导入 `pyspark_huggingface` 以激活反向移植并启用“huggingface”数据源：

```python
>>> import pyspark_huggingface
huggingface datasource enabled for pyspark 3.x.x (backport from pyspark 4)
```

## 阅读

“huggingface”数据源允许从 Hugging Face 读取数据集，并使用 `pyarrow` 来传输 Arrow 数据。
这与 Hugging Face 上[supported format](https://huggingface.co/docs/hub/datasets-adding#file-formats) 中的所有数据集兼容，例如 Parquet 数据集。

例如，以下是如何加载 [stanfordnlp/imdb](https://huggingface.co/stanfordnlp/imdb) 数据集：

```python
>>> import pyspark_huggingface
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.appName("demo").getOrCreate()
>>> df = spark.read.format("huggingface").load("stanfordnlp/imdb")
```

这是 [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) 数据集的另一个示例。
它是一个封闭的存储库，用户在访问它之前必须接受使用条款。
它还具有多个子集，即“3M”和“7M”。所以我们需要指定要加载哪一个。

    
    

我们使用`.format()`函数来使用“huggingface”数据源，并使用`.load()`来加载数据集（更准确地说是名为“7M”的配置或子集，包含7M样本）。然后我们计算每种语言的对话数量并过滤数据集。

登录访问门控存储库后，我们可以运行：

```python
>>> import pyspark_huggingface
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.appName("demo").getOrCreate()
>>> df = spark.read.format("huggingface").option("config", "7M").load("BAAI/Infinity-Instruct")
>>> df.show()
+---+----------------------------+-----+----------+--------------------+        
| id|               conversations|label|langdetect|              source|
+---+----------------------------+-----+----------+--------------------+
|  0|        [{human, def exti...|     |        en|      code_exercises|
|  1|        [{human, See the ...|     |        en|                flan|
|  2|        [{human, This is ...|     |        en|                flan|
|  3|        [{human, If you d...|     |        en|                flan|
|  4|        [{human, In a Uni...|     |        en|                flan|
|  5|        [{human, Read the...|     |        en|                flan|
|  6|        [{human, You are ...|     |        en|          code_bagel|
|  7|        [{human, I want y...|     |        en|          Subjective|
|  8|        [{human, Given th...|     |        en|                flan|
|  9|[{human, 因果联系原则是法...|     |     zh-cn|          Subjective|
| 10|        [{human, Provide ...|     |        en|self-oss-instruct...|
| 11|        [{human, The univ...|     |        en|                flan|
| 12|        [{human, Q: I am ...|     |        en|                flan|
| 13|        [{human, What is ...|     |        en|      OpenHermes-2.5|
| 14|        [{human, In react...|     |        en|                flan|
| 15|        [{human, Write Py...|     |        en|      code_exercises|
| 16|        [{human, Find the...|     |        en|            MetaMath|
| 17|        [{human, Three of...|     |        en|            MetaMath|
| 18|        [{human, Chandra ...|     |        en|            MetaMath|
| 19|[{human, 用经济学知识分析...|     |     zh-cn|          Subjective|
+---+----------------------------+-----+----------+--------------------+
```这会以流方式加载数据集，并且输出 DataFrame 数据集中的每个数据文件都有一个分区，以实现高效的分布式处理。

为了计算每种语言的对话数量，我们运行使用 `columns` 选项和 `groupBy()` 操作的代码。
`columns` 选项对于仅加载我们需要的数据很有用，因为 PySpark 不支持使用数据源 API 进行谓词下推。
还有一个 `filters` 选项，仅加载特定范围内值的数据。

```python
>>> df_langdetect_only = (
...     spark.read.format("huggingface")
...     .option("config", "7M")
...     .option("columns", '["langdetect"]')
...     .load("BAAI/Infinity-Instruct")
... )
>>> df_langdetect_only.groupBy("langdetect").count().show()
+----------+-------+                                                            
|langdetect|  count|
+----------+-------+
|        en|6697793|
|     zh-cn| 751313|
+----------+-------+
```

过滤数据集并仅保留中文对话：

```python
>>> df_chinese_only = (
...     spark.read.format("huggingface")
...     .option("config", "7M")
...     .option("filters", '[("langdetect", "=", "zh-cn")]')
...     .load("BAAI/Infinity-Instruct")
... )
>>> df_chinese_only.show()
+---+----------------------------+-----+----------+----------+                  
| id|               conversations|label|langdetect|    source|
+---+----------------------------+-----+----------+----------+
|  9|[{human, 因果联系原则是法...|     |     zh-cn|Subjective|
| 19|[{human, 用经济学知识分析...|     |     zh-cn|Subjective|
| 38| [{human, 某个考试共有A、...|     |     zh-cn|Subjective|
| 39|[{human, 撰写一篇关于斐波...|     |     zh-cn|Subjective|
| 57|[{human, 总结世界历史上的...|     |     zh-cn|Subjective|
| 61|[{human, 生成一则广告词。...|     |     zh-cn|Subjective|
| 66|[{human, 描述一个有效的团...|     |     zh-cn|Subjective|
| 94|[{human, 如果比利和蒂芙尼...|     |     zh-cn|Subjective|
|102|[{human, 生成一句英文名言...|     |     zh-cn|Subjective|
|106|[{human, 写一封感谢信，感...|     |     zh-cn|Subjective|
|118| [{human, 生成一个故事。}...|     |     zh-cn|Subjective|
|174|[{human, 高胆固醇水平的后...|     |     zh-cn|Subjective|
|180|[{human, 基于以下角色信息...|     |     zh-cn|Subjective|
|192|[{human, 请写一篇文章，概...|     |     zh-cn|Subjective|
|221|[{human, 以诗歌形式表达对...|     |     zh-cn|Subjective|
|228|[{human, 根据给定的指令，...|     |     zh-cn|Subjective|
|236|[{human, 打开一个新的生成...|     |     zh-cn|Subjective|
|260|[{human, 生成一个有关未来...|     |     zh-cn|Subjective|
|268|[{human, 如果有一定数量的...|     |     zh-cn|Subjective|
|273| [{human, 题目：小明有5个...|     |     zh-cn|Subjective|
+---+----------------------------+-----+----------+----------+
```

还可以在加载的 DataFrame 上应用过滤器或删除列，但在加载时执行此操作会更有效，尤其是在 Parquet 数据集上。
事实上，Parquet 包含文件和行组级别的元数据，这允许跳过不包含满足条件的样本的数据集的整个部分。 Parquet 中的列也可以独立加载，这允许跳过排除的列并避免加载不必要的数据。

### 选项

以下是您可以传递给 `read..option()` 的可用选项列表：* `config`（字符串）：选择数据集子集/配置
* `split`（字符串）：选择数据集分割（默认为“train”）
* `token`（字符串）：你的拥抱脸令牌

您可以选择手动加载哪些文件，而不是指定配置或拆分：

* `data_dir`（字符串）：选择目录
* `data_files`（字符串）：选择一个或多个文件，例如`"data/*.parquet"` 或 `'["part1.parquet", "par2.parquet"]'`

对于 Parquet 数据集：

* `columns`（字符串）：选择要加载的列子集，例如`'["id"]'`
* `filters`（字符串）：跳过不符合条件的文件和行组，例如`'[("source", "=", "code_exercises")]'`。过滤器被传递到[pyarrow.parquet.ParquetDataset](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetDataset.html)。

任何其他选项都作为参数传递给 [datasets.load_dataset] (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset)

### 运行 SQL 查询

准备好 PySpark Dataframe 后，您可以使用 `spark.sql` 运行 SQL 查询：

```python
>>> import pyspark_huggingface
>>> from pyspark.sql import SparkSession
>>> spark = SparkSession.builder.appName("demo").getOrCreate()
>>> df = (
...     spark.read.format("huggingface")
...     .option("config", "7M")
...     .option("columns", '["source"]')
...     .load("BAAI/Infinity-Instruct")
... )
>>> spark.sql("SELECT source, count(*) AS total FROM {df} GROUP BY source ORDER BY total DESC", df=df).show()
+--------------------+-------+
|              source|  total|
+--------------------+-------+
|                flan|2435840|
|          Subjective|1342427|
|      OpenHermes-2.5| 855478|
|            MetaMath| 690138|
|      code_exercises| 590958|
|Orca-math-word-pr...| 398168|
|          code_bagel| 386649|
|        MathInstruct| 329254|
|python-code-datas...|  88632|
|instructional_cod...|  82920|
|        CodeFeedback|  79513|
|self-oss-instruct...|  50467|
|Evol-Instruct-Cod...|  43354|
|CodeExercise-Pyth...|  27159|
|code_instructions...|  23130|
|  Code-Instruct-700k|  10860|
|Glaive-code-assis...|   9281|
|python_code_instr...|   2581|
|Python-Code-23k-S...|   2297|
+--------------------+-------+
```

同样，指定 `columns` 选项不是必需的，但对于避免加载不必要的数据并使查询更快很有用。

## 写

您可以使用“huggingface”数据源将 PySpark Dataframe 写入 Hugging Face。
它以分布式方式并行上传 Parquet 文件，并且仅在文件全部上传后才提交文件。
它的工作原理如下：

```python
>>> import pyspark_huggingface
>>> df.write.format("huggingface").save("username/dataset_name")
```以下是我们如何使用此函数将 [BAAI/Infinity-Instruct](https://huggingface.co/datasets/BAAI/Infinity-Instruct) 数据集的过滤版本写回 Hugging Face。

首先你需要[create a dataset repository](https://huggingface.co/new-dataset)，例如`username/Infinity-Instruct-Chinese-Only`（如果您愿意，可以将其设置为私人）。
然后，确保您已通过身份验证，并且可以使用“huggingface”数据源，将 `mode` 设置为“覆盖”（如果要扩展现有数据集，则设置为“追加”），然后使用 `.save()` 推送到 Hugging Face：

```python
>>> df_chinese_only.write.format("huggingface").mode("overwrite").save("username/Infinity-Instruct-Chinese-Only")
```

    
    

### 模式

将数据集推送到 Hugging Face 时有两种模式可用：

*“覆盖”：如果数据集已存在，则覆盖该数据集
* “append”：将数据集追加到现有数据集

### 选项

以下是您可以传递给 `write.option()` 的可用选项列表：

* `token`（字符串）：你的拥抱脸令牌

欢迎贡献以在此处添加更多选项，特别是`subset`和`split`。

## 存储桶

在数据集存储库中发布 AI 就绪数据之前，通常会在 [Storage Buckets](/docs/hub/storage-buckets) 中处理原始数据并进行实验。

访问存储桶的方式与数据集存储库相同，但使用 `buckets/` 前缀以及 `data_dir` 或 `data_files` 选项：

```python
>>> df = spark.read.format("huggingface").option("data_dir", "data").load("buckets/username/my-bucket")
>>> # OR with a glob pattern
>>> # df = spark.read.format("huggingface").option("data_files", "data/*.parquet").load("buckets/username/my-bucket")
>>> df.write.format("huggingface").option("data_dir", "new-data").save("buckets/username/my-bucket")
```

### 使用 GPG 签署提交
https://huggingface.co/docs/hub/security-gpg.md