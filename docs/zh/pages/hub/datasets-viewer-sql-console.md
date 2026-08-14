<!-- huggingface-docs: machine-translated zh-CN from English source -->

# SQL Console：在浏览器中查询拥抱人脸数据集

您可以使用 SQL 控制台在浏览器中对数据集运行 SQL 查询。 SQL 控制台由 [DuckDB](https://duckdb.org/) WASM 提供支持，完全在浏览器中运行。您可以从 Data Studio 访问 SQL 控制台。

    
    

    要了解有关 SQL 控制台的更多信息，请参阅 SQL 控制台博客文章。

通过 SQL 控制台，您可以：

- 在数据集上运行[DuckDB SQL queries](https://duckdb.org/docs/sql/query_syntax/select)（_签出[SQL Snippets](https://huggingface.co/spaces/cfahlgren1/sql-snippets)以获取有用的查询_） 
- 通过链接与其他人分享查询结果（_check out [this example](https://huggingface.co/datasets/gretelai/synthetic-gsm8k-reflection-405b?sql_console=true&sql=FROM+histogram%28%0A++train%2C%0A++topic%2C%0A++bin_count+%3A%3D+10%0A%29)_） 
- 将查询结果下载到 Parquet 或 CSV 文件 
- 使用 iframe 将查询结果嵌入您自己的网页中 
- 使用自然语言查询数据集

> [!提示]
> 您还可以在本地通过CLI使用DuckDB通过`hf://`协议查询数据集。有关更多信息，请参阅 DuckDB 数据集文档。 SQL 控制台提供了一个方便的 `Copy to DuckDB CLI` 按钮，可生成 SQL 查询以在 DuckDB CLI 中创建视图并执行查询。或者，使用 [⟦T9⟧](/docs/huggingface_hub/package_reference/cli#hf-datasets-sql) 从终端或脚本运行相同的查询。

## 示例

### 过滤SQL 控制台使过滤数据集变得非常容易。例如，如果要过滤 `SkunkworksAI/reasoning-0.01` 数据集以获取推理长度至少为 10 的指令和响应，则可以使用以下查询：

    
    

这是按推理长度排序的 SQL
```sql
SELECT *
FROM train
WHERE LENGTH(reasoning_chains) > 10;
```

### 直方图

许多数据集作者选择包含有关数据集中数据分布的统计数据。使用 DuckDB `histogram` 函数，我们可以绘制列值的直方图。

例如，要绘制 [Lichess/chess-puzzles](https://huggingface.co/datasets/Lichess/chess-puzzles) 数据集中的 `Rating` 列的直方图，您可以使用以下查询：

    
    

    在此处了解有关 `histogram` 功能和参数的更多信息。

```sql
from histogram(train, Rating)
```

### 正则表达式匹配

DuckDB 最强大的功能之一是对正则表达式的深度支持。您可以使用 `regexp` 函数来匹配数据中的模式。

 使用[regexp_matches](https://duckdb.org/docs/sql/functions/char.html#regexp_matchesstring-pattern)函数，我们可以过滤[GeneralReasoning/GeneralThought-195k](https://huggingface.co/datasets/GeneralReasoning/GeneralThought-195K)数据集以查找包含Markdown代码块的指令。

 
    
    

    在此处了解有关 DuckDB 正则表达式函数的更多信息。

```sql
SELECT *
FROM train
WHERE regexp_matches(model_answer, '```')
限制 10；
```

### Saved Queries and Embeds API

You can create, update, and delete SQL Console embeds programmatically. Embeds are saved queries that can be shared via link or embedded in other pages.

**Create an embed:**
```
POST /api/datasets/{namespace}/{repo}/sql-console/embed
内容类型：application/json
授权：持有者{token}{
  "sql": "从火车 LIMIT 10 中选择 *",
  "title": "示例行",
  “私人”：假，
  “视图”：[{“键”：“默认/火车”，“显示名称”：“火车”，“视图名称”：“火车”}]
}
```

**Update an embed:**
```
补丁 /api/datasets/{namespace}/{repo}/sql-console/embed/{embed_id}
内容类型：application/json
授权：持有者{token}

{
  "sql": "SELECT * FROM train LIMIT 20",
  "title": "更新后的标题",
  “私人”：真实
}
```

**Delete an embed:**
```
删除 /api/datasets/{namespace}/{repo}/sql-console/embed/{embed_id}
授权：持有者{token}
```

### Leakage Detection

Leakage detection is the process of identifying whether data in a dataset is present in multiple splits, for example, whether the test set is present in the training set.

<div class="flex justify-center">
    <img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/datastudio-leakage.png"/>
    <img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/datastudio-leakage-dark.png"/>
</div>

<p class="text-sm text-center italic">
    Learn more about leakage detection <a href="https://huggingface.co/blog/lbourdois/lle">here</a>.
</p>

```sql
与
    重叠行 AS (
        选择合并(
            (SELECT COUNT(*) AS 重叠计数
             从火车出发
             相交
             SELECT COUNT(*) AS 重叠计数
             来自测试），
            0
        ) AS 重叠计数
    ),
    总唯一行数 AS (
        选择 COUNT(*) AS 总计数
        来自（
            从火车中选择 *
            联盟
            从测试中选择 *
        ) 合并
    ）
选择
    重叠计数，
    总数，
    案例 
        当总计数 > 0 时，则（重叠计数 * 100.0 / 总计数）
        其他 0
    END AS 重叠百分比
FROM 重叠行数、总唯一行数；
````### 在 Hugging Face 中使用 spaCy
https://huggingface.co/docs/hub/spacy.md