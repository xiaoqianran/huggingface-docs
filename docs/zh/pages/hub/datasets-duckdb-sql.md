<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 执行SQL操作

使用 DuckDB 执行 SQL 操作为高效查询数据集开辟了无限可能。让我们深入研究一些展示 DuckDB 函数强大功能的示例。

在我们的演示中，我们将探索一个有趣的数据集。 [MMLU](https://huggingface.co/datasets/cais/mmlu)数据集是一个多任务测试，包含跨越各个知识领域的多项选择题。

要预览数据集，我们选择 3 行的样本：

```bash
FROM 'hf://datasets/cais/mmlu/all/test-*.parquet' USING SAMPLE 3;

┌──────────────────────┬──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┐
│       question       │       subject        │                                                                         choices                                                                          │ answer │
│       varchar        │       varchar        │                                                                        varchar[]                                                                         │ int64  │
├──────────────────────┼──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ The model of light…  │ conceptual_physics   │ [wave model, particle model, Both of these, Neither of these]                                                                                            │      1 │
│ A person who is lo…  │ professional_psych…  │ [his/her life scripts., his/her own feelings, attitudes, and beliefs., the emotional reactions and behaviors of the people he/she is interacting with.…  │      1 │
│ The thermic effect…  │ nutrition            │ [is substantially higher for carbohydrate than for protein, is accompanied by a slight decrease in body core temperature., is partly related to sympat…  │      2 │
└──────────────────────┴──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┘

```

此命令从数据集中检索 3 行的随机样本供我们检查。

让我们首先检查数据集的架构。下表概述了我们的数据集的结构：

```bash
DESCRIBE FROM 'hf://datasets/cais/mmlu/all/test-*.parquet' USING SAMPLE 3;
┌─────────────┬─────────────┬─────────┬─────────┬─────────┬─────────┐
│ column_name │ column_type │  null   │   key   │ default │  extra  │
│   varchar   │   varchar   │ varchar │ varchar │ varchar │ varchar │
├─────────────┼─────────────┼─────────┼─────────┼─────────┼─────────┤
│ question    │ VARCHAR     │ YES     │         │         │         │
│ subject     │ VARCHAR     │ YES     │         │         │         │
│ choices     │ VARCHAR[]   │ YES     │         │         │         │
│ answer      │ BIGINT      │ YES     │         │         │         │
└─────────────┴─────────────┴─────────┴─────────┴─────────┴─────────┘

```
接下来，我们来分析一下我们的数据集中是否有重复的记录：

```bash
SELECT   *,
         COUNT(*) AS counts
FROM     'hf://datasets/cais/mmlu/all/test-*.parquet'
GROUP BY ALL
HAVING   counts > 2; 

┌──────────┬─────────┬───────────┬────────┬────────┐
│ question │ subject │  choices  │ answer │ counts │
│ varchar  │ varchar │ varchar[] │ int64  │ int64  │
├──────────┴─────────┴───────────┴────────┴────────┤
│                      0 rows                      │
└──────────────────────────────────────────────────┘

```

幸运的是，我们的数据集不包含任何重复记录。

让我们看看条形表示中基于主题的问题的比例：

```bash
SELECT 
    subject, 
    COUNT(*) AS counts, 
    BAR(COUNT(*), 0, (SELECT COUNT(*) FROM 'hf://datasets/cais/mmlu/all/test-*.parquet')) AS percentage 
FROM 
    'hf://datasets/cais/mmlu/all/test-*.parquet' 
GROUP BY 
    subject 
ORDER BY 
    counts DESC;

┌──────────────────────────────┬────────┬────────────────────────────────────────────────────────────────────────────────┐
│           subject            │ counts │                                   percentage                                   │
│           varchar            │ int64  │                                    varchar                                     │
├──────────────────────────────┼────────┼────────────────────────────────────────────────────────────────────────────────┤
│ professional_law             │   1534 │ ████████▋                                                                      │
│ moral_scenarios              │    895 │ █████                                                                          │
│ miscellaneous                │    783 │ ████▍                                                                          │
│ professional_psychology      │    612 │ ███▍                                                                           │
│ high_school_psychology       │    545 │ ███                                                                            │
│ high_school_macroeconomics   │    390 │ ██▏                                                                            │
│ elementary_mathematics       │    378 │ ██▏                                                                            │
│ moral_disputes               │    346 │ █▉                                                                             │
├──────────────────────────────┴────────┴────────────────────────────────────────────────────────────────────────────────┤
│ 57 rows (8 shown)                                                                                           3 columns  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

现在，让我们准备包含与**营养**相关的问题的数据集子集，并创建问题到正确答案的映射。
请注意，我们有 **choices** 列，我们可以使用 **answer** 列作为索引从中获得正确答案。

```bash
SELECT *
FROM   'hf://datasets/cais/mmlu/all/test-*.parquet'
WHERE  subject = 'nutrition' LIMIT 3;

┌──────────────────────┬───────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────┐
│       question       │  subject  │                                                                               choices                                                                               │ answer │
│       varchar        │  varchar  │                                                                              varchar[]                                                                              │ int64  │
├──────────────────────┼───────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────┤
│ Which foods tend t…  │ nutrition │ [Meat, Confectionary, Fruits and vegetables, Potatoes]                                                                                                              │      2 │
│ In which one of th…  │ nutrition │ [If the incidence rate of the disease falls., If survival time with the disease increases., If recovery of the disease is faster., If the population in which the…  │      1 │
│ Which of the follo…  │ nutrition │ [The flavonoid class comprises flavonoids and isoflavonoids., The digestibility and bioavailability of isoflavones in soya food products are not changed by proce…  │      0 │
└──────────────────────┴───────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────┘

```

```bash
SELECT question,
       choices[answer] AS correct_answer
FROM   'hf://datasets/cais/mmlu/all/test-*.parquet'
WHERE  subject = 'nutrition' LIMIT 3;

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
│                                                              question                                                               │               correct_answer                │
│                                                               varchar                                                               │                   varchar                   │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Which foods tend to be consumed in lower quantities in Wales and Scotland (as of 2020)?\n                                           │ Confectionary                               │
│ In which one of the following circumstances will the prevalence of a disease in the population increase, all else being constant?\n │ If the incidence rate of the disease falls. │
│ Which of the following statements is correct?\n                                                                                     │                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

```为了确保数据的整洁，让我们删除问题末尾的所有换行符并过滤掉任何空答案：

```bash
SELECT regexp_replace(question, '\n', '') AS question,
       choices[answer] AS correct_answer
FROM   'hf://datasets/cais/mmlu/all/test-*.parquet'
WHERE  subject = 'nutrition' AND LENGTH(correct_answer) > 0 LIMIT 3;

┌───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬─────────────────────────────────────────────┐
│                                                             question                                                              │               correct_answer                │
│                                                              varchar                                                              │                   varchar                   │
├───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼─────────────────────────────────────────────┤
│ Which foods tend to be consumed in lower quantities in Wales and Scotland (as of 2020)?                                           │ Confectionary                               │
│ In which one of the following circumstances will the prevalence of a disease in the population increase, all else being constant? │ If the incidence rate of the disease falls. │
│ Which vitamin is a major lipid-soluble antioxidant in cell membranes?                                                             │ Vitamin D                                   │
└───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴─────────────────────────────────────────────┘

```

最后，让我们重点介绍本节中使用的一些 DuckDB 函数：
- `DESCRIBE`，返回表模式。
- `USING SAMPLE`，样本用于随机选择数据集的子集。
- `BAR`，绘制一条带，其宽度与 (x - min) 成正比，并且当 x = max 时等于宽度字符。宽度默认为 80。
- `string[begin:end]`，使用切片约定提取字符串。缺少 begin 或 end 参数分别被解释为列表的开头或结尾。接受负值。
- `regexp_replace`，如果字符串包含正则表达式模式，则用replacement替换匹配部分。
- `LENGTH`，获取字符串中的字符数。

> [!提示]
> DuckDB 的[SQL functions overview](https://duckdb.org/docs/sql/functions/overview) 中有很多有用的功能。最好的部分是您可以直接在 Hugging Face 数据集上使用它们。

### 🟧 空间标签工作室
https://huggingface.co/docs/hub/spaces-sdks-docker-label-studio.md