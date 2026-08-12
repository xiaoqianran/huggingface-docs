<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 执行向量相似度搜索

DuckDB 0.10.0 版本中添加了固定长度数组功能。这使您可以在 DuckDB 表中使用向量嵌入，使您的数据分析更加强大。

此外，还引入了 array_cosine_similarity 函数。该函数测量两个向量之间角度的余弦，表明它们的相似性。值 1 表示它们完全对齐，0 表示它们垂直，-1 表示它们完全相反。

让我们探讨一下如何使用此函数进行相似性搜索。在本节中，我们将向您展示如何使用 DuckDB 执行相似性搜索。

我们将使用 [asoria/awesome-chatgpt-prompts-embeddings](https://huggingface.co/datasets/asoria/awesome-chatgpt-prompts-embeddings) 数据集。

首先，让我们预览一下数据集中的一些记录：

```bash
FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet' SELECT act, prompt, len(embedding) as embed_len LIMIT 3;

┌──────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬───────────┐
│         act          │                                                                                    prompt                                                                                    │ embed_len │
│       varchar        │                                                                                   varchar                                                                                    │   int64   │
├──────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼───────────┤
│ Linux Terminal       │ I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output insid…  │       384 │
│ English Translator…  │ I want you to act as an English translator, spelling corrector and improver. I will speak to you in any language and you will detect the language, translate it and answer…  │       384 │
│ `position` Intervi…  │ I want you to act as an interviewer. I will be the candidate and you will ask me the interview questions for the `position` position. I want you to only reply as the inte…  │       384 │
└──────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴───────────┘

```

接下来，让我们选择用于相似性搜索的嵌入：

```bash
FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet' SELECT  embedding  WHERE act = 'Linux Terminal';

┌─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                                    embedding                                                                                                    │
│                                                                                                     float[]                                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ [-0.020781303, -0.029143505, -0.0660217, -0.00932716, -0.02601602, -0.011426172, 0.06627567, 0.11941507, 0.0013917526, 0.012889079, 0.053234346, -0.07380514, 0.04871567, -0.043601237, -0.0025319182, 0.0448…  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

```

现在，让我们使用选定的嵌入来查找相似的记录：

```bash
SELECT act,
       prompt,
       array_cosine_similarity(embedding::float[384], (SELECT embedding FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet' WHERE  act = 'Linux Terminal')::float[384]) AS similarity 
FROM 'hf://datasets/asoria/awesome-chatgpt-prompts-embeddings/data/*.parquet'
ORDER BY similarity DESC
LIMIT 3;

┌──────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┬────────────┐
│         act          │                                                                                   prompt                                                                                    │ similarity │
│       varchar        │                                                                                   varchar                                                                                   │   float    │
├──────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┼────────────┤
│ Linux Terminal       │ I want you to act as a linux terminal. I will type commands and you will reply with what the terminal should show. I want you to only reply with the terminal output insi…  │        1.0 │
│ JavaScript Console   │ I want you to act as a javascript console. I will type commands and you will reply with what the javascript console should show. I want you to only reply with the termin…  │  0.7599728 │
│ R programming Inte…  │ I want you to act as a R interpreter. I'll type commands and you'll reply with what the terminal should show. I want you to only reply with the terminal output inside on…  │  0.7303775 │
└──────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┴────────────┘

```

就是这样！您已使用 DuckDB 成功执行了矢量相似性搜索。

### 在拥抱脸部时使用`Transformers.js`
https://huggingface.co/docs/hub/transformers-js.md