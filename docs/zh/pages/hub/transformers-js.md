<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用`Transformers.js`

Transformers.js 是一个 JavaScript 库，用于直接在浏览器中运行 🤗 Transformers，无需服务器！它的设计在功能上与原始 [Python library](https://github.com/huggingface/transformers) 相同，这意味着您可以使用非常相似的 API 运行相同的预训练模型。

## 在 Hub 中探索 `transformers.js`

您可以通过[models page](https://huggingface.co/models?library=transformers.js)中的库过滤找到`transformers.js`型号。

## 快速浏览

从现有代码翻译起来非常简单！就像 Python 库一样，我们支持 `pipeline` API。管道将预训练模型与输入预处理和输出后处理组合在一起，使其成为使用库运行模型的最简单方法。

Python（原始）
JavaScript（我们的）

```python
from transformers import pipeline

# Allocate a pipeline for sentiment-analysis
pipe = pipeline('sentiment-analysis')

out = pipe('I love transformers!')
# [{'label': 'POSITIVE', 'score': 0.999806941}]
```

```javascript
import { pipeline } from '@huggingface/transformers';

// Allocate a pipeline for sentiment-analysis
let pipe = await pipeline('sentiment-analysis');

let out = await pipe('I love transformers!');
// [{'label': 'POSITIVE', 'score': 0.999817686}]
```

您还可以通过指定模型 ID 或路径作为 `pipeline` 函数的第二个参数来使用不同的模型。例如：
```javascript
// Use a different model for sentiment-analysis
let pipe = await pipeline('sentiment-analysis', 'nlptown/bert-base-multilingual-uncased-sentiment');
```

有关支持的任务和模型的完整列表，请参阅[documentation](https://huggingface.co/docs/transformers.js)。

## 安装

要通过 [NPM](https://www.npmjs.com/package/@huggingface/transformers) 安装，请运行：
```bash
npm i @huggingface/transformers
```

有关更多信息，包括如何通过 CDN 或静态托管在 vanilla JS（无需任何捆绑程序）中使用它，请参阅 [README](https://github.com/huggingface/transformers.js/blob/main/README.md#installation)。

## 其他资源

* Transformers.js [repository](https://github.com/huggingface/transformers.js)
* Transformers.js [docs](https://huggingface.co/docs/transformers.js)
* Transformers.js [demo](https://huggingface.github.io/transformers.js/)### 五十一
https://huggingface.co/docs/hub/datasets-fiftyone.md