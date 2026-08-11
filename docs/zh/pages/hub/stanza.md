<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 Stanza

`stanza`是一系列准确高效的工具集合，用于对多种人类语言进行语言分析。从原始文本到句法分析和实体识别，Stanza 为您选择的语言带来了最先进的 NLP 模型。

## 探索 Hub 中的节

您可以通过[models page](https://huggingface.co/models?library=stanza&sort=downloads)左侧筛选找到`stanza`型号。您可以找到 70 多种不同语言的模型！

Hub 上的所有型号均具有以下功能：
1. 自动生成的模型卡，其中包含有助于发现的简短描述和元数据标签。
2. 一个交互式小部件，您可以使用它直接在浏览器中运行模型（用于命名实体识别和词性）。
3. 推理提供程序小部件，允许发出推理请求（用于命名实体识别和词性）。

## 使用现有模型

`stanza` 库自动从 Hub 下载模型。您可以使用`stanza.Pipeline`从Hub下载模型并进行推理。

```python
import stanza

nlp = stanza.Pipeline('en') # download th English model and initialize an English neural pipeline
doc = nlp("Barack Obama was born in Hawaii.") # run annotation over a sentence
```

## 分享你的模型要添加新的官方 Stanza 模型，您可以按照流程进行[add a new language](https://stanfordnlp.github.io/stanza/new_language.html)，然后[share your models with the Stanza team](https://stanfordnlp.github.io/stanza/new_language.html#contributing-back-to-stanza)。您还可以找到将模型上传到 Hub [here](https://github.com/stanfordnlp/huggingface-models/blob/main/hugging_stanza.py) 的官方脚本。

## 其他资源

*`stanza`[docs](https://stanfordnlp.github.io/stanza/)。

### Webhooks 自动化
https://huggingface.co/docs/hub/jobs-webhooks.md