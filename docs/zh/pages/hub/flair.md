<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用天赋

[Flair](https://github.com/flairNLP/flair) 是一个非常简单的最先进的 NLP 框架。
由[Humboldt University of Berlin](https://www.informatik.hu-berlin.de/en/forschung-en/gebiete/ml-en/)和朋友开发。

## 探索中心的天赋

您可以通过[models page](https://huggingface.co/models?library=flair)左侧筛选找到`flair`型号。

Hub 上的所有型号都具有以下有用的功能：

1. 自动生成的模型卡，附有简要说明。
2. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
3. 推理提供程序小部件，允许您发出推理请求。

## 安装

首先，您可以按照[Flair installation guide](https://github.com/flairNLP/flair?tab=readme-ov-file#requirements-and-installation)进行操作。
您还可以通过 pip 使用以下一行安装：

```
$ pip install -U flair
```

## 使用现有模型

所有 `flair` 模型都可以轻松地从 Hub 加载：

```py
from flair.data import Sentence
from flair.models import SequenceTagger

# load tagger
tagger = SequenceTagger.load("flair/ner-multi")
```

加载后，您可以使用`predict()`进行推理：

```py
sentence = Sentence("George Washington ging nach Washington.")
tagger.predict(sentence)

# print sentence
print(sentence)
```

它输出以下内容：

```text
Sentence[6]: "George Washington ging nach Washington." → ["George Washington"/PER, "Washington"/LOC]
```

如果您想加载特定的 Flair 模型，您可以单击模型卡中的`Use in Flair`，您将获得一个工作片段！

## 其他资源

*天赋[repository](https://github.com/flairNLP/flair)
* 天赋[docs](https://flairnlp.github.io/docs/intro)
* Hub上的官方天赋[models](https://huggingface.co/flair)（主要由[@alanakbik](https://huggingface.co/alanakbik)和[@stefan-it](https://huggingface.co/stefan-it)训练）

### 执行向量相似度搜索
https://huggingface.co/docs/hub/datasets-duckdb-vector-similarity-search.md