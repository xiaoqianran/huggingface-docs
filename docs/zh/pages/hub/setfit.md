<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 SetFit 和拥抱脸部

SetFit 是一个高效且无提示的框架，用于对 [Sentence Transformers](https://sbert.net/) 进行少量微调。它用很少的标记数据实现了高精度 - 例如，在客户评论情感数据集上每类只有 8 个标记示例，SetFit 在 3000 个示例的完整训练集上与微调 RoBERTa Large 相比具有竞争力！

与其他few-shot学习方法相比，SetFit有几个独特的特点：

* 🗣 **没有提示或语言器：** 当前的小样本微调技术需要手工制作的提示或语言器将示例转换为适合底层语言模型的格式。 SetFit 直接从文本示例生成丰富的嵌入，从而完全省去了提示。
* 🏎 **快速训练：** SetFit 不需要像 [T0](https://huggingface.co/bigscience/T0) 或 GPT-3 这样的大规模模型来实现高精度。因此，训练和运行推理的速度通常要快一个数量级（或更多）。
* 🌎 **多语言支持**：SetFit 可以与 Hub 上的任何[Sentence Transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads) 一起使用，这意味着您只需微调多语言检查点即可对多种语言的文本进行分类。

## 探索 Hub 上的 SetFit您可以通过[models page](https://huggingface.co/models?library=setfit)左侧筛选找到SetFit型号。

Hub 上的所有型号都具有以下有用的功能：
1. 自动生成的模型卡，附有简要说明。
2. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
3. 推理提供程序小部件，允许您发出推理请求。

## 安装

首先，您可以按照[SetFit installation guide](https://huggingface.co/docs/setfit/installation)进行操作。您还可以通过 pip 使用以下一行安装：

```
pip install -U setfit
```

## 使用现有模型

所有`setfit`模型都可以轻松地从集线器加载。

```py
from setfit import SetFitModel

model = SetFitModel.from_pretrained("tomaarsen/setfit-paraphrase-mpnet-base-v2-sst2-8-shot")
```

加载后，您可以使用[⟦T5⟧](https://huggingface.co/docs/setfit/reference/main#setfit.SetFitModel.predict)进行推理。

```py
model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
```
```bash
['positive', 'negative']
```

如果您想加载特定的SetFit模型，您可以单击`Use in SetFit`，您将获得一个工作片段！

## 其他资源
* [All SetFit models available on the Hub](https://huggingface.co/models?library=setfit)
* 设置适合[repository](https://github.com/huggingface/setfit)
* 设置适合[docs](https://huggingface.co/docs/setfit)
* 设置适合[paper](https://arxiv.org/abs/2209.11055)

### 极地
https://huggingface.co/docs/hub/datasets-polars.md