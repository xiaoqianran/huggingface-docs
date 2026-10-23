<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 SpanMarker

[SpanMarker](https://github.com/tomaarsen/SpanMarkerNER) 是一个使用熟悉的编码器（例如 BERT、RoBERTa 和 DeBERTa）训练强大的命名实体识别模型的框架。 SpanMarker 紧密实现在 🤗 Transformers 库之上，可以很好地利用它。因此，熟悉 Transformer 的任何人都可以直观地使用 SpanMarker。

## 在 Hub 中探索 SpanMarker

您可以通过[models page](https://huggingface.co/models?library=span-marker)左侧筛选找到`span_marker`型号。

Hub 上的所有型号都具有以下有用的功能：
1. 自动生成的模型卡，附有简要说明。
2. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
3. 推理提供程序小部件，允许您发出推理请求。

## 安装

首先，您可以按照[SpanMarker installation guide](https://tomaarsen.github.io/SpanMarkerNER/install.html)进行操作。您还可以通过 pip 使用以下一行安装：

```
pip install -U span_marker
```

## 使用现有模型

所有`span_marker`模型都可以轻松地从集线器加载。

```py
from span_marker import SpanMarkerModel

model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-bert-base-fewnerd-fine-super")
```

加载后，您可以使用[⟦T6⟧](https://tomaarsen.github.io/SpanMarkerNER/api/span_marker.modeling.html#span_marker.modeling.SpanMarkerModel.predict)进行推理。

```py
model.predict("Amelia Earhart flew her single engine Lockheed Vega 5B across the Atlantic to Paris.")
```
```json
[
    {"span": "Amelia Earhart", "label": "person-other", "score": 0.7629689574241638, "char_start_index": 0, "char_end_index": 14},
    {"span": "Lockheed Vega 5B", "label": "product-airplane", "score": 0.9833564758300781, "char_start_index": 38, "char_end_index": 54},
    {"span": "Atlantic", "label": "location-bodiesofwater", "score": 0.7621214389801025, "char_start_index": 66, "char_end_index": 74},
    {"span": "Paris", "label": "location-GPE", "score": 0.9807717204093933, "char_start_index": 78, "char_end_index": 83}
]
```

如果您想加载特定的 SpanMarker 模型，您可以单击`Use in SpanMarker`，您将获得一个工作片段！

## 其他资源

* 跨度标记[repository](https://github.com/tomaarsen/SpanMarkerNER)
* 跨度标记[docs](https://tomaarsen.github.io/SpanMarkerNER)### 模型卡指南
https://huggingface.co/docs/hub/model-card-guidebook.md