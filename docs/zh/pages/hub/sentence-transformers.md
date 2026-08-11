<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用句子转换器

`sentence-transformers` 是一个库，提供简单的方法来计算句子、段落和图像的嵌入（密集向量表示）。文本嵌入到向量空间中，使得相似文本接近，从而实现语义搜索、聚类和检索等应用。 

## 探索 Hub 中的句子转换器

通过[models page](https://huggingface.co/models?library=sentence-transformers&sort=downloads)左侧筛选，您可以找到超过50000个`sentence-transformer`型号。这些模型大多数都支持不同的任务，例如使用 [⟦T5⟧](https://huggingface.co/models?library=sentence-transformers&pipeline_tag=feature-extraction&sort=downloads) 来生成嵌入，以及使用 [⟦T6⟧](https://huggingface.co/models?library=sentence-transformers&pipeline_tag=sentence-similarity&sort=downloads) 来确定给定句子与其他句子的相似程度。您还可以在[the official docs](https://www.sbert.net/docs/pretrained_models.html)找到官方预训练模型的概述。

Hub 上的所有型号均具有以下功能：
1. 自动生成的模型卡，其中包含描述、示例代码片段、架构概述等。 
2. 有助于发现并包含许可证等信息的元数据标签。
3. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
4. 推理提供程序小部件，允许发出推理请求。

## 使用现有模型只需一行代码即可加载 Hub 上预训练的模型

```py
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('model_name')
```

这是一个对句子进行编码，然后计算它们之间的距离以进行语义搜索的示例。

```py
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

query_embedding = model.encode('How big is London')
passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its financial district'])

print("Similarity:", util.dot_score(query_embedding, passage_embedding))
```

如果您想了解如何加载特定模型，可以单击`Use in sentence-transformers`，您将获得一个可以加载它的工作片段！ 

## 分享你的模型

您可以使用经过训练的模型中的 `save_to_hub` 方法来共享您的句子转换器。

```py
from sentence_transformers import SentenceTransformer

# Load or train a model
model.save_to_hub("my_new_model")
```

此命令创建一个存储库，其中包含自动生成的模型卡、推理小部件、示例代码片段等！ [Here](https://huggingface.co/osanseviero/my_new_model) 就是一个例子。

## 其他资源

* 句子变形金刚[library](https://github.com/UKPLab/sentence-transformers)。
* 句子变形金刚[docs](https://www.sbert.net/)。
* 与集线器[announcement](https://huggingface.co/blog/sentence-transformers-in-the-hub)集成。

### 空间闪亮
https://huggingface.co/docs/hub/spaces-sdks-docker-shiny.md