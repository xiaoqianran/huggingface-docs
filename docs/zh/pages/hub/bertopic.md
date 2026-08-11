<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 BERTopic

[BERTopic](https://github.com/MaartenGr/BERTopic) 是一个主题建模框架，它利用 🤗 转换器和 c-TF-IDF 创建密集的集群，允许轻松解释主题，同时在主题描述中保留重要的单词。 

BERTopic 支持各种主题建模技术：  

  
    引导
    监督
    半监督
 
   
    手册
    多主题分布
    分层的
 
 
    基于类别
    动态
    在线/增量
 
 
    多式联运
    多方面
    文本生成/法学硕士
 
 
    零射击（新！）
    合并模型（新！）
    种子词（新！）
 

## 探索 Hub 上的 BERTopic

您可以通过[models page](https://huggingface.co/models?library=bertopic&sort=trending)左侧的过滤找到BERTopic模型。

Hub 上托管的 BERTopic 模型有一个模型卡，其中包含有关模型的有用信息。由于 BERTopic Hugging Face Hub 集成，您可以使用几行代码加载 BERTopic 模型。您还可以使用[Inference Endpoints](https://huggingface.co/inference-endpoints)部署这些模型。

## 安装

首先，您可以按照[BERTopic installation guide](https://github.com/MaartenGr/BERTopic#installation)进行操作。
您还可以通过 pip 使用以下一行安装：

```bash
pip install bertopic
```

## 使用现有模型

所有 BERTopic 模型都可以轻松地从 Hub 加载：

```py
from bertopic import BERTopic
topic_model = BERTopic.load("MaartenGr/BERTopic_Wikipedia")
```加载后，您可以使用 BERTopic 的功能来预测新实例的主题：

```py
topic, prob = topic_model.transform("This is an incredible movie!")
topic_model.topic_labels_[topic]
```

这给了我们以下主题：

```text
64_rating_rated_cinematography_film
```

## 共享模型

创建 BERTopic 模型后，您可以通过 Hugging Face Hub 轻松与其他人共享。为此，我们可以利用 `push_to_hf_hub` 函数，它允许我们直接将模型推送到 Hugging Face Hub：

```python
from bertopic import BERTopic

# Train model
topic_model = BERTopic().fit(my_docs)

# Push to HuggingFace Hub
topic_model.push_to_hf_hub(
    repo_id="MaartenGr/BERTopic_ArXiv",
    save_ctfidf=True
)
```

请注意，保存的模型不包括降维和聚类算法。这些被删除，因为它们只需要训练模型和查找相关主题。推理是通过主题和文档嵌入之间的直接余弦相似度来完成的。这不仅加快了模型的速度，而且使我们能够拥有一个可以使用的小型 BERTopic 模型。

## 其他资源

* [BERTopic repository](https://github.com/MaartenGr/BERTopic)
* [BERTopic docs](https://maartengr.github.io/BERTopic/)
* [BERTopic models in the Hub](https://huggingface.co/models?library=bertopic&sort=trending)

### 如何使用 Okta 配置 SCIM
https://huggingface.co/docs/hub/security-sso-okta-scim.md