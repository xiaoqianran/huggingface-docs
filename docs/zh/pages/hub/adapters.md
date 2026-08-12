<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱面部使用_Adapters_

> 注意：_Adapters_已替换了`adapter-transformers`库，并且在模型权重方面完全兼容。更多信息请参见[here](https://docs.adapterhub.ml/transitioning.html)。

[_Adapters_](https://github.com/adapter-hub/adapters) 是 🤗 `transformers` 的附加库，用于使用适配器和其他参数有效的方法有效地微调预训练的语言模型。
_Adapters_还提供了在训练和推理过程中组合适配器模块的各种方法。
您可以在[_Adapters_ paper](https://arxiv.org/abs/2311.11077)了解更多相关信息。

## 探索集线器上的_适配器_

您可以通过[models page](https://huggingface.co/models?library=adapter-transformers&sort=downloads)左侧的过滤来查找_Adapters_型号。一些适配器型号可以在适配器中心[repository](https://github.com/adapter-hub/hub)中找到。两个来源的模型都汇总在 [AdapterHub website](https://adapterhub.ml/explore/) 上。

## 安装

首先，您可以参考[AdapterHub installation guide](https://docs.adapterhub.ml/installation.html)。您还可以通过 pip 使用以下一行安装：

```
pip install adapters
```

## 使用现有模型

有关加载预训练适配器的完整指南，我们建议查看[official guide](https://docs.adapterhub.ml/loading.html)。 

简而言之，完整的设置包括三个步骤：1. 使用 _Adapters_ 提供的 `AutoAdapterModel` 类加载基本 `transformers` 模型。
2. 使用`load_adapter()`方法加载并添加适配器。
3. 通过`active_adapters`激活适配器（用于推理）或通过`train_adapter()`激活并将其设置为可训练（用于训练）。请务必查看[composition of adapters](https://docs.adapterhub.ml/adapter_composition.html)。

```py
from adapters import AutoAdapterModel

# 1.
model = AutoAdapterModel.from_pretrained("FacebookAI/roberta-base")
# 2.
adapter_name = model.load_adapter("AdapterHub/roberta-base-pf-imdb")
# 3.
model.active_adapters = adapter_name
# or model.train_adapter(adapter_name)
```

您还可以使用 `list_adapters` 以编程方式查找所有适配器型号：

```py
from adapters import list_adapters

# source can be "ah" (AdapterHub), "hf" (hf.co) or None (for both, default)
adapter_infos = list_adapters(source="hf", model_name="FacebookAI/roberta-base")
```

如果您想了解如何加载特定模型，可以单击`Use in Adapters`，您将获得一个可以加载它的工作片段！ 

## 分享你的模型

有关使用_Adapters_共享模型的完整指南，我们建议查看[official guide](https://docs.adapterhub.ml/huggingface_hub.html#uploading-to-the-hub)。 

您可以使用已包含适配器的模型中的 `push_adapter_to_hub` 方法来共享适配器。

```py
model.push_adapter_to_hub(
    "my-awesome-adapter",
    "awesome_adapter",
    adapterhub_tag="sentiment/imdb",
    datasets_tag="imdb"
)
```

此命令创建一个存储库，其中包含自动生成的模型卡和所有必要的元数据。

## 其他资源

* _适配器_ [repository](https://github.com/adapter-hub/adapters)
* _适配器_ [docs](https://docs.adapterhub.ml)
* _适配器_ [paper](https://arxiv.org/abs/2311.11077)
* 与集线器[docs](https://docs.adapterhub.ml/huggingface_hub.html)集成

### 如何使用 Microsoft Entra ID (Azure AD) 配置 SAML SSO
https://huggingface.co/docs/hub/security-sso-azure-saml.md