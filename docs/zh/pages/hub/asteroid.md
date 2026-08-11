<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用小行星拥抱脸部

`asteroid`是一个用于音频源分离的Pytorch工具包。它可以在常见数据集上进行快速实验，并支持大量数据集和重现论文的方法。

## 探索中心的小行星

您可以通过[models page](https://huggingface.co/models?filter=asteroid)左侧筛选找到`asteroid`型号。 

Hub 上的所有型号均具有以下功能：
1. 自动生成的模型卡，其中包含描述、训练配置、指标等。
2. 元数据标签有助于发现并包含许可证和数据集等信息。
3. 一个交互式小部件，您可以使用它直接在浏览器中玩模型。
4. 推理提供程序小部件，允许发出推理请求。

## 使用现有模型

有关加载预训练模型的完整指南，我们建议查看[official guide](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md)。 

所有模型类（`BaseModel`、`ConvTasNet` 等）都有一个 `from_pretrained` 方法，允许从 Hub 加载模型。

```py
from asteroid.models import ConvTasNet
model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM_sepclean')
```

如果您想了解如何加载特定模型，可以单击`Use in Adapter Transformers`，您将获得一个可以加载它的工作片段！ 

## 分享你的模型目前没有自动方法将模型上传到 Hub，但上传模型的过程记录在 [official guide](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md#share-your-models) 中。

所有配方都会创建将模型上传到集线器所需的所有文件。该过程通常涉及以下步骤：
1. 创建并克隆模型存储库。
2. 将文件从配方输出移动到存储库（模型卡、模型过滤器、TensorBoard 跟踪）。
3. 推送文件（`git add` + `git commit` + `git push`）。

完成此操作后，您可以直接在浏览器中尝试您的模型并与社区其他成员共享。

## 其他资源

* 小行星[website](https://asteroid-team.github.io/)。
* 小行星[library](https://github.com/asteroid-team/asteroid)。
* 集成[docs](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md)。

### 私有和门控数据集的身份验证
https://huggingface.co/docs/hub/datasets-duckdb-auth.md