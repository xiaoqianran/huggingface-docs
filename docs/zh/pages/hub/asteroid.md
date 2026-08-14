<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用小行星拥抱脸部

`asteroid` is a Pytorch toolkit for audio source separation. It enables fast experimentation on common datasets with support for a large range of datasets and recipes to reproduce papers.

## 探索中心的小行星

You can find `asteroid` models by filtering at the left of the [models page](https://huggingface.co/models?filter=asteroid). 

All models on the Hub come up with the following features:
1. An automatically generated model card with a description, training configuration, metrics, and more.
2. Metadata tags that help for discoverability and contain information such as licenses and datasets.
3. An interactive widget you can use to play out with the model directly in the browser.
4. An Inference Providers widget that allows to make inference requests.

## 使用现有模型

For a full guide on loading pre-trained models, we recommend checking out the [official guide](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md). 

All model classes (`BaseModel`, `ConvTasNet`, etc) have a `from_pretrained` method that allows to load models from the Hub.

```py
from asteroid.models import ConvTasNet
model = ConvTasNet.from_pretrained('mpariente/ConvTasNet_WHAM_sepclean')
```

If you want to see how to load a specific model, you can click `Use in Adapter Transformers` and you will be given a working snippet that you can load it! 

## 分享你的模型目前没有自动方法将模型上传到 Hub，但上传模型的过程记录在 [official guide](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md#share-your-models) 中。

All the recipes create all the needed files to upload a model to the Hub.该过程通常涉及以下步骤：
1. 创建并克隆模型存储库。
2. Moving files from the recipe output to the repository (model card, model filte, TensorBoard traces).
3. 推送文件（`git add` + `git commit` + `git push`）。

完成此操作后，您可以直接在浏览器中尝试您的模型并与社区其他成员共享。

## 其他资源

* 小行星[website](https://asteroid-team.github.io/)。
* 小行星[library](https://github.com/asteroid-team/asteroid)。
* 集成[docs](https://github.com/asteroid-team/asteroid/blob/master/docs/source/readmes/pretrained_models.md)。

### How to configure SAML SSO with Google Workspace
https://huggingface.co/docs/hub/security-sso-google-saml.md