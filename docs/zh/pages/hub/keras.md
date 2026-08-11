<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在拥抱脸部时使用 Keras

Keras 是一个开源多后端深度学习框架，支持 JAX、TensorFlow 和 PyTorch。您可以在[keras.io](https://keras.io/)找到更多详细信息。

## 在 Hub 中探索 Keras

您可以通过在 [models page](https://huggingface.co/models?library=keras&sort=downloads) 上按库名称过滤来在 Hub 上列出 `keras` 模型。

当直接从 Keras 库上传时，Hub 上的 Keras 模型会提供有用的功能：
1. 生成的模型卡，其中包含描述、模型图等。
2. 下载计数以监控模型的受欢迎程度。
3. 快速开始使用模型的代码片段。

## 使用现有模型

Keras 与 Hugging Face Hub 深度集成。这意味着您可以直接从库在 Hub 上加载和保存模型。为此，您需要安装最新版本的 Keras 和 `huggingface_hub`。 `huggingface_hub` 库是 Keras 用于与 Hub 交互的轻量级 Python 客户端。

```
pip install -U keras huggingface_hub
```

安装该库后，您只需通过传递 Hugging Face 路径作为参数来使用常规 `keras.saving.load_model` 方法。 HF 路径是带有 `hf://` 前缀的 `repo_id`，例如`"hf://keras-io/weather-prediction"`。在 [Keras documentation](https://keras.io/api/models/model_saving_apis/model_saving_and_loading/#load_model-function) 中了解有关 `load_model` 的更多信息。

```py
import keras

model = keras.saving.load_model("hf://Wauplin/mnist_example")
```如果您想了解如何加载特定模型，可以单击模型页面上的**使用此模型**来获取有效的代码片段！ 

## 分享你的模型

与 `load_model` 类似，您可以使用带有 HF 路径的 `model.save()` 在 Hub 上保存并共享 `keras` 模型：

```py
model = ...
model.save("hf://your-username/your-model-name")
```

如果 Hub 上不存在存储库，则会为您创建该存储库。上传的模型包含模型卡、模型图、`metadata.json`和`config.json`文件以及包含模型权重的`model.weights.h5`文件。

默认情况下，存储库将包含最小模型卡。查看 [Model Card guide](https://huggingface.co/docs/hub/model-cards) 了解有关模型卡以及如何完成它们的更多信息。您还可以使用 `huggingface_hub.ModelCard` 以编程方式更新模型卡（请参阅 [guide](https://huggingface.co/docs/huggingface_hub/guides/model-cards)）。

> [!提示]
> 您可能已经熟悉 `.keras` 文件。事实上，`.keras` 文件只是一个包含 `.json` 和 `model.weights.h5` 文件的 zip 文件。当推送到中心时，模型将保存为解压缩的文件夹，以便您浏览文件。请注意，如果您手动将 `.keras` 文件上传到 Hub 上的模型存储库，该存储库将自动标记为 `keras`，但您将无法使用 `keras.saving.load_model` 加载它。

## 其他资源

* Keras 开发者[Guides](https://keras.io/guides/)。
* Keras [examples](https://keras.io/examples/)。### 数据文件配置
https://huggingface.co/docs/hub/datasets-data-files-configuration.md