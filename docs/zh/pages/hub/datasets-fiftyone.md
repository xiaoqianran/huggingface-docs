<!-- huggingface-docs: machine-translated zh-CN from English source -->

#五十一

FiftyOne 是一个开源工具包，用于策划、可视化和
管理非结构化视觉数据。该库简化了以数据为中心的流程
工作流程，从查找低置信度预测到识别质量较差的预测
样本并发现数据中隐藏的模式。该库支持所有
各种视觉数据，从图像和视频到 PDF、点云和网格。

FiftyOne 支持对象检测、关键点、折线和自定义模式。

FiftyOne 与 Hugging Face Hub 集成，以便您可以加载和共享
直接来自 Hub 的 FiftyOne 数据集。

🚀 尝试[Colab](https://colab.research.google.com/drive/1l0kzfbJ2wtUw1EGS1tq1PJYoWenMlihp?usp=sharing) 中的 FiftyOne 🤝 拥抱脸部集成！

## 先决条件

第一个[login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)：

```bash
hf auth login
```

确保您已安装`fiftyone>=0.24.0`：

```bash
pip install -U fiftyone
```

## 从 Hub 加载可视化数据集

使用 FiftyOne 的 Hugging Face utils 中的 `load_from_hub()`，您可以加载：

- 上传到中心的任何 FiftyOne 数据集
- 大多数基于图像的数据集存储在 Parquet 文件中（这是通过 `datasets` 库上传到集线器的数据集的标准）

### 从 Hub 加载 FiftyOne 数据集推送到 FiftyOne 的 [supported common formats](https://docs.voxel51.com/user_guide/dataset_creation/datasets.html#supported-import-formats) 之一的中心的任何数据集
其数据集存储库中应包含所有必要的配置信息
hub，因此您可以通过指定其`repo_id`来加载数据集。举个例子，要
加载[VisDrone detection dataset](https://huggingface.co/datasets/Voxel51/VisDrone2019-DET)：

```python
import fiftyone as fo
from fiftyone.utils import load_from_hub

## load from the hub
dataset = load_from_hub("Voxel51/VisDrone2019-DET")

## visualize in app
session = fo.launch_app(dataset)
```

![FiftyOne VisDrone dataset](https://cdn-uploads.huggingface.co/production/uploads/63127e2495407887cb79c5ea/0eKxe_GSsBjt8wMjT9qaI.jpeg)

您可以[customize the download process](https://docs.voxel51.com/integrations/huggingface.html#configuring-the-download-process)，包括样本数量
下载、创建的数据集对象的名称或是否持久化
到磁盘。

您可以使用以下方法列出 Hub 上所有可用的 FiftyOne 数据集：

```python
from huggingface_hub import HfApi
api = HfApi()
api.list_datasets(tags="fiftyone")
```

### 使用 FiftyOne 从 Hub 加载 Parquet 数据集

您还可以使用 `load_from_hub()` 函数从 Parquet 加载数据集
文件。为您处理类型转换，并从 URL 下载图像
如果需要的话。

借助此功能，[you can load](https://docs.voxel51.com/integrations/huggingface.html#basic-examples) 以下任何一项：

- [FiftyOne-Compatible Image Classification Datasets](https://huggingface.co/collections/Voxel51/fiftyone-compatible-image-classification-datasets-665dfd51020d8b66a56c9b6f)，如 [Food101](https://huggingface.co/datasets/food101) 和 [ImageNet-Sketch](https://huggingface.co/datasets/imagenet_sketch)
- [FiftyOne-Compatible Object Detection Datasets](https://huggingface.co/collections/Voxel51/fiftyone-compatible-object-detection-datasets-665e0279c94ae552c7159a2b) 类似 [CPPE-5](https://huggingface.co/datasets/cppe-5) 和 [WIDER FACE](https://huggingface.co/datasets/wider_face)
- [FiftyOne-Compatible Segmentation Datasets](https://huggingface.co/collections/Voxel51/fiftyone-compatible-image-segmentation-datasets-665e15b6ddb96a4d7226a380) 类似 [SceneParse150](https://huggingface.co/datasets/scene_parse_150) 和 [Sidewalk Semantic](https://huggingface.co/datasets/segments/sidewalk-semantic)
- [FiftyOne-Compatible Image Captioning Datasets](https://huggingface.co/collections/Voxel51/fiftyone-compatible-image-captioning-datasets-665e16e29350244c06084505) 类似 [COYO-700M](https://huggingface.co/datasets/kakaobrain/coyo-700m) 和 [New Yorker Caption Contest](https://huggingface.co/datasets/jmhessel/newyorker_caption_contest)
- [FiftyOne-Compatible Visual Question-Answering Datasets](https://huggingface.co/collections/Voxel51/fiftyone-compatible-vqa-datasets-665e16424ecc8a718156248a) 类似 [TextVQA](https://huggingface.co/datasets/textvqa) 和 [ScienceQA](https://huggingface.co/datasets/derek-thomas/ScienceQA)

例如，我们可以加载前 1,000 个样本
[WikiArt dataset](https://huggingface.co/datasets/huggan/wikiart) 进入 FiftyOne：

```python
import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub

dataset = load_from_hub(
    "huggan/wikiart",  ## repo_id
    format="parquet",  ## for Parquet format
    classification_fields=["artist", "style", "genre"], ## columns to treat as classification labels
    max_samples=1000,  # number of samples to load
    name="wikiart",  # name of the dataset in FiftyOne
)
```

![WikiArt Dataset](https://cdn-uploads.huggingface.co/production/uploads/63127e2495407887cb79c5ea/PCqCvTlNTG5SLtcK5fwuQ.jpeg)

## 将 FiftyOne 数据集推送到集线器

您可以使用以下方法将数据集推送到中心：

```python
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone.utils.huggingface import push_to_hub

## load example dataset
dataset = foz.load_zoo_dataset("quickstart")

## push to hub
push_to_hub(dataset, "my-hf-dataset")
```当你调用`push_to_hub()`时，数据集将被上传到repo
在您的用户名下使用指定的存储库名称，然后将创建存储库
如果需要的话。将自动生成 [Dataset Card](./datasets-cards) 并填充从集线器加载数据集的说明。您可以使用 `preview_path` 参数上传缩略图/gif 以显示在数据集卡上。

下面是一个使用许多这些参数的示例，它将使用标签、MIT 许可证、描述和预览图像将 FiftyOne 的 [Quickstart Video](https://docs.voxel51.com/user_guide/dataset_zoo/datasets.html#quickstart-video) 数据集的前三个样本上传到私有存储库 `username/my-quickstart-video-dataset`：

```python
dataset = foz.load_from_zoo("quickstart-video", max_samples=3)

push_to_hub(
    dataset,
    "my-quickstart-video-dataset",
    tags=["video", "tracking"],
    license="mit",
    description="A dataset of video samples for tracking tasks",
    private=True,
    preview_path="<path/to/preview.png>"
)
```

## 📚 资源

- [🚀 Code-Along Colab Notebook](https://colab.research.google.com/drive/1l0kzfbJ2wtUw1EGS1tq1PJYoWenMlihp?usp=sharing)
- [🗺️ User Guide for FiftyOne Datasets](https://docs.voxel51.com/user_guide/using_datasets.html#)
- [🤗 FiftyOne 🤝 Hub Integration Docs](https://docs.voxel51.com/integrations/huggingface.html#huggingface-hub)
- [🤗 FiftyOne 🤝 Transformers Integration Docs](https://docs.voxel51.com/integrations/huggingface.html#transformers-library)
- [🧩 FiftyOne Hugging Face Hub Plugin](https://github.com/voxel51/fiftyone-huggingface-plugins)

### 网络安全
https://huggingface.co/docs/hub/enterprise-network-security.md