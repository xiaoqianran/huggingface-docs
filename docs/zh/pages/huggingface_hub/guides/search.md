<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 搜索中心

在本教程中，您将学习如何使用`huggingface_hub`在Hub上搜索模型、数据集和空间。

## 如何列出存储库？

`huggingface_hub` 库包含一个 HTTP 客户端 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 来与集线器交互。
除此之外，它还可以列出存储在 Hub 上的模型、数据集和空间：

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> models = api.list_models()
```

[list_models()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_models) 的输出是存储在 Hub 上的模型的迭代器。

同样，您可以使用 [list_datasets()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_datasets) 列出数据集，使用 [list_spaces()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_spaces) 列出空间。

## 如何过滤存储库？

列出存储库很棒，但现在您可能想要过滤搜索。
列表助手有几个属性，例如：
- `filter`
- `author`
- `search`
- `num_parameters`
- ...

让我们看一个示例，获取 Hub 上进行图像分类、已在 imagenet 数据集上进行训练并与 PyTorch 一起运行的所有模型。

```py
models = hf_api.list_models(filter=["image-classification", "pytorch", "imagenet"])
```

您还可以使用与 Hub UI 相同的范围语法按参数计数过滤模型：

```py
models = hf_api.list_models(num_parameters="min:6B,max:128B")
```

过滤时，您还可以对模型进行排序并仅获取最靠前的结果。例如，
以下示例获取 Hub 上下载次数最多的前 5 个数据集：

```py
>>> list(list_datasets(sort="downloads", limit=5))
[DatasetInfo(
	id='argilla/databricks-dolly-15k-curated-en',
	author='argilla',
	sha='4dcd1dedbe148307a833c931b21ca456a1fc4281',
	last_modified=datetime.datetime(2023, 10, 2, 12, 32, 53, tzinfo=datetime.timezone.utc),
	private=False,
	downloads=8889377,
	(...)
```要探索 Hub 上的可用过滤器，请访问 [models](https://huggingface.co/models) 和 [datasets](https://huggingface.co/datasets) 页面
在浏览器中搜索一些参数并查看 URL 中的值。

## 使用 CLI

您还可以使用 `hf` 命令行界面列出和搜索模型、数据集和空间：

```bash
# List models
>>> hf models ls --search "llama" --sort downloads --limit 5

# List datasets
>>> hf datasets ls --author Qwen

# List Spaces
>>> hf spaces ls --search "3d"

# Get info about a specific model
>>> hf models info Lightricks/LTX-2
```

欲了解更多详情，请参阅[CLI guide](./cli#hf-models)。

### 了解缓存
https://huggingface.co/docs/huggingface_hub/v1.27.0/guides/manage-cache.md