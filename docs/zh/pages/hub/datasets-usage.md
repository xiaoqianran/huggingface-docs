<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用🤗数据集

在 Hugging Face Hub 上找到有趣的数据集后，您可以使用 🤗 数据集加载该数据集。您可以点击[**Use this dataset** button](https://huggingface.co/datasets/nyu-mll/glue?library=datasets)复制代码来加载数据集。

首先你需要[Login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)，例如使用：

```
hf auth login
```

然后您可以使用 Hugging Face Hub 加载数据集

```python
from datasets import load_dataset

dataset = load_dataset("username/my_dataset")

# or load the separate splits if the dataset has train/validation/test splits
train_dataset = load_dataset("username/my_dataset", split="train")
valid_dataset = load_dataset("username/my_dataset", split="validation")
test_dataset  = load_dataset("username/my_dataset", split="test")
```

您还可以将数据集上传到 Hugging Face Hub：

```python
my_new_dataset.push_to_hub("username/my_new_dataset")
```

这将创建一个数据集存储库`username/my_new_dataset`，其中包含 Parquet 格式的数据集，您可以稍后重新加载。

有关使用 🤗 数据集的更多信息，请查看 🤗 数据集文档中提供的 [tutorials](/docs/datasets/tutorial) 和 [how-to guides](/docs/datasets/how_to)。

### 安全
https://huggingface.co/docs/hub/security.md