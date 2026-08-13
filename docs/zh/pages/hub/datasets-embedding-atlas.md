<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 嵌入图集

[Embedding Atlas](https://apple.github.io/embedding-atlas/)是一种用于探索大型嵌入空间的交互式可视化工具。它使您能够可视化、交叉过滤和搜索嵌入以及关联的元数据，帮助您了解高维数据中的模式和关系。所有计算都在您的计算机中进行，确保您的数据保持私密和安全。

以下是作为静态空间托管的 [MegaScience](https://huggingface.co/datasets/MegaScience/MegaScience) 数据集的 [example atlas](https://huggingface.co/spaces/davanstrien/megascience)：

  
    
  

## 主要特点

- **交互式探索**：通过流畅、响应式可视化浏览数百万个嵌入
- **基于浏览器的计算**：本地计算嵌入和投影，无需将数据发送到外部服务器
- **交叉过滤**：跨多个元数据列链接和过滤数据
- **搜索功能**：查找与给定查询或现有项目相似的数据点
- **多个集成选项**：通过命令行、Jupyter 小部件或 Web 界面使用

## 先决条件

首先，安装 Embedding Atlas：

```bash
pip install embedding-atlas
```

如果您计划从 Hugging Face Hub 加载私有数据集，您还需要 [login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)：

```bash
hf auth login
```

## 从 Hub 加载数据集Embedding Atlas 提供与 Hugging Face Hub 的无缝集成，使您可以直接可视化任何数据集的嵌入。

### 使用命令行

可视化拥抱脸部数据集的最简单方法是通过命令行界面。尝试使用 IMDB 数据集：

```bash
# Load the IMDB dataset from the Hub
embedding-atlas stanfordnlp/imdb

# Specify the text column for embedding computation
embedding-atlas stanfordnlp/imdb --text "text"

# Load only a sample for faster exploration
embedding-atlas stanfordnlp/imdb --text "text" --sample 5000
```

对于您自己的数据集，请使用相同的模式：

```bash
# Load your dataset from the Hub
embedding-atlas username/dataset-name

# Load multiple splits
embedding-atlas username/dataset-name --split train --split test

# Specify custom text column
embedding-atlas username/dataset-name --text "content"
```

### 使用 Python 和 Jupyter

您还可以在 Jupyter Notebook 中使用 Embedding Atlas 进行交互式探索：

```python
from embedding_atlas.widget import EmbeddingAtlasWidget
from datasets import load_dataset
import pandas as pd

# Load the IMDB dataset from Hugging Face Hub
dataset = load_dataset("stanfordnlp/imdb", split="train[:5000]")

# Convert to pandas DataFrame
df = dataset.to_pandas()

# Create interactive widget
widget = EmbeddingAtlasWidget(df)
widget
```

对于您自己的数据集：

```python
from embedding_atlas.widget import EmbeddingAtlasWidget
from datasets import load_dataset
import pandas as pd

# Load your dataset from the Hub
dataset = load_dataset("username/dataset-name", split="train")
df = dataset.to_pandas()

# Create interactive widget
widget = EmbeddingAtlasWidget(df)
widget
```

### 使用预先计算的嵌入

如果您有带有预先计算嵌入的数据集，则可以直接加载它们：

```bash
# Load dataset with pre-computed coordinates
embedding-atlas username/dataset-name \
    --x "embedding_x" \
    --y "embedding_y"

# Load with pre-computed nearest neighbors
embedding-atlas username/dataset-name \
    --neighbors "neighbors_column"
```

## 自定义嵌入

Embedding Atlas 默认使用[SentenceTransformers](https://huggingface.co/sentence-transformers)，但支持自定义嵌入模型：

```bash
# Use a specific embedding model
embedding-atlas stanfordnlp/imdb \
    --text "text" \
    --model "sentence-transformers/all-MiniLM-L6-v2"

# For models requiring remote code execution
embedding-atlas username/dataset-name \
    --model "custom/model" \
    --trust-remote-code
```

### UMAP 投影参数

针对您的特定用例微调降维：

```bash
embedding-atlas stanfordnlp/imdb \
    --text "text" \
    --umap-n-neighbors 30 \
    --umap-min-dist 0.1 \
    --umap-metric "cosine"
```

## 用例

### 探索文本数据集

可视化和探索文本语料库以识别聚类、异常值和模式：

```python
from embedding_atlas.widget import EmbeddingAtlasWidget
from datasets import load_dataset
import pandas as pd

# Load a text classification dataset
dataset = load_dataset("stanfordnlp/imdb", split="train[:5000]")
df = dataset.to_pandas()

# Visualize with metadata
widget = EmbeddingAtlasWidget(df)
widget
```

## 其他资源

- [Embedding Atlas GitHub Repository](https://github.com/apple/embedding-atlas)
- [Official Documentation](https://apple.github.io/embedding-atlas/)
- [Interactive Demo](https://apple.github.io/embedding-atlas/upload/)
- [Command Line Reference](https://apple.github.io/embedding-atlas/tool.html)

### 拉取请求和讨论
https://huggingface.co/docs/hub/repositories-pull-requests-discussions.md