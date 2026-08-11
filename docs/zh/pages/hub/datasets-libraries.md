<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 图书馆

数据集中心支持开源生态系统中的多个库。
借助 [huggingface_hub Python library](/docs/huggingface_hub)，您可以轻松地在 Hub 上共享数据集。
我们很高兴欢迎来到该中心的一组开源库，这些库正在推动机器学习的发展。

## 库表

下表总结了支持的库及其集成级别。

|图书馆 |描述 |从集线器下载 |从集线器传输 |推送到集线器 |流式传输到集线器 |优化的 Parquet 文件 |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | ----------------- | ---------------- | ----------- | ------------- | ----------------------- |
| [Argilla](./datasets-argilla) |面向重视高质量数据的人工智能工程师和领域专家的协作工具。                                           | ✅ | ❌ | ✅ | ❌ | ❌ || [Daft](./datasets-daft) |使用 Python 原生接口进行大规模、多模式数据处理的数据引擎。                                        | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Dask](./datasets-dask) |并行和分布式计算库，可扩展现有的 Python 和 PyData 生态系统。                               | ✅ | ✅ | ✅ | ✅ | ✅* |
| [Data Designer](./datasets-data-designer) | NVIDIA NeMo 框架，用于使用 LLM 生成合成数据集。                                                      | ✅ | ❌ | ✅ | ❌ | ❌ |
| [Datasets](./datasets-usage) | 🤗 Datasets 是一个用于访问和共享音频、计算机视觉和自然语言处理 (NLP) 数据集的库。 | ✅ | ✅ | ✅ | ✅ | ✅ |
| [Distilabel](./datasets-distilabel) |合成数据生成和人工智能反馈的框架。                                                                   | ✅ | ❌ | ✅ | ❌ | ❌ || [DuckDB](./datasets-duckdb) |进程内 SQL OLAP 数据库管理系统。                                                                                | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Embedding Atlas](./datasets-embedding-atlas) |用于大型嵌入的交互式可视化和探索工具。                                                 | ✅ | ✅ | ❌ | ❌ | ❌ |
| [Fenic](./datasets-fenic) |受 PySpark 启发的 DataFrame 框架，用于构建生产 AI 和代理应用程序。                                      | ✅ | ✅ | ❌ | ❌ | ❌ |
| [FiftyOne](./datasets-fiftyone) | FiftyOne 是一个用于图像、视频和 3D 数据的管理和可视化的库。                                             | ✅ | ✅ | ✅ | ❌ | ❌ |
| [Lance](./datasets-lance) |用于多模式人工智能的开放湖屋格式。                                                                                    | ✅ | ✅ | ❌ | ❌ | ❌ || [Pandas](./datasets-pandas) | Python数据分析工具包。                                                                                                  | ✅ | ❌ | ✅ | ❌ | ✅* |
| [Polars](./datasets-polars) | OLAP 查询引擎之上的 DataFrame 库。                                                                            | ✅ | ✅ | ✅ | ❌ | ❌ |
| [PyArrow](./datasets-pyarrow) | Apache Arrow 是一种柱状格式和用于快速数据交换和内存分析的工具箱。                             | ✅ | ✅ | ✅ | ❌ | ✅* |
| [Spark](./datasets-spark) |分布式环境下的实时、大规模数据处理工具。                                                      | ✅ | ✅ | ✅ | ✅ | ✅ |
| [WebDataset](./datasets-webdataset) |用于为大型数据集编写 I/O 管道的库。                                                                             | ✅ | ✅ | ❌ | ❌ | ❌ |_ * 需要传递额外的参数来写入优化的 Parquet 文件_

## 数据处理库

### 流媒体

数据集流允许逐步迭代 Hugging Face 中的数据集，而无需完全下载。
它节省了本地磁盘空间，因为数据永远不在磁盘上。它可以节省内存，因为一次仅使用数据集的一小部分。而且它还可以节省时间，因为无需在 CPU 或 GPU 工作负载之前下载数据。

除了从“Hugging Face”流式传输之外，许多库还支持“返回”Hugging Face 流式传输。
因此，他们可以运行端到端的流传输管道：从源进行流传输并逐步写入 Hugging Face，通常会重叠下载、上传和处理步骤。

有关如何进行流式传输的更多详细信息，请查看支持流式传输的库的文档（参见上表），或者如果您想自己从 Hugging Face 流式传输数据集，请查看 [streaming datasets](./datasets-streaming) 文档。

### 优化的 Parquet 文件

Hugging Face 上的 Parquet 文件经过优化，可提高存储效率、加速下载和上传，并实现高效的数据集流式传输和编辑。优化的 Parquet 文件是具有附加功能的 Parquet 文件：

* [Parquet Content Defined Chunking](https://huggingface.co/blog/parquet-cdc) 针对 Hugging Face 的存储后端[Xet](https://huggingface.co/docs/hub/en/xet/index) 优化了 Parquet。由于基于块的重复数据删除，它可以加速上传和下载，并允许高效的文件编辑
* 页面索引可在流式传输时加速过滤器并实现高效的随机访问，例如在[Dataset Viewer](https://huggingface.co/docs/dataset-viewer)

有些库需要额外的参数来编写优化的 Parquet 文件，例如 `Pandas` 和 `PyArrow`：

* `use_content_defined_chunking=True` 为 [deduplication](https://huggingface.co/blog/parquet-cdc) 和 [editing](./datasets-editing) 启用 Parquet 内容定义分块
* `write_page_index=True` 在 Parquet 元数据中包含页面索引，对于 [streaming and random access](./datasets-streaming)

## 培训库

与 Hub 数据集集成以进行模型训练的训练库。下表显示了它们的流媒体功能 - 无需先下载数据集即可进行训练的能力。|图书馆 |描述 |从集线器传输 |
| -------- | ----------- | ---------------- |
| [Axolotl](https://docs.axolotl.ai/docs/streaming.html) |低代码LLM微调框架| ✅ |
| [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) | 100+ LLM 统一微调 | ✅ |
| [Sentence Transformers](https://sbert.net/docs/sentence_transformer/training_overview.html) |文本嵌入和语义相似度 | ✅ |
| [Transformers](https://huggingface.co/docs/transformers/trainer) | 🤗 用于微调模型的 Transformers Trainer | ✅ |
| [TRL](https://huggingface.co/docs/trl) |通过强化学习（SFT、DPO、GRPO）培训法学硕士 | ⚠️* |
| [Unsloth](https://docs.unsloth.ai) |快速 LLM 微调（加速 2 倍，内存减少 70%）| ✅ |

_* SFTTrainer 和 DPOTrainer 支持流式传输； GRPPOTrainer 尚不支持流式输入_

### 从集线器流式传输

流式传输允许对大量数据集进行训练，而无需先下载它们。这在以下情况下很有价值：
- 您的数据集太大，无法容纳在磁盘上
- 您想立即开始训练
- 您正在使用 [HF Jobs](https://huggingface.co/docs/hub/jobs)，其中共置计算可提供更快的流传输

最近的改进使流式传输[up to 100x more efficient](https://huggingface.co/blog/streaming-datasets)具有更快的启动、预取和更好的扩展到许多工作人员的能力。

**注意：** 流式处理需要在训练参数中使用`max_steps`，因为数据集长度未知，并且使用基于缓冲区的改组。更多详情请参见[streaming datasets](./datasets-streaming)。

### 记录到集线器

一些工具可以在训练期间将训练数据流回中心：- **[Trackio](https://github.com/huggingface/trackio)**：将训练指标实时传输到 Hub 数据集

## 将数据库和工具与 Hub 集成

本指南专为想要与 Hugging Face Hub 集成的数据库和工具的开发人员和维护人员而设计。无论您是构建数据处理库、分析工具还是任何需要与数据集交互的软件，本文档都将帮助您实现 Hub 集成。

该指南涵盖：

- 将数据从 Hub 加载到库/工具中的可能方法
- 将数据从库/工具上传到中心的可能方法

### 从集线器加载数据

如果您有一个用于处理数据的库，那么您的用户从 Hub 加载数据会很有帮助。

一般来说，我们建议依靠现有的库（如 `datasets`、`pandas` 或 `polars`）来执行此操作，除非您有特定原因要实现自己的库。如果您需要对加载过程进行更多控制，可以使用 `huggingface_hub` 库，例如，它允许您从存储库下载特定的文件子集。

您可以找到有关从 Hub [here](https://huggingface.co/docs/hub/datasets-downloading) 加载数据的更多信息。

#### 通过数据集查看器和 Parquet 文件集成Hub 的数据集查看器和 Parquet 转换系统提供了一种与数据集集成的标准化方法，无论其原始格式如何。该基础设施是集线器和外部库之间的可靠集成层。

如果数据集尚未采用 Parquet 格式，则 Hub 会自动将每个数据集的前 5GB 转换为 Parquet 格式，以支持数据集查看器并提供一致的访问模式。这种标准化为库集成提供了多种好处：

- 无论原始格式如何，一致的数据访问模式
- 通过 Hub 的数据集查看器进行内置数据集预览和探索。数据集查看器还可以作为 iframe 嵌入到您的应用程序中，从而轻松提供丰富的数据集预览。有关嵌入查看器的更多信息，请参阅[dataset viewer embedding documentation](https://huggingface.co/docs/hub/en/datasets-viewer-embed)。
- 针对查询优化的高效列式存储。例如，您可以使用[DuckDB](https://duckdb.org/)之类的工具来查询或过滤特定的数据子集。
- Parquet 在机器学习和数据科学生态系统中得到了良好的支持。

有关使用数据集查看器 API 的更多详细信息，请参阅 [Dataset Viewer API documentation](https://huggingface.co/docs/dataset-viewer/index)

### 上传数据到Hub本节介绍了添加将数据上传到库中的集线器的功能的可能方法，即如何实现 `push_to_hub` 方法。

本指南将介绍将数据上传到 Hub 的三种主要方法：

- 使用`datasets`库和`push_to_hub`方法
- 使用`pandas`写入Hub
- 使用`huggingface_hub`库和`hf_hub_download`方法
- 直接使用 API 或 Git 与 git-xet

#### 使用`datasets`库

将数据推送到 Hub 的最直接方法是依赖 `datasets` 库中现有的 [⟦T19⟧](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.push_to_hub) 方法。 `push_to_hub`方法会自动处理：

- 存储库的创建
- 将数据集转换为 Parquet
- 将数据集分成合适的部分
- 上传数据

例如，如果您有一个返回字典列表的合成数据生成库，您可以简单地执行以下操作：

```python
from datasets import Dataset

data = [{"prompt": "Write a cake recipe", "response": "Measure 1 cup ..."}]
ds = Dataset.from_list(data)
ds.push_to_hub("USERNAME_OR_ORG/repo_ID")
```

这种集成的示例：

- [Distilabel](https://github.com/argilla-io/distilabel/blob/8ad48387dfa4d7bd5639065661f1975dcb44c16a/src/distilabel/distiset.py#L77)

#### 依赖现有库与 Hub 的集成

Polars、Pandas、Dask、Spark、DuckDB 和 Daft 都可以写入 Hugging Face Hub 存储库。更多详情请参见[datasets libraries](https://huggingface.co/docs/hub/datasets-libraries)。如果您已经在代码中使用这些库之一，则添加推送到 Hub 的功能非常简单。例如，如果您有一个可以返回 Pandas DataFrame 的合成数据生成库，则以下是您需要写入 Hub 的代码：

```python
from huggingface_hub import HfApi

# Initialize the Hub API
hf_api = HfApi(token=os.getenv("HF_TOKEN"))

# Create a repository (if it doesn't exist)
hf_api.create_repo(repo_id="username/my-dataset", repo_type="dataset")

# Convert your data to a DataFrame and save directly to the Hub
df.to_parquet("hf://datasets/username/my-dataset/data.parquet")
```

#### 使用 Huggingface_hub Python 库

`huggingface_hub` Python 库提供了一种更灵活的方法来将数据上传到 Hub。该库允许您将特定文件或文件子集上传到存储库。如果您有一个大型数据集，但不想将其转换为 Parquet，想要上传特定的文件子集，或者想要对存储库结构进行更多控制，则此功能非常有用。

根据您的使用案例，您可以在代码中的特定点上传文件或文件夹，即当用户单击“推送到集线器”时，将注释从工具导出到集线器。例如，

```python
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)

api.upload_folder(
    folder_path="/my-cool-library/data-folder",
    repo_id="username/my-cool-space",
    repo_type="dataset",
    commit_message="Push annotations to Hub"
    allow_patterns="*.jsonl",
)
```

您可以找到有关如何将数据上传到集线器[here](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload)的更多信息。

或者，在某些情况下，您可能希望在后台上传数据，例如每 10 分钟生成一次合成数据。在这种情况下，您可以使用 `huggingface_hub` 库的 `scheduled_uploads` 功能。欲了解更多详情，请参阅[scheduled uploads documentation](https://huggingface.co/docs/huggingface_hub/main/en/guides/upload#scheduled-uploads)。您可以在以下位置查看使用此方法将数据上传到 Hub 的示例：

- [fastdata](https://github.com/AnswerDotAI/fastdata/blob/main/nbs/00_core.ipynb)图书馆
- 这个[magpie](https://huggingface.co/spaces/davanstrien/magpie/blob/fc79672c740b8d3d098378dca37c0f191c208de0/app.py#L67)演示空间

## 更多支持

有关集成的技术问题，请随时通过 datasets@huggingface.co 联系数据集团队。

### 空间上的 ZenML
https://huggingface.co/docs/hub/spaces-sdks-docker-zenml.md