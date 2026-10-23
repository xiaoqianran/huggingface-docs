<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据集概述

## Hub 上的数据集

Hugging Face Hub 托管一个[large number of community-curated datasets](https://huggingface.co/datasets)，用于执行各种任务，例如翻译、自动语音识别和图像分类。除了 [dataset card](./datasets-cards) 中包含的信息之外，许多数据集（例如 [GLUE](https://huggingface.co/datasets/nyu-mll/glue)）还包含 [Dataset Viewer](./data-studio) 来展示数据。

每个数据集都是一个[Git repository](./repositories)，其中包含生成训练、评估和测试分割所需的数据。有关如何构建数据集存储库的信息，请参阅[Data files Configuration page](./datasets-data-files-configuration)。遵循受支持的存储库结构将确保 Hub 上的数据集页面具有查看器。

## 搜索数据集

与模型和空间一样，您可以使用顶部导航或 [main datasets page](https://huggingface.co/datasets) 中的搜索栏在 Hub 中搜索数据集。您可以使用大量语言、任务和许可证来过滤结果，以找到适合您的数据集。

## 隐私

由于数据集是存储库，因此您可以通过“设置”选项卡[toggle their visibility between private and public](./repositories-settings#private-repositories)。如果数据集归 [organization](./organizations) 所有，则隐私设置适用于组织的所有成员。

### 鸭数据库
https://huggingface.co/docs/hub/datasets-duckdb.md