<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 手动配置

本指南将向您展示如何为数据集存储库配置自定义结构。 [companion collection of example datasets](https://huggingface.co/collections/datasets-examples/manual-configuration-655e293cea26da0acab95b87) 展示了文档的每个部分。

具有受支持结构和 [file formats](./datasets-adding#file-formats) 的数据集会自动在 Hub 上的数据集页面上有一个数据集查看器。您可以使用 YAML 定义查看器使用的拆分、子集和构建器参数。

还可以为同一数据集定义多个子集（也称为“配置”）（例如，如果数据集具有各种独立文件）。

## 分裂

如果您有多个文件并且想要定义哪个文件进入哪个拆分，您可以在 README.md 顶部使用 YAML。

例如，给定一个像这样的存储库：

```
my_dataset_repository/
├── README.md
├── data.csv
└── holdout.csv
```

您可以通过在 README.md 顶部的 YAML 块中添加 `configs` 字段来定义拆分子集：

```yaml
---
configs:
- config_name: default
  data_files:
  - split: train
    path: "data.csv"
  - split: test
    path: "holdout.csv"
---
```

您可以使用路径列表为每个拆分选择多个文件：

```
my_dataset_repository/
├── README.md
├── data/
│   ├── abc.csv
│   └── def.csv
└── holdout/
    └── ghi.csv
```

```yaml
---
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - "data/abc.csv"
    - "data/def.csv"
  - split: test
    path: "holdout/ghi.csv"
---
```

或者您可以使用 glob 模式自动列出您需要的所有文件：

```yaml
---
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/*.csv"
  - split: test
    path: "holdout/*.csv"
---
```

> [!警告]
> 请注意，即使您只有一个子集，`config_name` 字段也是必需的。

## 多个子集您的数据集可能有多个您希望能够单独使用的数据子集。
例如，每个子集在数据集查看器 Hugging Face Hub 中都有自己的下拉菜单。

在这种情况下，您可以在 YAML 中的 `configs` 字段内定义子集列表：

```
my_dataset_repository/
├── README.md
├── main_data.csv
└── additional_data.csv
```

```yaml
---
configs:
- config_name: main_data
  data_files: "main_data.csv"
- config_name: additional_data
  data_files: "additional_data.csv"
---
```

请注意，查看器中显示的子集顺序首先是默认顺序，然后是字母顺序。

> [!提示]
> 您可以使用 `default: true` 设置默认子集
>
> ```yaml
> - config_name: main_data
>   data_files: "main_data.csv"
>   default: true
> ```
>
> 这对于设置数据集查看器首先显示哪个子集以及默认加载哪个子集数据库很有用。

## 数据目录

您可以使用 `data_dir` 来指向目录，而不是使用 `data_files` 列出单个文件。该目录内的文件会根据文件扩展名自动解析。当您的数据组织在子目录中时，这特别有用：

例如，在这种情况下，您可以简单地使用`data_dir`，因为每个子集的数据都位于其自己的目录中：

```
my_dataset_repository/
├── README.md
├── main/
│   ├── train.csv
│   └── test.csv
└── extra/
    ├── train.csv
    └── test.csv
```

```yaml
---
configs:
- config_name: main
  data_dir: "main"
- config_name: extra
  data_dir: "extra"
---
```

当设置 `data_dir` 时，构建器将解析相对于该目录的文件。如果目录包含与默认拆分命名模式匹配的文件（例如 `train.csv`、`test.csv`），则会自动分配拆分，无需显式 `data_files`。您还可以将 `data_dir` 与 `data_files` 结合使用以获得更多控制：

```yaml
---
configs:
- config_name: default
  data_dir: "data"
  data_files:
  - split: train
    path: "training_*.csv"
  - split: test
    path: "eval_*.csv"
---
```

在这种情况下，`data_files`中的`path`模式是相对于`data_dir`解析的。

## 构建器参数

不仅是`data_files`，其他构建器特定的参数也可以通过 YAML 传递，从而可以更灵活地加载数据，同时不需要任何自定义代码。例如，定义在哪个子集中使用哪个分隔符来加载 `csv` 文件：

```yaml
---
configs:
- config_name: tab
  data_files: "main_data.csv"
  sep: "\t"
- config_name: comma
  data_files: "additional_data.csv"
  sep: ","
---
```

请参阅[specific builders' documentation](/docs/datasets/package_reference/builder_classes)查看它们有哪些参数。

### 带注释的模型卡模板
https://huggingface.co/docs/hub/model-card-annotated.md