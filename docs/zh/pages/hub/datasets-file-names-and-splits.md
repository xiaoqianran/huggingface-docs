<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 文件名和分割

To host and share your dataset, create a dataset repository on the Hugging Face Hub and upload your data files.

本指南将向您展示在上传数据集存储库时如何命名数据集存储库中的文件和目录，并启用所有数据集中心功能（例如数据集查看器）。查看[companion collection of example datasets](https://huggingface.co/collections/datasets-examples/file-names-and-splits-655e28af4471bd95709eb135)了解更多详情。

A dataset with a supported structure and [file formats](./datasets-adding#file-formats) automatically has a dataset viewer on its page on the Hub.

Note that if none of the structures below suits your case, you can have more control over how you define splits and subsets with the [Manual Configuration](./datasets-manual-configuration).

## 基本用例

如果您的数据集未拆分为 [train/validation/test splits](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets)，最简单的数据集结构是有一个文件：`data.csv`（这适用于任何 [supported file format](./datasets-adding#file-formats) 和任何文件名）。

Your repository will also contain a `README.md` file, the [dataset card](./datasets-cards) displayed on your dataset page.

```
my_dataset_repository/
├── README.md
└── data.csv
```

## 分裂

Some patterns in the dataset repository can be used to assign certain files to train/validation/test splits.

### 文件名

您可以按照 `train`、`test` 和 `validation` 分割来命名数据文件：

```
my_dataset_repository/
├── README.md
├── train.csv
├── test.csv
└── validation.csv
```如果您没有任何非传统拆分，则可以将拆分名称放置在数据文件中的任何位置。唯一的规则是拆分名称必须由非单词字符分隔，例如 `test-file.csv` 而不是 `testfile.csv`。支持的分隔符包括下划线、破折号、空格、点和数字。

例如，以下文件名都是可接受的：

- 火车分班：`train.csv`、`my_train_file.csv`、`train1.csv`
- 验证分割：`validation.csv`、`my_validation_file.csv`、`validation1.csv`
- 测试拆分：`test.csv`、`my_test_file.csv`、`test1.csv`

### 目录名称

You can place your data files into different directories named `train`, `test`, and `validation` where each directory contains the data files for that split:

```
my_dataset_repository/
├── README.md
└── data/
    ├── train/
    │   └── data.csv
    ├── test/
    │   └── more_data.csv
    └── validation/
        └── even_more_data.csv
```

### 关键词

有多种方法可以引用训练/验证/测试分割。验证拆分有时称为“dev”，测试拆分可能称为“eval”。
还支持这些其他拆分名称，并且以下关键字是等效的：

- 训练，训练
- 验证、有效、val、dev
- 测试、测试、评估、评估

因此，下面的结构是一个有效的存储库：

```
my_dataset_repository/
├── README.md
└── data/
    ├── training.csv
    ├── eval.csv
    └── valid.csv
```

### 每个分割多个文件

拆分可以跨越多个文件，例如：

```
my_dataset_repository/
├── README.md
├── train_0.csv
├── train_1.csv
├── train_2.csv
├── train_3.csv
├── test_0.csv
└── test_1.csv
```确保 `train` 集的所有文件名称中都包含 *train* （测试和验证相同）。
您甚至可以在文件名中添加前缀或后缀`train`（例如`my_train_file_00001.csv`）。

为了方便起见，您还可以将数据文件放在不同的目录中。
在这种情况下，分割名称是从目录名称推断出来的。

```
my_dataset_repository/
├── README.md
└── data/
    ├── train/
    │   ├── shard_0.csv
    │   ├── shard_1.csv
    │   ├── shard_2.csv
    │   └── shard_3.csv
    └── test/
        ├── shard_0.csv
        └── shard_1.csv
```

### 自定义拆分名称

如果您的数据集分割的自定义名称不是 `train`、`test` 或 `validation`，那么您可以将数据文件命名为 `data/<split_name>-xxxxx-of-xxxxx.csv`。

这是一个具有三个分割的示例：`train`、`test` 和 `random`：

```
my_dataset_repository/
├── README.md
└── data/
    ├── train-00000-of-00003.csv
    ├── train-00001-of-00003.csv
    ├── train-00002-of-00003.csv
    ├── test-00000-of-00001.csv
    ├── random-00000-of-00003.csv
    ├── random-00001-of-00003.csv
    └── random-00002-of-00003.csv
```

### 安全模型
https://huggingface.co/docs/hub/xet/security.md