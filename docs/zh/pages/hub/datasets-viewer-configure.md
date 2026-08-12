<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 配置数据集查看器

数据集查看器支持多种 [data files formats](./datasets-adding#file-formats)，从文本到表格，从图像到音频格式。
它还根据文件和文件夹名称分隔训练/验证/测试拆分。

要为您的数据集配置数据集查看器，首先确保您的数据集位于 [supported data format](./datasets-adding#file-formats) 中。

## 配置拆分或子集的下拉菜单

在数据集查看器中，您可以查看数据集的 [train/validation/test](https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets) 分割，有时还可以在多个子集之间进行选择（例如，每种语言一个）。

要定义这些下拉列表，您可以在数据文件或其拆分名称之后命名它们的文件夹（训练/验证/测试）。
还可以使用 YAML 手动自定义拆分。

如需了解更多信息，请随时查看 [Data files Configuration](./datasets-data-files-configuration) 和 [collections of example datasets](https://huggingface.co/datasets-examples) 的文档。 [Image Dataset doc page](./datasets-image) 提出了各种方法来用图像构建数据集。

## 禁用查看器

可以禁用数据集查看器。为此，请将 YAML 部分添加到数据集的 `README.md` 文件（如果尚不存在，则创建一个），并添加值为 `false` 的 `viewer` 属性。

```yaml
---
viewer: false
---
```

## 私有数据集

对于 **私有** 数据集，为 [PRO users](https://huggingface.co/pricing) 和 [Team or Enterprise organizations](https://huggingface.co/enterprise) 启用数据集查看器。### 管理工作
https://huggingface.co/docs/hub/jobs-manage.md