<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 3D 数据集

本指南介绍如何在 Hub 上预览 3D 文件，以及如何构建 [OpenUSD](https://openusd.org) 资产数据集，例如用于机器人模拟的 [SimReady](https://github.com/NVIDIA/simready-foundation) 包。 3D 文件不会显示在 [Dataset Viewer](./data-studio) 中，但您可以在其页面上预览每个文件，并且 SimReady 数据集会在数据集页面上获取其资产的图库。

## 预览 3D 文件

单击数据集的 **文件和版本** 选项卡中的 3D 文件或 [Storage Bucket](./storage-buckets) 以在交互式 3D 查看器中打开它。您可以旋转、缩放和全屏打开查看器。

|格式|扩展 |
| -------- | ---------------------------------- |
|开盘美元 | `.usd`、`.usda`、`.usdc`、`.usdz` |
| GLTF | `.glb`、`.gltf` |

### OpenUSD 图层和纹理

美元资产通常分为多个文件：引用子层、有效负载、材质和纹理的根文件。查看器从同一存储库加载这些引用的文件，因此请保留您创作资产所用的文件夹结构，并使用相对路径进行引用。无法加载对绝对路径或存储库外部文件的引用。

## SimReady 数据集[SimReady](https://github.com/NVIDIA/simready-foundation) 是 OpenUSD 资产的规范，可用于模拟，具有物理、材料和语义标签。包含 SimReady 包的数据集在其页面上显示 **3D 资产** 卡，其中包含包的数量以及整个数据集中采样的最多 15 个缩略图。单击缩略图会在查看器中打开其 USD 文件。

例如，请参阅 [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets) 或 [nvidia/PhysicalAI-SimReady-Warehouse-01](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01)。

### 包结构

包是包含 `com.nvidia.simready.packaging.json` 文件的文件夹。包可以位于存储库中的任何深度，包内的文件夹是该包的一部分。

每个包都需要其根 USD 文件的缩略图才能出现在卡中。根据 SimReady 对缩略图的要求，`<folder>/<file>.usd` 的缩略图是 `<folder>/.thumbs/256x256/<file>.usd.png`（或 `.usda.png`、`.usdc.png`、`.usdz.png`），位于包文件夹或其下一个文件夹中：

```
my_dataset_repository/
├── IV_pole_a01
│   ├── com.nvidia.simready.packaging.json
│   └── simready_usd
│       ├── .thumbs
│       │   └── 256x256
│       │       ├── sm_IV_pole_a01_01.usd.png
│       │       └── sm_IV_pole_a01_01.usd_light.png
│       ├── sm_IV_pole_a01_01.usd
│       └── payloads
│           └── ...
└── IV_pole_a02
    ├── com.nvidia.simready.packaging.json
    └── ...
```

- 可选的`<file>.usd_light.png`在浅色模式下显示，`<file>.usd.png`在深色模式下显示。
- 超过 4 MB 的缩略图将被跳过。

每次推送到数据集后，该卡都会更新。

## 模态标签

具有 3D 文件的数据集具有 `3d` 模态，因此它们在过滤 [datasets by modality](https://huggingface.co/datasets?modality=modality:3d) 时会显示。您也可以在数据集卡中设置，参见[Force set a dataset modality](./datasets-cards#force-set-a-dataset-modality)。### SQL Console：在浏览器中查询拥抱人脸数据集
https://huggingface.co/docs/hub/datasets-viewer-sql-console.md