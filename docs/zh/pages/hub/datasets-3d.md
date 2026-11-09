<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 3D 数据集

Hub 可在浏览器中预览 3D 文件，因此您无需下载即可检查网格、场景、点云和高斯图。本指南涵盖：

- 交互式查看器中的[Previewing 3D files](#previewing-3d-files)
- [3D files in the Dataset Viewer](#3d-files-in-the-dataset-viewer)，带有 `Mesh` 列和资产画廊
- [Structuring your dataset](#structuring-your-dataset) 因此 3D 文件会显示在其他列旁边
- 用于机器人模拟的 [SimReady datasets](#simready-datasets) [OpenUSD](https://openusd.org) 资产

## 预览 3D 文件

单击数据集的 **文件和版本** 选项卡中的 3D 文件或 [Storage Bucket](./storage-buckets) 以在交互式 3D 查看器中打开它。您可以旋转、缩放和全屏打开查看器。

|格式|扩展 |示例|
| -------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| GLTF | `.glb`、`.gltf` | [DamagedHelmet.glb](https://huggingface.co/datasets/cfahlgren1/3d-model-samples/blob/main/DamagedHelmet.glb) |
|网布| `.obj`、`.stl`、`.ply` | [cone.stl](https://huggingface.co/datasets/cfahlgren1/mesh-doc-test-formats/blob/main/meshes/cone.stl) |
|点云| `.ply` | [birmingham_block_13.ply](https://huggingface.co/datasets/WhiteFlamesCN/sensaturban-citynav-ply/blob/main/birmingham_block_13.ply) |
|高斯分布| `.ply`、`.splat` | [a_monarch_butterfly.ply](https://huggingface.co/datasets/cfahlgren1/3d-splat-samples/blob/main/a_monarch_butterfly.ply) |
|开盘美元 | `.usd`、`.usda`、`.usdc`、`.usdz` | [sm_box_corrugated_brown_b08_01.usd](https://huggingface.co/datasets/nvidia/simready-assets/blob/main/box_corrugated_brown_b08/simready_usd/sm_box_corrugated_brown_b08_01.usd) |> [!提示]
> 单击“**加载 3D 预览**”时，会加载超过 100 MB 的 glTF、网格、点云和高斯 splat 文件。无法预览超过 1 GB 的文件。

### 网格和 glTF

glTF 文件使用其材质和纹理进行渲染。使用右上角的按钮在**实体**、**线框**和**实体+线框**之间切换。

### 点云

仅包含顶点的 `.ply` 文件（例如 LiDAR 或摄影测量扫描）显示为带有顶点颜色的点云：

### 高斯分布

高斯图可以存储为 `.splat` 或 `.ply` 文件：

### 开放美元

美元资产通常分为多个文件：引用子层、有效负载、材质和纹理的根文件。查看器从同一存储库加载这些引用的文件，因此请保留您创作资产所用的文件夹结构，并使用相对路径进行引用。无法加载对绝对路径或存储库外部文件的引用。

## 数据集查看器中的 3D 文件

### 网格列

数据集查看器显示 [⟦T21⟧](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Mesh) 列的每个单元格的缩略图。可以为 `.glb`、`.ply`、`.stl` 和 `.obj` 文件呈现缩略图。单击缩略图以 3D 方式打开文件，位于该行的其他值旁边：

### 3D 资产卡

数据集页面还显示一个 **3D 资源** 卡，其中包含所选子集和拆分的第一行中最多 15 个缩略图，如 [top of this page](#3d-dataset) 中的缩略图。确保第一行有网格，否则卡片不会出现。

例如，参见[cfahlgren1/smithsonian-3d-glb](https://huggingface.co/datasets/cfahlgren1/smithsonian-3d-glb)。

## 构建数据集

### 仅 3D 文件

如果您的数据集仅包含 `.glb`、`.ply` 或 `.stl` 文件，则无需执行任何操作：它们会自动获取 `mesh` 列。

```
my_dataset_repository/
└── meshes
    ├── chair.glb
    ├── lamp.glb
    └── table.glb
```

### 附加列

要添加标题、标签或其他列，请在 3D 文件旁边添加带有 `file_name` 列的 `metadata.jsonl`（或 `metadata.csv`）文件：

```
my_dataset_repository/
└── train
    ├── box.glb
    ├── sphere.glb
    └── metadata.jsonl
```

有关更多详细信息，请参阅 [Create a mesh dataset](https://huggingface.co/docs/datasets/main/en/mesh_dataset)，有关示例，请参阅 [cfahlgren1/mesh-doc-test-metadata](https://huggingface.co/datasets/cfahlgren1/mesh-doc-test-metadata)。

### 镶木地板

如果您的数据集存储为 Parquet，则数据集查看器会读取 Parquet 文件，并且不会将存储库中的 3D 文件作为列添加。要在其他列旁边显示 3D 文件，请将文件保留在存储库中并添加引用它们的 `Mesh` 列：

```
my_dataset_repository/
├── meshes
│   ├── box.glb
│   ├── cone.stl
│   └── ...
└── train.parquet
```

```python
import json
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Features, Mesh, Value

table = pq.read_table("train.parquet")

cells = [
    {"bytes": None, "path": f"meshes/{row_id}.glb"}  # relative path from parquet directory to meshes
    for row_id in table["id"].to_pylist()
]
table = table.append_column("mesh", pa.array(cells, type=Mesh().pa_type))

# List all the columns of your dataset
features = Features({"id": Value("string"), "prompt": Value("string"), "mesh": Mesh()})
table = table.replace_schema_metadata(
    {"huggingface": json.dumps({"info": {"features": features.to_dict()}})}
)
pq.write_table(table, "train.parquet")
```- 路径必须相对于包含 Parquet 文件的目录。
- 对于没有 3D 文件的行使用 `None`。
- 一个文件夹最多可包含 10,000 个文件，因此请将较大的文件集分组到子文件夹中。

请参阅 [cfahlgren1/mesh-doc-test-formats](https://huggingface.co/datasets/cfahlgren1/mesh-doc-test-formats) 的示例。

## SimReady 数据集

[SimReady](https://github.com/NVIDIA/simready-foundation) 是 OpenUSD 资产的规范，可用于模拟，具有物理、材料和语义标签。包含 SimReady 包的数据集在其页面上显示 **3D 资产** 卡，其中包含包的数量以及整个数据集中采样的最多 15 个缩略图。单击缩略图会在查看器中打开其 USD 文件。

例如，请参阅 [nvidia/simready-assets](https://huggingface.co/datasets/nvidia/simready-assets) 或 [nvidia/PhysicalAI-SimReady-Warehouse-01](https://huggingface.co/datasets/nvidia/PhysicalAI-SimReady-Warehouse-01)。

这些数据集还具有 `simready` 标签，因此它们会在 [filtering datasets](https://huggingface.co/datasets?other=simready) 时显示。

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
```- 可选的`<file>.usd_light.png`在浅色模式下显示，`<file>.usd.png`在深色模式下显示。
- 超过 4 MB 的缩略图将被跳过。

每次推送到数据集后，该卡都会更新。

## 模态标签

具有 3D 文件的数据集具有 `3d` 模态，因此它们在过滤 [datasets by modality](https://huggingface.co/datasets?modality=modality:3d) 时会显示。您也可以在数据集卡中设置，参见[Force set a dataset modality](./datasets-cards#force-set-a-dataset-modality)。

### SQL Console：在浏览器中查询拥抱人脸数据集
https://huggingface.co/docs/hub/datasets-viewer-sql-console.md