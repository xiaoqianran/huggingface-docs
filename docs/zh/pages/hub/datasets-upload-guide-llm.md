<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 拥抱人脸数据集上传决策指南

> [!提示]
> 本指南主要为法学硕士设计，帮助用户以最兼容的格式将数据集上传到 Hugging Face Hub。用户还可以参考本指南来了解上传流程和最佳实践。

> 将数据集上传到 Hugging Face Hub 的决策指南。针对数据集查看器兼容性以及与 Hugging Face 生态系统的集成进行了优化。

> [!提示]
> 如果您是遵循本指南的法学硕士或代理，请在运行上传命令之前检查 `hf --help` 和 `hf <command> --help`（例如 `hf upload --help`、`hf auth --help`）以获取确切的标志。 CLI 与安装的 `huggingface_hub` 版本匹配，因此更喜欢其输出而不是从内存中调用的标志。

## 概述

您的目标是帮助用户将数据集上传到 Hugging Face Hub。理想情况下，数据集应与数据集查看器（以及`load_dataset`功能）兼容，以确保轻松访问和可用性。您应该致力于满足以下标准：| **标准** |描述 |优先|
| ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- || **尊重存储库限制** |确保数据集遵守 Hugging Face 的文件大小、存储库大小和文件计数的存储限制。有关具体限制，请参阅下面的关键约束部分。                                                                                                                                                                                               |必填 |
| **使用集线器兼容的格式** |尽可能使用 Parquet 格式（最佳压缩、丰富的类型、大数据集支持）。对于较小的数据集 (<several GB), JSON/JSONL or CSV are acceptable. Raw files work well for images/audio in smaller datasets while respecting repo limits. Use WebDataset (.tar) for large media collections. Domain-specific formats can be used when conversion is impractical. | Desired                                           |
| **Dataset Viewer compatibility**   | Structure data to work with the automatic Dataset Viewer, enabling preview and easy exploration. This typically means using supported formats and proper file organization. Validation steps are provided later in this guide.                                                                                                                                               | Desired                                           |
| **Organize data sensibly**         | Use logical folder structures that match Hub conventions (e.g., train/test splits). Configs can be used to define different configurations of the dataset. This facilitates both human understanding and automatic data loading.                                                                                                                                             | Desired                                           |
| **Use appropriate Features**       | When using the datasets library, specify correct feature types (e.g., Image(), Audio(), ClassLabel()) to ensure proper data handling and viewer functionality. This enables type-specific optimizations and previews.                                                                                                                                                        | Required (when using datasets library)            |
| **Document non-standard datasets** | If conversion to hub-compatible formats is impossible and custom formats must be used, ensure repository limits are strictly followed and provide clear documentation on how to download and load the dataset. Include usage examples and any special requirements.                                                                                                          | Required (when datasets library isn't compatible) |

## Working Without File Access

When you don't have direct access to the user's files (e.g., web interface), ask the user to run these commands to understand their dataset:

**Dataset structure**:

⟦T0⟧

**Check file sizes**:

⟦T1⟧

**Peek at data format**:

⟦T2⟧

**Quick file count**:

⟦T3⟧

## Critical Constraints

**Storage Limits**:

⟦T4⟧

**Human-readable summary**:

- Free: 100GB private datasets
- Pro (for individuals) | Team or Enterprise (for organizations): 1TB+ private storage per seat (see ⟦T93⟧)
- Public: 1TB (contact datasets@huggingface.co for larger)
- Per file: 200GB max, <50GB recommended
- Per folder: <10k files

See https://huggingface.co/docs/hub/storage-limits#repository-limitations-and-recommendations for current limits for current recommendations for repository sizes and file counts.

## Quick Reference by Data Type

| Your Data                           | Recommended Approach                                                                                         | Quick Command                                                                                    |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| **CSV/JSON files**                  | Use built-in loaders (handles any size via memory mapping)                                                   | ⟦T26⟧                     |
| **Images in folders**               | Use ⟦T94⟧ for automatic class detection          | ⟦T27⟧               |
| **Audio files**                     | Use ⟦T95⟧ for automatic organization             | ⟦T28⟧                |
| **Video files**                     | Use ⟦T96⟧ for automatic organization | ⟦T29⟧               |
| **PDF documents**                   | Use ⟦T97⟧ for text extraction      | ⟦T30⟧                   |
| **Very large datasets (100GB+)**    | Use ⟦T31⟧ to control memory usage                                                                 | ⟦T32⟧                                  |
| **Many files / directories (>10k)** |使用`upload_folder`/`hf upload`（处理大/多文件上传）| `hf upload username/dataset ./data --repo-type=dataset` |
| **流媒体大媒体** |用于高效流式传输的 WebDataset 格式 |创建 .tar 分片，然后 `hf upload` |
| **科学数据（HDF5、NetCDF）** |使用数组功能转换为 Parquet |参见[Scientific Data](#scientific-data)部分|| **自定义/专有格式** |如果无法转换，请彻底记录 | `hf upload` 带有全面的自述文件 |

## 上传工作流程

0. ✓ **收集数据集信息**（如果需要）：

   - 什么类型的数据？ （图像、文本、音频、CSV 等）
   - 它是如何组织的？ （文件夹结构、单文件、多文件）
   - 大概尺寸是多少？
   - 文件是什么格式？
   - 有什么特殊要求吗？ （例如，流媒体、私人访问）
   - 检查描述数据集的现有自述文件或文档文件

1. ✓ **验证**：
   - CLI：`hf auth login`
   - 或使用令牌：`HfApi(token="hf_...")`或设置`HF_TOKEN`环境变量
2. ✓ **识别您的数据类型**：检查上面的[Quick Reference](#quick-reference-by-data-type)表
3. ✓ **选择上传方式**：

   - **小文件（<1GB) with hub-compatible format**: Can use ⟦T100⟧ for quick uploads
   - **Built-in loader available**: Use the loader + ⟦T41⟧ (see Quick Reference table)
   - **Large datasets or many files**: Use ⟦T42⟧ / ⟦T43⟧ which handles large uploads (auto multi-commit, resumable).
   - **Custom formats**: Convert to hub-compatible format if possible, otherwise document thoroughly

4. ✓ **Test locally** (if using built-in loader):

   ⟦T5⟧

5. ✓ **Upload to Hub**:

   ⟦T6⟧

6. ✓ **Verify your upload**:
   - Check Dataset Viewer: ⟦T44⟧
   - Test loading: ⟦T45⟧
   - If viewer shows errors, check the ⟦T101⟧ section

## Common Conversion Patterns

When built-in loaders don't match your data structure, use the datasets library as a compatibility layer. Convert your data to a Dataset object, then use ⟦T46⟧ for maximum flexibility and Dataset Viewer compatibility.

### From DataFrames

If you already have your data working in pandas, polars, or other dataframe libraries, you can convert directly:

⟦T7⟧

## Custom Format Conversion

When built-in loaders don't match your data format, convert to Dataset objects following these principles:

### Design Principles

**1. Prefer wide/flat structures over joins**

- Denormalize relational data into single rows for better usability
- Include all relevant information in each example
- Lean towards bigger but more usable data - Hugging Face's infrastructure uses advanced deduplication (XetHub) and Parquet optimizations to handle redundancy efficiently

**2. Use configs for logical dataset variations**

- Beyond train/test/val splits, use configs for different subsets or views of your data
- Each config can have different features or data organization
- Example: language-specific configs, task-specific views, or data modalities

### Conversion Methods

**Small datasets (fits in memory) - use ⟦T47⟧**:

⟦T8⟧

**Large datasets (memory-efficient) - use ⟦T48⟧**:

⟦T9⟧

**Tip**: For large datasets, test with a subset first by adding a limit to your generator or using ⟦T49⟧ after creation.

### Using Configs for Dataset Variations

⟦T10⟧

### Multi-modal Examples

**Text + Audio (speech recognition)**:

⟦T11⟧

**Multiple images per example**:

⟦T12⟧

**Note**: For text + images, consider using ImageFolder with metadata.csv which handles this automatically.

## Essential Features

Features define the schema and data types for your dataset columns. Specifying correct features ensures:

- Proper data handling and type conversion
- Dataset Viewer functionality (e.g., image/audio previews)
- Efficient storage and loading
- Clear documentation of your data structure

For complete feature documentation, see: ⟦T102⟧

### Feature Types Overview

**Basic Types**:

- ⟦T50⟧: Scalar values - ⟦T51⟧, ⟦T52⟧, ⟦T53⟧, ⟦T54⟧, ⟦T55⟧, and other numeric types
- ⟦T56⟧: Categorical data with named classes
- ⟦T57⟧: Lists of any feature type
- ⟦T58⟧: For very large lists

**Media Types** (enable Dataset Viewer previews):

- ⟦T59⟧: Handles various image formats, returns PIL Image objects
- ⟦T60⟧: Audio with array data and optional sampling rate
- ⟦T61⟧: Video files
- ⟦T62⟧: PDF documents with text extraction

**Array Types** (for tensors/scientific data):

- ⟦T63⟧, ⟦T64⟧, ⟦T65⟧, ⟦T66⟧: Fixed or variable-length arrays
- Example: ⟦T67⟧
- First dimension can be ⟦T68⟧ for variable length

**Translation Types**:

- ⟦T69⟧: For translation pairs with fixed languages
- ⟦T70⟧: For translations with varying language pairs

**Note**: New feature types are added regularly. Check the documentation for the latest additions.

## Upload Methods

**Dataset objects (use push_to_hub)**:
Use when you've loaded/converted data using the datasets library

⟦T13⟧

**Pre-existing files (use ⟦T71⟧ / ⟦T72⟧)**:
Use when you have hub-compatible files (e.g., Parquet files) already prepared and organized.

⟦T14⟧

⟦T15⟧

**Important**: Before uploading, verify the files meet repository limits:

- Check folder structure if you have file access: ensure no folder contains >10k 文件
- 要求用户确认：“您的文件是否采用集线器兼容的格式（Parquet/CSV/JSON）并且组织得当？”
- 对于非标准格式，请考虑先转换为Dataset对象以确保兼容性

## 验证**考虑小的重新格式化**：如果数据接近内置加载器格式，建议进行细微更改：

- 重命名列（例如，ImageFolder 的“文件名”→“文件名”）
- 重新组织文件夹（例如，将图像移动到类子文件夹中）
- 重命名文件以匹配预期模式（例如，“data.csv”→“train.csv”）

**预上传**：

- 本地测试：`load_dataset("imagefolder", data_dir="./data")`
- 验证功能是否正常工作：

  ```python
  # Test first example
  print(dataset[0])

  # For images: verify they load
  if 'image' in dataset.features:
      dataset[0]['image']  # Should return PIL Image

  # Check dataset size before upload
  print(f"Size: {len(dataset)} examples")
  ```

- 检查metadata.csv是否有“file_name”列
- 验证相对路径，没有前导斜杠
- 确保没有文件夹>10k个文件

**上传后**：

- 检查查看器：`https://huggingface.co/datasets/username/dataset`
- 测试负载：`load_dataset("username/dataset")`
- 验证保留的功能：`print(dataset.features)`

## 常见问题 → 解决方案

|问题 |解决方案 |
| -------------------------- | ------------------------------------------------ |
| “找不到存储库”|运行`hf auth login` |
|内存错误 |使用`max_shard_size="500MB"` |
|数据集查看器无法工作 |等待5-10分钟，检查README.md配置|
|文件 >50GB |分割成更小的文件 |
| “找不到文件” |在元数据中使用相对路径 |

## 数据集查看器配置**注意**：此部分主要适用于直接上传到 Hub 的数据集（通过 UI 或`upload_large_folder`）。使用 `push_to_hub()` 上传的数据集通常会自​​动配置查看器。

### 自动检测何时工作

数据集查看器自动检测标准结构：

- 文件名为：`train.csv`、`test.json`、`validation.parquet`
- 目录名为：`train/`、`test/`、`validation/`
- 使用分隔符分割名称：`test-data.csv` ✓（不是 `testdata.csv` ✗）

### 手动配置

对于自定义结构，请将 YAML 添加到 README.md 中：

```yaml
---
configs:
  - config_name: default # Required even for single config!
    data_files:
      - split: train
        path: "data/train/*.parquet"
      - split: test
        path: "data/test/*.parquet"
---
```

多种配置示例：

```yaml
---
configs:
  - config_name: english
    data_files: "en/*.parquet"
  - config_name: french
    data_files: "fr/*.parquet"
---
```

### 常见观众问题

- **上传后无查看者**：等待 5-10 分钟进行处理
- **“配置名称错误”**：添加 `config_name` 字段（必填！）
- **未检测到文件**：检查命名模式（需要分隔符）
- **查看器已禁用**：从 README YAML 中删除 `viewer: false`

## 快速模板

```python
# ImageFolder with metadata
dataset = load_dataset("imagefolder", data_dir="./images")
dataset.push_to_hub("username/dataset")

# Memory-efficient upload
dataset.push_to_hub("username/dataset", max_shard_size="500MB")

# Multiple CSV files
dataset = load_dataset('csv', data_files={'train': 'train.csv', 'test': 'test.csv'})
dataset.push_to_hub("username/dataset")
```

## 文档

**核心文档**：[Adding datasets](https://huggingface.co/docs/hub/datasets-adding) | [Dataset viewer](https://huggingface.co/docs/hub/datasets-viewer) | [Storage limits](https://huggingface.co/docs/hub/storage-limits) | [Upload guide](https://huggingface.co/docs/datasets/upload_dataset)

## 数据集卡

提醒用户添加数据集卡 (README.md)：

- 数据集描述和使用
- 许可证信息
- 引文详细信息

详情请参阅[Dataset Cards guide](https://huggingface.co/docs/hub/datasets-cards)。

---

## 附录：特殊情况

### Web数据集结构

对于流式传输大型媒体数据集：- 创建 1-5GB tar 碎片
- 一致的内部结构
- 使用`hf upload`上传

### 科学数据

- HDF5/NetCDF → 使用数组功能转换为 Parquet
- 时间序列 → Array2D(shape=(None, n))
- 复杂的元数据 → 存储为 JSON 字符串

### 社区资源

对于非常专业或定制的格式：

- 在中心搜索类似数据集：`https://huggingface.co/datasets`
- 寻求有关[Hugging Face Forums](https://discuss.huggingface.co/c/datasets/10)的建议
- 加入[Hugging Face Discord](https://hf.co/join/discord)获取实时帮助
- 许多特定领域的格式已经在 Hub 上有示例

### 关于空间的证据
https://huggingface.co/docs/hub/spaces-sdks-docker-evidence.md