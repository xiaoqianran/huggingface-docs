<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 数据设计师

[Data Designer](https://github.com/NVIDIA-NeMo/DataDesigner) 是 NVIDIA NeMo 的框架，用于使用 LLM 生成高质量的合成数据集。它使您能够使用统计采样器、LLM 或现有种子数据集创建不同的数据。

## 先决条件

```bash
pip install data-designer
```

## 从 Hub 下载数据集作为种子

使用`HuggingFaceSeedSource`直接从Hub加载数据集作为种子数据进行生成。

```python
import data_designer.config as dd
from data_designer.interface import DataDesigner

data_designer = DataDesigner()
config_builder = dd.DataDesignerConfigBuilder()

# Load seed data from HuggingFace
seed_source = dd.HuggingFaceSeedSource(
    path="datasets/gretelai/symptom_to_diagnosis/data/train.parquet",
    token="hf_...",  # Optional, for private datasets
)
config_builder.with_seed_dataset(seed_source)

# Reference seed columns in prompts
config_builder.add_column(
    dd.LLMTextColumnConfig(
        name="physician_notes",
        model_alias="openai-gpt-5",
        prompt="Write notes for a patient with {{ diagnosis }}. Symptoms: {{ patient_summary }}",
    )
)

preview = data_designer.preview(config_builder, num_records=5)
```

## 将生成的数据集推送到 Hub

使用内置的`push_to_hub`方法将生成的数据集上传到Hub。

```python
# Generate dataset
results = data_designer.create(config_builder, num_records=1000, dataset_name="my-dataset")

# Push to Hub
url = results.push_to_hub(
    repo_id="username/my-synthetic-dataset",
    description="Synthetic dataset generated with Data Designer.",
    tags=["medical", "notes"],
    private=False,
)
```

## 资源

- [Data Designer Documentation](https://nvidia-nemo.github.io/DataDesigner/)
- [GitHub Repository](https://github.com/NVIDIA-NeMo/DataDesigner)
- [Seed Datasets Guide](https://nvidia-nemo.github.io/DataDesigner/latest/concepts/seed-datasets/)
- [Guide to using Data Designer with Inference Providers](https://huggingface.co/docs/inference-providers/integrations/datadesigner)

### 数据集下载统计
https://huggingface.co/docs/hub/datasets-download-stats.md