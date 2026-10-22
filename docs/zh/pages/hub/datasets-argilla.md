<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 阿尔吉拉

Argilla 是一款协作工具，适用于需要为其项目构建高质量数据集的人工智能工程师和领域专家。

![image](https://github.com/user-attachments/assets/0e6ce1d8-65ca-4211-b4ba-5182f88168a0)

Argilla 可用于收集各种 AI 项目的人类反馈，例如传统 NLP（文本分类、NER 等）、LLM（RAG、偏好调整等）或多模态模型（文本到图像等）。 Argilla 的编程方法可让您构建用于持续评估和模型改进的工作流程。 Argilla 的目标是通过快速迭代正确的数据和模型来确保您的数据工作获得回报。

## 人们用 Argilla 构建什么？

社区使用 Argilla 创建了令人惊叹的开源 [datasets](https://huggingface.co/datasets?library=library:argilla&sort=trending) 和 [models](https://huggingface.co/models?other=distilabel)。 

### 开源数据集和模型

Argilla 也向开源贡献了一些模型和数据集。- [Cleaned UltraFeedback dataset](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned) 用于微调 [Notus](https://huggingface.co/argilla/notus-7b-v1) 和 [Notux](https://huggingface.co/argilla/notux-8x7b-v1) 模型。原始 UltraFeedback 数据集是使用 Argilla UI 过滤器进行整理的，以查找并报告原始数据生成代码中的错误。基于此数据管理流程，Argilla 构建了新版本的 UltraFeedback 数据集并对 Notus 进行了微调，在多个基准测试中表现优于 Zephyr。
- [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) 用于微调[improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B)。该数据集是通过将 Argilla 中的人工管理与 distilabel 的人工智能反馈相结合而构建的，从而形成了英特尔 Orca 数据集的改进版本，并超越了在原始数据集上微调的模型。

### 示例 用例

[the Red Cross](https://510.global/)、[Loris.ai](https://loris.ai/)、[Prolific](https://www.prolific.com/)等公司的AI团队使用Argilla来提高AI项目的质量和效率。他们在我们的[AI community meetup](https://lu.ma/embed-checkout/evt-IQtRiSuXZCIW6FB)分享了他们的经验。- 人工智能造福人类：[the Red Cross presentation](https://youtu.be/ZsCqrAhzkFU?feature=shared)展示了红十字会领域专家和人工智能团队如何合作，对乌克兰危机难民的请求进行分类和重定向，以简化红十字会的支持流程。
- 客户支持：在[the Loris meetup](https://youtu.be/jWrtgf2w4VU?feature=shared)期间，他们展示了他们的人工智能团队如何使用无监督和少样本对比学习来帮助他们快速验证并获得大量多标签分类器的标记样本。
- Research studies: [the showcase from Prolific](https://youtu.be/ePDlhIxnuAs?feature=shared) announced their integration with our platform. They use it to actively distribute data collection projects among their annotating workforce. This allows Prolific to quickly and efficiently collect high-quality data for research studies.

## Prerequisites

First [login with your Hugging Face account](/docs/huggingface_hub/quick-start#login):

```bash
hf auth login
```

确保您已安装`argilla>=2.0.0`：

```bash
pip install -U argilla
```

Lastly, you will need to deploy the Argilla server and UI, which can be done [easily on the Hugging Face Hub](https://argilla-io.github.io/argilla/latest/getting_started/quickstart/#run-the-argilla-server).

## 导入和导出数据集和记录

This guide shows how to import and export your dataset to the Hugging Face Hub.在 Argilla 中，您可以导入/导出数据集的两个主要组成部分：
- `rg.Settings`中定义的数据集的完整配置。如果您想共享您的反馈任务或稍后在 Argilla 中恢复它，这非常有用。
- 数据集中存储的记录，包括`Metadata`、`Vectors`、`Suggestions`、`Responses`。如果您想在 Argilla 之外使用数据集的记录，这非常有用。

### 将 Argilla 数据集推送到 Hugging Face Hub

您可以将数据集从 Argilla 推送到 Hugging Face Hub。如果您想与社区共享数据集或对其进行版本控制，这非常有用。您可以使用 `rg.Dataset.to_hub` 方法将数据集推送到 Hugging Face Hub。

```python
import argilla as rg

client = rg.Argilla(api_url="<api_url>", api_key="<api_key>")
dataset = client.datasets(name="my_dataset")
dataset.to_hub(repo_id="<repo_id>")
```

#### 有或没有记录
    
上面的示例将数据集的 `Settings` 和记录推送到集线器。如果您只想推送数据集的配置，可以将`with_records`参数设置为`False`。如果您只对特定数据集模板感兴趣或者想要更改数据集设置和/或记录，这非常有用。

```python
dataset.to_hub(repo_id="<repo_id>", with_records=False)
```

### 从 Hugging Face Hub 中提取 Argilla 数据集You can pull a dataset from the Hugging Face Hub to Argilla. This is useful if you want to restore a dataset and its configuration. You can pull the dataset from the Hugging Face Hub using the `rg.Dataset.from_hub` method.

```python

import argilla as rg

client = rg.Argilla(api_url="<api_url>", api_key="<api_key>")
dataset = rg.Dataset.from_hub(repo_id="<repo_id>")
```

The `rg.Dataset.from_hub` method loads the configuration and records from the dataset repo. If you only want to load records, you can pass a `datasets.Dataset` object to the `rg.Dataset.log` method. This enables you to configure your own dataset and reuse existing Hub datasets. 

#### 有或没有记录

The example above will pull the dataset's `Settings` and records from the hub. If you only want to pull the dataset's configuration, you can set the `with_records` parameter to `False`.如果您只对特定数据集模板感兴趣或者想要更改数据集设置和/或记录，这非常有用。

```python
dataset = rg.Dataset.from_hub(repo_id="<repo_id>", with_records=False)
```

With the dataset's configuration you could then make changes to the dataset. For example, you could adapt the dataset's settings for a different task:

```python
dataset.settings.questions = [rg.TextQuestion(name="answer")]
```

You could then log the dataset's records using the `load_dataset` method of the `datasets` package and pass the dataset to the `rg.Dataset.log` method.

```python
hf_dataset = load_dataset("<repo_id>")
dataset.log(hf_dataset)
```

## 📚 资源

- [🚀 Argilla Docs](https://argilla-io.github.io/argilla/)
- [🚀 Argilla Docs - import export guides](https://argilla-io.github.io/argilla/latest/how_to_guides/import_export/)### 在拥抱脸部使用 ESPnet
https://huggingface.co/docs/hub/espnet.md