<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 迪斯蒂标签

Distilabel 是一个合成数据和人工智能反馈框架，适用于需要基于经过验证的研究论文的快速、可靠和可扩展管道的工程师。

Distilabel 可用于为各种项目生成合成数据和 AI 反馈，包括传统的预测 NLP（分类、提取等）或生成和大型语言模型场景（指令跟踪、对话生成、判断等）。 Distilabel 的编程方法允许您构建可扩展的管道数据生成和 AI 反馈。 distilabel 的目标是根据经过验证的研究方法快速生成高质量、多样化的数据集，用于生成和判断 AI 反馈，从而加速您的 AI 开发。

## 人们用 distilabel 构建了什么？

Argilla 社区使用 distilabel 创建了令人惊叹的 [datasets](https://huggingface.co/datasets?other=distilabel) 和 [models](https://huggingface.co/models?other=distilabel)。- [1M OpenHermesPreference](https://huggingface.co/datasets/argilla/OpenHermesPreferences) 是使用 [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) LLM 生成的包含约 100 万个 AI 偏好的数据集。这是一个很好的示例，说明如何使用 distilabel 来扩展和增加数据集开发。
- [distilabeled Intel Orca DPO dataset](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs) 用于微调[improved OpenHermes model](https://huggingface.co/argilla/distilabeled-OpenHermes-2.5-Mistral-7B)。该数据集是通过将 Argilla 中的人工管理与 distilabel 的人工智能反馈相结合而构建的，从而形成了英特尔 Orca 数据集的改进版本，并超越了在原始数据集上微调的模型。
- [haiku DPO data](https://github.com/davanstrien/haiku-dpo) 是任何人都可以为特定任务创建合成数据集的示例，该数据集经过管理和评估后可用于微调自定义 LLM。

## 先决条件

第一个[login with your Hugging Face account](/docs/huggingface_hub/quick-start#login)：

```bash
hf auth login
```

确保您已安装`distilabel`：

```bash
pip install -U distilabel[vllm]
```

## Distilabel 管道Distilabel 管道可以使用任意数量的互连步骤或任务来构建。一个步骤或任务的输出作为另一步骤或任务的输入。可以将一系列步骤链接在一起，以使用法学硕士构建复杂的数据处理和生成管道。每个步骤的输入是一批数据，包含一个字典列表，其中每个字典代表数据集的一行，键是列名。为了向 Hugging Face 中心提供数据或向 Hugging Face 中心提供数据，我们定义了 `Distiset` 类作为 `datasets.DatasetDict` 的抽象。

## Distiset 作为数据集对象

distilabel 中的 Pipeline 返回一种特殊类型的 Hugging Face `datasets.DatasetDict`，称为 `Distiset`。

Pipeline 可以在 Distiset 中输出多个子集，Distiset 是一个类似字典的对象，每个子集有一个条目。然后可以将 Distiset 无缝推送到 Hugging Face Hub，所有子集都位于同一存储库中。

## 将数据从 Hub 加载到 Distiset为了展示从 Hub 加载数据的示例，我们将重现 [Prometheus 2 paper](https://arxiv.org/pdf/2405.01535) 并使用 distilabel 中实现的 PrometheusEval 任务。 Prometheus 2 和 Prometheusval 任务直接评估和成对排名任务，即评估给定指令（有或没有参考答案）的单个独立响应的质量，以及分别针对有或没有参考答案的给定指令评估一个响应相对于另一个响应的质量。我们将在从 Hub 加载的数据集上使用这些任务，该数据集由 Hugging Face H4 团队创建，名为 [HuggingFaceH4/instruction-dataset](https://huggingface.co/datasets/HuggingFaceH4/instruction-dataset)。 

```python
from distilabel.llms import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import KeepColumns, LoadDataFromHub
from distilabel.steps.tasks import PrometheusEval

if __name__ == "__main__":
    with Pipeline(name="prometheus") as pipeline:
        load_dataset = LoadDataFromHub(
            name="load_dataset",
            repo_id="HuggingFaceH4/instruction-dataset",
            split="test",
            output_mappings={"prompt": "instruction", "completion": "generation"},
        )

        task = PrometheusEval(
            name="task",
            llm=vLLM(
                model="prometheus-eval/prometheus-7b-v2.0",
                chat_template="[INST] {{ messages[0]['content'] }}\n{{ messages[1]['content'] }}[/INST]",
            ),
            mode="absolute",
            rubric="factual-validity",
            reference=False,
            num_generations=1,
            group_generations=False,
        )

        keep_columns = KeepColumns(
            name="keep_columns",
            columns=["instruction", "generation", "feedback", "result", "model_name"],
        )

        load_dataset >> task >> keep_columns
```

然后我们需要使用运行时参数调用`pipeline.run`，以便启动管道并将数据存储在`Distiset`对象中。

```python
distiset = pipeline.run(
    parameters={
        task.name: {
            "llm": {
                "generation_kwargs": {
                    "max_new_tokens": 1024,
                    "temperature": 0.7,
                },
            },
        },
    },
)
```

## 将 distilabel Distiset 推送到 Hub

将 `Distiset` 推送到 Hugging Face 存储库，其中每个子集将对应于不同的配置：

```python
distiset.push_to_hub(
    "my-org/my-dataset",
    commit_message="Initial commit",
    private=False,
    token=os.getenv("HF_TOKEN"),
)
```

## 📚 资源

- [🚀 Distilabel Docs](https://distilabel.argilla.io/latest/)
- [🚀 Distilabel Docs - distiset](https://distilabel.argilla.io/latest/sections/how_to_guides/advanced/distiset/)
- [🚀 Distilabel Docs - prometheus](https://distilabel.argilla.io/1.2.0/sections/pipeline_samples/papers/prometheus/)
- [🆕 Introducing distilabel](https://argilla.io/blog/introducing-distilabel-1/)

### 存储限制
https://huggingface.co/docs/hub/storage-limits.md