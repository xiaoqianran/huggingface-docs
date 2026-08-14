<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 评估结果

> [!警告]
> 这是一项正在进行中的功能。

该中心提供了一个去中心化的系统来跟踪模型评估结果。基准数据集托管排行榜，模型存储库存储自动显示在模型页面和基准排行榜上的评估分数。

## 基准数据集

数据集存储库可以定义为**基准**（例如，[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)、[HLE](https://huggingface.co/datasets/cais/hle)、[GPQA](https://huggingface.co/datasets/Idavidrein/gpqa)）。它们显示“基准”标签，并自动汇总整个中心模型存储库的评估结果，并显示顶级模型的排行榜。

![Benchmark Dataset](https://huggingface.co/huggingface/documentation-images/resolve/main/evaluation-results/benchmark-preview.png)

## 模型评估结果

评估分数作为 YAML 文件存储在模型存储库中的 `.eval_results/` 文件夹中。这些结果：

- 出现在模型页面上，并包含基准排行榜的链接
- 汇总到基准数据集的排行榜中
- 可以通过 PR 提交并标记为“社区提供”

![Model Evaluation Results](https://huggingface.co/huggingface/documentation-images/resolve/main/evaluation-results/eval-results-previw.png)

### 添加评估结果

要将评估结果添加到模型中，您可以使用 `.eval_results/` 文件夹中的 YAML 文件向模型存储库提交 PR。

在模型存储库的 `.eval_results/*.yaml` 中创建一个 YAML 文件：

```yaml
- dataset:
    id: cais/hle                  # Required. Hub dataset ID (must be a Benchmark)
    task_id: default              # Required. ID of the Task, as defined in the dataset's eval.yaml
    revision: <hash>              # Optional. Dataset revision hash
  value: 20.90                    # Required. Metric value
  verifyToken: <token>            # Optional. Cryptographic proof of auditable evaluation
  date: "2025-01-15"              # Optional. ISO-8601 date or datetime of when the eval was run (defaults to git commit time)
  source:                         # Optional. Attribution for this result, for instance a repo containing output traces or a Paper
    url: https://huggingface.co/spaces/SaylorTwift/smollm3-mmlu-pro  # Required if source provided
    name: Eval traces             # Optional. Display name
    user: SaylorTwift             # Optional. HF username/org
  notes: "no-tools"               # Optional. Details about the evaluation setup (e.g., "tools", "no-tools", etc.)
```

或者，仅具有必需的属性：

```yaml
- dataset:
    id: Idavidrein/gpqa
    task_id: gpqa_diamond
  value: 0.412
```结果根据 YAML 文件中的元数据显示徽章：

|徽章|状况 |
|------|----------|
|已验证 | `verifyToken` 有效（使用 Inspect-ai 在 HF 作业中运行评估）|
|社区 |通过开放 PR 提交的结果（未合并到 main）|
|排行榜 |基准数据集链接 |
|来源 |评估日志或外部来源的链接 |

有关如何格式化此数据的更多详细信息，请查看 [Eval Results](https://github.com/huggingface/hub-docs/blob/main/eval_results.yaml) 规范。

### 社区贡献

任何人都可以通过 Pull Request 向任何模型提交评估结果：

1. 转到模型页面，单击“社区”选项卡并打开拉取请求。
3. 添加包含结果的 `.eval_results/*.yaml` 文件。
4. PR打开时将在模型页面上显示为“社区提供”。

如需评估模型的帮助，请参阅 [Evaluating models with Inspect](https://huggingface.co/docs/inference-providers/guides/evaluation-inspect-ai) 指南。 Inspect 的评估日志可以直接写入[HF Storage Buckets](./storage-buckets-integrations#inspect-ai)。

> [!提示]
> PR 开放时，社区分数可见。如果分数有争议，模型作者可以关闭 PR 将其删除。目标是透明地公开现有的评估数据，同时通过经过验证的分数建立完全可重复的标准。

## 注册基准

要将您的数据集注册为基准：1. 创建包含评估数据的数据集存储库
2. 使用您的基准配置将 `eval.yaml` 文件添加到存储库根目录，符合下面定义的规范。
3. 文件在推送时验证
4.（**测试版**）联系我们，以便我们将其添加到允许列表中。

可以在这些基准测试中找到示例：[GPQA](https://huggingface.co/datasets/Idavidrein/gpqa/blob/main/eval.yaml)、[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/blob/main/eval.yaml)、[HLE](https://huggingface.co/datasets/cais/hle/blob/main/eval.yaml)、[GSM8K](https://huggingface.co/datasets/openai/gsm8k/blob/main/eval.yaml)。

## Eval.yaml 规范

`eval.yaml` 应包含以下字段：

- `name` — 人类可读的基准显示名称（例如 `"Humanity's Last Exam"`）。
- `description` — 基准测量内容的简短描述。
- `evaluation_framework` — 此基准测试的规范评估框架标识符。这是 Hugging Face 团队维护的一个枚举。将您自己的添加到列表[here](https://github.com/huggingface/huggingface.js/blob/main/packages/tasks/src/eval.ts)。每个基准测试仅支持一个框架。
- `tasks[]` — 此基准定义的任务列表（子排行榜）（见下文）。

每个 `tasks[]` 项目中的必填字段：

- `id` — 任务的唯一标识符。 （例如`"gpqa_diamond"`）。一个基准测试可以定义多个任务，每个任务都会生成自己的排行榜。请随意为每个任务选择一个排行榜标识符。

每个 `tasks[]` 项目中的可选字段：- `config` — 要评估的拥抱脸部数据集的配置（例如`"default"`）。默认为数据集的默认配置。
- `split` — 要评估的拥抱脸部数据集的分割（例如`"test"`）。默认为`"test"`。

设置`evaluation_framework: inspect-ai`时，还需要设置以下字段：

- `field_spec` — 输入和输出字段的规范。包含 `input`、`target`、`choices` 和可选的 `input_image` 子字段。更多详情请参阅[docs](https://inspect.aisi.org.uk/tasks.html#hugging-facehttps://inspect.aisi.org.uk/tasks.html#hugging-face)。
- `solvers` — 求解器过去使用 AI 模型从输入到输出。这可以是从简单的系统提示到自我批评循环。更多详情请参阅[docs](https://inspect.aisi.org.uk/solvers.html)。
- `scores` — 使用得分手。评分者确定求解器是否成功地为数据集中定义的目标找到了正确的输出，以及以什么方式进行。更多详情请参阅[docs](https://inspect.aisi.org.uk/scorers.html)。

最小示例（仅限必填字段）：

```yaml
name: MathArena AIME 2026
description: The American Invitational Mathematics Exam (AIME).
evaluation_framework: math-arena

tasks:
  - id: MathArena/aime_2026
```
扩展示例：

```yaml
name: MathArena AIME 2026
description: The American Invitational Mathematics Exam (AIME).
evaluation_framework: "math-arena"

tasks:
  - id: MathArena/aime_2026
    config: default
    split: test
```

扩展示例（`"inspect-ai"`特定）：

```yaml
name: Humanity's Last Exam
description: >
  Humanity's Last Exam (HLE) is a multi-modal benchmark at the frontier of human
  knowledge, designed to be the final closed-ended academic benchmark of its
  kind with broad subject coverage. Humanity's Last Exam consists of 2,500
  questions across dozens of subjects, including mathematics, humanities, and
  the natural sciences. HLE is developed globally by subject-matter experts and
  consists of multiple-choice and short-answer questions suitable for automated
  grading.
evaluation_framework: "inspect-ai"

tasks:
  - id: hle
    config: default
    split: test

    field_spec:
      input: question
      input_image: image
      target: answer

    solvers:
      - name: system_message
        args:
          template: |
            Your response should be in the following format:
            Explanation: {your explanation for your answer choice}
            Answer: {your chosen answer}
            Confidence: {your confidence score between 0% and 100% for your answer}
      - name: generate

    scorers:
      - name: model_graded_fact
        args:
          model: openai/o3-mini
```

### 后续步骤
https://huggingface.co/docs/hub/repositories-next-steps.md