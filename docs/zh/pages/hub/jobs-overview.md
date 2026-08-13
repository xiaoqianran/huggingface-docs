<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 职位概览

使用熟悉的 UV 和类似 Docker 的界面在 Hugging Face 基础设施上运行计算作业！

UV 和类似 Docker 的 CLI uv,run,ps,logs,stats,inspect 任何硬件 CPU 到 A100s 和 TPU 运行任何东西 UV、Docker、HF 空间等 按使用量付费 仅按使用秒数付费

Hugging Face Hub 通过 Jobs 为人工智能和数据工作流程提供计算。

Jobs 在 Hugging Face 基础设施上运行，旨在为 AI 构建者、数据工程师、开发人员和 AI 代理提供轻松访问云基础设施来运行其工作负载的能力。它们非常适合微调 AI 模型和使用 GPU 运行推理，也适用于数据摄取和处理。

作业是通过要运行的命令（例如 UV 或 python 命令）、硬件风格（CPU、GPU、TPU）以及可选的来自 Hugging Face Spaces 或 Docker Hub 的 Docker 映像来定义的。许多作业可以并行运行，这很有用，例如用于参数调整或并行推理和数据处理。

## 从任何地方运行作业

您可以使用多种工具来运行作业：* `hf` 命令行界面（请参阅 [CLI installation steps](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 和 [Jobs CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) 了解更多信息）
* `huggingface_hub` Python 客户端（更多信息请参阅[⟦T4⟧ Jobs documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs)）
* Jobs HTTP API（更多信息请参阅[Jobs OpenAPI](https://huggingface-openapi.hf.space/#tag/jobs)）

## 运行任何工作负载

`hf` Jobs CLI 和 `huggingface_hub` Python 客户端提供类似 UV 的界面来运行 Python 工作负载。 UV 安装所需的 Python 依赖项并通过一个命令运行 Python 脚本。 Python 依赖项也可以在独立的 UV 脚本中定义，在这种情况下，除了 UV 脚本之外不需要指定任何内容来运行作业。

```diff
- uv run <script.py>
+ hf jobs uv run <script.py>
```

更一般地说，Hugging Face Jobs 支持基于 Docker 和命令的任何工作负载。 Jobs 提供了一个类似 Docker 的界面来 rub Jobs，您可以在其中指定来自 Hugging Face Spaces 或 Docker Hub 的 Docker 镜像，以及要运行的命令。 Docker 提供了将即用环境打包为由社区共享或定制的 Docker 映像的能力。因此，您可以根据工作负载的需要（例如 python、torch、vllm）选择或定义 Docker 映像并运行任何命令。这比使用 UV 更先进，但提供了更大的灵活性。

```diff
- docker run <image> <command>
+ hf jobs run <image> <command>
```

## 自动化作业通过计划或使用 Webhook 自动触发作业。

通过计划，您可以每 X 分钟、每小时、每天、每周或每月运行一次作业。调度作业使用 `cron` 语法，如 `"*/5 * * * *"` 表示“每 5 分钟”，或使用 `"@hourly"`、`"@daily"`、`"weekly"` 或 `"@monthly"` 等别名。

借助 Webhooks，只要 Hugging Face 存储库有更新，作业就可以运行。例如，您可以将 Webhook 配置为触发给定帐户下的每个模型更新，并从作业中的 Webhook 负载检索更新的模型。

### 使用 GPU 空间
https://huggingface.co/docs/hub/spaces-gpus.md