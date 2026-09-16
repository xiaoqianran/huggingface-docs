<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 职位概览

Hugging Face Jobs 在远程 CPU 和 GPU 上运行您的代码。使用它来微调模型、对数据集运行推理或处理数据。

作业在您选择的硬件上的环境中运行命令。您可以从终端或集线器上跟踪其状态和日志。

[Start with the Quickstart](./jobs-quickstart) 用于运行您的第一个 CPU 和 GPU 作业，或 [browse examples](./jobs-examples) 用于适应工作负载。

UV 和类似 Docker 的 CLI uv、run、ps、logs、stats、inspect CPU 和 GPU 为您的工作负载选择硬件 运行您的代码 Python 脚本和 Docker 镜像 按使用量付费 为您使用的计算付费

## 从任何地方运行作业

您可以使用多种工具来运行作业：

* `hf` 命令行界面（请参阅 [CLI installation steps](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) 和 [Jobs CLI documentation](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) 了解更多信息）
* `huggingface_hub` Python 客户端（更多信息请参阅[⟦T4⟧ Jobs documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs)）
* Jobs HTTP API（请参阅[Jobs OpenAPI](https://huggingface-openapi.hf.space/#tag/jobs)了解更多信息）

## 运行任何工作负载

### Python 脚本

使用`hf jobs uv run`远程运行Python脚本。使用 `--with` 或在 [script header](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) 中指定其依赖项。乔布斯在运行代码之前使用 uv 安装这些依赖项。

```diff
- uv run <script.py>
+ hf jobs uv run <script.py>
```

### Docker 镜像将 `hf jobs run` 与 Docker 映像和要运行的命令结合使用。无论您使用的是 Python 还是其他语言，该映像都会提供您的工作负载所需的工具和库。选择一个 [existing image](./jobs-popular-images)，使用由 [Docker Space](./spaces-sdks-docker) 构建的一个，或者构建您自己的。

```diff
- docker run <image> <command>
+ hf jobs run <image> <command>
```

许多作业可以并行运行，用于参数调整、推理和数据处理等任务。

## 自动化作业

通过计划或使用 Webhook 自动触发作业。

通过计划，您可以每 X 分钟、几小时、几天、几周或几个月运行一次作业。调度作业使用 `cron` 语法，如 `"*/5 * * * *"` 表示“每 5 分钟”，或使用 `"@hourly"`、`"@daily"`、`"weekly"` 或 `"@monthly"` 等别名。

借助 Webhooks，只要 Hugging Face 存储库有更新，作业就可以运行。例如，您可以将 Webhook 配置为触发给定帐户下的每个模型更新，并从作业中的 Webhook 负载检索更新的模型。

### 上传模型
https://huggingface.co/docs/hub/models-uploading.md