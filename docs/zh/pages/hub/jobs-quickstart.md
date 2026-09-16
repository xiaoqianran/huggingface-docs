<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 快速入门

在 Hugging Face CPU 和 GPU 上运行 Python 代码。在本指南中，您将在 CPU 上运行一个简单的命令，然后在 GPU 上使用小型语言模型生成文本。

您需要一个包含 [pre-paid credits](https://huggingface.co/settings/billing) 的 Hugging Face 帐户。有关计算成本，请参阅[Pricing and Billing](./jobs-pricing)。

## 1. 设置 CLI

[Install the Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli#getting-started)，然后登录您的帐户：

```bash
>>> hf auth login
```

## 2. 运行Hello World

在终端中运行此命令：

```bash
>>> hf jobs uv run python -c 'print("Hello from the cloud!")'
```

`hf jobs uv run` 在 Hugging Face 基础设施上的 Python 环境中运行该命令。它默认使用 CPU 并将作业日志传输到您的终端。启动后，你会看到：

```text
Hello from the cloud!
```

CLI 还会打印您的作业 ID 及其页面的链接。打开链接以在浏览器中查看其状态和日志。您可以在 [Jobs page](https://huggingface.co/settings/jobs) 上再次找到您的职位，或者通过以下 CLI 命令使用该 ID。

作业还可以在任何 Docker 映像中运行命令：

```bash
>>> hf jobs run ubuntu echo 'Hello from the cloud!'
```

本指南的其余部分使用 `hf jobs uv run`。请参阅 [Docker Jobs](./jobs-configuration#docker-jobs) 了解何时以及如何使用图像。

## 3. 在 GPU 上运行模型

运行这个准备好的脚本来生成机器人名称。您可以[view it on GitHub](https://github.com/huggingface/hub-docs/blob/main/examples/jobs/hello_gpu.py)或阅读以下代码。

```bash
hf jobs uv run \
    --flavor t4-small \
    --timeout 5m \
    https://raw.githubusercontent.com/huggingface/hub-docs/main/examples/jobs/hello_gpu.py
```

- `--flavor t4-small` 选择具有 NVIDIA T4 GPU 的机器。
- `--timeout 5m` 对作业设置五分钟限制。作业下载模型并在日志中打印其答案。例如：

```text
RoboLearnbot
```

这是完整的脚本：

```python
# /// script
# dependencies = ["torch", "transformers"]
# ///

from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="HuggingFaceTB/SmolLM2-360M-Instruct",
    dtype="float16",
)
messages = [{
    "role": "user",
    "content": "Suggest a name for a robot that helps people learn Python. Answer with only the name.",
}]
outputs = generator(messages, max_new_tokens=48, do_sample=False, return_full_text=False)
print(outputs[0]["generated_text"])
```

[dependency header](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies) 告诉 uv 在作业中安装 `torch` 和 `transformers`。您还可以使用 `--with` 指定依赖关系。您只需要本地的 `hf` CLI。

该脚本在作业的 GPU 上运行 [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct)。 `max_new_tokens` 限制答案长度。

启动时间因硬件可用性、依赖项安装和模型下载而异。

> [!提示]
> 按 Ctrl+C 停止流日志；作业继续运行。要停止作业，请使用 `hf jobs cancel JOB_ID`，将 `JOB_ID` 替换为 CLI 打印的 ID。

您还可以使用 [⟦T21⟧ client](https://huggingface.co/docs/huggingface_hub/guides/jobs) 从 Python 启动相同的作业：

```python
from huggingface_hub import run_uv_job

job = run_uv_job(
    "https://raw.githubusercontent.com/huggingface/hub-docs/main/examples/jobs/hello_gpu.py",
    flavor="t4-small",
    timeout="5m",
)
print(job.url)
```

## 4. 检查结果

使用 GPU 作业的 ID 检查其状态并再次读取其日志：

```bash
>>> hf jobs inspect JOB_ID
>>> hf jobs logs JOB_ID
```

成功运行的状态为`COMPLETED`，其日志包含生成的答案。

作业完成后，答案仍保留在作业日志中。当您调整脚本来生成文件时，[save those results to a bucket or Hub repository](./jobs-manage#persist-your-results)，以便它们在作业中存活下来。

## 尝试你自己的脚本（可选）

将上面的代码复制到`hello_gpu.py`中，编辑`messages`中的提示符，然后运行本地文件：

```bash
hf jobs uv run --flavor t4-small --timeout 5m hello_gpu.py
```

CLI 会自动上传您编辑的脚本。如果您将其保存在其他地方，请将 `hello_gpu.py` 替换为其路径。## 后续步骤

在此示例的基础上构建更大的工作负载：

- [Annotate a dataset with OCR, classification or batch inference](./jobs-examples#uv-scripts)。
- [Fine-tune and save a model](./jobs-examples#guides-to-train-with-jobs) 使用 TRL 或 Unsloth。
- [Read datasets or buckets and save processed results](./jobs-large-datasets)。
- [Run commands in Docker images](./jobs-configuration#docker-jobs)。
- [Use Jobs from a coding agent](./jobs-examples#coding-agent-skills)。

### GGUF 在 LM Studio 中的使用
https://huggingface.co/docs/hub/lmstudio.md