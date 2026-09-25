<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 在工作中训练模型

本页面展示了如何使用库的脚本或您自己的脚本在作业上微调和训练模型。每个示例都限制为短期运行，在单个 A10G 上只需几分钟即可完成，计算费用约为 0.10 美元。该作业在 Hugging Face 的机器上运行，因此即使您关闭终端或笔记本电脑，它也会继续运行。如果您之前没有运行过作业，[Quickstart](./jobs-quickstart) 涵盖了安装 CLI、登录以及作业所需的积分。

## 第一次运行

此命令在 2,000 个 [Food-101](https://huggingface.co/datasets/ethz/food101) 图像上微调图像分类器，并将其推送到 Hub 上的命名空间，大约需要三分钟：

```bash
hf jobs uv run --flavor a10g-small --timeout 30m -s HF_TOKEN -- \
  https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/image-classification/run_image_classification.py \
  --model_name_or_path google/vit-base-patch16-224-in21k \
  --dataset_name ethz/food101 \
  --do_train --do_eval \
  --remove_unused_columns False \
  --max_train_samples 2000 --max_eval_samples 500 --num_train_epochs 1 \
  --output_dir vit-food101 \
  --push_to_hub
```

[Transformers](#transformers) 部分涵盖了完整运行和其他示例脚本。

## 训练作业是如何组合在一起的

上面的命令有五个部分，此页面上的每个命令都有相同的部分。- **运行的内容。** 使用 `hf jobs uv run` 启动的 uv 脚本：一个 Python 脚本，在顶部附近的注释块中声明其依赖项（`# /// script`，[PEP 723](https://peps.python.org/pep-0723/) 格式），uv 将其安装到新环境中。 TRL 部分显示了其中之一。或者使用 `hf jobs run` 启动的库的 Docker 映像，它使用映像中已安装的库。当使用 `pip` 安装库时，从 uv 脚本开始。当库发布带有已编译依赖项的映像时，或其文档要求使用该映像时，请使用该映像。有关权衡，请参阅[Using Docker images](./jobs-images)。
- **一个令牌。** 默认情况下，作业不会获得 Hugging Face 令牌。 `-s HF_TOKEN` 将您的模型作为秘密转发，因此运行可以推送其模型并读取门控或私有输入。其他秘密也以同样的方式传播，例如`-s WANDB_API_KEY`。
- **硬件和时间。** `--flavor` 选择 GPU。 `--timeout` 设置时间限制，默认30分钟。达到超时的运行将停止，因此将 `--timeout` 设置为高于预期运行时间。参见[Hardware flavor](./jobs-configuration#hardware-flavor)和[Timeout](./jobs-configuration#timeout)。- **`hf` 标志和脚本之间有 `--`。** `--` 之前的标志适用于 `hf jobs`。之后是脚本路径和脚本自己的参数。如果没有它，与 `hf` 标志共享名称的脚本参数（例如 `--timeout` 或 `--token`）将由 `hf` 获取。镜像形式中，`--`后面就是在容器中运行的命令。
- **输出的去向。** 下面的每个库都可以将完成的模型推送到 Hub 存储库，每个示例都显示了该选项。对于 Transformers 和 TRL，存储库采用 `--output_dir` 名称。运行还可以在运行时写入已安装的 [bucket](./storage-buckets)，使用 `-v` 将其安装到容器中。参见[After it ends](#after-it-ends)。

**您自己的训练代码。** 使用适合的最简单形式：

- **一个文件。** 在脚本头中声明其依赖项并运行 `hf jobs uv run train.py`。 TRL 部分显示了其中之一。
- **一个项目文件夹**，包含本地导入、`pyproject.toml` 或配置文件。 `hf jobs uv run` 仅上传脚本文件，因此请挂载文件夹。挂载是只读的，因此该命令将项目复制到可写目录并在那里运行它，就像在本地一样：

  ```bash
  hf jobs run --flavor a10g-small --timeout 30m -s HF_TOKEN \
    -v ./my-project:/code ghcr.io/astral-sh/uv:python3.12-bookworm -- \
    bash -c "cp -r /code /tmp/project && cd /tmp/project && uv run train.py --config configs/run.yaml"
  ````uv run` 从 `pyproject.toml` 或脚本标头安装依赖项，本地导入和相对路径保持不变。参见[Local directories](./jobs-configuration#local-directories)。
- **需要系统包或 CUDA 工具包的代码。** 构建一次映像并使用 `hf jobs run` 运行它。参见[Build your own image with a Docker Space](./jobs-images#build-your-own-image-with-a-docker-space)。

脚本还可以在脚本标头的 `[tool.hf-jobs]` 表中携带自己的启动配置，如 TRL 部分所示。参见[Define the launch config in the script](./jobs-configuration#define-the-launch-config-in-the-script)。 Python 中提供了与 `run_uv_job()` 和 `run_job()` 相同的命令，在 [Configuration](./jobs-configuration) 中进行了介绍。

## 长跑前的检查

在取下盖子之前，进行一些检查可以避免浪费运行。- **首先进行冒烟测试。** 在小版本上运行带有步骤上限的命令，如本页上的示例所示。它证明了依赖项安装、数据加载、模型适合以及推送有效。然后取下盖子并启动完整运行。
- **检查数据是否相符。** 每个风味都有一个固定磁盘，列在[Pricing and Billing](./jobs-pricing#pricing)的临时存储列中。权重、数据集和保存的检查点共享它。对于磁盘来说太大的数据集可以流式传输或安装，而不是下载。参见[Process Large Datasets](./jobs-large-datasets)。
- **估计运行时间。** 冒烟测试的最终 `train_*` 指标包括 `train_steps_per_second`，训练器在开始时打印总步数。他们一起提供完整的跑步训练时间。将 `--timeout` 设置在其上方，因为作业还花费时间安装依赖项和下载模型。每种口味的价格为[Pricing and Billing](./jobs-pricing)。
- **在多 GPU 风格上，每个 GPU 启动一个进程。** 以 `x2`、`x4` 或 `x8` 结尾的风格在一台机器上提供多个 GPU。确保您的启动方式使用了它们。 Transformer 和 TRL 需要 `accelerate launch`，如 TRL 部分所示。蝾螈自己做这件事。普通的`python train.py`使用一个GPU，而`Trainer`则回退到`DataParallel`，这比每个GPU一个进程要慢。- **如果要重新运行，则固定。** 将脚本 URL 固定到提交而不是 `main`，并将图像固定到特定标签而不是 `latest`。重新运行会得到相同的软件。下面的脚本 URL 跟踪 `main`，因此请在重新运行之前固定提交。
- **检查长运行**到已安装的存储桶，因此超时或崩溃不会丢失运行。参见[After it ends](#after-it-ends)。

## 当它运行时

`hf jobs run` 和 `hf jobs uv run` 都会传输日志并按住您的终端直到运行结束。 Ctrl+C 仅停止日志流。作业将持续运行，直到完成或使用 `hf jobs cancel <job_id>` 停止它。

对于较长的运行，请传递 `-d`（分离）以立即获取作业 ID，然后使用 `hf jobs logs -f <job_id>` 跟踪运行并确认 GPU 正忙于 `hf jobs stats <job_id>`。分离运行后，`hf jobs wait <job_id>` 会阻塞，直到作业结束，如果失败则以非零值退出，这是脚本或代理循环所需要的。非分离运行已经做到了这一点。参见[Manage Jobs](./jobs-manage)。

作业日志将损失值打印为文本。对于曲线，将训练器指向实验跟踪器（例如 [trackio](https://huggingface.co/docs/trackio)），对于托管跟踪器（例如权重和偏差），将其密钥作为第二个秘密传递。

## 结束后当作业结束时，作业的磁盘将被丢弃，无论是完成、失败还是超时。您想要保留的任何东西都必须在此之前离开容器。

**将模型推送到 Hub 存储库。** 此页面上的每个库都有一个选项：`--push_to_hub` 用于 Transformers 和 TRL，`--output-repo` 用于 Unsloth 脚本，`hub_model_id` 在 Axolotl YAML 中。运行结束时，库上传权重、分词器和生成的模型卡，记录基本模型和训练参数。如果存储库不存在，则创建该存储库。要使其私有，请先使用 `hf repos create <name> --private` 创建它。细粒度令牌需要写入并创建对模型存储库的访问权限。如果没有它，运行会训练到最后，然后上传失败。

**随时写入存储桶。** 对于需要数小时的运行，请挂载现有的 [Storage Bucket](./storage-buckets) 读写（使用 `hf buckets create` 创建一个）并将库的输出目录指向它。检查点在保存时会落入存储桶中，因此超时或崩溃不会丢失运行，并且下一个作业可以从中恢复。同样的路线适用于评估输出、日志和其他任何非模型的内容。

```bash
hf jobs uv run --flavor a10g-large --timeout 8h -s HF_TOKEN \
  -v hf://buckets/your-username/checkpoints:/ckpt -- \
  train.py --output_dir /ckpt/run-01
```Transformer 和 TRL 脚本采用 `--output_dir`。 Axolotl 在 YAML 中采用 `output_dir`。如果您还传递了`--push_to_hub`，请也设置`--hub_model_id`，或者存储库以输出路径的最后一部分（`run-01`）命名。要继续中断的运行，请再次挂载相同的存储桶并传递库的恢复选项，例如 Transformers 的 `--resume_from_checkpoint` `Trainer`。有关安装选项，请参阅[Volumes](./jobs-configuration#volumes)。

**读取失败的运行。**失败的作业会保留其日志：`hf jobs logs <job_id>`在结束后继续工作，`hf jobs inspect <job_id>`给出最终状态和错误消息。无论运行是否成功，`hf jobs logs -f`都会在日志流结束时返回，因此在假设它有效之前检查`inspect`。

## 变形金刚

Transformers 存储库中的 [example scripts](https://github.com/huggingface/transformers/tree/main/examples/pytorch) 在脚本标头中声明它们的依赖项，因此它们直接从 GitHub URL 在作业上运行。 URL 后面的参数转到脚本。 [A first run](#a-first-run) 使用图像分类脚本。对于完整运行，从该命令中删除 `--max_train_samples 2000 --max_eval_samples 500 --num_train_epochs 1`：75,000 个 Food-101 训练图像上的三个 epoch 在 `a10g-small` 上大约需要一个小时，按照该风味的速度大约需要 1 美元，并达到 90% 的准确率。在启动之前将 `--timeout` 调至 `2h`。 `--push_to_hub` 使用输出目录名称在您的命名空间下上传模型。脚本可用于文本分类、摘要、翻译、标记分类、语音识别等。

## TRL

[TRL](https://huggingface.co/docs/trl) 为每个训练器（SFT、DPO、GRPO 等）都有一个可立即运行的脚本，每个训练器都声明自己的依赖项，因此它们可以直接从其 URL 在作业上运行，就像 Transformers 脚本一样。此命令对 [chat dataset](https://huggingface.co/datasets/trl-lib/Capybara) 上的小模型进行微调：

```bash
hf jobs uv run --flavor a10g-small --timeout 30m -s HF_TOKEN -- \
  https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/trl/scripts/sft.py \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --dataset_name trl-lib/Capybara \
  --max_steps 100 \
  --output_dir Qwen2-0.5B-SFT \
  --push_to_hub
```

这将在大约六分钟内完成。删除 `--max_steps` 以进行完整运行：Capybara 的三个纪元（脚本的默认值）在 `a10g-small` 上花费大约 2 小时 20 分钟，因此用它来提高 `--timeout`。 TRL自己的文档使用`a100-large`，速度更快。

完整指南（包括编写您自己的 TRL 脚本和运行 `huggingface/trl` 映像）位于 TRL 文档中的 [Training with Jobs](https://huggingface.co/docs/trl/jobs_training)。

对于多个 GPU，切换到 TRL 映像，该映像附带 `accelerate`，并让它为每个 GPU 启动一个进程：

```bash
hf jobs run --flavor a10g-largex2 --timeout 30m -s HF_TOKEN huggingface/trl -- \
  accelerate launch --num_processes 2 -m trl.scripts.sft \
  --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
  --dataset_name trl-lib/Capybara \
  --max_steps 100 \
  --output_dir Qwen2-0.5B-SFT \
  --push_to_hub
```如果没有 `--max_steps`，完整运行（Capybara 的三个 epoch，脚本的默认值）在两台 A10G 上大约需要 1 小时 35 分钟。

当您编写自己的 TRL 脚本时，启动配置可以随之移动。标头中的 `[tool.hf-jobs]` 表设置风味、超时和秘密：

```python
# /// script
# dependencies = ["trl"]
#
# [tool.hf-jobs]
# flavor  = "a10g-small"
# timeout = "1h"
# secrets = ["HF_TOKEN"]
# ///
from trl import SFTConfig, SFTTrainer
...
```

`hf jobs uv run train.py` 则不需要标志，并且您通过的标志仍然获胜。参见[Define the launch config in the script](./jobs-configuration#define-the-launch-config-in-the-script)。

## 不懒惰

[Unsloth](https://unsloth.ai) 在 [⟦T88⟧](https://huggingface.co/datasets/unsloth/jobs) 数据集中提供可立即运行的脚本，每个模型系列一个。他们从脚本头安装 Unsloth 并将数据集和输出存储库作为参数：

```bash
hf jobs uv run --flavor a10g-small --timeout 30m -s HF_TOKEN -- \
  https://huggingface.co/datasets/unsloth/jobs/resolve/main/sft-lfm2.5.py \
  --dataset mlabonne/FineTome-100k \
  --max-steps 50 \
  --output-repo your-username/lfm-finetuned
```

这将在大约五分钟内完成并推送 LoRA 适配器。对于完整的纪元，将`--max-steps 50`替换为`--num-epochs 1`并提高`--timeout`。演练是 Hugging Face 博客上的[Fine-tune with Unsloth on Jobs](https://huggingface.co/blog/unsloth-jobs)。

## 蝾螈

[Axolotl](https://docs.axolotl.ai) 采用 YAML 配置并从其自己的 Docker 映像运行，因此本节使用带有固定标签的 `hf jobs run` 并将本地目录中的配置与 `-v` 同步。将 [Axolotl examples](https://github.com/axolotl-ai-cloud/axolotl/tree/main/examples) 中的配置保存为 `./configs/lora.yml` 并添加命令后显示的键。本示例使用`examples/phi/lora-3.5.yaml`，Phi-3.5-mini 的 LoRA 微调，其他方面不变。该映像已包含 `axolotl` 命令。

```bash
hf jobs run --flavor a10g-small --timeout 30m -s HF_TOKEN \
  -v ./configs:/configs \
  axolotlai/axolotl:0.19.0-py3.12-cu130-2.13.0 -- \
  axolotl train /configs/lora.yml
````-v ./configs:/configs` 将本地 `configs` 目录上传到您的私有 `jobs-artifacts` 存储桶（在首次使用时为您创建）并将其以只读方式挂载在容器中，因此磁盘上的 YAML 是运行使用的 YAML。输出在 YAML 中设置。这些键在运行结束时推动模型，并且 `max_steps` 限制此试运行：

```yaml
hub_model_id: your-username/my-adapter
hub_strategy: end
max_steps: 20
```

这将在大约七分钟内完成，并将适配器推送到`hub_model_id`，作为私人存储库。移除`max_steps`以进行完整运行。图像标签格式在[Axolotl's Docker guide](https://docs.axolotl.ai/docs/docker.html#sec-main-tags)中描述。

对于更多 GPU，只需更改风格即可：在 `a10g-largex4` 上，`axolotl train` 为每个 GPU 单独启动一个进程。 DeepSpeed 和 FSDP 就是 YAML 密钥的问题，在 [Axolotl's multi-GPU guide](https://docs.axolotl.ai/docs/multi-gpu.html) 中介绍。

## 更进一步- [Serve Models](./jobs-serving) 将您训练的模型放在临时端点后面，以进行评估运行或演示。 [Inference Endpoints](https://huggingface.co/docs/inference-endpoints) 运行一个持续运行的系统。
- [Configuration](./jobs-configuration) 用于秘密、环境变量、卷和 `[tool.hf-jobs]` 表，让脚本携带自己的风格和超时。
- [Manage Jobs](./jobs-manage) 用于列出、检查、调试和取消作业。
- [Process Large Datasets](./jobs-large-datasets) 用于流式传输和安装不适合磁盘的数据。
- [Schedule Jobs](./jobs-schedule) 在计时器上运行训练命令。
- [Examples & Tutorials](./jobs-examples) 用于社区撰写，包括视觉语言微调和将大型数据集流式传输到训练运行中。
- [Use Jobs from a coding agent](./jobs-examples#coding-agent-skills)：`hf` CLI 技能可让 Claude Code、Codex 和 Cursor 启动并为您观看这些运行。

### 空间作为 API 端点
https://huggingface.co/docs/hub/spaces-api-endpoints.md