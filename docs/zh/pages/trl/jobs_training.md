<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 工作训练

[⟦T22⟧](https://huggingface.co/models?other=hf_jobs,trl)

[Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/guides/jobs) 让您可以在完全托管的基础设施上运行训练脚本，无需管理 GPU 或本地环境设置。

在本指南中，您将学习如何：

* 使用[TRL Jobs](https://github.com/huggingface/trl-jobs)轻松运行预先优化的TRL训练
* 使用 uv 脚本运行任何 TRL 训练脚本

有关 Hugging Face 作业的一般详细信息（硬件选择、作业监控等），请参阅[Jobs documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs)。

## 要求

* 一个带有积极[credit balance](https://huggingface.co/settings/billing)的拥抱脸账户。 Jobs 是按需付费的——您只需为使用的秒数付费。
* 登录抱脸中心(`hf auth login`)

## 使用 TRL 作业

[TRL Jobs](https://github.com/huggingface/trl-jobs) 是 Hugging Face Jobs 和 TRL 的高级包装，可简化培训。它提供优化的默认配置，因此您无需手动调整参数即可快速启动。

示例：

```bash
pip install trl-jobs
trl-jobs sft --model_name Qwen/Qwen3-0.6B --dataset_name trl-lib/Capybara
```

TRL Jobs 支持本指南中涵盖的所有内容，并进行了额外的优化以简化工作流程。

## 使用 uv 脚本

为了获得更多控制，您可以使用 [uv scripts](https://docs.astral.sh/uv/guides/scripts/) 使用自己的脚本直接运行 Hugging Face Jobs。

创建包含训练代码的 Python 脚本（例如 `train.py`）：

```python
from datasets import load_dataset
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")
trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
)
trainer.train()
trainer.push_to_hub("Qwen2.5-0.5B-SFT")
```

使用 [⟦T17⟧ CLI](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-jobs) 或 Python API 启动作业：

```bash
hf jobs uv run \
    --flavor a100-large \
    --with trl \
    --secrets HF_TOKEN \
    train.py
```

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    dependencies=["trl"],
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
)
```

要成功运行该脚本需要：* **安装 TRL**：使用 `--with trl` 标志或 `dependencies` 参数。 uv 在运行脚本之前自动安装这些依赖项。
* **身份验证令牌**：需要推送经过训练的模型（或执行其他经过身份验证的操作）。为其提供 `--secrets HF_TOKEN` 标志或 `secrets` 参数。

> [!警告]
> 使用乔布斯进行培训时，请务必：
>
> * **设置足够的超时**。默认情况下，作业会在 30 分钟后超时。如果您的作业超过超时时间，它将失败并且所有进度都将丢失。参见[Setting a custom timeout](https://huggingface.co/docs/huggingface_hub/guides/jobs#setting-a-custom-timeout)。
> * **将模型推送到 Hub**。作业环境是短暂的——作业结束时文件将被删除。如果你不推送模型，它就会丢失。

您还可以直接从 URL 运行脚本：

```bash
hf jobs uv run \
    --flavor a100-large \
    --with trl \
    --secrets HF_TOKEN \
    "https://gist.githubusercontent.com/qgallouedec/eb6a7d20bd7d56f9c440c3c8c56d2307/raw/69fd78a179e19af115e4a54a1cdedd2a6c237f2f/train.py"
```

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "https://gist.githubusercontent.com/qgallouedec/eb6a7d20bd7d56f9c440c3c8c56d2307/raw/69fd78a179e19af115e4a54a1cdedd2a6c237f2f/train.py",
    flavor="a100-large",
    dependencies=["trl"],
    secrets={"HF_TOKEN": "hf_..."},
)
```

要使脚本独立，请在顶部声明依赖项：

```python
# /// script
# dependencies = [
#     "trl",
#     "peft",
# ]
# ///

from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer

dataset = load_dataset("trl-lib/Capybara", split="train")

trainer = SFTTrainer(
    model="Qwen/Qwen2.5-0.5B",
    train_dataset=dataset,
    peft_config=LoraConfig(),
)
trainer.train()
trainer.push_to_hub("Qwen2.5-0.5B-SFT")
```

然后您可以运行该脚本而不指定依赖项：

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    train.py
```

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
)
```

TRL 示例脚本完全兼容 uv，因此您可以直接在作业上运行完整的训练工作流程。您可以使用标准脚本参数以及硬件和机密来自定义训练：

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/examples/scripts/prm.py \
    --model_name_or_path Qwen/Qwen2-0.5B-Instruct \
    --dataset_name trl-lib/prm800k \
    --output_dir Qwen2-0.5B-Reward \
    --push_to_hub
```

```python
from huggingface_hub import run_uv_job
run_uv_job(
    "https://raw.githubusercontent.com/huggingface/trl/refs/heads/main/examples/scripts/prm.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
    script_args=[
        "--model_name_or_path", "Qwen/Qwen2-0.5B-Instruct",
        "--dataset_name", "trl-lib/prm800k",
        "--output_dir", "Qwen2-0.5B-Reward",
        "--push_to_hub"
    ]
)
```

请参阅[Example Scripts](example_overview#scripts)中的完整示例列表。

### Docker 镜像[huggingface/trl](https://hub.docker.com/r/huggingface/trl) 提供了具有所有 TRL 依赖项的最新 Docker 映像，并且可以直接与 Hugging Face Jobs 一起使用：

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --image huggingface/trl \
    train.py
```

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
    image="huggingface/trl",
)
```

作业在来自 Hugging Face Spaces 或 Docker Hub 的 Docker 映像上运行，因此您还可以指定任何自定义映像：

```bash
hf jobs uv run \
    --flavor a100-large \
    --secrets HF_TOKEN \
    --image <docker-image> \
    --secrets HF_TOKEN \
    train.py
```

```python
from huggingface_hub import run_uv_job

run_uv_job(
    "train.py",
    flavor="a100-large",
    secrets={"HF_TOKEN": "hf_..."},
    image="<docker-image>",
)
```

### 不懒惰的集成
https://huggingface.co/docs/trl/v1.9.2/unsloth_integration.md