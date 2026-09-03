<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 运行和管理作业

Hugging Face Hub 通过 Jobs 为人工智能和数据工作流程提供计算。

**有关工作和定价的总体概述，请参阅 [Hub Jobs documentation](https://huggingface.co/docs/hub/jobs)。**

作业在 Hugging Face 基础设施上运行，并使用要运行的命令（例如 python 命令）、来自 Hugging Face Spaces 或 Docker Hub 的 Docker 映像以及硬件风格（CPU、GPU、TPU）进行定义。本指南将向您展示如何与 Hub 上的作业进行交互，特别是：

- 运行作业。
- 检查工作状态。
- 选择硬件。
- 配置环境变量和秘密。
- 运行 UV 脚本。

如果您想在 Hub 上运行和管理作业，您的机器必须登录。如果您没有登录，请参阅
[this section](../quick-start#authentication)。在本指南的其余部分中，我们将假设您的计算机已登录。

> [!提示]
> **拥抱脸部工作**适用于任何具有积极[credit balance](https://huggingface.co/settings/billing) 的用户或组织。详情请参阅[Jobs pricing and billing](https://huggingface.co/docs/hub/jobs-pricing)。

## 作业命令行界面

使用 [⟦T39⟧ CLI](./cli#hf-jobs) 从命令行运行作业，并传递 `--flavor` 指定您的硬件。

`hf jobs run` 使用 Docker 映像和命令以及熟悉的类似 Docker 的界面来运行作业。想想`docker run`，但对于在任何硬件上运行代码：

```bash
>>> hf jobs run --name hello-world python:3.12 python -c "print('Hello world')"
>>> hf jobs run --name gpu-check --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python -c "import torch; print(torch.cuda.get_device_name())"
```

使用`hf jobs uv run`运行本地或远程UV脚本：

```bash
>>> hf jobs uv run --name my-script my_script.py
>>> hf jobs uv run --name sft-training --flavor a10g-small "https://raw.githubusercontent.com/huggingface/trl/main/trl/scripts/sft.py"
```UV 脚本是 Python 脚本，使用 [UV documentation](https://docs.astral.sh/uv/guides/scripts/) 中定义的特殊注释语法将其依赖项直接包含在文件中。

现在本指南的其余部分将向您展示 python API。
如果您想查看所有可用的 `hf jobs` 命令和选项，请查看 [guide on the ⟦T45⟧ command line interface](./cli#hf-jobs)。

> [!提示]
> 需要一台“交互式”机器，而不是一劳永逸的工作——例如运行人工智能生成的代码，通过命令执行和文件传输？查看基于 Jobs 构建的 [Sandboxes](./sandbox)。

## 运行作业

在 Hugging Face 基础设施（包括 GPU 和 TPU）上运行使用命令和 Docker 映像定义的计算作业。

您只能管理您拥有的作业（在您的用户名命名空间下）或来自您具有写入权限的组织的作业。
此功能是按使用量付费的：您只需为使用的秒数付费。

[run_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_job) 允许您在 Hugging Face 的基础设施上运行任何命令：

```python
# Directly run Python code
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello from the cloud!')"],
... )

# Use GPUs without any setup
>>> run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "-c", "import torch; print(torch.cuda.get_device_name())"],
...     flavor="a10g-small",
... )

# Run in an organization account
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Running in an org account')"],
...     namespace="my-org-name",
... )

# Run from Hugging Face Spaces
>>> run_job(
...     image="hf.co/spaces/lhoestq/duckdb",
...     command=["duckdb", "-c", "select 'hello world'"],
... )

# Run a Python script with `uv` (experimental)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")
```

> [!警告]
> **重要**：作业有默认超时（30 分钟），之后它们将自动停止。对于模型训练等长时间运行的任务，请确保使用 `timeout` 参数设置自定义超时。详情请参阅[Configure Job Timeout](#configure-job-timeout)。[run_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_job) 返回[JobInfo](/docs/huggingface_hub/v1.30.0/en/package_reference/jobs#huggingface_hub.JobInfo)，其中包含 Hugging Face 上作业的 URL，您可以在其中查看作业状态和日志。
保存[JobInfo](/docs/huggingface_hub/v1.30.0/en/package_reference/jobs#huggingface_hub.JobInfo)的作业ID以管理作业：

```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello from the cloud!')"]
... )
>>> job.url
https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a
>>> job.id
687f911eaea852de79c4a50a
```

作业在后台运行。下一节将引导您通过 [inspect_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.inspect_job) 了解作业状态、[fetch_job_logs()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_job_logs) 查看日志以及 [fetch_job_metrics()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_job_metrics) 监控资源使用情况。

## 检查作业状态

```python
# List your jobs (results are paginated and returned as an iterator)
>>> from huggingface_hub import list_jobs
>>> jobs = list_jobs()
>>> next(iter(jobs))
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# Materialize the iterator into a list
>>> all_jobs = [job for job in list_jobs()]

# List your running jobs
>>> running_jobs = list_jobs(status="RUNNING")

# Filter by one or more statuses and/or labels
>>> list_jobs(status=["RUNNING", "SCHEDULING"], labels={"env": "prod"})

# Inspect the status of a job
>>> from huggingface_hub import inspect_job
>>> inspect_job(job_id=job_id)
JobInfo(id='687f911eaea852de79c4a50a', created_at=datetime.datetime(2025, 7, 22, 13, 24, 46, 909000, tzinfo=datetime.timezone.utc), docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', status=JobStatus(stage='COMPLETED', message=None), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687f911eaea852de79c4a50a')

# View logs from a job
>>> from huggingface_hub import fetch_job_logs
>>> for log in fetch_job_logs(job_id=job_id):
...     print(log)
Hello from the cloud!

# View resources usage metrics from a job
>>> from huggingface_hub import fetch_job_metrics
>>> for metrics in fetch_job_metrics(job_id=job_id):
...     print(metrics)
{
    "cpu_usage_pct": 0,
    "cpu_millicores": 2000,
    "memory_used_bytes": 929792,
    "memory_total_bytes": 17179869184,
    "rx_bps": 0,
    "tx_bps": 0,
    "gpus": {},
    "replica": "4dzsh"
}

# Cancel a job
>>> from huggingface_hub import cancel_job
>>> cancel_job(job_id=job_id)
```

## 等待作业完成

使用 [wait_for_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.wait_for_job) 进行阻塞，直到作业到达终止阶段（`COMPLETED`、`CANCELED`、`ERROR` 或 `DELETED`）。最终的[JobInfo](/docs/huggingface_hub/v1.30.0/en/package_reference/jobs#huggingface_hub.JobInfo)总是返回——失败的作业不会引发异常——所以检查`job.status.stage`以对结果采取行动。传递作业 ID 列表以立即等待整批作业。

```python
>>> from huggingface_hub import run_job, wait_for_job
>>> job = run_job(image="python:3.12", command=["python", "-c", "print('hello')"])
>>> wait_for_job(job_id=job.id).status.stage
'COMPLETED'

# Run multiple jobs in parallel and wait for all of them to finish
>>> jobs = [run_job(image=image, command=command) for command in commands]
>>> finished_jobs = wait_for_job(job_id=[job.id for job in jobs], timeout=3600)
```

CLI 中的 `hf jobs wait` 也提供了同样的功能，只有当所有作业成功完成时，它才会以代码 0 退出——这对于在 shell 脚本或 CI 中链接命令很方便：

```bash
# Chain on success
hf jobs wait <job_id> && hf jobs run --detach python:3.12 python eval.py

# Wait for all currently running jobs
hf jobs ls -q | xargs hf jobs wait
```

请注意，如果作业失败，非分离的 `hf jobs run` （或 `hf jobs uv run`）也会以非零代码退出，因此 `hf jobs run ... && next-step` 可以正确链接，无需显式等待。

## 选择硬件

在许多情况下，在 GPU 上运行作业很有用：- **模型训练**：在 GPU（T4、A10G、A100）上微调或训练模型，无需管理基础设施
- **合成数据生成**：在强大的硬件上使用法学硕士生成大规模数据集
- **数据处理**：使用高 CPU 配置处理海量数据集以实现并行工作负载
- **批量推理**：使用优化的 GPU 设置对数千个样本运行离线推理
- **实验和基准**：在一致的硬件上运行机器学习实验以获得可重现的结果
- **开发和调试**：无需本地 CUDA 设置即可测试 GPU 代码

使用 `flavor` 参数在 GPU 或 TPU 上运行作业。例如，要在 A10G GPU 上运行 PyTorch 作业：

```python
# Use an A10G GPU to check PyTorch CUDA
>>> from huggingface_hub import run_job
>>> run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "-c", "import torch; print(f'This code ran with the following GPU: {torch.cuda.get_device_name()}')"],
...     flavor="a10g-small",
... )
```

运行此命令将显示以下输出！

```bash
This code ran with the following GPU: NVIDIA A10G
```

使用它来运行带有 UV 的微调脚本，例如 [trl/scripts/sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py)：

```python
>>> from huggingface_hub import run_uv_job
>>> run_uv_job(
...     "sft.py",
...     script_args=["--model_name_or_path", "Qwen/Qwen2-0.5B", ...],
...     dependencies=["trl"],
...     env={"HF_TOKEN": ...},
...     flavor="a10g-small",
... )
```

> [!提示]
> 有关在 Hugging Face 基础设施上使用 TRL 运行模型训练作业的综合指南，请查看 [TRL Jobs Training documentation](https://huggingface.co/docs/trl/main/en/jobs_training)。它涵盖了微调配方、硬件选择以及有效训练模型的最佳实践。

以下是运行作业的可用硬件的完整列表：|名称 |漂亮的名字|中央处理器|公羊|存储|加速器|成本/分钟 |成本/小时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cpu-basic` | CPU基础| 2 个虚拟CPU | 16GB| 50GB| - | 0.0002 美元 | 0.01 美元 |
| `cpu-upgrade` | CPU升级| 8 个 vCPU | 32GB| 50GB| - | 0.0005 美元 | 0.03 美元 |
| `cpu-performance` | CPU性能| 32 个 vCPU | 256 GB | 256 GB 1024 GB | 1024 GB - | 0.0317 美元 | 1.90 美元 |
| `cpu-xl` |中央处理器XL | 16 个 vCPU | 124GB| 1000 GB | - | 0.0167 美元 | 1.00 美元 |
| `t4-small` | Nvidia T4 - 小| 4 个虚拟CPU | 15GB| 50GB| 1 个 T4 (16 GB) | 0.0067 美元 | 0.40 美元 |
| `t4-medium` | Nvidia T4 - 中 | 8 个 vCPU | 30GB| 100GB| 1 个 T4 (16 GB) | 0.0100 美元 | 0.60 美元 |
| `a10g-small` | Nvidia A10G - 小| 4 个虚拟CPU | 15GB| 110 GB | 110 GB 1 个 A10G (24 GB) | 0.0167 美元 | 1.00 美元 |
| `a10g-large` | Nvidia A10G - 大| 12 个 vCPU | 46GB| 200GB| 1 个 A10G (24 GB) | 0.0250 美元 | 1.50 美元 |
| `a10g-largex2` | 2x Nvidia A10G - 大 | 24 个 vCPU | 92GB| 1000 GB | 2 个 A10G (48 GB) | 0.0500 美元 | $3.00 |
| `a10g-largex4` | 4x Nvidia A10G - 大 | 48 个 vCPU | 184 GB | 184 GB 2000 GB | 4 个 A10G (96 GB) | 0.0833 美元 | 5.00 美元 |
| `a100-large` | Nvidia A100 - 大| 12 个 vCPU | 142 GB | 142 GB 1000 GB | 1 个 A100 (80 GB) | 0.0417 美元 | 2.50 美元 |
| `a100x4` | 4x Nvidia A100 | 48 个 vCPU | 568GB| 4000 GB | 4 个 A100 (320 GB) | 0.1667 美元 | 10.00 美元 || `a100x8` | 8x Nvidia A100 | 96 个 vCPU | 1136 GB | 1136 GB 8000 GB | 8 个 A100 (640 GB) | 0.3333 美元 | 20.00 美元 |
| `h200` |英伟达 H200 | 23 个 vCPU | 256 GB | 256 GB 3000 GB | 1 个 H200 (141 GB) | 0.0833 美元 | 5.00 美元 |
| `h200x2` |英伟达 H200 | 46 个 vCPU | 512GB| 6000 GB | 2 个 H200 (282 GB) | 0.1667 美元 | 10.00 美元 |
| `h200x4` |英伟达 H200 | 92 个 vCPU | 1024 GB | 1024 GB 12000 GB | 4 个 H200 (564 GB) | 0.3333 美元 | 20.00 美元 |
| `h200x8` |英伟达 H200 | 184 个 vCPU | 2048GB | 24000 GB | 8 个 H200 (1128 GB) | 0.6667 美元 | 40.00 美元 |
| `rtx-pro-6000` |英伟达 RTX PRO 6000 | 23 个 vCPU | 256 GB | 256 GB 475GB| 1 个 RTX PRO 6000 (96 GB) | 0.0458 美元 | 2.75 美元 |
| `rtx-pro-6000x2` |英伟达 RTX PRO 6000 | 46 个 vCPU | 512GB| 950GB| 2 个 RTX PRO 6000 (192 GB) | 0.0917 美元 | 5.50 美元 |
| `rtx-pro-6000x4` |英伟达 RTX PRO 6000 | 92 个 vCPU | 1024 GB | 1024 GB 1900GB| 4 个 RTX PRO 6000 (384 GB) | 0.1833 美元 | 11.00 美元 |
| `rtx-pro-6000x8` |英伟达 RTX PRO 6000 | 184 个 vCPU | 2048GB | 3800 GB | 3800 GB 8 个 RTX PRO 6000 (768 GB) | 0.3667 美元 | 22.00 美元 |
| `l4x1` | 1x Nvidia L4 | 8 个 vCPU | 30GB| 400GB| 1 个 L4 (24 GB) | 0.0133 美元 | 0.80 美元 |
| `l4x4` | 4x Nvidia L4 | 48 个 vCPU | 186 GB | 186 GB 3200 GB | 3200 GB 4 个 L4 (96 GB) | 0.0633 美元 | 3.80 美元 |
| `l40sx1` | 1x Nvidia L40S | 8 个 vCPU | 62GB| 380GB| 1 个 L40S (48 GB) | 0.0300 美元 | 1.80 美元 || `l40sx4` | 4x Nvidia L40S | 48 个 vCPU | 382GB| 3200 GB | 3200 GB 4 个 L40S (192 GB) | 0.1383 美元 | 8.30 美元 |
| `l40sx8` | 8x Nvidia L40S | 192 个 vCPU | 1534GB| 6500 GB | 6500 GB 8 个 L40S (384 GB) | $0.3917 | $0.3917 23.50 美元 |

您可以通过运行以下命令以编程方式获取此列表：

```bash
>>> hf jobs hardware
```

或者使用Python API：

```python
>>> from huggingface_hub import list_jobs_hardware
>>> list_jobs_hardware()
```

就是这样！您现在正在 Hugging Face 的基础设施上运行代码。

## 挂载卷

使用 [Volume](/docs/huggingface_hub/v1.30.0/en/package_reference/jobs#huggingface_hub.Volume) 列表在作业磁盘上挂载卷。

您可以安装任何 Hugging Face 存储库（模型/数据集/空间）或 [Storage Bucket](/docs/hub/storage-buckets)。例如：

* 挂载模型库：`Volume(type="model", source="openai/gpt-oss-120b", mount_path="/model")`
* 挂载数据集存储库：`Volume(type="dataset", source="HuggingFaceFW/fineweb", mount_path="/data")`
* 安装储物桶：`Volume(type="bucket", source="username/my-bucket", mount_path="/mnt")`

然后你可以使用挂载的卷作为本地目录：

```python
>>> from huggingface_hub import run_job, Volume
>>> job = run_job(
...     image="duckdb/duckdb",
...     command=["duckdb", "-c", "SELECT * FROM '/data/**/*.parquet' LIMIT 5"],
...     volumes=[Volume(type="dataset", source="HuggingFaceFW/fineweb", mount_path="/data")],
... )
```

例如，您还可以写入已安装的存储桶，以在训练模型时保存检查点：

```python
>>> from huggingface_hub import run_uv_job, Volume
>>> script = "my_sft.py"
>>> script_args = ["--output_dir", "/training-outputs/training-v3-final", ...]
>>> checkpoints_bucket = Volume(type="bucket", source="username/my-bucket", mount_path="/training-outputs")
>>> run_uv_job(script, script_args=script_args, volumes=[checkpoints_bucket])
```

默认情况下，挂载的存储桶具有读+写能力。
这对于存储桶特别有用，它为经常更改的数据提供快速、可变的存储——文件可以就地覆盖或删除。

使用`read_only=True`启用只读：`Volume(type="bucket", read_only=True, ...)`。

### 挂载本地数据要针对计算机上的数据运行作业，请使用 [sync_job_volume()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_job_volume)：它将本地目录同步到您的 `jobs-artifacts` [Storage Bucket](https://huggingface.co/docs/hub/storage-buckets)（如果需要，会自动创建）并返回准备安装的 [Volume](/docs/huggingface_hub/v1.30.0/en/package_reference/jobs#huggingface_hub.Volume)：

```python
>>> from huggingface_hub import run_uv_job, sync_job_volume

# Upload ./training-data once...
>>> volume = sync_job_volume("./training-data", "/data")

# ...then run as many Jobs as you want against it
>>> run_uv_job("train.py", script_args=["--learning-rate", "0.01"], volumes=[volume])
>>> run_uv_job("train.py", script_args=["--learning-rate", "0.05"], volumes=[volume])
```

每个目录在存储桶中都有自己的稳定文件夹：在同一目录上重新运行[sync_job_volume()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_job_volume)只会上传新的或修改的文件。默认情况下，该卷以只读方式安装。

要检索作业写入的文件，请安装一个读写卷（空的输出目录也可以）并在作业结束后将其同步回来：

```python
>>> from huggingface_hub import run_uv_job, sync_bucket, sync_job_volume

>>> outputs = sync_job_volume("./outputs", "/outputs", read_only=False)
>>> job = run_uv_job("process.py", volumes=[outputs])

# ...once the Job completes, pull back the data:
>>> sync_bucket(f"hf://buckets/{outputs.source}/{outputs.path}", "./outputs")
```

在 CLI 中，只需传递本地目录作为 `-v` 的源端：

```bash
>>> hf jobs uv run -v ./pdfs:/input -v ./md-out:/output:rw ocr.py
```

## 通过 SSH 进入作业

将 `ssh=True` 传递给 [run_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_job)（或 [run_uv_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_uv_job)）以使作业的容器可通过 SSH 访问。 SSH 端点在作业状态中可用：

```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="python:3.12",
...     command=["sleep", "infinity"],
...     ssh=True,
... )
>>> job.status.ssh_url
'ssh://68498e23210b3a4f4e6e2a23@ssh.hf.jobs'
```

从终端连接`hf jobs ssh <job_id>`（或直接使用`ssh <job_id>@ssh.hf.jobs`）：

```bash
>>> hf jobs ssh 68498e23210b3a4f4e6e2a23
```

仅允许对作业命名空间具有写入权限的用户（作业创建者或所有者组织的成员），并通过在 https://huggingface.co/settings/keys 注册的 SSH 公钥进行身份验证。

## 配置作业超时作业有一个默认超时（30 分钟），之后它们将自动停止。在运行模型训练等长时间运行的任务时，了解这一点很重要。

### 设置自定义超时

您可以在运行作业时使用 `timeout` 参数指定自定义超时值。可以通过两种方式指定超时：

1. **作为数字**（解释为秒）：
```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "train_model.py"],
...     flavor="a10g-large",
...     timeout=7200,  # 2 hours in seconds
... )
```

2. **作为带有时间单位的字符串**：
```python
>>> # Using different time units
>>> job = run_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python", "train_model.py"],
...     flavor="a10g-large",
...     timeout="2h",  # 2 hours
... )

>>> # Other examples:
>>> # timeout="30m"    # 30 minutes
>>> # timeout="1.5h"   # 1.5 hours
>>> # timeout="1d"     # 1 day
>>> # timeout="3600s"  # 3600 seconds
```

支持的时间单位：
- `s` - 秒
- `m` - 分钟  
- `h` - 小时
- `d` - 天

### 对 UV 作业使用超时

对于 UV 作业，您还可以指定超时：

```python
>>> from huggingface_hub import run_uv_job
>>> job = run_uv_job(
...     "training_script.py",
...     flavor="a10g-large",
...     timeout="90m",  # 90 minutes
... )
```

> [!警告]
> 如果您不指定超时，则默认超时将应用于您的作业。对于模型训练等可能需要数小时的长时间运行的任务，请确保设置适当的超时以避免作业意外终止。

### 监控作业持续时间

运行长时间任务时，最好的做法是：
- 估计您的工作的预期持续时间并设置带有一些缓冲区的超时
- 通过日志监控您的工作进度
- 检查作业状态以确保其未超时

```python
>>> from huggingface_hub import inspect_job, fetch_job_logs
>>> # Check job status
>>> job_info = inspect_job(job_id=job.id)
>>> if job_info.status.stage == "ERROR":
...     print(f"Job failed: {job_info.status.message}")
...     # Check logs for more details
...     for log in fetch_job_logs(job_id=job.id):
...         print(log)
```

有关超时参数的更多详细信息，请参阅[⟦T98⟧ API reference](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api#huggingface_hub.HfApi.run_job.timeout)。

## 传递环境变量和 Secret您可以使用 `env` 和 `secrets` 将环境变量传递给您的作业：

```python
# Pass environment variables
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['FOO'], os.environ['BAR'])"],
...     env={"FOO": "foo", "BAR": "bar"},
... )
```

```python
# Pass secrets - they will be encrypted server side
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['MY_SECRET'])"],
...     secrets={"MY_SECRET": "psswrd"},
... )
```

### 内置环境变量

在作业容器内，以下环境变量自动可用：

|变量|描述 |
|----------|-------------|
| `JOB_ID` |当前作业的唯一标识符。使用它以编程方式引用作业，例如以唯一名称将输出存储在数据集中。 |
| `ACCELERATOR` |可用的加速器类型（例如，`t4-medium`、`a10g-small`、`a100x4`）。如果没有加速器则为空。 |
| `CPU_CORES` |作业可用的 CPU 核心数量（例如，`2`、`4`、`8`）。 |
| `MEMORY` |作业可用的内存量（例如，`16Gi`、`32Gi`）。 |

```python
# Access job environment information
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", """
... import os
... print(f"Job ID: {os.environ.get('JOB_ID')}")
... print(f"Accelerator: {os.environ.get('ACCELERATOR', 'none')}")
... print(f"CPU cores: {os.environ.get('CPU_CORES')}")
... print(f"Memory: {os.environ.get('MEMORY')}")
... """],
... )
```

当您需要为输出创建唯一标识符、根据可用硬件调整代码或记录资源信息时，这些变量非常有用。

## 标签

标签是将元数据附加到作业的键=值对：

使用 `name` 可以更轻松地在 UI 中查找和识别职位。名称是可选的，并且不必是唯一的：

```python
>>> from huggingface_hub import run_job
>>> run_job(name="daily-report", image="python:3.12", command=["python", "report.py"])
```

在 CLI 中，创建作业时传递 `--name`，或通过 labels 命令命名现有作业：

```bash
>>> hf jobs run --name daily-report python:3.12 python report.py
>>> hf jobs labels <job_id> --name daily-report
````hf jobs ls` 显示`NAME` 列，并且可以按名称过滤（`--label name=NAME` 的快捷方式）：

```bash
>>> hf jobs ls -a --name daily-report
```

```python
# Pass extra metadata with Labels
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "import os; print(os.environ['MY_SECRET'])"],
...     labels={"my-label": "my-value", "foo": "bar"},
... )
```

如果您不通过 `--name`，则会从 Docker 映像或脚本自动派生一个名称，加上命令的简短哈希值，因此同一命令的重新运行会共享一个名称（例如 `python:3.12 foo --truc` → `python-3-12-1a2b3c4d`）。

### 更新标签

使用 [update_job_labels()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_job_labels) 替换现有作业上的标签。这将替换所有现有的用户提供的标签：

```python
>>> from huggingface_hub import update_job_labels
>>> update_job_labels(job_id, labels={"env": "prod", "team": "ml"})
```

这也适用于 [update_scheduled_job_labels()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.update_scheduled_job_labels) 的预定作业：

```python
>>> from huggingface_hub import update_scheduled_job_labels
>>> update_scheduled_job_labels(scheduled_job_id, labels={"env": "staging"})
```

从 CLI：

```bash
>>> hf jobs labels <job_id> --label env=prod --label team=ml
>>> hf jobs scheduled labels <scheduled_job_id> --label env=staging
```

要删除所有标签，请传递`--clear`：

```bash
>>> hf jobs labels <job_id> --clear
```

## UV 脚本（实验）

> [!提示]
> 正在寻找即用型 UV 脚本？查看 Hugging Face Hub 上的 [uv-scripts organization](https://huggingface.co/uv-scripts)，它提供了 UV 脚本的社区集合，用于执行模型训练、合成数据生成、数据处理等任务。

在 HF 基础设施上运行 UV 脚本（具有内联依赖项的 Python 脚本）：

```python
# Run a UV script (creates temporary repo)
>>> from huggingface_hub import run_uv_job
>>> run_uv_job("my_script.py")

# Run with GPU
>>> run_uv_job("ml_training.py", flavor="gpu-t4-small")

# Run with dependencies
>>> run_uv_job("inference.py", dependencies=["transformers", "torch"])

# Run a script directly from a URL
>>> run_uv_job("https://huggingface.co/datasets/username/scripts/resolve/main/example.py")

# Run a command
>>> run_uv_job("python", script_args=["-c", "import lighteval"], dependencies=["lighteval"])
```

UV 脚本是 Python 脚本，使用特殊的注释语法将其依赖项直接包含在文件中。这使得它们非常适合不需要复杂项目设置的独立任务。在 [UV documentation](https://docs.astral.sh/uv/guides/scripts/) 中了解有关 UV 脚本的更多信息。

#### UV 脚本的 Docker 镜像虽然 UV 脚本可以内联指定其依赖关系，但 ML 工作负载通常具有复杂的依赖关系。使用已安装这些库的预构建 Docker 映像可以显着加快作业启动速度并避免依赖性问题。

默认情况下，当您运行 `hf jobs uv run` 时，将使用 `astral-sh/uv:python3.12-bookworm` 图像。该图像基于预装了 uv 的 Python 3.12 Bookworm 发行版。

您可以使用 `--image` 标志指定不同的图像：

```bash
hf jobs uv run \
 --flavor a10g-large \
 --image vllm/vllm-openai:latest \
...
```

上述命令将使用 `vllm/vllm-openai:latest` 图像运行。如果您使用 vLLM 进行合成数据生成，此方法可能会很有用。

> [!提示]
> 许多推理框架提供优化的 docker 镜像。随着 uv 在 Python 生态系统中越来越多地被采用，更多的这些也将预安装 uv，这意味着它们将在使用 hf jobs uv run 时工作。

### 预定的工作

安排和管理将在 HF 基础设施上运行的作业。

将 [create_scheduled_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_scheduled_job) 或 [create_scheduled_uv_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_scheduled_uv_job) 与时间表 `@annually`、`@yearly`、`@monthly`、`@weekly`、`@daily`、`@hourly` 或 CRON 时间表表达式（例如， `"0 9 * * 1"`每周一上午 9 点）：

```python
# Schedule a job that runs every hour
>>> from huggingface_hub import create_scheduled_job
>>> create_scheduled_job(
...     image="python:3.12",
...     command=["python",  "-c", "print('This runs every hour!')"],
...     schedule="@hourly"
... )

# Use the CRON syntax
>>> create_scheduled_job(
...     image="python:3.12",
...     command=["python",  "-c", "print('This runs every 5 minutes!')"],
...     schedule="*/5 * * * *"
... )

# Schedule with GPU
>>> create_scheduled_job(
...     image="pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel",
...     command=["python",  "-c", 'import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")'],
...     schedule="@hourly",
...     flavor="a10g-small",
... )

# Schedule a UV script
>>> from huggingface_hub import create_scheduled_uv_job
>>> create_scheduled_uv_job("my_script.py", schedule="@hourly")
```

使用与[run_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_job)和[run_uv_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_uv_job)相同的参数来传递环境变量、秘密、超时等。使用 `list_scheduled_jobs`、[inspect_scheduled_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.inspect_scheduled_job)、[suspend_scheduled_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.suspend_scheduled_job)、[resume_scheduled_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.resume_scheduled_job)、[trigger_scheduled_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.trigger_scheduled_job) 和 [delete_scheduled_job()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_scheduled_job) 管理计划作业：

```python
# List your active scheduled jobs
>>> from huggingface_hub import list_scheduled_jobs
>>> list_scheduled_jobs()

# Only list scheduled jobs with the given labels
>>> list_scheduled_jobs(labels={"env": "prod"})

# Inspect the status of a job
>>> from huggingface_hub import inspect_scheduled_job
>>> inspect_scheduled_job(scheduled_job_id)

# Suspend (pause) a scheduled job
>>> from huggingface_hub import suspend_scheduled_job
>>> suspend_scheduled_job(scheduled_job_id)

# Resume a scheduled job
>>> from huggingface_hub import resume_scheduled_job
>>> resume_scheduled_job(scheduled_job_id)

# Trigger a scheduled job to run right now (does not change the schedule)
>>> from huggingface_hub import trigger_scheduled_job
>>> job = trigger_scheduled_job(scheduled_job_id)
>>> job.url

# Delete a scheduled job
>>> from huggingface_hub import delete_scheduled_job
>>> delete_scheduled_job(scheduled_job_id)
```

### 使用 webhooks 触发作业

Webhooks 允许您监听特定存储库或属于特定用户/组织集的所有存储库（不仅仅是您的存储库，而是任何存储库）的新更改。

使用 [create_webhook()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_webhook) 创建一个 Webhook，当 Hugging Face 存储库中发生更改时触发作业：

```python
from huggingface_hub import create_webhook

# Example: Creating a webhook that triggers a Job
webhook = create_webhook(
    job_id=job_id,
    watched=[{"type": "user", "name": "your-username"}, {"type": "org", "name": "your-org-name"}],
    domains=["repo", "discussion"],
    secret="your-secret"
)
```

Webhook 使用环境变量 `WEBHOOK_PAYLOAD` 中的 Webhook 负载触发作业。
您可以在 [Webhooks documentation](./webhooks) 中找到有关 webhooks 的更多信息。

### 将任何 ML 框架与 Hub 集成
https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/integrations.md