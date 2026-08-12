<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 配置

## 身份验证

您需要通过`hf auth login`进行身份验证才能运行作业，并使用具有启动和管理作业权限的令牌。

或者，使用 CLI 中的 `--token` 或 Python 中的 `token` 参数手动传递 Hugging Face 令牌。

## 紫外线工作

指定要运行的 UV 脚本或 python 命令，就像使用 UV 一样：

```bash
>>> hf jobs uv run train.py
```

```bash
>>> hf jobs uv run python -c 'print("Hello from the cloud!")'
```

`hf jobs uv run` 命令接受 UV 参数，如 `--with` 和 `--python`。 `--with` 参数允许您指定 python 依赖项，`--python` 允许您选择要使用的 python 版本：

```bash
>>> hf jobs uv run --with trl train.py
>>> hf jobs uv run --python 3.12 train.py
```

命令（或脚本）后面的参数不会解释为 uv 的参数。 uv 的所有选项都必须在命令之前提供，例如 uv run --verbose foo。为了清晰起见，可以使用 `--` 将命令与 jobs/uv 选项分开，例如

```bash
>>> hf jobs uv run --with trl-jobs -- trl-jobs sft --model_name Qwen/Qwen3-0.6B --dataset_name trl-lib/Capybara
```

查找 [CLI documentation](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-jobs-uv-run) 和 [UV Commands documentation](https://docs.astral.sh/uv/reference/cli/#uv-run) 中所有参数的列表。

默认情况下，UV 作业使用 `ghcr.io/astral-sh/uv:python3.12-bookworm` Docker 映像运行，但只要安装了 UV，您就可以使用其他映像，即 `--image <docker-image>`。

## Docker 工作

指定 Docker 映像和要运行的命令，就像使用 docker 一样：

```bash
>>> hf jobs run ubuntu echo "Hello from the cloud!"
```作业的所有选项都必须在命令之前提供。为了清晰起见，可以使用 `--` 将命令与 jobs/uv 选项分开，例如

```bash
>>> hf jobs run --token hf_xxx ubuntu -- echo "Hello from the cloud!"
```

在 [CLI documentation](https://huggingface.co/docs/huggingface_hub/package_reference/cli#hf-jobs-run) 中找到所有参数的列表。

## 环境变量和秘密

### 内置环境变量

与[built-in environment variables in Spaces](./spaces-overview#built-in-environment-variables)类似，Jobs自动在容器内提供以下环境变量：

|变量|描述 |
|----------|-------------|
| `JOB_ID` |当前作业的唯一标识符（例如，`699d874f1aad19adb8aaeadc`）。这与 UI 和作业 URL 中显示的 ID 相同。 |
| `ACCELERATOR` |可用加速器的类型（例如，`t4-medium`、`a10g-small`、`a100x4`）或`none`（适用于仅 CPU 作业）。 |
| `CPU_CORES` |分配给作业的 CPU 核心数。 |
| `MEMORY` |分配给作业的内存量（例如，`8Gi`）。 |

您可以使用这些变量来跟踪输出，使代码适应可用资源，或以编程方式引用当前作业：

```bash
# Access job environment information
>>> hf jobs run python:3.12 python -c "import os; print(f'Job: {os.environ.get(\"JOB_ID\")}, CPU: {os.environ.get(\"CPU_CORES\")}, Mem: {os.environ.get(\"MEMORY\")}')"
```

### 用户定义的环境变量

您可以使用以下命令将环境变量传递给您的作业 

```bash
# Pass environment variables
>>> hf jobs uv run -e FOO=foo -e BAR=bar python -c 'import os; print(os.environ["FOO"], os.environ["BAR"])'
```

```bash
# Pass an environment from a local .env file
>>> hf jobs uv run --env-file .env python -c 'import os; print(os.environ["FOO"], os.environ["BAR"])'
```

```bash
# Pass secrets - they will be encrypted server side
>>> hf jobs uv run -s MY_SECRET=psswrd python -c 'import os; print(os.environ["MY_SECRET"])'
```

```bash
# Pass secrets from a local .env.secrets file - they will be encrypted server side
>>> hf jobs uv run --secrets-file .env.secrets python -c 'import os; print(os.environ["MY_SECRET"])'
```> [!提示]
> 使用 `--secrets HF_TOKEN` 隐式传递您本地的 Hugging Face 令牌。
> 使用此语法，可以从环境变量中检索机密。
> 对于`HF_TOKEN`，如果未设置环境变量，它可能会读取位于 Hugging Face 主文件夹中的令牌文件。

## 卷

使用 `-v` 或 `--volume` 将 Hugging Face 存储库（模型、数据集）、[Storage Buckets](./storage-buckets) 或本地目录作为卷安装在作业容器中。 Hub 源使用 `hf://` URL 方案：`hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]`；本地目录直接作为源传递。

> [!提示]
> 由于装载的文件是延迟获取的，因此装载可以让作业处理远大于其本地磁盘的数据集。有关在作业上安装、流式传输和处理大数据的信息，请参阅[Process Large Datasets](./jobs-large-datasets)。

卷类型：

|类型 |示例|
|------|---------|
|模型仓库 | `-v hf://openai/gpt-oss-120b:/model` |
|数据集存储库 | `-v hf://datasets/stanfordnlp/imdb:/data` |
|储物桶| `-v hf://buckets/username/my-bucket:/mnt` |
|子文件夹| `-v hf://datasets/org/my-dataset/train:/data` |
|本地目录| `-v ./training-data:/data` |

然后使用已安装的卷作为容器内的本地目录：

```bash
# Mount a dataset and query it with DuckDB
>>> hf jobs run -v hf://datasets/stanfordnlp/imdb:/dataset \
...     duckdb/duckdb duckdb -c "SELECT * FROM '/dataset/**/*.parquet' LIMIT 5"

# Mount a bucket to save training checkpoints
>>> hf jobs uv run -v hf://buckets/username/my-bucket:/training-outputs \
...     sft.py --output-dir /training-outputs/training-v3-final
```

可以通过重复 `-v` 标志来安装多个卷：

```bash
>>> hf jobs run -v hf://datasets/username/my-dataset:/data -v hf://buckets/username/my-bucket:/output \
...     python:3.12 python script.py
```模型和数据集始终以**只读**方式安装。默认情况下，存储桶是**读写**的，这对于保存输出、检查点或中间结果非常有用。使用 `:ro` 以只读模式挂载存储桶：

```bash
>>> hf jobs run -v hf://buckets/username/my-bucket:/mnt:ro python:3.12 ls /mnt
```

### 本地目录

传递本地目录作为源，在作业启动之前将其同步到您的私有`jobs-artifacts`[Storage Bucket](./storage-buckets)（自动创建），然后将其安装到容器中。本地目录默认挂载为**只读**；使用 `:rw` 写入输出：

```bash
>>> hf jobs uv run -v ./pdfs:/input -v ./md-out:/output:rw ocr.py
```

重新同步同一目录只会上传新的或修改的文件。要检索作业写入读写卷的文件，请在作业结束后同步其存储桶文件夹 — CLI 在作业启动时打印确切的 `hf buckets sync` 命令。计划作业也可以工作：创建计划时目录会同步一次，并且每个触发器都会安装相同的文件夹。在Python中，使用[⟦T74⟧](https://huggingface.co/docs/huggingface_hub/guides/jobs#mount-local-data)。

在 Python 中，使用 [⟦T75⟧](https://huggingface.co/docs/huggingface_hub/package_reference/jobs#huggingface_hub.Volume) 类：

```python
from huggingface_hub import Volume, run_job

job = run_job(
    image="python:3.12",
    command=["python", "-c", "import os; print(os.listdir('/data'))"],
    volumes=[
        Volume(type="dataset", source="username/my-dataset", mount_path="/data"),
        Volume(type="bucket", source="username/my-bucket", mount_path="/output"),
    ],
)
```

> [!注意]
> 批量安装需要 `huggingface_hub` >= 1.8.0。更多详情请参阅[Python client documentation](https://huggingface.co/docs/huggingface_hub/guides/jobs#mount-a-volume)和[installation guide](https://huggingface.co/docs/huggingface_hub/installation)。

## 硬件风味

使用 `flavor` 参数在 GPU 或 TPU 上运行作业。例如，要在 A10G GPU 上运行 PyTorch 作业：

```bash
>>> hf jobs uv run --with torch --flavor a10g-small python -c "import torch; print(f'This code ran with the following GPU: {torch.cuda.get_device_name()}')"
```

运行此命令将显示以下输出！```
This code ran with the following GPU: NVIDIA A10G
```

这是运行像 [trl/scripts/sft.py](https://github.com/huggingface/trl/blob/main/trl/scripts/sft.py) 这样的微调脚本的另一个示例：

```bash
>>> hf jobs uv run --with trl --flavor a10g-small -s HF_TOKEN -- sft.py --model_name_or_path Qwen/Qwen2-0.5B ...
```

> [!提示]
> 有关在 Hugging Face 基础设施上使用 TRL 运行模型训练作业的综合指南，请查看 [TRL Jobs Training documentation](https://huggingface.co/docs/trl/main/en/jobs_training)。它涵盖了微调配方、硬件选择以及有效训练模型的最佳实践。

使用 `hf jobs hardware` 命令查看可用 `--flavor` 选项列表（默认为 `cpu-basic`）：

```bash
>>> hf jobs hardware
name             pretty name             cpu       ram      storage   accelerator               cost/min  cost/hour
cpu-basic        CPU Basic               2 vCPU    16 GB    50 GB                               $0.0002   $0.01
cpu-upgrade      CPU Upgrade             8 vCPU    32 GB    50 GB                               $0.0005   $0.03
cpu-performance  CPU Performance         32 vCPU   256 GB   1024 GB                             $0.0317   $1.90
cpu-xl           CPU XL                  16 vCPU   124 GB   1000 GB                             $0.0167   $1.00
t4-small         Nvidia T4 - small       4 vCPU    15 GB    50 GB     1x T4 (16 GB)             $0.0067   $0.40
t4-medium        Nvidia T4 - medium      8 vCPU    30 GB    100 GB    1x T4 (16 GB)             $0.0100   $0.60
a10g-small       Nvidia A10G - small     4 vCPU    15 GB    110 GB    1x A10G (24 GB)           $0.0167   $1.00
a10g-large       Nvidia A10G - large     12 vCPU   46 GB    200 GB    1x A10G (24 GB)           $0.0250   $1.50
a10g-largex2     2x Nvidia A10G - large  24 vCPU   92 GB    1000 GB   2x A10G (48 GB)           $0.0500   $3.00
a10g-largex4     4x Nvidia A10G - large  48 vCPU   184 GB   2000 GB   4x A10G (96 GB)           $0.0833   $5.00
a100-large       Nvidia A100 - large     12 vCPU   142 GB   1000 GB   1x A100 (80 GB)           $0.0417   $2.50
a100x4           4x Nvidia A100          48 vCPU   568 GB   4000 GB   4x A100 (320 GB)          $0.1667   $10.00
a100x8           8x Nvidia A100          96 vCPU   1136 GB  8000 GB   8x A100 (640 GB)          $0.3333   $20.00
h200             Nvidia H200             23 vCPU   256 GB   3000 GB   1x H200 (141 GB)          $0.0833   $5.00
h200x2           Nvidia H200             46 vCPU   512 GB   6000 GB   2x H200 (282 GB)          $0.1667   $10.00
h200x4           Nvidia H200             92 vCPU   1024 GB  12000 GB  4x H200 (564 GB)          $0.3333   $20.00
h200x8           Nvidia H200             184 vCPU  2048 GB  24000 GB  8x H200 (1128 GB)         $0.6667   $40.00
rtx-pro-6000     Nvidia RTX PRO 6000     23 vCPU   256 GB   475 GB    1x RTX PRO 6000 (96 GB)   $0.0458   $2.75
rtx-pro-6000x2   Nvidia RTX PRO 6000     46 vCPU   512 GB   950 GB    2x RTX PRO 6000 (192 GB)  $0.0917   $5.50
rtx-pro-6000x4   Nvidia RTX PRO 6000     92 vCPU   1024 GB  1900 GB   4x RTX PRO 6000 (384 GB)  $0.1833   $11.00
rtx-pro-6000x8   Nvidia RTX PRO 6000     184 vCPU  2048 GB  3800 GB   8x RTX PRO 6000 (768 GB)  $0.3667   $22.00
l4x1             1x Nvidia L4            8 vCPU    30 GB    400 GB    1x L4 (24 GB)             $0.0133   $0.80
l4x4             4x Nvidia L4            48 vCPU   186 GB   3200 GB   4x L4 (96 GB)             $0.0633   $3.80
l40sx1           1x Nvidia L40S          8 vCPU    62 GB    380 GB    1x L40S (48 GB)           $0.0300   $1.80
l40sx4           4x Nvidia L40S          48 vCPU   382 GB   3200 GB   4x L40S (192 GB)          $0.1383   $8.30
l40sx8           8x Nvidia L40S          192 vCPU  1534 GB  6500 GB   8x L40S (384 GB)          $0.3917   $23.50
```

## 暴露端口

作业可以使用 `--expose <port>` (CLI) 或 `expose=[<port>]` (Python API) 通过公共作业代理公开容器端口。每个公开的端口都可以通过 `https://<job_id>--<port>.hf.jobs` 访问，并且需要具有 `read` 访问作业命名空间的 HF 令牌：

```bash
curl -H "Authorization: Bearer $HF_TOKEN" https://<job_id>--<port>.hf.jobs/
```

这适用于 `hf jobs run`、`hf jobs uv run` 及其预定变体。重复该标志以公开多个端口（`--expose 8000 --expose 8001`），或传递列表中的多个端口（`expose=[8000, 8001]`）。

> [!注意]
> 仅当作业运行时，暴露的端口才会在作业硬件价格的基础上按小时固定小费率计费。详情请参阅[pricing page](./jobs-pricing)。

### 命令行界面

```bash
# Expose a web server running on port 8000
>>> hf jobs run --expose 8000 python:3.12 python -m http.server 8000
✓ Job started
  id: 6a2aa7cec4f53f9fc5aa4cff
  url: https://huggingface.co/jobs/Wauplin/6a2aa7cec4f53f9fc5aa4cff
Hint: Exposed ports are reachable at (requires an HF token with read access to the job):
  https://6a2aa7cec4f53f9fc5aa4cff--8000.hf.jobs
Serving HTTP on 0.0.0.0 port 8000 (http://0.0.0.0:8000/) ...
```

###Python

```python
>>> from huggingface_hub import run_job
>>> job = run_job(image="python:3.12", command=["python", "-m", "http.server", "8000"], expose=[8000])
>>> job.status.expose_urls
['https://6a2ab384c4f53f9fc5aa4d4f--8000.hf.jobs']
```

## SSH您可以在正在运行的作业中打开交互式 SSH 会话，以直接在容器内进行调试、检查或工作。在创建作业时使用 `--ssh` (CLI) 或 `ssh=True` (Python API) 启用它，然后与 `hf jobs ssh <job_id>` 连接。

仅允许对作业的命名空间具有写入权限的用户（即作业创建者或具有写入权限的所有者组织的成员）。通过在 [https://huggingface.co/settings/keys](https://huggingface.co/settings/keys) 注册的 SSH 公钥执行身份验证。

SSH 在 `hf jobs run` 和 `hf jobs uv run` 上可用。计划作业不支持它。

### 命令行界面

```bash
# Start a job with SSH enabled
>>> hf jobs run --ssh --detach --timeout 10m python:3.12 sleep infinity
✓ Job started
  id: 6a2bd1f1871c005b5352ad31
  url: https://huggingface.co/jobs/Wauplin/6a2bd1f1871c005b5352ad31
Hint: Use `hf jobs ssh 6a2bd1f1871c005b5352ad31` to open an SSH session into the job.

# Open an SSH session into the job
>>> hf jobs ssh 6a2bd1f1871c005b5352ad31
```

您还可以打印 SSH 命令而不运行它（`--dry-run`），或传递特定的身份文件（`-i`/`--identity-file`）：

```bash
>>> hf jobs ssh 6a2bd1f1871c005b5352ad31 --dry-run
ssh 6a2bd1f1871c005b5352ad31@ssh.hf.jobs

>>> hf jobs ssh 6a2bd1f1871c005b5352ad31 -i ~/.ssh/id_ed25519
```

###Python

```python
>>> from huggingface_hub import run_job
>>> job = run_job(image="python:3.12", command=["sleep", "infinity"], ssh=True)
>>> job.status.ssh_url
'ssh://6a2bd1f1871c005b5352ad31@ssh.hf.jobs'
```

然后从终端使用`hf jobs ssh <job_id>`连接，或直接使用`ssh <job_id>@ssh.hf.jobs`连接。

### 端口转发

由于这是常规 SSH 连接，因此您可以使用 SSH 的 `-L` 和 `-R` 标志在您的计算机和作业之间转发端口。直接与`ssh`连接（使用`hf jobs ssh <job_id> --dry-run`获取确切的目的地）并添加转发标志。

使用`-L`（本地转发）从您的机器访问作业内运行的服务。例如，要达到在训练作业中启动的 TensorBoard：

```bash
# Forward local port 6006 to the Job's TensorBoard on port 6006
>>> ssh -L 6006:localhost:6006 6a2bd1f1871c005b5352ad31@ssh.hf.jobs
```

然后在浏览器中打开[http://localhost:6006](http://localhost:6006)。使用`-R`（远程转发）让作业访问您机器上运行的服务。例如，要将本地数据库或 API 公开给作业：

```bash
# Make your local port 8080 reachable from inside the Job on port 8080
>>> ssh -R 8080:localhost:8080 6a2bd1f1871c005b5352ad31@ssh.hf.jobs
```

## 超时

作业有一个默认超时（30 分钟），之后它们将自动停止。在运行模型训练等长时间运行的任务时，了解这一点很重要。

您可以在运行作业时使用 `--timeout` 参数指定自定义超时值。可以通过两种方式指定超时：

1. **作为数字**（解释为秒）：

使用 `--timeout` 并传递以秒为单位的数字（这里 2 小时 = 7200 秒）：

```bash
>>> hf jobs uv run --timeout 7200 --with torch --flavor a10g-large train.py
```

2. **作为带有时间单位的字符串**：

或者使用 `--timeout` 并使用不同的时间单位：

```bash
>>> hf jobs uv run --timeout 2h --with torch --flavor a10g-large train.py
```

其他例子：

```bash
--timeout 30m    # 30 minutes
--timeout 1.5h   # 1.5 hours
--timeout 1d     # 1 day
--timeout 3600s  # 3600 seconds
```

支持的时间单位：
- `s` - 秒
- `m` - 分钟  
- `h` - 小时
- `d` - 天

> [!警告]
> 如果您不指定超时，则默认超时将应用于您的作业。对于模型训练等可能需要数小时的长时间运行的任务，请确保设置适当的超时以避免作业意外终止。

## 命名空间使用 `--namespace` 参数在您的组织帐户下运行作业。确保您使用有权在您的组织帐户下启动和管理作业的令牌登录。

```bash
>>> hf jobs uv run --namespace my-org-name python -c "print('Running in an org account')"
```

请注意，您可以手动传递具有正确权限的令牌：

```bash
>>> hf jobs uv run --namespace my-org-name --token hf_xxx python -c "print('Running in an org account')"
```

## 标签

向作业添加一个或多个标签，以添加一些带有 `-l` 或 `--label` 的元数据。
您可以稍后使用此类元数据来过滤网站上或 CLI 中的作业。

添加带有 `--label my-label` 的标签或带有 `--label key=value` 的键值标签。
例如：

```bash
hf jobs uv run --label fine-tuning --label model=Qwen3-0.6B --label dataset=Capybara ...
```

请注意，多次使用相同的`key`会导致最后一个`key=value`覆盖并丢弃任何先前带有`key`的标签。

### 命名一个工作

为作业命名，以便更容易在 UI 中查找和识别。该名称存储为 `name` 标签。名称是可选的，并且不必是唯一的。在 UI 中，作业将按名称分组。

```bash
hf jobs run --name daily-report python:3.12 python report.py
```

### 更新标签

使用 `hf jobs labels` 更新现有作业上的标签。通过`--label`将替换所有现有标签；单独通过 `--name` 可以保留它们：

```bash
# Replace the labels on a Job
hf jobs labels <job_id> --label env=prod --label team=ml

# Name an existing Job (keeps its other labels)
hf jobs labels <job_id> --name daily-report

# Remove all labels from a Job
hf jobs labels <job_id> --clear
```

### 吉斯卡德谈空间
https://huggingface.co/docs/hub/spaces-sdks-docker-giskard.md