<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 管理工作

## 列出职位

在“职位”页面或您的组织“职位”页面（用户/组织页面 > 设置 > 职位）中查找您的职位列表：

它也可以在 Hugging Face CLI 中使用。使用 `hf jobs ps` 显示正在运行的作业列表，并使用 `-a` 显示所有作业：

```bash
>>> hf jobs ps
JOB ID       IMAGE/SPACE      COMMAND     CREATED             STATUS  
------------ ---------------- ----------- ------------------- ------- 
69402ea6c... ghcr.io/astra... uv run p... 2025-12-15 15:52:06 RUNNING
>>> hf jobs ps -a
JOB ID       IMAGE/SPACE COMMAND         CREATED             STATUS    
------------ ---------- --------------- ------------------- --------- 
69402ea6c... ghcr.io... uv run pytho... 2025-12-15 15:52:06 RUNNING   
693b06b8c... ghcr.io... uv run pytho... 2025-12-11 18:00:24 CANCELED  
693b069fc... ghcr.io... uv run pytho... 2025-12-11 17:59:59 ERROR     
693aef401... ghcr.io... uv run pytho... 2025-12-11 16:20:16 COMPLETED 
693aee76c... ubuntu     echo Hello f... 2025-12-11 16:16:54 COMPLETED 
693ae8e3c... python:... python -c pr... 2025-12-11 15:53:07 COMPLETED
```

指定您的组织`namespace`以列出您组织下的职位：

```bash
>>> hf jobs ps --namespace <my-org-name>
```

## 过滤职位

您最常用的标签显示在作业列表上方，并显示使用每个标签的作业数量；单击一个进行过滤，或输入任何 `key=value` 标签：

在 CLI 中，使用 `--status` 按状态过滤，并使用 `--label key=value` 按标签过滤。重复`--label`一次需要多个标签：

```bash
# Jobs that ended in error
>>> hf jobs ps --status error

# Running or scheduling Jobs with a given label
>>> hf jobs ps --status running,scheduling --label env=prod

# All Jobs (any status) that have both labels
>>> hf jobs ps -a --label model=Qwen3-06B --label dataset=Capybara

# A label added without a value (`--label fine-tuning`) is matched with a trailing "="
>>> hf jobs ps -a --label fine-tuning=
```

使用 `--name`（`--label name=<name>` 的快捷方式）按名称过滤，例如列出指定作业的每次运行：

```bash
>>> hf jobs ps -a --name daily-report
JOB_ID      NAME         IMAGE/SPACE COMMAND      CREATED      STATUS    RUNTIME
----------- ------------ ----------- ------------ ------------ --------- -------
6a9042c2... daily-report python:3.12 python -c... 2026-08-2... COMPLETED 0s     
6a904162... daily-report python:3.12 python -c... 2026-08-2... CANCELED  --     
```

> [!提示]
> 默认情况下`hf jobs ps` 显示正在运行和调度的作业。使用`-a`显示所有状态； `-a` 不能与`--status` 组合使用。

## 监控资源使用情况

使用`hf jobs stats`获取正在运行的作业的CPU、内存、网络和GPU（如果有）的使用统计信息：

```bash
>>> hf jobs stats
JOB ID                   CPU % NUM CPU MEM % MEM USAGE        NET I/O         GPU UTIL % GPU MEM % GPU MEM USAGE   
------------------------ ----- ------- ----- ---------------- --------------- ---------- --------- --------------- 
695e83c5d2f3efac77e8cf18 8%    12.0    7.18% 10.9GB / 152.5GB 0.0bps / 0.0bps 100%       31.92%    25.9GB / 81.2GB
```

指定一个或多个 Job id 以仅显示某些 Job 的统计信息：

```bash
>>> hf jobs stats [job-ids]...
```

## 检查作业

您可以在 Job 页面中查看 Job 的状态日志：

或者使用 CLI

```bash
>>> hf jobs inspect 693994e21a39f67af5a41ad0 
[
    {
        "id": "693994e21a39f67af5a41ad0",
        "created_at": "2025-12-10 15:42:26.835000+00:00",
        "docker_image": "ghcr.io/astral-sh/uv:python3.12-bookworm",
        "space_id": null,
        "command": ["bash", "-c", "python -c \"import urllib.request; import os; from pathlib import Path; o = urllib.request.build_opener(); o.addheaders = [(\\\"Authorization\\\", \\\"Bearer \\\" + os.environ[\\\"UV_SCRIPT_HF_TOKEN\\\"])]; Path(\\\"/tmp/script.py\\\").write_bytes(o.open(os.environ[\\\"UV_SCRIPT_URL\\\"]).read())\" && uv run --with trl /tmp/script.py"],
        "arguments": [],
        "environment": {"UV_SCRIPT_URL": "https://huggingface.co/datasets/lhoestq/hf-cli-jobs-uv-run-scripts/resolve/728cc5682eb402d7ffe66a2f6f97645b34cb08dd/train.py"},
        "secrets": ["HF_TOKEN", "UV_SCRIPT_HF_TOKEN"],
        "flavor": "a100-large",
        "status": {"stage": "COMPLETED", "message": null},
        "owner": {"id": "5e9ecfc04957053f60648a3e", "name": "lhoestq", "type": "user"},
        "endpoint": "https://huggingface.co",
        "url": "https://huggingface.co/jobs/lhoestq/693994e21a39f67af5a41ad0"
    }
]
```和日志

```bash
>>> hf jobs logs 693994e21a39f67af5a41ad0
Downloading nvidia-cuda-nvrtc-cu12 (84.0MiB)
Downloading numpy (15.8MiB)
Downloading nvidia-cuda-cupti-cu12 (9.8MiB)
Downloading tokenizers (3.1MiB)
Downloading nvidia-cusolver-cu12 (255.1MiB)
Downloading nvidia-cufft-cu12 (184.2MiB)
Downloading transformers (11.4MiB)
Downloading setuptools (1.1MiB)
... 
```

指定您的组织`namespace`来检查您组织下的作业：

```bash
hf jobs inspect --namespace <my-org-name> <job_id>
hf jobs logs --namespace <my-org-name> <job_id>
```

## 等待作业完成

使用 `hf jobs wait` 进行阻塞，直到一个或多个作业达到终止状态（`COMPLETED`、`CANCELED`、`ERROR` 或 `DELETED`）。仅当每个作业成功完成时，它才会以代码 `0` 退出，否则为非零代码 - 方便在 shell 脚本或 CI 中链接步骤：

```bash
# Wait for a single Job
>>> hf jobs wait 693994e21a39f67af5a41ad0

# Wait for several Jobs at once (all must be in the same namespace)
>>> hf jobs wait <job_id_1> <job_id_2>

# Wait for every currently running Job
>>> hf jobs ps -q | xargs hf jobs wait
```

使用 `--timeout` 设置最长等待时间（接受 `s`、`m`、`h` 或 `d`）：

```bash
>>> hf jobs wait --timeout 30m 693994e21a39f67af5a41ad0
```

因为当作业失败时 `hf jobs wait` 返回非零退出代码，所以您可以将其与 `&&` 链接。非分离的`hf jobs run`（或`hf jobs uv run`）已经阻塞并在失败时以非零值退出，因此仅对于分离的作业（`-d`）或等待批处理时才需要显式等待：

```bash
>>> hf jobs wait 693994e21a39f67af5a41ad0 && echo "job completed successfully"
```

## 坚持你的结果

当作业结束时，作业的文件系统将被删除。在作业退出之前写下您想要在某处持久保存的任何内容：

- **中间工件、检查点和日志 → 存储桶卷。** 安装存储桶并在安装路径下写入输出 - 有关示例，请参阅 [Volumes](./jobs-configuration#volumes)。创建作业时，卷安装将使用您的 Hugging Face 身份进行授权，因此您的脚本不需要令牌即可写入它们。- **最终模型和数据集 → 推送到 Hub 存储库。** 作业没有 Hugging Face 令牌，除非您传递一个令牌，例如`--secrets HF_TOKEN`（裸表单自动解析为您登录的令牌）。如果您的脚本调用`push_to_hub()`或`create_repo()`，请确保令牌具有写入权限（细粒度令牌需要存储库写入和创建权限）。常见的失败模式是作业完成数小时的计算，然后在最终上传时出错，因为令牌无法写入 — 计算已完成，但结果未保存。

- **将关键结果打印到日志中。** 作业日志在作业结束后保留​​，并且可以随时使用 `hf jobs logs <job-id>` 获取。即使上传步骤失败，将最终指标打印到标准输出也能让它们恢复。

作业完成后，检查您的输出是否实际落地（例如使用 `hf buckets list`） - `COMPLETED` 状态意味着命令成功退出，而不是文件已保留在您期望的位置。

## 调试作业

如果作业有错误，您可以在作业页面上看到它

查看“作业”页面上的状态消息和日志，了解出了什么问题。

您还可以查看日志的最后几行，以了解作业失败之前发生的情况。您可以在“作业”页面中或使用 CLI 来查看：```bash
>>> hf jobs logs 69405cf51a39f67af5a41f29 | tail -n 10     
 Downloaded nvidia-cudnn-cu12
 Downloaded torch
Installed 66 packages in 226ms
Generating train split: 100%|██████████| 15806/15806 [00:00<00:00, 73330.17 examples/s]
Generating test split: 100%|██████████| 200/200 [00:00<00:00, 45427.32 examples/s]
Traceback (most recent call last):
  File "/tmp/script.py", line 7, in <module>
    train_dataset=train_dataset,
                  ^^^^^^^^^^^^^
NameError: name 'train_dataset' is not defined. Did you mean: 'load_dataset'?
```

使用本地 UV 或 Docker 设置在本地调试作业：

* `hf jobs uv run ...` -> `uv run ...`
* `hf jobs run ...` -> `docker run ...`

状态消息可以显示“作业超时”：这意味着作业在超时（默认为 30 分钟）之前没有及时完成，因此被停止。
在这种情况下，您需要在 CLI 中使用 `--timeout` 指定更高的超时，例如

```bash
hf jobs uv run --timeout 3h ...
```

## 取消作业

使用作业页面上的“取消”按钮取消作业：

或在 CLI 中：

```bash
hf jobs cancel 693b06b8c67c9f186cfe239e
```

指定您的组织`namespace`以取消您组织下的作业：

```bash
hf jobs cancel --namespace <my-org-name> <job_id>
```

## MacOS 菜单栏

在 MacOS [⟦T59⟧](https://github.com/drbh/hfjobs-menubar) 客户端中找到您的职位列表：

获取作业信息，并监控日志和资源使用统计信息：

### 空格
https://huggingface.co/docs/hub/spaces.md