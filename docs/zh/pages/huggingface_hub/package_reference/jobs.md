<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 职位

查看 [HfApi](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi) 文档页面，获取在 Hub 上管理作业的方法参考。

- 运行作业：[run_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_job)
- 获取日志：[fetch_job_logs()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_job_logs)
- 获取指标：[fetch_job_metrics()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_job_metrics)
- 检查作业：[inspect_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.inspect_job)
- 等待作业完成：[wait_for_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.wait_for_job)
- 列出工作：[list_jobs()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_jobs)
- 取消作业：[cancel_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.cancel_job)
- 运行 UV 作业：[run_uv_job()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_uv_job)
- 同步本地目录以将其挂载到作业中：[sync_job_volume()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.sync_job_volume)

## 数据结构

### JobInfo[[huggingface_hub.JobInfo]]

#### Huggingface_hub.JobInfo[[huggingface_hub.JobInfo]]

```python
huggingface_hub.JobInfo(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_jobs_api.py#L163)

**参数：**

id (`str`) ：作业 ID。

created_at（`datetime`或`None`）：创建作业的时间。

started_at（`datetime`或`None`）：作业开始运行的时间。当作业仍在调度时没有。

Finished_at (`datetime` 或 `None`) ：作业完成时。当作业仍在调度或运行时没有。

docker_image（`str` 或 `None`）：用于作业的 Docker Hub 中的 Docker 映像。如果存在 space_id，则可以为 None。

space_id (`str` 或 `None`) ：用于作业的 Hugging Face Spaces 的 Docker 映像。如果 docker_image 存在，则可以为 None。

命令（`list[str]` 或 `None`）：作业的命令，例如`["python", "-c", "print('hello world')"]`

参数（`list[str]`或`None`）：传递给命令的参数环境（`dict[str]`或`None`）：作业的环境变量作为字典。

Secrets (`dict[str]` 或 `None`) ：作业的秘密环境变量（加密）。

风味（`str`或`None`）：硬件风味。请参阅`JobHardware`了解可能的值。例如。 `"cpu-basic"`。

labels（`dict[str, str]`或`None`）：附加到作业的标签（键值对）。

卷（`list[Volume]`或`None`）：安装在作业容器中的卷（存储桶、模型、数据集、空间）。

status : (`JobStatus` 或 `None`): 作业的状态，例如`JobStatus(stage="RUNNING", message=None)` 请参阅[JobStage](/docs/huggingface_hub/v1.27.0/en/package_reference/jobs#huggingface_hub.JobStage) 了解可能的阶段值。

持续时间（`JobDurations`或`None`）：作业的时间细分。适用于所有作业状态，包括调度。

所有者：（`JobOwner`或`None`）：作业的所有者，例如`JobOwner(id="5e9ecfc04957053f60648a3e", name="lhoestq", type="user")`

发起者（`JobInitiator` 或 `None`）：触发作业的原因，例如`JobInitiator(type="scheduled-job", id="...")` 用于 cron 触发的运行。

hide_urls (`list[str]` 或 `None`) ：可访问作业公开端口的公共 URL（每个通过 `expose=` 公开的端口一个），例如`["https://687fb701029421ae5549d998--8000.hf.jobs"]`。 `None` 当没有端口暴露时。访问 URL 需要具有对作业命名空间的读取访问权限的 HF 令牌。ssh_url（`str`或`None`）：作业的SSH端点，例如`"ssh://687fb701029421ae5549d998@ssh.hf.jobs"`。仅当作业以 `ssh=True` 开始时存在。连接需要对作业命名空间的写入权限以及在集线器上注册的 SSH 公钥 (https://huggingface.co/settings/keys)。

包含有关作业的信息。

示例：

```python
>>> from huggingface_hub import run_job
>>> job = run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello from the cloud!')"]
... )
>>> job
JobInfo(id='687fb701029421ae5549d998', created_at=datetime.datetime(2025, 7, 22, 16, 6, 25, 79000, tzinfo=datetime.timezone.utc), started_at=datetime.datetime(2025, 7, 22, 16, 6, 31, 79000, tzinfo=datetime.timezone.utc), finished_at=None, docker_image='python:3.12', space_id=None, command=['python', '-c', "print('Hello from the cloud!')"], arguments=[], environment={}, secrets={}, flavor='cpu-basic', labels=None, status=JobStatus(stage='RUNNING', message=None), durations=JobDurations(scheduling_secs=6, running_secs=2, total_secs=8), owner=JobOwner(id='5e9ecfc04957053f60648a3e', name='lhoestq', type='user'), initiator=JobInitiator(type='user', id='5e9ecfc04957053f60648a3e', name='lhoestq'), endpoint='https://huggingface.co', url='https://huggingface.co/jobs/lhoestq/687fb701029421ae5549d998')
>>> job.id
'687fb701029421ae5549d998'
>>> job.url
'https://huggingface.co/jobs/lhoestq/687fb701029421ae5549d998'
>>> job.status.stage
'RUNNING'
```

### JobOwner[[huggingface_hub.JobOwner]]

#### Huggingface_hub.JobOwner[[huggingface_hub.JobOwner]]

```python
huggingface_hub.JobOwner(id: str, name: str, type: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_jobs_api.py#L113)

### JobStage[[huggingface_hub.JobStage]]

#### Huggingface_hub.JobStage[[huggingface_hub.JobStage]]

```python
huggingface_hub.JobStage(value, names = None, module = None, qualname = None, type = None, start = 1)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_jobs_api.py#L71)

枚举 Hub 上作业的可能阶段。

值可以与字符串进行比较：
```py
assert JobStage.COMPLETED == "COMPLETED"
```

可能的值为：`COMPLETED`、`CANCELED`、`ERROR`、`DELETED`、`SCHEDULING`、`RUNNING`。
取自https://github.com/huggingface/moon-landing/blob/main/server/job_types/JobInfo.ts#L61（私人网址）。

### JobStatus[[huggingface_hub.JobStatus]]

#### Huggingface_hub.JobStatus[[huggingface_hub.JobStatus]]

```python
huggingface_hub.JobStatus(stage: JobStage, message: str | None, expose_urls: list[str] | None, ssh_url: str | None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_jobs_api.py#L105)

### 音量[[huggingface_hub.Volume]]

#### Huggingface_hub.Volume[[huggingface_hub.Volume]]

```python
huggingface_hub.Volume(**kwargs)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L122)

**参数：**

type (`str`) ：卷类型：`"bucket"`、`"model"`、`"dataset"` 或 `"space"`。

source (`str`) : 源标识符，例如`"username/my-bucket"` 或 `"username/my-model"`。mount_path (`str`) : 容器内的挂载路径，例如`"/data"`。必须以`/`开头。

revision (`str` or `None`) ：Git 版本（仅适用于 repos，默认为 `"main"`）。

read_only (`bool` 或 `None`) ：只读挂载。对于存储库强制为`True`，对于存储桶默认为`False`。

路径（`str`或`None`）：要挂载的存储桶/存储库内的子文件夹前缀，例如`"path/to/dir"`。

描述要安装在空间或作业容器中的卷。

#### to_dict[[huggingface_hub.Volume.to_dict]]

```python
to_dict()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L158)

序列化为 Hub API 期望的 JSON 负载。

#### to_uri[[huggingface_hub.Volume.to_uri]]

```python
to_uri()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/_space_api.py#L173)

以 CLI 期望的格式将卷作为 HF 挂载 URI 返回。

### OAuth 和 FastAPI
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/oauth.md