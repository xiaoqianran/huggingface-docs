<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 管理你的空间

在本指南中，我们将了解如何管理您的 Space 运行时
([secrets](https://huggingface.co/docs/hub/spaces-overview#managing-secrets),
[hardware](https://huggingface.co/docs/hub/spaces-gpus) 和卷）使用 `huggingface_hub`。

## 搜索空间

您可以使用 [search_spaces()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.search_spaces) 使用语义搜索来搜索 Hub 上的空间。这对多词查询使用基于嵌入的搜索，对单词查询使用全文搜索。

```py
>>> from huggingface_hub import search_spaces
>>> results = list(search_spaces("generate image"))
>>> results[0]
SpaceSearchResult(id='mrfakename/Z-Image-Turbo', title='Z Image Turbo', sdk='gradio', likes=2867, ...)
```

您可以按 SDK 或标签过滤结果：

```py
>>> results = search_spaces("chatbot", sdk="gradio", filter="mcp-server")
```

或者通过 CLI：

```bash
>>> hf spaces search "generate image"
>>> hf spaces search "chatbot" --sdk gradio --limit 5
```

## 一个简单的例子：配置秘密和硬件。

以下是在 Hub 上创建和设置空间的端到端示例。

### 在 Hub 上创建空间

```py
>>> from huggingface_hub import HfApi
>>> repo_id = "Wauplin/my-cool-training-space"
>>> api = HfApi()

# For example with a Gradio SDK
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="gradio")
```

### 从模板创建空间

您可以从 Hub 上提供的官方模板之一（例如 JupyterLab、Gradio 聊天机器人、Streamlit 应用程序等）播种新空间，而不是从空空间开始。使用 [list_space_templates()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.list_space_templates) 列出可用模板，然后将模板的 `repo_id` （或其简称 `name`）作为 `space_template` 传递给 [create_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo)。请注意，`space_sdk` 仍然是必需的：模板在 SDK 设置卡元数据时播种文件。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# List available templates
>>> for template in api.list_space_templates():
...     print(f"{template.name} ({template.repo_id})")
Streamlit (streamlit/streamlit-template-space)
JupyterLab (SpacesExamples/jupyterlab)
chatbot (gradio-templates/chatbot)
...

# Create a Space from a template, by repo id or by short name
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_template="SpacesExamples/jupyterlab")
>>> api.create_repo(repo_id=repo_id, repo_type="space", space_template="JupyterLab")
```

建议将某些模板设为私有（例如 JupyterLab）。如果您没有明确设置可见性，此类空间将自动创建为私有空间。在 CLI 中，可以通过 `hf spaces templates` 和 `hf repos create --template` 获得相同的功能：

```bash
# List available templates
>>> hf spaces templates
NAME            REPO_ID                                        SDK    PREFERRED_PRIVATE
--------------- ---------------------------------------------- ------ -----------------
Streamlit       streamlit/streamlit-template-space             docker
JupyterLab      SpacesExamples/jupyterlab                      docker ✔
Argilla         argilla/argilla-template-space                 docker
Livebook        livebook-dev/livebook                          docker
...

# Create a Space from a template
>>> hf repos create my-jupyterlab --type space --template jupyterlab
```

### 复制一个空格

如果您想从现有空间构建而不是从头开始，这可能会很有用。
如果您想要控制公共空间的配置/设置，它也很有用。更多详情请参见[duplicate_repo()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.duplicate_repo)。

```py
>>> api.duplicate_repo("multimodalart/dreambooth-training", repo_type="space")
```

### 使用您首选的解决方案上传您的代码

以下是将本地文件夹`src/`从您的计算机上传到您的空间的示例：

```py
>>> api.upload_folder(repo_id=repo_id, repo_type="space", folder_path="src/")
```

在此步骤中，您的应用程序应该已经在 Hub 上免费运行！
但是，您可能希望使用机密和升级的硬件进一步配置它。

### 配置秘密和变量

您的空间可能需要一些密钥、令牌或变量才能工作。
更多详情请参见[docs](https://huggingface.co/docs/hub/spaces-overview#managing-secrets)。
例如，从您的空间生成后，用于将图像数据集上传到集线器的 HF 令牌。

```py
>>> api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value="hf_api_***")
>>> api.add_space_variable(repo_id=repo_id, key="MODEL_REPO_ID", value="user/repo")
```

您可以列出现有的秘密和变量。秘密值是只写的，因此仅返回键、描述和更新时间戳：

```py
>>> api.get_space_secrets(repo_id=repo_id)
{'HF_TOKEN': SpaceSecret(key='HF_TOKEN', description=None, updated_at=datetime.datetime(...))}
>>> api.get_space_variables(repo_id=repo_id)
{'MODEL_REPO_ID': SpaceVariable(key='MODEL_REPO_ID', value='user/repo', description=None, updated_at=...)}
```

秘密和变量也可以被删除：
```py
>>> api.delete_space_secret(repo_id=repo_id, key="HF_TOKEN")
>>> api.delete_space_variable(repo_id=repo_id, key="MODEL_REPO_ID")
```> [!提示]
> 在您的空间内，秘密可作为环境变量（或
> Streamlit 机密管理（如果使用 Streamlit）。无需通过 API 获取它们！

> [!警告]
> 空间配置（机密或硬件）的任何更改都将触发应用程序的重新启动。

**奖励：在创建或复制空间时设置秘密和变量！**

创建或复制空间时可以设置秘密和变量：

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     space_variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_secrets=[{"key"="HF_TOKEN", "value"="hf_api_***"}, ...],
...     space_variables=[{"key"="MODEL_REPO_ID", "value"="user/repo"}, ...],
... )
```

### 配置硬件

默认情况下，您的 Space 将免费运行在 CPU 环境上。你可以升级硬件
在 GPU 上运行它。需要支付卡或社区补助金才能升级您的
空间。更多详情请参见[docs](https://huggingface.co/docs/hub/spaces-gpus)。

```py
# Use `SpaceHardware` enum
>>> from huggingface_hub import SpaceHardware
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM)

# Or simply pass a string value
>>> api.request_space_hardware(repo_id=repo_id, hardware="t4-medium")
```

硬件更新不会立即完成，因为您的空间必须重新加载到我们的服务器上。
您可以随时检查您的空间正在运行哪些硬件，以了解您的请求是否有效
已经满足了。

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.stage
"RUNNING_BUILDING"
>>> runtime.hardware
"cpu-basic"
>>> runtime.requested_hardware
"t4-medium"
```

您现在已经拥有一个完全配置的空间。确保将您的空间降级回“cpu-classic”
当你使用完它时。

**奖励：创建或复制空间时请求硬件！**

升级后的硬件将在您的空间建成后自动分配给您。

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="cpu-upgrade",
...     space_sleep_time="7200", # 2 hours in secs
... )
```
```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_hardware="cpu-upgrade",
...     space_sleep_time="7200", # 2 hours in secs
... )
```### 暂停并重新启动您的空间

默认情况下，如果您的空间在升级的硬件上运行，它将永远不会停止。但为了避免被扣款，
当您不使用它时，您可能想暂停它。使用 [pause_space()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.pause_space) 可以实现这一点。暂停的空间将是
在空间所有者通过 UI 或使用 [restart_space()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.restart_space) 通过 API 重新启动它之前，该空间处于非活动状态。
有关暂停模式的更多详细信息，请参阅[this section](https://huggingface.co/docs/hub/spaces-gpus#pause)

```py
# Pause your Space to avoid getting billed
>>> api.pause_space(repo_id=repo_id)
# (...)
# Restart it when you need it
>>> api.restart_space(repo_id=repo_id)
```

另一种可能性是为您的空间设置超时。如果您的空间处于非活动状态的时间超过超时时间，
它会去睡觉。任何登陆您空间的访客都会将其重新启动。您可以使用设置超时
[set_space_sleep_time()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.set_space_sleep_time)。有关睡眠模式的更多详细信息，请参阅[this section](https://huggingface.co/docs/hub/spaces-gpus#sleep-time)。

```py
# Put your Space to sleep after 1h of inactivity
>>> api.set_space_sleep_time(repo_id=repo_id, sleep_time=3600)
```

注意：如果您使用“cpu-basic”硬件，则无法配置自定义睡眠时间。您的空间将自动
48 小时不活动后暂停。

**奖励：在请求硬件时设置睡眠时间**

升级后的硬件将在您的空间建成后自动分配给您。

```py
>>> api.request_space_hardware(repo_id=repo_id, hardware=SpaceHardware.T4_MEDIUM, sleep_time=3600)
```

**奖励：在创建或复制空间时设置睡眠时间！**

```py
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio"
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```
```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_hardware="t4-medium",
...     space_sleep_time="3600",
... )
```

### 通过读取日志来调试失败的空间当空间无法构建或在运行时崩溃时，您通常在浏览器中查看的日志也可以通过[fetch_space_logs()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.fetch_space_logs)以编程方式获得。这对于无法打开浏览器的脚本或代理工作流程特别有用。

```py
# Drain the currently available run logs and return immediately (like `docker logs`)
>>> for line in api.fetch_space_logs(repo_id=repo_id):
...     print(line, end="")

# Read the container build logs instead (useful when the Space is stuck in BUILD_ERROR)
>>> for line in api.fetch_space_logs(repo_id=repo_id, build=True):
...     print(line, end="")

# Stream run logs in real time until the server closes the stream (Ctrl-C to stop)
>>> for line in api.fetch_space_logs(repo_id=repo_id, follow=True):
...     print(line, end="")
```

CLI 提供相同的功能：

```bash
hf spaces logs username/my-space             # drain run logs
hf spaces logs username/my-space --build     # read build logs
hf spaces logs username/my-space -f          # stream in real time
hf spaces logs username/my-space -n 50       # last 50 lines only
```

### SSH 进入空间（开发模式）

[Dev Mode](https://huggingface.co/docs/hub/spaces-dev-mode) 可让您通过 SSH 连接到正在运行的 Space 容器，以进行实时调试和开发。使用 `hf spaces ssh` 直接从终端打开会话。如果空间上尚未启用开发模式，CLI 将提示您启用它（或通过 `--auto` 跳过提示）。

您的 SSH 公钥必须在 [in your settings](https://huggingface.co/settings/keys) 注册。

```bash
# SSH into a Space (enables Dev Mode if needed)
hf spaces ssh username/my-space

# Auto-enable Dev Mode without prompting
hf spaces ssh username/my-space --auto

# Print the SSH command without running it
hf spaces ssh username/my-space --dry-run

# Use a specific SSH key
hf spaces ssh username/my-space -i ~/.ssh/id_ed25519
```

您还可以使用 `hf spaces dev-mode` 在没有 SSH 的情况下启用开发模式，它会打印 SSH、VS Code、Cursor 和 Windsurf 的连接说明：

```bash
hf spaces dev-mode username/my-space
```

### 在您的空间中挂载卷

您可以将 Hub 资源（模型、数据集或存储桶）作为卷装载到空间容器中。这使您的 Space 可以直接文件系统访问这些资源，而无需在代码中下载它们。创建或复制空间时可以直接设置卷：

```py
>>> from huggingface_hub import Volume
>>> api.create_repo(
...     repo_id=repo_id,
...     repo_type="space",
...     space_sdk="gradio",
...     space_volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/data"),
...     ],
... )
```
```py
>>> api.duplicate_repo(
...     from_id=repo_id,
...     repo_type="space",
...     space_volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/data"),
...     ],
... )
```您可以通过 Space 运行时检查当前安装了哪些卷：

```py
>>> runtime = api.get_space_runtime(repo_id=repo_id)
>>> runtime.volumes
[Volume(type='model', source='username/my-model', mount_path='/models', read_only=True), ...]
```

如果您需要更新现有空间上的卷，请使用[set_space_volumes()](/docs/huggingface_hub/v1.30.0/en/package_reference/hf_api#huggingface_hub.HfApi.set_space_volumes)。请注意，这会替换所有先前安装的卷。

```py
>>> api.set_space_volumes(
...     repo_id=repo_id,
...     volumes=[
...         Volume(type="model", source="username/my-model", mount_path="/models", read_only=True),
...         Volume(type="dataset", source="username/my-dataset", mount_path="/data", read_only=True),
...         Volume(type="bucket", source="username/my-bucket", mount_path="/output"),
...     ],
... )
```

要从空间中删除所有卷：

```py
>>> api.delete_space_volumes(repo_id=repo_id)
```

> [!注意]
> 模型、数据集和空间始终以只读方式安装。仅存储桶支持读写挂载。

> [!警告]
> 设置卷将替换任何先前安装的卷。要将卷添加到现有列表，请首先从运行时读取当前卷并将它们包含在新列表中。

所有卷操作也可以从 CLI 进行：

```bash
# List current volumes
hf spaces volumes ls username/my-space

# Set (replace) volumes
hf spaces volumes set username/my-space \
    -v hf://models/username/my-model:/models \
    -v hf://buckets/username/my-bucket:/data

# Remove all volumes
hf spaces volumes delete username/my-space
```

## 更高级：暂时升级你的空间！

空间允许许多不同的用例。有时，您可能想要
要在特定硬件上临时运行空间，请执行某些操作，然后将其关闭。在
在本节中，我们将探讨如何利用 Spaces 按需微调模型。
这只是解决这个特定问题的一种方法。必须将其视为建议
并适应您的用例。假设我们有一个空间来微调模型。这是一个 Gradio 应用程序，它作为输入
模型 ID 和数据集 ID。工作流程如下：

0.（提示用户输入模型和数据集）
1. 从 Hub 加载模型。
2. 从 Hub 加载数据集。
3. 在数据集上微调模型。
4. 将新模型上传到 Hub。

步骤 3. 需要定制硬件，但您不希望您的 Space 一直在付费上运行
图形处理器。解决方案是动态请求硬件进行训练并关闭它
之后就下来了。由于请求硬件会重新启动您的空间，因此您的应用程序必须以某种方式“记住”
它正在执行的当前任务。有多种方法可以做到这一点。在本指南中
我们将看到一种使用数据集作为“任务调度程序”的解决方案。

### 应用程序骨架

您的应用程序如下所示。启动时，检查是否计划了任务，如果是，
在正确的硬件上运行它。完成后，将硬件设置回自由计划 CPU 并
提示用户执行新任务。> [!警告]
> 这样的工作流程不支持像普通演示那样的并发访问。
> 特别是，当训练发生时，该界面将被禁用。
> 最好将您的存储库设置为私有，以确保您是唯一的用户。

```py
# Space will need your token to request hardware: set it as a Secret !
HF_TOKEN=REDACTED

# Space own repo_id
TRAINING_SPACE_ID = "Wauplin/dreambooth-training"

from huggingface_hub import HfApi, SpaceHardware
api = HfApi(token=HF_TOKEN)

# On Space startup, check if a task is scheduled. If yes, finetune the model. If not,
# display an interface to request a new task.
task = get_task()
if task is None:
    # Start Gradio app
    def gradio_fn(task):
        # On user request, add task and request hardware
        add_task(task)
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)

    gr.Interface(fn=gradio_fn, ...).launch()
else:
    runtime = api.get_space_runtime(repo_id=TRAINING_SPACE_ID)
    # Check if Space is loaded with a GPU.
    if runtime.hardware == SpaceHardware.T4_MEDIUM:
        # If yes, finetune base model on dataset !
        train_and_upload(task)

        # Then, mark the task as "DONE"
        mark_as_done(task)

        # DO NOT FORGET: set back CPU hardware
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.CPU_BASIC)
    else:
        api.request_space_hardware(repo_id=TRAINING_SPACE_ID, hardware=SpaceHardware.T4_MEDIUM)
```

### 任务调度器

可以通过多种方式来完成任务安排。这是一个如何使用它来完成的示例
存储为数据集的简单 CSV。

```py
# Dataset ID in which a `tasks.csv` file contains the tasks to perform.
# Here is a basic example for `tasks.csv` containing inputs (base model and dataset)
# and status (PENDING or DONE).
#     multimodalart/sd-fine-tunable,Wauplin/concept-1,DONE
#     multimodalart/sd-fine-tunable,Wauplin/concept-2,PENDING
TASK_DATASET_ID = "Wauplin/dreambooth-task-scheduler"

def _get_csv_file():
    return hf_hub_download(repo_id=TASK_DATASET_ID, filename="tasks.csv", repo_type="dataset", token=HF_TOKEN)

def get_task():
    with open(_get_csv_file()) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if row[2] == "PENDING":
                return row[0], row[1] # model_id, dataset_id

def add_task(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Quick and dirty way to add a task
        path_or_fileobj=(tasks + f"\n{model_id},{dataset_id},PENDING").encode()
    )

def mark_as_done(task):
    model_id, dataset_id = task
    with open(_get_csv_file()) as csv_file:
        with open(csv_file, "r") as f:
            tasks = f.read()

    api.upload_file(
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo="tasks.csv",
        # Quick and dirty way to set the task as DONE
        path_or_fileobj=tasks.replace(
            f"{model_id},{dataset_id},PENDING",
            f"{model_id},{dataset_id},DONE"
        ).encode()
    )
```

### 搜索中心
https://huggingface.co/docs/huggingface_hub/v1.30.0/guides/search.md