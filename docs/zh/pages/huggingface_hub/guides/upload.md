<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 上传文件到Hub

共享文件和工作是该中心的一个重要方面。 `huggingface_hub` 提供了多种将文件上传到 Hub 的选项。您可以独立使用这些函数，也可以将它们集成到您的库中，使您的用户更方便地与 Hub 交互。

每当您想要将文件上传到 Hub 时，您都需要登录您的 Hugging Face 帐户。有关身份验证的更多详细信息，请查看[this section](../quick-start#authentication)。

## 上传文件

使用 [create_repo()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo) 创建存储库后，您可以使用 [upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 将文件上传到存储库。

指定要上传的文件的路径、要将文件上传到存储库中的位置以及要将文件添加到的存储库的名称。根据您的存储库类型，您可以选择将存储库类型设置为 `dataset`、`model` 或 `space`。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.upload_file(
...     path_or_fileobj="/path/to/local/folder/README.md",
...     path_in_repo="README.md",
...     repo_id="username/test-dataset",
...     repo_type="dataset",
... )
```

## 上传文件夹使用 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 函数将本地文件夹上传到现有存储库。指定本地文件夹的路径
要上传的位置、要将文件夹上传到存储库中的位置以及要添加的存储库的名称
文件夹到.根据您的存储库类型，您可以选择将存储库类型设置为 `dataset`、`model` 或 `space`。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
>>> api.upload_folder(
...     folder_path="/path/to/local/space",
...     repo_id="username/my-cool-space",
...     repo_type="space",
... )
```

默认情况下，将考虑 `.gitignore` 文件来了解哪些文件应该提交或不提交。默认情况下，我们检查提交中是否存在 `.gitignore` 文件，如果不存在，我们检查集线器上是否存在。请注意，只会使用目录根目录下的 `.gitignore` 文件。我们不会检查子目录中的 `.gitignore` 文件。

如果您不想使用硬编码的 `.gitignore` 文件，可以使用 `allow_patterns` 和 `ignore_patterns` 参数来过滤要上传的文件。这些参数接受单个模式或模式列表。模式是标准通配符（通配符模式），如文档[here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)所述。如果同时提供了`allow_patterns`和`ignore_patterns`，则两个约束都适用。

除了 `.gitignore` 文件和允许/忽略模式之外，任何子目录中存在的任何 `.git/` 文件夹都将被忽略。

```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder",
...     path_in_repo="my-dataset/train", # Upload to a specific folder
...     repo_id="username/test-dataset",
...     repo_type="dataset",
...     ignore_patterns="**/logs/*.txt", # Ignore all text logs
... )
```您还可以使用 `delete_patterns` 参数来指定要在同一提交中从存储库中删除的文件。
如果您想在将文件推入其中之前清理远程文件夹并且您不知道哪些文件，这可能会很有用
已经存在。

以下示例将本地 `./logs` 文件夹上传到远程 `/experiment/logs/` 文件夹。仅上传txt文件
但在此之前，存储库上的所有先前日志都将被删除。所有这些都在一次提交中。
```py
>>> api.upload_folder(
...     folder_path="/path/to/local/folder/logs",
...     repo_id="username/trained-model",
...     path_in_repo="experiment/logs/",
...     allow_patterns="*.txt", # Upload all local text files
...     delete_patterns="*.txt", # Delete all remote text files before
... )
```

### 文件如何上传

安装 `hf_xet` 时（默认情况下是这种情况），[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 通过流式管道上传文件：根据集线器检查文件，上传到 Xet 存储后端（内部分块、重复数据删除和重试传输），并以自适应批次提交，所有操作都是并行的。实际上这意味着：- **任何大小的文件夹**：小文件夹在一次提交中上传，而包含许多文件的文件夹会自动拆分为多个提交以保持低于服务器限制。发生这种情况时，后续提交会在提交消息上获得 ` (part 2)`、` (part 3)`、... 后缀。
- **可恢复**：如果上传因任何原因中断，只需重新运行相同的调用即可。已提交的文件会被检测并跳过，并且已上传的块会被重复数据删除——重新上传它们（几乎）不会传输任何数据。不涉及本地状态：您甚至可以从另一台机器恢复。一个例外：对于 `create_pr=True`，重新运行会打开一个新的拉取请求。我们建议在恢复上传时使用 `revision="refs/pr/N"` 重新运行。
- **无双重读取**：文件在单次读取过程中分块上传时进行哈希处理。上传开始之前没有单独的“哈希”阶段。

实时进度显示跟踪三个阶段：

```
Found 5,000 files to upload
  Preparing   ████████████████████  5,000 / 5,000 ✓
  Uploading   ██████████████░░░░░░  423 / 603 files  3.8GB · 19.7MB/s
  Committing  ██████████████████░░  4,580 / 5,000  6 commits
```

如果未安装 `hf_xet`，[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 会回退到旧行为：首先对所有内容进行哈希处理，通过 HTTP 上传，然后创建单个提交。我们始终建议保持安装`hf_xet`以获得更好的稳健性

## 从 CLI 上传您可以从终端使用`hf upload`命令直接将文件上传到Hub。在内部，它使用与上述相同的 [upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 和 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 帮助器。

您可以上传单个文件或整个文件夹：

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors

>>> hf upload Wauplin/my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

`local_path` 和 `path_in_repo` 是可选的并且可以隐式推断。如果未设置`local_path`，工具将
检查本地文件夹或文件是否与`repo_id`同名。如果是这种情况，其内容将被上传。
否则，会引发异常，要求用户显式设置`local_path`。无论如何，如果`path_in_repo`不是
设置后，文件将上传到存储库的根目录。

目标也可以表示为遵循语法 `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]` 的单个`hf://` URI（有关完整语法，请参阅[HF URIs reference](../package_reference/hf_uris)）。然后从 URI 中读取存储库类型、修订版本和 `path_in_repo`，该 URI 不能与 `--repo-type` 和 `--revision` 选项组合使用：

```bash
# Upload a single file to a dataset on a specific branch
>>> hf upload hf://datasets/Wauplin/my-cool-dataset@my-branch/data/train.csv ./train.csv
https://huggingface.co/datasets/Wauplin/my-cool-dataset/blob/my-branch/data/train.csv
```

> [!提示]
> 要获得大文件的最大上传吞吐量，请设置 [⟦T59⟧](../package_reference/environment_variables#hf_xet_high_performance) 环境变量。这启用了`hf_xet`的高性能模式，使可用带宽和CPU内核饱和。注意：旧的 `HF_HUB_ENABLE_HF_TRANSFER=1` 标志不再使用，因为 `hf_transfer` 已被删除，取而代之的是 `hf_xet` — 改为设置 `HF_XET_HIGH_PERFORMANCE=1`。有关CLI上传命令的更多详细信息，请参阅[CLI guide](./cli#hf-upload)。

## 上传一个大文件夹

[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 和 `hf upload` 命令是将文件上传到集线器（包括非常大的文件夹）的首选解决方案。文件通过多次提交流式传输到集线器，如果过程中断，该过程会自动恢复。只需重新运行相同的调用即可跳过已上传的文件。

```py
>>> api.upload_folder(
...     repo_id="HuggingFaceM4/Docmatix",
...     repo_type="dataset",
...     folder_path="/path/to/local/docmatix",
... )
```

或从终端：

```sh
hf upload HuggingFaceM4/Docmatix --repo-type=dataset /path/to/local/docmatix
```

> [!警告]
> 旧版 [upload_large_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_large_folder) 方法和 `hf upload-large-folder` 命令已**弃用**，并将在未来版本中删除。请使用 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) / `hf upload` 代替。

### 大型上传的提示和技巧

处理存储库中的大量数据时需要注意一些限制。考虑到传输数据所需的时间，无论是在 hf.co 上还是在本地工作时，上传/推送在流程结束时失败或遇到体验下降可能会非常烦人。

查看我们的 [Repository limitations and recommendations](https://huggingface.co/docs/hub/repositories-recommendations) 指南​​，了解如何在 Hub 上构建存储库的最佳实践。让我们继续介绍一些实用技巧，以使您的上传过程尽可能顺利。- **从小规模开始**：我们建议从少量数据开始测试您的上传脚本。当失败只需要一点时间时，迭代脚本会更容易。
- **预期失败**：传输大量数据具有挑战性。您不知道会发生什么，但最好始终考虑到某些事情至少会失败一次 - 无论是由于您的机器、您的连接还是我们的服务器造成的。例如，如果您计划上传大量文件，最好在上传下一批文件之前在本地跟踪已上传的文件。您可以确保已提交的 LFS 文件永远不会重新上传两次，但在客户端检查它仍然可以节省一些时间。这就是 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 为您所做的。
- **使用`hf_xet`**：这利用了 Hub 的新存储后端，由 Rust 编写，现在可供所有人使用。事实上，使用`huggingface_hub`时，`hf_xet`已经默认启用！为了获得最佳性能，请将 [⟦T71⟧](../package_reference/environment_variables#hf_xet_high_performance) 设置为环境变量。请注意，启用高性能模式后，该工具将尝试使用所有可用带宽和 CPU 内核。

## 高级功能在大多数情况下，您不需要超过 [upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 和 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 即可将文件上传到 Hub。
然而，`huggingface_hub`具有更高级的功能，使事情变得更容易。让我们来看看他们吧！

### 更快的上传

通过 `hf_xet` 实现更快的上传，这是与 [⟦T74⟧](https://github.com/huggingface/xet-core) 库的 Python 绑定，可实现基于块的重复数据删除，从而实现更快的上传和下载。 `hf_xet` 与 `huggingface_hub` 无缝集成，但使用 Rust `xet-core` 库和 Xet 存储而不是 LFS。 

`hf_xet` 使用 Xet 存储系统，该系统将文件分解为不可变的块，远程存储这些块的集合（称为块或 xorb），并在请求时检索它们以重新组装文件。上传时，在确认用户有权写入此存储库后，`hf_xet`将扫描文件，将它们分解为块并将这些块收集到xorb中（并在已知块中进行重复数据删除），然后将这些xorb上传到Xet内容可寻址服务（CAS），该服务将验证xorb的完整性，注册xorb元数据以及LFS SHA256哈希（以支持查找/下载），并将 xorb 写入远程存储。要启用它，只需安装最新版本的`huggingface_hub`：

```bash
pip install -U "huggingface_hub"
```

从 `huggingface_hub` 0.32.0 开始，这也将安装 `hf_xet`。 

所有其他 `huggingface_hub` API 将继续工作，无需任何修改。要了解有关 Xet 存储和 `hf_xet` 的优势的更多信息，请参阅此 [section](https://huggingface.co/docs/hub/xet/index)。

**集群/分布式文件系统上传注意事项**

从集群上传时，上传的文件通常驻留在分布式或网络文件系统（NFS、EBS、Lustre、Fsx 等）上。 Xet 存储会将这些文件分块并在本地写入块（也称为 xorb），一旦块完成就会上传它们。为了从分布式文件系统上传时获得更好的性能，请确保将 [⟦T85⟧](../package_reference/environment_variables#hfxetcache) 设置为本地磁盘（例如本地 NVMe 或 SSD 磁盘）上的目录。 Xet 缓存的默认位置位于 (`~/.cache/huggingface/xet`) 处的 `HF_HOME` 下，并且该位置位于用户的主目录中，通常也位于分布式文件系统上。

### 非阻塞上传在某些情况下，您希望在不阻塞主线程的情况下推送数据。这对于上传日志和
继续训练时出现工件。为此，您可以在 [upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 和 `run_as_future` 中使用 `run_as_future` 参数
[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)。这将返回一个[⟦T89⟧](https://docs.python.org/3/library/concurrent.futures.html#future-objects)
可用于检查上传状态的对象。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> future = api.upload_folder( # Upload in the background (non-blocking action)
...     repo_id="username/my-model",
...     folder_path="checkpoints-001",
...     run_as_future=True,
... )
>>> future
Future(...)
>>> future.done()
False
>>> future.result() # Wait for the upload to complete (blocking action)
...
```

> [!提示]
> 使用 `run_as_future=True` 时，后台作业会排队。这意味着您可以保证获得这些工作
> 以正确的顺序执行。

尽管后台作业对于上传数据/创建提交最有用，但您可以使用您喜欢的任何方法进行排队
[run_as_future()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.run_as_future)。例如，您可以使用它创建一个存储库，然后在后台将数据上传到其中。的
上传方法中的内置 `run_as_future` 参数只是它的别名。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()
>>> api.run_as_future(api.create_repo, "username/my-model", exists_ok=True)
Future(...)
>>> api.upload_file(
...     repo_id="username/my-model",
...     path_in_repo="file.txt",
...     path_or_fileobj=b"file content",
...     run_as_future=True,
... )
Future(...)
```

### 在存储库之间复制文件

使用 [copy_files()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.copy_files) 在 Hub 上的存储库之间复制文件或文件夹，而无需下载或重新上传大数据。当您想要跨模型变体复制权重、在存储库之间复制数据集文件或跨存储库重新组织文件时，这非常有用。在底层，它使用 [CommitOperationCopy](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy) 操作创建一个提交。

```py
>>> from huggingface_hub import HfApi
>>> api = HfApi()

# Copy a single file between repos
>>> api.copy_files(
...     "hf://username/source-model/weights.safetensors",
...     "hf://username/target-model/weights.safetensors",
... )

# Copy an entire folder
>>> api.copy_files(
...     "hf://datasets/username/source-dataset/data/",
...     "hf://datasets/username/target-dataset/data/",
... )
```

您还可以在同一存储库中复制：

```py
# Duplicate a file in the same repo
>>> api.copy_files(
...     "hf://username/my-model/config.json",
...     "hf://username/my-model/backup/config.json",
... )
```> [!提示]
> 复制文件夹时，源上的尾随 `/` 使用 rsync 样式语义，这意味着复制文件夹的*内容*，而不嵌套文件夹本身。如果没有尾随 `/`，文件夹本身将嵌套在目标位置。

> [!提示]
> [copy_files()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.copy_files)还支持复制文件到[Buckets](./buckets)。更多详情请参阅[Buckets guide](./buckets#copy-files-to-bucket)。

### 预定上传

Hugging Face Hub 可以轻松保存和版本数据。但是，在更新同一文件数千次时存在一些限制。例如，您可能想要保存培训过程或用户的日志
对已部署空间的反馈。在这些情况下，将数据作为数据集上传到 Hub 上是有意义的，但可能很难正确执行。主要原因是您不想对数据的每次更新进行版本控制，因为这会使 git 存储库无法使用。 [CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler)类提供了这个问题的解决方案。这个想法是运行一个后台作业，定期将本地文件夹推送到集线器。假设您有一个
Gradio Space 将一些文本作为输入并生成它的两个翻译。然后，用户可以选择他们喜欢的翻译。对于每次运行，您希望保存输入、输出和用户首选项以分析结果。这是一个
[CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler) 的完美用例；您想要将数据保存到 Hub（可能有数百万用户反馈），但是
您不需要实时保存每个用户的输入。相反，您可以将数据本地保存在 JSON 文件中，然后
每10分钟上传一次。例如：

```py
>>> import json
>>> import uuid
>>> from pathlib import Path
>>> import gradio as gr
>>> from huggingface_hub import CommitScheduler

# Define the file where to save the data. Use UUID to make sure not to overwrite existing data from a previous run.
>>> feedback_file = Path("user_feedback/") / f"data_{uuid.uuid4()}.json"
>>> feedback_folder = feedback_file.parent

# Schedule regular uploads. Remote repo and local folder are created if they don't already exist.
>>> scheduler = CommitScheduler(
...     repo_id="report-translation-feedback",
...     repo_type="dataset",
...     folder_path=feedback_folder,
...     path_in_repo="data",
...     every=10,
... )

# Define the function that will be called when the user submits its feedback (to be called in Gradio)
>>> def save_feedback(input_text:str, output_1: str, output_2:str, user_choice: int) -> None:
...     """
...     Append input/outputs and user feedback to a JSON Lines file using a thread lock to avoid concurrent writes from different users.
...     """
...     with scheduler.lock:
...         with feedback_file.open("a") as f:
...             f.write(json.dumps({"input": input_text, "output_1": output_1, "output_2": output_2, "user_choice": user_choice}))
...             f.write("\n")

# Start Gradio
>>> with gr.Blocks() as demo:
>>>     ... # define Gradio demo + use `save_feedback`
>>> demo.launch()
```

就是这样！用户输入/输出和反馈将作为数据集在中心提供。通过使用唯一的 JSON 文件名，可以保证您不会覆盖先前运行的数据或另一次运行的数据
空间/副本同时推送到同一存储库。有关[CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler)的更多详细信息，您需要了解以下信息：
- **仅附加：**
    假设您仅将内容添加到该文件夹。您只能将数据附加到现有文件或创建
    新文件。删除或覆盖文件可能会损坏您的存储库。
- **git 历史**：
    调度程序将每隔 `every` 分钟提交该文件夹。为了避免过多污染git仓库，
    建议将最小值设置为 5 分钟。此外，调度程序旨在避免空提交。如果没有
    在文件夹中检测到新内容，计划的提交将被删除。
- **错误：**
    调度程序作为后台线程运行。它在您实例化该类时启动，并且永远不会停止。特别是，
    如果上传过程中发生错误（例如：连接问题），调度程序将默默地忽略它并重试
    在下一次预定的提交时。
- **线程安全：**
    在大多数情况下，可以安全地假设您可以写入文件而不必担心锁定文件。的
    如果您在上传时将内容写入文件夹，调度程序不会崩溃或损坏。在实践中，重载应用程序_有可能_发生并发问题。在这种情况下，我们建议使用
    `scheduler.lock`锁，保证线程安全。仅当调度程序扫描文件夹时才会阻止锁定
    更改，而不是在上传数据时更改。您可以放心地假设它不会影响您空间上的用户体验。

#### 空间持久性演示

将数据从空间持久保存到集线器上的数据集是 [CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler) 的主要用例。根据用途而定
在这种情况下，您可能希望以不同的方式构建数据。该结构必须对并发用户具有鲁棒性，并且
重新启动通常意味着生成 UUID。除了稳健性之外，您还应该以 🤗 数据集库可读的格式上传数据，以便以后重用。我们创建了一个[Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
演示了如何保存几种不同的数据格式（您可能需要根据自己的特定需求进行调整）。

#### 自定义上传[CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler) 假设您的数据是仅附加的，并且应该“按原样”上传。然而，你
可能想要自定义数据上传的方式。您可以通过创建一个继承自 [CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler) 的类来做到这一点
并覆盖 `push_to_hub` 方法（随意以任何你想要的方式覆盖它）。你保证它会
在后台线程中每 `every` 分钟调用一次。您不必担心并发和错误，但您
必须小心其他方面，例如推送空提交或重复数据。

在下面的（简化的）示例中，我们覆盖 `push_to_hub` 以将所有 PNG 文件压缩到一个存档中，以避免
超载集线器上的存储库：

```py
class ZipScheduler(CommitScheduler):
    def push_to_hub(self):
        # 1. List PNG files
          png_files = list(self.folder_path.glob("*.png"))
          if len(png_files) == 0:
              return None  # return early if nothing to commit

        # 2. Zip png files in a single archive
        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "train.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zip:
                for png_file in png_files:
                    zip.write(filename=png_file, arcname=png_file.name)

            # 3. Upload archive
            self.api.upload_file(..., path_or_fileobj=archive_path)

        # 4. Delete local png files to avoid re-uploading them later
        for png_file in png_files:
            png_file.unlink()
```

当您覆盖 `push_to_hub` 时，您可以访问 [CommitScheduler](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitScheduler) 的属性，尤其是：
- [HfApi](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi)客户端：`api`
- 文件夹参数：`folder_path`和`path_in_repo`
- 回购参数：`repo_id`、`repo_type`、`revision`
- 螺纹锁：`lock`

> [!提示]
> 有关自定义调度程序的更多示例，请查看我们的 [demo Space](https://huggingface.co/spaces/Wauplin/space_to_dataset_saver)
> 根据您的用例包含不同的实现。

### 创建提交[upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)和[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)函数是高级API，通常使用起来很方便。我们推荐
如果您不需要在较低级别工作，请先尝试这些功能。但是，如果您想在提交级别工作，
您可以直接使用[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)功能。

[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)支持三种类型的操作：

- [CommitOperationAdd](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd) 将文件上传到集线器。如果文件已存在，则覆盖文件内容。此操作接受两个参数：

  - `path_in_repo`：上传文件的存储库路径。
  - `path_or_fileobj`：文件系统上文件的路径或类似文件的对象。这是要上传到集线器的文件的内容。

- [CommitOperationDelete](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitOperationDelete) 从存储库中删除文件或文件夹。此操作接受 `path_in_repo` 作为参数。

- [CommitOperationCopy](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitOperationCopy) 在存储库内或跨存储库复制文件。此操作接受以下参数：- `src_path_in_repo`：要复制的文件的存储库路径。
  - `path_in_repo`：文件应复制到的存储库路径。
  - `src_revision`：可选 - 如果要从不同分支/修订版复制文件，则要复制文件的修订版。
  - `src_repo_id`：可选 - 要从中复制的源存储库（例如`"username/source-model"`）。默认为目标存储库。
  - `src_repo_type`：可选 - 源存储库的类型（`"model"`、`"dataset"` 或 `"space"`）。当`src_repo_id`设置时需要。

例如，如果您要上传两个文件并删除 Hub 存储库中的一个文件：

1. 使用相应的`CommitOperation`添加或删除文件以及删除文件夹：

```py
>>> from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
>>> api = HfApi()
>>> operations = [
...     CommitOperationAdd(path_in_repo="LICENSE.md", path_or_fileobj="~/repo/LICENSE.md"),
...     CommitOperationAdd(path_in_repo="weights.h5", path_or_fileobj="~/repo/weights-final.h5"),
...     CommitOperationDelete(path_in_repo="old-weights.h5"),
...     CommitOperationDelete(path_in_repo="logs/"),
...     CommitOperationCopy(src_path_in_repo="image.png", path_in_repo="duplicate_image.png"),
... ]
```

2. 将您的操作传递给[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)：

```py
>>> api.create_commit(
...     repo_id="lysandre/test-model",
...     operations=operations,
...     commit_message="Upload my model weights and license",
... )
```

除了[upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)和[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)之外，以下函数也在底层使用[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)：

- [delete_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_file) 从 Hub 上的存储库中删除单个文件。
- [delete_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.delete_folder) 从 Hub 上的存储库中删除整个文件夹。
- [metadata_update()](/docs/huggingface_hub/v1.29.0/en/package_reference/cards#huggingface_hub.metadata_update) 更新存储库的元数据。

有关更多详细信息，请查看 [HfApi](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi) 参考。

### 在提交之前预上传 LFS 文件在某些情况下，您可能希望在**进行提交调用之前**将大文件上传到 S3。例如，如果您是
将数据集提交到内存中生成的多个分片中，您需要将分片一一上传
以避免内存不足的问题。解决方案是将每个分片作为单独的提交上传到存储库上。当被
完全有效，该解决方案的缺点是可能会通过生成数十次提交来弄乱 git 历史记录。
为了解决这个问题，您可以将文件逐个上传到 S3，然后在最后创建一个提交。这个
可以将[preupload_lfs_files()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.preupload_lfs_files)与[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)结合使用。

> [!警告]
> 这是高级用户方法。直接使用[upload_file()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file)、[upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder)或[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)代替处理
> 在绝大多数情况下，预上传文件的低级逻辑是可行的方法。主要警告
> [preupload_lfs_files()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.preupload_lfs_files) 是，在实际提交之前，无法在存储库上访问上传文件
> 枢纽。如果您有任何疑问，请随时在我们的 Discord 或 GitHub 问题中联系我们。

这是一个简单的示例，说明如何预上传文件：

```py
>>> from huggingface_hub import CommitOperationAdd, preupload_lfs_files, create_commit, create_repo

>>> repo_id = create_repo("test_preupload").repo_id

>>> operations = [] # List of all `CommitOperationAdd` objects that will be generated
>>> for i in range(5):
...     content = ... # generate binary content
...     addition = CommitOperationAdd(path_in_repo=f"shard_{i}_of_5.bin", path_or_fileobj=content)
...     preupload_lfs_files(repo_id, additions=[addition])
...     operations.append(addition)

>>> # Create commit
>>> create_commit(repo_id, operations=operations, commit_message="Commit all shards")
```首先，我们一一创建[CommitOperationAdd](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.CommitOperationAdd)对象。在现实世界的示例中，这些将包含
生成的碎片。每个文件都会在生成下一个文件之前上传。在[preupload_lfs_files()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.preupload_lfs_files)步骤中，**
`CommitOperationAdd` 对象发生变异**。您只能使用它直接将其传递给[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)。主要
对象的更新是**从中删除**二进制内容，这意味着如果
您不存储对它的另一个引用。这是预料之中的，因为我们不想将以下内容保留在内存中
已经上传了。最后，我们通过将所有操作传递给[create_commit()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.create_commit)来创建提交。你可以通过
尚未处理的其他操作（添加、删除或复制）将被正确处理。

### 从集线器下载文件
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/download.md