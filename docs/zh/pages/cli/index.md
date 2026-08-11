<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 命令行界面 (CLI)

`huggingface_hub` Python 包附带一个名为 `hf` 的内置 CLI。该工具允许您直接从终端与 Hugging Face Hub 进行交互。例如，您可以登录帐户、创建存储库、上传和下载文件等。它还具有配置计算机或管理缓存的便捷功能。在本指南中，我们将了解 CLI 的主要功能以及如何使用它们。

> [!提示]
> 本指南涵盖了 `hf` CLI 最重要的功能。
> 有关所有命令和选项的完整参考，请参阅 [CLI reference](../package_reference/cli)。

> [!提示]
> 将 `hf` CLI 与 AI 代理一起使用？安装技能并查看[Hugging Face CLI for AI Agents](https://huggingface.co/docs/hub/agents-cli)指南。
> ```bash
> # for Codex, Cursor, OpenCode, Pi and other agents that load skills from `.agents/skills`
> hf skills add
> # includes the above + Claude Code
> hf skills add --claude
> ```
> 独立安装程序会为您安装它（见下文），然后 `hf update` 会刷新它。

## 开始使用

### 独立安装程序（推荐）

您可以使用单个命令安装 `hf` CLI：

在 macOS 和 Linux 上：

```bash
>>> curl -LsSf https://hf.co/cli/install.sh | bash
```

在 Windows 上：

```powershell
>>> powershell -ExecutionPolicy ByPass -c "irm https://hf.co/cli/install.ps1 | iex"
```

安装程序还会在全球范围内安装 [⟦T194⟧ skill](https://huggingface.co/docs/hub/agents-cli)，供 Claude Code 和任何读取 `~/.agents/skills` 的代理使用。通过 `--exclude-skill` 跳过它：

```bash
>>> curl -LsSf https://hf.co/cli/install.sh | bash -s -- --exclude-skill
```

```powershell
>>> powershell -ExecutionPolicy ByPass -c "& ([scriptblock]::Create((irm https://hf.co/cli/install.ps1))) -ExcludeSkill"
```

安装后，您可以检查 CLI 是否设置正确：

```bash
>>> hf --help
Usage: hf [OPTIONS] COMMAND [ARGS]...

  Hugging Face Hub CLI

Options:
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or customize the installation.
  -h, --help            Show this message and exit.

Main commands:
  auth                 Manage authentication (login, logout, etc.).
  buckets              Commands to interact with buckets.
  cache                Manage local cache directory.
  collections          Interact with collections on the Hub.
  datasets             Interact with datasets on the Hub.
  download             Download files from the Hub.
  endpoints            Manage Hugging Face Inference Endpoints.
  extensions           Manage hf CLI extensions.
  jobs                 Run and manage Jobs on the Hub.
  models               Interact with models on the Hub.
  papers               Interact with papers on the Hub.
  repo                 Manage repos on the Hub.
  skills               Manage skills for AI assistants.
  spaces               Interact with spaces on the Hub.
  sync                 Sync files between local directory and a bucket.
  upload               Upload a file or a folder to the Hub.
  upload-large-folder  [Deprecated] Use 'hf upload' instead.

Help commands:
  env      Print information about the environment.
  version  Print information about the hf version.
```如果 CLI 安装正确，您应该会看到 CLI 中所有可用选项的列表。如果您收到诸如`command not found: hf`之类的错误消息，请参阅[Installation](../installation)指南。

> [!提示]
> `--help` 选项对于获取有关命令的更多详细信息非常方便。您可以随时使用它来列出所有可用选项及其详细信息。例如，`hf upload --help` 提供了有关如何使用 CLI 上传文件的更多信息。

### 使用紫外线

使用 `hf` CLI 最简单的方法是使用 [⟦T201⟧](https://docs.astral.sh/uv/concepts/tools/)。它始终在隔离环境中运行最新版本 - 无需安装！

确保首先安装`uv`。请参阅[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) 了解说明。

然后直接使用CLI：

```bash
>>> uvx hf auth login
>>> uvx hf download
>>> uvx hf ...
```

> [!提示]
> `uvx hf` 使用[⟦T204⟧ PyPI package](https://pypi.org/project/hf/)。

### 使用 pip 安装

CLI 还随核心 `huggingface_hub` 包一起提供：

```bash
>>> pip install -U "huggingface_hub"
```

### 使用自制软件

您还可以使用 [Homebrew](https://brew.sh/) 安装 CLI：

```bash
>>> brew install hf
```

查看 Homebrew Huggingface 页面 [here](https://formulae.brew.sh/formula/hf) 了解更多详细信息。

### 更新中

要升级到最新版本，请运行：

```bash
>>> hf update
```这会检测`hf`的安装方式（Homebrew、独立安装程序或 pip）并运行匹配的更新命令。如果全局安装了`hf-cli`技能，它也会刷新，以便代理看到新的命令界面。如果不是，它会保持卸载状态：如果您跳过或删除它，更新永远不会将其恢复。

默认情况下，当 PyPI 上有更新版本时，CLI 还会向 stderr 打印一行黄色警告。要使其静音（例如在离线 CI 中），请设置 `HF_HUB_DISABLE_UPDATE_CHECK=1`。

## 输出格式

大多数 `hf` 命令接受同一组全局格式化标志。它们记录在每个 `--help` 页面的专用 `Formatting options` 部分中，并且可以添加到任何命令中，而无需为每个命令声明它们：|旗帜|同等|描述 |
| ---- | ---------- | ----------- |
| `--format <value>` | — |明确选择输出格式。接受的值：`auto`（默认）、`human`、`agent`、`json`、`quiet`。 |
| `--json` | `--format json` |打印结构化 JSON。对于通过管道输送到 `jq` 或其他脚本很有用。 |
| `-q`、`--quiet` | `--format quiet` |仅打印 ID（每行一个）。对于将 ID 通过管道传输到其他命令中非常有用。 |
| `--no-truncate` | — |在人类表格中显示完整值，而不是使用 `...` 缩短长值。该表可能会溢出终端宽度。使用 `--format json` 进行列表/字典列的结构化输出。 |

`auto`（默认）为交互式终端选​​择 `human`，当 AI 代理调用 CLI 时选择`agent`。 `human`增加了颜色和漂亮的桌子； `agent` 生成不截断的制表符分隔值； `json` 发出一个紧凑的 JSON 对象或数组。混合两个输出模式标志（例如 `--json` 与 `--format table`）会引发使用错误。人类表格会自动适应您的终端宽度：仅当自然宽度溢出屏幕时，最宽的列才会缩短（因此在宽终端上，您通常会看到没有`...`的完整值）。通过 `--no-truncate` 强制使用全值，无论宽度如何；使用 `--format json` 进行列表/字典列的结构化输出。

```bash
# JSON output for scripting
>>> hf models ls --search bert --limit 2 --json | jq '.[].id'

# IDs only, one per line
>>> hf collections ls --owner nvidia -q
nvidia/nemotron-supervised-fine-tuning-69eab9824c9120a3a3b1e25e
nvidia/nvidia-nemotron-v3-69388dda16167bb1607171ea
```

> [!提示]

## hf 身份验证登录

在许多情况下，您必须登录 Hugging Face 帐户才能与 Hub 交互（下载私人存储库、上传文件、创建 PR 等）。为此，请在终端中运行以下命令：

```bash
>>> hf auth login
```

如果您已经登录，此命令将跳过提示并显示一条消息。要强制重新登录（例如切换令牌），请使用`--force`：

```bash
>>> hf auth login --force
```

默认情况下，该命令会使用浏览器登录：它会打印 URL 和短代码。打开 URL，输入代码，批准请求，CLI 将检索访问令牌并将其保存在您的计算机上。令牌会在一段时间后过期，但只要您继续使用它就会自动刷新。

```
? How would you like to log in?  [Use arrows, Enter to confirm]
> Log in with your browser
  Paste an access token

    Open this URL in your browser:
        https://huggingface.co/oauth/device

    And enter the code: ABCD-EFGH

    Waiting for authorization...
Token is valid.
The token `oauth-wauplin` has been saved to /home/wauplin/.cache/huggingface/stored_tokens
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful.
```您还可以选择以交互方式（选择*粘贴访问令牌*）或直接从命令行粘贴从 [Settings page](https://huggingface.co/settings/tokens) 生成的 [User Access Token](https://huggingface.co/docs/hub/security-tokens)。为了更安全，我们建议将您的令牌作为环境变量传递，以避免将其粘贴到命令历史记录中。

```bash
# Or using an environment variable
>>> hf auth login --token $HF_TOKEN --add-to-git-credential
Token is valid (permission: write).
The token `token_name` has been saved to /home/wauplin/.cache/huggingface/stored_tokens
Your token has been saved in your configured git credential helpers (store).
Your token has been saved to /home/wauplin/.cache/huggingface/token
Login successful
The current active token is: `token_name`
```

当由 AI 代理（自动检测或使用`--format agent`）运行时，该命令永远不会提示：它运行浏览器流程并打印代理可以转发给用户的简单指令，然后等待授权：

```bash
>>> hf auth login --format agent
Ask the user to open https://huggingface.co/oauth/device in a browser and enter the code ABCD-EFGH. The code expires in 900 seconds. Waiting for authorization...
Login successful: logged in as wauplin (token saved as 'oauth-wauplin').
```

`hf auth login` 是交互式的，因此不支持 `--format json` 和 `--format quiet`：通过 `--token` 进行脚本化、非交互式登录。

有关身份验证的更多详细信息，请查看[this section](../quick-start#authentication)。

## hf auth whoami

如果你想知道你是否登录，可以使用`hf auth whoami`。此命令没有任何选项，只是打印您的用户名和您在 Hub 上所属的组织：

```bash
hf auth whoami
Wauplin
orgs:  huggingface,eu-test,OAuthTesters,hf-accelerate,HFSmolCluster
```

如果您没有登录，将会打印一条错误消息。

## hf 身份验证注销

该命令将您注销。实际上，它将删除您计算机上存储的所有令牌。如果要删除特定令牌，可以指定令牌名称作为参数。如果您使用 `HF_TOKEN` 环境变量登录（请参阅 [reference](../package_reference/environment_variables#hftoken)），此命令不会让您注销。如果是这种情况，您必须在计算机配置中取消设置环境变量。

## 高频下载

使用`hf download`命令直接从Hub下载文件。在内部，它使用 [Download](./download) 指南​​中描述的相同 [hf_hub_download()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 和 [snapshot_download()](/docs/huggingface_hub/v1.27.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 帮助程序，并将返回的路径打印到终端。在下面的示例中，我们将介绍最常见的用例。要获得可用选项的完整列表，您可以运行：

```bash
hf download --help
```

### 下载单个文件

要从存储库下载单个文件，只需提供 repo_id 和文件名，如下所示：

```bash
>>> hf download gpt2 config.json
downloading https://huggingface.co/gpt2/resolve/main/config.json to /home/wauplin/.cache/huggingface/hub/tmpwrq8dm5o
(…)ingface.co/gpt2/resolve/main/config.json: 100%|██████████████████████████████████| 665/665 [00:00<00:00, 2.49MB/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

该命令将始终在最后一行打印本地计算机上文件的路径。

要下载位于存储库子目录中的文件，您应该以 posix 格式提供存储库中文件的路径，如下所示：

```bash
>>> hf download HiDream-ai/HiDream-I1-Full text_encoder/model.safetensors
```

### 下载整个存储库

在某些情况下，您只想从存储库下载所有文件。只需指定 repo id 即可完成此操作：

```bash
>>> hf download HuggingFaceH4/zephyr-7b-beta
Fetching 23 files:   0%|                                                | 0/23 [00:00<?, ?it/s]
...
...
/home/wauplin/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/3bac358730f8806e5c3dc7c7e19eb36e045bf720
```

### 下载多个文件您还可以使用单个命令从存储库下载文件的子集。这可以通过两种方式完成。如果您已经有了要下载的文件的精确列表，您只需按顺序提供它们即可：

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files:   0%|                                                                        | 0/2 [00:00<?, ?it/s]
downloading https://huggingface.co/gpt2/resolve/11c5a3d5811f50298f278a704980280950aedb10/model.safetensors to /home/wauplin/.cache/huggingface/hub/tmpdachpl3o
(…)8f278a7049802950aedb10/model.safetensors: 100%|██████████████████████████████| 8.09k/8.09k [00:00<00:00, 40.5MB/s]
Fetching 2 files: 100%|████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00,  3.76it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

另一种方法是使用 `--include` 和 `--exclude` 提供模式来过滤要下载的文件。例如，如果要下载 [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 中的所有 safetensors 文件，除了 FP16 精度的文件：

```bash
>>> hf download stabilityai/stable-diffusion-xl-base-1.0 --include "*.safetensors" --exclude "*.fp16.*"*
Fetching 8 files:   0%|                                                                         | 0/8 [00:00<?, ?it/s]
...
...
Fetching 8 files: 100%|█████████████████████████████████████████████████████████████████████████| 8/8 (...)
/home/wauplin/.cache/huggingface/hub/models--stabilityai--stable-diffusion-xl-base-1.0/snapshots/462165984030d82259a11f4367a4eed129e94a7b
```

### 下载数据集或空间

上面的示例展示了如何从模型存储库下载。要下载数据集或空间，请使用 `--repo-type` 选项：

```bash
# https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k
>>> hf download HuggingFaceH4/ultrachat_200k --repo-type dataset

# https://huggingface.co/spaces/HuggingFaceH4/zephyr-chat
>>> hf download HuggingFaceH4/zephyr-chat --repo-type space

...
```

### 下载特定版本

上面的示例展示了如何从主分支上的最新提交下载。要从特定修订版（提交哈希、分支名称或标签）下载，请使用 `--revision` 选项：

```bash
>>> hf download bigcode/the-stack --repo-type dataset --revision v1.1
...
```

### 使用 hf:// URI

您可以提供单个 `hf://` URI，而不是将存储库类型、修订版和文件路径作为单独的参数和选项传递。 URI 立即对所有内容进行编码，遵循语法 `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]`（有关完整语法，请参阅 [HF URIs reference](../package_reference/hf_uris)）：

```bash
# Equivalent to: hf download bigcode/the-stack --repo-type dataset --revision v1.1
>>> hf download hf://datasets/bigcode/the-stack@v1.1

# Download a single file from a specific revision
>>> hf download hf://datasets/HuggingFaceM4/FineVision@refs/pr/1/data/train.parquet

# Download a subfolder (note the trailing slash)
>>> hf download hf://datasets/HuggingFaceM4/FineVision/art/

# A bare id still works and defaults to a model repo
>>> hf download hf://openai-community/gpt2/config.json
```当给定 URI 时，也无法设置 `--repo-type` 和 `--revision`，因为它们已经是 URI 的一部分（否则会引发错误），并且嵌入在 URI 中的文件路径无法与位置文件名组合。路径上的尾随 `/` 表示子文件夹（与位置参数一样）。包含 `/` 的分支名称必须进行 URL 编码为 `%2F`（例如 `hf://my-org/my-model@feature%2Ffoo`）。

### 下载到本地文件夹

从集线器下载文件的推荐（也是默认）方法是使用缓存系统。但是，在某些情况下，您想要下载文件并将其移动到特定文件夹。这对于使工作流程更接近 git 命令提供的功能很有用。您可以使用 `--local-dir` 选项来做到这一点。

将在本地目录的根目录下创建一个 `.cache/huggingface/` 文件夹，其中包含有关下载文件的元数据。如果文件已经是最新的，这可以防止重新下载文件。如果元数据已更改，则下载新的文件版本。这使得`local-dir`针对仅拉取最新更改进行了优化。

> [!提示]
> 有关如何下载到本地文件的更多详细信息，请查看 [download](./download#download-files-to-a-local-folder) 指南。

```bash
>>> hf download adept/fuyu-8b model-00001-of-00002.safetensors --local-dir fuyu
...
fuyu/model-00001-of-00002.safetensors
```

### 试运行模式在某些情况下，您希望在实际下载之前检查将下载哪些文件。您可以使用 `--dry-run` 参数进行检查。它列出了存储库上要下载的所有文件，并检查它们是否已下载。这可以了解必须下载的文件数量及其大小。

```sh
>>> hf download openai-community/gpt2 --dry-run
[dry-run] Fetching 26 files: 100%|█████████████| 26/26 [00:04<00:00,  6.26it/s]
[dry-run] Will download 11 files (out of 26) totalling 5.6G.
File                              Bytes to download
--------------------------------- -----------------
.gitattributes                    -
64-8bits.tflite                   125.2M
64-fp16.tflite                    248.3M
64.tflite                         495.8M
README.md                         -
config.json                       -
flax_model.msgpack                497.8M
generation_config.json            -
merges.txt                        -
model.safetensors                 548.1M
onnx/config.json                  -
onnx/decoder_model.onnx           653.7M
onnx/decoder_model_merged.onnx    655.2M
onnx/decoder_with_past_model.onnx 653.7M
onnx/generation_config.json       -
onnx/merges.txt                   -
onnx/special_tokens_map.json      -
onnx/tokenizer.json               -
onnx/tokenizer_config.json        -
onnx/vocab.json                   -
pytorch_model.bin                 548.1M
rust_model.ot                     702.5M
tf_model.h5                       497.9M
tokenizer.json                    -
tokenizer_config.json             -
vocab.json                        -
```

欲了解更多详情，请查看[download guide](./download#dry-run-mode)。

###指定缓存目录

如果不使用`--local-dir`，所有文件都会默认下载到`HF_HOME`[environment variable](../package_reference/environment_variables#hfhome)定义的缓存目录中。您可以使用 `--cache-dir` 指定自定义缓存：

```bash
>>> hf download adept/fuyu-8b --cache-dir ./path/to/cache
...
./path/to/cache/models--adept--fuyu-8b/snapshots/ddcacbcf5fdf9cc59ff01f6be6d6662624d9c745
```

### 指定一个令牌

要访问私有或封闭的存储库，您必须使用令牌。默认情况下，将使用本地保存的令牌（使用`hf auth login`）。如果您想显式进行身份验证，请使用 `--token` 选项：

```bash
>>> hf download gpt2 config.json --token=hf_****
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

### 安静模式

默认情况下，`hf download`命令将是详细的。它将打印警告消息、有关下载文件的信息和进度条等详细信息。如果您想消除所有这些，请使用 `--quiet` 选项。仅打印最后一行（即下载文件的路径）。如果您想将输出传递给脚本中的另一个命令，这会很有用。

```bash
>>> hf download gpt2 --quiet
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```### 下载超时

在连接速度较慢的计算机上，您可能会遇到如下超时问题：

```bash
`httpx.TimeoutException: (TimeoutException("HTTPSConnectionPool(host='cdn-lfs-us-1.huggingface.co', port=443): Read timed out. (read timeout=10)"), '(Request ID: a33d910c-84c6-4514-8362-c705e2039d38)')`
```

为了缓解此问题，您可以将 `HF_HUB_DOWNLOAD_TIMEOUT` 环境变量设置为更高的值（默认值为 10）：

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=30
```

欲了解更多详情，请查看[environment variables reference](../package_reference/environment_variables#hfhubdownloadtimeout)。并重新运行您的下载命令。

## 高频上传

使用`hf upload`命令直接将文件上传到Hub。在内部，它使用 [Upload](./upload) 指南中描述的相同 [upload_file()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file) 和 [upload_folder()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 帮助器。在下面的示例中，我们将介绍最常见的用例。要获得可用选项的完整列表，您可以运行：

```bash
>>> hf upload --help
```

### 上传整个文件夹

该命令的默认用法是：

```bash
# Usage:  hf upload [repo_id] [local_path] [path_in_repo]
```

要上传存储库根目录下的当前目录，请使用：

```bash
>>> hf upload my-cool-model . .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

> [!提示]
> 如果存储库尚不存在，则会自动创建。

您还可以上传特定文件夹：

```bash
>>> hf upload my-cool-model ./models .
https://huggingface.co/Wauplin/my-cool-model/tree/main/
```

最后，您可以将文件夹上传到存储库上的特定目的地：

```bash
>>> hf upload my-cool-model ./path/to/curated/data /data/train
https://huggingface.co/Wauplin/my-cool-model/tree/main/data/train
```

### 上传单个文件

您还可以通过设置 `local_path` 指向您计算机上的文件来上传单个文件。如果是这种情况，`path_in_repo`是可选的，并且默认为本地文件的名称：

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/model.safetensors
```如果您想上传单个文件到特定目录，请相应设置`path_in_repo`：

```bash
>>> hf upload Wauplin/my-cool-model ./models/model.safetensors /vae/model.safetensors
https://huggingface.co/Wauplin/my-cool-model/blob/main/vae/model.safetensors
```

### 上传多个文件

要一次从文件夹上传多个文件而不上传整个文件夹，请使用 `--include` 和 `--exclude` 模式。它还可以与 `--delete` 选项结合使用，在上传新文件的同时删除存储库上的文件。在下面的例子中，我们通过删除远程文件并上传除`/logs`中的文件之外的所有文件来同步本地空间：

```bash
# Sync local Space with Hub (upload new files except from logs/, delete removed files)
>>> hf upload Wauplin/space-example --repo-type=space --exclude="/logs/*" --delete="*" --commit-message="Sync local Space with Hub"
...
```

### 上传到数据集或空间

要上传到数据集或空间，请使用 `--repo-type` 选项：

```bash
>>> hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
...
```

### 上传到组织

要将内容上传到组织拥有的存储库而不是个人存储库，您必须在 `repo_id` 中明确指定：

```bash
>>> hf upload MyCoolOrganization/my-cool-model . .
https://huggingface.co/MyCoolOrganization/my-cool-model/tree/main/
```

### 上传到特定版本

默认情况下，文件上传到`main`分支。如果要将文件上传到另一个分支或引用，请使用 `--revision` 选项：

```bash
# Upload files to a PR
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
...
```

**注意：**如果`revision`不存在且`--create-pr`未设置，则会从`main`分支自动创建分支。

### 使用 hf:// URI与 `hf download` 一样，目标可以表示为遵循语法 `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]` 的单个 `hf://` URI（有关完整语法，请参阅 [HF URIs reference](../package_reference/hf_uris)）。存储库类型、修订版本和 `path_in_repo` 均从 URI 中读取：

```bash
# Equivalent to: hf upload Wauplin/my-cool-dataset ./train.csv data/train.csv --repo-type dataset --revision my-branch
>>> hf upload hf://datasets/Wauplin/my-cool-dataset@my-branch/data/train.csv ./train.csv

# Upload a whole folder to the root of a model repo
>>> hf upload hf://Wauplin/my-cool-model ./models
```

当给定 URI 时，也无法设置 `--repo-type` 和 `--revision`，因为它们已经是 URI 的一部分（否则会引发错误），并且嵌入在 URI 中的路径无法与 `path_in_repo` 参数组合。

### 上传并创建 PR

如果您没有推送到存储库的权限，则必须打开 PR 并让作者知道您想要进行的更改。这可以通过设置 `--create-pr` 选项来完成：

```bash
# Create a PR and upload the files to it
>>> hf upload bigcode/the-stack . . --repo-type dataset --revision refs/pr/104
https://huggingface.co/datasets/bigcode/the-stack/blob/refs%2Fpr%2F104/
```

### 定期上传

在某些情况下，您可能希望将定期更新推送到存储库。例如，如果您正在训练模型并且希望每 10 分钟上传一次日志文件夹，则这非常有用。您可以使用 `--every` 选项来执行此操作：

```bash
# Upload new logs every 10 minutes
hf upload training-model logs/ --every=10
```

### 指定提交消息

使用 `--commit-message` 和 `--commit-description` 为您的提交设置自定义消息和描述，而不是默认的消息和描述

```bash
>>> hf upload Wauplin/my-cool-model ./models . --commit-message="Epoch 34/50" --commit-description="Val accuracy: 68%. Check tensorboard for more details."
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### 指定一个令牌要上传文件，您必须使用令牌。默认情况下，将使用本地保存的令牌（使用`hf auth login`）。如果您想显式进行身份验证，请使用 `--token` 选项：

```bash
>>> hf upload Wauplin/my-cool-model ./models . --token=hf_****
...
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

### 安静模式

默认情况下，`hf upload`命令将是详细的。它将打印警告消息、有关上传文件的信息和进度条等详细信息。如果您想静音所有这些，请使用 `--quiet` 选项。仅打印最后一行（即上传文件的 URL）。如果您想将输出传递给脚本中的另一个命令，这会很有用。

```bash
>>> hf upload Wauplin/my-cool-model ./models . --quiet
https://huggingface.co/Wauplin/my-cool-model/tree/main
```

## hf 上传大文件夹

> [!警告]
> `hf upload-large-folder` 已弃用，并将在未来版本中删除。请使用 [⟦T302⟧](#hf-upload) 代替。现在，它可以开箱即用地处理非常大的文件夹，并在重新运行时自动恢复。

```bash
# Upload a large folder to a model repository
>>> hf upload Wauplin/my-cool-model ./large_model_dir

# Upload a dataset
>>> hf upload Wauplin/my-cool-dataset ./large_data_dir --repo-type dataset
```

## 高频桶使用 `hf buckets` 管理 Hugging Face Hub 上的存储桶。 Buckets 在 Hugging Face 上提供类似 S3 的对象存储，由 Xet 存储后端提供支持。与存储库（基于 git 并跟踪文件历史记录）不同，存储桶是远程对象存储容器，专为具有内容可寻址重复数据删除功能的大型文件而设计。它们专为需要简单、快速、可变存储的用例而设计，例如存储训练检查点、日志、中间工件或任何不需要版本控制的大型文件集合。在下面的示例中，我们将介绍最常见的用例。如需完整指南，请参阅[Buckets guide](./buckets)。

### 创建一个桶

要创建新存储桶，请使用`hf buckets create`。默认情况下，该存储桶将在您的命名空间下创建：

```bash
>>> hf buckets create my-bucket
```

您还可以使用 `--private` 标志创建私有存储桶：

```bash
>>> hf buckets create my-bucket --private
```

### 列出并检查存储桶

要列出所有存储桶，请使用 `hf buckets list`（或其简写 `hf buckets ls`）。您还可以列出特定组织中的存储桶：

```bash
>>> hf buckets list
ID                   PRIVATE       SIZE TOTAL_FILES CREATED_AT
-------------------- ------- ---------- ----------- ----------
username/my-bucket                   32           5 2026-02-16
username/checkpoints         117609095         700 2026-02-13
username/logs                321757477        2000 2026-02-13

# Human-readable sizes
>>> hf buckets list -h
ID                   PRIVATE     SIZE TOTAL_FILES CREATED_AT
-------------------- ------- -------- ----------- ----------
username/my-bucket               32 B           5 2026-02-16
username/checkpoints         117.6 MB         700 2026-02-13
username/logs                321.8 MB        2000 2026-02-13

# List buckets in a specific namespace
>>> hf buckets ls my-org

# Filter buckets by name
>>> hf buckets list --search "checkpoint"
```

要获取有关特定存储桶的详细信息（以 JSON 形式返回），请使用 `hf buckets info`：

```bash
>>> hf buckets info username/my-bucket
{
  "id": "username/my-bucket",
  "private": false,
  "created_at": "2026-02-16T15:28:32+00:00",
  "size": 32,
  "total_files": 5
}
```

### 删除一个桶要删除存储桶，请使用`hf buckets delete`。除非您通过`--yes`，否则系统会提示您确认：

```bash
>>> hf buckets delete username/my-bucket --yes
```

### 删除文件

使用`hf buckets remove`（或其简写`hf buckets rm`）从存储桶中删除文件。

要删除单个文件，请指定其路径：

```bash
>>> hf buckets rm username/my-bucket/old-model.bin
```

要删除某个前缀下的所有文件，请使用 `--recursive`：

```bash
>>> hf buckets rm username/my-bucket/logs/ --recursive
```

您还可以定位存储桶中不带前缀的所有文件：

```bash
>>> hf buckets rm username/my-bucket --recursive --include "*.tmp"
```

使用 `--dry-run` 预览将要删除的内容，而无需实际删除任何内容：

```bash
>>> hf buckets rm username/my-bucket/checkpoints/ --recursive --dry-run
```

### 浏览文件

使用 `hf buckets list` 和存储桶 ID 来列出存储桶中的文件：

```bash
>>> hf buckets list username/my-bucket
        2048  2026-01-15 10:30:00  big.bin
           5  2026-01-15 10:30:00  file.txt
              2026-01-15 10:30:00  sub/
```

添加 `-R` 用于递归列表，添加 `-h` 用于人类可读的文件大小和短日期。您还可以使用 `--tree` 显示 ASCII 树视图，或使用 `--tree --quiet` 来显示没有元数据的干净树：

```bash
# Recursive with human-readable sizes
>>> hf buckets list username/my-bucket -R -h
      2.0 KB         Jan 15 10:30  big.bin
         5 B         Jan 15 10:30  file.txt
        14 B         Jan 15 10:30  sub/nested.txt
         4 B         Jan 15 10:30  sub/deep/file.txt

# Tree with human-readable sizes
>>> hf buckets list username/my-bucket --tree -h -R
2.0 KB  Jan 15 10:30  ├── big.bin
   5 B  Jan 15 10:30  ├── file.txt
                      └── sub/
                          ├── deep/
   4 B  Jan 15 10:30  │       └── file.txt
  14 B  Jan 15 10:30  └── nested.txt

# Clean tree without metadata
>>> hf buckets list username/my-bucket --tree --quiet -R
├── big.bin
├── file.txt
└── sub/
    ├── deep/
    │   └── file.txt
    └── nested.txt
```

要按前缀过滤，请将前缀附加到存储桶路径：

```bash
>>> hf buckets list username/my-bucket/sub -R
```

### 复制文件

使用 `hf cp` 在本地计算机、存储库和存储桶之间复制单个文件。源和目标都可以是本地路径、`hf://` URI（存储库或存储桶）或`-`（stdin/stdout）。

> [!提示]
> `hf cp` 也公开为 `hf repos cp` 和 `hf buckets cp` — 所有三个都是完全相同的命令。使用最适合您的工作流程的选项。上传文件（本地 → 存储库或存储桶）：

```bash
# To a repository
>>> hf cp ./model.safetensors hf://username/my-model/model.safetensors

# To a bucket (uses the local filename when the destination ends with /)
>>> hf cp ./data.csv hf://buckets/username/my-bucket/logs/
```

下载文件（存储库或存储桶 → 本地）：

```bash
# From a repository
>>> hf cp hf://datasets/username/my-dataset@refs/pr/1/data.csv ./data.csv

# From a bucket, to the current directory (destination omitted)
>>> hf cp hf://buckets/username/my-bucket/config.json
```

您还可以使用 `-` 流式传输到标准输出或从标准输入：

```bash
# Download to stdout
>>> hf cp hf://buckets/username/my-bucket/config.json - | jq .

# Upload from stdin
>>> echo "hello" | hf cp - hf://username/my-model/hello.txt
```

要在集线器上的两个位置之间进行复制（存储库/存储桶→存储库/存储桶）：

```bash
# Repo to repo
>>> hf cp hf://username/source-model/config.json hf://username/dest-model/config.json

# Repo to bucket
>>> hf cp hf://datasets/username/my-dataset/data/train/ hf://buckets/username/my-bucket/datasets/train/

# Bucket to bucket
>>> hf cp hf://buckets/username/source-bucket/logs/ hf://buckets/username/archive-bucket/logs/
```

在两个 Hub 位置之间复制文件夹时，源路径上的尾随 `/` 控制是嵌套文件夹本身还是仅复制其内容（rsync 样式）：

```bash
# Without trailing slash: "logs" dir is nested => archive/logs/...
>>> hf cp hf://buckets/username/my-bucket/logs hf://buckets/username/archive-bucket/

# With trailing slash: only contents of "logs" are copied => archive/...
>>> hf cp hf://buckets/username/my-bucket/logs/ hf://buckets/username/archive-bucket/
```

注意事项：

- `hf cp` 当涉及本地路径时复制单个文件。要将整个目录复制到本地/从本地复制整个目录，请使用`hf upload`/`hf download`（存储库）或`hf buckets sync`（存储桶）。
- 尚不支持存储桶到存储库的复制。
- 不支持本地到本地复制（使用 shell 的 `cp`）。
- 两个 Hub 位置之间的副本只能在同一 [storage region](https://huggingface.co/docs/hub/storage-regions) 内使用。

### 同步目录

使用 `hf buckets sync` 同步本地机器和存储桶之间的目录。它比较源和目标并仅传输更改的文件。

将本地目录上传到存储桶：

```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket
```

从存储桶下载到本地目录：

```bash
>>> hf buckets sync hf://buckets/username/my-bucket ./data
```

使用 `--delete` 删除源中不存在的目标文件：

```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket --delete
```

您可以使用 `--include` 和 `--exclude` 模式过滤要同步的文件：```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket --include "*.safetensors" --exclude "*.tmp"
```

要仅更新现有文件（跳过新文件），请使用 `--existing`。要仅创建新文件（跳过现有文件），请使用 `--ignore-existing`：

```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket --existing
>>> hf buckets sync ./data hf://buckets/username/my-bucket --ignore-existing
```

为了更加安全，您可以在执行之前生成一个计划进行审查，然后应用它：

```bash
# Generate a plan
>>> hf buckets sync ./data hf://buckets/username/my-bucket --plan sync-plan.jsonl

# Review and apply the plan
>>> hf buckets sync --apply sync-plan.jsonl
```

使用 `--dry-run` 将同步计划作为 JSONL 打印到 stdout，而不执行任何操作。这对于通过管道传输到 `jq` 或其他工具非常方便：

```bash
>>> hf buckets sync ./data hf://buckets/username/my-bucket --dry-run | jq .
```

> [!提示]
> `hf sync` 是`hf buckets sync` 的一个方便的顶级别名。有关所有同步选项的完整详细信息，请参阅[Buckets guide](./buckets#sync-directories)。

## 高频型号

使用 `hf models` 列出 Hub 上的型号并获取有关特定型号的详细信息。

### 列出型号

```bash
# List trending models
>>> hf models ls

# Search for models
>>> hf models ls --search "lora"

# Filter by author
>>> hf models ls --author Qwen

# Filter by pipeline tag
>>> hf models ls --pipeline-tag summarization

# Filter by parameter count
>>> hf models ls --num-parameters min:6B,max:128B

# Only non-gated models
>>> hf models ls --no-gated --author google

# Only models runnable by a given app
>>> hf models ls --apps llama.cpp

# Sort by downloads
>>> hf models ls --sort downloads --limit 10
```

当使用模型 ID 调用时，`hf models ls` 会列出该模型存储库中的文件：

```bash
# List files in a model repo
>>> hf models ls meta-llama/Llama-3.2-1B-Instruct

# List files recursively
>>> hf models ls meta-llama/Llama-3.2-1B-Instruct -R

# Tree view with human-readable sizes
>>> hf models ls meta-llama/Llama-3.2-1B-Instruct --tree -h

# List files at a specific revision
>>> hf models ls meta-llama/Llama-3.2-1B-Instruct --revision main
```

### 获取型号信息

```bash
>>> hf models info Lightricks/LTX-2
```

使用 `--expand` 获取其他属性，例如 `downloads`、`likes`、`tags` 等。

### 获取模型卡

使用 `hf models card` 获取模型的模型卡 (README)。默认情况下，将完整的卡片内容打印到标准输出。

```bash
# Full card (metadata + text)
>>> hf models card google/gemma-4-31B-it

# Just the metadata (from the YAML frontmatter)
>>> hf models card google/gemma-4-31B-it --metadata

# Metadata as JSON (useful for scripting and agents)
>>> hf models card google/gemma-4-31B-it --metadata --format json

# Just the text body (no YAML frontmatter)
>>> hf models card google/gemma-4-31B-it --text
```

## 高频数据集

使用 `hf datasets` 列出 Hub 上的数据集并获取有关特定数据集的详细信息。

### 列出数据集

```bash
# List trending datasets
>>> hf datasets ls

# Search for datasets
>>> hf datasets ls --search "code"

# List official benchmark datasets
>>> hf datasets ls --filter benchmark:official

# Sort by downloads
>>> hf datasets ls --sort downloads --limit 10
```

当使用数据集 ID 调用时，`hf datasets ls` 列出该数据集存储库中的文件：

```bash
# List files in a dataset repo
>>> hf datasets ls HuggingFaceFW/fineweb

# List files recursively with human-readable sizes
>>> hf datasets ls HuggingFaceFW/fineweb -R -h

# Tree view
>>> hf datasets ls HuggingFaceFW/fineweb --tree
```

### 列出数据集排行榜使用 `hf datasets leaderboard` 显示提交到基准数据集的模型分数，以便您可以找到任务的最佳模型或通过基准分数比较模型。

```bash
>>> hf datasets leaderboard SWE-bench/SWE-bench_Verified
>>> hf datasets leaderboard SWE-bench/SWE-bench_Verified --limit 5 --format json
```

### 获取数据集信息

```bash
>>> hf datasets info HuggingFaceFW/fineweb
```

### 获取数据集卡

使用 `hf datasets card` 获取数据集的数据集卡 (README)。默认情况下，将完整的卡片内容打印到标准输出。

```bash
# Full card (metadata + text)
>>> hf datasets card HuggingFaceFW/fineweb

# Just the metadata (from the YAML frontmatter)
>>> hf datasets card HuggingFaceFW/fineweb --metadata

# Metadata as JSON (useful for scripting and agents)
>>> hf datasets card HuggingFaceFW/fineweb --metadata --format json

# Just the text body (no YAML frontmatter)
>>> hf datasets card HuggingFaceFW/fineweb --text
```

### 列出 parquet URL

在编写 SQL 查询之前，使用 `hf datasets parquet` 发现数据集的 parquet 文件 URL。
Hub 上的数据集会通过数据集查看器服务在后端自动转换为 Parquet（对于符合条件的数据集）。
详情请参阅[Parquet conversion guide](https://huggingface.co/docs/dataset-viewer/parquet)。

```bash
>>> hf datasets parquet cfahlgren1/hub-stats
>>> hf datasets parquet cfahlgren1/hub-stats --subset models
>>> hf datasets parquet cfahlgren1/hub-stats --split train
>>> hf datasets parquet cfahlgren1/hub-stats --format json
```

默认表输出包括来自 Hub API 的子集、拆分和镶木地板文件 URL。

### 在数据集 parquet 上运行 SQL

使用 `hf datasets sql` 使用 DuckDB 针对数据集 parquet URL 执行原始 SQL 查询。
首先使用`hf datasets parquet`发现URL，然后使用`read_parquet(...)`直接查询它们。

```bash
>>> hf datasets sql "SELECT COUNT(*) AS rows FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet')"
>>> hf datasets sql "SELECT * FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet') LIMIT 5" --format json
```

如果需要，请先安装 DuckDB：

```bash
# Python package
>>> pip install duckdb

# or standalone DuckDB CLI (Homebrew: macOS/Linux)
>>> brew install duckdb
```

## 高频空格

使用 `hf spaces` 列出 Hub 上的空间并获取有关特定空间的详细信息。

### 列表空间

```bash
# List trending Spaces
>>> hf spaces ls

# Search for Spaces
>>> hf spaces ls --search "3d"

# Sort by likes
>>> hf spaces ls --sort likes --limit 10
```

当使用空间 ID 调用时，`hf spaces ls` 会列出该空间存储库中的文件：

```bash
# List files in a Space repo
>>> hf spaces ls victor/deepsite

# List files recursively with tree view
>>> hf spaces ls victor/deepsite --tree -R -h
```

### 获取空间信息

```bash
>>> hf spaces info victor/deepsite
```

### 获取太空卡使用`hf spaces card`获取空间的空间卡（README）。默认情况下，将完整的卡片内容打印到标准输出。

```bash
# Full card (metadata + text)
>>> hf spaces card mteb/leaderboard

# Just the card metadata (from the YAML frontmatter)
>>> hf spaces card mteb/leaderboard --metadata

# Card metadata as JSON
>>> hf spaces card mteb/leaderboard --metadata --format json

# Just the text body (no YAML frontmatter)
>>> hf spaces card mteb/leaderboard --text
```

> [!提示]
> 暂停或重新启动空间会破坏其容器，因此写入临时文件系统的任何内容都会丢失。要在重新启动后保留数据，请使用 `hf spaces volumes set` 安装卷或存储桶（运行 `hf spaces volumes --help` 了解详细信息）。

### 暂停一个空格

当您不使用空间时，使用`hf spaces pause`暂停该空间（暂停时间不计费）。稍后使用`hf spaces restart`重新启动它。

```bash
>>> hf spaces pause username/my-space
```

### 重新启动空间

使用`hf spaces restart`重新启动空间。通过 `--factory-reboot` 从头开始​​重建 Space，而不使用构建缓存。

```bash
>>> hf spaces restart username/my-space
>>> hf spaces restart username/my-space --factory-reboot
```

### 等待一个空间

使用`hf spaces wait`进行封锁，直到空间完成构建/启动并达到稳定阶段。如果 Space 为 `RUNNING`，则以代码 0 退出，否则为非零（例如 `BUILD_ERROR`）。方便在重新启动或硬件更改后编写脚本。

```bash
>>> hf spaces wait username/my-space

# With a timeout
>>> hf spaces wait username/my-space --timeout 5m

# Chain with restart
>>> hf spaces restart username/my-space && hf spaces wait username/my-space
```

### 列出可用硬件

使用 `hf spaces hardware` 列出 Spaces 的所有可用硬件选项，包括定价。

```bash
>>> hf spaces hardware
```

### 更新空间设置

使用`hf spaces settings`更新空间的设置。

```bash
>>> hf spaces settings username/my-space --sleep-time 3600
>>> hf spaces settings username/my-space --hardware t4-medium
```- `--sleep-time`：空间休眠前的空闲时间（以秒为单位）。使用`-1`永不睡觉。仅适用于升级的硬件（请参阅[Spaces sleep time docs](https://huggingface.co/docs/hub/spaces-gpus#sleep-time)）。
- `--hardware`：硬件风味（例如`cpu-basic`、`t4-medium`、`l4x4`）。运行 `hf spaces hardware` 查看所有选项。

### 管理空间秘密

使用 `hf spaces secrets ls` 列出空间上的秘密，使用 `hf spaces secrets add` 添加或更新一个或多个秘密，使用 `hf spaces secrets delete` 删除一个。通过 `--secrets-file PATH` 从 `.env` 样式文件加载机密。现有密钥将被覆盖。

```bash
>>> hf spaces secrets ls username/my-space
>>> hf spaces secrets add username/my-space -s OPENAI_API_KEY=sk-...
>>> hf spaces secrets add username/my-space --secrets-file .env.secrets
>>> hf spaces secrets delete username/my-space OPENAI_API_KEY --yes
```

> [!注意]
> 秘密值是只写的，因此 `hf spaces secrets ls` 显示密钥、描述和更新时间戳，但从不显示秘密值本身。

### 管理空间环境变量

使用 `hf spaces variables` 管理空间上的非秘密环境变量。与秘密不同，变量是可读的，因此`ls`同时显示键和值。在 `add` 上传递 `--env-file PATH` 以从 `.env` 样式文件加载。

```bash
>>> hf spaces variables ls username/my-space
>>> hf spaces variables add username/my-space -e MODEL_ID=gpt2 -e MAX_TOKENS=512
>>> hf spaces variables add username/my-space --env-file .env
>>> hf spaces variables delete username/my-space MAX_TOKENS --yes
```

### SSH 进入空间（开发模式）

使用 `hf spaces ssh` 打开与 Space 的开发模式容器的 SSH 会话。如果未启用开发模式，CLI 将提示您启用它（或使用 `--auto` 跳过提示）。

您的 SSH 公钥必须注册 [in your settings](https://huggingface.co/settings/keys)。更多详情请参阅[Dev Mode documentation](https://huggingface.co/docs/hub/spaces-dev-mode)。

```bash
# SSH into a Space
>>> hf spaces ssh username/my-space

# Print the SSH command without running it
>>> hf spaces ssh username/my-space --dry-run

# Auto-enable Dev Mode if disabled
>>> hf spaces ssh username/my-space --auto

# Use a specific SSH key
>>> hf spaces ssh username/my-space -i ~/.ssh/id_ed25519
```

## 高频论文使用`hf papers`列出、搜索、获取结构化信息以及阅读 Hub 上论文的 Markdown 内容。

### 列出论文

```bash
# List most recent daily papers
>>> hf papers ls

# List trending papers
>>> hf papers ls --sort=trending

# List papers from a specific date
>>> hf papers ls --date=2025-01-23

# List today's papers
>>> hf papers ls --date=today

# List papers from a specific week
>>> hf papers ls --week=2025-W09

# List papers from a specific month
>>> hf papers ls --month=2025-02

# List papers submitted by a specific user
>>> hf papers ls --submitter=akhaliq

# Limit results
>>> hf papers ls --sort=trending --limit=5
```

### 搜索论文

```bash
# Search papers by keyword
>>> hf papers search "vision language"

# Limit search results
>>> hf papers search "diffusion models" --limit=10

# Output as JSON
>>> hf papers search "attention" --format=json
```

### 获取论文信息

```bash
# Get structured metadata for a paper (returns JSON)
>>> hf papers info 2601.15621
```

### 以 Markdown 形式阅读论文

```bash
# Read the full paper content as markdown
>>> hf papers read 2601.15621
```

## 高频讨论

使用 `hf discussions` 直接从终端管理 Hub 存储库上的讨论和拉取请求。您可以列出、查看、创建、评论、关闭、重新打开和合并讨论和 PR。有关 Hub 社区功能如何工作的完整指南，请参阅 [Discussions and Pull Requests guide](./community)。

### 列出讨论

要列出存储库上的开放讨论和 PR，请将存储库 ID 传递给 `hf discussions list`（或其简写 `hf discussions ls`）：

```bash
>>> hf discussions list username/my-model
```

您可以按种类（`discussion` 或 `pull_request`）、状态（`open`、`closed`、`merged` 或 `all`）或作者缩小结果范围：

```bash
>>> hf discussions list username/my-model --kind pull_request --status merged
>>> hf discussions list username/my-model --author alice
```

对于脚本编写，使用 `--format json` 获取结构化输出，或使用 `--quiet` 仅打印讨论编号（每行一个）：

```bash
>>> hf discussions list username/my-model --format json
>>> hf discussions ls username/my-model --quiet
```

### 获取讨论或 PR 的信息

要检查特定讨论或 PR，请传递存储库 ID 和讨论编号：

```bash
>>> hf discussions info username/my-model 5
```

默认情况下，仅显示讨论元数据（标题、状态、作者等）。添加 `--comments` 以包含完整的对话线程，或添加 `--diff` 以显示 PR 差异：

```bash
>>> hf discussions info username/my-model 5 --comments
>>> hf discussions info username/my-model 5 --diff
```使用 `--format json` 进行机器可读输出，并在管道到其他工具时使用 `--no-color` 去除 ANSI 颜色。

### 创建讨论或 PR

要开启新讨论，请提供带有 `--title` 的标题。您可以选择包含与 `--body` 内联的描述，或从带有 `--body-file` 的文件加载它：

```bash
>>> hf discussions create username/my-model --title "Bug report"
>>> hf discussions create username/my-model --title "Feature request" --body "Please add X"
>>> hf discussions create username/my-model --title "Report" --body-file report.md
```

要创建拉取请求而不是简单的讨论，请添加 `--pull-request` 标志：

```bash
>>> hf discussions create username/my-model --title "Fix typo" --pull-request
```

### 对讨论或 PR 发表评论

通过指定编号向现有讨论或 PR 添加评论。注释正文可以使用 `--body` 内联传递，使用 `--body-file` 从文件中读取，或者使用 `--body-file -` 从标准输入通过管道传输：

```bash
>>> hf discussions comment username/my-model 5 --body "Thanks for reporting!"
>>> hf discussions comment username/my-model 5 --body-file review.md
>>> echo "LGTM" | hf discussions comment username/my-model 5 --body-file -
```

### 编辑评论

通过传递讨论编号和评论 ID 来编辑现有评论。评论 ID 可以通过 `hf discussions info` 检索：

```bash
>>> hf discussions edit username/my-model 5 abc123 --body "Updated comment."
>>> hf discussions edit username/my-model 5 abc123 --body-file fixed.md
```

### 关闭、重新打开和合并

您可以通过 `hf discussions close` 结束讨论或 PR。默认情况下，系统将提示您确认。通过 `--yes` 跳过提示，并通过 `--comment` 留下结束消息：

```bash
>>> hf discussions close username/my-model 5
>>> hf discussions close username/my-model 5 --yes --comment "Resolved"
```

要重新打开之前关闭的讨论，请使用 `hf discussions reopen`：

```bash
>>> hf discussions reopen username/my-model 5 --yes
```

要合并拉取请求，请使用 `hf discussions merge`：

```bash
>>> hf discussions merge username/my-model 5 --yes
```

### 重命名和差异

您可以通过提供新标题来重命名讨论：

```bash
>>> hf discussions rename username/my-model 5 "Updated title"
```要直接在终端中查看拉取请求的差异，请使用 `hf discussions diff`：

```bash
>>> hf discussions diff username/my-model 5
```

## 高频存储库

`hf repos` 允许您列出、创建、删除、移动存储库、更新其设置以及删除 Hugging Face Hub 上的文件。它还包括用于管理分支和标签的子命令。

### 列出存储库

使用 `hf repos ls` 列出所有存储库（模型、数据集、空间和存储桶）以及存储信息，并按存储使用情况排序。默认情况下，仅显示前 30 个存储库：

```bash
# List all your repos (first 30)
>>> hf repos ls
REPOSITORY                TYPE     UPDATED      VISIBILITY   STORAGE  % OF TOTAL
------------------------  -------  ----------   ----------  --------  ----------
username/bucket-raw       bucket   2026-04-29   public        1.7 TB       72.3%
username/my-model         model    2026-05-06   public        4.8 GB       18.1%
username/my-dataset       dataset  2024-09-14   private     598.4 MB        5.2%
Hint: Showing 30 of 42 repos. Use `--limit 0` to list all.
```

按存储库类型过滤、按名称搜索或调整限制：

```bash
# List only models
>>> hf repos ls --type model

# Search by name
>>> hf repos ls --search "bert"

# List repos from an organization
>>> hf repos ls --namespace my-org

# Combine filters
>>> hf repos ls --namespace my-org --type dataset --search "train"

# List all repos (no limit)
>>> hf repos ls --limit 0
```

使用 `--format json` 进行脚本编写，或仅使用 `-q` 进行 ID。管道传输时，使用`--limit 0`导出所有存储库：

```bash
>>> hf repos ls --limit 0 --format json | jq '.[].id'
>>> hf repos ls -q
```

### 创建一个仓库

```bash
>>> hf repos create Wauplin/my-cool-model
Successfully created Wauplin/my-cool-model on the Hub.
Your repo is now available at https://huggingface.co/Wauplin/my-cool-model
```

创建私有数据集或空间：

```bash
>>> hf repos create my-cool-dataset --repo-type dataset --private
>>> hf repos create my-gradio-space --repo-type space --sdk gradio
```

如果存储库可能已存在，请使用 `--exist-ok`，并使用 `--resource-group-id` 来定位企业资源组。

在特定区域创建存储库：

```bash
>>> hf repos create my-model --region us
```

### 删除仓库

```bash
>>> hf repos delete Wauplin/my-cool-model
```

数据集和空间：

```bash
>>> hf repos delete my-cool-dataset --repo-type dataset
>>> hf repos delete my-gradio-space --repo-type space
```

### 移动存储库

```bash
>>> hf repos move old-namespace/my-model new-namespace/my-model
```

### 更新仓库设置

```bash
>>> hf repos settings Wauplin/my-cool-model --gated auto
>>> hf repos settings Wauplin/my-cool-model --private true
>>> hf repos settings Wauplin/my-cool-model --private false
```

- `--gated`：`auto`、`manual`、`false` 之一
- `--private true|false`：设置存储库隐私

### 从存储库中删除文件

`hf repos delete-files <repo_id>` 子命令允许您从存储库中删除文件。以下是一些使用示例。删除文件夹：

```bash
>>> hf repos delete-files Wauplin/my-cool-model folder/
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

删除多个文件：

```bash
>>> hf repos delete-files Wauplin/my-cool-model file.txt folder/pytorch_model.bin
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

使用通配符模式删除文件集。模式是标准通配符（通配符模式），如文档[here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)所述。模式匹配基于[⟦T435⟧](https://docs.python.org/3/library/fnmatch.html)。

> [!警告]
> 请注意，`fnmatch` 跨路径边界匹配 `*`，这与传统的 Unix shell 通配不同。例如，`"data/*.json"` 将匹配`data/file.json` **和** `data/subdir/file.json`。要仅匹配直接目录中的文件，您需要显式列出它们或使用更具体的模式。

```bash
>>> hf repos delete-files Wauplin/my-cool-model "*.txt" "folder/*.bin"
Files correctly deleted from repo. Commit: https://huggingface.co/Wauplin/my-cool-mo...
```

要从存储库中删除文件，您必须经过身份验证和授权。默认情况下，将使用本地保存的令牌（使用`hf auth login`）。如果您想显式进行身份验证，请使用 `--token` 选项：

```bash
>>> hf repos delete-files --token=hf_**** Wauplin/my-cool-model file.txt
```

### hf 仓库分支

使用 `hf repos branch` 在 Hub 上创建和删除存储库的分支。

```bash
# Create a branch
>>> hf repos branch create Wauplin/my-cool-model dev

# Create a branch from a specific revision
>>> hf repos branch create Wauplin/my-cool-model release-1 --revision refs/pr/104

# Delete a branch
>>> hf repos branch delete Wauplin/my-cool-model dev
```

> [!提示]
> 如果您需要显式进行身份验证，所有命令都接受 `--repo-type`（`model`、`dataset`、`space` 之一）和 `--token`。在任何命令上使用 `--help` 即可查看所有选项。

## 高频缓存

使用 `hf cache` 管理本地 Hugging Face 缓存目录。缓存存储从 Hub 下载的模型、数据集和其他文件。

```bash
# List cached repositories
>>> hf cache ls

# List cached revisions
>>> hf cache ls --revisions

# Remove specific items from cache
>>> hf cache rm model/gpt2

# Remove unreferenced revisions
>>> hf cache prune

# Verify cached file checksums
>>> hf cache verify gpt2
```

### hf 缓存 ls使用 `hf cache ls` 检查本地存储在 Hugging Face 缓存中的内容。默认情况下，它按存储库聚合信息：

```bash
>>> hf cache ls
ID                          SIZE     LAST_ACCESSED LAST_MODIFIED REFS
--------------------------- -------- ------------- ------------- -----------
dataset/nyu-mll/glue          157.4M 2 days ago    2 days ago    main script
model/LiquidAI/LFM2-VL-1.6B     3.2G 4 days ago    4 days ago    main
model/microsoft/UserLM-8b      32.1G 4 days ago    4 days ago    main

Found 3 repo(s) for a total of 5 revision(s) and 35.5G on disk.
```

添加 `--revisions` 以深入了解特定快照，并链接过滤器以关注重要事项：

```bash
>>> hf cache ls --filter "size>30g" --revisions
ID                        REVISION                                 SIZE     LAST_MODIFIED REFS
------------------------- ---------------------------------------- -------- ------------- ----
model/microsoft/UserLM-8b be8f2069189bdf443e554c24e488ff3ff6952691    32.1G 4 days ago    main

Found 1 repo(s) for a total of 1 revision(s) and 32.1G on disk.
```

该命令支持多种脚本输出格式：`--format json` 打印结构化对象，`--format csv` 写入逗号分隔的行，`--quiet` 仅打印 ID。使用 `--sort` 按 `accessed`、`modified`、`name` 或 `size` 对条目进行排序（附加 `:asc` 或 `:desc` 来控制顺序），并使用 `--limit` 将结果限制为前 N 个条目。将它们与 `--cache-dir` 结合起来以定位替代缓存位置。请参阅 [Manage your cache](./manage-cache) 高级工作流程指南。

通过将 ID 通过管道传输到 `hf cache rm` 来删除使用 `hf cache ls --q` 选择的缓存条目：

```bash
>>> hf cache rm $(hf cache ls --filter "accessed>1y" -q) -y
About to delete 2 repo(s) totalling 5.31G.
  - model/meta-llama/Llama-3.2-1B-Instruct (entire repo)
  - model/hexgrad/Kokoro-82M (entire repo)
Delete repo: ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct
Delete repo: ~/.cache/huggingface/hub/models--hexgrad--Kokoro-82M
Cache deletion done. Saved 5.31G.
Deleted 2 repo(s) and 2 revision(s); freed 5.31G.
```

### hf 缓存 rm

`hf cache rm` 删除缓存的存储库或单个修订。传递一个或多个存储库 ID (`model/bert-base-uncased`)、存储库级别 `hf://` URI 或修订哈希值：

```bash
>>> hf cache rm model/LiquidAI/LFM2-VL-1.6B
About to delete 1 repo(s) totalling 3.2G.
  - model/LiquidAI/LFM2-VL-1.6B (entire repo)
Proceed with deletion? [y/N]: y
Delete repo: ~/.cache/huggingface/hub/models--LiquidAI--LFM2-VL-1.6B
Cache deletion done. Saved 3.2G.
Deleted 1 repo(s) and 2 revision(s); freed 3.2G.
```

还支持存储库级别 `hf://` URI：

```bash
>>> hf cache rm hf://models/openai-community/gpt2 --dry-run
About to delete 1 repo(s) totalling 1.1G.
  - model/openai-community/gpt2 (entire repo)
Dry run: no files were deleted.
```

在同一个调用中混合存储库和特定修订。使用 `--dry-run` 预览影响，或使用 `--yes` 跳过确认提示 - 在自动化脚本中非常方便：

```bash
>>> hf cache rm model/t5-small 8f3ad1c --dry-run
About to delete 1 repo(s) and 1 revision(s) totalling 1.1G.
  - model/t5-small:
      8f3ad1c [main] 1.1G
Dry run: no files were deleted.
```在默认缓存位置之外工作时，请将命令与 `--cache-dir PATH` 配对。

### hf 缓存修剪

`hf cache prune` 是一个方便的快捷方式，可以回收缓存垃圾占用的空间：每个分离的（未引用的）修订版（仅保留仍然可以通过分支或标签访问的修订版）以及来自中断下载的任何剩余的 `.incomplete` 文件：

```bash
>>> hf cache prune
About to delete 3 unreferenced revision(s) and 2 incomplete download(s) (2.4G total).
  - model/t5-small:
      1c610f6b [refs/pr/1] 820.1M
      d4ec9b72 [(detached)] 640.5M
  - dataset/google/fleurs:
      2b91c8dd [(detached)] 937.6M
Proceed? [y/N]: y
Deleted 3 unreferenced revision(s) and 2 incomplete download(s); freed 2.4G.
```

`.incomplete` 文件是下载中断时留下的部分下载。它们不会被 `hf cache ls`/`rm` （在修订级别工作）跟踪，因此 `hf cache ls` 用提示指出它们，`hf cache prune` 自动删除它们。

与其他缓存命令一样，可以使用 `--dry-run`、`--yes` 和 `--cache-dir`。有关更多示例，请参阅[Manage your cache](./manage-cache)指南。

### hf 缓存验证

使用 `hf cache verify` 根据集线器上的校验和验证本地文件。您可以验证缓存快照或常规本地目录。

示例：

```bash
# Verify main revision of a model in cache
>>> hf cache verify deepseek-ai/DeepSeek-OCR

# Verify a specific revision
>>> hf cache verify deepseek-ai/DeepSeek-OCR --revision refs/pr/5
>>> hf cache verify deepseek-ai/DeepSeek-OCR --revision ef93bf4a377c5d5ed9dca78e0bc4ea50b26fe6a4

# Verify a private repo
>>> hf cache verify me/private-model --token hf_***

# Verify a dataset
>>> hf cache verify karpathy/fineweb-edu-100b-shuffle --repo-type dataset

# Verify files in a local directory
>>> hf cache verify deepseek-ai/DeepSeek-OCR --local-dir /path/to/repo
```

默认情况下，该命令会警告丢失或多余的文件。使用标志将这些警告转变为错误：

```bash
>>> hf cache verify deepseek-ai/DeepSeek-OCR --fail-on-missing-files --fail-on-extra-files
```

成功后，您将看到一个摘要：

```text
✅ Verified 13 file(s) for 'deepseek-ai/DeepSeek-OCR' (model) in ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6
  All checksums match.
```

如果检测到不匹配，该命令将打印详细列表并以非零状态退出。

### hf 存储库标签使用 `hf repos tag` 创建、列出和删除 Hub 上存储库的标签。

```bash
# Create a tag
>>> hf repos tag create my-model v1.0

# List tags
>>> hf repos tag list my-model

# Delete a tag
>>> hf repos tag delete my-model v1.0
```

### 标记模型

要标记存储库，您需要提供 `repo_id` 和 `tag` 名称：

```bash
>>> hf repos tag create Wauplin/my-cool-model v1.0
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### 在特定版本上标记模型

如果您想标记特定修订版，可以使用 `--revision` 选项。默认情况下，标签将在 `main` 分支上创建：

```bash
>>> hf repos tag create Wauplin/my-cool-model v1.0 --revision refs/pr/104
You are about to create tag v1.0 on model Wauplin/my-cool-model
Tag v1.0 created on Wauplin/my-cool-model
```

### 标记数据集或空间

如果要标记数据集或空间，则必须指定 `--repo-type` 选项：

```bash
>>> hf repos tag create bigcode/the-stack v1.0 --repo-type dataset
You are about to create tag v1.0 on dataset bigcode/the-stack
Tag v1.0 created on bigcode/the-stack
```

### 列出标签

要列出存储库的所有标签，请使用 `list` （或 `ls`）子命令：

```bash
>>> hf repos tag list Wauplin/gradio-space-ci --repo-type space
Tags for space Wauplin/gradio-space-ci:
0.2.2
0.2.1
0.2.0
0.1.2
0.0.2
0.0.1
```

### 删除标签

要删除标签，请使用 `delete` 子命令：

```bash
>>> hf repos tag delete Wauplin/my-cool-model v1.0
You are about to delete tag v1.0 on model Wauplin/my-cool-model
Proceed? [Y/n] y
Tag v1.0 deleted on Wauplin/my-cool-model
```

您也可以通过`-y`跳过确认步骤。

## 高频环境

`hf env` 命令打印有关机器设置的详细信息。当您在 [GitHub](https://github.com/huggingface/huggingface_hub) 上提出问题以帮助维护人员调查您的问题时，这非常有用。

```bash
>>> hf env

Copy-and-paste the text below in your GitHub issue.

- huggingface_hub version: 1.0.0.rc6
- Platform: Linux-6.8.0-85-generic-x86_64-with-glibc2.35
- Python version: 3.11.14
- Running in iPython ?: No
- Running in notebook ?: No
- Running in Google Colab ?: No
- Running in Google Colab Enterprise ?: No
- Token path ?: /home/wauplin/.cache/huggingface/token
- Has saved token ?: True
- Who am I ?: Wauplin
- Configured git credential helpers: store
- Installation method: unknown
- Torch: N/A
- httpx: 0.28.1
- hf_xet: 1.1.10
- gradio: 5.41.1
- tensorboard: N/A
- pydantic: 2.11.7
- ENDPOINT: https://huggingface.co
- HF_HUB_CACHE: /home/wauplin/.cache/huggingface/hub
- HF_ASSETS_CACHE: /home/wauplin/.cache/huggingface/assets
- HF_TOKEN_PATH: /home/wauplin/.cache/huggingface/token
- HF_STORED_TOKENS_PATH: /home/wauplin/.cache/huggingface/stored_tokens
- HF_HUB_OFFLINE: False
- HF_HUB_DISABLE_TELEMETRY: False
- HF_HUB_DISABLE_PROGRESS_BARS: None
- HF_HUB_DISABLE_SYMLINKS_WARNING: False
- HF_HUB_DISABLE_EXPERIMENTAL_WARNING: False
- HF_HUB_DISABLE_IMPLICIT_TOKEN: False
- HF_HUB_DISABLE_XET: False
- HF_HUB_ETAG_TIMEOUT: 10
- HF_HUB_DOWNLOAD_TIMEOUT: 10
```

## 高频工作

使用熟悉的类似 Docker 的界面在 Hugging Face 基础设施上运行计算作业。

`hf jobs` 是一个命令行工具，可让您通过简单的命令在 Hugging Face 的基础设施（包括 GPU 和 TPU！）上运行任何内容。想想`docker run`，但在 A100 上运行代码。**有关作业和定价的一般概述，请参阅 [Hub Jobs documentation](https://huggingface.co/docs/hub/jobs)。**有关与 CLI 一起使用 Python API 的信息，请参阅 [Run and manage Jobs guide](./jobs)。

```bash
# Directly run Python code
>>> hf jobs run --name hello-world python:3.12 python -c 'print("Hello from the cloud!")'

# Use GPUs without any setup
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(torch.cuda.get_device_name())"

# Run in an organization account
>>> hf jobs run --namespace my-org-name python:3.12 python -c 'print("Running in an org account")'

# Run from Hugging Face Spaces
>>> hf jobs run hf.co/spaces/lhoestq/duckdb duckdb -c 'select "hello world"'

# Run a Python script with `uv` (experimental)
>>> hf jobs uv run --name my-script my_script.py
```

> [!提示]
> **拥抱脸部作业**可供任何拥有 [pre-paid credits](https://huggingface.co/settings/billing) 的用户或组织使用。

### 快速入门

#### 1. 运行你的第一份工作

```bash
# Run a simple Python script
>>> hf jobs run python:3.12 python -c 'print("Hello from HF compute!")'
```

此命令运行作业并显示日志。您可以通过 `--detach` 在后台运行作业并仅打印作业 ID。

#### 2.检查作业状态

```bash
# List your running jobs
>>> hf jobs ls
# List all jobs
>>> hf jobs ls -a

# Inspect the status of a job
>>> hf jobs inspect <job_id>

# View logs from a job
>>> hf jobs logs <job_id>

# View resources usage stats and metrics of running jobs
>>> hf jobs stats
# View resources usage stats and metrics of some jobs
>>> hf jobs stats [job_ids]...

# Cancel a job
>>> hf jobs cancel <job_id>

# Wait until one or more jobs finish (exit code 0 only if all jobs completed successfully)
>>> hf jobs wait <job_id> [<job_id>...]

# Wait for all currently running jobs
>>> hf jobs ls -q | xargs hf jobs wait
```

如果作业失败，非分离的 `hf jobs run` 和 `hf jobs wait` 会以非零代码退出，因此您可以使用 `&&` 链接命令：

```bash
>>> hf jobs run python:3.12 python train.py && echo "training succeeded"
```

#### 3. 在 GPU 上运行

您还可以使用 `--flavor` 选项在 GPU 或 TPU 上运行作业。例如，要在 A10G GPU 上运行 PyTorch 作业：

```bash
# Use an A10G GPU to check PyTorch CUDA
>>> hf jobs run --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c 'import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")'
```

运行此命令将显示以下输出！

```bash
This code ran with the following GPU: NVIDIA A10G
```

为了清晰起见，`--` 可用于将命令与作业选项分开，例如 `hf jobs run --flavor a10g-small -- python -c '...'`

就是这样！您现在正在 Hugging Face 的基础设施上运行代码。

### 常见用例- **模型训练**：在 GPU（T4、A10G、A100）上微调或训练模型，无需管理基础设施
- **合成数据生成**：在强大的硬件上使用法学硕士生成大规模数据集
- **数据处理**：使用高 CPU 配置处理海量数据集以实现并行工作负载
- **批量推理**：使用优化的 GPU 设置对数千个样本运行离线推理
- **实验和基准**：在一致的硬件上运行机器学习实验以获得可重现的结果
- **开发和调试**：无需本地 CUDA 设置即可测试 GPU 代码

### 传递环境变量和 Secret

您可以使用以下命令将环境变量传递给您的作业

```bash
# Pass environment variables
>>> hf jobs run -e FOO=foo -e BAR=bar python:3.12 python -c 'import os; print(os.environ["FOO"], os.environ["BAR"])'
```

```bash
# Pass an environment from a local .env file
>>> hf jobs run --env-file .env python:3.12 python -c 'import os; print(os.environ["FOO"], os.environ["BAR"])'
```

```bash
# Pass secrets - they will be encrypted server side
>>> hf jobs run -s MY_SECRET=psswrd python:3.12 python -c 'import os; print(os.environ["MY_SECRET"])'
```

```bash
# Pass secrets from a local .env.secrets file - they will be encrypted server side
>>> hf jobs run --secrets-file .env.secrets python:3.12 python -c 'import os; print(os.environ["MY_SECRET"])'
```

> [!提示]
> 使用 `--secrets HF_TOKEN` 隐式传递您本地的 Hugging Face 令牌。
> 使用此语法，可以从环境变量中检索机密。
> 对于`HF_TOKEN`，如果未设置环境变量，它可能会读取位于 Hugging Face 主文件夹中的令牌文件。

#### 内置环境变量

在作业容器内，以下环境变量自动可用：|变量|描述 |
| ------------- | ----------------------------------------------------------------------------------------------------------- |
| `JOB_ID` |当前作业的唯一标识符。使用它以编程方式引用作业。               |
| `ACCELERATOR` |可用加速器的类型（例如，`t4-medium`、`a10g-small`、`a100x4`）。如果没有加速器则为空。 |
| `CPU_CORES` |作业可用的 CPU 核心数量（例如，`2`、`4`、`8`）。                                     |
| `MEMORY` |作业可用的内存量（例如，`16Gi`、`32Gi`）。                                       |

```bash
# Access job environment information
>>> hf jobs run python:3.12 python -c "import os; print(f'Job: {os.environ.get(\"JOB_ID\")}, CPU: {os.environ.get(\"CPU_CORES\")}, Mem: {os.environ.get(\"MEMORY\")}')"
```

### 作业超时

作业的默认超时时间为 30 分钟，之后会自动停止。对于模型训练等长时间运行的任务，请使用 `--timeout` 选项设置自定义超时：

```bash
# Set timeout in seconds (default unit)
>>> hf jobs run --timeout 7200 python:3.12 python train.py

# Use time units: s (seconds), m (minutes), h (hours), d (days)
>>> hf jobs run --timeout 2h pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel python train.py
>>> hf jobs run --timeout 90m python:3.12 python process_data.py
>>> hf jobs run --timeout 1.5h python:3.12 python train.py  # floats are supported
```

`--timeout` 选项还适用于 UV 脚本和计划作业：

```bash
# UV script with timeout
>>> hf jobs uv run --timeout 2h training_script.py

# Scheduled job with timeout
>>> hf jobs scheduled run @daily --timeout 4h python:3.12 python daily_task.py
```

> [!警告]
> 如果您的作业超过超时时间，它将自动终止。始终为长时间运行的任务设置适当的超时和一些缓冲区，以避免作业意外终止。### 硬件

可用的`--flavor`选项：

- CPU：`cpu-basic`、`cpu-upgrade`
- GPU：`t4-small`、`t4-medium`、`l4x1`、`l4x4`、`a10g-small`、`a10g-large`、`a10g-largex2`、`a10g-largex4`、`a100-large`
- TPU：`v5e-1x1`，`v5e-2x2`，`v5e-2x4`

(07/2025更新于拥抱脸[suggested_hardware docs](https://huggingface.co/docs/hub/en/spaces-config-reference))

## 卷

使用 `-v` 或 `--volume` 在作业磁盘上安装卷。

您可以使用 `hf://` URL 方案挂载任何 Hugging Face 存储库（模型/数据集/空间）或 [Storage Bucket](/docs/hub/storage-buckets)。例如：

* 挂载模型存储库：`-v hf://openai/gpt-oss-120b:/model`
* 挂载数据集存储库：`-v hf://datasets/HuggingFaceFW/fineweb:/data`
* 安装储物桶：`-v hf://buckets/username/my-bucket:/mnt`
*挂载空间：`-v hf://spaces/username/my-space:/app`
* 在存储库中安装子文件夹：`-v hf://datasets/org/ds/train:/data`

然后你可以使用挂载的卷作为本地目录：

```bash
# Docker Job with a mounted volume as input
>>> hf jobs run -v hf://datasets/HuggingFaceFW/fineweb:/dataset \
...     duckdb/duckdb duckdb -c "SELECT * FROM '/dataset/**/*.parquet' LIMIT 5"

# UV Job with a mounted volume to save checkpoints when training a model
>>> hf jobs uv run -v hf://buckets/username/my-bucket:/training-outputs \
...     sft.py --output-dir /training-outputs/training-v3-final ...
```

模型、数据集和空间始终以只读方式安装。存储桶默认是读+写的——这对于经常更改的数据特别有用，因为文件可以就地覆盖或删除。

使用 `:ro` 启用只读：

* 以只读方式挂载存储桶：`-v hf://buckets/username/my-bucket:/mnt:ro`

### 挂载本地数据

`-v`的源端也可以是本地目录。它首先同步到您的 `jobs-artifacts` [Storage Bucket](/docs/hub/storage-buckets)，并将生成的存储桶文件夹安装在作业中：

```bash
>>> hf jobs uv run -v ./my-data:/data process.py
```重新运行该命令仅上传新的或修改的文件。默认情况下，本地目录以只读方式安装。使用 `:rw` 让作业写入卷，例如在作业完成后检索输出（空的本地目录也可以）：

```bash
>>> hf jobs uv run -v ./pdfs:/input -v ./md-out:/output:rw ocr.py
...
Hint: Volume '/output' is mounted read-write. Once the job is over, pull back its data with:
  hf buckets sync hf://buckets/username/jobs-artifacts/md-out-a1b2c3d4 ./md-out
```

### 标签

使用 `-l` 或 `--label` 将标签添加到作业。标签是将元数据应用于作业的键=值对。要使用两个标签来标记作业，请重复标签标志（`-l` 或 `--label`）：

```bash
>>> hf jobs run -l my-label --label foo=bar ubuntu echo "This Job has multiple labels"
```

my-label 键未指定值，因此其值默认为空字符串 ("")。

创建Job时使用`--name`添加`name`标签。名称使职位更容易在 UI 中查找和识别；它们是可选的并且不必是唯一的。如果您没有通过 `--name`，则会从 Docker 映像或脚本自动派生一个名称，加上命令的简短哈希值，因此同一命令的重新运行会共享一个名称（例如 `python:3.12 foo --truc` → `python-3-12-1a2b3c4d`）。您还可以重命名现有作业：

```bash
>>> hf jobs run --name training-v2 python:3.12 python train.py
>>> hf jobs labels <job_id> --name training-v2
```

使用`hf jobs ls`中的`--status`、`--label`和`--name`来过滤职位。 `--status` 接受一个或多个状态，`--label` 接受`key=value` 对，`--name` 是`--label name=NAME` 的快捷方式。作业必须匹配要列出的每个过滤器。作业名称也显示为其自己的 `NAME` 列：```bash
# Show completed Jobs
>>> hf jobs ls -a --status completed

# Show running or scheduling Jobs
>>> hf jobs ls --status running,scheduling

# Show Jobs named `training-v2`
>>> hf jobs ls -a --name training-v2

# Show Jobs with the `model=Qwen3-06B` label
>>> hf jobs ls -a --label model=Qwen3-06B

# Combine filters: running Jobs labelled both `env=prod` and `team=ml`
>>> hf jobs ls --status running --label env=prod --label team=ml
```

默认情况下，`hf jobs ps`最多显示 100 个作业，以避免终端臃肿。使用 `--limit` 更改此设置，或使用 `--limit 0` 显示所有内容：

```bash
# Show up to 500 Jobs
>>> hf jobs ps -a --limit 500

# Show all Jobs (no limit)
>>> hf jobs ps -a --limit 0
```

> [!警告]
> `-f`/`--filter` 已弃用，取而代之的是 `--status` 和 `--label`。精确匹配：不支持 glob 模式 (`data-*`) 和否定 (`key!=value`)，并且无法按 `id`、`image` 或 `command` 进行过滤。

### 通过 SSH 进入作业

将 `--ssh` 传递给 `hf jobs run`（或 `hf jobs uv run`）以使作业的容器可通过 SSH 访问，然后与 `hf jobs ssh` 连接：

```bash
# Start a job with SSH enabled
>>> hf jobs run --ssh --detach python:3.12 sleep infinity

# Open an SSH session into it
>>> hf jobs ssh <job_id>

# Print the SSH command instead of running it
>>> hf jobs ssh <job_id> --dry-run

# Use a specific identity file
>>> hf jobs ssh <job_id> -i ~/.ssh/id_ed25519
```

仅允许对作业命名空间具有写入权限的用户（作业创建者或所有者组织的成员），并通过在 https://huggingface.co/settings/keys 注册的 SSH 公钥进行身份验证。

### UV 脚本（实验）

在 HF 基础设施上运行 UV 脚本（具有内联依赖项的 Python 脚本）。 UV 脚本是 Python 脚本，使用特殊的注释语法将其依赖项直接包含在文件中。

```bash
# Run a UV script (creates temporary repo)
>>> hf jobs uv run my_script.py

# Run with persistent repo
>>> hf jobs uv run --repo my-uv-scripts my_script.py

# Run with GPU
>>> hf jobs uv run --flavor gpu-t4-small ml_training.py

# Pass arguments to script
>>> hf jobs uv run process.py input.csv output.parquet

# Add dependencies
>>> hf jobs uv run --with transformers --with torch train.py

# Run a script directly from a URL
>>> hf jobs uv run https://huggingface.co/datasets/username/scripts/resolve/main/example.py

# Run a command
>>> hf jobs uv run --with lighteval python -c 'import lighteval'
```UV 脚本是 Python 脚本，使用特殊的注释语法将其依赖项直接包含在文件中。这使得它们非常适合不需要复杂项目设置的独立任务。在 [UV documentation](https://docs.astral.sh/uv/guides/scripts/) 中了解有关 UV 脚本的更多信息。

为了清晰起见，`--`可用于将命令与作业/uv选项分开，例如`hf jobs uv run --flavor gpu-t4-small --with torch -- python -c '...'`

### hf 已安排工作

安排和管理将在 HF 基础设施上运行的作业。

时间表应为 `@annually`、`@yearly`、`@monthly`、`@weekly`、`@daily`、`@hourly` 或 CRON 时间表表达式之一（例如，`"0 9 * * 1"` 每周一上午 9 点）。

```bash
# Schedule a job that runs every hour
>>> hf jobs scheduled run @hourly --name hourly-task python:3.12 python -c 'print("This runs every hour!")'

# Use the CRON syntax
>>> hf jobs scheduled run "*/5 * * * *" python:3.12 python -c 'print("This runs every 5 minutes!")'

# Schedule with GPU
>>> hf jobs scheduled run @hourly --flavor a10g-small pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel \
... python -c "import torch; print(f"This code ran with the following GPU: {torch.cuda.get_device_name()}")"

# Schedule a UV script
>>> hf jobs scheduled uv run @hourly --name hourly-script my_script.py
```

使用与`hf jobs run`相同的参数来传递环境变量、秘密、超时等。

使用管理计划作业

```bash
# List your active scheduled jobs
>>> hf jobs scheduled ls

# Inspect the status of a job
>>> hf jobs scheduled inspect <scheduled_job_id>

# Suspend (pause) a scheduled job
>>> hf jobs scheduled suspend <scheduled_job_id>

# Resume a scheduled job
>>> hf jobs scheduled resume <scheduled_job_id>

# Trigger a scheduled job to run right now (does not change the schedule)
>>> hf jobs scheduled trigger <scheduled_job_id>

# Delete a scheduled job
>>> hf jobs scheduled delete <scheduled_job_id>
```

## 高频沙箱

`hf sandbox` 启动基于作业构建的隔离云计算机：创建一台、使用实时流输出运行命令以及将文件复制进出。任何带有 `/bin/sh` 的 Docker 镜像都可以。请参阅 [Sandboxes guide](./sandbox) 了解 Python API，以及 [conceptual guide](../concepts/sandbox) 了解其底层工作原理。

```bash
# Create a sandbox (waits until it is ready, prints its id)
>>> hf sandbox create
✓ Sandbox ready id=687f911eaea852de79c4a50a image=python:3.12 elapsed=6.0s

# Run commands inside it (output is streamed, exit code is propagated)
>>> hf sandbox exec 687f911eaea852de79c4a50a -- python -c "print('hi')"
hi

# Copy files in and out (docker-style)
>>> hf sandbox cp data.csv 687f911eaea852de79c4a50a:/data/data.csv
>>> hf sandbox cp 687f911eaea852de79c4a50a:/app/results.json results.json

# Terminate a sandbox
>>> hf sandbox kill 687f911eaea852de79c4a50a
```使用 `--flavor` 选择硬件（例如 `a10g-small`），使用 `--idle-timeout` 限制沙箱生命周期，使用 `-e` / `--secrets` 作为环境变量。要扇出许多廉价的 CPU 沙箱，请使用 `hf sandbox pool create` 预热池并使用 `hf sandbox create --pool <id>` 生成到其中（请参阅 [Sandboxes guide](./sandbox#from-the-cli)）。

## 高频网络钩子

`hf webhooks` 可让您直接从终端管理 Hugging Face Hub 上的 Webhook。 Webhooks 允许您监听存储库、用户或组织上的事件（推送、讨论等）并触发操作 - 通过 ping 远程 URL 或在 Hugging Face 基础设施上运行作业。

### 列出网络钩子

```bash
>>> hf webhooks ls
ID           URL                                DISABLED  DOMAINS  WATCHED
wh-abc123    https://example.com/hook            False     repo     model:bert-base-uncased
wh-def456    https://example.com/other-hook      False     repo     org:HuggingFace
```

使用 `--format json` 进行机器可读输出，或使用 `-q` 仅打印 ID：

```bash
>>> hf webhooks ls --format json
>>> hf webhooks ls -q
```

### 获取 webhook 信息

```bash
>>> hf webhooks info wh-abc123
```

将完整的 Webhook 详细信息打印为 JSON。

### 创建一个网络钩子

创建一个将有效负载发送到 URL 的 Webhook：

```bash
>>> hf webhooks create --url https://example.com/hook --watch model:bert-base-uncased
>>> hf webhooks create --url https://example.com/hook --watch org:HuggingFace --watch model:gpt2 --domain repo
```

或者创建一个触发作业的 Webhook：

```bash
>>> hf webhooks create --job-id 687f911eaea852de79c4a50a --watch user:julien-c
```

`--watch` 选项使用格式 `type:name`，其中 type 为 `model`、`dataset`、`space`、`org` 或 `user` 之一。可以重复观看多个项目。使用`--domain`过滤事件到`repo`或`discussions`，并使用`--secret`设置签名秘密。

### 更新网络钩子

```bash
>>> hf webhooks update wh-abc123 --url https://new-url.com/hook
>>> hf webhooks update wh-abc123 --watch model:gpt2 --domain repo
```仅更改提供的选项。请注意，指定时，`--watch` 会替换整个观看列表。

### 启用/禁用网络钩子

```bash
>>> hf webhooks enable wh-abc123
>>> hf webhooks disable wh-abc123
```

### 删除网络钩子

```bash
>>> hf webhooks delete wh-abc123
Are you sure you want to delete webhook 'wh-abc123'? [y/N]: y
Webhook deleted: wh-abc123
```

使用`--yes`跳过确认提示：

```bash
>>> hf webhooks delete wh-abc123 --yes
```

> [!提示]
> 所有命令都接受 `--token` 来覆盖身份验证。在任何命令上使用 `--help` 即可查看所有选项。

## 高频端点

使用`hf endpoints`直接从终端列出、部署、描述和管理推理端点。遗产
`hf inference-endpoints` 别名仍可用于兼容性。

```bash
# Lists endpoints in your namespace
>>> hf endpoints ls

# Deploy an endpoint from Model Catalog
>>> hf endpoints catalog deploy --repo openai/gpt-oss-120b --name my-endpoint

# Deploy an endpoint from the Hugging Face Hub
>>> hf endpoints deploy my-endpoint --repo gpt2 --framework pytorch --accelerator cpu --instance-size x2 --instance-type intel-icl

# List catalog entries
>>> hf endpoints catalog ls

# Show status and metadata
>>> hf endpoints describe my-endpoint

# Pause the endpoint
>>> hf endpoints pause my-endpoint

# Delete without confirmation prompt
>>> hf endpoints delete my-endpoint --yes
```

> [!提示]
> 添加 `--namespace` 以定位组织，添加 `--token` 以覆盖身份验证。

#### 部署自定义容器

要部署您自己的 Docker 映像而不是 Hugging Face 托管映像，请将 `--framework custom` 与 `--custom-image` 一起传递。模型存储库安装在容器内的`/repository`处。使用`--container-args`（以及可选的`--container-command`）传递带引号的启动字符串，使用`--env`/`--secrets`注入环境变量，使用`--type`设置访问类型（`public`、`authenticated`或`private`）：

```bash
>>> hf endpoints deploy nex-n2-pro \
      --repo nex-agi/Nex-N2-Pro \
      --framework custom \
      --accelerator gpu --vendor aws --region us-east-1 \
      --instance-type nvidia-h200 --instance-size x8 \
      --custom-image nexagi/sglang:v0.5.12 \
      --health-route /health --port 30000 \
      --container-args "--reasoning-parser qwen3 --tool-call-parser qwen3_coder --mamba-scheduler-strategy extra_buffer --tp 8" \
      --env MODEL_ID=/repository \
      --type authenticated
````--container-args` 和 `--container-command` 不限于自定义映像：它们映射到 API 负载中的 `model.args` 和 `model.command`，这也适用于托管映像。在现有端点上，这两个标志都会替换当前值而不是添加到当前值，因此请传递所需的完整列表并使用空字符串来清除它。从目录部署的端点已经带有调整后的引擎标志，因此在覆盖它们之前检查`hf endpoints describe`：

```bash
>>> hf endpoints update my-endpoint \
      --container-args "--enable-auto-tool-choice --tool-call-parser lfm2"
```

### 高频端点目录

使用 `hf endpoints catalog` 与推理端点模型目录交互。使用优化配置直接从目录部署模型。

```bash
# List available catalog models
>>> hf endpoints catalog ls

# Deploy a model from the catalog
>>> hf endpoints catalog deploy --repo meta-llama/Llama-3.2-1B-Instruct

# Deploy with a custom name
>>> hf endpoints catalog deploy --repo meta-llama/Llama-3.2-1B-Instruct --name my-llama-endpoint
```