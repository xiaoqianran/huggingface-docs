<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 警告
# 整个文件是通过 `hf` CLI 实现生成的。
# 要重新生成代码，请运行`make style`或`python ./utils/generate_cli_reference.py --update`。
# 警告
-->

#`hf`

拥抱 Face Hub CLI

**用法**：

```console
$ hf [OPTIONS] [COMMAND] [ARGS]...
```

**选项**：

* `--install-completion`：当前shell的安装完成。
* `--show-completion`：显示当前 shell 的完成情况，以复制它或自定义安装。
* `--help`：显示此消息并退出。

**命令**：

* `auth`：管理身份验证（登录、注销等）。
* `buckets`：与存储桶交互的命令。
* `cache`：管理本地缓存目录。
* `collections`：与 Hub 上的收藏互动。
* `cp`：在本地路径之间复制文件，...
* `datasets`：与 Hub 上的数据集交互。
* `discussions`：管理讨论并拉取请求...
* `download`：从集线器下载文件。
* `endpoints`：管理拥抱脸部推理端点。
* `env`：打印有关环境的信息。
* `extensions`：管理 hf CLI 扩展。 [别名：分机]
* `jobs`：在 Hub 上运行和管理作业。
* `lfs-enable-largefiles`：配置您的存储库以启用上传...
* `lfs-multipart-upload`：内部 git-lfs 自定义传输代理...
* `models`：与 Hub 上的模型交互。
* `papers`：与 Hub 上的论文互动。* `repos`：管理 Hub 上的存储库。 [别名：回购]
* `sandbox`：在 Hugging Face 上运行和管理沙箱...
* `skills`：管理AI助手的技能。
* `spaces`：与 Hub 上的空间互动。
* `sync`：在本地目录和...之间同步文件
* `update`：将`hf` CLI 更新到最新版本。
* `upload`：上传文件或文件夹到集线器。
* `upload-large-folder`: [已弃用] 将大文件夹上传到...
* `version`：打印有关hf版本的信息。
* `webhooks`：管理 Hub 上的 webhook。

## `hf auth`

管理身份验证（登录、注销等）。

**用法**：

```console
$ hf auth [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `list`：列出所有存储的访问令牌。 [别名：ls]
* `login`：从浏览器登录，或使用令牌...
* `logout`：从特定令牌注销。
* `switch`：在访问令牌之间切换。
* `token`：将当前访问令牌打印到标准输出。
* `whoami`：找出您所在的huggingface.co 帐户...

### `hf auth list`

列出所有存储的访问令牌。 [别名：ls]

**用法**：

```console
$ hf auth list [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 授权列表了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf auth login`

从浏览器登录，或使用 Huggingface.co/settings/tokens 中的令牌登录。

**用法**：

```console
$ hf auth login [OPTIONS]
```

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--add-to-git-credential / --no-add-to-git-credential`：保存到 git 凭证助手。仅当您打算直接运行 git 命令时才有用。  [默认值：不添加到 git-credential]
* `--force / --no-force`: 即使已经登录也强制重新登录。 [默认: no-force]
* `--help`：显示此消息并退出。

示例
  $ hf 身份验证登录
  $ hf 身份验证登录 --token $HF_TOKEN
  $ hf auth login --token $HF_TOKEN --add-to-git-credential
  $ hf 身份验证登录 --force

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf auth logout`

从特定令牌注销。

**用法**：

```console
$ hf auth logout [OPTIONS]
```

**选项**：

* `--token-name TEXT`：要注销的令牌名称
* `--help`：显示此消息并退出。

示例
  $ hf 身份验证注销
  $ hf auth logout --token-name my-token了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf auth switch`

在访问令牌之间切换。

**用法**：

```console
$ hf auth switch [OPTIONS]
```

**选项**：

* `--token-name TEXT`：要切换到的代币名称
* `--add-to-git-credential / --no-add-to-git-credential`：保存到 git 凭证助手。仅当您打算直接运行 git 命令时才有用。  [默认值：不添加到 git-credential]
* `--help`：显示此消息并退出。

示例
  $ hf 身份验证开关
  $ hf auth switch --token-name my-token

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf auth token`

将当前访问令牌打印到标准输出。

**用法**：

```console
$ hf auth token [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 身份验证令牌
  $ hf 身份验证令牌 | xargs curl -H '授权：持有者{}'

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf auth whoami`

找出您登录的huggingface.co 帐户。

**用法**：

```console
$ hf auth whoami [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。示例
  $ hf auth whoami
  $ hf auth whoami --format json

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf buckets`

与存储桶交互的命令。

**用法**：

```console
$ hf buckets [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `cp`：在本地路径之间复制文件，...
* `create`：创建一个新的桶。
* `delete`：删除桶。
* `info`：获取有关存储桶的信息。
* `list`：列出存储桶或存储桶中的文件。 [别名：ls]
* `move`：将存储桶移动（重命名）为新名称或...
* `remove`：从存储桶中删除文件。 [别名：rm]
* `settings`：更新存储桶设置（可见性）。
* `sync`：在本地目录和...之间同步文件

### `hf buckets cp`

在本地路径、存储库和存储桶之间复制文件。处理上传（本地/stdin -> repo/bucket）、下载（repo/bucket -> local/stdout）和
远程到远程副本（repo/bucket -> repo/bucket）。存储桶到存储库和本地到本地
不支持副本。对于目录，使用 `hf upload`/`hf download` (repos) 或
`hf buckets sync`（桶）。远程到远程复制仅在同一存储内工作
区域（https://huggingface.co/docs/hub/storage-regions）。

**用法**：

```console
$ hf buckets cp [OPTIONS] SRC [DST]
```

**参数**：

* `SRC`：来源：本地文件、hf:// URI（存储库或存储桶）或 - 用于标准输入。  [必填]
* `[DST]`：目标：本地路径、hf:// URI（存储库或存储桶）或 - 用于标准输出。

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 存储桶 cp hf://buckets/username/my-bucket/config.json config.json
  $ hf 存储桶 cp hf://buckets/username/my-bucket/data.csv data/
  $ hf 存储桶 cp hf://buckets/username/my-bucket/config.json -
  $ hf 桶 cp model.safetensors hf://buckets/username/my-bucket/model.safetensors
  $ hf 存储桶 cp config.json hf://buckets/username/my-bucket/logs/
  $ hf 存储桶 cp - hf://buckets/username/my-bucket/config.json
  $ hf 存储桶 cp hf://buckets/username/my-bucket/data.csv hf://buckets/username/dest-bucket/
  $ hf 存储桶 cp hf://buckets/username/source-bucket/logs/ hf://buckets/username/dest-bucket/logs/

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets create`

创建一个新存储桶。

**用法**：

```console
$ hf buckets create [OPTIONS] BUCKET_ID
```

**参数**：

* `BUCKET_ID`：存储桶 ID：bucket_name、namespace/bucket_name 或 hf://buckets/namespace/bucket_name [必填]

**选项**：* `--private`：创建私有存储桶。
* `--region [us|eu]`：创建存储桶的云区域。可以是“我们”或“欧盟”之一。需要团队计划或以上。
* `--exist-ok`：如果存储桶已经存在，则不要引发错误。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 桶创建 my-bucket
  $ hf 存储桶创建用户/我的存储桶
  $ hf 存储桶创建 hf://buckets/user/my-bucket
  $ hf 存储桶创建用户/my-bucket --private
  $ hf 存储桶创建用户/my-bucket --exist-ok
  $ hf Bucket 创建用户/my-bucket --region us

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets delete`

删除一个桶。

这将删除整个存储桶及其所有内容。使用 `hf buckets rm` 删除单个文件。

**用法**：

```console
$ hf buckets delete [OPTIONS] BUCKET_ID
```

**参数**：

* `BUCKET_ID`：存储桶 ID：namespace/bucket_name 或 hf://buckets/namespace/bucket_name [必填]

**选项**：* `-y, --yes`：跳过确认提示。
* `--missing-ok`：如果存储桶不存在，则不引发错误。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 存储桶删除用户/我的存储桶
  $ hf 存储桶删除 hf://buckets/user/my-bucket
  $ hf 存储桶删除用户/我的存储桶 --yes
  $ hf Bucket 删除用户/my-bucket --missing-ok

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets info`

获取有关存储桶的信息。

**用法**：

```console
$ hf buckets info [OPTIONS] BUCKET_ID
```

**参数**：

* `BUCKET_ID`: 存储桶 ID：namespace/bucket_name 或 hf://buckets/namespace/bucket_name [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 存储桶信息用户/我的存储桶
  $ hf 存储桶信息 hf://buckets/user/my-bucket

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets list`

列出存储桶或存储桶中的文件。 [别名：ls]当不带参数或名称空间调用时，列出存储桶。
当使用存储桶 ID (namespace/bucket_name) 调用时，列出存储桶中的文件。

**用法**：

```console
$ hf buckets list [OPTIONS] [ARGUMENT]
```

**参数**：

* `[ARGUMENT]`：用于列出存储桶的命名空间（用户或组织），或用于列出文件的存储桶 ID（命名空间/bucket_name(/prefix) 或 hf://buckets/...）。

**选项**：

* `-h, --human-readable`：以人类可读的格式显示尺寸。
* `--tree`：以树形格式列出文件（仅用于列出文件）。
* `-R, --recursive`：递归列出文件（仅用于列出文件）。
* `--search TEXT`：搜索查询。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 存储桶列表
  $ hf 存储桶列表 Huggingface
  $ hf 存储桶列表 --搜索“我的前缀”
  $ hf 存储桶列表用户/我的存储桶
  $ hf 存储桶列表用户/我的存储桶 -R
  $ hf 存储桶列表用户/我的存储桶 -h
  $ hf 存储桶列表用户/我的存储桶 --tree
  $ hf 存储桶列表用户/我的存储桶 --tree -h
  $ hf 存储桶列表 hf://buckets/user/my-bucket
  $ hf 存储桶列表 user/my-bucket/sub -R

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets move`将存储桶移动（重命名）到新名称或命名空间。

**用法**：

```console
$ hf buckets move [OPTIONS] FROM_ID TO_ID
```

**参数**：

* `FROM_ID`：源存储桶 ID：namespace/bucket_name 或 hf://buckets/namespace/bucket_name [必填]
* `TO_ID`：目标存储桶 ID：namespace/bucket_name 或 hf://buckets/namespace/bucket_name [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 存储桶移动用户/旧存储桶用户/新存储桶
  $ hf 存储桶移动 user/my-bucket my-org/my-bucket
  $ hf 存储桶移动 hf://buckets/user/old-bucket hf://buckets/user/new-bucket

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets remove`

从存储桶中删除文件。 [别名：rm]

要删除整个存储桶，请改用 `hf buckets delete`。

**用法**：

```console
$ hf buckets remove [OPTIONS] ARGUMENT
```

**参数**：

* `ARGUMENT`：存储桶路径：namespace/bucket_name/path 或 hf://buckets/namespace/bucket_name/path。使用 --recursive，namespace/bucket_name 也被接受以定位所有文件。  [必填]

**选项**：* `-R, --recursive`：递归删除给定前缀下的文件。
* `-y, --yes`：跳过确认提示。
* `--dry-run`：预览将要删除的内容，而不实际删除。
* `--include TEXT`：仅包含匹配模式的文件（可以指定多个）。需要--递归。
* `--exclude TEXT`：排除匹配模式的文件（可以指定多个）。需要--递归。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 存储桶删除 user/my-bucket/file.txt
  $ hf 桶 rm hf://buckets/user/my-bucket/file.txt
  $ hf Bucket rm user/my-bucket/logs/ --recursive
  $ hf Bucket rm user/my-bucket --recursive --include "*.tmp"
  $ hf Bucket rm user/my-bucket/data/ --recursive --dry-run

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets settings`

更新存储桶设置（可见性）。

**用法**：

```console
$ hf buckets settings [OPTIONS] BUCKET_ID
```

**参数**：

* `BUCKET_ID`: 存储桶 ID：namespace/bucket_name 或 hf://buckets/namespace/bucket_name [必填]

**选项**：* `--private`：将存储桶设为私有。
* `--public`：公开存储桶。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 存储桶设置 user/my-bucket --private
  $ hf 存储桶设置 user/my-bucket --public
  $ hf 存储桶设置 hf://buckets/user/my-bucket --private

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf buckets sync`

在本地目录和存储桶之间同步文件。

**用法**：

```console
$ hf buckets sync [OPTIONS] [SOURCE] [DEST]
```

**参数**：

* `[SOURCE]`: 源路径：本地目录或 hf://buckets/namespace/bucket_name(/prefix)
* `[DEST]`: 目标路径：本地目录或 hf://buckets/namespace/bucket_name(/prefix)

**选项**：* `--delete / --no-delete`：删除源中不存在的目标文件。  [默认：不删除]
* `--ignore-times`：仅根据大小跳过文件，忽略修改时间。
* `--ignore-sizes`：仅根据修改时间跳过文件，忽略大小。
* `--plan TEXT`：将同步计划保存到 JSONL 文件以供审核而不是执行。
* `--apply TEXT`：应用之前保存的计划文件。
* `--dry-run`：将同步计划作为 JSONL 打印到标准输出而不执行。
* `--include TEXT`：包含匹配模式的文件（可以指定多个）。
* `--exclude TEXT`：排除匹配模式的文件（可指定多个）。
* `--filter-from TEXT`：从文件中读取包含/排除模式。
* `--existing`：跳过在接收器上创建新文件（仅更新现有文件）。
* `--ignore-existing`：跳过更新接收器上存在的文件（仅创建新文件）。
* `-v, --verbose`：显示带有推理的详细日志记录。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 存储桶同步 ./data hf://buckets/user/my-bucket
  $ hf 存储桶同步 hf://buckets/user/my-bucket ./data
  $ hf 存储桶同步 ./data hf://buckets/user/my-bucket --delete
  $ hf 存储桶同步 hf://buckets/user/my-bucket ./data --include "*.safetensors" --exclude "*.tmp"
  $ hf 存储桶同步 ./data hf://buckets/user/my-bucket --plansync-plan.jsonl
  $ hf 存储桶同步 --applysync-plan.jsonl
  $ hf 存储桶同步 ./data hf://buckets/user/my-bucket --dry-run
  $ hf 存储桶同步 ./data hf://buckets/user/my-bucket --dry-run | jq .

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf cache`

管理本地缓存目录。

**用法**：

```console
$ hf cache [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `list`：列出缓存的存储库或修订版本。 [别名：ls]
* `prune`：删除独立的修订和不完整的...
* `rm`：删除缓存的存储库或修订版本。
* `verify`：验证单个存储库的校验和...

### `hf cache list`

列出缓存的存储库或修订版本。 [别名：ls]

**用法**：

```console
$ hf cache list [OPTIONS]
```

**选项**：* `--cache-dir TEXT`：要扫描的缓存目录（默认为Hugging Face缓存）。
* `--revisions / --no-revisions`：在输出中包含修订，而不是聚合存储库。  [默认值：无修改]
* `-f, --filter TEXT`：过滤条目（例如“大小> 1GB”、“类型=型号”、“访问> 7d”）。可以多次使用。
* `--sort [accessed|accessed:asc|accessed:desc|modified|modified:asc|modified:desc|name|name:asc|name:desc|size|size:asc|size:desc]`：按键对条目进行排序。支持的键：“访问”、“修改”、“名称”、“大小”。附加 ':asc' 或 ':desc' 以显式设置顺序（例如，'modified:asc'）。默认值：“已访问”、“已修改”、“大小”默认为“desc”（最新/最大的在前）； “name”默认为“asc”（按字母顺序）。
* `--limit INTEGER`：限制返回结果的数量。排序后仅返回前 N 个条目。
* `--show-warnings / --no-show-warnings`：显示有关缓存不一致的警告。  [默认值：未显示警告]
* `--help`：显示此消息并退出。

示例
  $ hf 缓存 ls
  $ hf 缓存 ls --revisions
  $ hf 缓存 ls --filter "大小>1GB" --limit 20
  $ hf 缓存 ls --format json

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf cache prune`

从缓存中删除分离的修订和不完整的下载。

**用法**：

```console
$ hf cache prune [OPTIONS]
```

**选项**：* `--cache-dir TEXT`：要扫描的缓存目录（默认为Hugging Face缓存）。
* `-y, --yes`：跳过确认提示。
* `--dry-run / --no-dry-run`：预览删除内容而不删除任何内容。  [默认值：不试运行]
* `--help`：显示此消息并退出。

示例
  $ hf 缓存修剪
  $ hf 缓存修剪 --dry-run

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf cache rm`

删除缓存的存储库或修订版本。

**用法**：

```console
$ hf cache rm [OPTIONS] TARGETS...
```

**参数**：

* `TARGETS...`：一个或多个存储库 ID（例如 model/bert-base-uncased）、存储库级别 hf:// URI 或要删除的修订哈希值。  [必填]

**选项**：

* `--cache-dir TEXT`：要扫描的缓存目录（默认为Hugging Face缓存）。
* `-y, --yes`：跳过确认提示。
* `--dry-run / --no-dry-run`：预览删除内容而不删除任何内容。  [默认值：不试运行]
* `--help`：显示此消息并退出。

示例
  $ hf 缓存 rm 模型/gpt2
  $ hf 缓存 rm hf://models/openai-community/gpt2
  $ hf 缓存 rm 
  $ hf 缓存 rm 模型/gpt2 --dry-run
  $ hf 缓存 rm 模型/gpt2 --是了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf cache verify`

从缓存或本地目录验证单个存储库修订的校验和。

示例：
  - 验证缓存中的主要版本：`hf cache verify gpt2`
  - 验证具体修订：`hf cache verify gpt2 --revision refs/pr/1`
  - 验证数据集：`hf cache verify karpathy/fineweb-edu-100b-shuffle --repo-type dataset`
  - 验证本地目录：`hf cache verify deepseek-ai/DeepSeek-OCR --local-dir /path/to/repo`

**用法**：

```console
$ hf cache verify [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--cache-dir TEXT`：验证缓存中的文件时使用的缓存目录（默认为 Hugging Face 缓存）。
* `--local-dir TEXT`：如果设置，则验证该目录下的文件而不是缓存。
* `--fail-on-missing-files`：如果某些文件在远程存在但在本地丢失，则失败。
* `--fail-on-extra-files`：如果某些文件本地存在但远程版本中不存在，则会失败。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 缓存验证 gpt2
  $ hf 缓存验证 gpt2 --revision refs/pr/1
  $ hf 缓存验证我的数据集 --repo-type 数据集

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf collections`

与 Hub 上的集合进行交互。

**用法**：

```console
$ hf collections [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `add-item`：将项目添加到集合中。
* `create`：在 Hub 上创建一个新集合。
* `delete`：从 Hub 中删除集合。
* `delete-item`：从集合中删除项目。
* `info`：获取有关 Hub 上的集合的信息。
* `list`：列出 Hub 上的收藏。 [别名：ls]
* `update`：更新 Hub 上集合的元数据。
* `update-item`：更新集合中的项目。

### `hf collections add-item`

将项目添加到集合中。

**用法**：

```console
$ hf collections add-item [OPTIONS] COLLECTION_SLUG ITEM_ID ITEM_TYPE:{model|dataset|space|paper|collection|bucket}
```

**参数**：

* `COLLECTION_SLUG`：集合 slug（例如，“用户名/集合-slug”）。  [必填]
* `ITEM_ID`：要添加的项目的 ID（repo 为 repo_id，论文为 paper ID）。  [必填]
* `ITEM_TYPE:{model|dataset|space|paper|collection|bucket}`：项目的类型（模型、数据集、空间、论文、集合或存储桶）。  [必填]

**选项**：* `--note TEXT`​​：附加到商品上的注释（最多 500 个字符）。
* `--exists-ok / --no-exists-ok`：如果该项目已在集合中，则不要引发错误。  [默认值：不存在-确定]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf collections add-item 用户名/my-collection Moonshotai/kimi-k2 模型
  $ hf collections add-item username/my-collection Qwen/DeepPlanning 数据集 --note “有用的数据集”
  $ hf collections add-item username/my-collection Tongyi-MAI/Z-图像空间

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf collections create`

在 Hub 上创建一个新集合。

**用法**：

```console
$ hf collections create [OPTIONS] TITLE
```

**参数**：

* `TITLE`：集合的标题。  [必填]

**选项**：* `--namespace TEXT`：命名空间（用户名或组织）。默认为经过身份验证的用户。
* `--description TEXT`：集合的描述（最多 150 个字符）。
* `--private / --no-private`：创建私人收藏。  [默认值：非私有]
* `--exists-ok / --no-exists-ok`：如果集合已经存在，则不要引发错误。  [默认值：不存在-确定]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 集合创建“我的模型”
  $ hf collections create "My Models" --description "我最喜欢的模型的集合" --private
  $ hf collections 创建“Org Collection”--namespace my-org

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf collections delete`

从中心删除集合。

**用法**：

```console
$ hf collections delete [OPTIONS] COLLECTION_SLUG
```

**参数**：

* `COLLECTION_SLUG`：集合 slug（例如，“用户名/集合-slug”）。  [必填]

**选项**：

* `--missing-ok / --no-missing-ok`：如果集合不存在，请勿引发错误。  [默认值：无缺失-确定]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 集合删除用户名/我的集合
  $ hf collections 删除用户名/my-collection --missing-ok

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf collections delete-item`

从集合中删除项目。

**用法**：

```console
$ hf collections delete-item [OPTIONS] COLLECTION_SLUG ITEM_OBJECT_ID
```

**参数**：

* `COLLECTION_SLUG`：集合 slug（例如，“用户名/集合-slug”）。  [必填]
* `ITEM_OBJECT_ID`：集合中项目的ID（从“hf collections info”返回的`item_object_id`字段中检索。[必需]

**选项**：

* `--missing-ok / --no-missing-ok`：如果该项目不存在，请勿引发错误。  [默认值：无缺失-确定]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

### `hf collections info`

获取有关 Hub 上的集合的信息。

**用法**：

```console
$ hf collections info [OPTIONS] COLLECTION_SLUG
```

**参数**：

* `COLLECTION_SLUG`：集合 slug（例如，“用户名/集合-slug”）。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 集合信息用户名/my-collection-slug了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf collections list`

列出 Hub 上的集合。 [别名：ls]

**用法**：

```console
$ hf collections list [OPTIONS]
```

**选项**：

* `--owner TEXT`：按所有者用户名或组织过滤。
* `--item TEXT`：过滤包含特定项目的集合（例如，“models/gpt2”、“datasets/squad”、“papers/2311.12983”）。
* `--sort [lastModified|trending|upvotes]`：按上次修改、趋势或点赞对结果进行排序。
* `--limit INTEGER`：限制结果数量。  [默认值：10]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 集合 ls
  $ hf 集合 ls --owner nvidia
  $ hf 集合 ls --item models/teknium/OpenHermes-2.5-Mistral-7B --limit 10

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf collections update`

更新 Hub 上集合的元数据。

**用法**：

```console
$ hf collections update [OPTIONS] COLLECTION_SLUG
```

**参数**：

* `COLLECTION_SLUG`：集合 slug（例如，“用户名/集合-slug”）。  [必填]

**选项**：* `--title TEXT`：集合的新标题。
* `--description TEXT`：集合的新描述（最多 150 个字符）。
* `--position INTEGER`：集合在所有者列表中的新位置。
* `--private / --no-private`：集合是否应该是私有的。
* `--theme TEXT`：集合的主题颜色（例如“绿色”、“蓝色”）。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf collections update username/my-collection --title "新标题"
  $ hf collections update username/my-collection --description "更新描述"
  $ hf collections 更新用户名/my-collection --private --theme green

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf collections update-item`

更新集合中的项目。

**用法**：

```console
$ hf collections update-item [OPTIONS] COLLECTION_SLUG ITEM_OBJECT_ID
```

**参数**：

* `COLLECTION_SLUG`：集合 slug（例如，“用户名/集合-slug”）。  [必填]
* `ITEM_OBJECT_ID`：集合中项目的 ID（来自“item_object_id”字段，而不是 repo_id）。  [必填]

**选项**：* `--note TEXT`：该项目的新注释（最多 500 个字符）。
* `--position INTEGER`：项目在集合中的新位置。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf collections update-item username/my-collection ITEM_OBJECT_ID --note "更新的注释"
  $ hf 集合更新项目用户名/我的集合 ITEM_OBJECT_ID --位置 0

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf cp`

在本地路径、存储库和存储桶之间复制文件。

处理上传（本地/stdin -> repo/bucket）、下载（repo/bucket -> local/stdout）和
远程到远程副本（repo/bucket -> repo/bucket）。存储桶到存储库和本地到本地
不支持副本。对于目录，使用 `hf upload`/`hf download` (repos) 或
`hf buckets sync`（桶）。远程到远程复制仅在同一存储内工作
区域（https://huggingface.co/docs/hub/storage-regions）。

**用法**：

```console
$ hf cp [OPTIONS] SRC [DST]
```

**参数**：* `SRC`：来源：本地文件、hf:// URI（存储库或存储桶）或 - 用于标准输入。  [必填]
* `[DST]`：目标：本地路径、hf:// URI（存储库或存储桶）或 - 用于标准输出。

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf cp hf://用户名/my-model/config.json
  $ hf cp hf://用户名/my-model/config.json ./config.json
  $ hf cp hf://datasets/用户名/my-dataset/data.csv ./data/
  $ hf cp hf://buckets/username/my-bucket/config.json -
  $ hf cp ./model.safetensors hf://用户名/my-model/model.safetensors
  $ hf cp ./config.json hf://buckets/用户名/my-bucket/logs/
  $ hf cp - hf://buckets/username/my-bucket/config.json
  $ hf cp hf://用户名/源模型/ hf://用户名/目标模型/
  $ hf cp hf://datasets/用户名/my-dataset/processed/ hf://buckets/用户名/my-bucket/processed/
  $ hf cp hf://buckets/username/my-bucket/logs/ hf://buckets/username/archive-bucket/ # 仅复制内容

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf datasets`

与 Hub 上的数据集交互。**用法**：

```console
$ hf datasets [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `card`：获取数据集卡（自述文件）...
* `info`：获取有关 Hub 上数据集的信息。
* `leaderboard`：列出数据集排行榜中的模型分数。
* `list`：列出 Hub 上的数据集，或... [别名：ls]
* `parquet`：列出可用于...的 parquet 文件 URL
* `sql`：使用 DuckDB 执行原始 SQL 查询...

### `hf datasets card`

获取 Hub 上数据集的数据集卡 (README)。

**用法**：

```console
$ hf datasets card [OPTIONS] DATASET_ID
```

**参数**：

* `DATASET_ID`：数据集ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--metadata`：仅输出卡中的元数据。
* `--text`：仅输出文本正文（无元数据）。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 数据集卡 HuggingFaceFW/fineweb
  $ hf 数据集卡 HuggingFaceFW/fineweb --metadata
  $ hf 数据集卡 HuggingFaceFW/fineweb --metadata --format json
  $ hf 数据集卡 HuggingFaceFW/fineweb --text

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf datasets info`获取有关 Hub 上数据集的信息。

**用法**：

```console
$ hf datasets info [OPTIONS] DATASET_ID
```

**参数**：

* `DATASET_ID`：数据集ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--expand TEXT`：要返回的逗号分隔属性。使用时，仅返回列出的属性（和 id）。示例：“--expand=下载、喜欢、标签”。有效：作者、cardData、引用、createdAt、描述、禁用、下载、downloadsAllTime、gate、lastModified、likes、mainSize、paperswithcode_id、private、resourceGroup、sha、siblings、tags、trendingScore、usedStorage。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 数据集信息 HuggingFaceFW/fineweb
  $ hf 数据集信息 my-dataset --expand 下载、喜欢、标签

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf datasets leaderboard`列出数据集排行榜中的模型分数。此命令有助于找到任务的最佳模型或通过基准分数比较模型。使用“hf datasets ls --filter benchmark:official”列出可用的排行榜。

**用法**：

```console
$ hf datasets leaderboard [OPTIONS] DATASET_ID
```

**参数**：

* `DATASET_ID`：基准数据集ID（例如`SWE-bench/SWE-bench_Verified`）。  [必填]

**选项**：

* `--limit INTEGER`：限制结果数量。  [默认值：20]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 数据集排行榜 SWE-bench/SWE-bench_Verified
  $ hf 数据集排行榜 SWE-bench/SWE-bench_Verified --limit 5 --format json
  $ hf datasets ls --filter benchmark:official # 列出可用的排行榜

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf datasets list`

列出 Hub 上的数据集或数据集存储库中的文件。 [别名：ls]

当不带参数调用时，列出集线器上的数据集。
当使用数据集 ID 调用时，列出该数据集存储库中的文件。

**用法**：

```console
$ hf datasets list [OPTIONS] [REPO_ID]
```

**参数**：* `[REPO_ID]`：用于列出文件的数据集 ID（例如 `username/repo-name`）。如果省略，则列出数据集。

**选项**：

* `--search TEXT`：搜索查询。
* `--author TEXT`：按作者或组织过滤。
* `--filter TEXT`：按标签过滤（例如“文本分类”）。可以多次使用。
* `--sort [created_at|downloads|last_modified|likes|trending_score]`：对结果进行排序。
* `--limit INTEGER`：限制结果数量。  [默认值：30]
* `--expand TEXT`：要返回的逗号分隔属性。使用时，仅返回列出的属性（和 id）。示例：“--expand=下载、喜欢、标签”。有效：作者、cardData、引用、createdAt、描述、禁用、下载、downloadsAllTime、gate、lastModified、likes、mainSize、paperswithcode_id、private、resourceGroup、sha、siblings、tags、trendingScore、usedStorage。
* `-h, --human-readable`：以人类可读的格式显示大小（仅用于列出文件）。
* `--tree`：以树形格式列出文件（仅用于列出文件）。
* `-R, --recursive`：递归列出文件（仅用于列出文件）。
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 数据集 ls
  $ hf 数据集 ls --sort downloads --limit 10
  $ hf 数据集 ls --搜索“代码”
  $ hf datasets ls --filter benchmark:official
  $ hf 数据集 ls HuggingFaceFW/fineweb
  $ hf 数据集 ls HuggingFaceFW/fineweb -R
  $ hf 数据集 ls HuggingFaceFW/fineweb --tree -h

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf datasets parquet`

列出可用于数据集的 parquet 文件 URL。

**用法**：

```console
$ hf datasets parquet [OPTIONS] DATASET_ID
```

**参数**：

* `DATASET_ID`：数据集ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--subset TEXT`：按子集/配置过滤镶木地板条目。
* `--split TEXT`：通过分割过滤镶木地板条目。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 数据集镶木地板 cfahlgren1/hub-stats
  $ hf 数据集 parquet cfahlgren1/hub-stats --subset models
  $ hf 数据集镶木地板 cfahlgren1/hub-stats --split train
  $ hf 数据集镶木地板 cfahlgren1/hub-stats --format json了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf datasets sql`

使用 DuckDB 针对数据集 parquet URL 执行原始 SQL 查询。

**用法**：

```console
$ hf datasets sql [OPTIONS] SQL
```

**参数**：

* `SQL`：要执行的原始 SQL 查询。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf datasets sql "SELECT COUNT(*) AS rows FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet')"
  $ hf 数据集 sql "SELECT * FROM read_parquet('https://huggingface.co/api/datasets/cfahlgren1/hub-stats/parquet/models/train/0.parquet') LIMIT 5" --format json

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf discussions`

管理 Hub 上的讨论和拉取请求。

**用法**：

```console
$ hf discussions [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：* `close`：关闭讨论或拉取请求。
* `comment`：对讨论或拉取请求发表评论。
* `create`：创建新的讨论或拉取请求...
* `diff`：显示拉取请求的差异。
* `edit`：编辑讨论中的现有评论...
* `info`：获取有关讨论或拉取请求的信息。
* `list`：列出存储库上的讨论和拉取请求。 [别名：ls]
* `merge`：合并拉取请求。
* `rename`：重命名讨论或拉取请求。
* `reopen`：重新开启封闭式讨论或拉取请求。

### `hf discussions close`

关闭讨论或拉取请求。

**用法**：

```console
$ hf discussions close [OPTIONS] REPO_ID NUM
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]

**选项**：

* `--comment TEXT`：关闭时发布的可选评论。
* `-y, --yes`：跳过确认提示。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 讨论关闭用户名/我的模型 5
  $ hf Discussion close username/my-model 5 --comment "关闭已解决。"了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions comment`

对讨论或拉取请求发表评论。

**用法**：

```console
$ hf discussions comment [OPTIONS] REPO_ID NUM
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]

**选项**：

* `--body TEXT`：评论文本（支持Markdown）。
* `--body-file PATH`：从文件中读取注释。使用“-”作为标准输入。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf Discussion comment username/my-model 5 --body "感谢您的报告！"
  $ hf 讨论评论 用户名/我的模型 5 --body "LGTM!"

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions create`

在存储库上创建新的讨论或拉取请求。

**用法**：

```console
$ hf discussions create [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：* `--title TEXT`：讨论或拉取请求的标题。  [必填]
* `--body TEXT`：描述（支持Markdown）。
* `--body-file PATH`：从文件中读取描述。使用“-”作为标准输入。
* `--pull-request, --pr`：创建拉取请求而不是讨论。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf Discussion 创建用户名/我的模型 --title “Bug 报告”
  $ hf Discussion create username/my-model --title "功能请求" --body "请添加 X"
  $ hf Discussion 创建用户名/我的模型 --title "修复拼写错误" --pull-request
  $ hf Discussion create username/my-dataset --type dataset --title "数据质量问题"

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions diff`

显示拉取请求的差异。

**用法**：

```console
$ hf discussions diff [OPTIONS] REPO_ID NUM
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]

**选项**：* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 讨论 diff 用户名/我的模型 5

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions edit`

编辑讨论或拉取请求的现有评论。

**用法**：

```console
$ hf discussions edit [OPTIONS] REPO_ID NUM COMMENT_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]
* `COMMENT_ID`：要编辑的评论的 ID（请参阅“hf 讨论信息 ... --format json”）。  [必填]

**选项**：

* `--body TEXT`：新的评论文本（支持Markdown）。
* `--body-file PATH`：从文件中读取新评论。使用“-”作为标准输入。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf Discussion edit username/my-model 5 abc123 --body“更新评论。”
  $ hf Discussion 编辑用户名/我的模型 5 abc123 --body-filefixed.md

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions info`

获取有关讨论或拉取请求的信息。

**用法**：

```console
$ hf discussions info [OPTIONS] REPO_ID NUM
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 讨论信息 用户名/我的模型 5
  $ hf 讨论信息 用户名/我的模型 5 --format json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions list`

列出存储库上的讨论和拉取请求。 [别名：ls]

**用法**：

```console
$ hf discussions list [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：* `-s, --status [open|closed|merged|draft|all]`：按状态过滤（打开、关闭、合并、草稿、全部）。  [默认：打开]
* `-k, --kind [all|discussion|pull_request]`：按种类过滤（讨论、pull_request、全部）。  [默认值：全部]
* `--author TEXT`：按作者或组织过滤。
* `--limit INTEGER`：限制结果数量。  [默认值：30]
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 讨论列表 用户名/我的模型
  $ hf 讨论列表 用户名/我的模型 --kind pull_request --status 已合并
  $ hf 讨论列表用户名/我的数据集--类型数据集--状态已关闭
  $ hf 讨论列表 用户名/我的模型 --author alice --format json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions merge`

合并拉取请求。

**用法**：

```console
$ hf discussions merge [OPTIONS] REPO_ID NUM
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]

**选项**：* `--comment TEXT`：合并时发布的可选评论。
* `-y, --yes`：跳过确认提示。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 讨论 合并用户名/我的模型 5
  $ hf Discussions merge username/my-model 5 --comment "合并，谢谢！"

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions rename`

重命名讨论或拉取请求。

**用法**：

```console
$ hf discussions rename [OPTIONS] REPO_ID NUM NEW_TITLE
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]
* `NEW_TITLE`：新标题。  [必填]

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf Discussions 重命名用户名/my-model 5“更新的标题”了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf discussions reopen`

重新开启封闭式讨论或拉取请求。

**用法**：

```console
$ hf discussions reopen [OPTIONS] REPO_ID NUM
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `NUM`：讨论或拉取请求编号。  [必填]

**选项**：

* `--comment TEXT`：重新打开时发布的可选评论。
* `-y, --yes`：跳过确认提示。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 讨论重新打开用户名/我的模型 5
  $ hf 讨论重新打开用户名/我的模型 5 --comment“重新打开以进行进一步调查。”

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf download`

从集线器下载文件。

**用法**：

```console
$ hf download [OPTIONS] REPO_ID [FILENAMES]...
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `[FILENAMES]...`：要下载的文件（例如`config.json`、`data/metadata.jsonl`）。

**选项**：* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认值：（型号）]
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--include TEXT`：从要下载的文件中包含的全局模式。例如：*.json
* `--exclude TEXT`：从要下载的文件中排除的全局模式。
* `--cache-dir TEXT`：保存文件的目录。
* `--local-dir TEXT`：如果设置，下载的文件将放在该目录下。查看 https://huggingface.co/docs/huggingface_hub/guides/download#download-files-to-a-local-folder 了解更多详细信息。
* `--force-download / --no-force-download`：如果为True，即使文件已经缓存，也会下载它们。  [默认值：不强制下载]
* `--dry-run / --no-dry-run`：如果为 True，则执行试运行而不实际下载文件。  [默认值：不试运行]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--max-workers INTEGER`：用于下载文件的最大工作线程数。默认值为 8。[默认值：8]
* `--help`：显示此消息并退出。示例
  $ hf 下载meta-llama/Llama-3.2-1B-Instruct
  $ hf 下载 meta-llama/Llama-3.2-1B-Instruct config.json tokenizer.json
  $ hf 下载meta-llama/Llama-3.2-1B-Instruct --include "*.safetensors" --exclude "*.bin"
  $ hf 下载meta-llama/Llama-3.2-1B-Instruct --local-dir ./models/llama
  $ hf 下载 HuggingFaceM4/FineVision art/ --repo-type 数据集
  $ hf 下载 hf://datasets/HuggingFaceH4/ultrachat_200k

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf endpoints`

管理拥抱脸部推理端点。

**用法**：

```console
$ hf endpoints [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：* `catalog`：与推理端点交互...
* `delete`：永久删除推理端点。
* `deploy`：从集线器部署推理端点...
* `describe`：获取现有端点的信息。
* `hardware`：列出可用于部署...的硬件
* `list`：列出...的所有推理端点 [别名：ls]
* `list-catalog`：列出可用的目录型号。
* `pause`：暂停推理端点。
* `resume`：恢复推理端点。
* `scale-to-zero`：将推理端点缩放为零。
* `update`：更新现有端点。

### `hf endpoints catalog`

与推理端点目录交互。

**用法**：

```console
$ hf endpoints catalog [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `deploy`：从...部署推理端点
* `list`：列出可用的目录型号。 [别名：ls]

#### `hf endpoints catalog deploy`

从模型目录部署推理端点。

**用法**：

```console
$ hf endpoints catalog deploy [OPTIONS]
```

**选项**：* `--repo TEXT`：与推理端点关联的模型存储库的名称（例如“openai/gpt-oss-120b”）。  [必填]
* `--name TEXT`：端点名称。
* `--accelerator TEXT`：用于推理的硬件加速器（例如“cpu”、“gpu”、“neuron”）。
* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点目录部署 --repo meta-llama/Llama-3.2-1B-Instruct

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf endpoints catalog list`

列出可用的目录型号。 [别名：ls]

**用法**：

```console
$ hf endpoints catalog list [OPTIONS]
```

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点目录 ls

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints delete`

永久删除推理端点。

**用法**：

```console
$ hf endpoints delete [OPTIONS] NAME
```

**参数**：* `NAME`：端点名称。  [必填]

**选项**：

* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--yes`：跳过确认提示。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点删除我的端点

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints deploy`

从 Hub 存储库部署推理端点。

运行 `hf endpoints hardware` 列出有效的 `--vendor`、`--region`、`--accelerator`、`--instance-type` 和
`--instance-size` 组合。

**用法**：

```console
$ hf endpoints deploy [OPTIONS] NAME
```

**参数**：

* `NAME`：端点名称。  [必填]

**选项**：* `--repo TEXT`：与推理端点关联的模型存储库的名称（例如“openai/gpt-oss-120b”）。  [必填]
* `--framework TEXT`：模型使用的机器学习框架（例如“vllm”）。  [必填]
* `--accelerator TEXT`：用于推理的硬件加速器（例如“cpu”）。  [必填]
* `--instance-size TEXT`：用于托管模型的实例的大小或类型（例如“x4”）。  [必填]
* `--instance-type TEXT`：将部署推理端点的云实例类型（例如“intel-icl”）。  [必填]
* `--region TEXT`：将在其中创建推理端点的云区域（例如“us-east-1”）。  [必填]
* `--vendor TEXT`：将托管推理端点的云提供商或供应商（例如“aws”）。  [必填]
* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--task TEXT`：部署模型的任务（例如“文本分类”）。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--min-replica INTEGER`：推理端点保持运行的最小副本（实例）数量。  [默认值：1]* `--max-replica INTEGER`：推理端点可扩展的最大副本（实例）数量。  [默认值：1]
* `--scale-to-zero-timeout INTEGER`：非活动端点缩放为零之前的持续时间（以分钟为单位）。
* `--scaling-metric [pendingRequests|hardwareUsage]`：缩放的公制参考。
* `--scaling-threshold FLOAT`：用于触发扩展的扩展指标阈值。未提供缩放指标时将被忽略。
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--custom-image TEXT`：要运行的容器的 Docker 镜像 URL（例如“nexagi/sglang:v0.5.12”）。需要“--框架定制”。
* `--engine [custom|hf-serve|llamacpp|sglang|tei|tgi|tgi-neuron|vllm|vllm-neuron]`：用于运行 --custom-image 的托管引擎映像（例如“vllm”）。默认为任意容器。
* `--health-route TEXT`：容器暴露的健康检查路径（例如'/health'）。需要--自定义图像。
* `--port INTEGER`：容器监听的端口（例如30000）。需要--自定义图像。
* `--tensor-parallel-size INTEGER`：跨单个模型副本分片的加速器数量（仅限 vLLM 和 SGLang 引擎）。
* `--data-parallel-size INTEGER`：要运行的模型副本数量，每个加速器一个（仅限 vLLM 引擎）。
* `--container-command TEXT`：覆盖容器入口点，作为拆分为标记的带引号的字符串（例如“python -m sglang.launch_server”）。* `--container-args TEXT`：附加到容器入口点的参数，作为拆分为标记的带引号的字符串（例如“--tp 8 --reasoning-parser qwen3”）。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `--env-file TEXT`：读入环境变量文件。
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `--type [public|protected|authenticated|private]`：端点访问类型。默认为“经过身份验证”（令牌门控，可公开访问）。
* `--help`：显示此消息并退出。

示例
  $ hf 端点部署 my-endpoint --repo gpt2 --framework pytorch ...
  $ hf 端点部署 my-endpoint --repo openai/gpt-oss-120b --framework custom --engine vllm --custom-image vllm/vllm-openai:v0.23.0 --tensor-parallel-size 8 ...

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints describe`

获取有关现有端点的信息。

**用法**：

```console
$ hf endpoints describe [OPTIONS] NAME
```

**参数**：

* `NAME`：端点名称。  [必填]

**选项**：* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点描述我的端点

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints hardware`

列出可用于部署推理端点的硬件。

仅列出命名空间现在可以部署的硬件：可用状态和足够的加速器
一个副本的剩余配额。使用 `--all` 列出 API 返回的每个组合。

配额是针对每个命名空间的，因此请传递您将传递给 `hf endpoints deploy` 的相同 `--namespace`。价格在
美元，每个副本每小时。

**用法**：

```console
$ hf endpoints hardware [OPTIONS]
```

**选项**：* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--vendor TEXT`：仅显示此云提供商托管的硬件（例如“aws”）。
* `--region TEXT`：仅显示此云区域中可用的硬件（例如“us-east-1”）。
* `--accelerator TEXT`：仅显示具有此加速器的硬件（例如“cpu”、“gpu”、“neuron”）。
* `--instance-type TEXT`：仅显示该实例类型的硬件（例如“nvidia-l4”）。
* `-a, --all`：还显示当前无法部署的硬件（不可用、已弃用或超出配额）。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点硬件
  $ hf 端点硬件 --vendor aws --accelerator gpu

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints list`

列出给定命名空间的所有推理端点。 [别名：ls]

**用法**：

```console
$ hf endpoints list [OPTIONS]
```

**选项**：* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点 ls
  $ hf 端点 ls --namespace my-org

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints list-catalog`

列出可用的目录型号。

**用法**：

```console
$ hf endpoints list-catalog [OPTIONS]
```

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

### `hf endpoints pause`

暂停推理端点。

**用法**：

```console
$ hf endpoints pause [OPTIONS] NAME
```

**参数**：

* `NAME`：端点名称。  [必填]

**选项**：

* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点暂停我的端点了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints resume`

恢复推理端点。

**用法**：

```console
$ hf endpoints resume [OPTIONS] NAME
```

**参数**：

* `NAME`：端点名称。  [必填]

**选项**：

* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--fail-if-already-running`：如果`True`，如果推理端点已在运行，该方法将引发错误。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点恢复我的端点

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints scale-to-zero`

将推理端点缩放为零。

**用法**：

```console
$ hf endpoints scale-to-zero [OPTIONS] NAME
```

**参数**：

* `NAME`：端点名称。  [必填]

**选项**：

* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 端点缩放至零 my-endpoint

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf endpoints update`

更新现有端点。

**用法**：

```console
$ hf endpoints update [OPTIONS] NAME
```

**参数**：

* `NAME`：端点名称。  [必填]

**选项**：

* `--namespace TEXT`：与推理端点关联的命名空间。默认为当前用户的命名空间。
* `--repo TEXT`：与推理端点关联的模型存储库的名称（例如“openai/gpt-oss-120b”）。
* `--accelerator TEXT`：用于推理的硬件加速器（例如“cpu”）。
* `--instance-size TEXT`：用于托管模型的实例的大小或类型（例如“x4”）。
* `--instance-type TEXT`：将部署推理端点的云实例类型（例如“intel-icl”）。
* `--framework TEXT`：模型使用的机器学习框架（例如“自定义”）。
* `--revision TEXT`：要在推理端点上部署的特定模型修订版（例如“6c0e6080953db56375760c0471a8c5f2929baf11”）。
* `--task TEXT`：部署模型的任务（例如“文本分类”）。* `--custom-image TEXT`：要运行的容器的 Docker 镜像 URL（例如“nexagi/sglang:v0.5.12”）。替换端点上当前配置的图像而不是修补它，因此传递您想要保留的引擎和容器设置，首先运行“hf端点描述名称”来查看它们。
* `--engine [custom|hf-serve|llamacpp|sglang|tei|tgi|tgi-neuron|vllm|vllm-neuron]`：用于运行 --custom-image 的托管引擎映像（例如“vllm”）。默认为任意容器。
* `--health-route TEXT`：容器暴露的健康检查路径（例如'/health'）。需要--自定义图像。
* `--port INTEGER`：容器监听的端口（例如30000）。需要--自定义图像。
* `--tensor-parallel-size INTEGER`：跨单个模型副本分片的加速器数量（仅限 vLLM 和 SGLang 引擎）。
* `--data-parallel-size INTEGER`：要运行的模型副本数量，每个加速器一个（仅限 vLLM 引擎）。
* `--container-command TEXT`：覆盖容器入口点，作为拆分为标记的带引号的字符串（例如“python -m sglang.launch_server”）。替换当前值；传递一个空字符串来清除它。* `--container-args TEXT`：附加到容器入口点的参数，作为拆分为标记的带引号的字符串（例如“--enable-auto-tool-choice --tool-call-parser lfm2”）。替换当前在端点上设置的参数，而不是添加到它们中，因此请包括您想要保留的参数，首先运行“hf endpoints describe NAME”来查看它们。传递一个空字符串来清除它们。
* `--min-replica INTEGER`：推理端点保持运行的最小副本（实例）数量。
* `--max-replica INTEGER`：推理端点可扩展的最大副本（实例）数量。
* `--scale-to-zero-timeout INTEGER`：非活动端点缩放为零之前的持续时间（以分钟为单位）。
* `--scaling-metric [pendingRequests|hardwareUsage]`：缩放的公制参考。
* `--scaling-threshold FLOAT`：用于触发扩展的扩展指标阈值。未提供缩放指标时将被忽略。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 端点更新 my-endpoint --min-replica 2
  $ hf 端点更新 my-endpoint --tensor-parallel-size 8
  $ hf 端点更新我的端点 --container-args "--enable-auto-tool-choice --tool-call-parser lfm2"了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf env`

打印有关环境的信息。

**用法**：

```console
$ hf env [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

## `hf extensions`

管理 hf CLI 扩展。 [别名：分机]

安全警告：扩展是第三方可执行文件或 Python 包。仅从您信任的来源安装。

**用法**：

```console
$ hf extensions [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `exec`：执行已安装的扩展。
* `install`：从公共 GitHub 安装扩展...
* `list`：列出已安装的扩展命令。 [别名：ls]
* `remove`：删除已安装的扩展。 [别名：rm]
* `search`：搜索 GitHub 上可用的扩展...
* `update`：将已安装的扩展更新为其...

### `hf extensions exec`

执行已安装的扩展。

**用法**：

```console
$ hf extensions exec [OPTIONS] NAME
```

**参数**：

* `NAME`：扩展名（带或不带`hf-`前缀）。  [必填]

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 扩展 exec claude -- --help
  $ hf 扩展 exec claude --model zai-org/GLM-5了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf extensions install`

从公共 GitHub 存储库安装扩展。

安全警告：这会安装第三方可执行文件或 Python 包。
仅从您信任的来源安装。

**用法**：

```console
$ hf extensions install [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：`[OWNER/]hf-<name>` 格式的 GitHub 扩展存储库。  [必填]

**选项**：

* `--force`：如果已安装则覆盖。
* `--help`：显示此消息并退出。

示例
  $ hf 扩展安装 hf-claude
  $ hf 扩展安装 hanouticelina/hf-claude
  $ hf 扩展安装 alvarobartt/hf-mem

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf extensions list`

列出已安装的扩展命令。 [别名：ls]

**用法**：

```console
$ hf extensions list [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 扩展名列表

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf extensions remove`

删除已安装的扩展。 [别名：rm]**用法**：

```console
$ hf extensions remove [OPTIONS] NAME
```

**参数**：

* `NAME`：要删除的扩展名（带或不带`hf-`前缀）。  [必填]

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 扩展删除克劳德

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf extensions search`

搜索 GitHub 上可用的扩展（标有“hf-extension”主题）。

**用法**：

```console
$ hf extensions search [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 扩展名搜索

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf extensions update`

将已安装的扩展更新到最新版本。

**用法**：

```console
$ hf extensions update [OPTIONS] [NAME]
```

**参数**：

* `[NAME]`：更新扩展（带或不带`hf-`前缀，可选为`OWNER/hf-<name>`）。如果省略，将检查所有已安装的扩展并更新过时的扩展。

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 扩展更新
  $ hf 扩展更新 hf-claude
  $ hf 扩展更新 alvarobartt/hf-mem了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf jobs`

在 Hub 上运行和管理作业。

**用法**：

```console
$ hf jobs [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `cancel`：取消作业
* `hardware`：列出作业可用的硬件选项
* `inspect`：显示一个或多个的详细信息
* `labels`：更新作业上的标签。
* `list`：列出职位。 [别名：ls、ps]
* `logs`：获取Job的日志。
* `run`：运行作业。
* `scheduled`：在 Hub 上创建和管理计划作业。
* `ssh`：通过 SSH 连接到正在运行的作业。
* `stats`：获取资源使用统计信息并...
* `uv`：运行 UV 脚本（Python 内联...
* `wait`：等待一个或多个作业达到...

### `hf jobs cancel`

取消工作

**用法**：

```console
$ hf jobs cancel [OPTIONS] JOB_ID
```

**参数**：

* `JOB_ID`：作业 ID（或“namespace/job_id”）[必需]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作取消了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs hardware`

列出作业可用的硬件选项

**用法**：

```console
$ hf jobs hardware [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

示例
  $ hf 工作 硬件

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs inspect`

显示一项或多项职位的详细信息

**用法**：

```console
$ hf jobs inspect [OPTIONS] JOB_IDS...
```

**参数**：

* `JOB_IDS...`：要检查的作业 ID（或“namespace/job_id”）[必需]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作检查 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs labels`

更新作业上的标签。传递 --label 会替换所有现有标签；仅通过 --name 就可以保留它们。

**用法**：

```console
$ hf jobs labels [OPTIONS] JOB_ID
```

**参数**：* `JOB_ID`：作业 ID（或“namespace/job_id”）[必需]

**选项**：

* `--name TEXT`：为作业命名。存储为 `name` 标签。名称不必是唯一的。默认为图像或脚本名称加上命令的短哈希。
* `-l, --label TEXT`：设置标签。例如。 --label KEY=VALUE 或 --label LABEL
* `--clear`：从作业中删除所有标签。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作标签 --name Training-v2
  $ hf 工作标签 --label env=prod --label team=ml
  $ hf 工作标签 --clear

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs list | ls | ps`

列出职位。

使用 `--status` 按状态过滤（请参阅 [JobStage](/docs/huggingface_hub/v1.29.0/en/package_reference/jobs#huggingface_hub.JobStage) 了解可能的值）并使用 `--label` 按 `key=value` 过滤
标签。作业必须匹配要列出的每个过滤器。

**用法**：

```console
$ hf jobs list | ls | ps [OPTIONS]
```

**选项**：* `-a, --all`：显示所有作业（默认显示正在运行和正在调度）。不能与--status 结合使用。
* `--status [COMPLETED|CANCELED|ERROR|DELETED|SCHEDULING|RUNNING]`：仅显示给定状态的作业。以逗号分隔或重复，例如`--status running,scheduling`。
* `-l, --label TEXT`：仅显示具有给定 `key=value` 标签的职位。重复以需要多个标签，例如`--label env=prod --label team=ml`。
* `--name TEXT`：仅显示具有给定名称的职位（`--label name=NAME` 的快捷方式）。
* `--limit INTEGER`：要显示的最大作业数。设置为 0 以显示全部（无限制）。  [默认值：100]
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `-f, --filter TEXT`：（已弃用）使用 `--status` 和 `--label` 代替。
* `--help`：显示此消息并退出。

示例
  $ hf 工作 ls
  $ hf 工作 ls -a
  $ hf jobs ls --status running,scheduling
  $ hf jobs ls --名称训练-v2
  $ hf jobs ls --label env=prod --label team=ml
  $ hf jobs ls --all --label hf-sandbox=1

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs logs`

获取作业的日志。默认情况下，打印当前可用的日志并退出（非阻塞）。
使用 --follow/-f 实时流式传输日志，直到作业完成。
使用 --tail/-n 限制返回的行数（如果支持，则在服务器端）。

注意：无论Job是否结束，下面都会在日志流结束时退出
成功或失败。运行`hf jobs inspect <job_id>`查看最终状态。

**用法**：

```console
$ hf jobs logs [OPTIONS] JOB_ID
```

**参数**：

* `JOB_ID`：作业 ID（或“namespace/job_id”）[必需]

**选项**：

* `-f, --follow`：跟随日志输出（流式传输直到作业完成）。如果没有此标志，则仅打印当前可用的日志。
* `-n, --tail INTEGER`：从日志末尾开始显示的行数。与 --follow 结合使用时，从最后 N 行开始流式传输。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 作业日志 
  $ hf 作业日志 -f 
  $ hf 作业日志 --tail 20 
  $ hf 作业日志 -f --tail 100 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档### `hf jobs run`

运行作业。

**用法**：

```console
$ hf jobs run [OPTIONS] IMAGE COMMAND...
```

**参数**：

* `IMAGE`：要使用的 Docker 镜像。  [必填]
* `COMMAND...`：要运行的命令。  [必填]

**选项**：

* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--name TEXT`：为作业命名。存储为 `name` 标签。名称不必是唯一的。默认为图像或脚本名称加上命令的短哈希。
* `-l, --label TEXT`：设置标签。例如。 --label KEY=VALUE 或 --label LABEL
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro|:rw] 或 LOCAL_DIR:/MOUNT_PATH[:ro|:rw]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。本地目录源首先同步到存储桶，并默认以只读方式挂载。例如。 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro 或 -v ./inputs:/inputs
* `--env-file TEXT`：读入环境变量文件。
* `--secrets-file TEXT`：读入秘密环境变量文件。* `--flavor [cpu-basic|cpu-upgrade|cpu-performance|cpu-xl|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8|h200|h200x2|h200x4|h200x8|rtx-pro-6000|rtx-pro-6000x2|rtx-pro-6000x4|rtx-pro-6000x8]`：硬件的味道。运行“hf jobs hardware”以列出可用的口味。默认为`cpu-basic`。
* `--timeout TEXT`：最大持续时间：带 s（秒，默认）、m（分钟）、h（小时）或 d（天）的整数。
* `-d, --detach`：在后台运行作业并打印作业ID。
* `--expose INTEGER`：通过作业代理公开容器端口。对多个端口重复该标志（例如`--expose 8000 --expose 8001`）。每个公开的端口都可以在公共作业域上访问；访问需要具有对作业命名空间的读取访问权限的 HF 令牌。
* `--ssh`：使作业的容器可通过 SSH 访问。连接`hf jobs ssh <job_id>`。需要在 https://huggingface.co/settings/keys 上注册的 SSH 公钥。
* `--resource-group-id TEXT`：要在其中创建作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf jobs run --name hello-world python:3.12 python -c 'print("Hello!")'
  $ hf jobs run --detach python:3.12 python script.py
  $ hf jobs run -e FOO=foo python:3.12 python script.py
  $ hf jobs run --secrets HF_TOKEN python:3.12 python script.py
  $ hf jobs run -v hf://org/my-model:/data -v hf://buckets/org/b:/mnt python:3.12 python script.py

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs scheduled`

在 Hub 上创建和管理计划作业。

**用法**：

```console
$ hf jobs scheduled [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `delete`：删除预定作业。
* `inspect`：显示一个或多个的详细信息
* `labels`：更新计划作业上的标签。
* `list`: 列出预定的作业 [别名: ls, ps]
* `resume`：恢复（取消暂停）预定的作业。
* `run`：安排作业。
* `suspend`：暂停（暂停）预定的作业。
* `trigger`：触发预定的Job立即运行...
* `uv`：在 HF 基础设施上安排 UV 脚本。

#### `hf jobs scheduled delete`

删除预定作业。

**用法**：

```console
$ hf jobs scheduled delete [OPTIONS] SCHEDULED_JOB_ID
```

**参数**：

* `SCHEDULED_JOB_ID`：计划作业 ID（或“namespace/scheduled_job_id”）[必需]**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 作业计划删除 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled inspect`

显示一项或多项计划作业的详细信息

**用法**：

```console
$ hf jobs scheduled inspect [OPTIONS] SCHEDULED_JOB_IDS...
```

**参数**：

* `SCHEDULED_JOB_IDS...`：要检查的计划作业 ID（或“namespace/scheduled_job_id”）[必需]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作安排检查 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled labels`

更新计划作业上的标签。传递 --label 会替换所有现有标签；仅通过 --name 就可以保留它们。

**用法**：

```console
$ hf jobs scheduled labels [OPTIONS] SCHEDULED_JOB_ID
```**参数**：

* `SCHEDULED_JOB_ID`：计划作业 ID（或“namespace/scheduled_job_id”）[必需]

**选项**：

* `--name TEXT`：为作业命名。存储为 `name` 标签。名称不必是唯一的。默认为图像或脚本名称加上命令的短哈希。
* `-l, --label TEXT`：设置标签。例如。 --label KEY=VALUE 或 --label LABEL
* `--clear`：从计划作业中删除所有标签。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf jobs 计划标签 --name daily-script
  $ hf 作业计划标签 --label env=prod --label team=ml
  $ hf 作业计划标签 --clear

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled list | ls | ps`

列出预定的作业

**用法**：

```console
$ hf jobs scheduled list | ls | ps [OPTIONS]
```

**选项**：* `-a, --all`：显示所有计划的作业（默认隐藏暂停）
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `-f, --filter TEXT`：根据提供的条件过滤输出（格式：key=value）
* `--help`：显示此消息并退出。

示例
  $ hf 已安排的工作 ls

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled resume`

恢复（取消暂停）计划的作业。

**用法**：

```console
$ hf jobs scheduled resume [OPTIONS] SCHEDULED_JOB_ID
```

**参数**：

* `SCHEDULED_JOB_ID`：计划作业 ID（或“namespace/scheduled_job_id”）[必需]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作计划简历 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled run`

安排工作。

**用法**：

```console
$ hf jobs scheduled run [OPTIONS] SCHEDULE IMAGE COMMAND...
```

**参数**：* `SCHEDULE`：每年、每年、每月、每周、每天、每小时或 CRON 计划表达式之一。  [必填]
* `IMAGE`：要使用的 Docker 镜像。  [必填]
* `COMMAND...`：要运行的命令。  [必填]

**选项**：

* `--suspend / --no-suspend`：暂停（暂停）预定的Job
* `--concurrency / --no-concurrency`：允许该Job的多个实例同时运行
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--name TEXT`：为作业命名。存储为 `name` 标签。名称不必是唯一的。默认为图像或脚本名称加上命令的短哈希。
* `-l, --label TEXT`：设置标签。例如。 --label KEY=VALUE 或 --label LABEL
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro|:rw] 或 LOCAL_DIR:/MOUNT_PATH[:ro|:rw]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。本地目录源首先同步到存储桶，并默认以只读方式挂载。例如。 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro 或 -v ./inputs:/inputs* `--env-file TEXT`：读入环境变量文件。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `--flavor [cpu-basic|cpu-upgrade|cpu-performance|cpu-xl|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8|h200|h200x2|h200x4|h200x8|rtx-pro-6000|rtx-pro-6000x2|rtx-pro-6000x4|rtx-pro-6000x8]`：硬件的味道。运行“hf jobs hardware”以列出可用的口味。默认为`cpu-basic`。
* `--timeout TEXT`：最大持续时间：带 s（秒，默认）、m（分钟）、h（小时）或 d（天）的整数。
* `--expose INTEGER`：通过作业代理公开容器端口。对多个端口重复该标志（例如`--expose 8000 --expose 8001`）。每个公开的端口都可以在公共作业域上访问；访问需要具有对作业命名空间的读取访问权限的 HF 令牌。
* `--resource-group-id TEXT`：要在其中创建作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf jobs Scheduled run "0 0 * * *" --name daily-script python:3.12 python script.py

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档#### `hf jobs scheduled suspend`

暂停（暂停）计划的作业。

**用法**：

```console
$ hf jobs scheduled suspend [OPTIONS] SCHEDULED_JOB_ID
```

**参数**：

* `SCHEDULED_JOB_ID`：计划作业 ID（或“namespace/scheduled_job_id”）[必需]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作计划暂停 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled trigger`

触发计划的作业立即运行（不更改计划）。

**用法**：

```console
$ hf jobs scheduled trigger [OPTIONS] SCHEDULED_JOB_ID
```

**参数**：

* `SCHEDULED_JOB_ID`：计划作业 ID（或“namespace/scheduled_job_id”）[必需]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 作业计划触发器了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf jobs scheduled uv`

在 HF 基础设施上安排 UV 脚本。

**用法**：

```console
$ hf jobs scheduled uv [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `run`：在 HF 上运行 UV 脚本（本地文件或 URL）...

##### `hf jobs scheduled uv run`

在 HF 基础设施上运行 UV 脚本（本地文件或 URL）

**用法**：

```console
$ hf jobs scheduled uv run [OPTIONS] SCHEDULE SCRIPT [SCRIPT_ARGS]...
```

**参数**：

* `SCHEDULE`：每年、每年、每月、每周、每天、每小时或 CRON 计划表达式之一。  [必填]
* `SCRIPT`：要运行的UV脚本（本地文件或URL）[必需]
* `[SCRIPT_ARGS]...`：脚本的参数

**选项**：* `--suspend / --no-suspend`：暂停（暂停）预定的Job
* `--concurrency / --no-concurrency`：允许该Job的多个实例同时运行
* `--image TEXT`：使用安装了`uv`的自定义 Docker 镜像。
* `--flavor [cpu-basic|cpu-upgrade|cpu-performance|cpu-xl|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8|h200|h200x2|h200x4|h200x8|rtx-pro-6000|rtx-pro-6000x2|rtx-pro-6000x4|rtx-pro-6000x8]`：硬件的味道。运行“hf jobs hardware”以列出可用的口味。默认为`cpu-basic`。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--name TEXT`：为作业命名。存储为 `name` 标签。名称不必是唯一的。默认为图像或脚本名称加上命令的短哈希。
* `-l, --label TEXT`：设置标签。例如。 --label KEY=VALUE 或 --label LABEL
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro|:rw] 或 LOCAL_DIR:/MOUNT_PATH[:ro|:rw]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。本地目录源首先同步到存储桶，并默认以只读方式挂载。例如。 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro 或 -v ./inputs:/inputs* `--env-file TEXT`：读入环境变量文件。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `--timeout TEXT`：最大持续时间：带 s（秒，默认）、m（分钟）、h（小时）或 d（天）的 int。
* `--expose INTEGER`：通过作业代理公开容器端口。对多个端口重复该标志（例如`--expose 8000 --expose 8001`）。每个公开的端口都可以在公共作业域上访问；访问需要具有对作业命名空间的读取访问权限的 HF 令牌。
* `--resource-group-id TEXT`：要在其中创建作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--with TEXT`：使用安装的给定软件包运行
* `-p, --python TEXT`：运行环境使用的Python解释器
* `--help`：显示此消息并退出。

示例
  $ hf jobs Scheduled uv run "0 0 * * *" --name daily-script script.py
  $ hf jobs Scheduled uv run "0 0 * * *" script.py --with pandas了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs ssh`

通过 SSH 连接到正在运行的作业。

如果作业尚未运行，则等待其达到 RUNNING 状态后再执行
连接。需要在启用 SSH 的情况下启动作业 (`hf jobs run --ssh ...`)
以及要在 https://huggingface.co/settings/keys 注册的 SSH 公钥。

**用法**：

```console
$ hf jobs ssh [OPTIONS] JOB_ID
```

**参数**：

* `JOB_ID`：作业 ID（或“namespace/job_id”）[必需]

**选项**：

* `-i, --identity-file PATH`：SSH 身份文件的路径（转发到`ssh -i`）。
* `--dry-run`：打印 SSH 命令而不是运行它。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 工作 ssh 
  $ hf jobs ssh --dry-run
  $ hf 工作 ssh -i ~/.ssh/id_ed25519

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs stats`

获取Jobs的资源使用统计和指标

**用法**：

```console
$ hf jobs stats [OPTIONS] [JOB_IDS]...
```**参数**：

* `[JOB_IDS]...`：作业 ID（或“namespace/job_id”）

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 就业统计 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs uv`

在 HF 基础设施上运行 UV 脚本（具有内联依赖项的 Python）。

**用法**：

```console
$ hf jobs uv [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `run`：在 HF 上运行 UV 脚本（本地文件或 URL）...

#### `hf jobs uv run`

在 HF 基础设施上运行 UV 脚本（本地文件或 URL）

**用法**：

```console
$ hf jobs uv run [OPTIONS] SCRIPT [SCRIPT_ARGS]...
```

**参数**：

* `SCRIPT`：要运行的UV脚本（本地文件或URL）[必需]
* `[SCRIPT_ARGS]...`：脚本的参数

**选项**：* `--image TEXT`：使用安装了`uv`的自定义 Docker 镜像。
* `--flavor [cpu-basic|cpu-upgrade|cpu-performance|cpu-xl|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8|h200|h200x2|h200x4|h200x8|rtx-pro-6000|rtx-pro-6000x2|rtx-pro-6000x4|rtx-pro-6000x8]`：硬件的味道。运行“hf jobs hardware”以列出可用的口味。默认为`cpu-basic`。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--name TEXT`：为作业命名。存储为 `name` 标签。名称不必是唯一的。默认为图像或脚本名称加上命令的短哈希。
* `-l, --label TEXT`：设置标签。例如。 --label KEY=VALUE 或 --label LABEL
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro|:rw] 或 LOCAL_DIR:/MOUNT_PATH[:ro|:rw]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。本地目录源首先同步到存储桶，并默认以只读方式挂载。例如。 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro 或 -v ./inputs:/inputs
* `--env-file TEXT`：读入环境变量文件。
* `--secrets-file TEXT`：读入秘密环境变量文件。* `--timeout TEXT`：最大持续时间：带 s（秒，默认）、m（分钟）、h（小时）或 d（天）的整数。
* `-d, --detach`：在后台运行作业并打印作业ID。
* `--expose INTEGER`：通过作业代理公开容器端口。对多个端口重复该标志（例如`--expose 8000 --expose 8001`）。每个公开的端口都可以在公共作业域上访问；访问需要具有对作业命名空间的读取访问权限的 HF 令牌。
* `--ssh`：使作业的容器可通过 SSH 访问。连接`hf jobs ssh <job_id>`。需要在 https://huggingface.co/settings/keys 上注册的 SSH 公钥。
* `--resource-group-id TEXT`：要在其中创建作业的资源组的 ID。用于控制对组织内资源的访问以及成本归因/支出限制功能。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--with TEXT`：使用安装的给定软件包运行
* `-p, --python TEXT`：运行环境使用的Python解释器
* `--help`：显示此消息并退出。示例
  $ hf jobs uv run --name my-script my_script.py
  $ hf jobs uv run --detach my_script.py
  $ hf jobs uv run ml_training.py --flavor a10g-small
  $ hf jobs uv run --with 变形金刚 train.py
  $ hf jobs uv run -v hf://org/my-model:/data -v hf://buckets/org/b:/mnt script.py

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf jobs wait`

等待一个或多个作业达到最终状态。

阻塞直到每个作业完成，然后如果所有作业完成则以代码 0 退出
成功，或者如果任何作业被取消、出错或删除，则返回非零退出代码。

所有作业必须属于同一名称空间。

**用法**：

```console
$ hf jobs wait [OPTIONS] JOB_IDS...
```

**参数**：

* `JOB_IDS...`：要等待的作业 ID（或“namespace/job_id”）。  [必填]

**选项**：

* `--timeout TEXT`：最长等待时间：带 s（秒，默认）、m（分钟）、h（小时）或 d（天）的 int。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 工作等待 
  $ hf 工作等待  
  $ hf 工作 ls -q | xargs hf 作业等待

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf lfs-enable-largefiles`

配置您的存储库以允许上传 > 5GB 的文件。

此命令将 git-lfs 设置为使用自定义多部分传输代理
这使得能够高效地上传大文件的块。

**用法**：

```console
$ hf lfs-enable-largefiles [OPTIONS] PATH
```

**参数**：

* `PATH`：要配置的存储库的本地路径。  [必填]

**选项**：

* `--help`：显示此消息并退出。

## `hf lfs-multipart-upload`

用于分段上传的内部 git-lfs 自定义传输代理。

该函数实现了 git-lfs 分段上传的自定义传输协议。
处理大文件分块上传到 Hugging Face Hub。

**用法**：

```console
$ hf lfs-multipart-upload [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

## `hf models`

与 Hub 上的模型进行交互。

**用法**：

```console
$ hf models [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `card`：获取模型的模型卡（自述文件）...
* `info`：获取 Hub 上模型的信息。
* `list`：列出 Hub 上的模型，或... [别名：ls]### `hf models card`

获取 Hub 上模型的模型卡 (README)。

**用法**：

```console
$ hf models card [OPTIONS] MODEL_ID
```

**参数**：

* `MODEL_ID`：型号 ID（例如 `username/repo-name`）。  [必填]

**选项**：

* `--metadata`：仅输出卡中的元数据。
* `--text`：仅输出文本正文（无元数据）。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 型号卡 google/gemma-4-31B-it
  $ hf 模型卡 google/gemma-4-31B-it --metadata
  $ hf 模型卡 google/gemma-4-31B-it --metadata --format json
  $ hf 模型卡 google/gemma-4-31B-it --text

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf models info`

获取有关 Hub 上模型的信息。

**用法**：

```console
$ hf models info [OPTIONS] MODEL_ID
```

**参数**：

* `MODEL_ID`：型号 ID（例如 `username/repo-name`）。  [必填]

**选项**：* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--expand TEXT`：要返回的逗号分隔属性。使用时，仅返回列出的属性（和 id）。示例：“--expand=下载、喜欢、标签”。有效：作者、baseModels、cardData、childrenModelCount、config、createdAt、disabled、downloads、downloadsAllTime、evalResults、gate、gguf、inference、inferenceProviderMapping、lastModified、library_name、likes、mask_token、model-index、pipeline_tag、private、resourceGroup、safetensors、sha、siblings、spaces、tags、transformersInfo、trendingScore、usedStorage、widgetData。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 模型信息 meta-llama/Llama-3.2-1B-Instruct
  $ hf 型号信息 Qwen/Qwen3.5-9B --expand 下载、点赞、标签

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf models list`

列出 Hub 上的模型或模型存储库中的文件。 [别名：ls]当不带参数调用时，列出集线器上的模型。
当使用模型 ID 调用时，列出该模型存储库中的文件。

**用法**：

```console
$ hf models list [OPTIONS] [REPO_ID]
```

**参数**：

* `[REPO_ID]`：用于列出文件的模型 ID（例如 `username/repo-name`）。如果省略，则列出型号。

**选项**：

* `--search TEXT`：搜索查询。
* `--author TEXT`：按作者或组织过滤。
* `--filter TEXT`：按标签过滤（例如“文本分类”）。可以多次使用。
* `--pipeline-tag TEXT`：按管道标签（规范任务）过滤，例如‘总结’。
* `--gated / --no-gated`：按门控状态过滤。 '--gate' 仅适用于门控，'--no-gate' 仅适用于非门控。
* `--apps TEXT`：按可以运行模型的应用程序过滤，例如“ollama”或“vllm”。
* `--num-parameters TEXT`：按参数计数过滤，例如'最小值：6B，最大值：128B'。
* `--inference-provider [baseten|cerebras|cohere|deepinfra|fal-ai|featherless-ai|fireworks-ai|groq|hf-inference|novita|nscale|openai|ovhcloud|publicai|replicate|scaleway|together|wavespeed|zai-org]`：按为模型提供服务的推理提供者进行过滤，例如'烟花-ai'。
* `--warm`：仅列出当前由至少一个推理提供者提供服务的模型。
* `--sort [created_at|downloads|last_modified|likes|trending_score]`：对结果进行排序。
* `--limit INTEGER`：限制结果数量。  [默认值：30]* `--expand TEXT`：要返回的逗号分隔属性。使用时，仅返回列出的属性（和 id）。示例：“--expand=下载、喜欢、标签”。有效：作者、baseModels、cardData、childrenModelCount、config、createdAt、disabled、downloads、downloadsAllTime、evalResults、gate、gguf、inference、inferenceProviderMapping、lastModified、library_name、likes、mask_token、model-index、pipeline_tag、private、resourceGroup、safetensors、sha、siblings、spaces、tags、transformersInfo、trendingScore、usedStorage、widgetData。
* `-h, --human-readable`：以人类可读的格式显示大小（仅适用于列出文件）。
* `--tree`：以树形格式列出文件（仅用于列出文件）。
* `-R, --recursive`：递归列出文件（仅用于列出文件）。
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf models ls --sort downloads --limit 10
  $ hf models ls --search "llama" --author meta-llama
  $ hf 模型 ls --pipeline-tag 文本生成 --warm
  $ hf models ls --num-parameters 最小值：6B，最大值：128B --排序喜欢
  $ hf 模型 ls --no-gate --author google
  $ hf 模型 ls --apps llama.cpp --apps vllm
  $ hf models ls --inference-provider fireworks-ai --sort downloads
  $ hf 模型 ls --warm --search llama
  $ hf 模型 ls meta-llama/Llama-3.2-1B-Instruct
  $ hf 模型 ls meta-llama/Llama-3.2-1B-Instruct -R
  $ hf 模型 ls meta-llama/Llama-3.2-1B-Instruct --tree -h

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf papers`

与 Hub 上的论文互动。

**用法**：

```console
$ hf papers [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `info`：获取有关 Hub 上一篇论文的信息。
* `list`：列出 Hub 上的日报。 [别名：ls]
* `read`：以 Markdown 形式阅读论文。
* `search`：在 Hub 上搜索论文。

### `hf papers info`

获取有关 Hub 上论文的信息。

**用法**：

```console
$ hf papers info [OPTIONS] PAPER_ID
```

**参数**：

* `PAPER_ID`：arXiv 论文 ID（例如“2502.08025”）。  [必需的]**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ 高频论文信息 2601.15621

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf papers list`

列出 Hub 上的每日报纸。 [别名：ls]

**用法**：

```console
$ hf papers list [OPTIONS]
```

**选项**：

* `--date TEXT`：ISO 格式的日期 (YYYY-MM-DD) 或“今天”。
* `--week TEXT`：过滤的 ISO 周，例如“2025-W09”。
* `--month TEXT`：ISO 格式 (YYYY-MM) 中筛选的月份，例如“2025 年 2 月”。
* `--submitter TEXT`：按提交者的用户名过滤。
* `--sort [publishedAt|trending]`：对结果进行排序。
* `--limit INTEGER`：限制结果数量。  [默认值：50]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 论文 ls
  $ hf 论文 ls --排序趋势
  $ hf 论文 ls --date 2025-01-23
  $ hf 论文 ls --week 2025-W09
  $ hf 论文 ls --提交者 akhaliq
  $ hf 论文 ls --format json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档### `hf papers read`

以 Markdown 形式阅读论文。

**用法**：

```console
$ hf papers read [OPTIONS] PAPER_ID
```

**参数**：

* `PAPER_ID`：arXiv 论文 ID（例如“2502.08025”）。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 论文阅读 2601.15621

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf papers search`

在 Hub 上搜索论文。

**用法**：

```console
$ hf papers search [OPTIONS] QUERY
```

**参数**：

* `QUERY`：搜索查询字符串。  [必填]

**选项**：

* `--limit INTEGER`：限制结果数量。  [默认值：20]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf论文搜索“视觉语言”
  $ hf 论文搜索“注意力机制”--limit 10
  $ hf 论文搜索“扩散”--format json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf repos`

管理 Hub 上的存储库。 [别名：回购]

**用法**：

```console
$ hf repos [OPTIONS] [COMMAND] [ARGS]...
```

**选项**：* `--help`：显示此消息并退出。

**命令**：

* `branch`：管理 Hub 上存储库的分支。
* `cp`：在本地路径之间复制文件，...
* `create`：在 Hub 上创建一个新的存储库。
* `delete`：从 Hub 中删除存储库。
* `delete-files`：从 Hub 上的存储库中删除文件。
* `duplicate`：在 Hub 上复制一个存储库（型号，...
* `list`：列出所有存储库（模型、数据集、空间... [别名：ls]
* `move`：将存储库从名称空间移动到...
* `settings`：更新存储库的设置。
* `tag`：管理 Hub 上存储库的标签。

### `hf repos branch`

管理 Hub 上存储库的分支。

**用法**：

```console
$ hf repos branch [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `create`：为 Hub 上的存储库创建一个新分支。
* `delete`：从 Hub 上的存储库中删除分支。

#### `hf repos branch create`

为 Hub 上的存储库创建一个新分支。

**用法**：

```console
$ hf repos branch create [OPTIONS] REPO_ID BRANCH
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `BRANCH`：要创建的分支的名称。  [必填]

**选项**：* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--exist-ok / --no-exist-ok`：如果设置为 True，则分支已存在时不会引发错误。  [默认值：不存在-确定]
* `--help`：显示此消息并退出。

示例
  $ hf repos 分支创建 my-model dev
  $ hf repos 分支创建 my-model dev --revision abc123

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf repos branch delete`

从 Hub 上的存储库中删除分支。

**用法**：

```console
$ hf repos branch delete [OPTIONS] REPO_ID BRANCH
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `BRANCH`：要删除的分支的名称。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--help`：显示此消息并退出。

示例
  $ hf repos 分支删除 my-model dev了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos cp`

在本地路径、存储库和存储桶之间复制文件。

处理上传（本地/stdin -> repo/bucket）、下载（repo/bucket -> local/stdout）和
远程到远程副本（repo/bucket -> repo/bucket）。存储桶到存储库和本地到本地
不支持副本。对于目录，使用 `hf upload`/`hf download` (repos) 或
`hf buckets sync`（桶）。远程到远程复制仅在同一存储内工作
区域（https://huggingface.co/docs/hub/storage-regions）。

**用法**：

```console
$ hf repos cp [OPTIONS] SRC [DST]
```

**参数**：

* `SRC`：来源：本地文件、hf:// URI（存储库或存储桶）或 - 用于标准输入。  [必填]
* `[DST]`：目标：本地路径、hf:// URI（存储库或存储桶）或 - 用于标准输出。

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf repos cp hf://用户名/my-model/config.json config.json
  $ hf repos cp hf://datasets/username/my-dataset/data.csv data/
  $ hf repos cp hf://username/my-model/config.json -
  $ hf repos cp model.safetensors hf://username/my-model/model.safetensors
  $ hf repos cp config.json hf://用户名/my-model/logs/
  $ hf repos cp - hf://用户名/my-model/config.json
  $ hf repos cp hf://username/source-model/config.json hf://username/dest-model/config.json
  $ hf repos cp hf://datasets/username/my-dataset/processed/ hf://datasets/username/dest-dataset/processed/
  $ hf repos cp hf://用户名/my-model/logs/ hf://用户名/archive-model/logs/

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos create`

在 Hub 上创建一个新的存储库。

**用法**：

```console
$ hf repos create [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--sdk, --space-sdk TEXT`：Hugging Face Spaces SDK 类型。当 --type 设置为“space”时必需。
* `--template TEXT`：根据官方模板创建空间。传递模板存储库 ID（例如“SpacesExamples/jupyterlab”）或其短名称（例如“JupyterLab”）。使用 `hf spaces templates` 列出可用模板。仅限空格。
* `--private / --no-private`: 如果 Hub 上不存在存储库，是否创建私有存储库。如果存储库已存在，则忽略。
* `--public`：是否公开仓库。如果存储库已存在，则忽略。
* `--protected`：是否对Space进行保护（仅限Spaces）。如果存储库已存在，则忽略。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--exist-ok / --no-exist-ok`：如果存储库已存在，请勿引发错误。  [默认值：不存在-确定]
* `--resource-group-id TEXT`：要在其中创建存储库的资源组。资源组仅适用于企业中心组织。
* `--region [us|eu]`：要在其中创建存储库的云区域。可以是“我们”或“欧盟”之一。需要团队计划或以上。
* `--flavor [cpu-basic|cpu-upgrade|zero-a10g|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8]`：太空硬件风格（例如“cpu-basic”、“t4-medium”、“l4x4”）。仅适用于空间。* `--storage [small|medium|large]`：（已弃用，请改用卷）空间持久存储层（“小”、“中”或“大”）。仅适用于空间。
* `--sleep-time INTEGER`：空间进入睡眠状态之前不活动的秒数。使用-1 禁用。仅适用于空间。
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `--env-file TEXT`：读入环境变量文件。
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。例如。 -v hf://org/m:/data 或 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro
* `--help`：显示此消息并退出。示例
  $ hf repos 创建我的模型
  $ hf repos create my-dataset --repo-type dataset --private
  $ hf repos create my-space --type space --sdk gradio --flavor t4-medium --secrets HF_TOKEN -e THEME=dark --protected
  $ hf repos create my-jupyterlab --type space --template SpacesExamples/jupyterlab
  $ hf repos create my-space --type space --sdk gradio -v hf://org/my-model:/models -v hf://buckets/org/b:/data
  $ hf repos create my-model --region us

了解更多
  使用 `hf <command> --help` 获取有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos delete`

从 Hub 中删除存储库。这是不可逆的操作。

**用法**：

```console
$ hf repos delete [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--missing-ok / --no-missing-ok`：如果设置为 True，则在 repo 不存在时不会引发错误。  [默认值：无缺失-确定]
* `-y, --yes`：回答“是”会自动提示。
* `--help`：显示此消息并退出。

示例
  $ hf repos 删除我的模型了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos delete-files`

从 Hub 上的存储库中删除文件。

**用法**：

```console
$ hf repos delete-files [OPTIONS] REPO_ID PATTERNS...
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `PATTERNS...`：匹配要删除的文件的全局模式。基于fnmatch，'*'递归匹配文件。  [必填]

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--commit-message TEXT`：生成的提交的摘要/标题/第一行。
* `--commit-description TEXT`：生成的提交的描述。
* `--create-pr / --no-create-pr`：是否为这些更改创建新的 Pull Request。  [默认值：no-create-pr]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf repos 删除文件 my-model file.txt
  $ hf repos 删除文件 my-model "*.json"
  $ hf repos 删除文件 my-model 文件夹/了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos duplicate`

在 Hub（模型、数据集或空间）上复制存储库。

**用法**：

```console
$ hf repos duplicate [OPTIONS] FROM_ID [TO_ID]
```

**参数**：

* `FROM_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `[TO_ID]`：目标存储库 ID（例如 `myorg/my-copy`）。默认为具有相同存储库名称的命名空间。

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--private / --no-private`: 如果 Hub 上不存在存储库，是否创建私有存储库。如果存储库已存在，则忽略。
* `--public`：是否公开仓库。如果存储库已存在，则忽略。
* `--protected`：是否对Space进行保护（仅限Spaces）。如果存储库已存在，则忽略。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--exist-ok / --no-exist-ok`：如果存储库已存在，请勿引发错误。  [默认值：不存在-确定]
* `--flavor [cpu-basic|cpu-upgrade|zero-a10g|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8]`：太空硬件风格（例如“cpu-basic”、“t4-medium”、“l4x4”）。仅适用于空间。
* `--storage [small|medium|large]`：（已弃用，请改用卷）空间持久存储层（“小”、“中”或“大”）。仅适用于空间。* `--sleep-time INTEGER`：空间进入睡眠状态之前不活动的秒数。使用-1 禁用。仅适用于空间。
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `--env-file TEXT`：读入环境变量文件。
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。例如。 -v hf://org/m:/data 或 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro
* `--help`：显示此消息并退出。

示例
  $ hf repos 重复 openai/gdpval --type 数据集
  $ hf repos 重复的 multimodalart/dreambooth-training my-dreambooth --type space --flavor l4x4 --secrets HF_TOKEN --private
  $ hf repos 重复 org/my-space my-space --type space -v hf://org/my-model:/models -v hf://buckets/org/b:/data了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos list`

列出所有存储库（模型、数据集、空间、存储桶）以及存储信息。 [别名：ls]

**用法**：

```console
$ hf repos list [OPTIONS]
```

**选项**：

* `--namespace TEXT`：组织名称。如果未提供，则列出经过身份验证的用户的存储库。
* `--type, --repo-type [model|dataset|space|bucket]`：按存储库类型（模型、数据集、空间或存储桶）过滤。
* `--search TEXT`：搜索查询。
* `--limit INTEGER`：限制结果数量。  [默认值：30]
* `--explore`：探索作为交互式 3D 城市的存储库。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 仓库 ls
  $ hf repos ls --explore
  $ hf repos ls --namespace my-org --search bert

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos move`

将存储库从一个命名空间移动到另一个命名空间。

**用法**：

```console
$ hf repos move [OPTIONS] FROM_ID TO_ID
```

**参数**：

* `FROM_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `TO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必需的]**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--help`：显示此消息并退出。

示例
  $ hf repos 移动旧命名空间/我的模型 新命名空间/我的模型

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos settings`

更新存储库的设置。

**用法**：

```console
$ hf repos settings [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：

* `--gated [auto|manual|false]`：存储库的门控状态。
* `--private / --no-private`: 如果 Hub 上不存在存储库，是否创建私有存储库。如果存储库已存在，则忽略。
* `--public`：是否公开仓库。如果存储库已存在，则忽略。
* `--protected`：是否对Space进行保护（仅限Spaces）。如果存储库已存在，则忽略。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--help`：显示此消息并退出。示例
  $ hf 存储库设置 my-model --private
  $ hf 存储库设置 my-model --gate auto
  $ hf 存储库设置 my-space --repo-type space --protected

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf repos tag`

管理 Hub 上存储库的标签。

**用法**：

```console
$ hf repos tag [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `create`：为存储库创建标签。
* `delete`：删除存储库的标签。
* `list`：列出存储库的标签。 [别名：ls]

#### `hf repos tag create`

为存储库创建标签。

**用法**：

```console
$ hf repos tag create [OPTIONS] REPO_ID TAG
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `TAG`：要创建的标签的名称。  [必填]

**选项**：

* `-m, --message TEXT`：要创建的标签的描述。
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--help`：显示此消息并退出。示例
  $ hf repos 标签创建 my-model v1.0
  $ hf repos tag create my-model v1.0 -m "First release"

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf repos tag delete`

删除存储库的标签。

**用法**：

```console
$ hf repos tag delete [OPTIONS] REPO_ID TAG
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `TAG`：要删除的标签名称。  [必填]

**选项**：

* `-y, --yes`：回答“是”则自动提示
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--help`：显示此消息并退出。

示例
  $ hf repos 标签删除 my-model v1.0

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf repos tag list`

列出存储库的标签。 [别名：ls]

**用法**：

```console
$ hf repos tag list [OPTIONS] REPO_ID
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]

**选项**：* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--help`：显示此消息并退出。

示例
  $ hf repos 标签列表 my-model

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf sandbox`

在 Hugging Face Jobs 上运行和管理沙箱。

**用法**：

```console
$ hf sandbox [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `cp`：在本地计算机和...之间复制文件
* `create`：创建沙箱：专用虚拟机...
* `exec`：在沙箱中运行命令，流式输出。
* `kill`：终止沙箱、整个共享主机……
* `pool`：主机虚拟机的暖池并廉价生成......
* `process`：列出并停止正在运行的后台进程...
* `spawn`：在...中启动长时间运行的命令

### `hf sandbox cp`

在本地计算机和沙箱（docker 样式）之间复制文件。

**用法**：

```console
$ hf sandbox cp [OPTIONS] SRC DST
```

**参数**：

* `SRC`：来源：本地路径或:.  [必填]
* `DST`: 目的地：本地路径或:.  [必填]

**选项**：* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱 cp data.csv :/data/data.csv
  $ hf 沙箱 cp :/app/result.json result.json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf sandbox create`

创建一个沙箱：默认情况下是一个专用虚拟机，或者是一个带有 `--pool` 的廉价共享虚拟机。

Env 和idle-timeout 在两种模式下都适用于沙箱。有了`--pool`，图像和
味道来自池子，所以将它们传递到这里是一个错误； `--secrets` 也是
被拒绝，因为池化沙箱没有加密秘密通道（使用`--env`）。定义
先有一个泳池，`hf sandbox pool create`。

**用法**：

```console
$ hf sandbox create [OPTIONS] [IMAGE]
```

**参数**：

* `[IMAGE]`：Docker 镜像（需要 /bin/sh）。

**选项**：* `--pool TEXT`：在此池中生成一个廉价的共享沙箱（来自`hf sandbox pool create`）。
* `--flavor [cpu-basic|cpu-upgrade|cpu-performance|cpu-xl|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8|h200|h200x2|h200x4|h200x8|rtx-pro-6000|rtx-pro-6000x2|rtx-pro-6000x4|rtx-pro-6000x8]`：硬件的味道。运行“hf jobs hardware”以列出可用的口味。默认为`cpu-basic`。
* `--idle-timeout TEXT`：在长时间不活动后自动终止沙箱（例如“10m”）。默认为 10m。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--env-file TEXT`：读入环境变量文件。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。例如。 -v hf://org/m:/data 或 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--forward-hf-token`：将您的 HF 代币作为 HF_TOKEN 注入沙箱中。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 沙箱创建
  $ hf 沙箱创建 ubuntu:24.04
  $ hf sandbox create --flavor a10g-small
  $ hf 沙箱创建 --pool pool-ab12cd34ef56 --env LOG_LEVEL=debug

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf sandbox exec`

在沙箱中运行命令，流式输出。使用命令的退出代码退出。

要在后台启动长时间运行的命令而不是等待它，请使用
`hf sandbox spawn`。

**用法**：

```console
$ hf sandbox exec [OPTIONS] SANDBOX_ID COMMAND...
```

**参数**：

* `SANDBOX_ID`：`hf sandbox create`打印的沙箱ID。  [必填]
* `COMMAND...`：要运行的命令。  [必填]

**选项**：

* `-w, --workdir TEXT`：工作目录。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `--env-file TEXT`：读入环境变量文件。
* `--timeout FLOAT`：在这么多秒后终止命令。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱执行程序 -- python -c "print(42)"
  $ hf 沙箱 exec -w /app -- pytest -x了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf sandbox kill`

终止沙箱、整个共享主机或所有内容（--all）。

**用法**：

```console
$ hf sandbox kill [OPTIONS] [SANDBOX_ID]
```

**参数**：

* `[SANDBOX_ID]`：要终止的沙箱或主机 ID。

**选项**：

* `--all`：终止命名空间中的每个沙箱和主机。
* `-y, --yes`：自动对提示回答“是”。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱杀死 
  $ hf sandbox kill # 杀死整个共享主机（其所有沙箱）
  $ hf 沙箱杀死 --all

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf sandbox pool`

主机虚拟机的热池并从中生成廉价的共享沙箱。

**用法**：

```console
$ hf sandbox pool [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：* `create`：热池：立即启动一个主机虚拟机，标记...
* `delete`：终止池中的每个主机虚拟机（并且... [别名：rm]
* `ls`：列出正在运行的沙箱池（从...分组[别名：列表]

#### `hf sandbox pool create`

预热池：立即启动一台主机虚拟机，并进行标记，以便稍后可以通过其池 ID 找到它。

**用法**：

```console
$ hf sandbox pool create [OPTIONS] [IMAGE]
```

**参数**：

* `[IMAGE]`：主机的 Docker 镜像（需要 /bin/sh）。

**选项**：

* `--flavor [cpu-basic|cpu-upgrade|cpu-performance|cpu-xl|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8|h200|h200x2|h200x4|h200x8|rtx-pro-6000|rtx-pro-6000x2|rtx-pro-6000x4|rtx-pro-6000x8]`：硬件的味道。运行“hf jobs hardware”以列出可用的口味。默认为`cpu-basic`。
* `--per-host INTEGER RANGE`：每个主机虚拟机打包的沙箱（默认 50）。  [默认值：50； x>=1]
* `--max-hosts INTEGER RANGE`：主机虚拟机数量的可选上限。  [x>=1]
* `--idle-timeout TEXT`：一旦主机在这么长时间内没有沙箱（例如“10m”），就将其关闭。默认为 10m。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱池创建
  $ hf 沙箱池创建 python:3.12 --flavor cpu-basic
  $ hf 沙箱池创建 --per-host 50 --idle-timeout 30m了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf sandbox pool delete`

终止池中的每个主机虚拟机（及其所有沙箱）。 [别名：rm]

**用法**：

```console
$ hf sandbox pool delete [OPTIONS] POOL_ID
```

**参数**：

* `POOL_ID`：要删除的池 ID。  [必填]

**选项**：

* `-y, --yes`：自动对提示回答“是”。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱池删除 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf sandbox pool ls`

列出正在运行的沙箱池（根据其主机虚拟机分组）。 [别名：列表]

**用法**：

```console
$ hf sandbox pool ls [OPTIONS]
```

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱池 ls了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf sandbox process`

列出并停止沙箱中运行的后台进程。

**用法**：

```console
$ hf sandbox process [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `kill`：停止在...中运行的后台进程
* `ls`: 列出... [别名: list] 中运行的后台进程

#### `hf sandbox process kill`

停止沙箱中运行的后台进程。

**用法**：

```console
$ hf sandbox process kill [OPTIONS] SANDBOX_ID PID
```

**参数**：

* `SANDBOX_ID`：`hf sandbox create`打印的沙箱ID。  [必填]
* `PID`：`hf sandbox process ls`打印的pid。  [必填]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱进程终止  

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf sandbox process ls`

列出沙箱中运行的后台进程（以`hf sandbox spawn`开头）。 [别名：列表]

**用法**：```console
$ hf sandbox process ls [OPTIONS] SANDBOX_ID
```

**参数**：

* `SANDBOX_ID`：`hf sandbox create`打印的沙箱ID。  [必填]

**选项**：

* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱进程 ls 

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf sandbox spawn`

在后台启动一个长时间运行的命令并返回其 pid（无需等待）。

使用 `hf sandbox process ls` 列出沙箱的进程并使用以下命令停止一个进程
`hf sandbox process kill`。

**用法**：

```console
$ hf sandbox spawn [OPTIONS] SANDBOX_ID COMMAND...
```

**参数**：

* `SANDBOX_ID`：`hf sandbox create`打印的沙箱ID。  [必填]
* `COMMAND...`：后台运行的命令。  [必填]

**选项**：* `-w, --workdir TEXT`：工作目录。
* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `--env-file TEXT`：读入环境变量文件。
* `--namespace TEXT`：作业将运行的命名空间。默认为当前用户的命名空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 沙箱生成 -- python -m http.server 8000
  $ hf sandbox spawn -w /app -- uvicorn app:app

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf skills`

管理人工智能助手的技能。

**用法**：

```console
$ hf skills [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `add`：为AI安装抱脸技能...
* `list`：列出拥抱中可用的技能... [别名：ls]
* `preview`: 将生成的`hf-cli` SKILL.md打印到...
* `update`：更新已安装的 Hugging Face 市场...

### `hf skills add`

为AI助手安装抱脸技能。默认的`hf-cli`技能是从安装的CLI版本本地生成的；
其他技能可以从 Hugging Face 市场下载。
默认位置位于当前目录 (.agents/skills) 或用户级别 (~/.agents/skills)。
如果指定了`--claude`，该技能也会符号链接到 Claude 的旧技能目录中。

**用法**：

```console
$ hf skills add [OPTIONS] [NAME]
```

**参数**：

* `[NAME]`：市场技能名称。

**选项**：

* `--claude`：为克劳德安装。
* `-g, --global`：全局安装（用户级）而不是安装在当前项目目录中。
* `--dest PATH`：安装到自定义目标（技能目录的路径）。
* `--force`：覆盖目的地已有的技能。
* `--help`：显示此消息并退出。

示例
  $ hf 技能添加
  $ hf 技能添加 Huggingface-gradio --dest=~/my-skills
  $ hf 技能添加 --global
  $ hf 技能添加 --claude
  $ hf 技能添加 Huggingface-gradio --claude --global

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf skills list`

列出 Hugging Face 市场中的可用技能。 [别名：ls]

**用法**：

```console
$ hf skills list [OPTIONS]
```

**选项**：* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ 高频技能列表
  $ hf 技能列表 --format json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf skills preview`

将生成的 `hf-cli` SKILL.md 打印到标准输出。

**用法**：

```console
$ hf skills preview [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

### `hf skills update`

更新已安装的 Hugging Face 市场技能。

**用法**：

```console
$ hf skills update [OPTIONS] [NAME]
```

**参数**：

* `[NAME]`：可选安装的技能名称进行更新。

**选项**：

* `--claude`：更新为克劳德安装的技能。
* `-g, --global`：使用全局技能目录代替当前项目。
* `--dest PATH`：更新自定义技能目录中的技能。
* `--help`：显示此消息并退出。

示例
  $ 高频技能更新
  $ hf 技能更新 hf-cli
  $ hf 技能更新 Huggingface-gradio --dest=~/my-skills
  $ hf 技能更新 --claude

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf spaces`

与 Hub 上的空间互动。**用法**：

```console
$ hf spaces [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `card`：获取空间卡（自述文件）...
* `dev-mode`：在空间上启用或禁用开发模式。
* `hardware`：列出 Spaces 的可用硬件选项。
* `hot-reload`：热重载 Space 的任何 Python 文件...
* `info`：获取有关 Hub 上空间的信息。
* `list`：列出 Hub 上的空间，或... [别名：ls]
* `logs`：获取Space的运行或构建日志。
* `pause`：暂停一个空格。
* `restart`：重启空间。
* `search`：使用语义搜索 Hub 上的空间...
* `secrets`：管理 Hub 上空间的机密。
* `settings`：更新空间的设置。
* `ssh`：通过 SSH 连接到 Space 的开发模式容器。
* `templates`：列出可用的空间模板。
* `variables`：管理空间的环境变量...
* `volumes`：管理 Hub 上空间的卷。
* `wait`：等待空间完成构建/启动。

### `hf spaces card`

获取 Hub 上空间的空间卡（自述文件）。

**用法**：

```console
$ hf spaces card [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：* `--metadata`：仅输出卡中的元数据。
* `--text`：仅输出文本正文（无元数据）。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间卡 mteb/排行榜
  $ hf 空间卡 mteb/leaderboard --metadata
  $ hf 空格卡 mteb/leaderboard --metadata --format json
  $ hf 空格卡 mteb/leaderboard --text

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces dev-mode`

在空间上启用或禁用开发模式。

Spaces 开发模式可简化应用程序的调试，并允许您更快地迭代 Spaces
重新启动您的应用程序而不停止 Space 容器本身。此功能作为 PRO 的一部分提供
或团队和企业计划。

请参阅文档：https://huggingface.co/docs/hub/spaces-dev-mode

**用法**：

```console
$ hf spaces dev-mode [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：* `--stop / --no-stop`：停止开发模式。  [默认：不间断]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间 开发模式 我的用户名/deepsite

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces hardware`

列出 Spaces 的可用硬件选项。

**用法**：

```console
$ hf spaces hardware [OPTIONS]
```

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间硬件

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces hot-reload`

热重新加载空间的任何 Python 文件，无需完全重建 + 重新启动。

⚠ 此功能是实验性的 ⚠

仅适用于 Gradio SDK (6.1+)
打开交互式编辑器，除非指定了 --local-file/-f 。此命令使用 https://github.com/breuleux/jurigged 修补实时 Python 进程
（基于 AST 的差异、就地功能更新等），与 Gradio 的本机热重载支持集成
（意味着 Gradio 演示对象的更改反映在 UI 中）

该命令创建远程提交。
如果您使用本地克隆，请随后运行 `git pull --autostash`
恢复提交并保持本地 git 状态同步。

**用法**：

```console
$ hf spaces hot-reload [OPTIONS] SPACE_ID [FILENAME]
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]
* `[FILENAME]`：Space 存储库中 Python 文件的路径。当指定 --local-file 并且存储库中的路径匹配时可以省略。

**选项**：

* `-f, --local-file PATH`：本地文件路径。交互式编辑器模式（如果未指定）
* `--skip-checks / --no-skip-checks`：跳过热重载兼容性检查。  [默认值：不跳过检查]
* `--skip-summary / --no-skip-summary`: 触发热重载后跳过摘要显示[默认: no-skip-summary]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf space hot-reload username/repo-name app.py # 打开远程 app.py 文件的交互式编辑器
  $ hf space hot-reload username/repo-name -f app.py # 从 ./app.py 获取本地版本并远程修补 app.py
  $ hf space hot-reload username/repo-name app.py -f src/app.py # 从 ./src/app.py 获取本地版本

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces info`

获取有关 Hub 上空间的信息。

**用法**：

```console
$ hf spaces info [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--expand TEXT`：要返回的逗号分隔属性。使用时，仅返回列出的属性（和 id）。示例：“--expand=likes,tags”。有效：作者、cardData、createdAt、数据集、disabled、lastModified、likes、models、private、region、resourceGroup、runtime、sdk、sha、siblings、subdomain、tags、trendingScore、usedStorage。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 空间信息 enzostvs/deepsite
  $ hf 空间信息 gradio/theme_builder --expand sdk、运行时、喜欢

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces list`

列出 Hub 上的空间或空间存储库中的文件。 [别名：ls]

当不带参数调用时，列出集线器上的空格。
当使用空间 ID 调用时，列出该空间存储库中的文件。

**用法**：

```console
$ hf spaces list [OPTIONS] [REPO_ID]
```

**参数**：

* `[REPO_ID]`：用于列出文件的空间 ID（例如 `username/repo-name`）。如果省略，则列出空格。

**选项**：* `--search TEXT`：搜索查询。
* `--author TEXT`：按作者或组织过滤。
* `--filter TEXT`：按标签过滤（例如“文本分类”）。可以多次使用。
* `--sort [created_at|last_modified|likes|trending_score]`：对结果进行排序。
* `--limit INTEGER`：限制结果数量。  [默认值：30]
* `--expand TEXT`：要返回的逗号分隔属性。使用时，仅返回列出的属性（和 id）。示例：“--expand=likes,tags”。有效：作者、cardData、createdAt、数据集、disabled、lastModified、likes、models、private、region、resourceGroup、runtime、sdk、sha、siblings、subdomain、tags、trendingScore、usedStorage。
* `-h, --human-readable`：以人类可读的格式显示大小（仅适用于列出文件）。
* `--tree`：以树形格式列出文件（仅用于列出文件）。
* `-R, --recursive`：递归列出文件（仅用于列出文件）。
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空格 ls --limit 10
  $ hf space ls --search "chatbot" --author Huggingface
  $ hf 空间 ls victor/deepsite
  $ hf 空间 ls victor/deepsite -R
  $ hf 空间 ls victor/deepsite --tree -h了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces logs`

获取空间的运行或构建日志。

默认情况下，打印当前可用的运行日志并退出（非阻塞，例如
`docker logs`）。使用 --follow/-f 进行流式传输，直到服务器关闭流式传输。
使用 --build 来查看容器构建日志（当空间是
陷入 BUILD_ERROR）。

**用法**：

```console
$ hf spaces logs [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--build`：获取容器构建日志而不是运行日志。当 Space 陷入 BUILD_ERROR 时很有用。
* `-f, --follow`：跟随日志输出（流式传输直到服务器关闭流）。如果没有此标志，则仅打印当前可用的日志。
* `-n, --tail INTEGER`：从日志末尾开始显示的行数。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf space 记录用户名/我的空间
  $ hf space 记录用户名/my-space --build
  $ hf 空间日志 -f 用户名/我的空间
  $ hf 空间日志 -n 50 用户名/我的空间了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces pause`

暂停一个空格。

**用法**：

```console
$ hf spaces pause [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空格 暂停用户名/我的空间

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces restart`

重新启动空间。

**用法**：

```console
$ hf spaces restart [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--factory-reboot`：不使用构建缓存从头开始重建空间。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf space 重新启动用户名/我的空间
  $ hf space restart 用户名/我的空间 --factory-reboot

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档### `hf spaces search`

使用语义搜索在 Hub 上搜索空间。

**用法**：

```console
$ hf spaces search [OPTIONS] QUERY
```

**参数**：

* `QUERY`：搜索查询。  [必填]

**选项**：

* `--filter TEXT`：按标签过滤（例如“文本分类”）。可以多次使用。
* `--sdk TEXT`：按 SDK 过滤（例如 gradio、docker、static）。
* `--include-non-running / --no-include-non-running`：结果中包含非运行空格。  [默认值：不包含非运行]
* `--description / --no-description`：显示AI生成的描述。  [默认值：无描述]
* `--limit INTEGER`：限制结果数量。  [默认值：10]
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空格搜索“生成图像”
  $ hf space search“识别图片中的对象”--sdk gradio --limit 5
  $ hf space search“从照片中删除背景”--description --json

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces secrets`

管理 Hub 上空间的机密。

**用法**：

```console
$ hf spaces secrets [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：* `add`：添加或更新空间的秘密。
* `delete`：从空间中删除秘密。
* `list`：列出空间的秘密。 [别名：ls]

#### `hf spaces secrets add`

添加或更新空间的机密。

**用法**：

```console
$ hf spaces secrets add [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `-s, --secrets TEXT`：设置秘密环境变量。例如。 --secrets SECRET=value 或 `--secrets HF_TOKEN` 传递您的 Hugging Face 令牌。
* `--secrets-file TEXT`：读入秘密环境变量文件。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间秘密添加用户名/我的空间 -s HF_TOKEN
  $ hf 空间秘密添加用户名/我的空间 -s OPENAI_API_KEY=sk-... -s ANTHROPIC_API_KEY=sk-...
  $ hf 空间秘密添加用户名/我的空间 --secrets-file .env.secrets

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf spaces secrets delete`

从空间中删除秘密。

**用法**：

```console
$ hf spaces secrets delete [OPTIONS] SPACE_ID KEY
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]
* `KEY`：要删除的秘密的名称。  [必填]

**选项**：* `-y, --yes`：回答“是”会自动提示。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf Spaces Secrets 删除用户名/我的空间 HF_TOKEN
  $ hf Spaces Secrets 删除用户名/我的空间 HF_TOKEN --yes

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf spaces secrets list`

列出空间的秘密。秘密值是只写且不返回的。 [别名：ls]

**用法**：

```console
$ hf spaces secrets list [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间秘密 ls 用户名/我的空间

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces settings`

更新空间的设置。

**用法**：

```console
$ hf spaces settings [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：* `--sleep-time INTEGER`：空间进入睡眠状态之前的空闲时间（以秒为单位）。使用 -1 表示从不休眠。仅适用于升级的硬件。
* `--hardware [cpu-basic|cpu-upgrade|zero-a10g|t4-small|t4-medium|l4x1|l4x4|l40sx1|l40sx4|l40sx8|a10g-small|a10g-large|a10g-largex2|a10g-largex4|a100-large|a100x4|a100x8]`：太空硬件风格（例如“cpu-basic”、“t4-medium”、“l4x4”）。运行“hf space hardware”以列出可用选项。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间设置用户名/我的空间 --sleep-time 300
  $ hf 空间设置用户名/我的空间 --hardware t4-medium

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces ssh`

通过 SSH 连接到 Space 的开发模式容器。

需要在空间上运行开发模式，并在 https://huggingface.co/settings/keys 上注册您的 SSH 公钥。

请参阅：https://huggingface.co/docs/hub/spaces-dev-mode

**用法**：

```console
$ hf spaces ssh [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：* `-i, --identity-file PATH`：SSH 身份文件的路径（转发到`ssh -i`）。
* `--dry-run`：打印 SSH 命令而不是运行它。
* `--auto`：如果尚未启用，则启用开发模式而不提示。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf space ssh 用户名/我的空间
  $ hf space ssh 用户名/my-space --dry-run
  $ hf 空格 ssh 用户名/my-space -i ~/.ssh/id_ed25519
  $ hf space ssh 用户名/my-space --auto

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces templates`

列出可用的空间模板。

模板的`repo_id`（或`name`）可以传递给`hf repos create --template ...`来
从该模板创建一个新空间。

**用法**：

```console
$ hf spaces templates [OPTIONS]
```

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空格模板

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces variables`管理 Hub 上空间的环境变量。

**用法**：

```console
$ hf spaces variables [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `add`：添加或更新环境变量...
* `delete`：从空间中删除环境变量。
* `list`：列出空间的环境变量。 [别名：ls]

#### `hf spaces variables add`

添加或更新空间的环境变量。

**用法**：

```console
$ hf spaces variables add [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `-e, --env TEXT`：设置环境变量。例如。 --env ENV=值
* `--env-file TEXT`：读入环境变量文件。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空格变量添加用户名/我的空间 -e DEBUG=1
  $ hf 空间变量添加用户名/我的空间 -e MODEL_ID=gpt2 -e MAX_TOKENS=512
  $ hf 空间变量添加用户名/我的空间 --env-file .env

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf spaces variables delete`

从空间中删除环境变量。

**用法**：

```console
$ hf spaces variables delete [OPTIONS] SPACE_ID KEY
```

**参数**：* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]
* `KEY`：要删除的变量的名称。  [必填]

**选项**：

* `-y, --yes`：回答“是”会自动提示。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf space 变量删除用户名/my-space DEBUG
  $ hf 空间变量删除用户名/我的空间 DEBUG --yes

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf spaces variables list`

列出空间的环境变量。 [别名：ls]

**用法**：

```console
$ hf spaces variables list [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空格变量 ls 用户名/我的空间

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces volumes`

管理 Hub 上空间的卷。

**用法**：

```console
$ hf spaces volumes [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。**命令**：

* `delete`：从空间中删除所有卷。
* `list`：列出空间中安装的卷。 [别名：ls]
* `set`：设置（替换）空间的卷。

#### `hf spaces volumes delete`

从空间中删除所有卷。

**用法**：

```console
$ hf spaces volumes delete [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `-y, --yes`：回答“是”会自动提示。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间卷 删除用户名/我的空间
  $ hf 空间卷删除用户名/我的空间 --yes

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf spaces volumes list`

列出空间中安装的卷。 [别名：ls]

**用法**：

```console
$ hf spaces volumes list [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间卷 ls 用户名/我的空间了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

#### `hf spaces volumes set`

设置（替换）空间的卷。

**用法**：

```console
$ hf spaces volumes set [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `-v, --volume TEXT`：安装一个或多个卷。格式：hf://[TYPE/]SOURCE:/MOUNT_PATH[:ro]。 TYPE 是以下之一：模型、数据集、空间、存储桶。如果省略，TYPE 默认为型号。模型、数据集和空间始终以只读方式安装。默认情况下，存储桶是读+写的。例如。 -v hf://org/m:/data 或 -v hf://datasets/org/ds:/data 或 -v hf://buckets/org/b:/mnt:ro
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf 空间卷设置用户名/my-space -v hf://models/username/my-model:/models
  $ hf 空间卷集 username/my-space -v hf://buckets/username/my-bucket:/data -v hf://datasets/username/my-dataset:/datasets:ro

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf spaces wait`等待空间完成构建/启动。

阻塞直到空间离开中间阶段（BUILDING、APP_STARTING 等）
并达到一个稳定的阶段。如果空间正在运行，则退出并显示代码 0，
或非零退出代码（例如 BUILD_ERROR、RUNTIME_ERROR）。

**用法**：

```console
$ hf spaces wait [OPTIONS] SPACE_ID
```

**参数**：

* `SPACE_ID`：空间ID（例如`username/repo-name`）。  [必填]

**选项**：

* `--timeout TEXT`：最长等待时间：带 s（秒，默认）、m（分钟）、h（小时）或 d（天）的 int。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf space 等待用户名/我的空间
  $ hf space 等待用户名/我的空间 --timeout 5m

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf sync`

在本地目录和存储桶之间同步文件。

**用法**：

```console
$ hf sync [OPTIONS] [SOURCE] [DEST]
```

**参数**：

* `[SOURCE]`: 源路径：本地目录或 hf://buckets/namespace/bucket_name(/prefix)
* `[DEST]`: 目标路径：本地目录或 hf://buckets/namespace/bucket_name(/prefix)

**选项**：* `--delete / --no-delete`：删除源中不存在的目标文件。  [默认：不删除]
* `--ignore-times`：仅根据大小跳过文件，忽略修改时间。
* `--ignore-sizes`：仅根据修改时间跳过文件，忽略大小。
* `--plan TEXT`：将同步计划保存到 JSONL 文件以供审核而不是执行。
* `--apply TEXT`：应用之前保存的计划文件。
* `--dry-run`：将同步计划作为 JSONL 打印到标准输出而不执行。
* `--include TEXT`：包含匹配模式的文件（可以指定多个）。
* `--exclude TEXT`：排除匹配模式的文件（可以指定多个）。
* `--filter-from TEXT`：从文件中读取包含/排除模式。
* `--existing`：跳过在接收器上创建新文件（仅更新现有文件）。
* `--ignore-existing`：跳过更新接收器上存在的文件（仅创建新文件）。
* `-v, --verbose`：显示带有推理的详细日志记录。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

## `hf update`

将`hf` CLI 更新到最新版本。

**用法**：

```console
$ hf update [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

## `hf upload`

将文件或文件夹上传到集线器。建议用于单次提交上传。

**用法**：

```console
$ hf upload [OPTIONS] REPO_ID [LOCAL_PATH] [PATH_IN_REPO]
```

**参数**：* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `[LOCAL_PATH]`：要上传的文件或文件夹的本地路径。支持通配符模式。默认为当前目录。
* `[PATH_IN_REPO]`：存储库中文件或文件夹的路径。默认为文件或文件夹的相对路径。

**选项**：

* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认值：（型号）]
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--private / --no-private`: 如果 Hub 上不存在存储库，是否创建私有存储库。如果存储库已存在，则忽略。
* `--include TEXT`：匹配要上传的文件的全局模式。
* `--exclude TEXT`：从要上传的文件中排除的全局模式。
* `--delete TEXT`：提交时从存储库中删除的文件的全局模式。
* `--commit-message TEXT`：生成的提交的摘要/标题/第一行。
* `--commit-description TEXT`：生成的提交的描述。
* `--create-pr / --no-create-pr`：是否将内容作为新的 Pull Request 上传。  [默认值：no-create-pr]
* `--every FLOAT`：如果设置，则计划后台作业每`every` 分钟创建提交。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf 上传 my-cool-model 。 。
  $ hf upload Wauplin/my-cool-model ./models/model.safetensors
  $ hf upload Wauplin/my-cool-dataset ./data /train --repo-type=dataset
  $ hf upload Wauplin/my-cool-model ./models 。 --commit-message="Epoch 34/50" --commit-description="Val 准确度：68%"
  $ hf upload bigcode/the-stack 。 。 --repo-type 数据集 --create-pr

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf upload-large-folder`

[已弃用] 将大文件夹上传到集线器。请使用 `hf upload` 代替。

**用法**：

```console
$ hf upload-large-folder [OPTIONS] REPO_ID LOCAL_PATH
```

**参数**：

* `REPO_ID`：存储库的 ID（例如 `username/repo-name` 或 `spaces/username/repo-name`）。  [必填]
* `LOCAL_PATH`：要上传的文件夹的本地路径。  [必填]

**选项**：* `--type, --repo-type [model|dataset|space]`：存储库的类型（模型、数据集或空间）。  [默认：型号]
* `--revision TEXT`：Git 修订 ID，可以是分支名称、标签或提交哈希。
* `--private / --no-private`: 如果 Hub 上不存在存储库，是否创建私有存储库。如果存储库已存在，则忽略。
* `--include TEXT`：匹配要上传的文件的全局模式。
* `--exclude TEXT`：从要上传的文件中排除的全局模式。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--num-workers INTEGER`：用于散列、上传和提交文件的工作人员数量。
* `--no-report / --no-no-report`：是否关闭定期状态报告。  [默认值：不报告]
* `--no-bars / --no-no-bars`: 是否禁用进度条。  [默认值：无栏]
* `--help`：显示此消息并退出。

示例
  $ hf 上传大文件夹 Wauplin/my-cool-model ./large_model_dir
  $ hf upload-large-folder Wauplin/my-cool-model ./large_model_dir --revision v1.0

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

## `hf version`

打印有关 hf 版本的信息。

**用法**：

```console
$ hf version [OPTIONS]
```

**选项**：

* `--help`：显示此消息并退出。

## `hf webhooks`管理 Hub 上的 Webhook。

**用法**：

```console
$ hf webhooks [OPTIONS] COMMAND [ARGS]...
```

**选项**：

* `--help`：显示此消息并退出。

**命令**：

* `create`：创建一个新的 webhook。
* `delete`：永久删除 webhook。
* `disable`：禁用活动的 webhook。
* `enable`：启用禁用的 webhook。
* `info`：显示单个 webhook 的完整详细信息。
* `list`：列出当前用户的所有 webhook。 [别名：ls]
* `update`：更新现有的 webhook。

### `hf webhooks create`

创建一个新的网络钩子。

提供 --url（用于 ping 远程服务器）或 --job-id（用于触发作业），但不能同时提供两者。

**用法**：

```console
$ hf webhooks create [OPTIONS]
```

**选项**：

* `--watch TEXT`：要观看的项目，采用“类型：名称”格式（例如“型号：bert-base-uncased”）。可重复。  [必填]
* `--url TEXT`：将 Webhook 负载发送到的 URL。与 --job-id 互斥。
* `--job-id TEXT`：要触发的作业 ID（来自 job.id），而不是 ping URL。与 --url 互斥。
* `--domain [repo|discussions]`：要观看的域：“repo”或“discussions”。可重复。默认为所有域。
* `--secret TEXT`：用于签署 Webhook 有效负载的可选密钥。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。示例
  $ hf webhooks create --url https://example.com/hook --watch model:bert-base-uncased
  $ hf webhooks create --url https://example.com/hook --watch org:HuggingFace --watch model:gpt2 --domain repo
  $ hf webhooks create --job-id 687f911eaea852de79c4a50a --watch 用户：julien-c

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf webhooks delete`

永久删除 Webhook。

**用法**：

```console
$ hf webhooks delete [OPTIONS] WEBHOOK_ID
```

**参数**：

* `WEBHOOK_ID`：要删除的Webhook的ID。  [必填]

**选项**：

* `-y, --yes`：跳过确认提示。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf webhooks 删除 abc123
  $ hf webhooks 删除 abc123 --yes

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf webhooks disable`

禁用活动的 Webhook。

**用法**：

```console
$ hf webhooks disable [OPTIONS] WEBHOOK_ID
```

**参数**：

* `WEBHOOK_ID`：要禁用的 webhook 的 ID。  [必填]

**选项**：* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf webhooks 禁用 abc123

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf webhooks enable`

启用已禁用的 Webhook。

**用法**：

```console
$ hf webhooks enable [OPTIONS] WEBHOOK_ID
```

**参数**：

* `WEBHOOK_ID`：要启用的 webhook 的 ID。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf webhooks 启用 abc123

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf webhooks info`

显示单个 Webhook 的完整详细信息。

**用法**：

```console
$ hf webhooks info [OPTIONS] WEBHOOK_ID
```

**参数**：

* `WEBHOOK_ID`：webhook 的 ID。  [必填]

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf webhooks 信息 abc123了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf webhooks list`

列出当前用户的所有 Webhook。 [别名：ls]

**用法**：

```console
$ hf webhooks list [OPTIONS]
```

**选项**：

* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf webhooks ls
  $ hf webhooks ls --format json
  $ hf webhooks ls --format 安静

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### `hf webhooks update`

更新现有的 Webhook。仅更改提供的选项。

**用法**：

```console
$ hf webhooks update [OPTIONS] WEBHOOK_ID
```

**参数**：

* `WEBHOOK_ID`：要更新的 webhook 的 ID。  [必填]

**选项**：* `--url TEXT`：用于发送 Webhook 负载的新 URL。
* `--watch TEXT`：要观看的新项目列表，采用“类型：名称”格式。可重复。替换整个现有的监视列表。
* `--domain [repo|discussions]`：要观看的新域名列表：“repo”或“discussions”。可重复。
* `--secret TEXT`：用于签署 Webhook 有效负载的新密钥。
* `--token TEXT`：从 https://huggingface.co/settings/tokens 生成的用户访问令牌。
* `--help`：显示此消息并退出。

示例
  $ hf webhooks update abc123 --url https://new-url.com/hook
  $ hf webhooks update abc123 --watch model:gpt2 --domain repo
  $ hf webhooks update abc123 --secret newsecret

了解更多
  使用 `hf <command> --help` 了解有关命令的更多信息。
  阅读 https://huggingface.co/docs/huggingface_hub/en/guides/cli 上的文档

### 与讨论和 Pull 请求交互
https://huggingface.co/docs/huggingface_hub/v1.29.0/package_reference/community.md