<!-- huggingface-docs: machine-translated zh-CN from English source -->

# HF URI

*HF URI* 是一个类似 URI 的字符串，用于标识 Hugging Face Hub 上的位置。在整个库和 CLI 中，`hf://...` 字符串用于指向：

- 模型、数据集、空间或内核存储库（可以选择固定在修订版上）；
- 此类存储库内的文件或子文件夹；
- [bucket](../guides/buckets) 或存储桶内的子文件夹。

*HF 安装* 使用本地安装路径和可选的 `:ro` / `:rw` 标志包装 HF URI，由 [Spaces](../guides/manage-spaces) 和 [Jobs](../guides/jobs) 卷使用。

本页记录了 HF URI 和 HF 安装的规范语法。库中的任何地方都使用相同的解析器，因此在一个上下文中有效的 URI（例如 [HfFileSystem](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_file_system#huggingface_hub.HfFileSystem)）在另一个上下文中进行相同的解析。

[parse_hf_uri()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.parse_hf_uri) 还接受 Hugging Face **网络 URL**（例如 `https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv`），因此您可以从网站复制粘贴链接。请参阅下面的[Web URLs](#web-urls)。

## HF URI 语法

```text
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
```|组件|必填 |允许值 |
| ---------------- | -------- | ---------------------------------------------------------------------------------- |
| `hf://` |是的 |文字协议前缀。                                            |
| `<TYPE>/` |没有| `models/`、`datasets/`、`spaces/`、`kernels/`、`buckets/`（复数）。 |
| `<ID>` |是的 | `<namespace>/<name>` |
| `@<REVISION>` |没有|分支、标记、提交 SHA 或特殊引用（`refs/pr/N`、`refs/convert/...`）。仅回购。 |
| `/<PATH>` |没有|存储库或存储桶内的路径。                                     |

## HF 安装语法

```text
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

挂载是一个 HF URI，后跟 `:<MOUNT_PATH>` 和可选的 `:ro` / `:rw` 标志。

|组件|必填 |允许值 |
| ---------------- | -------- | ---------------------------------------------------------------------------------- |
| `<MOUNT_PATH>` |是的 |绝对安装路径（必须以`/`开头）。                          |
| `:ro` / `:rw` |没有|只读/读写标志。                                        |

## 什么是 HF URI以下是**所有有效的 HF URI**：

```text
# Models (type prefix is optional, but the id is always 'namespace/name')
hf://my-org/my-model                            # implicit type prefix
hf://models/my-org/my-model                     # explicit type prefix
hf://models/my-org/my-model/config.json         # file inside a model repo
hf://models/my-org/my-model@v1.0/config.json    # pinned to a revision

# Datasets, Spaces, Kernels (type prefix is required)
hf://datasets/my-org/my-dataset
hf://datasets/my-org/my-dataset@dev/train.csv
hf://spaces/my-user/my-space
hf://kernels/my-org/my-kernel

# Special revisions (preserved as-is)
hf://datasets/my-org/my-dataset@refs/pr/10/data.csv
hf://datasets/my-org/my-dataset@refs/convert/parquet/data.parquet

# Buckets (always 'namespace/name', no revision)
hf://buckets/my-org/my-bucket
hf://buckets/my-org/my-bucket/sub/folder
```

以下是**有效的 HF 安装**（体积规格）：

```text
hf://my-org/my-model:/data
hf://datasets/my-org/my-dataset:/mnt:ro
hf://datasets/my-org/my-dataset/train:/mnt:rw    # mount a sub-folder
hf://buckets/my-org/my-bucket:/storage:rw
```

## 什么是**不是** HF URI

解析器是故意严格的。以下内容被**拒绝**：

|无效的 URI |原因 |
| ------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `my-org/my-model`、`./local/path` |缺少 `hf://` 前缀且不是可识别的拥抱脸部 URL。           |
| `hf://dataset/org/m`、`hf://model/org/m` |禁止使用单数类型形式，请使用复数 (`datasets/`, ...)。   |
| `hf://datasets`、`hf://buckets/` |单独的类型前缀不是有效的 URI，需要 `<ID>`。          |
| `hf://gpt2`、`hf://datasets/squad` |不支持规范存储库（没有命名空间）。                |
| `hf://buckets/single-segment` |铲斗必须始终为 `namespace/name`。                                |
| `hf://buckets/org/b@v1` |存储桶不支持修订标记。                               |
| `hf://org/m@`、`hf://datasets/foo/bar@/x` | `@`之后的空修订。                                               || `hf://a/b/c@v1` |存储库 ID 必须是`namespace/name`，额外的段是路径。           |
| `hf://org/m:/` |挂载路径必须是非空绝对路径。                           |

## 网址

为了方便起见，[parse_hf_uri()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.parse_hf_uri) 还接受您从浏览器复制粘贴的 Hugging Face Web URL。它们被标准化为规范的 `hf://` 形式，因此您可以将 URL 直接粘贴到 CLI 或库中，然后“它就可以工作”：

```python
>>> from huggingface_hub import parse_hf_uri
>>> parse_hf_uri("https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv")
HfUri(type='dataset', id='my-org/my-dataset', revision='main', path_in_repo='train.csv')
```

来自 `huggingface.co`（及其 `hf.co` 短域、登台主机和自定义 `HF_ENDPOINT` 的主机）的 URL 均可识别，无论是否带有 `https://` 方案。查询字符串 (`?download=true`) 和片段 (`#L10`) 将被忽略。

要解析来自另一个 Hub 部署（例如自托管或代理 Hub）的 URL，请将其基本 URL 作为 `endpoint` 传递。然后，除了默认主机之外，还会识别其主机，并在解析之前删除路径前缀（例如 `http://localhost:8080/hf`）：

```python
>>> parse_hf_uri("https://hub.my-company.com/datasets/my-org/my-dataset", endpoint="https://hub.my-company.com")
HfUri(type='dataset', id='my-org/my-dataset', revision=None, path_in_repo='')
```

### 支持的 URL 格式

下表列出了支持的路由。为了简洁起见，仅显示**URL路径**（主机后面的部分，例如`huggingface.co`）；隐含了公认的主机。|指向|网址路径|
| -------------------------------------------------------------------------- | ---------------------------------------------------------- |
|模型库| `/<ns>/<name>` |
|数据集存储库（`spaces/`、`kernels/`、`models/`相同）| `/datasets/<ns>/<name>` |
|存储库内的文件夹，固定在 `<rev>` | `/<ns>/<name>/tree/<rev>[/<path>]` |
|存储库内的文件（文件查看器路径）| `/<ns>/<name>/blob/<rev>/<path>` |
|存储库内的文件（下载路径）| `/<ns>/<name>/resolve/<rev>/<path>` |
|存储库内的文件（原始路径）| `/<ns>/<name>/raw/<rev>/<path>` |
|存储库内的文件（归咎路径）| `/<ns>/<name>/blame/<rev>/<path>` |
|桶| `/buckets/<ns>/<name>` |
|存储桶内的文件（存储桶没有版本控制）| `/buckets/<ns>/<name>/resolve/<path>` |
|桶内的文件夹| `/buckets/<ns>/<name>/tree/<path>` |该修订取自`blob`/`resolve`/`raw`/`tree`/`blame`之后的单个片段。特殊引用（`refs/pr/N`、`refs/convert/...`）会被急切地匹配，即使它们包含`/`；任何其他包含 `/` 的分支/标签名称必须经过 URL 编码 (`feature%2Ffoo`)。

### 未解析的 URL

当 URL 不明确或未指向具体的 Hub 位置时，它会被**拒绝**（永远不会被猜测）：

下面的路径再次显示没有主机，除了最后一行，其中主机本身是拒绝的原因。|原因 |网址路径|
| -------------------------------------------------------------------------------- | --------------------------------------- |
|单段 URL：用户/组织页面、列表页面或规范存储库 — 不明确。 | `/<username>` |
|列表页面，没有`<ns>/<name>`。                                                  | `/datasets` |
|不支持规范存储库（没有命名空间）。                         | `/gpt2` |
|不是文件/文件夹位置（与 `commits`、`discussions`、`settings`、... 相同）。   | `/<ns>/<name>/commit/<rev>` |
|集合不是存储库。                                                | `/collections/<ns>/<slug>` |
|主持人不是公认的 Hugging Face 主持人。                                      | `https://example.com/<ns>/<name>` |

## 渲染 URL

[HfUri.to_url()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.HfUri.to_url) 是解析 URL 的逆过程：它呈现 HF URI 的可浏览 Web URL。

```python
>>> uri = parse_hf_uri("hf://datasets/my-org/my-dataset@v1/train.csv")
>>> uri.to_url()
'https://huggingface.co/datasets/my-org/my-dataset/blob/v1/train.csv'
```当未设置路径或修订时，它指向存储库/存储桶登录页面；仅设置修订时，它指向文件夹查看器（`/tree/<rev>`）；对于存储库文件，它指向文件查看器（`/blob/<rev>/<path>`，修订版本默认为`main`）；对于存储桶文件，它指向树路由（`/tree/<path>`）。路径中的特殊字符（空格、`#`、...）采用百分比编码。传递 `endpoint=...` 来定位自定义主机。

## Python 解析

### 解析 URI

[parse_hf_uri()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.parse_hf_uri) 是集中式 URI 解析器。它是一个纯字符串解析器（无网络调用）并返回冻结的 [HfUri](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.HfUri) 数据类。

```python
>>> from huggingface_hub import parse_hf_uri
>>> parse_hf_uri("hf://datasets/my-org/my-dataset@refs/pr/3/train.json")
HfUri(type='dataset', id='my-org/my-dataset', revision='refs/pr/3', path_in_repo='train.json')
```

[HfUri](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.HfUri) 可通过 [HfUri.to_uri()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.HfUri.to_uri) 进行往返，它始终发出规范形式（带有显式类型前缀）：

```python
>>> uri = parse_hf_uri("hf://my-org/my-model@v1/config.json")
>>> uri.to_uri()
'hf://models/my-org/my-model@v1/config.json'
```

直接使用 `type` 和 `id` 字段。布尔属性 `is_repo` 和 `is_bucket` 在需要时消除存储库 URI 和存储桶 URI 之间的歧义。

### 解析挂载

`parse_hf_mount` 解析挂载规范（具有本地挂载路径和可选 `:ro`/`:rw` 标志的 HF URI）并返回冻结的 `HfMount` 数据类。它在底层使用[parse_hf_uri()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.parse_hf_uri)。

```python
>>> from huggingface_hub import parse_hf_mount
>>> parse_hf_mount("hf://buckets/my-org/my-bucket/sub/dir:/mnt:ro")
HfMount(source=HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir'), mount_path='/mnt', read_only=True)
```

`HfMount` 可通过 `HfMount.to_uri` 往返：

```python
>>> mount = parse_hf_mount("hf://my-org/my-model:/data:ro")
>>> mount.to_uri()
'hf://models/my-org/my-model:/data:ro'
```

## 参考[[huggingface_hub.HfUri]]

#### Huggingface_hub.HfUri[[huggingface_hub.HfUri]]

```python
huggingface_hub.HfUri(type: typing.Literal['model', 'dataset', 'space', 'kernel', 'bucket'], id: str, revision: str | None = None, path_in_repo: str = '', _raw: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_hf_uris.py#L79)

**参数：**type (`str`) ：“模型”、“数据集”、“空间”、“内核”或“存储桶”之一。

id (`str`) ：repo URI 的存储库 id（'namespace/name'，例如 'my-org/my-model'），或存储桶 URI 的存储桶 id（'namespace/name'）。

revision (`str`, *可选*) : URI 中“@”后指定的修订版本，已 URL 解码。如果未指定修订版，或者对于存储桶 URI（从不携带修订版），则为“无”。像“refs/pr/10”和“refs/convert/parquet”这样的特殊参考将按原样保留。

path_in_repo (`str`) ：存储库或存储桶内的路径。如果 URI 指向根，则为空字符串。

Hugging Face Hub URI 的解析表示（'hf://...'）。

#### to_uri[[huggingface_hub.HfUri.to_uri]]

```python
to_uri()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_hf_uris.py#L137)

将 URI 呈现为规范的“hf://”字符串。

类型前缀始终显式写入（例如“hf://models/my-org/my-model”）。

#### to_url[[huggingface_hub.HfUri.to_url]]

```python
to_url(endpoint: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_hf_uris.py#L155)

**参数：**

端点（`str`，*可选*）：要使用的基本端点。默认为“constants.ENDPOINT”（即“https://huggingface.co”）。

**返回：** `str`

网址。

将 URI 呈现为 Hugging Face **web URL**（您在浏览器中打开的那种）。这是使用 [parse_hf_uri()](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.parse_hf_uri) 解析 URL 的逆过程。返回的 URL 指向：

- 未设置路径或修订时的存储库/存储桶登陆页面；
- 仅设置修订版时的文件夹查看器（'/tree/'）；
- 存储库文件的文件查看器（'/blob//'）（修订版默认为'main'）；
- 存储桶文件的树路由（'/tree/'）（存储桶没有版本控制）。

示例：
```py
>>> from huggingface_hub import parse_hf_uri
>>> parse_hf_uri("hf://datasets/my-org/my-dataset@v1/train.csv").to_url()
'https://huggingface.co/datasets/my-org/my-dataset/blob/v1/train.csv'
```

#### Huggingface_hub.parse_hf_uri[[huggingface_hub.parse_hf_uri]]

```python
huggingface_hub.parse_hf_uri(uri: str, endpoint: str | None = None)
```

**参数：**

uri (`str`) ：要解析的 URI。必须以“hf://”开头，或者是 Hugging Face URL（例如“https://huggingface.co/...”）。

端点（`str`，*可选*）：自定义集线器端点（例如自托管或代理集线器，如“https://hub.my-company.com”或“http://localhost:8080/hf”）。提供后，除了默认的 Hugging Face 主机之外，还会识别该端点上的 Web URL。对“hf://”URI 没有影响。

**返回：** [HfUri](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.HfUri)

解析后的 URI。

**加薪：** `HfUriError`

- `HfUriError` -- 
  如果 URI 格式错误（缺少前缀、无效类型、缺少 id、不支持的 URL 路由等）。

解析 Hugging Face Hub URI ('hf://...') 或 Hugging Face Web URL。HF URI 是一个类似 URI 的字符串，用于标识 Hugging Face Hub 上的位置。完整的语法是：

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]
```

为了方便起见，Hugging Face **网址**（您从网站复制粘贴的网址）也
接受并标准化为规范的“hf://”形式，例如
'https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv'。仅明确的 URL
（存储库/存储桶页面和文件/文件夹查看器路由）被接受；任何其他路线都会被拒绝。

有关完整规范，请参阅“docs/source/en/package_reference/hf_uris.md”。

示例：
```py
>>> from huggingface_hub.utils import parse_hf_uri
>>> parse_hf_uri("hf://my-org/my-model")
HfUri(type='model', id='my-org/my-model', revision=None, path_in_repo='')
>>> parse_hf_uri("hf://datasets/my-org/my-dataset@refs/pr/3/train.json")
HfUri(type='dataset', id='my-org/my-dataset', revision='refs/pr/3', path_in_repo='train.json')
>>> parse_hf_uri("https://huggingface.co/datasets/my-org/my-dataset/blob/main/train.csv")
HfUri(type='dataset', id='my-org/my-dataset', revision='main', path_in_repo='train.csv')
```

#### Huggingface_hub.utils.HfMount[[huggingface_hub.utils.HfMount]]

```python
huggingface_hub.utils.HfMount(source: HfUri, mount_path: str, read_only: bool | None = None, _raw: str | None = None)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_hf_uris.py#L207)

**参数：**

source ([HfUri](/docs/huggingface_hub/v1.27.0/en/package_reference/hf_uris#huggingface_hub.HfUri)) ：解析后的 HF URI，标识要挂载的集线器资源。

mount_path (`str`) ：本地挂载路径（始终以“/”开头）。

read_only (`bool`, *可选*) ：如果挂载以 ':ro' 结尾则为 True，如果以 ':rw' 结尾则为 False，如果未提供标志则为 'None'。

HF URI 与本地安装路径和可选的只读标志配对。

由空间和作业用来描述卷安装。完整的语法是：

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

#### to_uri[[huggingface_hub.utils.HfMount.to_uri]]

```python
to_uri()
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_hf_uris.py#L238)

将安装渲染为规范的“hf://”字符串。示例：'hf://models/my-org/my-model:/data:ro'

#### Huggingface_hub.utils.parse_hf_mount[[huggingface_hub.utils.parse_hf_mount]]

```python
huggingface_hub.utils.parse_hf_mount(mount_str: str)
```

[Source](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/src/huggingface_hub/utils/_hf_uris.py#L443)

**参数：**

mount_str (`str`) ：要解析的挂载字符串。必须以“hf://”开头并包含“:”段。

**返回：** `HfMount`

解析后的挂载。

**加薪：** `HfUriError`

- `HfUriError` -- 
  如果挂载字符串格式错误（缺少挂载路径、无效 URI 等）。

解析 HF 安装规范 ('hf://...:[:ro|:rw]')。

安装规范是一个 HF URI，后跟一个本地安装路径和一个可选的只读/读写标志。

完整的语法是：

```
hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]:<MOUNT_PATH>[:ro|:rw]
```

有关完整规范，请参阅“docs/source/en/package_reference/hf_uris.md”。

示例：
```py
>>> from huggingface_hub.utils import parse_hf_mount
>>> parse_hf_mount("hf://my-org/my-model:/data:ro")
HfMount(source=HfUri(type='model', id='my-org/my-model', revision=None, path_in_repo=''), mount_path='/data', read_only=True)
>>> parse_hf_mount("hf://buckets/my-org/my-bucket/sub/dir:/mnt:rw")
HfMount(source=HfUri(type='bucket', id='my-org/my-bucket', revision=None, path_in_repo='sub/dir'), mount_path='/mnt', read_only=False)
```

### 存储卡
https://huggingface.co/docs/huggingface_hub/v1.27.0/package_reference/cards.md