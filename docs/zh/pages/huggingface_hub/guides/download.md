<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 从 Hub 下载文件

`huggingface_hub`库提供从存储库下载文件的功能
存储在集线器上。您可以独立使用这些功能，也可以将它们集成到您的
自己的库，让您的用户更方便地与 Hub 交互。这个
指南将向您展示如何：

* 下载并缓存单个文件。
* 下载并缓存整个存储库。
* 下载文件到本地文件夹。

## 下载单个文件

[hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)功能是从Hub下载文件的主要功能。
它下载远程文件，将其缓存在磁盘上（以版本感知的方式），然后返回其本地文件路径。

> [!提示]
> 返回的文件路径是指向 HF 本地缓存的指针。因此，重要的是不要修改该文件，以避免
> 缓存已损坏。如果您有兴趣了解有关文件如何缓存的更多信息，请参阅我们的
> [caching guide](./manage-cache)。

### 从最新版本开始

使用 `repo_id`、`repo_type` 和 `filename` 参数选择要下载的文件。默认情况下，该文件将
被视为 `model` 存储库的一部分。

```python
>>> from huggingface_hub import hf_hub_download
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json")
'/root/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade/config.json'

# Download from a dataset
>>> hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
'/root/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34/fleurs.py'
```

### 从特定版本开始默认情况下，会下载 `main` 分支的最新版本。但是，在某些情况下您想要下载文件
在特定版本（例如，来自特定分支、PR、标签或提交哈希）。
为此，请使用 `revision` 参数：

```python
# Download from the `v1.0` tag
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="v1.0")

# Download from the `test-branch` branch
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="test-branch")

# Download from Pull Request #3
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="refs/pr/3")

# Download from a specific commit hash
>>> hf_hub_download(repo_id="lysandre/arxiv-nlp", filename="config.json", revision="877b84a8f93f2d619faa2a6e514a32beef88ab0a")
```

**注意：** 使用提交哈希时，它必须是全长哈希，而不是 7 个字符的提交哈希。

### 构建下载URL

如果您想构造用于从存储库下载文件的 URL，您可以使用 [hf_hub_url()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_url) 返回 URL。
请注意，它是由[hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)内部使用的。

## 下载整个存储库

[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 下载给定版本的整个存储库。它内部使用[hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)
意味着所有下载的文件也缓存在您的本地磁盘上。同时进行下载以加快进程。

要下载整个存储库，只需传递 `repo_id` 和 `repo_type`：

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp")
'/home/lysandre/.cache/huggingface/hub/models--lysandre--arxiv-nlp/snapshots/894a9adde21d9a3e3843e6d5aeaaf01875c7fade'

# Or from a dataset
>>> snapshot_download(repo_id="google/fleurs", repo_type="dataset")
'/home/lysandre/.cache/huggingface/hub/datasets--google--fleurs/snapshots/199e4ae37915137c555b1765c01477c216287d34'
```

[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)默认下载最新版本。如果您想要特定的存储库修订版，请使用
`revision`参数：

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", revision="refs/pr/1")
```

### 过滤要下载的文件[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 提供了一种下载存储库的简单方法。然而，您并不总是想下载
存储库的全部内容。例如，您可能希望阻止下载所有 `.bin` 文件（如果您知道会这样做）
仅使用 `.safetensors` 权重。您可以使用 `allow_patterns` 和 `ignore_patterns` 参数来做到这一点。

这些参数接受单个模式或模式列表。模式是标准通配符（通配符
模式）如记录的[here](https://tldp.org/LDP/GNU-Linux-Tools-Summary/html/x11655.htm)。模式匹配是
基于[⟦T29⟧](https://docs.python.org/3/library/fnmatch.html)。

例如，您可以使用 `allow_patterns` 仅下载 JSON 配置文件：

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", allow_patterns="*.json")
```

另一方面，`ignore_patterns`可以排除某些文件的下载。的
以下示例忽略 `.msgpack` 和 `.h5` 文件扩展名：

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="lysandre/arxiv-nlp", ignore_patterns=["*.msgpack", "*.h5"])
```

最后，您可以将两者结合起来以精确过滤您的下载。下面是下载所有json和markdown的示例
除 `vocab.json` 之外的文件。

```python
>>> from huggingface_hub import snapshot_download
>>> snapshot_download(repo_id="gpt2", allow_patterns=["*.md", "*.json"], ignore_patterns="vocab.json")
```

## 下载文件到本地文件夹

默认情况下，我们建议使用[cache system](./manage-cache)从Hub下载文件。您可以使用[hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download)和[snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download)中的`cache_dir`参数或通过设置[⟦T36⟧](../package_reference/environment_variables#hf_home)环境变量来指定自定义缓存位置。但是，如果您需要将文件下载到特定文件夹，可以将 `local_dir` 参数传递给下载函数。这对于使工作流程更接近 `git` 命令提供的功能很有用。下载的文件将在指定文件夹中保留其原始文件结构。例如，如果 `filename="data/train.csv"` 和 `local_dir="path/to/folder"`，则生成的文件路径将为 `"path/to/folder/data/train.csv"`。

将在本地目录的根目录下创建一个 `.cache/huggingface/` 文件夹，其中包含有关下载文件的元数据。如果文件已经是最新的，这可以防止重新下载文件。如果元数据已更改，则下载新的文件版本。这使得`local_dir`针对仅拉取最新更改进行了优化。

下载完成后，如果不再需要`.cache/huggingface/`文件夹，您可以安全地删除它。但是，请注意，在没有此文件夹的情况下重新运行脚本可能会导致恢复时间更长，因为元数据将会丢失。请放心，您的本地数据将保持完整且不受影响。

> [!提示]
> 向 Hub 提交更改时，不必担心 `.cache/huggingface/` 文件夹！ `git` 和 [upload_folder()](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder) 都会自动忽略此文件夹。

## 从 CLI 下载您可以从终端使用`hf download`命令直接从Hub下载文件。
在内部，它使用与上述相同的 [hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 和 [snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 帮助器并打印
返回终端的路径。

```bash
>>> hf download gpt2 config.json
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/config.json
```

您可以一次下载多个文件，它会显示进度条并返回文件所在的快照路径
位于：

```bash
>>> hf download gpt2 config.json model.safetensors
Fetching 2 files: 100%|████████████████████████████████████████████| 2/2 [00:00<00:00, 23831.27it/s]
/home/wauplin/.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10
```

您还可以使用单个 `hf://` URI 指向存储库（以及可选的修订版和文件）。 URI 遵循语法 `hf://[<TYPE>/]<ID>[@<REVISION>][/<PATH>]`（请参阅 [HF URIs reference](../package_reference/hf_uris) 了解完整语法）并替换 `--repo-type` 和 `--revision` 选项，这些选项不能与它一起设置：

```bash
# Download a single file from a dataset at a given revision
>>> hf download hf://datasets/google/fleurs@refs/pr/1/fleurs.py

# Download a subfolder (note the trailing slash)
>>> hf download hf://datasets/google/fleurs/data/

# Download an entire repo
>>> hf download hf://datasets/google/fleurs
```

更多关于CLI下载命令的详细信息，请参考[CLI guide](./cli#hf-download)。

## 试运行模式

在某些情况下，您希望在实际下载之前检查将下载哪些文件。您可以使用 `--dry-run` 参数进行检查。它列出了存储库上要下载的所有文件，并检查它们是否已下载。这可以了解必须下载的文件数量及其大小。

这是一个检查单个文件的示例：

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 1 files (out of 1) totalling 655.2M
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx 655.2M
```

如果文件已经被缓存：

```sh
>>> hf download openai-community/gpt2 onnx/decoder_model_merged.onnx --dry-run
[dry-run] Will download 0 files (out of 1) totalling 0.0.
File                           Bytes to download
------------------------------ -----------------
onnx/decoder_model_merged.onnx -
```

您还可以对整个存储库执行试运行：

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
```以及文件过滤：

```sh
>>> hf download openai-community/gpt2 --include "*.json"  --dry-run
[dry-run] Fetching 11 files: 100%|█████████████| 11/11 [00:00<00:00, 80518.92it/s]
[dry-run] Will download 0 files (out of 11) totalling 0.0.
File                         Bytes to download
---------------------------- -----------------
config.json                  -
generation_config.json       -
onnx/config.json             -
onnx/generation_config.json  -
onnx/special_tokens_map.json -
onnx/tokenizer.json          -
onnx/tokenizer_config.json   -
onnx/vocab.json              -
tokenizer.json               -
tokenizer_config.json        -
vocab.json                   -
```

最后，您还可以通过将 `dry_run=True` 传递给 [hf_hub_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.hf_hub_download) 和 [snapshot_download()](/docs/huggingface_hub/v1.29.0/en/package_reference/file_download#huggingface_hub.snapshot_download) 以编程方式进行试运行。它将返回一个 [DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) （分别是 [DryRunFileInfo](/docs/huggingface_hub/v1.29.0/en/package_reference/hf_api#huggingface_hub.DryRunFileInfo) 的列表），其中包含每个文件的提交哈希、文件名和文件大小、文件是否被缓存以及文件是否会被下载。实际上，如果未缓存或通过了`force_download=True`，则将下载该文件。

## 更快的下载

通过 `hf_xet` 利用更快的下载速度，`hf_xet` 是与 [⟦T56⟧](https://github.com/huggingface/xet-core) 库的 Python 绑定，可实现 
基于块的重复数据删除可加快下载和上传速度。 `hf_xet` 与 `huggingface_hub` 无缝集成，但使用 Rust `xet-core` 库和 Xet 存储而不是 LFS。`hf_xet` 使用 Xet 存储系统，该系统将文件分解为不可变的块，远程存储这些块的集合（称为块或 xorb），并在请求时检索它们以重新组合文件。下载时，在确认用户有权访问文件后，`hf_xet`将使用该文件的LFS SHA256哈希查询Xet内容可寻址服务（CAS），以接收重建元数据（xorbs内的范围）来组装这些文件，以及预签名的URL以直接下载xorbs。然后`hf_xet`将有效地下载所需的xorb范围并将文件写到磁盘上。

要启用它，只需安装最新版本的`huggingface_hub`：

```bash
pip install -U "huggingface_hub"
```

从 `huggingface_hub` 0.32.0 开始，这也将安装 `hf_xet`。

所有其他 `huggingface_hub` API 将继续工作，无需任何修改。要了解有关 Xet 存储和 `hf_xet` 的优势的更多信息，请参阅此 [section](https://huggingface.co/docs/hub/xet/index)。

注意：`hf_transfer`以前与LFS存储后端一起使用，现已弃用；使用 `hf_xet` 代替。

### 操作指南
https://huggingface.co/docs/huggingface_hub/v1.29.0/guides/overview.md