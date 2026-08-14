<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 下载数据集

## 集成库

如果集线器上的数据集绑定到[supported library](./datasets-libraries)，则只需几行即可加载数据集。有关访问数据集的信息，您可以单击数据集页面上的“使用此数据集”按钮以了解如何执行此操作。例如，[⟦T5⟧](https://huggingface.co/datasets/knkarthick/samsum?library=datasets) 展示了如何使用下面的 `datasets` 执行此操作。

## 使用 Hugging Face 客户端库

您可以使用 [⟦T7⟧](/docs/huggingface_hub) 库来创建、删除、更新和检索存储库中的信息。例如，要从命令行下载`HuggingFaceH4/ultrachat_200k`数据集，请运行

```bash
hf download HuggingFaceH4/ultrachat_200k --repo-type dataset
```

请参阅 [HF CLI download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-a-dataset-or-a-space) 了解更多信息。

您还可以将其集成到您自己的库中！例如，您可以使用 Pandas 快速加载包含几行的 CSV 数据集。
```py
from huggingface_hub import hf_hub_download
import pandas as pd

REPO_ID = "YOUR_REPO_ID"
FILENAME = "data.csv"

dataset = pd.read_csv(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset")
)
```

## 使用 Git

由于 Hub 上的所有数据集都是 Xet 支持的 Git 存储库，因此您可以通过 [installing git-xet](./xet/using-xet-storage#git-xet) 在本地克隆数据集并运行：

```bash
git xet install
git lfs install
git clone git@hf.co:datasets/<dataset ID> # example: git clone git@hf.co:datasets/allenai/c4
```

如果您拥有对特定数据集存储库的写入访问权限，您还可以提交并推送对数据集的修订。

将您的 SSH 公钥添加到 [your user settings](https://huggingface.co/settings/keys) 以推送更改和/或访问私有存储库。

## 更快的下载您可以从 Hugging Face 的 CDN [fast.hf.co](https://fast.hf.co) 测试您的下载速度。这会针对最近的 HF 边缘服务器运行快速带宽测试，帮助您了解基线吞吐量。您还可以使用 [⟦T9⟧](https://github.com/julien-c/hf-speedtest) CLI 扩展测量终端的下载速度：

```bash
hf extensions install julien-c/hf-speedtest
hf speedtest
```

为了实现更快的传输，集线器使用具有自适应并发功能的[Xet storage backend](https://huggingface.co/docs/hub/en/xet/index)，可根据网络条件自动调整并行流。有关 `HF_XET_HIGH_PERFORMANCE=1` 等调整选项的详细信息，请参阅 [Faster downloads for models](./models-downloading#faster-downloads)。

## 使用 hf-mount

对于大型数据集，您可以使用 [hf-mount](https://github.com/huggingface/hf-mount) 将存储库安装为本地文件系统，而不是下载完整的存储库。文件是延迟获取的——只有代码读取的字节才会到达网络。当您的工作流程需要本地文件路径（例如 `tarfile`、`zipfile`、`imagefolder`）而不是 Python 迭代器时很有用。

```bash
brew install hf-mount
hf-mount start repo datasets/stanfordnlp/imdb /tmp/imdb
```

存储库以只读方式安装。有关完整设置详细信息、后端选项和缓存，请参阅[Mount as a Local Filesystem](./storage-buckets-access#mount-as-a-local-filesystem)。

## 在代理或防火墙后面下载如果您的网络通过防火墙或代理限制出站流量，则下载数据集需要的不仅仅是`huggingface.co`。文件内容由单独的存储和 CDN 主机名提供，如果无法访问这些内容，即使 `huggingface.co` 本身已列入白名单，`load_dataset` / `hf download` 也会失败。

将以下主机名列入白名单（全部通过 HTTPS/端口 443）：

|主机名 |目的|
|------------------------------------------|--------------------------------------------------------|
| `huggingface.co` | Hub API、元数据和下载重定向 |
| `cas-server.xethub.hf.co` | Xet 存储协议 API + 上传（美国）|
| `cas-server.xethub-eu.hf.co` | Xet 存储协议 API + 上传（欧盟）|
| `transfer.xethub.hf.co` | Xet 存储下载 API（美国）|
| `transfer.xethub-eu.hf.co` | Xet 存储下载 API（欧盟）|
| `us.aws.cdn.hf.co` | CDN 边缘（美国）|
| `us.gcp.cdn.hf.co` | CDN 边缘（美国）|
| `cdn-lfs-us-1.hf.co` | LFS CDN（美国）|
| `cdn-lfs-eu-1.hf.co` | LFS CDN（欧盟）|> [!提示]
> 下载遵循从 `huggingface.co` 到这些主机名的 HTTP 重定向，因此
> 仅将 `huggingface.co` 列入白名单是不够的。 A`ReadTimeoutError`（而不是
> 连接错误）下载中途通常意味着初始连接
> 成功，但存储或 CDN 主机被阻止。

> [!提示]
> 通配符行为取决于您的代理如何匹配域。多家企业代理
> 将允许列表条目视为覆盖任意深度子域的后缀匹配。如果你的
> 确实如此，最简单的选择是将后缀 `hf.co` 和 `huggingface.co` 列入白名单 —
> 这些涵盖当前和未来的每个存储和 CDN 端点。
>
> 如果您的代理仅支持单标签通配符（其中 `*.hf.co` 匹配
> `cdn-lfs-us-1.hf.co` 但不是更深的 `us.aws.cdn.hf.co` 或 `cas-server.xethub.hf.co`），
> 将上表中的显式主机名列入白名单。请注意，`*.xethub.hf.co`确实
> 不涵盖`xethub-eu.hf.co`下的欧盟主机，`*.cdn.hf.co`不涵盖
> 双标签`us.aws.cdn.hf.co` / `us.gcp.cdn.hf.co`。

> [!警告]
> 随着我们的存储和 CDN 基础设施的发展，这些主机名可能会发生变化。你在哪里
> 安全策略允许，将 `hf.co` 和 `huggingface.co` 后缀列入白名单（所有
> 子域），这样当特定端点发生变化时，您的规则就不会被破坏。### 空间小组
https://huggingface.co/docs/hub/spaces-sdks-docker-panel.md