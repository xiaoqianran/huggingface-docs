<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 下载模型

## 集成库

如果 Hub 上的模型与 [supported library](./models-libraries) 绑定，只需几行即可加载模型。有关访问模型的信息，您可以单击模型页面上的“在_Library_中使用_”按钮来查看如何操作。例如，`distilbert/distilgpt2`展示了如何使用下面的🤗变形金刚来做到这一点。

## 使用 Hugging Face 客户端库

您可以使用 [⟦T7⟧](https://github.com/huggingface/huggingface_hub) 库来创建、删除、更新和检索存储库中的信息。例如，要从命令行下载`HuggingFaceH4/zephyr-7b-beta`模型，请运行
```bash
hf download HuggingFaceH4/zephyr-7b-beta
```
请参阅[CLI download documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli#download-an-entire-repository)了解更多信息。

您还可以将其集成到您自己的库中。例如，您可以用几行代码快速加载 Scikit-learn 模型。
```py
from huggingface_hub import hf_hub_download
import joblib

REPO_ID = "YOUR_REPO_ID"
FILENAME = "sklearn_model.joblib"

model = joblib.load(
    hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
)
```

## 使用 Git

由于模型中心上的所有模型都是 Xet 支持的 Git 存储库，因此您可以通过 [installing git-xet](./xet/using-xet-storage#git-xet) 在本地克隆模型并运行：

```bash
git xet install
git lfs install
git clone git@hf.co:<MODEL ID> # example: git clone git@hf.co:bigscience/bloom
```

如果您拥有对特定模型存储库的写入访问权限，您还可以提交并推送对模型的修订。

将您的 SSH 公钥添加到 [your user settings](https://huggingface.co/settings/keys) 以推送更改和/或访问私有存储库。

## 更快的下载

### 测试你的下载速度您可以从 Hugging Face 的 CDN [fast.hf.co](https://fast.hf.co) 测试您的下载速度。这会针对最近的 HF 边缘服务器运行快速带宽测试，帮助您在调整任何设置之前了解基线吞吐量。

您还可以使用 [⟦T9⟧](https://github.com/julien-c/hf-speedtest) CLI 扩展直接从终端测量下载速度：

```bash
hf extensions install julien-c/hf-speedtest
hf speedtest
```

### hf_xet 的自适应并发

`hf_xet` 是一个基于 Rust 的软件包，利用 [Xet storage backend](https://huggingface.co/docs/hub/en/xet/index) 通过基于块的重复数据删除来优化文件传输。默认情况下，`hf_xet`使用**自适应并发** - 它根据实时网络条件自动调整并行传输流的数量，保守地启动（1个流）并在带宽允许的情况下扩展到64个并发流。

对于大多数机器（包括数据中心环境），默认设置已经使可用网络带宽饱和。对于使用具有高带宽**和至少 64 GB RAM** 的计算机的高级用户，`HF_XET_HIGH_PERFORMANCE=1` 提高了并发范围并显着增加了内存缓冲区大小，这在并行下载许多大文件时会有所帮助。

```bash
HF_XET_HIGH_PERFORMANCE=1 hf download ...
```

## 使用 hf-mount对于大型模型，您可以使用 [hf-mount](https://github.com/huggingface/hf-mount) 将存储库安装为本地文件系统，而不是下载完整的存储库。文件是延迟获取的——只有代码读取的字节才会到达网络。

```bash
brew install hf-mount
hf-mount start repo openai-community/gpt2 /tmp/gpt2
```

存储库以只读方式安装。有关完整设置详细信息、后端选项和缓存，请参阅[Mount as a Local Filesystem](./storage-buckets-access#mount-as-a-local-filesystem)。

## 在代理或防火墙后面下载

如果您的网络通过防火墙或代理限制出站流量，则下载模型和数据集需要的不仅仅是 `huggingface.co`。文件内容由单独的存储和 CDN 主机名提供，如果无法访问，`from_pretrained` / `hf download` 将失败，即使 `huggingface.co` 本身已列入白名单。

将以下主机名列入白名单（全部通过 HTTPS/端口 443）：|主机名 |目的|
|------------------------------------------|--------------------------------------------------------|
| `huggingface.co` | Hub API、元数据和下载重定向 |
| `cas-server.xethub.hf.co` | Xet 存储协议 API + 上传（美国）|
| `cas-server.xethub-eu.hf.co` | Xet 存储协议 API + 上传（欧盟）|
| `transfer.xethub.hf.co` | Xet 存储下载 API（美国）|
| `transfer.xethub-eu.hf.co` | Xet 存储下载 API（欧盟）|
| `us.aws.cdn.hf.co` | CDN 边缘（美国）|
| `us.gcp.cdn.hf.co` | CDN 边缘（美国）|
| `cdn-lfs-us-1.hf.co` | LFS CDN（美国）|
| `cdn-lfs-eu-1.hf.co` | LFS CDN（欧盟）|

> [!提示]
> 下载遵循从 `huggingface.co` 到这些主机名的 HTTP 重定向，因此
> 仅将 `huggingface.co` 列入白名单是不够的。 A `ReadTimeoutError`（而不是
> 连接错误）下载中途通常意味着初始连接
> 成功，但存储或 CDN 主机被阻止。> [!提示]
> 通配符行为取决于您的代理如何匹配域。多家企业代理
> 将允许列表条目视为覆盖任意深度子域的后缀匹配。如果你的
> 确实如此，最简单的选择是将后缀 `hf.co` 和 `huggingface.co` 列入白名单 —
> 这些涵盖当前和未来的每个存储和 CDN 端点。
>
> 如果您的代理仅支持单标签通配符（其中 `*.hf.co` 匹配
> `cdn-lfs-us-1.hf.co` 但不是更深的 `us.aws.cdn.hf.co` 或 `cas-server.xethub.hf.co`），
> 将上表中的显式主机名列入白名单。请注意 `*.xethub.hf.co` 确实
> 不涵盖`xethub-eu.hf.co`下的欧盟主机，`*.cdn.hf.co`不涵盖
> 双标签`us.aws.cdn.hf.co` / `us.gcp.cdn.hf.co`。

> [!警告]
> 随着我们的存储和 CDN 基础设施的发展，这些主机名可能会发生变化。你在哪里
> 安全策略允许，将 `hf.co` 和 `huggingface.co` 后缀列入白名单（所有
> 子域），这样当特定端点发生变化时，您的规则就不会被破坏。

### 机器学习文档工具的概况
https://huggingface.co/docs/hub/model-card-landscape-analysis.md