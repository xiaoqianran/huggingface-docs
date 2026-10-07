<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 遥测

从版本 1.7.0 开始，`hf_xet` Python 包和`hf-xet` Rust 箱在每次上传和下载后都会向 Hub 发送一个简短的报告。我们使用这些报告来查看传输失败的频率以及跨 `hf_xet` 版本、操作系统和网络的传输速度，以便我们可以发现并解决问题。

Git Xet 0.2.1 及更早版本不发送遥测数据。

## 收集什么

每份报告都描述一次传输。在一次提交中上传多个文件是一次传输，一次下载多个文件也是一次传输。|数据|详情 |
|---|---|
|传输类型和结果 |上传或下载，以及是否成功、失败或被取消。对于故障，一般错误类别，例如 `network`、`timeout` 或 `server_error`。 |
|时间 |转移花了多长时间。对于上传，还包括分块和上传数据所花费的时间以及完成提交所花费的时间。 |
|尺寸|文件数、文件总大小以及通过网络发送的字节数。 |
|重复数据删除 |对于上传，有多少数据已存储在集线器上并被跳过，有多少是新数据，以及新数据的压缩效果如何。 |
|速度|平均吞吐量和使用的最大并行请求数。 |
|客户| `hf_xet` 版本、操作系统、CPU 架构、CPU 数量和用户代理字符串。这是 `huggingface_hub` 随每个请求发送到集线器的同一个用户代理，其中包含调用库的名称和版本（例如 `transformers`）以及 `huggingface_hub`、Python 和 PyTorch 的版本。 |
|身份证 |传输的随机 ID、`hf_xet` 会话的随机 ID 以及处理传输的 Xet 存储服务器的主机名。 |报告不包括文件名、文件路径、文件内容、文件哈希、存储库名称或您的用户名。

报告使用与传输相同的访问令牌发送到 Xet 存储服务器。当服务器存储报告时，它会添加您的 IP 地址、令牌所属的 Hub 帐户（如果您未登录，则为“匿名”帐户）、颁发令牌的存储库以及令牌是否允许读取或写入访问。

## 报告何时发送

`hf_xet` 在上传或下载完成时发送一份报告，无论是否成功。运行时间超过 5 分钟的传输还会每 5 分钟发送一次进度报告，直至完成。

报告大约 1 KB。它们与传输一起发送，不会减慢传输速度。如果无法发送报告，该报告将被丢弃并且不会重试。传输完成后，`hf_xet` 最多等待 2 秒才能发出最后一份报告，因此即使您的程序立即退出，报告也不会丢失。

## 选择退出

遥测默认处于开启状态。要关闭它，请设置`HF_HUB_DISABLE_TELEMETRY=1`：

```bash
export HF_HUB_DISABLE_TELEMETRY=1
```

这与关闭 `huggingface_hub` 和其他 Hugging Face Python 库中的遥测功能相同。请参阅 `huggingface_hub` 文档中的 [⟦T16⟧](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisabletelemetry)。当`DO_NOT_TRACK`、`DISABLE_TELEMETRY`、`HF_HUB_OFFLINE`或`TRANSFORMERS_OFFLINE`设置为`1`、`true`、`yes`或`on`时，`hf_xet`也会关闭遥测。

要仅关闭`hf_xet`的遥测，请设置`HF_XET_TELEMETRY_ENABLED=0`。

|环境变量 |默认 |描述 |
|---|---|---|
| `HF_HUB_DISABLE_TELEMETRY` |取消设置 |设置为 `1` 以关闭 `hf_xet` 和其他 Hugging Face Python 库中的遥测。 |
| `HF_XET_TELEMETRY_ENABLED` | `true` |设置为 `0` 仅关闭 `hf_xet` 遥测。将其设置为 `1` 不会覆盖上面的选择退出变量。 |
| `HF_XET_TELEMETRY_HEARTBEAT_AFTER` | `300s` |传输运行多长时间后才开始发送进度报告。 `0` 关闭进度报告。 |
| `HF_XET_TELEMETRY_HEARTBEAT_INTERVAL` | `300s` |进度报告之间的时间。 |
| `HF_XET_TELEMETRY_FINAL_FLUSH_TIMEOUT` | `2s` |传输完成后`hf_xet`等待最后报告的时间。 `0` 表示不等待。 |

### 使用 Xet 存储
https://huggingface.co/docs/hub/xet/using-xet-storage.md