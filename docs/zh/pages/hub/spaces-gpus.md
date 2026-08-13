<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 使用 GPU 空间

您可以使用空间顶部导航栏中的_设置_按钮升级您的空间以使用 GPU 加速器。如果您正在为业余项目构建一个很酷的演示，您甚至可以请求免费升级！

> [!提示]
> 从长远来看，我们还希望公开非 GPU 硬件，例如 HPU、IPU 或 TPU。如果您想在特定的 AI 硬件上运行，请告知我们（网站：huggingface.co）。

一旦您的 Space 在 GPU 上运行，您就可以直接从此徽章查看它正在运行的硬件：

## 硬件规格

在下表中，您可以查看不同升级选项的规格。

###CPU

| **硬件** | **CPU** | **内存** | **GPU 内存** | **磁盘** | **每小时价格** |
|------------------------ |-------------- |------------- |---------------- |------------------------ | ----------------- |
| CPU基础| 2 个虚拟CPU | 16GB|  - | 50GB|自由的！             |
| CPU升级 | 8 个 vCPU | 32GB|  - | 50GB| 0.03 美元 |> [!注意]
> CPU Basic 没有每小时成本，但创建在计算（Gradio 或 Docker）上运行的新空间需要付费计划。静态空间对所有人免费。详情请参阅[Spaces Overview](./spaces-overview#hardware-resources)。

### GPU

| **硬件** | **CPU** | **内存** | **GPU 内存** | **磁盘** | **每小时价格** |
|------------------------ |-------------- |------------- |---------------- |------------------------ | ----------------- |
| Nvidia T4 - 小| 4 个虚拟CPU | 15GB| 16GB| 50GB| 0.40 美元 |
| Nvidia T4 - 中 | 8 个 vCPU | 30GB| 16GB| 100GB| 0.60 美元 |
| 1x Nvidia L4 | 8 个 vCPU | 30GB| 24GB| 400GB| 0.80 美元 |
| 4x Nvidia L4 | 48 个 vCPU | 186 GB | 186 GB 96GB | 3200 GB | 3200 GB 3.80 美元 |
| 1x Nvidia L40S | 8 个 vCPU | 62GB| 48GB| 380GB| 1.80 美元 |
| 4x Nvidia L40S | 48 个 vCPU | 382GB| 192 GB | 192 GB 3200 GB | 3200 GB 8.30 美元 |
| 8x Nvidia L40S | 192 个 vCPU | 1534GB| 384GB| 6500 GB | 6500 GB 23.50 美元 || Nvidia A10G - 小| 4 个虚拟CPU | 15GB| 24GB| 110 GB | 110 GB 1.00 美元 |
| Nvidia A10G - 大| 12 个 vCPU | 46GB| 24GB| 200GB| 1.50 美元 |
| 2x Nvidia A10G - 大 | 24 个 vCPU | 92GB| 48GB| 1000 GB | $3.00 |
| 4x Nvidia A10G - 大 | 48 个 vCPU | 184 GB | 184 GB 96GB | 2000 GB | 5.00 美元 |
| Nvidia A100 - 大| 12 个 vCPU | 142 GB | 142 GB 80GB| 1000 GB | 2.50 美元 |
| ~~Nvidia H100~~ *（2025 年 12 月删除）* | | | | | |
| ~~8x Nvidia H100~~ *（2025 年 12 月删除）* | | | | | |
| 4x Nvidia A100 | 48 个 vCPU | 568GB| 320GB| 4000 GB | 10.00 美元 |         
| 8x Nvidia A100 | 96 个 vCPU | 1136 GB | 1136 GB 640GB| 8000 GB | 20.00 美元 | 

## 以编程方式配置硬件

您可以使用 `huggingface_hub` 以编程方式配置 Space 硬件。这允许您在需要动态分配 GPU 的广泛用例中使用。
查看[this guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/manage_spaces)了解更多详情。

## 框架具体要求[[框架]]大多数 Spaces 在 GPU 升级后应该可以开箱即用，但有时您需要安装您使用的机器学习框架的 CUDA 版本。请遵循本指南，以确保您的空间利用改进的硬件。

### PyTorch

您需要安装与内置 CUDA 驱动程序兼容的 PyTorch 版本。将以下两行添加到您的 `requirements.txt` 文件中应该可以工作：

```
--extra-index-url https://download.pytorch.org/whl/cu113
torch
```

您可以通过在 `app.py` 中运行以下代码并检查 Space 日志中的输出来验证安装是否成功：

```Python
import torch
print(f"Is CUDA available: {torch.cuda.is_available()}")
# True
print(f"CUDA device: {torch.cuda.get_device_name(torch.cuda.current_device())}")
# Tesla T4
```

许多框架会自动使用 GPU（如果可用）。 🤗 `transformers`、`fastai` 等管道就是这种情况。在其他情况下，或者如果您直接使用 PyTorch，您可能需要将模型和数据移动到 GPU，以确保计算在加速器上完成，而不是在 CPU 上完成。您可以使用 PyTorch 的 `.to()` 语法，例如：

```Python
model = load_pytorch_model()
model = model.to("cuda")
```

###贾克斯

如果使用 JAX，则需要指定包含 CUDA 兼容包的 URL。请将以下行添加到您的 `requirements.txt` 文件中：

```
-f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
jax[cuda11_pip]
jaxlib
```之后，您可以通过打印以下代码的输出并在您的空间日志中检查它来验证安装。

```Python
import jax

print(f"JAX devices: {jax.devices()}")
# JAX devices: [StreamExecutorGpuDevice(id=0, process_index=0)]
print(f"JAX device type: {jax.devices()[0].device_kind}")
# JAX device type: Tesla T4
```

### 张量流

默认 `tensorflow` 安装应识别 CUDA 设备。只需将 `tensorflow` 添加到您的 `requirements.txt` 文件中，并在 `app.py` 中使用以下代码在您的 Space 日志中进行验证。

```Python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
# [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

## 计费

空间计费基于硬件使用情况并按分钟计算：空间在请求的硬件上运行的每分钟都会向您收取费用， 
无论空间是否被使用。

在空间的生命周期内，仅当空间处于`Starting`或`Running`时才会计费。这意味着构建过程中没有任何成本。

如果正在运行的空间开始出现故障，它将自动暂停并停止计费。

如果长时间（例如两天）不使用在免费硬​​件上运行的空间，则会自动暂停。默认情况下，即使没有使用，升级的空间也会无限期运行。您可以通过空间设置中的[setting a custom "sleep time"](#sleep-time)更改此行为。要中断您的空间的计费，您可以将硬件更改为 CPU 基本，或[pause](#pause)。

有关计费的更多信息可以在 [dedicated Hub-wide section](./billing) 中找到。### 社区 GPU 资助 

您是否拥有很棒的 Space，但需要帮助支付 GPU 硬件升级成本？我们热衷于为那些拥有创新空间的人提供帮助，因此请随时申请社区 GPU 资助，看看您的空间是否能成功！该应用程序可以在左下角“睡眠时间设置”下的 Space 硬件存储库设置中找到：

![Community GPU Grant](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/ask-for-community-grant.png)

## 设置自定义睡眠时间[[sleep-time]]

如果您的 Space 在默认的 `cpu-basic` 硬件上运行，如果不活动时间超过设定时间（当前为 48 小时），它将进入睡眠状态。任何访问您空间的人都会自动重新启动它。

如果您希望 Space 永远不会停用，或者想要设置自定义睡眠时间，则需要升级到付费硬件。

默认情况下，升级后的 Space 永远不会进入休眠状态。但是，您可以使用此设置让升级后的空间在未使用时变为空闲状态（`stopped`阶段）😴。当硬件处于睡眠状态时，您无需为升级的硬件付费。一旦收到新访客，空间就会“醒来”或重新启动。

然后，您的 Spaces 硬件设置中将提供以下界面：

可以使用以下选项：

## 副本您可以通过请求多个副本来水平扩展您的空间。这会在空间的多个实例之间分配流量，以提高可用性和吞吐量。您可以通过 API 设置副本数量：

```
POST https://huggingface.co/api/spaces/{namespace}/{repo}/replicas
Content-Type: application/json

{
  "replicas": 2
}
```

> [!注意]
> 副本仅适用于升级（付费）的硬件。每个副本都是独立计费的。

## 流式传输日志、事件和指标[[streaming]]

您可以通过 SSE（服务器发送的事件）从您的空间传输实时日志、状态事件和指标：

- **构建或运行日志**：`GET /api/spaces/{namespace}/{repo}/logs/{build|run}`
- **状态事件**：`GET /api/spaces/{namespace}/{repo}/events`
- **指标**：`GET /api/spaces/{namespace}/{repo}/metrics`

这些端点需要使用[SSE protocol](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)进行身份验证并返回数据。

日志端点接受可选的 `tail` 查询参数（非负整数）以限制对日志的最后 N 行的响应：

```
GET /api/spaces/{namespace}/{repo}/logs/{build|run}?tail=100
```

## 暂停一个空格[[pause]]

您可以从存储库设置中`pause` 一个空间。 “已暂停”空间意味着该空间处于暂停状态，在手动重新启动之前不会使用资源，并且只有已暂停空间的所有者才能重新启动它。暂停时间不计费。

### 虎斑在空间
https://huggingface.co/docs/hub/spaces-sdks-docker-tabby.md