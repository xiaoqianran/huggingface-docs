<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 定价和计费

Hugging Face Jobs 让您可以在 Hugging Face 基础设施上运行计算任务，而无需自行管理。只需在各种 CPU 和 GPU 选项中定义命令、Docker 映像和硬件风格即可。

> [!提示]
> 任何具有积极 [credit balance](https://huggingface.co/settings/billing) 的用户或组织都可以获得工作。 [PRO](./pro)、团队或 [Enterprise](./enterprise) 订阅中包含的每月计算积分计入该余额，并可用于工作。

作业计费基于硬件使用情况并按分钟计算：作业在请求的硬件上运行的每分钟都会向您收费。

在作业的生命周期中，仅在作业启动或运行时才会计费。这意味着构建过程中没有任何成本。

如果正在运行的作业开始失败，它将自动暂停并停止计费。

## 定价

作业根据所使用的硬件按分钟计费。以下是可用的硬件选项及其定价。

＃＃＃ 中央处理器| **硬件** | **CPU** | **内存** | **临时存储** | **每小时价格** |
|------------------------ |-------------- |------------- |------------------------ | ----------------- |
| CPU基础| 2 个虚拟CPU | 16GB| 50GB| 0.01 美元 |
| CPU升级| 8 个 vCPU | 32GB| 50GB| 0.03 美元 |
|中央处理器XL | 16 个 vCPU | 124GB| 1000 GB | 1.00 美元 |
| CPU性能| 32 个 vCPU | 256 GB | 256 GB 1024 GB | 1024 GB 1.90 美元 |

### GPU| **硬件** | **CPU** | **内存** | **GPU 内存** | **临时存储** | **每小时价格** |
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
| 4x Nvidia A100 - 大 | 48 个 vCPU | 568GB| 320GB| 4000 GB | 10.00 美元 |
| 8x Nvidia A100 - 大 | 96 个 vCPU | 1136 GB | 1136 GB 640GB| 8000 GB | 20.00 美元 |
|英伟达 H200 | 23 个 vCPU | 256 GB | 256 GB 141 GB | 141 GB 3000 GB | 5.00 美元 |
| 2 个 Nvidia H200 | 46 个 vCPU | 512GB| 282GB| 6000 GB | 10.00 美元 |
| 4x Nvidia H200 | 92 个 vCPU | 1024 GB | 1024 GB 564GB| 12000 GB | 20.00 美元 || 8x Nvidia H200 | 184 个 vCPU | 2048GB | 1128GB| 24000 GB | 40.00 美元 |
|英伟达 RTX PRO 6000 | 23 个 vCPU | 256 GB | 256 GB 96GB | 475GB| 2.75 美元 |
| 2 个 Nvidia RTX PRO 6000 | 46 个 vCPU | 512GB| 192 GB | 192 GB 950GB| 5.50 美元 |
| 4 个 Nvidia RTX PRO 6000 | 92 个 vCPU | 1024 GB | 1024 GB 384GB| 1900GB| 11.00 美元 |
| 8 个 Nvidia RTX PRO 6000 | 184 个 vCPU | 2048GB | 768 GB | 3800 GB | 3800 GB 22.00 美元 |

您还可以通过 `GET /api/jobs/hardware` 的 API 或通过 CLI 以编程方式检索可用硬件和定价：

```bash
>>> hf jobs hardware
```

### 暴露端口

作业可以[expose one or more ports](./jobs-configuration#expose-ports) 使它们在作业运行时可以从外部访问。除硬件价格外，公开一个或多个端口按每个作业的统一费率计费：

| **产品** | **每小时价格** |
|----------------| ----------------- |
|暴露端口 | 0.01 美元 |

与硬件一样，它按分钟计费，仅在作业启动或运行时计费。

## 管理账单

### 向您的组织开具账单默认情况下，计费是针对用户的命名空间进行的，但您可以通过指定正确的 `namespace` 来向您的组织计费：

```bash
hf jobs run --namespace my-org-name ...
```

在这种情况下，作业在组织帐户下运行，您可以在组织作业页面（组织页面 > 设置 > 作业）中看到它。

> [!注意]
> 在企业计划及以上版本中，组织管理员可以通过 [Granular feature access](./security-resource-groups#granular-feature-access) 设置限制谁可以运行和查看向组织计费的作业（仅限组织管理员或选定资源组的成员）。

### 向资源组开具账单

> [!警告]
> 此功能是企业计划及以上版本的一部分。

如果您的组织设置了 [Resource Groups](./security-resource-groups)，您可以将作业成本归因于特定资源组。为此：

1. 您必须是资源组的成员。
2. 运行作业时将资源组的 ID 作为`namespace` 传递。

您可以在组织的资源组设置页面中找到资源组的 ID。

```bash
hf jobs run --namespace <resource-group-id> ...
```

在Python中：

```python
>>> from huggingface_hub import run_job
>>> run_job(
...     image="python:3.12",
...     command=["python", "-c", "print('Hello!')"],
...     namespace="<resource-group-id>",
... )
```

### 查看当前计算使用情况

您可以在 [Billing](https://huggingface.co/settings/billing) 页面的“计算使用情况”部分下查看作业的当前账单信息：

有关计费的更多信息可以在 [dedicated Hub documentation](https://huggingface.co/docs/hub/en/billing) 中找到。

### 建议

#### 设置超时限制创建Job时设置一个`timeout`以确保它不能运行超过一定的持续时间。
达到`timeout`持续时间的作业运行将自动停止，其计费也将自动停止。
以下是使用 CLI 设置超时的方法：

```bash
hf jobs run --timeout 3h ...
```

请注意，默认超时设置为 **30 分钟**。
因此，如果您的作业需要更多时间来运行，您必须指定更长的超时。

#### 取消不相关的作业

如果正在运行的作业不再相关，您可以通过作业页面或 CLI 提前取消它以停止其计费：

```bash
hf jobs cancel <job-id>
```

### 门控数据集
https://huggingface.co/docs/hub/datasets-gate.md