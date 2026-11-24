<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 守夜人

我们看到越来越多的攻击使用人工智能代理来针对推理基础设施以及模型或数据集的提供者。攻击者使用公共 Hub 存储库以及其他协作平台作为枢纽或攻击媒介。

我们利用从 2026 年 7 月的 OpenAI/HF 事件中学到的经验，构建了 Nightwatch，即我们的**主动代理安全**服务。其安全代理实时扫描集线器活动日志以查找这些攻击的迹象。

![Hub access logs, metrics, and external OSINT feed into the Nightwatch agent, which flags unusual activity for review.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/nightwatch/live-scan-sources-53c80514.png)

Nightwatch 是一款仅适用于企业客户的付费附加组件。价格反映了用于连续扫描的大量代币（开放和封闭模型）。 [Contact our team](https://huggingface.co/contact/sales?from=enterprise) 为您的组织探索 Nightwatch。

## 实施

![Illustrative Hub logs showing one account reading three repositories in the same second.](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/enterprise/nightwatch/log-excerpt.png)

当 Nightwatch 检测到异常活动时，Slack 代理会向您的团队发送警报。您还可以将这些警报路由到组织的安全事件摄取系统。

### Slack 警报示例

#### 异常访问令牌活动> 我们检测到您组织成员之一的 Hugging Face 帐户存在异常活动。
>
> 自 2026 年 10 月 29 日以来，我们发现来自两个可疑 IP 地址的请求使用其 `xyz` 访问令牌。该令牌具有写入角色，创建于 2025 年 10 月 12 日。
>
> 2026 年 10 月 29 日和 11 月 9 日，该令牌用于读取您组织的 60 个私有存储库。其中，22 个被克隆。
>
> 令牌已被撤销。我们可以分享受影响存储库的完整列表和时间表。

#### 私有数据集公开

> 在 UTC 时间 09:16，管理员帐户将您的私有数据集 `acme/research-preview` 公开。该请求来自地理定位为新加坡的 IP 地址。八分钟前，同一个帐户在巴黎的常用网络上处于活动状态。
>
> 在 UTC 时间 09:20，我们看到来自您组织外部的首次下载。该数据集仍然是公开的。这是计划中的发布吗？
>
> 如果这是意外情况，您可以再次将数据集设为私有并查看管理员帐户的最近活动。我们可以共享时间线和两个 IP 地址来帮助您进行调查。

### 使用空格作为组织卡
https://huggingface.co/docs/hub/spaces-organization-cards.md