<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 团队和企业计划

> [!提示]
>
> 订阅团队或企业计划以获取适合您组织的高级功能。

团队和企业组织计划为组织增加了高级功能，为公司和团队在 Hugging Face 上实现安全、合规和托管的协作。

    
    

## 快速比较我们的计划

### 核心使用、存储、速率限制

|特色|免费|团队|企业 |企业增强版 |
| ---------------------------------------------------------------- | ----------- | -------------------- | -------------------- | ---------------------- |
|存储 – 公共存储库 |尽最大努力| 12TB 底座 + 1TB/座 | 200TB 基础 + 1TB/座 | 500TB 基础 + 1TB/座 |
|存储 – 私人存储库 | 100GB | 1TB/席位 + 付费 | 1TB/席位 + 付费 | 1TB/席位 + 付费 |
| [Extra storage](./storage-limits#pay-as-you-go-price) | ❌ | ✅ 付费 | ✅ 付费 | ✅ 付费 || API 请求/周期\* | 1,000 | 3,000 | 6,000 | 10,000 至 100,000† |
|解析器请求/周期\* | 5,000 | 20,000 | 50,000 | 100,000 至 500,000† |
|页数请求/周期\* | 200 | 200 400 | 600 | 1,000 至 10,000† |

\* 所有配额均按 5 分钟固定窗口计算

† 定义组织 IP 范围时

### 推理和 Hub 学分|特色 |免费|团队|企业 |企业增强版 |
| -------------------------------------------------------------------------------------------------------------------------- | -------- | --------------------------------- | --------------------------------- | --------------------------------- |
|通过推理提供程序提供模型服务 | ✅ 付费 | ✅ 包含 2 美元/座位/月 + 现收现付 | ✅ 包含 2 美元/座位/月 + 现收现付 | ✅ 包含 2 美元/座位/月 + 现收现付 |
| [Usage & billing control](https://huggingface.co/docs/inference-providers/pricing#inference-providers-usage-breakdown) | ❌ | ✅ | ✅ | ✅ |
|使用推理端点扩展部署 | ✅ 付费 | ✅ 付费 | ✅ 付费 | ✅ 付费 || Hub 积分\* 包含在计划中 | ❌ | ❌（可批量购买）| ❌（可批量购买）|含 ACV 5% |

\* Hub 积分可用于推理提供商、推理端点、作业、空间升级、ZeroGPU 配额扩展

### 空间和工作

|特色 |免费|团队|企业 |企业增强版 |
| -------------------------------------- | --------- | ----------- | ----------- | ---------------- |
|空间 – 基于 CPU 的运行时 | 8 单位\* | ✅ 无限制 | ✅ 无限制 | ✅ 无限制 |
|空间 – ZeroGPU 使用等级 | 5 分钟† | 40 分钟† | 60 分钟† | 60 分钟† |
|空间——硬件升级|现收现付 |现收现付 |现收现付 |现收现付 |
|空间的开发模式/自定义域 | ❌ | ✅ | ✅ | ✅ |
|作业和脚本（训练/微调、评估）|现收现付 |现收现付 |现收现付 |现收现付 |

\* 同时运行† 包括每日配额；付费计划可以使用积分超出配额，每 10 分钟 GPU 时间 1 美元

### 回购规则、访问控制、可见性

|特色 |                 免费|      团队|   企业 |      企业增强版 |
| -------------------------------------- | :----------------------------------: | :-------------: | :-------------: | :------------------------: |
|访问控制粒度 | [Standard](./organizations-security) | ✅ 细粒度 | ✅ 细粒度 | ✅ 细粒度+政策 |
|组织控制|                  ❌ |       ✅ |       ✅ |             ✅ |
|集线器控制|                  ❌ |       ❌ |       ❌ |             ✅ |
|默认私人仓库 |                  ❌ |       ✅ |       ✅ |             ✅ |
|禁用公共存储库（组织范围）|                  ❌ |       ✅ |       ✅ |             ✅ || [Data residency](./storage-regions) |                  ❌ |       ✅ |       ✅ |             ✅ |
| Data Studio（私有数据集）|                  ❌ |       ✅ |       ✅ |             ✅ |
|门控组集合 |                  ❌ |       ✅ |       ✅ |             ✅ |

### 身份、身份验证、组织安全

|特色|免费|     团队|  企业 |企业增强版 |
| -------------------------------------------------- | :--: | :----------: | :----------: | :-------------: |
| [SSO to private org](./enterprise-sso) |  ❌ | ✅ 基本单点登录 | ✅ 基本单点登录 | ✅ 托管单点登录 |
| [SSO to public Hub](./enterprise-advanced-sso) |  ❌ |      ❌ |      ❌ |       ✅ |
| [Enforce 2FA](./enterprise-advanced-security) |  ❌ |      ✅ |      ✅ |       ✅ |
| [OAuth Token Exchange](./oauth#token-exchange-for-organizations-rfc-8693) |  ❌ |      ❌ |      ✅ |       ✅ |
|禁用用户的个人公共存储库 |  ❌ |      ❌ |      ❌ |       ✅ |
|禁止用户加入其他组织 |  ❌ |      ❌ |      ❌ |       ✅ ||禁用 PRO 订阅 |  ❌ |      ❌ |      ❌ |       ✅ |
|隐藏会员列表 |  ❌ |      ✅ |      ✅ |       ✅ |

### 治理、审计、合规

|特色|免费|    团队|企业 |企业增强版 |
| ----------------------------------------------------------------------------------- | :--: | :---------: | :---------: | :-------------: |
|角色控制 |  ✅ | ✅ 高级 | ✅ 高级 |   ✅ 高级 |
| [Audit logs](./audit-logs) |  ❌ |     ✅ |     ✅ |       ✅ |
| [Resource groups](./enterprise-advanced-security) |  ❌ |     ✅ |     ✅ |       ✅ |
| [Tokens admin / management](./enterprise-tokens-management) |  ❌ |     ✅ |     ✅ |       ✅ |
| [Token revocation](./enterprise-tokens-management#revoking-via-api) |  ❌ |     ❌ |     ✅ |       ✅ |
| [Download analytics](./download-analytics) |  ✅ |     ✅ |     ✅ |       ✅ |
| [Content access / policy controls](./enterprise-network-security) |  ❌ |     ❌ |     ❌ |       ✅ |
| [Network access controls](./enterprise-network-security) |  ❌ |     ❌ |     ❌ |       ✅ || [Enforced authentication (advanced)](./enterprise-network-security) |  ❌ |     ❌ |     ❌ |       ✅ |

### 用户配置和管理

|特色|   免费|     团队|  企业 |企业增强版 |
| ---------------------- | :-----: | :------------: | :------------: | :-------------: |
|入职/离职 | ✅ 说明书 | ✅ 控制 | ✅ 控制 |  ✅ 自动化 |
| SCIM 配置 |    ❌ |      ❌ | ✅ 基于邀请 | ✅ 全生命周期 |
|管理用户|    ❌ |      ❌ |      ❌ |       ✅ |

### 支持、计费、采购|特色|     免费|          团队|       企业 |    企业增强版 |
| ------------------------------------------- | :----------: | :--------------------: | :--------------------: | :--------------------: |
|支持|论坛访问 |      尽最大努力| SLA 的电子邮件支持 |高级 Slack 支持 |
|计费|              |信用卡自助服务 |    使用发票付款 |    使用发票付款 |
|合同（包括采购订单）|      ❌ |           ❌ |     ✅ 高频模板 |   ✅ 客户论文 |
|法律审查|      ❌ |           ❌ |           ❌ |           ✅ |
|供应商入职和安全调查问卷 |      ❌ |           ❌ |           ❌ |           ✅ |

＃＃＃ 社区|特色 |免费|团队|企业 |企业增强版 |
| -------------------------------------------------------------------------------------------------------------------------- | :--: | :--: | :--------: | :-------------: |
|组织文章 |  ❌ |  ✅ |     ✅ |       ✅ |
| [Publisher Analytics Dashboard](./publisher-analytics) |  ❌ |  ✅ |     ✅ |       ✅ |
| [Set your primary org on your profile](https://huggingface.co/changelog/primary-organization-on-profiles) |  ❌ |  ✅ |     ✅ |       ✅ |

### 定价

|特色|免费|    团队|    企业 |企业增强版 |
| ------------------ | :--: | :---------: | :--------------: | :-------------: |
|定价|  - | 20 美元/用户/月 | 50 美元/用户/月起|     定制|
|飞行员可用性 |  ❌ |     ❌ |        ❌ |       ✅ |

## 进一步潜水

在以下部分中，我们将记录以下团队和企业功能：

- [Single Sign-On (SSO)](./enterprise-sso)
- [Audit Logs](./audit-logs)
- [Storage Regions](./storage-regions)
- [Data Studio for Private datasets](./enterprise-datasets)
- [Resource Groups](./security-resource-groups)
- [Advanced Compute Options](./advanced-compute-options)
- [Advanced Security](./enterprise-advanced-security)
- [Tokens Management](./enterprise-tokens-management)
- [OAuth Token Exchange](./oauth#token-exchange-for-organizations-rfc-8693)
- [Publisher Analytics](./publisher-analytics)
- [Gating Group Collections](./enterprise-gating-group-collections)
- [Network Security](./enterprise-network-security)
- [Higher Rate limits](./rate-limits)
- [Blog Articles](./enterprise-blog-articles)最后，团队和企业计划包括更多的 [included public storage](./storage-limits)，以及订阅中每个席位 1TB 的 [private storage](./storage-limits)，即，如果您的组织有 40 名成员，那么您将拥有 40TB 的存储空间用于存储您的私有模型和数据集。

### 嵌入图集
https://huggingface.co/docs/hub/datasets-embedding-atlas.md