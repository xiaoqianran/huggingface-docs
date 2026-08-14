<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 存储桶：安全性与合规性

存储桶构建在与 Hugging Face Hub 相同的基础设施上，内置企业级安全性和合规性。

## 加密

存储在存储桶中的所有数据均使用 **AES-256** 加密进行静态加密。传输中的数据通过 **TLS** 进行保护。

## 访问控制

存储桶使用集线器的标准访问控制机制：

- **SSO**：通过 [Single Sign-On](./security-sso) 通过组织的身份提供商进行身份验证
- **RBAC**：通过[Resource Groups](./security-resource-groups)的细粒度权限让您控制谁可以读取、写入或管理每个存储桶
- **令牌**：程序化访问通过具有范围权限的[User Access Tokens](./security-tokens)进行管理

## 审核日志

所有存储桶操作（上传、下载、删除和权限更改）都记录在您组织的 [Audit Logs](./audit-logs) 中，让您可以完整跟踪谁访问了什么内容以及何时访问内容。

## 数据驻留

存储桶数据存储在**美国和欧盟区域**。您可以在创建存储桶时选择数据所在的位置，[pre-warming](./storage-buckets#pre-warming-and-cdn) 可以让您在特定云区域中缓存更靠近计算的数据。

## 合规性

Hugging Face 拥有以下认证和合规标准：- **SOC 2 Type 2** 认证 — 主动监控和修补安全漏洞
- **GDPR** 合规 — 通过 [Enterprise Plans](https://huggingface.co/pricing) 提供数据处理协议

有关 Hugging Face 整体安全状况的更多详细信息，请参阅[Security](./security) 页面。如有疑问，请联系[security@huggingface.co](mailto:security@huggingface.co)。

### 用户访问令牌
https://huggingface.co/docs/hub/security-tokens.md