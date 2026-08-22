<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 计费

在 Hugging Face，我们为 ML 社区（即 Hub）构建了一个协作平台，并通过提供高级功能和简单的 AI 计算访问来实现盈利。

欢迎与计费相关的任何反馈或支持请求：billing@huggingface.co

## 团队和企业订阅

我们通过团队或企业计划为组织提供先进的安全性和合规性功能，其中包括用于存储库的[Single Sign-On](./enterprise-sso)、[Advanced Access Control](./enterprise-resource-groups)、对数据位置的控制、用于公共和私有存储库的更高[storage capacity](./storage-limits)等等。

团队和企业计划的计费方式与典型订阅相同。它们会自动续订，但您可以随时在组织的结算设置中选择取消。

您可以使用信用卡或 AWS 账户支付团队订阅费用，或通过年度合同升级到企业版。

续订后，您的订阅中的席位数量将更新，以匹配您组织的成员数量。
[included storage](./storage-limits) 以上的私有存储库存储将与您的订阅续订一起计费。

	
	

## PRO 订阅

PRO 订阅为认真的用户解锁了基本功能，包括：- 公共和私人存储库的[storage capacity](./storage-limits)更高
- 更高的带宽和 API [rate limits](./rate-limits)
- 包括每月计算积分，可在所有 Hugging Face 计算服务中使用：[Inference Providers](/docs/inference-providers/)、[Inference Endpoints](https://huggingface.co/docs/inference-endpoints)、[Spaces](./spaces-overview) 升级硬件和 [Jobs](./jobs-overview)
- 更高级别的 ZeroGPU 空间使用和即用即付配额扩展
- 能够创建在计算上运行的 Gradio 和 Docker 空间，托管最多 10 个 ZeroGPU 空间，并使用开发模式
- 能够发布社交帖子和社区博客
- 在私人数据集上利用[Data Studio](./data-studio)

在 https://huggingface.co/pro 查看完整的福利列表，然后在 https://huggingface.co/subscribe/pro 订阅

与团队和企业订阅类似，PRO 订阅的计费方式与典型订阅相同。订阅会自动为您续订。您可以随时在结算设置中选择取消订阅：https://huggingface.co/settings/billing

您只能使用信用卡支付 PRO 订阅费用。订阅与任何即用即付的计算使用量分开计费。
[included storage](./storage-limits) 以上的私有存储库存储将与您的订阅续订一起计费。

注意：PRO 福利也包含在 [Enterprise subscription](https://huggingface.co/enterprise) 中。## 按量付费私人存储 

除了 PRO、Team 和 Enterprise 中包含的 1TB（或每个席位 1TB）私有存储之外，额外的私有存储按 1TB 增量计费，基本价格为 **18 美元/TB/月**。 

在即用即付模式下，超额费用将通过您的付款方式收取。

我们的客户经理可以为大批量提供额外折扣。
请在[huggingface.co/pricing](https://huggingface.co/pricing#storage)查看完整的定价等级。

## Hub 上的计算服务

我们还直接提供[Spaces](./spaces)、[Inference Endpoints](https://huggingface.co/docs/inference-endpoints/index)和[Inference Providers](https://huggingface.co/docs/inference-providers/index)的计算服务。

虽然我们的一些计算服务有免费套餐，但用户和组织可以付费访问更强大的硬件加速器。

我们的计算服务的计费是基于使用量的，这意味着您只需为您使用的量付费。您可以随时从位于用户或组织的设置菜单中的计费仪表板监控您的使用情况。

	
	

计算服务使用量与 PRO 和团队/企业订阅（以及潜在的私人存储）分开计费。
计算服务的发票在每个月初编辑。

## 可用的付款方式

Hugging Face 使用 [Stripe](https://stripe.com) 安全地处理您的付款信息。Hugging Face 计算服务支持的唯一付款方式是信用卡。
您可以通过结算设置将信用卡添加到您的帐户。

	
	

## 云提供商合作伙伴关系

我们与[AWS](https://huggingface.co/blog/aws-partnership)、[Azure](https://huggingface.co/blog/hugging-face-endpoints-on-azure)等云提供商合作，
[Google Cloud](https://huggingface.co/blog/llama31-on-vertex-ai) 让客户可以轻松地直接在他们选择的云中使用 Hugging Face。
这些解决方案和使用量由云提供商直接计费。最终，我们希望人们无论何时何地都有使用 Hugging Face 的绝佳选择
构建机器学习驱动的产品。

您还可以选择通过 [AWS Marketplace](https://aws.amazon.com/marketplace/pp/prodview-n6vsyhdjkfng2) 将您的 Hugging Face 组织链接到您的 AWS 账户。
Hugging Face 计算服务的使用量将包含在您的 AWS 账单中。请阅读我们的[blog post](https://huggingface.co/blog/aws-marketplace)了解更多信息。

## 支持常见问题解答 

**问。为什么我需要添加学分？我可以用它们做什么？**

A. 积分可让您使用 HF 按量付费服务：

- 作业：在 GPU 上运行任何工作负载
- 推理提供者：通过 API 调用 250k+ 模型
- 推理端点：专用部署
- GPU 空间：托管在定制硬件上
- ZeroGPU：每日津贴之外的额外配额
- 私人存储：私人存储库的额外存储如果您有 PRO、Team 或 Enterprise 订阅，则订阅中包含的积分每月都会记入，并计入相同的服务。在对任何即用即付使用量进行计费之前，它们会自动应用。

**问。如果我用完积分会怎样？**

答：我们建议启用自动充值，以避免积分用完后服务中断。

**问。我在添加卡时遇到问题。怎么了？**

答：请确保该卡支持 3D 安全身份验证，并针对定期在线付款进行了正确配置。我们尚不支持在印度发行的信用卡，因为我们正在努力增加系统对最新 RBI 指令的合规性。在我们添加对印度信用卡的支持之前，您可以：
* 将组织帐户链接到 AWS 帐户，以便访问即用即付功能（端点、空间、AutoTrain）：[Hugging Face Platform on the AWS Marketplace: Pay with your AWS Account](https://huggingface.co/blog/aws-marketplace)
* 使用其他国家发行的信用卡

**问。如何添加我的税号或更新账单详细信息？**答：在 https://huggingface.co/settings/billing/ payment 上添加或更新卡时，请选中 **其他发票信息（可选）** 框。然后，您可以提供**公司名称**和**发票详细信息**（例如税号、采购订单号或公司注册号），这些信息将显示在您的发票上。如果您需要进一步帮助，请发送电子邮件至 billing@huggingface.co。

**问。几天前我刚刚收到 PRO/Team 订阅费用。为什么又向我收费？**

答：所有订阅均在每月 1 号续订。如果您在月中注册 Team 或 PRO 的第一个月，我们将按比例收取订阅费用。

**问。我需要过去发票的副本，在哪里可以找到这些？**

答：在此处查看并下载所有发票：https://huggingface.co/settings/billing/invoices。发票也通过电子邮件发送。

**问。我需要更新我帐户中的信用卡。该怎么办？**

答：前往 https://huggingface.co/settings/billing/ payment 并随时更新您的付款方式。 

**订阅**

**问。我需要暂停我的 PRO 订阅，在哪里可以执行此操作？**答：您可以随时在此处取消订阅：https://huggingface.co/settings/billing/subscription。 
请通过 billing@huggingface.co 向我们发送您的反馈。

**问。我的组织有团队或企业订阅，我需要更新席位数。我该怎么做？**

A. 席位数量将在续订时自动调整，以反映上一时期组织成员数量的任何增加。由于是固定费用订阅，因此无需在当月或当年更新订阅的席位数。

### 如何获取用户在 Spaces 中的计划和状态
https://huggingface.co/docs/hub/spaces-get-user-plan.md