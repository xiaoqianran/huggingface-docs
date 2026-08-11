<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 带注释的模型卡模板

> [!提示]
> 准备好发布了吗？请参阅[Model Release Checklist](./model-release-checklist)，了解简洁、信息丰富且用户友好的模型发布的基本步骤。

## 模板

[modelcard_template.md file](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)

## 路线

完整填写模型卡需要几个不同角色的输入。 （一个人可能拥有多个角色。）我们将这些角色称为**开发人员**，他负责编写代码并进行培训； **社会技术人员**，擅长分析技术与社会的长期相互作用（包括律师、伦理学家、社会学家或权利倡导者）； **项目组织者**，了解模型的整体范围和影响范围，可以大致填写卡片的每个部分，并作为模型卡片更新的联系人。

* 填写[Training Procedure](#training-procedure-optional)和[Technical Specifications](#technical-specifications-optional)需要**开发商**。它们对于 [Bias, Risks, and Limitations](#bias-risks-and-limitations) 的“限制”部分也特别有用。他们负责为评估提供[Results](#results)，并且最好与其他角色一起定义评估的其余部分：[Testing Data, Factors & Metrics](#testing-data-factors--metrics)。* **社会技术**对于填写[Bias, Risks, and Limitations](#bias-risks-and-limitations)中的“偏见”和“风险”是必要的，对于[Uses](#uses)中的“超出范围使用”特别有用。

* 需要**项目组织者**填写[Model Details](#model-details)和[Uses](#uses)。他们也可能会填写
[Training Data](#training-data)。项目组织者还可以负责[Citation](#citation-optional)、[Glossary](#glossary-optional)、 
[Model Card Contact](#model-card-contact)、[Model Card Authors](#model-card-authors-optional)、[More Information](#more-information-optional)。

_下面提供了斜体说明。_

模板变量名称出现在`monospace`中。

--- 

# 型号名称

**部分概述：** 提供模型名称以及模型内容的 1-2 句话摘要。 

`model_id`

`model_summary`

# 目录

**部分概述：** 提供每个部分的链接，使人们能够轻松跳转/在其他位置使用保留目录的文件/打印内容等。

# 型号详细信息

**部分概述：** 本部分提供有关模型是什么、其当前状态及其来源的基本信息。它对于任何想要参考该模型的人来说应该很有用。

## 型号说明

`model_description`_提供有关模型的基本详细信息。这包括架构、版本（是否在论文中介绍过）、原始实现是否可用以及创建者。任何版权均应归属于此。本节还可以提及有关训练程序、参数和重要免责声明的一般信息。_

* **开发者：** `developers`

_列出（最好是链接到）构建模型的人员。_

* **资助者：** `funded_by`
  
_列出（最好是链接到）在财务上、计算上或以其他方式支持或启用该模型的资金来源。_

* **分享者[可选]：** `shared_by`

_列出（最好是链接到）使模型在线可用的人员/组织。_

* **型号类型：** `model_type`

_您可以将“类型”命名为：_

_1.监督/学习方法_

_2.机器学习类型_

_3.情态_

* **语言** [NLP]：`language`

_当系统使用或处理自然（人类）语言时使用此字段。_

* **许可证：** `license`

_正在使用的许可证的名称和链接。_

* **根据模型进行微调[可选]：** `base_model`

_如果此模型有另一个模型作为其基础，请在此处链接到该模型。_

## 模型来源 [可选]* **存储库：** `repo`
* **纸张[可选]：** `paper`
* **演示[可选]：** `demo`

_为用户提供直接查看模型及其详细信息的来源。其他类型的资源 - 培训日志、经验教训等 - 属于 [More Information](#more-information-optional) 部分。如果您在本节中包含一件事，请链接到存储库。_

# 用途

**本节概述：**本节解决有关如何在不同应用环境中使用模型的问题，讨论模型的可预见用户（包括受模型影响的用户），并描述被认为超出模型范围或滥用的用途。  请注意，本节无意包含许可证使用详细信息。为此，请直接链接到许可证。

## 直接使用

`direct_use`

_解释如何在不进行微调、后处理或插入管道的情况下使用模型。建议使用示例代码片段。_

## 下游使用[可选]

`downstream_use`

_解释在针对任务进行微调或插入更大的生态系统或应用程序时如何使用此模型。建议使用示例代码片段。_

## 超出范围的使用

`out_of_scope_use`_列出模型如何可预见地被滥用（以不起作用的方式使用）并解决用户不应该对模型执行哪些操作。_

# 偏见、风险和局限性

**本节概述：** 本节指出可预见的危害、误解以及技术和社会技术限制。它还提供有关警告和潜在缓解措施的信息。偏见、风险和限制有时是密不可分的/指的是相同的问题。一般来说，偏见和风险是社会技术性的，而限制是技术性的： 
- **偏见**是对某些亚群体的刻板印象或不成比例的表现（偏差）。 
- **风险**是模型可能导致的社会相关问题。
- **限制**是一种可能的故障模式，可以按照列出的建议进行解决。

`bias_risks_limitations`

_此模型有哪些已知或可预见的问题？_

## 建议

`bias_recommendations`

_针对可预见的问题有哪些建议？这可以包括从“对图像进行缩减采样”到过滤露骨内容的所有内容。_

# 训练详情**部分概述：** 本部分提供描述和复制培训的信息，包括培训数据、培训元素的速度和规模以及培训对环境的影响。这也与 [Technical Specifications](#technical-specifications-optional) 密切相关，当与培训程序相关时，此处的内容应链接到该部分。  对于想要了解更多有关模型输入和训练足迹的人来说，它非常有用。
对于任何想要了解模型所学习内容的基础知识的人来说，它都是相关的。

## 训练数据

`training_data`

_写 1-2 个句子来说明训练数据是什么。理想情况下，这会链接到数据集卡以获取更多信息。与数据预处理或附加过滤相关的文档链接可以在此处以及[More Information](#more-information-optional)._中找到
 

## 培训程序[可选]

### 预处理

`preprocessing`

_详细标记化、调整大小/重写（取决于模态）等_

### 速度、尺寸、时间

`speeds_sizes_times`

_详细吞吐量、开始/结束时间、检查点大小等_

# 评价**部分概述：** 本部分描述评估协议、评估中测量的内容并提供结果。  理想情况下，评估至少包含两部分，其中一部分着眼于一般绩效的定量测量（[Testing Data, Factors & Metrics](#testing-data-factors--metrics)），例如可以通过基准测试来完成；另一种是考察特定社会安全问题的表现（[Societal Impact Assessment](#societal-impact-assessment-optional)），例如可以通过红队来完成。您还可以在模型卡元数据中以结构化方式指定模型的评估结果。结果由中心解析并显示在模型页面上的小部件中。请参阅 https://huggingface.co/docs/hub/model-cards#evaluation-results。

## 测试数据、因素和指标_理想情况下，评估是针对不同因素（例如任务、领域和人口亚组）进行**分解的**；并使用对可预见的使用环境最有意义的指标进行计算。不同子组之间的平等评估表现被认为是这些子组之间的“公平”；目标公平性指标应根据模型使用情况下哪些错误更可能出现问题来决定。然而，本节最常用于报告不同任务基准的总体评估性能。_

### 测试数据

`testing_data`

_描述测试数据或链接到其数据集卡。_

### 因素

`testing_factors`

_哪些可预见的特征会影响模型的行为？理想情况下，评估应根据这些因素进行分解，以便发现绩效差异。_

### 指标

`testing_metrics`

_将使用哪些指标进行评估？_

## 结果

`results`

_结果应基于上面定义的因素和指标。_

### 总结

`results_summary`

_结果说明了什么？这可以作为普通观众的一种 tl;dr 。_

## 社会影响评估[可选]_使用此自由文本部分解释如何评估该模型的社会危害风险，例如儿童安全、NCII、隐私和暴力。这可能采取以下问题的答案的形式：_

- _这个模型对于孩子来说使用安全吗？为什么或为什么不？_
- _该模型是否经过测试以评估与非自愿亲密图像（包括 CSEM）相关的风险？_
- _该模型是否经过测试以评估与暴力活动或暴力描述相关的风险？结果如何？_

_还可以提供每个问题的定量数字。_

# 模型检查[可选]

**部分概述：** 这是一些开发人员开始添加的实验部分，可解释性/可解释性方面的工作可能会在其中进行。

`model_examination`

# 环境影响

**部分概述：** 总结了计算环境影响所需的信息，例如用电量和碳排放量。

* **硬件类型：** `hardware_type`
* **使用时间：** `hours_used`
* **云提供商：** `cloud_provider`
* **计算区域：** `cloud_region`
* **碳排放量：** `co2_emitted`

_可以使用[Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700)中提供的[Machine Learning Impact calculator](https://mlco2.github.io/impact#compute)来估算碳排放量。_

# 技术规格[可选]**本节概述：** 本节包含有关模型目标和架构以及计算基础设施的详细信息。对于对模型开发感兴趣的人很有用。编写此部分通常需要模型开发人员直接参与。

## 模型架构和目标

`model_specs`

## 计算基础设施

`compute_infrastructure`

### 硬件

`hardware_requirements`

_最低硬件要求是什么，例如处理、存储和内存要求？_

### 软件

`software`

# 引文[可选]

**部分概述：** 开发人员对此模型的首选引用。这通常是一篇论文。

### BibTeX 

`citation_bibtex`

### APA 

`citation_apa`

# 术语表 [可选]

**部分概述：** 本部分定义常见术语以及指标的计算方式。

`glossary`

_明确定义术语，以便受众能够理解。_

# 更多信息 [可选]

**本节概述：** 本节提供有关数据集创建、技术规范、经验教训和初步结果的文章链接。

`more_information`

# 模型卡作者 [可选]**部分概述：** 本部分列出了创建模型卡的人员，为模型卡构建的详细工作提供认可和问责。

`model_card_authors`

# 型号卡联系人

**部分概述：** 为对模型卡有更新、建议或疑问的人员提供联系模型卡作者的方式

`model_card_contact`

# 如何开始使用模型

**部分概述：** 提供代码片段来展示如何使用该模型。

`get_started_code`

---

**请引用为：**
奥佐阿尼、埃齐和格奇克、玛丽莎和米切尔、玛格丽特。模型卡指南。拥抱脸，2022 年。https://huggingface.co/docs/hub/en/model-card-guidebook

### Web数据集
https://huggingface.co/docs/hub/datasets-webdataset.md