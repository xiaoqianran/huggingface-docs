<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 机器学习文档工具的概况
2018 年模型卡框架的开发受到自然语言处理数据语句 ([Bender & Friedman, 2018](https://aclanthology.org/Q18-1041/)) 和数据集数据表 ([Gebru et al., 2018](https://www.fatml.org/media/documents/datasheets_for_datasets.pdf)) 的主要文档框架工作的启发。自从提出模型卡以来，人们还提出了许多其他工具来记录和评估机器学习开发周期的各个方面。这些工具（包括模型卡和在模型卡之前提出的相关文档工作）可以根据其重点（例如，该工具关注机器学习系统生命周期的哪一部分？）及其目标受众（例如，该工具是为谁设计的？）来进行背景化。在下面的图 1-2 中，我们沿着这些维度总结了几个重要的文档工具，提供了每个工具的上下文描述，并链接到示例。我们将文档工具大致分为以下几组：* **以数据为中心**，包括专注于机器学习系统生命周期中使用的数据集的文档工具
* **以模型和方法为中心**，包括专注于机器学习模型和方法的文档工具；和 
* **以系统为中心**，包括专注于 ML 系统的文档工具，包括模型、方法、数据集、API 以及作为 ML 系统一部分相互交互的非 AI/ML 组件

这些分组并不相互排斥；它们确实包含机器学习系统生命周期的重叠方面。例如，**系统卡**专注于记录可能包含多个模型和数据集的机器学习系统，因此可能包含与以数据为中心或以模型为中心的文档工具重叠的内容。所描述的工具是机器学习系统生命周期的文档工具的非详尽列表。一般来说，我们包含的工具是： 

* 专注于机器学习系统生命周期的某些（或多个）方面的文档
* 包括发布旨在重复使用、采用和改编的模板

## 机器学习文档工具总结

### 图1| **机器学习系统生命周期的阶段** |  **工具** |  **简要说明** |  **示例** ||:--------------------------------: |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- ||数据| ***数据表*** [(Gebru et al., 2018)](https://www.fatml.org/media/documents/datasheets_for_datasets.pdf) | “我们建议每个数据集都附有一个数据表，记录其动机、创建、组成、预期用途、分发、维护和其他信息。”                                                                                                                                                                                                                	|例如，参见 [Ivy Lee’s repo](https://github.com/ivylee/model-cards-and-datasheets) 的示例 |
|数据| ***数据报表*** [(Bender & Friedman, 2018)(Bender et al., 2021)](https://techpolicylab.uw.edu/wp-content/uploads/2021/11/Data_Statements_Guide_V2.pdf) | “数据声明是数据集的表征，它提供了上下文，使开发人员和用户能够更好地理解实验结果如何推广、如何正确部署软件以及基于软件构建的系统中可能反映哪些偏差。”                                                                                                         	|参见[Data Statements for NLP Workshop](https://techpolicylab.uw.edu/events/event/data-statements-for-nlp/)||数据| ***数据集营养标签*** [(Holland et al., 2018)](https://huggingface.co/papers/1805.03677) | “数据集营养标签......是一个诊断框架，通过在人工智能模型开发之前提供数据集“成分”的精炼而全面的概述，降低标准化数据分析的障碍。”                                                                                                                                                                              	|参见[The Data Nutrition Label](https://datanutrition.org/labels/)||数据| ***NLP 数据卡*** [(McMillan-Major et al., 2021)](https://huggingface.co/papers/2108.07374) | “我们提出了在自然语言处理 (NLP) 中创建文档模板和指南的两个案例研究：Hugging Face (HF) 数据集中心 [^1] 以及 Generation 及其评估和指标 (GEM) 的基准。在这两种情况下，我们使用术语数据卡来指代数据集的文档。| 请参阅[(McMillan-Major et al., 2021)](https://huggingface.co/papers/2108.07374) ||数据| ***数据集开发生命周期文档框架*** [(Hutchinson et al., 2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445918) | “我们引入了严格的数据集开发透明度框架，支持决策和问责制。该框架利用数据集开发的循环、基础设施和工程性质来借鉴软件开发生命周期的最佳实践。”                                                                                                         	|请参阅[(Hutchinson et al., 2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445918)，附录 A 了解模板 ||数据| ***数据卡*** [(Pushkarna et al., 2021)](https://huggingface.co/papers/2204.01075) | “数据卡是关于 ML 数据集各个方面的基本事实的结构化摘要，利益相关者在数据集的整个生命周期中都需要这些数据集，以进行负责任的 AI 开发。这些摘要提供了对塑造数据以及模型的过程和基本原理的解释。”                                                                                                  	|请参阅[Data Cards Playbook github](https://github.com/PAIR-code/datacardsplaybook/) ||数据| ***CrowdWorkSheets*** [(Díaz et al., 2022)](https://huggingface.co/papers/2206.08931) | “我们为数据集开发人员引入了一个新颖的框架 CrowdWorkSheets，以促进数据注释管道各个阶段关键决策点的透明记录：任务制定、注释器选择、平台和基础设施选择、数据集分析和评估以及数据集发布和维护。”                                                 	|参见[(Díaz et al., 2022)](https://huggingface.co/papers/2206.08931)||模型和方法| ***模型卡*** [Mitchell et al. (2018)](https://huggingface.co/papers/1810.03993) | “模型卡是伴随训练有素的机器学习模型的简短文档，可在与预期应用领域相关的各种条件下提供基准评估。模型卡还公开了模型的使用环境、性能评估程序的详细信息以及其他相关信息。”                                	|请参阅 https://huggingface.co/models、[Model Card Guidebook](https://huggingface.co/docs/hub/model-card-guidebook) 和 [Model Card Examples](https://huggingface.co/docs/hub/model-card-appendix#model-card-examples) ||模型和方法| ***储值卡*** [Shen et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3442188.3445971) | “我们推出了 Value Cards，这是一个深思熟虑驱动的工具包，旨在让计算机科学专业的学生和从业者认识到基于机器学习的决策系统的社会影响……Value Cards 鼓励对不同的 ML 性能指标及其潜在权衡进行调查和辩论。”                                                               	|请参阅[Shen et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3442188.3445971)，第 3.3 节 ||模型和方法| ***方法卡*** [Adkins et al. (2022)](https://dl.acm.org/doi/pdf/10.1145/3491101.3519724) | “我们提出方法卡来指导 ML 工程师完成模型开发过程……这些信息包括规定性和描述性元素，主要重点是确保 ML 工程师能够正确使用这些方法。”                                                                                                                                          	|请参阅[Adkins et al. (2022)](https://dl.acm.org/doi/pdf/10.1145/3491101.3519724)，附录 A |
|模型和方法| ***ML 模型的消费者标签*** [Seifert et al. (2019)](https://ris.utwente.nl/ws/portalfiles/portal/158031484/Seifert2019_cogmi_consumer_labels_preprint.pdf) | “我们建议为经过训练和发布的 ML 模型颁发消费者标签。这些标签主要针对机器学习非专业人士，例如 ML 系统的操作员、决策执行者以及决策主体本身” |参见[Seifert et al. (2019)](https://ris.utwente.nl/ws/portalfiles/portal/158031484/Seifert2019_cogmi_consumer_labels_preprint.pdf)||系统| ***概况介绍*** [Arnold et al. (2019)](https://huggingface.co/papers/1808.07261) | “情况说明书将包含人工智能服务所有相关属性的部分，例如预期用途、性能、安全性和保障。性能将包括适当的准确性或风险测量以及时间信息。”                                                                                                                                                              	|请参阅 [IBM’s AI Factsheets 360](https://aifs360.res.ibm.com) 和 [Hind et al., (2020)](https://dl.acm.org/doi/abs/10.1145/3334480.3383051) |
|系统| ***系统卡*** [Procope et al. (2022)](https://ai.facebook.com/research/publications/system-level-transparency-of-machine-learning) | “System Cards 旨在通过向利益相关者提供 ML 系统的不同组件、这些组件如何交互以及系统如何使用不同数据和受保护信息的概述来提高 ML 系统的透明度。”                                                                                                                                 	|参见[Meta’s Instagram Feed Ranking System Card](https://ai.facebook.com/tools/system-cards/instagram-feed-ranking/)||系统| ***RL 奖励报告*** [Gilbert et al. (2022)](https://huggingface.co/papers/2204.10817) | “我们草拟了一个用于记录已部署的学习系统的框架，我们将其称为奖励报告……我们将奖励报告概述为动态文档，用于跟踪特定自动化系统优化背后的设计选择和假设的更新。它们旨在跟踪系统部署所产生的动态现象，而不仅仅是模型或数据的静态属性。” 	|请参阅 https://rewardreports.github.io ||系统| ***健壮健身房*** [Goel et al. (2021)](https://huggingface.co/papers/2101.04840) | “我们发现了评估 NLP 系统的挑战，并提出了 Robustness Gym (RG) 形式的解决方案，这是一个简单且可扩展的评估工具包，统一了 4 个标准评估范式：子群体、转换、评估集和对抗性攻击。”                                                                                                                  	|请参阅 https://github.com/robustness-gym/robustness-gym ||系统| ***关于 ML*** [Raji and Yang, (2019)](https://huggingface.co/papers/1912.06166) | “关于 ML（机器学习生命周期的理解和透明度的注释和基准）是由 PAI 领导的一项为期多年、多利益相关者的计划。该计划旨在汇集不同的观点，以大规模开发、测试和实施机器学习系统文档实践。”                                                                 	|参见[ABOUT ML’s resources library](https://partnershiponai.org/about-ml-resources-library/)|

### 以数据为中心的文档工具

一些提议的文档工具侧重于机器学习系统生命周期中使用的数据集，包括作为连续周期的一部分来训练、开发、验证、微调和评估机器学习模型。这些工具通常关注数据生命周期的许多方面（可能针对特定数据集、数据集组或更广泛），包括如何组装、收集、注释数据以及如何使用数据。* 扩展了电子行业中数据表的概念，[Gebru et al. (2018)](https://www.fatml.org/media/documents/datasheets_for_datasets.pdf)提出了数据集的数据表，以记录与数据集的创建、潜在用途和相关问题相关的详细信息。 
* [Bender and Friedman (2018)](https://aclanthology.org/Q18-1041/)提出用于自然语言处理的数据语句。 [Bender, Friedman and McMillan-Major (2021)](https://techpolicylab.uw.edu/wp-content/uploads/2021/11/Data_Statements_Guide_V2.pdf)更新原始数据语句框架并提供资源，包括编写数据语句以及在第一版模式和新版本模式之间进行转换的指南[^2]。 
* [Holland et al. (2018)](https://huggingface.co/papers/1805.03677)提出数据营养标签，类似于食品的营养成分和隐私披露的营养标签，作为数据集分析和决策的工具。数据营养标签团队于2020年发布了标签的更新设计和界面（[Chmielinski et al., 2020)](https://huggingface.co/papers/2201.03954)）。
* [McMillan-Major et al. (2021)](https://huggingface.co/papers/2108.07374) 以 Hugging Face Hub 上的数据卡的形式描述了 **NLP 数据卡** 的开发过程和生成的模板[^3]，以及作为 Generation 及其评估指标 (GEM) 环境的 NLP 基准的一部分的数据集的数据卡[^4]。* [Hutchinson et al. (2021)](https://dl.acm.org/doi/pdf/10.1145/3442188.3445918) 描述了对全面数据集文档的需求，并借鉴软件开发实践，提供了用于记录数据集开发生命周期的多个方面的模板（出于表 1 和表 2 的目的，我们将其框架称为 **数据集开发生命周期文档框架**）。
* [Pushkarna et al. (2021)](https://huggingface.co/papers/2204.01075) 提议将数据卡作为**数据卡手册**的一部分，这是一种以人为本的文档工具，专注于工业和研究中使用的数据集。 

### 以模型和方法为中心的文档工具

另一组文档工具可以被认为是专注于机器学习模型和机器学习方法。这些包括：

* [Mitchell et al. (2018)](https://huggingface.co/papers/1810.03993) 提出用于模型报告的**模型卡**，以伴随经过训练的 ML 模型并记录与评估、使用和其他问题相关的问题

* [Shen et al. (2021)](https://dl.acm.org/doi/abs/10.1145/3442188.3445971) 提出**价值卡**，用于向学生和从业者传授与机器学习模型相关的价值观

* [Seifert et al. (2019)](https://ris.utwente.nl/ws/portalfiles/portal/158031484/Seifert2019_cogmi_consumer_labels_preprint.pdf) 提出**ML 模型的消费者标签**，以帮助使用模型或受模型影响的非专家了解与模型相关的关键问题。* [Adkins et al. (2022)](https://dl.acm.org/doi/pdf/10.1145/3491101.3519724) 分析描述性文档工具的各个方面（他们认为包括**模型卡**和数据表），并主张为机器学习工程师增加规范性工具。他们提出了方法卡，重点关注机器学习方法，并主要考虑模型开发人员和审阅者等技术利益相关者的设计。

  *他们设想了模型卡和方法卡之间的关系，部分是这样说的：“我们建议的部分和提示......[在方法卡模板中]重点关注足以生成具有定义的输入、输出和任务的适当ML模型的ML方法。这些示例是对象检测方法（例如单次检测器）和语言建模方法（例如生成预训练变压器（GPT））。*可以为使用这些方法创建的模型创建模型卡*。”*他们还指出，“虽然模型卡和情况说明书主要关注记录现有模型，但方法卡更关注创建和训练这些模型时需要考虑的基本方法和算法选择。*作为一个粗略的类比，如果模型卡和情况说明书提供有关熟食的营养信息，那么方法卡提供食谱*。”
 

### 以系统为中心的文档工具

以系统为中心的文档工具不是关注特定的模型、数据集或方法，而是着眼于模型如何相互交互、与数据集、方法以及其他 ML 组件交互以形成 ML 系统。* [Procope et al. (2022)](https://ai.facebook.com/research/publications/system-level-transparency-of-machine-learning) 提出系统卡来记录和解释人工智能系统——可能包括多个机器学习模型、人工智能工具和非人工智能技术——它们协同工作来完成任务。
* [Arnold et al. (2019)](https://huggingface.co/papers/1808.07261) 将消费产品符合性声明的理念扩展到 AI 服务，提出情况说明书来记录“AI 服务”的各个方面，这些服务通常通过 API 访问，并且可能由多个不同的 ML 模型组成。 [Hind et al. (2020)](https://dl.acm.org/doi/abs/10.1145/3334480.3383051) 分享对构建情况说明书的思考。
* [Gilbert et al. (2022)](https://huggingface.co/papers/2204.10817) 提出**强化学习**系统的奖励报告，认识到机器学习系统的动态性质以及需要进行文档工作以纳入部署后性能的考虑，特别是对于强化学习系统。
* [Goel et al. (2021)](https://huggingface.co/papers/2101.04840) 开发**Robustness Gym**，一个评估工具包，用于测试现实系统中深度神经网络的多个方面，允许跨评估范式进行比较。 
* 通过[ABOUT ML project](https://partnershiponai.org/workstream/about-ml/) ([Raji and Yang, 2019](https://huggingface.co/papers/1912.06166))，AI 合作伙伴正在协调机器学习社区中各利益相关者群体的努力，为 ML 系统开发全面、可扩展的文档工具。 

## 模型卡的演变自从米切尔等人提出模型卡的提案以来。 2018 年，模型卡已被各种组织采用和改编，包括开发和托管机器学习模型的大型科技公司和初创公司[^5]、描述新技术的研究人员[^6]以及评估各种项目模型的政府利益相关者[^7]。模型卡也作为人工智能伦理教育工具包的一部分出现，许多组织和开发人员已经创建了用于自动或半自动创建模型卡的实现。附录A提供了一组由不同组织创建的各类ML模型的模型卡示例（包括大语言模型的模型卡）、模型卡生成工具和模型卡教育工具。### 拥抱脸部中心上的模型卡
2018年以来，承载和共享模型卡的新平台和媒介也不断涌现。例如，与该项目特别相关的是，Hugging Face 在 Hugging Face Hub 上托管模型卡，作为与 ML 模型关联的存储库中的 README 文件。因此，模型卡成为 Hugging Face Hub 上模型用户的一种重要文档形式。作为模型卡分析的一部分，我们为 Hugging Face Hub 上的几十个 ML 模型开发并提出了模型卡，使用 Hub 的拉取请求 (PR) 和讨论功能来收集模型卡的反馈，验证模型卡中包含的信息，并在 Hugging Face Hub 上发布模型的模型卡。在撰写本指南时，Hugging Face Hub 上的所有 Hugging Face 模型在 Hub 上都有关联的模型卡[^8]。上传到 Hugging Face Hub 的大量模型（在撰写本文时有 101,041 个模型），使我们能够探索该中心模型卡中的内容：
我们首先分析语言模型、模型卡，以识别模式（例如重复的部分和小节），目的是回答最初的问题，例如：

1) 这些型号中有多少有型号卡？
   
2) 有多少百分比的下载有关联的模型卡？

通过对中心上所有模型的分析，我们注意到下载量最多的是前 200 个模型。

随着对大型语言模型的持续关注，按照下载最多且仅带有模型卡的模型进行排序，我们注意到各自模型卡中最重复出现的部分。 

虽然模型卡中的某些标题可能因模型而异，但我们对每个模型卡内每个部分的组件/主题进行了分组，然后将它们映射到最常出现的部分标题（主要在前 200 个下载模型中找到，并在 Bloom 模型卡的帮助/指导下找到）

> [!提示]
> [Checkout the User Studies](./model-cards-user-studies)

> [!提示]
> [See Appendix](./model-card-appendix)[^1]：对于每个工具，描述摘自第二列中列出的链接论文。

[^2]：请参阅 https://techpolicylab.uw.edu/data-statements/ 。

[^3]：请参阅 https://techpolicylab.uw.edu/data-statements/ 。

[^4]：请参阅 https://techpolicylab.uw.edu/data-statements/ 。

[^5]：参见 Hugging Face Hub、Google Cloud 的模型卡 https://modelcards.withgoogle.com/about 等。

[^6]：参见附录 A。

[^7]：请参阅 GSA / 美国人口普查局关于模型卡生成器的合作。

[^8]：“Hugging Face 模型”是指 Hugging Face 在 Hub 上共享的模型，而不是其他组织。正式来说，这些模型的模型 ID 中没有“/”。

---

**请引用为：**
奥佐阿尼、埃齐和格奇克、玛丽莎和米切尔、玛格丽特。模型卡指南。拥抱脸，2022 年。https://huggingface.co/docs/hub/en/model-card-guidebook

### 关于空间的证据
https://huggingface.co/docs/hub/spaces-sdks-docker-evidence.md