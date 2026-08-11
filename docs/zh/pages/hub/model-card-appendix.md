<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 附录

## 附录 A：用户研究
_关键问题全文回复_

### 你如何定义模型卡？

***洞察力：受访者对模型卡的看法普遍相似：文档集中于培训、用例和偏见/限制等问题***

* 模型卡是模型描述，包括它们的训练方式、用例以及潜在的偏差和限制
* 描述模型基本特征的文件，以便读者/用户了解他/她面前的制品、背景/培训、如何使用它及其技术/道德限制。
* 它们作为模型的活生生的人工制品来记录它们。模型卡包含的信息包括特定模型用途的高级描述、限制、偏差、指标等等。它们主要用于理解模型的作用。
* 模型卡之于模型，就像 GitHub README 之于 GitHub 项目一样。它告诉人们他们需要了解的有关模型的所有信息。如果你不写一个，没有人会使用你的模型。* 据我了解，模型卡使用某些基准（地理、文化、性别等）来定义模型的可用性和局限性。它本质上是模型的“营养成分标签”，可以显示模型的创建方式并指导其他人了解其可重用性。
* 模型卡是关于模型的元数据和文档，是正确使用模型所需知道的一切：关于模型的信息、哪篇论文介绍了它、它在什么数据集上进行训练或微调、它属于谁、该模型是否存在已知的风险和限制、任何有用的技术信息。
* IMO 模型卡是模型的简要介绍，其中包括：
  * 模型架构特性的简短摘要
  * 描述其训练数据
  * 参考数据集的性能如何（如果可能的话，准确度和速度指标）
  * 限制
  * 如何在 Transformers 库的上下文中使用它
  * 来源（原始文章，Github 存储库，...）
* 易于访问的文档，任何背景都可以阅读并了解关键模型组件和社会影响

### 你喜欢模型卡的哪些方面？* 他们很有趣地教人们新模型
* 作为一个非技术人员，有机会了解模型，了解其基础知识，这是作者以透明且可解释（即值得信赖）的方式披露其创新的机会。
* 我喜欢带有视觉效果和小部件的交互式模型卡，它们允许我在不运行任何代码的情况下尝试模型。
* 我喜欢好的型号卡的原因是您可以找到有关该特定型号的所有所需信息。
* 模型卡对人工智能伦理世界来说是革命性的。这是减轻/教育机器学习偏见的第一个切实步骤之一。他们提高了意识和责任感！
* 结构化、详尽，信息越多越好。
* 它有助于了解模型的优点（或缺点）。
* 简洁性和可访问性

### 你不喜欢模型卡的什么？* 可能会涉及技术性和/或密集性
* 它们包含针对不同受众（研究人员、工程师、非工程师）的大量信息，因此很难探索具有预期用例的模型卡。 
  * [注意：此评论可以通过针对不同受众的切换视图来解决]
* 好的作品是需要时间来创造的。很难对它们进行测试以确保信息是最新的。很多时候，模型卡的格式完全不同——所以你必须弄清楚某个人是如何构建他们的模型卡的。 
  * [注意：此评论有助于展示标准化格式和自动化工具的价值，使创建模型卡变得更容易]
* 如果没有社区的帮助来进行补充评估，模型卡可能会受到开发人员可能没有意识到的固有偏见的影响。对他们来说还处于早期阶段，但如果没有更彻底的评估，模型卡的信息可能会太有限。
* 空模型卡。没有许可证信息 - 客户需要该信息，但通常没有。 
* 它们通常要么太简洁，要么太冗长。
*写下它们哈哈祝福你

### 其他重要的新见解* 模型卡最好由不同角色的人填写：技术规格一般只能由开发人员填写；整个过程中的道德考虑因素通常是由那些倾向于从事道德问题工作的人最了解的。
* 模型用户非常关心许可证——具体来说，模型是否可以合法地用于特定任务。

## 附录 B：景观分析
_机器学习中模型文档状态概述_

### 型号卡示例
模型卡和密切相关的变体的示例包括： 

* 谷歌云：[Face Detection](https://modelcards.withgoogle.com/face-detection)、[Object Detection](https://modelcards.withgoogle.com/object-detection)
* 谷歌研究：[ML Kit Vision Models](https://developers.google.com/s/results/ml-kit?q=%22Model%20Card%22)、[Face Detection](https://sites.google.com/view/perception-cv4arvr/blazeface)、[Conversation AI](https://github.com/conversationai/perspectiveapi/tree/main/model-cards)
* OpenAI：[GPT-3](https://github.com/openai/gpt-3/blob/master/model-card.md)、[GPT-2](https://github.com/openai/gpt-2/blob/master/model_card.md)、[DALL-E dVAE](https://github.com/openai/DALL-E/blob/master/model_card.md)、[CLIP](https://github.com/openai/CLIP-featurevis/blob/master/model-card.md)
* [NVIDIA Model Cards](https://catalog.ngc.nvidia.com/models?filters=&orderBy=weightPopularASC&query=)
* [Salesforce Model Cards](https://blog.salesforceairesearch.com/model-cards-for-ai-model-transparency/)
* [Allen AI Model Cards](https://github.com/allenai/allennlp-models/tree/main/allennlp_models/modelcards)
* [Co:here AI Model Cards](https://docs.cohere.ai/responsible-use/)
* [Duke PULSE Model Card](https://arxiv.org/pdf/2003.03808.pdf)
* [Stanford Dynasent](https://github.com/cgpotts/dynasent/blob/main/dynasent_modelcard.md)
* [GEM Model Cards](https://gem-benchmark.com/model_cards)
* Parl.AI: [Parl.AI sample model cards](https://github.com/facebookresearch/ParlAI/tree/main/docs/sample_model_cards), [BlenderBot 2.0 2.7B](https://github.com/facebookresearch/ParlAI/blob/main/parlai/zoo/blenderbot2/model_card.md)
* [Perspective API Model Cards](https://github.com/conversationai/perspectiveapi/tree/main/model-cards)
* 请参阅 https://github.com/ivylee/model-cards-and-datasheets 了解更多示例！

### 大型语言模型的模型卡
大型语言模型通常与相关文档一起发布。具有关联模型卡（或相关文档工具）的大型语言模型包括： 

* [Big Science BLOOM model card](https://huggingface.co/bigscience/bloom)
* [GPT-2 Model Card](https://github.com/openai/gpt-2/blob/master/model_card.md) 
* [GPT-3 Model Card](https://github.com/openai/gpt-3/blob/master/model-card.md)
* [DALL-E 2 Preview System Card](https://github.com/openai/dalle-2-preview/blob/main/system-card.md)
* [OPT-175B model card](https://arxiv.org/pdf/2205.01068.pdf)

### 模型卡生成工具
以编程方式或交互方式生成模型卡的工具包括：* [Salesforce Model Card Creation](https://help.salesforce.com/s/articleView?id=release-notes.rn_bi_edd_model_card.htm&type=5&release=232)
* [TensorFlow Model Card Toolkit](https://ai.googleblog.com/2020/07/introducing-model-card-toolkit-for.html)
  * [Python library](https://pypi.org/project/model-card-toolkit/)
* [GSA / US Census Bureau Collaboration on Model Card Generator](https://bias.xd.gov/resources/model-card-generator/)
* [Parl.AI Auto Generation Tool](https://parl.ai/docs/tutorial_model_cards.html)
* [VerifyML Model Card Generation Web Tool](https://www.verifyml.com)
* [RMarkdown Template for Model Card as part of vetiver package](https://cran.r-project.org/web/packages/vetiver/vignettes/model-card.html)
* [Databaseline ML Cards toolkit](https://databaseline.tech/ml-cards/)

### 模型卡教育工具
用于理解模型卡和了解如何创建模型卡的工具包括： 

* [Hugging Face Hub docs](https://huggingface.co/course/chapter4/4?fw=pt)
* [Perspective API](https://developers.perspectiveapi.com/s/about-the-api-model-cards)
* [Kaggle](https://www.kaggle.com/code/var0101/model-cards/tutorial)
* [Code.org](https://studio.code.org/s/aiml-2021/lessons/8)
* [UNICEF](https://unicef.github.io/inventory/data/model-card/)

---

**请引用为：**
奥佐阿尼、埃齐和格奇克、玛丽莎和米切尔、玛格丽特。模型卡指南。拥抱脸，2022 年。https://huggingface.co/docs/hub/en/model-card-guidebook

### 存储库设置
https://huggingface.co/docs/hub/repositories-settings.md