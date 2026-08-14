<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 模型卡指南 

模型卡是机器学习模型的重要文档和透明度框架。我们相信模型卡有潜力充当“边界对象”，这是一种单一的人工制品，在与模型卡交互时具有不同背景和目标的用户（包括开发人员、学生、政策制定者、伦理学家、受机器学习模型影响的人以及其他利益相关者）都可以访问。我们认识到，开发一个单一的产品来服务于如此多方面的目的是很困难的，需要仔细考虑潜在的用户和用例。作为 Hugging Face 科学团队的一员，我们过去几个月的目标是帮助模型卡实现这一愿景，同时考虑到 Hugging Face 和更广泛的 ML 社区中的这些挑战。为了实现这一目标，重要的是要认识到帮助模型卡发展到今天的深思熟虑和专注的努力，从采用模型卡作为许多大型组织的标准实践，到开发用于托管和生成模型卡的复杂工具。由于模型卡是由 Mitchell 等人提出的。 （2018），机器学习文档的领域已经扩大和发展。人们已经提出并开发了大量用于数据、模型和机器学习系统的文档工具和模板，反映了数百名研究人员、受影响的社区成员、倡导者和其他利益相关者的出色工作。关于机器学习文档与负责任的人工智能变革理论之间关系的重要讨论引发了持续的重要讨论，有时甚至出现分歧。我们还认识到模型卡面临的挑战，这在某种程度上反映了机器学习文档和负责任的人工智能工作更普遍面临的挑战，我们看到了未来的机会，可以帮助塑造模型卡及其在未来数月和数年中积极发挥作用的生态系统。我们的工作展示了我们认为模型卡现在所处的位置以及它们未来在 Hugging Face 及其他领域的发展方向。这项工作是模型卡当前状态的“快照”，通过对 ML 文档制品实例化的多种方式的景观分析得出结论。它代表了关于模型卡的当前状态和更理想的愿景的多种观点中的一种。在这篇博文中，我们总结了我们的工作，包括讨论更广泛、不断发展的 ML 文档工具、模型卡的不同受众和观点，以及模型卡内容的潜在新模板。我们还在 Hugging Face Hub 的背景下探索和开发机器学习模型的模型卡，利用 Hub 的功能来协作创建、讨论和传播 ML 模型的模型卡。 

随着本指南的推出，我们引入了一些新资源，并将之前关于模型卡的工作联系在一起：

1）更新的模型卡模板，在`huggingface_hub`库[modelcard_template.md file](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/modelcard_template.md)中发布，汇集了学术界和整个行业的模型卡工作。

2) [Annotated Model Card Template](./model-card-annotated)，详细说明如何填写该卡。3）[Model Card Creator Tool](https://huggingface.co/spaces/huggingface/Model_Cards_Writing_Tool)，无需编程即可轻松创建卡片，并帮助团队分担不同部分的工作。

4) A [User Study](./model-cards-user-studies) 关于 Hugging Face 模型卡的使用

5) 最先进的模型文档[Landscape Analysis and Literature Review](./model-card-landscape-analysis)。

我们还包括一个[Appendix](./model-card-appendix)，其中包含这项工作的更多详细信息。

---

**请引用为：** 
奥佐阿尼、埃齐和格奇克、玛丽莎和米切尔、玛格丽特。模型卡指南。拥抱脸，2022 年。https://huggingface.co/docs/hub/en/model-card-guidebook

### Langfuse 空间
https://huggingface.co/docs/hub/spaces-sdks-docker-langfuse.md