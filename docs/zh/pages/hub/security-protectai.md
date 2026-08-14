<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 第三方扫描仪：保护AI

> [!提示]
> 有兴趣加入我们的安全合作伙伴关系/在 Hub 上提供扫描信息吗？请通过 security@huggingface.co 与我们联系。*

[Protect AI](https://protectai.com/) 的 [Guardian](https://protectai.com/guardian) 捕获了 pickle、Keras 和其他漏洞，详情请参阅其 [Knowledge Base page](https://protectai.com/insights/knowledge-base/)。 Guardian 还受益于他们的赏金社区[Huntr](https://huntr.com/)s 发送的报告。

![Protect AI report for the danger.dat file contained in mcpotato/42-eicar-street](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/protect-ai-report.png)
*[danger.dat](https://huggingface.co/mcpotato/42-eicar-street/blob/main/danger.dat)的报告示例*

我们与 Protect AI 合作提供扫描服务，以使 Hub 更安全。与我们的内部扫描系统扫描文件的方式相同，Guardian 也扫描公共存储库的文件。

我们的前端专门为此目的进行了重新设计，以适应新的扫描仪：

这是一个示例存储库，您可以查看该存储库以查看该功能的实际应用：[mcpotato/42-eicar-street](https://huggingface.co/mcpotato/42-eicar-street)。

## 模型安全复习

为了共享模型，我们序列化用于与模型交互的数据结构，以方便存储和传输。某些序列化格式容易受到恶意攻击，例如任意代码执行（看看你的pickle），使共享模型具有潜在的危险。由于 Hugging Face 已成为流行的模型共享平台，我们希望保护社区免受此影响，因此我们开发了像 [picklescan](https://github.com/mmaitre314/picklescan) 这样的工具以及集成第三方扫描仪的原因。

Pickle 并不是唯一可利用的格式，[see for reference](https://github.com/Azure/counterfit/wiki/Abusing-ML-model-file-formats-to-create-malware-on-AI-systems:-A-proof-of-concept) 如何利用 Keras Lambda 层来实现任意代码执行。

### 拥抱人脸数据集上传决策指南
https://huggingface.co/docs/hub/datasets-upload-guide-llm.md