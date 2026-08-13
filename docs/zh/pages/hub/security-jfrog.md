<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 第三方扫描仪：JFrog

[JFrog](https://jfrog.com/) 的安全扫描器可检测机器学习模型中的恶意行为。

![JFrog report for the danger.dat file contained in mcpotato/42-eicar-street](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/jfrog-report.png)
*[danger.dat](https://huggingface.co/mcpotato/42-eicar-street/blob/main/danger.dat)的报告示例*

我们[partnered with JFrog](https://hf.co/blog/jfrog)提供扫描功能，以使Hub更安全。模型文件由 JFrog 扫描仪扫描，我们将扫描结果公开在 Hub 界面上。

JFrog 的扫描仪旨在减少误报。事实上，我们目前观察到的模型权重中包含的代码并不总是恶意的。当在文件中检测到代码时，JFrog 的扫描仪将解析它并进行分析，以检查是否存在潜在的恶意使用。

    
    

这是一个示例存储库，您可以查看该存储库以查看该功能的实际应用：[mcpotato/42-eicar-street](https://huggingface.co/mcpotato/42-eicar-street)。

## 模型安全复习

为了共享模型，我们序列化用于与模型交互的数据结构，以方便存储和传输。某些序列化格式容易受到恶意攻击，例如任意代码执行（看看你的pickle），使共享模型具有潜在的危险。由于 Hugging Face 已成为流行的模型共享平台，我们希望保护社区免受此影响，因此我们开发了像 [picklescan](https://github.com/mmaitre314/picklescan) 这样的工具以及集成第三方扫描仪的原因。

Pickle 并不是唯一可利用的格式，[see for reference](https://github.com/Azure/counterfit/wiki/Abusing-ML-model-file-formats-to-create-malware-on-AI-systems:-A-proof-of-concept) 如何利用 Keras Lambda 层来实现任意代码执行。

### HF PRO 订阅 🔥
https://huggingface.co/docs/hub/pro.md