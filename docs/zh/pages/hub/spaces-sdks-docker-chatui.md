<!-- huggingface-docs: machine-translated zh-CN from English source -->

# 空间上的 ChatUI

**HuggingChat** 是一个开源接口，使每个人都可以尝试开源大型语言模型，例如 Falcon、StarCoder 和 BLOOM。借助名为 ChatUI 的官方 Docker 模板，您只需使用 Hugging Face 的基础设施点击几下即可根据您选择的模型部署自己的 HuggingChat。

## 部署您自己的聊天 UI

要开始，只需点击[here](https://huggingface.co/new-space?template=huggingchat/chat-ui-template)。在该应用程序的后端，[text-generation-inference](https://github.com/huggingface/text-generation-inference)用于更好地优化模型推理。由于这些模型无法在 CPU 上运行，因此您可以根据您选择的模型来选择 GPU。 

    

您应该提供一个 MongoDB 端点，您的聊天记录将在其中写入。如果将此部分留空，您的日志将保存到空间内的数据库中。请注意，Hugging Face 无法访问您的聊天记录。您可以通过提供应用程序名称和应用程序颜色参数来配置空间的名称和主题。
在此下方，您可以选择您想要服务的模型的 Hugging Face Hub ID。您还可以在下面的字典中以 JSON 格式更改生成超参数。_注意_：如果您想部署具有门控访问权限的模型或私有存储库中的模型，您只需在存储库机密中提供 `HF_TOKEN` 即可。您需要将其值设置为可从 [here](https://huggingface.co/settings/tokens) 获取的访问令牌。

创建完成后，您将在您的空间上看到`Building`。构建完成后，您可以尝试自己的 HuggingChat！

开始聊天吧！

## 阅读更多

- [HF Docker Spaces](https://huggingface.co/docs/hub/spaces-sdks-docker)
- [chat-ui GitHub Repository](https://github.com/huggingface/chat-ui)
- [text-generation-inference GitHub repository](https://github.com/huggingface/text-generation-inference)

### 为模特提供工作服务
https://huggingface.co/docs/hub/jobs-serving.md